import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window needed

import pygame
import pytest

from battleship import MINI, GameConfig
from battleship.watch import AGENTS, WatchApp, main


def press(app, key):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
    app.render()


@pytest.mark.parametrize("agent", sorted(AGENTS))
def test_watching_plays_a_full_game(agent):
    app = WatchApp(MINI, agent, seed=3)
    try:
        app.render()
        press(app, pygame.K_h)                       # show hidden ships
        while not app.game.done:
            assert app.target in app.observation.legal_actions
            if app.heat is not None:
                assert all(0 <= p <= 1 + 1e-9 for p in app.heat)
            press(app, pygame.K_SPACE)
        assert app.target is None and "All ships sunk" in app.last
        press(app, pygame.K_SPACE)                   # ignored once the game is over
        press(app, pygame.K_n)                       # next board
        assert app.seed == 4 and app.observation.num_shots == 0
    finally:
        pygame.quit()


def test_heat_is_the_agents_belief():
    app = WatchApp(GameConfig(4, (3, 2), reveal_sunk_cells=False), "belief", seed=0)
    try:
        assert app.kind == "P(ship)"
        assert sum(app.heat) == pytest.approx(5)    # 5 ship cells on an empty board
        assert app.heat[app.target] == max(app.heat)
    finally:
        pygame.quit()


def test_planners_need_the_small_board():
    with pytest.raises(SystemExit):
        main(["--agent", "optimal", "--board", "classic"])


def test_q_view_shows_the_planners_values_and_the_agent_fires_at_the_smallest():
    from battleship.planning import planner
    app = WatchApp(MINI, "optimal", seed=12, show="q")
    try:
        plan = planner(MINI)
        while not app.game.done:
            app.render()
            q = app.q["q"]
            assert set(q) == set(app.observation.legal_actions)
            layouts, fired = plan.state(app.agent.belief)
            assert q[app.target] == pytest.approx(plan.optimal_value(layouts, fired))
            assert q[app.target] == pytest.approx(min(q.values()))
            # Q* = ship cells still unhit + P(miss now) + misses later, with misses later >= 0
            later = q[app.target] - app.q["unhit"] - (1 - app.heat[app.target])
            assert later >= -1e-9
            press(app, pygame.K_SPACE)
        assert app.q is None
    finally:
        pygame.quit()


def test_q_key_switches_views_for_the_greedy_agent():
    app = WatchApp(MINI, "belief", seed=12)
    try:
        assert app.q is None
        press(app, pygame.K_q)
        assert app.show == "q" and app.q is not None
        assert app.q["q"][app.target] >= app.q["best"] - 1e-9     # greedy need not be optimal
        press(app, pygame.K_SPACE)
        assert app.q is not None                                  # stays on for the next shot
        press(app, pygame.K_q)
        assert app.show == "p" and app.q is None
    finally:
        pygame.quit()


def test_q_view_is_only_for_pomdp_agents_on_the_small_board():
    for argv in (["--agent", "density", "--show", "q"],
                 ["--agent", "belief", "--board", "classic", "--show", "q"]):
        with pytest.raises(SystemExit):
            main(argv)
    app = WatchApp(MINI, "density", seed=0)
    try:
        press(app, pygame.K_q)                                     # ignored: no exact belief
        assert app.show == "p" and app.q is None
    finally:
        pygame.quit()
