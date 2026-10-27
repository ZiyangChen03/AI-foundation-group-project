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
