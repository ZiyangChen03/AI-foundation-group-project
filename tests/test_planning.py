import random
from functools import lru_cache

import pytest

from battleship import MINI, GameConfig, all_layouts, play_solo
from battleship.agents import BeliefAgent
from battleship.belief import Belief
from battleship.planning import Planner, planner

NO_REVEAL = GameConfig(4, (3, 2), reveal_sunk_cells=False)
TINY = [GameConfig(3, (2,)), GameConfig(3, (2, 1)), GameConfig(3, (2, 1), reveal_sunk_cells=False),
        GameConfig(3, (1, 1)), GameConfig(3, (1, 1), reveal_sunk_cells=False), GameConfig(3, (3,))]


def naive_optimal(config):
    """V* with no shortcuts: every untried cell, beliefs as Belief objects."""
    table = Belief.prior(config)._table

    @lru_cache(maxsize=None)
    def value(support, fired):
        belief = Belief(table, support, fired)
        if belief.done:
            return 0.0
        return min(1 + sum(len(nxt) / len(belief) * value(nxt.support, nxt.fired)
                           for nxt in belief.outcomes(cell).values())
                   for cell in range(config.num_cells) if not fired >> cell & 1)

    prior = Belief.prior(config)
    return value(prior.support, prior.fired)


@pytest.mark.parametrize("config", TINY)
def test_optimal_value_matches_a_solver_without_shortcuts(config):
    assert Planner(config).optimal_value(Planner(config).everything, 0) == pytest.approx(
        naive_optimal(config), abs=1e-12)


@pytest.mark.parametrize("config", TINY + [MINI, NO_REVEAL])
def test_values_are_ordered(config):
    """optimal <= rollout <= greedy (policy improvement), and each value is
    what following that policy actually costs."""
    plan = planner(config)          # shared, so the 4x4 tables are solved once
    start = plan.everything, 0
    optimal = plan.optimal_value(*start)
    greedy = plan.greedy_value(*start)
    rollout = plan.policy_value(plan.rollout_shots)
    assert optimal <= rollout + 1e-12 <= greedy + 2e-12
    assert plan.policy_value(plan.optimal_shots) == pytest.approx(optimal, abs=1e-12)
    assert plan.policy_value(plan.greedy_shots) == pytest.approx(greedy, abs=1e-12)


def test_known_values_on_4x4():
    for config, optimal, rollout, greedy in ((MINI, 8.7462, 8.7645, 8.9132),
                                             (NO_REVEAL, 8.7538, 8.7835, 8.9132)):
        plan = planner(config)
        assert plan.optimal_value(plan.everything, 0) == pytest.approx(optimal, abs=1e-4)
        assert plan.policy_value(plan.rollout_shots) == pytest.approx(rollout, abs=1e-4)
        assert plan.greedy_value(plan.everything, 0) == pytest.approx(greedy, abs=1e-4)


class FirstChoice(random.Random):
    """An rng whose choice() always takes the first option, so ties are
    broken the same way every game."""

    def choice(self, seq):
        return seq[0]


class FirstTieAgent(BeliefAgent):
    def new_game(self, config, rng):
        super().new_game(config, FirstChoice(0))


@pytest.mark.parametrize("config", [MINI, NO_REVEAL])
@pytest.mark.parametrize("name", ["greedy", "rollout", "optimal"])
def test_agents_play_exactly_as_the_planner_predicts(config, name):
    """With ties always broken to the first cell, the average over all 264
    layouts must equal the planner's exact value for that policy. (With
    random ties it only matches on average.)"""
    plan = planner(config)
    choose = {"greedy": plan.greedy_shots, "rollout": plan.rollout_shots,
              "optimal": plan.optimal_shots}[name]
    expected = plan.policy_value(lambda layouts, fired: choose(layouts, fired)[:1])
    agent = FirstTieAgent(planner=name)
    shots = [play_solo(agent, config, layout=layout).num_shots for layout in all_layouts(config)]
    assert sum(shots) / len(shots) == pytest.approx(expected, abs=1e-9)


def test_hand_worked_example():
    """4x4 with one 3-ship: 16 layouts, small enough to check every value by
    hand. After a miss at B2, greedy fires at C3 (4/12 to hit) but C1 (3/12)
    is better: Q*(C3) = 1 + (4/12)(3) + (8/12)(5) = 16/3, while
    Q*(C1) = 1 + (3/12)(8/3) + (9/12)(14/3) = 31/6. So V* = 45/8 and greedy
    costs 23/4."""
    config = GameConfig(4, (3,))
    plan = Planner(config)
    layouts = plan.table.layouts
    names = {i: {c for c in layout[0].cells} for i, layout in enumerate(layouts)}
    B2, C1, C2, C3 = 5, 2, 6, 10

    def belief(keep, fired):
        return (sum(1 << i for i, cells in names.items() if keep(cells)),
                sum(1 << c for c in fired))

    assert len(layouts) == 16
    assert plan.hit_counts(plan.everything, 0)[B2] == 4
    assert plan.optimal_value(plan.everything, 0) == pytest.approx(45 / 8)
    assert plan.greedy_value(plan.everything, 0) == pytest.approx(23 / 4)
    cross = belief(lambda cells: B2 in cells, [B2])
    assert plan.optimal_value(*cross) == pytest.approx(3)
    missed = belief(lambda cells: B2 not in cells, [B2])
    assert plan.optimal_value(*missed) == pytest.approx(31 / 6)
    assert plan.greedy_shots(*missed) == [C3]
    assert plan._q(*missed, C3, plan.optimal_value) == pytest.approx(16 / 3)
    assert plan._q(*missed, C1, plan.optimal_value) == pytest.approx(31 / 6)
    ring = belief(lambda cells: B2 not in cells and C3 not in cells, [B2, C3])
    assert plan.optimal_value(*ring) == pytest.approx(5)
    c1_hit = belief(lambda cells: B2 not in cells and C1 in cells, [B2, C1])
    c1_miss = belief(lambda cells: B2 not in cells and C1 not in cells, [B2, C1])
    assert plan.optimal_value(*c1_hit) == pytest.approx(8 / 3)
    assert plan.optimal_value(*c1_miss) == pytest.approx(14 / 3)
    assert C2 in plan.optimal_shots(*cross)


def test_optimal_shots_are_legal_and_optimal():
    plan = planner(MINI)
    belief = Belief.prior(MINI)
    layouts, fired = plan.state(belief)
    for cell in plan.optimal_shots(layouts, fired):
        assert not fired >> cell & 1
        assert plan._q(layouts, fired, cell, plan.optimal_value) == pytest.approx(
            plan.optimal_value(layouts, fired))


def test_optimal_planner_needs_the_exact_belief():
    with pytest.raises(ValueError, match="exact belief"):
        BeliefAgent(belief="particles", planner="optimal")
