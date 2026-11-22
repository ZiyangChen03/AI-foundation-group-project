"""The 4x4 comparison script (scripts/compare_4x4.py)."""

import importlib.util
import pathlib

import pytest

from battleship import MINI
from battleship.belief import Belief
from battleship.opponents import favorites, uniform

_PATH = pathlib.Path(__file__).parent.parent / "scripts" / "compare_4x4.py"
_spec = importlib.util.spec_from_file_location("compare_4x4", _PATH)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def test_weighted_stats():
    shots = [[4, 6], [10, 10], None]
    stats = compare.weighted_stats(shots, [0.5, 0.5, 0.0])
    assert stats["mean"] == pytest.approx(7.5)                 # (5 + 10) / 2
    assert stats["std"] == pytest.approx((0.5 * (3.5**2 + 1.5**2) / 2 + 0.5 * 2.5**2) ** 0.5)
    assert stats["worst"] == 10


def test_build_agents_has_baselines_and_pomdp_planners():
    labels = [label for label, _ in compare.build_agents()]
    for expected in ("Random", "Hunt-target + parity", "Probability density",
                     "POMDP greedy", "POMDP optimal"):
        assert expected in labels
    agents = [agent for _, agent in compare.build_agents()]
    assert len({id(a) for a in agents}) == len(agents)          # round_robin needs distinct objects


def test_every_layout_is_played_and_zero_weights_are_skipped():
    agent = dict(compare.build_agents())["POMDP greedy"]
    shots = compare.play_all_layouts(agent, range(1))
    assert len(shots) == len(Belief.prior(MINI).layouts) == 264
    habit = favorites(MINI, 3)
    weights = habit.distribution(MINI)
    some = compare.play_all_layouts(agent, range(1), weights)
    assert sum(row is not None for row in some) == sum(w > 0 for w in weights)


def test_pomdp_optimal_matches_the_exact_value():
    """Covering all layouts makes the comparison exact: the optimal planner's
    games average exactly the value the Bellman solver computes."""
    from battleship.planning import planner
    plan = planner(MINI)
    agent = dict(compare.build_agents())["POMDP optimal"]
    shots = compare.play_all_layouts(agent, range(2))
    mean = compare.weighted_stats(shots, uniform().distribution(MINI))["mean"]
    assert mean == pytest.approx(plan.optimal_value(plan.everything, 0), abs=1e-9)
