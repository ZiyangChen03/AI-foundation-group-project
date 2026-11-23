"""Information-gain planners: exact values on 4x4 and agreement with play."""

from math import log2

import pytest

from battleship import MINI, GameConfig, all_layouts, evaluate
from battleship.agents import BeliefAgent
from battleship.planning import planner

NO_REVEAL = GameConfig(4, (3, 2), reveal_sunk_cells=False)


@pytest.mark.parametrize("config", [MINI, NO_REVEAL])
def test_exact_values(config):
    plan = planner(config)
    info = plan.policy_value(plan.info_gain_shots)
    greedy_info = plan.policy_value(plan.greedy_info_shots)
    greedy = plan.greedy_value(plan.everything, 0)
    optimal = plan.optimal_value(plan.everything, 0)
    assert info == pytest.approx(8.958, abs=1e-3)
    assert greedy_info == pytest.approx(8.899, abs=1e-3)
    assert optimal <= greedy_info < greedy < info


def test_information_gain_is_never_negative():
    plan = planner(MINI)
    layouts, fired = plan.everything, 0
    prior = log2(layouts.bit_count())
    for cell in plan.hit_counts(layouts, fired):
        assert plan.expected_entropy(layouts, fired, cell) <= prior + 1e-9


@pytest.mark.parametrize("name", ["infogain", "greedy-info"])
def test_agent_plays_every_layout(name):
    layouts = list(all_layouts(MINI))
    report = evaluate(BeliefAgent(planner=name), MINI, layouts=layouts)
    assert report.games == len(layouts)
    assert all(len(record.shots) >= sum(MINI.fleet) for record in report.records)


@pytest.mark.parametrize("name", ["infogain", "greedy-info"])
def test_needs_exact_belief(name):
    with pytest.raises(ValueError):
        BeliefAgent(belief="particles", planner=name)
