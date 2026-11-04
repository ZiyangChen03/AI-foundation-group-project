"""Belief-state Q-learning: updates, equivalence with the environment, learning."""

import random

import pytest

from battleship import CLASSIC, MINI, all_layouts, evaluate, play_solo, train
from battleship.agents import BeliefQAgent
from battleship.planning import planner


def test_rejects_boards_without_an_exact_belief():
    with pytest.raises(ValueError, match="small boards"):
        BeliefQAgent(config=CLASSIC)


def test_update_moves_toward_the_target():
    agent = BeliefQAgent(alpha=0.5, initial=-20.0)    # fixed learning rate
    agent.update("s", 3, -1.0)                  # terminal shot
    assert agent.value("s", 3) == pytest.approx(-10.5)
    agent.update("s", 3, -1.0)
    assert agent.value("s", 3) == pytest.approx(-5.75)
    assert agent.value("s", 4) == -20.0         # untouched actions keep the initial value


def test_default_step_size_averages_the_targets():
    agent = BeliefQAgent()                     # alpha = 1 / visits
    for target in (-3.0, -5.0, -4.0):
        agent.update("s", 0, target)
    assert agent.value("s", 0) == pytest.approx(-4.0)


def test_untrained_agent_finishes_every_layout():
    agent = BeliefQAgent()
    report = evaluate(agent, MINI, layouts=list(all_layouts(MINI)))
    assert report.games == 264
    assert 9.5 < agent.exact_value() < 13.0      # uniform over cells that could hold a ship


def test_simulated_beliefs_match_the_environment():
    """fast_train's transitions are the ones the environment produces."""
    plan = planner(MINI)
    layouts = plan.table.layouts
    for i, layout in enumerate(list(all_layouts(MINI))[::11]):
        record = play_solo(BeliefQAgent(), MINI, layout=layout, agent_seed=i)
        hidden = layouts.index(layout)
        state, fired = plan.everything, 0
        for n, shot in enumerate(record.shots, 1):
            assert not plan._done(state, fired)
            state = next(g for g in plan.outcomes(state, fired, shot.cell) if g >> hidden & 1)
            fired |= 1 << shot.cell
            assert state >> hidden & 1
        assert plan._done(state, fired)


def test_fast_training_learns():
    agent = BeliefQAgent()
    before = agent.exact_value()
    agent.fast_train(60_000, random.Random(0))
    after = agent.exact_value()
    assert after < 10.0 < 10.2 < before         # untrained: uniform over possible cells
    assert agent.training is False


def test_training_through_the_environment_updates_the_table():
    agent = BeliefQAgent(epsilon=0.3)
    train(agent, MINI, 300)
    assert agent.num_states() > 0
    assert agent.training is False


def test_evaluation_does_not_change_the_table():
    agent = BeliefQAgent()
    agent.fast_train(2_000, random.Random(1))
    snapshot = {k: dict(v) for k, v in agent.q.items()}
    evaluate(agent, MINI, layouts=list(all_layouts(MINI)))
    assert agent.q == snapshot


def test_save_and_load(tmp_path):
    agent = BeliefQAgent()
    agent.fast_train(2_000, random.Random(2))
    path = tmp_path / "q.json.gz"
    agent.save(str(path))
    loaded = BeliefQAgent.load(str(path))
    assert loaded.q == agent.q and loaded.visits == agent.visits
    assert loaded.board == agent.board and loaded.initial == agent.initial
