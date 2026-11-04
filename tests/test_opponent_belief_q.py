import random
import pytest
from battleship import MINI, play_solo
from battleship.belief import Belief
from battleship.opponent_belief_q.agent import OpponentBeliefQAgent


def test_prior_validation():
    with pytest.raises(ValueError):
        OpponentBeliefQAgent([0] * 264)


def test_roundtrip_and_frozen_evaluation(tmp_path):
    agent = OpponentBeliefQAgent(range(1, 265))
    agent.fast_train(100, random.Random(7))
    path = tmp_path / 'model.gz'
    agent.save(path)
    restored = OpponentBeliefQAgent.load(path)
    assert restored.q == agent.q
    assert restored.visits == agent.visits
    assert restored.prior_weights == pytest.approx(agent.prior_weights)
    snapshot = {k: dict(v) for k, v in restored.q.items()}
    for layout in Belief.prior(MINI).layouts[::11]:
        record = play_solo(restored, MINI, layout=layout, agent_seed=0)
        assert len({shot.cell for shot in record.shots}) == record.num_shots
    assert restored.q == snapshot


def test_weighted_sampling_and_terminal_update():
    class RecordedRandom(random.Random):
        def choices(self, population, weights, **kwargs):
            self.seen = tuple(weights)
            return [0]
    rng = RecordedRandom(0)
    agent = OpponentBeliefQAgent(range(1, 265))
    agent.fast_train(1, rng)
    assert rng.seen == agent.prior_weights
    assert any(-1.0 in row.values() for row in agent.q.values())
    assert not agent.training


def test_new_game_uses_learned_weights():
    agent = OpponentBeliefQAgent(range(1, 265))
    agent.new_game(MINI, random.Random(0))
    assert agent.belief.weights == agent.prior_weights
