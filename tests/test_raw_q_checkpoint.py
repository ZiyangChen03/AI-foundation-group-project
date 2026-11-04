import gzip
import json
import random
import pytest
from battleship import MINI
from battleship.env import BattleshipEnv
from battleship.opponent_belief_q.raw_q_checkpoint import RawQCheckpointAgent


def test_raw_checkpoint_uses_original_state_and_legal_max(tmp_path):
    path = tmp_path / 'raw.gz'
    env = BattleshipEnv(MINI)
    obs = env.reset(seed=0)
    with gzip.open(path, 'wt') as stream:
        json.dump(dict(version=1, config='MINI', q={'?' * 16 + ':3,2': {'5': 10}}), stream)
    agent = RawQCheckpointAgent.load(path)
    agent.new_game(MINI, random.Random(0))
    assert agent.select_shot(obs) == 5
    obs = env.step(5).observation
    assert agent.select_shot(obs) in obs.legal_actions


def test_symmetry_checkpoints_are_not_silently_misread(tmp_path):
    path = tmp_path / 'raw.gz'
    with gzip.open(path, 'wt') as stream:
        json.dump(dict(version=4, config='MINI', symmetry=True, q={}), stream)
    with pytest.raises(ValueError, match='non-symmetry'):
        RawQCheckpointAgent.load(path)
