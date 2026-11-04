"""Evaluation-only reader for the project's original raw-board Q checkpoints.

Supports non-symmetry checkpoints, including the historical 1M-episode run.
No dependency on a different checkout or its training/reporting code.
"""
import gzip
import json
from ..agents.base import Agent
from ..config import MINI


class RawQCheckpointAgent(Agent):
    name = 'raw-board-q'

    @classmethod
    def load(cls, path):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            data = json.load(stream)
        if data.get('version') not in (1, 2, 3, 4) or data.get('config') != 'MINI':
            raise ValueError('unsupported raw Q checkpoint')
        if data.get('symmetry', False):
            raise ValueError('use the original non-symmetry raw Q baseline')
        agent = cls()
        agent.initial = data.get('initial', 0.0)
        agent.q = {s: {int(a): v for a, v in row.items()} for s, row in data['q'].items()}
        return agent

    def new_game(self, config, rng):
        if config != MINI:
            raise ValueError('only MINI supported')
        super().new_game(config, rng)

    def select_shot(self, obs):
        key = ''.join(c.value for c in obs.board) + ':' + ','.join(map(str, obs.remaining_ship_lengths))
        row = self.q.get(key, {})
        best = max(row.get(a, self.initial) for a in obs.legal_actions)
        return self.rng.choice([a for a in obs.legal_actions if row.get(a, self.initial) == best])
