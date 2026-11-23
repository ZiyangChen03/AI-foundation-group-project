"""A trained Q-network as an ``Agent``, so it plays through ``play_solo`` and
``evaluate`` like every other agent: it sees only the public observation and
fires at the legal cell with the highest Q (the first one on ties)."""

from __future__ import annotations

import torch

from ..agents.base import Agent
from ..config import GameConfig
from . import model as qmodel
from .vec_env import board_codes, remaining_counts


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, model: qmodel.QNet, config: GameConfig, name: str | None = None):
        super().__init__()
        self.model = model.to("cpu").eval()
        self.board = config
        if name:
            self.name = name

    def new_game(self, config, rng):
        if config != self.board:
            raise ValueError(f"this network was trained on {self.board}, not {config}")
        super().new_game(config, rng)

    def q_values(self, obs) -> torch.Tensor:
        board = board_codes(obs).unsqueeze(0)
        remaining = remaining_counts(obs).unsqueeze(0)
        x, legal, dens = qmodel.features(board, remaining, obs.config, self.model.net)
        with torch.no_grad():
            return self.model(x, legal, dens)[0]

    def select_shot(self, obs):
        return int(self.q_values(obs).argmax())

    def save(self, path: str) -> None:
        qmodel.save(path, self.model, self.board)

    @classmethod
    def load(cls, path: str) -> DQNAgent:
        model, config, info = qmodel.load(path)
        return cls(model, config, info.get("name"))
