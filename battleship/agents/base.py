"""The interface every agent implements."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import GameConfig
    from ..env import Observation, StepResult
    from ..runner import GameRecord


class Agent(ABC):
    """A shooter. Ships are always placed randomly by the runner.

    Subclasses that define ``__init__`` must call ``super().__init__()``.
    """

    name: str = "agent"

    def __init__(self):
        self.training = False   # learned agents explore / update only when True
        self.config: GameConfig | None = None
        self.rng = random.Random()

    def new_game(self, config: GameConfig, rng: random.Random) -> None:
        """Reset per-game state. ``rng`` is seeded by the runner for reproducibility."""
        self.config = config
        self.rng = rng

    @abstractmethod
    def select_shot(self, obs: Observation) -> int:
        """Return a cell in ``obs.legal_actions``."""

    def observe(self, obs: Observation, action: int, result: StepResult) -> None:
        """Called after every shot with the observation the shot was chosen from."""

    def end_game(self, record: GameRecord) -> None:
        """Called once after each game with the full record, including the
        revealed layout."""

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} has nothing to save")

    @classmethod
    def load(cls, path: str) -> Agent:
        raise NotImplementedError(f"{cls.__name__} cannot be loaded")

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"
