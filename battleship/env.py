"""The one-player Battleship environment and the public information agents see."""

import operator
import random
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import NamedTuple

from .config import MINI, GameConfig, label
from .layout import Layout, random_layout, validate_layout


class CellState(Enum):
    UNKNOWN = "?"
    MISS = "o"
    HIT = "x"
    SUNK = "#"


class Result(Enum):
    MISS = "miss"
    HIT = "hit"
    SUNK = "sunk"


@dataclass(frozen=True)
class Shot:
    cell: int
    result: Result
    sunk_length: int | None = None   # set only when result is SUNK


@dataclass(frozen=True)
class Observation:
    """Everything an agent is allowed to see."""

    config: GameConfig
    board: tuple[CellState, ...]
    history: tuple[Shot, ...]                 # in firing order
    remaining_ship_lengths: tuple[int, ...]   # sorted descending

    @cached_property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(cell for cell, state in enumerate(self.board)
                     if state is CellState.UNKNOWN)

    @property
    def num_shots(self) -> int:
        return len(self.history)


def is_consistent(cells: frozenset[int], obs: Observation) -> bool:
    """True if a ship could still occupy ``cells``: none of them is a miss or
    part of a ship already sunk."""
    return all(obs.board[cell] is not CellState.MISS
               and obs.board[cell] is not CellState.SUNK
               for cell in cells)


class StepResult(NamedTuple):
    observation: Observation
    reward: float
    done: bool
    shot: Shot


class BattleshipEnv:
    """Hidden fleet that an agent fires at until every ship is sunk.

    Reward is -1 per shot, so maximizing reward means winning in fewer shots.
    """

    REWARD = -1.0

    def __init__(self, config: GameConfig = MINI):
        self.config = config
        self.reset()

    def reset(self, seed: int | None = None, layout: Layout | None = None) -> Observation:
        """Start a new game on ``layout`` if given, else on a random layout
        generated from ``seed``."""
        if layout is None:
            layout = random_layout(self.config, random.Random(seed))
        else:
            layout = tuple(layout)
            validate_layout(self.config, layout)
        self._layout = layout
        self._ship_at = {cell: i for i, ship in enumerate(layout) for cell in ship.cells}
        self._hits_left = [ship.length for ship in layout]
        self._board = [CellState.UNKNOWN] * self.config.num_cells
        self._history = []
        return self.observation()

    @property
    def done(self) -> bool:
        return not any(self._hits_left)

    def observation(self) -> Observation:
        remaining = sorted((ship.length for ship, left in zip(self._layout, self._hits_left)
                            if left), reverse=True)
        return Observation(self.config, tuple(self._board), tuple(self._history),
                           tuple(remaining))

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        return [cell for cell, state in enumerate(self._board) if state is CellState.UNKNOWN]

    def step(self, cell: int) -> StepResult:
        if self.done:
            raise ValueError("game already finished")
        if isinstance(cell, bool):
            raise ValueError(f"cell must be an integer, got {cell!r}")
        try:
            cell = operator.index(cell)
        except TypeError:
            raise ValueError(f"cell must be an integer, got {cell!r}") from None
        if not 0 <= cell < self.config.num_cells:
            raise ValueError(f"cell must be between 0 and {self.config.num_cells - 1}, got {cell}")
        if self._board[cell] is not CellState.UNKNOWN:
            raise ValueError(f"{label(cell, self.config)} has already been shot")

        ship = self._ship_at.get(cell)
        sunk_length = None
        if ship is None:
            result = Result.MISS
            self._board[cell] = CellState.MISS
        else:
            self._hits_left[ship] -= 1
            self._board[cell] = CellState.HIT
            result = Result.HIT
            if self._hits_left[ship] == 0:
                result = Result.SUNK
                sunk_length = self._layout[ship].length
                if self.config.reveal_sunk_cells:
                    for sunk_cell in self._layout[ship].cells:
                        self._board[sunk_cell] = CellState.SUNK
        shot = Shot(cell, result, sunk_length)
        self._history.append(shot)
        return StepResult(self.observation(), self.REWARD, self.done, shot)

    def reveal_layout(self) -> Layout:
        """The hidden layout, available only once the game is over."""
        if not self.done:
            raise RuntimeError("the layout is hidden until the game is over")
        return self._layout
