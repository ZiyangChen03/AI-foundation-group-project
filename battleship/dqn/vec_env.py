"""Many Battleship games at once, stored as tensors, for fast DQN training.

Each row is one game. The rules are exactly those of ``BattleshipEnv``:
tests/test_dqn.py replays random games through both and compares every board.
Layouts come from ``random_layout``, so the boards are drawn from the same
distribution as ``play_solo`` and ``evaluate``.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from ..config import GameConfig
from ..env import CellState, Observation
from ..layout import Layout, random_layout

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3
CODES = {CellState.UNKNOWN: UNKNOWN, CellState.MISS: MISS, CellState.HIT: HIT,
         CellState.SUNK: SUNK}
MAX_LENGTH = 5          # one "ships remaining" plane per length 1..5


class VecBattleship:
    """``num_games`` games side by side. Finished games keep their final board
    until ``reset`` is called for them."""

    def __init__(self, config: GameConfig, num_games: int = 0, seed: int = 0,
                 layouts: list[Layout] | None = None):
        if max(config.fleet) > MAX_LENGTH:
            raise ValueError(f"ships longer than {MAX_LENGTH} are not supported")
        self.config = config
        self.rng = random.Random(seed)
        n = len(layouts) if layouts is not None else num_games
        cells, ships = config.num_cells, len(config.fleet)
        self.n = n
        self.board = torch.zeros(n, cells, dtype=torch.int8)
        self.ship_at = torch.full((n, cells), -1, dtype=torch.long)
        self.lengths = torch.zeros(n, ships, dtype=torch.long)
        self.left = torch.zeros(n, ships, dtype=torch.long)
        self.shots = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            self._place(i, None if layouts is None else layouts[i])

    def _place(self, i: int, layout: Layout | None) -> None:
        if layout is None:
            layout = random_layout(self.config, self.rng)
        self.board[i] = UNKNOWN
        self.ship_at[i] = -1
        for s, ship in enumerate(layout):
            self.ship_at[i, list(ship.cells)] = s
            self.lengths[i, s] = self.left[i, s] = ship.length
        self.shots[i] = 0

    def reset(self, rows: torch.Tensor) -> None:
        """Start a new random game in every row where ``rows`` is True."""
        for i in rows.nonzero().flatten().tolist():
            self._place(i, None)

    @property
    def done(self) -> torch.Tensor:
        return (self.left == 0).all(1)

    def legal(self) -> torch.Tensor:
        return self.board == UNKNOWN

    def remaining(self) -> torch.Tensor:
        """Ships still afloat, counted by length: column L-1 for length L."""
        counts = torch.zeros(self.n, MAX_LENGTH, dtype=torch.long)
        return counts.scatter_add_(1, self.lengths - 1, (self.left > 0).long())

    def step(self, actions: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """Fire ``actions[i]`` in every active row; returns which games are now done."""
        rows = torch.arange(self.n) if active is None else active.nonzero().flatten()
        cells = actions[rows]
        if (self.board[rows, cells] != UNKNOWN).any():
            raise ValueError("a cell was fired at twice")
        ship = self.ship_at[rows, cells]
        hit = ship >= 0
        self.board[rows, cells] = torch.where(hit, HIT, MISS).to(torch.int8)
        hit_rows, hit_ships = rows[hit], ship[hit]
        self.left[hit_rows, hit_ships] -= 1
        sunk = self.left[hit_rows, hit_ships] == 0
        if self.config.reveal_sunk_cells and sunk.any():
            sunk_rows, sunk_ships = hit_rows[sunk], hit_ships[sunk]
            cover = self.ship_at[sunk_rows] == sunk_ships[:, None]
            self.board[sunk_rows] = torch.where(cover, SUNK, self.board[sunk_rows]).to(torch.int8)
        self.shots[rows] += 1
        return self.done


def board_codes(obs: Observation) -> torch.Tensor:
    return torch.tensor([CODES[state] for state in obs.board], dtype=torch.int8)


def remaining_counts(obs: Observation) -> torch.Tensor:
    counts = torch.zeros(MAX_LENGTH, dtype=torch.long)
    for length in obs.remaining_ship_lengths:
        counts[length - 1] += 1
    return counts


def density(board: torch.Tensor, remaining: torch.Tensor, config: GameConfig,
            hit_weight: int = 50) -> torch.Tensor:
    """``ProbabilityDensityAgent.density`` for a batch of boards, in float64.

    Each remaining ship is tried in every position that avoids misses and sunk
    cells; a position through k open hits adds ``hit_weight ** k`` to each
    unknown cell it covers."""
    n, size = board.shape[0], config.size
    grid = board.view(n, 1, size, size)
    blocked = ((grid == MISS) | (grid == SUNK)).double()
    hits = (grid == HIT).double()
    score = torch.zeros(n, 1, size, size, dtype=torch.float64)
    for length in range(1, MAX_LENGTH + 1):
        count = remaining[:, length - 1].double().view(n, 1, 1, 1)
        if length > size or not count.any():
            continue
        for shape in ((1, length),) if length == 1 else ((1, length), (length, 1)):
            kernel = torch.ones(1, 1, *shape, dtype=torch.float64)
            fits = F.conv2d(blocked, kernel) == 0
            weight = fits * hit_weight ** F.conv2d(hits, kernel) * count
            score += F.conv_transpose2d(weight, kernel)
    return score.view(n, -1) * (board == UNKNOWN)


def normalized_density(board: torch.Tensor, remaining: torch.Tensor,
                       config: GameConfig) -> torch.Tensor:
    """The density map divided by its maximum, so every board's best cell is 1."""
    d = density(board, remaining, config)
    return (d / d.amax(1, keepdim=True).clamp(min=1e-300)).float()
