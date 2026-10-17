"""Game configuration and cell helpers.

Cells are integers ``0 .. size*size - 1`` in row-major order
(``cell = row * size + col``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Board size, fleet, and rules shared by every agent."""

    size: int = 10
    fleet: tuple[int, ...] = (5, 4, 3, 3, 2)
    reveal_sunk_cells: bool = True

    def __post_init__(self):
        object.__setattr__(self, "fleet", tuple(self.fleet))
        if self.size < 1:
            raise ValueError("size must be at least 1")
        if not self.fleet:
            raise ValueError("fleet must contain at least one ship")
        if any(not 1 <= length <= self.size for length in self.fleet):
            raise ValueError(f"ship lengths must be between 1 and {self.size}")
        if sum(self.fleet) > self.num_cells:
            raise ValueError("fleet does not fit on the board")

    @property
    def num_cells(self) -> int:
        return self.size * self.size


MINI = GameConfig(size=4, fleet=(3, 2))
CLASSIC = GameConfig(size=10, fleet=(5, 4, 3, 3, 2))

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def to_rc(cell: int, config: GameConfig) -> tuple[int, int]:
    return divmod(cell, config.size)


def to_cell(row: int, col: int, config: GameConfig) -> int:
    return row * config.size + col


def label(cell: int, config: GameConfig) -> str:
    """Human-readable name of a cell, e.g. 0 -> "A1"."""
    row, col = to_rc(cell, config)
    return f"{chr(65 + col)}{row + 1}"


def neighbors(cell: int, config: GameConfig) -> list[int]:
    """Cells directly above, below, left and right of ``cell``."""
    row, col = to_rc(cell, config)
    return [to_cell(row + dr, col + dc, config)
            for dr, dc in DIRECTIONS
            if 0 <= row + dr < config.size and 0 <= col + dc < config.size]
