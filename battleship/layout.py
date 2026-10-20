"""Ship layouts: random placement, predefined layouts, and enumeration.

Ships are straight (horizontal or vertical), may touch, and may not overlap.
"""

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache

from .config import GameConfig, to_cell


@dataclass(frozen=True)
class Ship:
    length: int
    cells: frozenset[int]


Layout = tuple[Ship, ...]


@cache
def placements(config: GameConfig, length: int) -> tuple[frozenset[int], ...]:
    """Every position a ship of ``length`` can take on an empty board."""
    found = {}
    for row in range(config.size):
        for col in range(config.size):
            for dr, dc in ((0, 1), (1, 0)):
                if (row + dr * (length - 1) < config.size
                        and col + dc * (length - 1) < config.size):
                    cells = frozenset(to_cell(row + dr * i, col + dc * i, config)
                                      for i in range(length))
                    found.setdefault(cells, None)
    return tuple(found)


def random_layout(config: GameConfig, rng: random.Random,
                  max_attempts: int = 100_000) -> Layout:
    """A layout drawn uniformly from all valid layouts (rejection sampling)."""
    options = [placements(config, length) for length in config.fleet]
    for _ in range(max_attempts):
        chosen = [rng.choice(cells) for cells in options]
        if sum(map(len, chosen)) == len(frozenset().union(*chosen)):
            return tuple(Ship(len(cells), cells) for cells in chosen)
    raise ValueError("could not place the fleet; the board may be too crowded")


def validate_layout(config: GameConfig, layout: Layout) -> None:
    """Raise ValueError unless ``layout`` is a legal placement of the fleet."""
    lengths = sorted(ship.length for ship in layout)
    if lengths != sorted(config.fleet):
        raise ValueError(f"ship lengths must be {sorted(config.fleet)}, got {lengths}")
    occupied = set()
    for ship in layout:
        if len(ship.cells) != ship.length or ship.cells not in placements(config, ship.length):
            raise ValueError(f"ship {sorted(ship.cells)} is not a straight line of "
                             f"{ship.length} cells on the board")
        if ship.cells & occupied:
            raise ValueError(f"ship {sorted(ship.cells)} overlaps another ship")
        occupied |= ship.cells


def layout_from_cells(config: GameConfig, ships: Iterable[Iterable[int]]) -> Layout:
    """Build and validate a predefined layout, e.g. ``[[0, 1, 2], [8, 12]]``."""
    layout = tuple(Ship(len(cells), cells)
                   for cells in (frozenset(ship) for ship in ships))
    validate_layout(config, layout)
    return layout


def all_layouts(config: GameConfig) -> Iterator[Layout]:
    """Every distinct layout, each exactly once.

    Ships of equal length are interchangeable, so only one ordering of them is
    produced. Only practical for small boards such as MINI.
    """
    fleet = config.fleet
    options = [placements(config, length) for length in fleet]
    previous_same = [max((j for j in range(i) if fleet[j] == fleet[i]), default=None)
                     for i in range(len(fleet))]
    indices, ships = [], []

    def extend(i, occupied):
        if i == len(fleet):
            yield tuple(ships)
            return
        start = 0 if previous_same[i] is None else indices[previous_same[i]] + 1
        for k in range(start, len(options[i])):
            cells = options[i][k]
            if not cells & occupied:
                indices.append(k)
                ships.append(Ship(fleet[i], cells))
                yield from extend(i + 1, occupied | cells)
                indices.pop()
                ships.pop()

    yield from extend(0, frozenset())
