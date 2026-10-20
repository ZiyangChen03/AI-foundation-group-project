import random
from itertools import product

import pytest

from battleship import (CLASSIC, MINI, GameConfig, Ship, all_layouts, label,
                        layout_from_cells, neighbors, placements, random_layout,
                        to_cell, to_rc, validate_layout)


def test_cell_helpers():
    assert to_rc(6, MINI) == (1, 2)
    assert to_cell(1, 2, MINI) == 6
    assert label(0, MINI) == "A1"
    assert label(99, CLASSIC) == "J10"
    assert sorted(neighbors(0, MINI)) == [1, 4]
    assert sorted(neighbors(5, MINI)) == [1, 4, 6, 9]


def test_config_rejects_bad_fleets():
    with pytest.raises(ValueError):
        GameConfig(size=4, fleet=(5,))
    with pytest.raises(ValueError):
        GameConfig(size=2, fleet=(2, 2, 2))


def test_placement_counts():
    assert len(placements(MINI, 3)) == 16
    assert len(placements(MINI, 2)) == 24
    assert len(placements(MINI, 1)) == 16          # no duplicate horizontal/vertical
    assert len(placements(CLASSIC, 5)) == 120


def brute_force_layouts(config):
    """Distinct boards, as sets of ship placements."""
    options = [placements(config, length) for length in config.fleet]
    boards = set()
    for chosen in product(*options):
        if sum(map(len, chosen)) == len(frozenset().union(*chosen)):
            boards.add(frozenset(zip(config.fleet, chosen)))
    return boards


@pytest.mark.parametrize("config", [MINI, GameConfig(size=4, fleet=(2, 2)),
                                    GameConfig(size=4, fleet=(2, 2, 3))])
def test_all_layouts_matches_brute_force(config):
    layouts = list(all_layouts(config))
    boards = {frozenset((ship.length, ship.cells) for ship in layout) for layout in layouts}
    assert len(boards) == len(layouts)             # no duplicates
    assert boards == brute_force_layouts(config)
    for layout in layouts:
        validate_layout(config, layout)


def test_random_layout_is_valid_and_seeded():
    for seed in range(200):
        validate_layout(CLASSIC, random_layout(CLASSIC, random.Random(seed)))
    assert random_layout(MINI, random.Random(7)) == random_layout(MINI, random.Random(7))


def test_random_layout_covers_every_mini_layout():
    rng = random.Random(0)
    seen = {random_layout(MINI, rng) for _ in range(20_000)}
    assert len(seen) == len(list(all_layouts(MINI)))


def test_layout_from_cells_and_validation():
    layout = layout_from_cells(MINI, [[0, 1, 2], [8, 12]])
    assert layout == (Ship(3, frozenset({0, 1, 2})), Ship(2, frozenset({8, 12})))
    with pytest.raises(ValueError, match="overlaps"):
        layout_from_cells(MINI, [[0, 1, 2], [2, 6]])
    with pytest.raises(ValueError, match="straight"):
        layout_from_cells(MINI, [[0, 1, 5], [8, 12]])   # bent
    with pytest.raises(ValueError, match="straight"):
        layout_from_cells(MINI, [[3, 4, 5], [8, 12]])   # wraps around the edge
    with pytest.raises(ValueError, match="lengths"):
        layout_from_cells(MINI, [[0, 1], [8, 12]])
