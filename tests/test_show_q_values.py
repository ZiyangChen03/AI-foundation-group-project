import importlib.util
import random
from pathlib import Path

import pytest

from battleship import GameConfig
from battleship.planning import Planner

_script = Path(__file__).resolve().parents[1] / "scripts" / "show_q_values.py"
_spec = importlib.util.spec_from_file_location("show_q_values", _script)
show = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(show)

ONE_SHIP = GameConfig(4, (3,))


def test_hand_worked_example_numbers():
    """One 3-ship, after a miss at B2 (tests/test_planning.py works it by hand)."""
    board = show.board_from_results(Planner(ONE_SHIP), ["B2:miss"])
    C1, C3 = show.parse_cell("C1", 4), show.parse_cell("C3", 4)
    assert board.count() == 12
    assert board.best(board.p_hit, largest=True) == [C3]
    assert board.best(board.info_gain, largest=True) == [C3]
    assert C1 in board.best(board.q, largest=False)
    assert board.q(C3) == pytest.approx(16 / 3)
    assert board.q(C1) == pytest.approx(31 / 6)
    assert board.misses_later(C3) == pytest.approx(16 / 3 - 3 - 8 / 12)
    assert board.misses_later(C1) == pytest.approx(31 / 6 - 3 - 9 / 12)


@pytest.mark.parametrize("config", [ONE_SHIP, GameConfig(4, (2, 2)), GameConfig(4, (2, 2), reveal_sunk_cells=False)])
def test_results_typed_by_hand_give_the_same_belief_as_the_true_layout(config):
    plan = Planner(config)
    rng = random.Random(0)
    for _ in range(30):
        index = rng.randrange(len(plan.table.layouts))
        board = show.Board(plan, plan.everything, 0, {})
        shots = []
        while not board.done():
            cell = rng.choice(board.untried())
            board = show.fire(board, cell, index)
            shot = f"{show.name(cell, 4)}:{board.shown[cell]}"
            if board.shown[cell] == "sunk" and config.reveal_sunk_cells:
                sunk = show._sunk_cells(plan, board.layouts, board.fired, cell)
                shot += ":" + "-".join(show.name(c, 4) for c in sunk)
            shots.append(shot)
            typed = show.board_from_results(plan, shots)
            assert (typed.layouts, typed.fired) == (board.layouts, board.fired)
            assert typed.layouts >> index & 1


def test_q_splits_into_unhit_cells_miss_now_and_misses_later():
    board = show.board_from_results(Planner(GameConfig(4, (2, 2))), ["B2:hit", "C2:miss"])
    for cell in board.untried():
        assert board.q(cell) == pytest.approx(board.unhit_ship_cells() + (1 - board.p_hit(cell))
                                              + board.misses_later(cell))
        assert board.misses_later(cell) >= -1e-12


def test_impossible_shots_are_refused():
    plan = Planner(ONE_SHIP)
    with pytest.raises(ValueError):
        show.board_from_results(plan, ["B2:miss", "B2:hit"])         # fired twice
    with pytest.raises(ValueError):
        show.board_from_results(plan, ["A1:sunk"])                   # a 3-ship cannot sink in one shot
