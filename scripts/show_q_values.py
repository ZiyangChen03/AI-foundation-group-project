"""Show the POMDP planner's numbers for every cell of a 4x4 board.

    uv run scripts/show_q_values.py                          # the empty board
    uv run scripts/show_q_values.py --shots B2:miss          # a board you describe
    uv run scripts/show_q_values.py --fleet 3 --shots B2:miss   # one 3-ship (the hand example)
    uv run scripts/show_q_values.py --play 17                # a whole optimal game on layout 17
    uv run scripts/show_q_values.py --disagreements          # boards where greedy is not optimal

For every cell not yet fired at, three numbers about the next shot:

  P(hit)     share of the layouts still possible that have a ship there.
             Greedy fires at the largest.
  info gain  how much a shot there narrows down the layouts: the entropy of
             its result, in bits. A planner that maximizes this is a third
             strategy, not the optimal one.
  Q*(b, a)   expected shots still needed if you fire there now and play
             optimally afterwards, with b the current belief (the set of
             layouts still possible):
                 Q*(b, a) = 1 + sum over results o of P(o | b, a) * V*(b after a, o)
                 V*(b)    = min over a of Q*(b, a)      (0 once every ship is sunk)
             The optimal planner fires at the smallest.

Every ship cell must be hit exactly once, so shots = ship cells + misses, and
with R ship cells still unhit, Q* splits exactly into two parts:

  Q*(b, a) - R  =  P(miss now)  +  E[misses later | fire at a, then play optimally]

Greedy minimizes the first part only. The optimal planner minimizes the sum:
the second part is what the shot's result is worth later (its value of
information, counted in shots). Raw information gain in bits is not the same
thing: a cell known to hold a ship teaches 0 bits but still has to be shot.

Shots are given as CELL:RESULT, with RESULT miss, hit or sunk; cells are a
column letter and a row number (A1 top left, D4 bottom right). When more than
one ship could have been sunk, name its cells: D1:sunk:C1-D1. Layouts are
numbered as in ``all_layouts`` (``--list-layouts`` prints them).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import GameConfig  # noqa: E402
from battleship.planning import Planner  # noqa: E402

RESULTS = ("miss", "hit", "sunk")


class Board:
    """One belief on a small board, with the planner that scores it."""

    def __init__(self, plan: Planner, layouts: int, fired: int, shown: dict[int, str]):
        self.plan, self.layouts, self.fired, self.shown = plan, layouts, fired, shown

    # --- the three numbers ---------------------------------------------------

    def count(self) -> int:
        return self.layouts.bit_count()

    def p_hit(self, cell: int) -> float:
        return (self.layouts & self.plan.covering[cell]).bit_count() / self.count()

    def info_gain(self, cell: int) -> float:
        n = self.count()
        return -sum(g.bit_count() / n * math.log2(g.bit_count() / n)
                    for g in self.plan.outcomes(self.layouts, self.fired, cell))

    def q(self, cell: int) -> float:
        return self.plan._q(self.layouts, self.fired, cell, self.plan.optimal_value)

    def misses_later(self, cell: int) -> float:
        """Expected misses after this shot with optimal play: Q* - R - P(miss now)."""
        return self.q(cell) - self.unhit_ship_cells() - (1 - self.p_hit(cell))

    def untried(self) -> list[int]:
        return [c for c in self.plan.cells if not self.fired >> c & 1]

    def unhit_ship_cells(self) -> int:
        return self.plan.total - (self.plan._ship_cells(self.layouts) & self.fired).bit_count()

    def done(self) -> bool:
        return self.plan._done(self.layouts, self.fired)

    # --- picks -----------------------------------------------------------------

    def best(self, score, largest: bool, tol: float = 1e-9) -> list[int]:
        values = {c: score(c) for c in self.untried()}
        target = max(values.values()) if largest else min(values.values())
        return [c for c, v in values.items() if abs(v - target) <= tol]


def name(cell: int, size: int) -> str:
    return "ABCDEFGH"[cell % size] + str(cell // size + 1)


def parse_cell(text: str, size: int) -> int:
    col, row = "ABCDEFGH".index(text[0].upper()), int(text[1:]) - 1
    if not (0 <= col < size and 0 <= row < size):
        raise ValueError(f"{text} is not on a {size}x{size} board")
    return row * size + col


def board_from_results(plan: Planner, shots: list[str]) -> Board:
    """The belief after shots given as CELL:RESULT."""
    size = plan.config.size
    layouts, fired, shown = plan.everything, 0, {}
    for shot in shots:
        text, _, result = shot.partition(":")
        result, _, ship = result.partition(":")
        cell, result = parse_cell(text, size), result.lower()
        if result not in RESULTS:
            raise ValueError(f"{shot}: the result must be one of {', '.join(RESULTS)}")
        if fired >> cell & 1:
            raise ValueError(f"{text} was already fired at")
        groups = plan.outcomes(layouts, fired, cell)
        covered = plan.covering[cell]
        miss = [g for g in groups if not g & covered]
        hit = [g for g in groups if g & covered and _sunk_cells(plan, g, fired | 1 << cell, cell) is None]
        sunk = [g for g in groups if g & covered and _sunk_cells(plan, g, fired | 1 << cell, cell) is not None]
        options = {"miss": miss, "hit": hit, "sunk": sunk}[result]
        if result == "sunk" and ship:
            named = sorted(parse_cell(c, size) for c in ship.split("-"))
            options = [g for g in options if _sunk_cells(plan, g, fired | 1 << cell, cell) == named]
        if not options:
            raise ValueError(f"{shot} cannot happen after the shots before it")
        if len(options) > 1:
            raise ValueError(f"{shot}: more than one ship could have been sunk; name its cells, "
                             f"e.g. {text}:sunk:<cell>-<cell>")
        layouts, fired = options[0], fired | 1 << cell
        shown[cell] = result
        if result == "sunk" and plan.reveal:
            for c in _sunk_cells(plan, layouts, fired, cell):
                shown[c] = "sunk"
    return Board(plan, layouts, fired, shown)


def _sunk_cells(plan: Planner, layouts: int, fired: int, cell: int):
    """Cells of the ship a shot at ``cell`` sank, the same in every layout of
    the group, or None if the shot only hit."""
    for mask, length, with_ship in plan.ships_at[cell]:
        if layouts & with_ship and not mask & ~fired:
            return [c for c in plan.cells if mask >> c & 1]
    return None


def board_on_layout(plan: Planner, index: int, cells: list[int]) -> Board:
    """The belief after firing at ``cells`` when the true layout is ``index``."""
    board = Board(plan, plan.everything, 0, {})
    for cell in cells:
        board = fire(board, cell, index)
    return board


def fire(board: Board, cell: int, index: int) -> Board:
    plan, after = board.plan, board.fired | 1 << cell
    group = next(g for g in plan.outcomes(board.layouts, board.fired, cell) if g >> index & 1)
    shown = dict(board.shown)
    sunk = _sunk_cells(plan, group, after, cell) if group & plan.covering[cell] else None
    if not group & plan.covering[cell]:
        shown[cell] = "miss"
    elif sunk is None:
        shown[cell] = "hit"
    else:
        for c in (sunk if plan.reveal else [cell]):
            shown[c] = "sunk"
    return Board(plan, group, after, shown)


# --- printing -------------------------------------------------------------------

MARK = {"miss": "  miss ", "hit": "  HIT  ", "sunk": "  SUNK "}


def grid(board: Board, title: str, value, fmt: str, picks: list[int]) -> list[str]:
    size = board.plan.config.size
    lines = [title, "     " + "".join(f"   {'ABCDEFGH'[c]}   " for c in range(size))]
    for r in range(size):
        row = f"  {r + 1}  "
        for c in range(size):
            cell = r * size + c
            if cell in board.shown:
                row += MARK[board.shown[cell]]
            elif board.fired >> cell & 1:
                row += "  hit  "
            else:
                row += (format(value(cell), fmt) + ("*" if cell in picks else " ")).rjust(7)
        lines.append(row)
    return lines


def show(board: Board, history: str = "") -> None:
    size = board.plan.config.size
    names = lambda cells: ", ".join(name(c, size) for c in cells)
    if history:
        print(f"Shots so far: {history}")
    if board.done():
        print("Every ship is sunk.\n")
        return
    n, left = board.count(), board.unhit_ship_cells()
    greedy = board.best(board.p_hit, largest=True)
    info = board.best(board.info_gain, largest=True)
    optimal = board.best(board.q, largest=False)
    v = board.q(optimal[0])
    print(f"{n} layout{'s' if n != 1 else ''} still possible; {left} ship cell{'s' if left != 1 else ''} not hit yet.")
    blocks = [grid(board, "P(hit)  (* = greedy's pick)", board.p_hit, ".2f", greedy),
              grid(board, "info gain, bits  (* = largest)", board.info_gain, ".3f", info),
              grid(board, "misses later if you fire here, playing optimally after", board.misses_later,
                   ".3f", []),
              grid(board, "Q*: expected shots left  (* = optimal pick)", board.q, ".3f", optimal)]
    for b in blocks:
        print("\n".join(b))
        print()
    print(f"  Q* = {left} unhit ship cells + P(miss now) + misses later")
    for label, picks in (("greedy fires at", greedy), ("max info gain fires", info), ("optimal fires at", optimal)):
        c = min(picks, key=board.q)
        print(f"  {label:21s} {names(picks):18s} Q* = {board.q(c):.3f} = {left} + "
              f"{1 - board.p_hit(c):.3f} + {board.misses_later(c):.3f}"
              + ("  = V*(b)" if picks is optimal else ""))
    print(f"  expected misses left with optimal play: {v - left:.3f}  (V* minus the {left} unhit ship cells)")
    print(f"  playing greedy from here instead:       "
          f"{board.plan.greedy_value(board.layouts, board.fired):.3f} shots expected\n")


def disagreements(plan: Planner):
    """Boards reached by optimal play (lowest cell among ties) where no greedy cell is optimal."""
    found, seen = [], set()

    def walk(board: Board, history: list[str]):
        if board.done():
            return
        key = plan._key(board.layouts, board.fired)
        if key in seen:
            return
        seen.add(key)
        optimal = board.best(board.q, largest=False)
        if not set(optimal) & set(board.best(board.p_hit, largest=True)):
            found.append((board, list(history)))
        cell = optimal[0]
        for index in _one_layout_per_outcome(plan, board, cell):
            nxt = fire(board, cell, index)
            walk(nxt, history + [label(nxt, cell)])

    walk(Board(plan, plan.everything, 0, {}), [])
    return found, len(seen)


def label(board: Board, cell: int) -> str:
    """The shot at ``cell`` as CELL:RESULT, naming the sunk ship's cells when revealed."""
    size, result = board.plan.config.size, board.shown.get(cell, "hit")
    text = f"{name(cell, size)}:{result}"
    if result == "sunk" and board.plan.reveal:
        cells = _sunk_cells(board.plan, board.layouts, board.fired, cell)
        text += ":" + "-".join(name(c, size) for c in cells)
    return text


def _one_layout_per_outcome(plan: Planner, board: Board, cell: int):
    for group in plan.outcomes(board.layouts, board.fired, cell):
        yield (group & -group).bit_length() - 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fleet", type=int, nargs="+", default=[3, 2], help="ship lengths (default 3 2)")
    p.add_argument("--no-reveal", action="store_true", help="sinking does not reveal the ship's cells")
    p.add_argument("--shots", nargs="*", default=[], help="shots so far, as CELL:RESULT")
    p.add_argument("--layout", type=int, help="true layout number: --shots then need only cells")
    p.add_argument("--play", type=int, metavar="LAYOUT", help="play a whole optimal game on this layout")
    p.add_argument("--disagreements", action="store_true", help="boards where greedy is not optimal")
    p.add_argument("--limit", type=int, default=3, help="boards to print with --disagreements")
    p.add_argument("--list-layouts", action="store_true")
    args = p.parse_args(argv)

    config = GameConfig(4, tuple(args.fleet), reveal_sunk_cells=not args.no_reveal)
    plan = Planner(config)
    layouts = plan.table.layouts
    v_start = plan.optimal_value(plan.everything, 0)
    print(f"4x4, ships {args.fleet}, {'sunk cells revealed' if plan.reveal else 'sunk cells not revealed'}: "
          f"{len(layouts)} layouts. V* of the empty board = {v_start:.4f}, "
          f"greedy = {plan.greedy_value(plan.everything, 0):.4f}\n")

    if args.list_layouts:
        for i, layout in enumerate(layouts):
            print(i, "  ".join("-".join(name(c, 4) for c in sorted(s.cells)) for s in layout))
        return
    if args.disagreements:
        found, total = disagreements(plan)
        print(f"Optimal play reaches {total} boards; at {len(found)} of them no greedy cell is optimal.\n")
        for board, history in found[:args.limit]:
            show(board, " ".join(history))
            print("  show again: --shots " + " ".join(history) + "\n")
        return
    if args.play is not None:
        board = Board(plan, plan.everything, 0, {})
        history = []
        while not board.done():
            show(board, " ".join(history) or "none")
            cell = board.best(board.q, largest=False)[0]
            board = fire(board, cell, args.play)
            history.append(label(board, cell))
        print(f"Sunk the fleet in {len(history)} shots: {' '.join(history)}")
        return
    if args.layout is not None:
        board = board_on_layout(plan, args.layout, [parse_cell(s.partition(":")[0], 4) for s in args.shots])
    else:
        board = board_from_results(plan, args.shots)
    show(board, " ".join(args.shots) or "none")


if __name__ == "__main__":
    main()
