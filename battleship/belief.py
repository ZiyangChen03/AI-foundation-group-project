"""Exact belief states for the one-player game, viewed as a POMDP.

The hidden state s is the layout. It never changes, so T(s, a, s') is 1 when
s' = s and 0 otherwise. The feedback to a shot is fully determined by the
layout, so Omega(o | s', a) is 1 for the outcome s' would produce and 0
otherwise. The Bayes update

    b'(s')  ∝  Omega(o | s', a) * sum_s T(s, a, s') b(s)

therefore keeps exactly the layouts that would have produced the observed
outcome, and renormalizes. With the uniform prior (the distribution
``random_layout`` samples from) the surviving layouts stay equally likely, so
a belief is the set of layouts still consistent with the history.

Listing every layout is only practical on small boards such as MINI.
"""

from dataclasses import dataclass
from functools import cache
from itertools import islice

from .config import GameConfig
from .env import CellState, Observation, Result
from .layout import Layout, all_layouts

# What one shot reveals: the result, the length of the ship it sank, and the
# cells of that ship when the rules reveal them.
Outcome = tuple[Result, int | None, frozenset[int] | None]

MAX_LAYOUTS = 100_000


@dataclass(frozen=True)
class _Table:
    """Every layout of a config, precomputed for fast outcome prediction."""

    config: GameConfig
    layouts: tuple[Layout, ...]
    ship_at: tuple[tuple[int, ...], ...]   # ship_at[i][cell] = ship index or -1
    masks: tuple[tuple[int, ...], ...]     # masks[i][ship] = bitmask of its cells
    cover: tuple[int, ...]                 # cover[i] = bitmask of all ship cells


@cache
def _table(config: GameConfig) -> _Table:
    layouts = tuple(islice(all_layouts(config), MAX_LAYOUTS + 1))
    if len(layouts) > MAX_LAYOUTS:
        raise ValueError(f"more than {MAX_LAYOUTS} layouts on a {config.size}x{config.size} "
                         "board; an exact belief is only practical on small boards")
    ship_at, masks = [], []
    for layout in layouts:
        at = [-1] * config.num_cells
        for s, ship in enumerate(layout):
            for cell in ship.cells:
                at[cell] = s
        ship_at.append(tuple(at))
        masks.append(tuple(sum(1 << cell for cell in ship.cells) for ship in layout))
    cover = tuple(sum(m) for m in masks)
    return _Table(config, layouts, tuple(ship_at), tuple(masks), cover)


def _predict(table: _Table, i: int, fired: int, cell: int) -> Outcome:
    """The outcome layout ``i`` produces for a shot at ``cell`` after ``fired``."""
    s = table.ship_at[i][cell]
    if s < 0:
        return Result.MISS, None, None
    if table.masks[i][s] & ~(fired | 1 << cell):
        return Result.HIT, None, None
    ship = table.layouts[i][s]
    return Result.SUNK, ship.length, ship.cells if table.config.reveal_sunk_cells else None


def observed_outcome(before: Observation, after: Observation) -> Outcome:
    """The outcome of the shot that turned ``before`` into ``after``."""
    shot = after.history[-1]
    revealed = None
    if shot.result is Result.SUNK and after.config.reveal_sunk_cells:
        revealed = frozenset(cell for cell, state in enumerate(after.board)
                             if state is CellState.SUNK
                             and before.board[cell] is not CellState.SUNK)
    return shot.result, shot.sunk_length, revealed


def explains(layout: Layout, obs: Observation) -> bool:
    """True if ``layout`` would have produced every result in ``obs.history``.

    A plain replay, kept independent of ``Belief`` so it can check it.

    One gap: an Observation records which cells are sunk now, not which cells
    each sinking revealed. With two ships of the same length, layouts that
    differ only in which of them was sunk by which shot all pass here, though
    the board seen after each shot ruled some out. ``Belief.update``, applied
    shot by shot, sees every reveal and stays exact.
    """
    ship_of = {cell: ship for ship in layout for cell in ship.cells}
    fired = set()
    for shot in obs.history:
        fired.add(shot.cell)
        ship = ship_of.get(shot.cell)
        if ship is None:
            expected = Result.MISS, None
        elif ship.cells <= fired:
            expected = Result.SUNK, ship.length
        else:
            expected = Result.HIT, None
        if (shot.result, shot.sunk_length) != expected:
            return False
    if obs.config.reveal_sunk_cells:
        sunk = {cell for ship in layout if ship.cells <= fired for cell in ship.cells}
        if sunk != {cell for cell, state in enumerate(obs.board) if state is CellState.SUNK}:
            return False
    return True


class Belief:
    """A uniform distribution over the layouts still consistent with the shots.

    Immutable: ``update`` returns a new belief.
    """

    __slots__ = ("_table", "support", "fired")

    def __init__(self, table: _Table, support: tuple[int, ...], fired: int):
        self._table = table
        self.support = support   # indices of the consistent layouts
        self.fired = fired       # bitmask of the cells shot so far

    @classmethod
    def prior(cls, config: GameConfig) -> "Belief":
        """The belief before any shot: every legal layout, equally likely."""
        table = _table(config)
        return cls(table, tuple(range(len(table.layouts))), 0)

    @classmethod
    def from_observation(cls, obs: Observation) -> "Belief":
        """The belief after the shots in ``obs``, rebuilt from scratch.

        Exact unless sunk cells are revealed and two ships share a length;
        then it can keep extra layouts (see ``explains``).
        """
        table = _table(obs.config)
        support = tuple(i for i, layout in enumerate(table.layouts) if explains(layout, obs))
        return cls(table, support, sum(1 << shot.cell for shot in obs.history))

    @property
    def config(self) -> GameConfig:
        return self._table.config

    @property
    def layouts(self) -> list[Layout]:
        return [self._table.layouts[i] for i in self.support]

    def __len__(self) -> int:
        return len(self.support)

    @property
    def done(self) -> bool:
        """True once every ship is sunk (all consistent layouts agree on this)."""
        return not self._table.cover[self.support[0]] & ~self.fired

    def occupancy(self) -> list[float]:
        """P(a ship covers the cell) for each untried cell; 0 for cells already shot."""
        counts = [0] * self.config.num_cells
        for i in self.support:
            cover = self._table.cover[i] & ~self.fired
            while cover:
                low = cover & -cover
                counts[low.bit_length() - 1] += 1
                cover ^= low
        return [count / len(self.support) for count in counts]

    def outcomes(self, cell: int) -> dict[Outcome, "Belief"]:
        """The possible outcomes of a shot at ``cell``, each with the belief it
        leads to. ``len(next) / len(self)`` is P(outcome | belief, cell)."""
        groups = {}
        for i in self.support:
            groups.setdefault(_predict(self._table, i, self.fired, cell), []).append(i)
        fired = self.fired | 1 << cell
        return {outcome: Belief(self._table, tuple(group), fired)
                for outcome, group in groups.items()}

    def update(self, cell: int, outcome: Outcome) -> "Belief":
        """Bayes update after shooting ``cell`` and observing ``outcome``."""
        support = tuple(i for i in self.support
                        if _predict(self._table, i, self.fired, cell) == outcome)
        if not support:
            raise ValueError("no layout explains this outcome")
        return Belief(self._table, support, self.fired | 1 << cell)
