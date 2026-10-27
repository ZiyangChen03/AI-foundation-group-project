"""Exact planning in the belief-state MDP (small boards only).

A POMDP becomes an MDP whose states are beliefs. Here a belief b is the set
of layouts still consistent with the shots, each shot costs 1, and there is
no discount, so the optimal expected number of shots left satisfies

    V*(b) = 0                                   if every ship is sunk
    V*(b) = min over cells a of Q*(b, a)        otherwise
    Q*(b, a) = 1 + sum over outcomes o of P(o | b, a) * V*(b_a^o)

where b_a^o is the Bayes update of b after firing at a and observing o.
``V`` solves this by recursion over the beliefs reachable from b, memoized.

The prior is uniform over legal layouts (how ``random_layout`` places ships), so
V* is optimal against random placement; an opponent with placement habits would
need a different prior.

The same recursion evaluates a fixed policy: replace the min by the average
over the cells the policy may pick. ``policy_value`` does this for greedy
(fire at the most likely cell), and ``rollout`` improves on greedy by one
step of lookahead: Q_greedy(b, a) = 1 + sum_o P(o | b, a) V_greedy(b_a^o).
By the policy improvement theorem rollout is never worse than greedy.

Beliefs are bitmasks over the layouts of ``belief._table`` (one bit per
layout), so splitting a belief by outcome takes a few integer operations.
Three shortcuts keep V* fast without changing it:

  - A cell no layout covers is a certain miss: it costs a shot and teaches
    nothing, so it is never fired.
  - A cell every layout covers is a certain hit. It must be fired at some
    point and firing it first only gives information sooner, so when one
    exists it is the only move considered.
  - With R ship cells still unhit, Q*(b, a) >= R + 1 - P(hit at a): every one
    of them takes a shot, and a miss at a costs one more. Cells are tried from
    the most to the least likely hit, stopping once this bound cannot beat
    the best value found.
"""

from functools import cache

from .belief import Belief, _table
from .config import GameConfig


class Planner:
    """Exact optimal, greedy and rollout values for one config."""

    def __init__(self, config: GameConfig):
        table = _table(config)
        self.config = config
        self.table = table
        self.cells = range(config.num_cells)
        self.total = sum(config.fleet)
        self.reveal = config.reveal_sunk_cells
        n = len(table.layouts)
        self.covering = [0] * config.num_cells     # cell -> layouts with a ship there
        ship_layouts = {}                           # (ship cells, length) -> layouts
        for i, layout in enumerate(table.layouts):
            for ship in layout:
                mask = sum(1 << cell for cell in ship.cells)
                key = (mask, ship.length)
                ship_layouts[key] = ship_layouts.get(key, 0) | 1 << i
                for cell in ship.cells:
                    self.covering[cell] |= 1 << i
        self.ships_at = [[(mask, length, layouts)
                          for (mask, length), layouts in ship_layouts.items() if mask >> cell & 1]
                         for cell in self.cells]
        self.everything = (1 << n) - 1
        self._optimal = {}
        self._greedy = {}

    # --- beliefs as bitmasks -------------------------------------------------

    def state(self, belief: Belief) -> tuple[int, int]:
        if belief._table is not self.table:
            raise ValueError("belief belongs to another config")
        return sum(1 << i for i in belief.support), belief.fired

    def _ship_cells(self, layouts: int) -> int:
        return self.table.cover[(layouts & -layouts).bit_length() - 1]

    def _done(self, layouts: int, fired: int) -> bool:
        return (self._ship_cells(layouts) & fired).bit_count() == self.total

    def _key(self, layouts: int, fired: int) -> tuple[int, int]:
        # Misses no consistent layout could cover no longer matter; the
        # fired hits do, and every consistent layout agrees on them.
        return layouts, fired & self._ship_cells(layouts)

    def hit_counts(self, layouts: int, fired: int) -> dict[int, int]:
        """Number of consistent layouts with a ship at each untried cell, for
        the cells where it is not zero."""
        counts = {}
        for cell in self.cells:
            if not fired >> cell & 1:
                k = (layouts & self.covering[cell]).bit_count()
                if k:
                    counts[cell] = k
        return counts

    def outcomes(self, layouts: int, fired: int, cell: int) -> list[int]:
        """The beliefs a shot at ``cell`` can lead to, one per outcome."""
        after = fired | 1 << cell
        groups, sunk = {}, 0
        for mask, length, with_ship in self.ships_at[cell]:
            if not mask & ~after:                        # this shot would sink it
                group = layouts & with_ship
                if group:
                    key = mask if self.reveal else length
                    groups[key] = groups.get(key, 0) | group
                    sunk |= group
        hit = layouts & self.covering[cell] & ~sunk
        miss = layouts & ~self.covering[cell]
        return [g for g in (*groups.values(), hit, miss) if g]

    def _q(self, layouts: int, fired: int, cell: int, value) -> float:
        size = layouts.bit_count()
        after = fired | 1 << cell
        return 1 + sum(g.bit_count() / size * value(g, after)
                       for g in self.outcomes(layouts, fired, cell))

    # --- optimal -------------------------------------------------------------

    def optimal_value(self, layouts: int, fired: int) -> float:
        """V*(b): the fewest shots still needed on average."""
        if self._done(layouts, fired):
            return 0.0
        key = self._key(layouts, fired)
        if key in self._optimal:
            return self._optimal[key]
        size = layouts.bit_count()
        ranked = sorted(self.hit_counts(layouts, fired).items(), key=lambda kv: -kv[1])
        if ranked[0][1] == size:
            ranked = ranked[:1]
        remaining = self.total - (self._ship_cells(layouts) & fired).bit_count()
        best = float("inf")
        for cell, count in ranked:
            if remaining + 1 - count / size >= best:
                break
            best = min(best, self._q(layouts, fired, cell, self.optimal_value))
        self._optimal[key] = best
        return best

    def optimal_shots(self, layouts: int, fired: int, tolerance: float = 1e-9) -> list[int]:
        """Every cell that achieves V*(b)."""
        best = self.optimal_value(layouts, fired)
        return [cell for cell in self.hit_counts(layouts, fired)
                if self._q(layouts, fired, cell, self.optimal_value) <= best + tolerance]

    # --- greedy and rollout ---------------------------------------------------

    def greedy_shots(self, layouts: int, fired: int) -> list[int]:
        counts = self.hit_counts(layouts, fired)
        best = max(counts.values())
        return [cell for cell, k in counts.items() if k == best]

    def greedy_value(self, layouts: int, fired: int) -> float:
        """Expected shots left when always firing at a most likely cell,
        breaking ties uniformly at random."""
        if self._done(layouts, fired):
            return 0.0
        key = self._key(layouts, fired)
        if key not in self._greedy:
            cells = self.greedy_shots(layouts, fired)
            self._greedy[key] = sum(self._q(layouts, fired, cell, self.greedy_value)
                                    for cell in cells) / len(cells)
        return self._greedy[key]

    def rollout_shots(self, layouts: int, fired: int, tolerance: float = 1e-9) -> list[int]:
        """The cells minimizing 1 + E[greedy's shots afterwards]."""
        q = {cell: self._q(layouts, fired, cell, self.greedy_value)
             for cell in self.hit_counts(layouts, fired)}
        best = min(q.values())
        return [cell for cell, v in q.items() if v <= best + tolerance]

    def policy_value(self, choose, layouts: int | None = None, fired: int = 0) -> float:
        """Exact expected shots of a policy ``choose(layouts, fired) -> cells``
        that picks uniformly among the cells it returns."""
        memo = {}

        def value(layouts, fired):
            if self._done(layouts, fired):
                return 0.0
            key = self._key(layouts, fired)
            if key not in memo:
                cells = choose(layouts, fired)
                memo[key] = sum(self._q(layouts, fired, cell, value)
                                for cell in cells) / len(cells)
            return memo[key]

        return value(self.everything if layouts is None else layouts, fired)


@cache
def planner(config: GameConfig) -> Planner:
    """The shared planner for ``config``; its tables fill up as it is used."""
    return Planner(config)
