"""Sampled belief states for boards too large to list every layout.

The exact belief is uniform over the layouts that explain every shot so far
(see ``belief.py``). On 10x10 there are far too many to list, so
``ParticleBelief`` keeps a few hundred of them ("particles"), drawn uniformly
from that set, and estimates P(ship at cell) from them.

Consistency splits into checks on single ships. Call a placement p of a ship
of length k *valid* when, judged on its own cells only:

  - it covers no miss;
  - if every cell of p has been shot, the last of those shots reported SUNK
    with length k (and revealed exactly p, when sunk cells are revealed), and
    none of the earlier ones reported SUNK;
  - otherwise none of its shot cells reported SUNK.

A layout explains the history exactly when every ship is valid, no two ships
overlap, and together they cover every cell that was hit. This holds because a
shot reports SUNK precisely when it hits the last unshot cell of its ship.

Particles stay uniform as follows:

  - Filtering: after a shot, particles that no longer explain the history are
    dropped. Survivors of a uniform sample are a uniform sample of the smaller
    set, so this step is exact.
  - Gibbs moves: one ship (or a pair of ships) is redrawn uniformly from every
    placement that keeps the layout consistent, given the other ships. Each
    move leaves the uniform distribution unchanged. Pair moves are needed
    because single-ship moves can get stuck, e.g. when the 2-ship sits exactly
    on two hits that the 3-ship could also cover.
  - Fresh draws: every remaining ship gets an independent uniform valid
    placement, kept only if the whole layout is consistent. Accepted draws are
    exact uniform samples and replace random particles.

P(ship at cell) is estimated with Rao-Blackwellization: while ship i is
redrawn, the exact conditional probability that it covers each cell (given
the others) is added, not just where it lands. This gives the same answer on
average with far less noise than counting particles.
"""

import random
from dataclasses import dataclass
from functools import cache

from .config import GameConfig, to_cell
from .env import Result
from .layout import Layout, random_layout


@dataclass(frozen=True)
class _Placements:
    """Every straight placement of every ship length in the fleet."""

    masks: tuple[int, ...]                       # bitmask of cells, by placement id
    cells: tuple[tuple[int, ...], ...]           # cells, by placement id
    lengths: tuple[int, ...]                     # length, by placement id
    by_length: dict[int, tuple[int, ...]]        # length -> placement ids
    containing: tuple[dict[int, tuple[int, ...]], ...]   # [cell][length] -> ids
    by_mask: dict[int, int]                      # bitmask -> placement id


@cache
def _placements(config: GameConfig) -> _Placements:
    masks, cells, lengths = [], [], []
    for length in sorted(set(config.fleet)):
        seen = set()
        for row in range(config.size):
            for col in range(config.size):
                for dr, dc in ((0, 1), (1, 0)):
                    if (row + dr * (length - 1) >= config.size
                            or col + dc * (length - 1) >= config.size):
                        continue
                    ship = tuple(to_cell(row + dr * i, col + dc * i, config)
                                 for i in range(length))
                    if ship in seen:            # length 1: both directions coincide
                        continue
                    seen.add(ship)
                    masks.append(sum(1 << cell for cell in ship))
                    cells.append(ship)
                    lengths.append(length)
    by_length = {length: tuple(p for p, k in enumerate(lengths) if k == length)
                 for length in set(lengths)}
    containing = tuple({length: tuple(p for p in by_length[length] if cell in cells[p])
                        for length in by_length}
                       for cell in range(config.num_cells))
    return _Placements(tuple(masks), tuple(cells), tuple(lengths), by_length, containing,
                       {mask: p for p, mask in enumerate(masks)})


class Constraints:
    """Everything the shots so far say about single ship placements.

    Updated one shot at a time with ``add``. ``valid[p]`` tells whether
    placement ``p`` passes the checks in the module docstring.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.placements = _placements(config)
        self.misses = 0           # cells shot that missed
        self.hits = 0             # cells shot that hit (HIT or SUNK)
        self.sunk_shots = 0       # cells whose own shot reported SUNK
        self.revealed_cells = 0   # cells of revealed sunk ships
        self.order = {}           # cell -> index of the shot that hit it
        self.sunk_length = {}     # sunk-shot cell -> length reported
        self.revealed = {}        # sunk-shot cell -> bitmask of the revealed ship
        self.valid = bytearray([1]) * len(self.placements.masks)
        self.options = dict(self.placements.by_length)   # length -> valid ids

    def add(self, cell: int, outcome) -> None:
        """Record the shot at ``cell`` and its outcome (see ``belief.Outcome``)."""
        result, sunk_length, revealed = outcome
        bit = 1 << cell
        recheck = set()
        for ids in self.placements.containing[cell].values():
            recheck.update(ids)
        if result is Result.MISS:
            self.misses |= bit
        else:
            self.hits |= bit
            self.order[cell] = len(self.order)
            if result is Result.SUNK:
                self.sunk_shots |= bit
                self.sunk_length[cell] = sunk_length
                if revealed is not None:
                    mask = sum(1 << c for c in revealed)
                    self.revealed[cell] = mask
                    self.revealed_cells |= mask
                    for c in revealed:
                        for ids in self.placements.containing[c].values():
                            recheck.update(ids)
        for p in recheck:
            self.valid[p] = self.check(p)
        self.options = {length: tuple(p for p in ids if self.valid[p])
                        for length, ids in self.placements.by_length.items()}

    def check(self, p: int) -> bool:
        """Whether placement ``p`` passes the single-ship checks."""
        mask = self.placements.masks[p]
        if mask & self.misses:
            return False
        if mask & self.revealed_cells and mask not in self.revealed.values():
            # Overlaps a revealed ship, which every consistent layout contains.
            return False
        if mask & ~self.hits:                     # not every cell shot yet
            return not mask & self.sunk_shots
        last = max(self.placements.cells[p], key=self.order.__getitem__)
        if mask & self.sunk_shots != 1 << last:
            return False
        if self.sunk_length[last] != self.placements.lengths[p]:
            return False
        return last not in self.revealed or self.revealed[last] == mask

    def explains(self, layout: tuple[int, ...]) -> bool:
        """Whether a layout, given as placement ids, explains every shot."""
        union = 0
        for p in layout:
            mask = self.placements.masks[p]
            if not self.valid[p] or mask & union:
                return False
            union |= mask
        return not self.hits & ~union


class ParticleBelief:
    """An approximately uniform sample of the layouts that explain the shots.

    ``occupancy()`` estimates P(ship at cell) for every cell. The estimate is
    refreshed by every ``update``. Randomness comes only from ``rng``.
    """

    def __init__(self, config: GameConfig, rng: random.Random, num_particles: int = 300,
                 fresh_draws: int = 50, max_search_nodes: int = 200_000):
        self.config = config
        self.rng = rng
        self.num_particles = num_particles
        self.fresh_draws = fresh_draws
        self.max_search_nodes = max_search_nodes
        self.constraints = Constraints(config)
        self.slots = tuple(config.fleet)      # the length of each particle entry
        self.searches = 0                     # times every particle died
        by_mask = self.constraints.placements.by_mask
        self.particles = [self._from_layout(random_layout(config, rng), by_mask)
                          for _ in range(num_particles)]
        self._estimate()

    def _from_layout(self, layout: Layout, by_mask) -> list[int]:
        # random_layout keeps fleet order, so each ship matches its slot.
        return [by_mask[sum(1 << cell for cell in ship.cells)] for ship in layout]

    # --- the belief update ---------------------------------------------------

    def update(self, cell: int, outcome) -> None:
        """Bayes update after shooting ``cell`` and observing ``outcome``."""
        self.constraints.add(cell, outcome)
        survivors = [state for state in self.particles if self.constraints.explains(state)]
        if not survivors:
            self.searches += 1
            survivors = [self._search() for _ in range(5)]
            for state in survivors:
                for _ in range(20):             # let the chain forget the search
                    self._move(state)
        self.particles = survivors + [list(self.rng.choice(survivors))
                                      for _ in range(self.num_particles - len(survivors))]
        self._fresh()
        self._estimate()

    def occupancy(self) -> list[float]:
        """Estimated P(a ship covers the cell); 0 for cells already shot."""
        return self._occupancy

    # --- moves that keep the uniform distribution ----------------------------

    def _choices(self, length: int, others: int, required: int) -> list[int]:
        """Valid placements of ``length`` that avoid ``others`` and cover ``required``."""
        masks = self.constraints.placements.masks
        if required:
            low = (required & -required).bit_length() - 1
            ids = [p for p in self.constraints.placements.containing[low][length]
                   if self.constraints.valid[p]]
        else:
            ids = self.constraints.options[length]
        return [p for p in ids if not masks[p] & others and masks[p] & required == required]

    def _sweep(self, state: list[int], weights: list[float] | None = None) -> None:
        """Redraw each ship in turn given the others (Gibbs). If ``weights`` is
        given, add each ship's conditional distribution to it."""
        masks = self.constraints.placements.masks
        hits = self.constraints.hits
        total = 0
        for p in state:
            total |= masks[p]
        n = len(state)
        for i, length in enumerate(self.slots):
            others = total ^ masks[state[i]]
            choices = self._choices(length, others, hits & ~others)
            if weights is not None:
                share = 1 / len(choices)
                for p in choices:
                    weights[p] += share
                for j in range(n):
                    if j != i:
                        weights[state[j]] += 1
            state[i] = self.rng.choice(choices)
            total = others | masks[state[i]]

    def _pair_move(self, state: list[int], i: int, j: int) -> None:
        """Redraw ships ``i`` and ``j`` together, uniformly over every pair of
        placements that keeps the layout consistent."""
        masks = self.constraints.placements.masks
        valid = self.constraints.valid
        containing = self.constraints.placements.containing
        others = 0
        for k, p in enumerate(state):
            if k != i and k != j:
                others |= masks[p]
        required = self.constraints.hits & ~others
        if not required:
            return            # single-ship moves already connect these states
        low = (required & -required).bit_length() - 1
        bit = 1 << low
        li, lj = self.slots[i], self.slots[j]
        pairs = []
        for p in containing[low][li]:
            if valid[p] and not masks[p] & others:
                pairs.extend((p, q) for q in self._choices(lj, others | masks[p],
                                                           required & ~masks[p]))
        for q in containing[low][lj]:
            if valid[q] and not masks[q] & others:
                pairs.extend((p, q) for p in self._choices(li, others | masks[q],
                                                           required & ~masks[q])
                             if not masks[p] & bit)
        state[i], state[j] = self.rng.choice(pairs)

    def _move(self, state: list[int]) -> None:
        self._sweep(state)
        if len(state) > 1:
            i, j = self.rng.sample(range(len(state)), 2)
            self._pair_move(state, i, j)

    def _fresh(self) -> None:
        """Replace random particles with independent exact draws, when the
        draws are cheap enough to succeed."""
        masks = self.constraints.placements.masks
        options = self.constraints.options
        hits = self.constraints.hits
        for _ in range(self.fresh_draws):
            draw, union = [], 0
            for length in self.slots:
                p = self.rng.choice(options[length])
                if masks[p] & union:
                    break
                draw.append(p)
                union |= masks[p]
            else:
                if not hits & ~union:
                    self.particles[self.rng.randrange(len(self.particles))] = draw

    def _estimate(self) -> None:
        """Move every particle once and estimate P(ship at cell)."""
        placements = self.constraints.placements
        weights = [0.0] * len(placements.masks)
        for state in self.particles:
            if len(state) > 1:
                i, j = self.rng.sample(range(len(state)), 2)
                self._pair_move(state, i, j)
            self._sweep(state, weights)
        scale = 1 / (len(self.particles) * len(self.slots))
        cover = [0.0] * self.config.num_cells
        for p, w in enumerate(weights):
            if w:
                for cell in placements.cells[p]:
                    cover[cell] += w
        shot = self.constraints.hits | self.constraints.misses
        self._occupancy = [0.0 if shot >> cell & 1 else cover[cell] * scale
                           for cell in range(self.config.num_cells)]

    # --- when every particle dies --------------------------------------------

    def _search(self) -> list[int]:
        """Find one consistent layout by backtracking (not uniform; the caller
        runs moves on it afterwards)."""
        masks = self.constraints.placements.masks
        valid = self.constraints.valid
        containing = self.constraints.placements.containing
        options = self.constraints.options
        state = [None] * len(self.slots)
        nodes = 0

        def place(free: list[int], occupied: int, required: int) -> bool:
            nonlocal nodes
            nodes += 1
            if nodes > self.max_search_nodes:
                raise RuntimeError("no consistent layout found; the history may be impossible")
            if not free:
                return not required
            if required:
                low = (required & -required).bit_length() - 1
                tried = set()
                for i in self.rng.sample(free, len(free)):
                    if self.slots[i] in tried:   # equal-length ships are interchangeable
                        continue
                    tried.add(self.slots[i])
                    ids = [p for p in containing[low][self.slots[i]]
                           if valid[p] and not masks[p] & occupied]
                    for p in self.rng.sample(ids, len(ids)):
                        state[i] = p
                        rest = [k for k in free if k != i]
                        if place(rest, occupied | masks[p], required & ~masks[p]):
                            return True
                return False
            i = free[0]
            ids = [p for p in options[self.slots[i]] if not masks[p] & occupied]
            for p in self.rng.sample(ids, len(ids)):
                state[i] = p
                if place(free[1:], occupied | masks[p], 0):
                    return True
            return False

        if not place(list(range(len(self.slots))), 0, self.constraints.hits):
            raise RuntimeError("no consistent layout exists for these shots")
        return state
