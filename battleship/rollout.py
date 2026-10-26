"""Monte Carlo rollout for boards too large for exact planning.

Rollout is one step of lookahead in the belief-state MDP: fire at the cell a
that minimizes

    Q(b, a) = 1 + E[ shots a base policy needs to finish | b, a ].

By the policy improvement theorem this is never worse than the base policy,
when Q is exact. On 10x10 Q is estimated by simulation: worlds (complete
layouts) are drawn from the particle belief, and in each one the game is
played to the end, firing at a first and then following the base policy.
Every candidate cell is scored on the same worlds with the same random
numbers, so differences between candidates are measured with little noise.

The base policy is the probability density baseline. It needs no belief to
run, so a simulated game is cheap. ``_Density`` keeps its scores up to date
incrementally: a shot only changes the scores of the placements that touch
the cell (or, after a sinking, the placements of that length).

On 4x4, exact rollout over greedy closes about 90% of greedy's gap to the
optimum, and about 95% of that gap comes from shots fired while a ship is hit
but not sunk. So by default rollout is used only in that situation, and the
agent fires greedily otherwise.
"""

import random

from .config import GameConfig
from .particles import Constraints, _placements


class _Density:
    """The probability density agent's cell scores, updated shot by shot.

    Mirrors ``ProbabilityDensityAgent.density``: every remaining ship is tried
    in every position that covers no miss and no revealed sunk cell, and each
    such position adds ``hit_weight ** (open hits it covers)`` to its untried
    cells. Open hits are hits not known to belong to a sunk ship.
    """

    __slots__ = ("p", "reveal", "hit_weight", "count", "fired", "blocked", "open",
                 "contribution", "score")

    def __init__(self, config: GameConfig, misses: int, hits: int, sunk_cells: int,
                 remaining: dict[int, int], hit_weight: int = 50):
        self.p = _placements(config)
        self.reveal = config.reveal_sunk_cells
        self.hit_weight = hit_weight
        self.count = dict(remaining)            # length -> ships of it not yet sunk
        self.fired = misses | hits
        self.blocked = misses | sunk_cells
        self.open = hits & ~sunk_cells
        self.contribution = [0] * len(self.p.masks)
        self.score = [0] * config.num_cells
        for q in range(len(self.p.masks)):
            self._set(q, self._value(q))

    def copy(self) -> "_Density":
        new = object.__new__(_Density)
        new.p, new.reveal, new.hit_weight = self.p, self.reveal, self.hit_weight
        new.count = dict(self.count)
        new.fired, new.blocked, new.open = self.fired, self.blocked, self.open
        new.contribution = list(self.contribution)
        new.score = list(self.score)
        return new

    def _value(self, q: int) -> int:
        mask = self.p.masks[q]
        if mask & self.blocked:
            return 0
        copies = self.count.get(self.p.lengths[q], 0)
        return copies * self.hit_weight ** (mask & self.open).bit_count() if copies else 0

    def _set(self, q: int, value: int) -> None:
        delta = value - self.contribution[q]
        if delta:
            self.contribution[q] = value
            for cell in self.p.cells[q]:
                self.score[cell] += delta

    def _refresh(self, cells: int) -> None:
        touched = set()
        while cells:
            low = cells & -cells
            for ids in self.p.containing[low.bit_length() - 1].values():
                touched.update(ids)
            cells ^= low
        for q in touched:
            self._set(q, self._value(q))

    def shoot(self, cell: int, hit: bool, sunk_ship: int | None, sunk_length: int | None) -> None:
        """Record a shot. ``sunk_ship`` is the bitmask of the ship it sank."""
        bit = 1 << cell
        self.fired |= bit
        changed = bit
        if not hit:
            self.blocked |= bit
        else:
            self.open |= bit
        if sunk_ship is not None:
            self.count[sunk_length] -= 1
            if self.reveal:
                self.blocked |= sunk_ship
                self.open &= ~sunk_ship
                changed |= sunk_ship
            for q in self.p.by_length[sunk_length]:     # one fewer ship of this length
                self._set(q, self._value(q))
        self._refresh(changed)

    def best(self, rng: random.Random, num_cells: int) -> int:
        top, cells = -1, []
        for cell in range(num_cells):
            if self.fired >> cell & 1:
                continue
            s = self.score[cell]
            if s > top:
                top, cells = s, [cell]
            elif s == top:
                cells.append(cell)
        return rng.choice(cells)


def density_from_constraints(constraints: Constraints, hit_weight: int = 50) -> _Density:
    """The density scores for the position the shots so far have reached."""
    config = constraints.config
    remaining = {}
    for length in config.fleet:
        remaining[length] = remaining.get(length, 0) + 1
    for length in constraints.sunk_length.values():
        remaining[length] -= 1
    sunk = constraints.revealed_cells if config.reveal_sunk_cells else 0
    return _Density(config, constraints.misses, constraints.hits, sunk, remaining, hit_weight)


def simulate(start: _Density, world: list[int], constraints: Constraints, first: int,
             rng: random.Random) -> int:
    """Shots needed to finish in ``world`` (placement ids), firing at ``first``
    and then following the density policy."""
    p = constraints.placements
    config = constraints.config
    density = start.copy()
    fired = constraints.hits | constraints.misses
    ships = [p.masks[q] for q in world]
    left = sum(1 for mask in ships if mask & ~fired)
    shots, cell = 0, first
    while True:
        shots += 1
        bit = 1 << cell
        fired |= bit
        ship = next((mask for mask in ships if mask & bit), None)
        sunk = ship is not None and not ship & ~fired
        if sunk:
            left -= 1
            if not left:
                return shots
        density.shoot(cell, ship is not None, ship if sunk else None,
                      ship.bit_count() if sunk else None)
        cell = density.best(rng, config.num_cells)


def _outcome(ships: tuple[int, ...], cell: int, fired: int, reveal: bool):
    """What a layout (ship bitmasks) reports for a shot at ``cell`` after ``fired``."""
    bit = 1 << cell
    for mask in ships:
        if mask & bit:
            if mask & ~(fired | bit):
                return 1                                  # hit
            return (2, mask) if reveal else (2, mask.bit_count())   # sunk
    return 0                                              # miss


def simulate_particles(particles: list[tuple[int, ...]], world: int, fired: int, first: int,
                       reveal: bool, num_cells: int, rng: random.Random) -> int:
    """Shots needed to finish in ``particles[world]``, firing at ``first`` and
    then greedily on the particles that agree with every simulated outcome."""
    ships = particles[world]
    alive = range(len(particles))
    unions = [sum(p) for p in particles]
    shots, cell = 0, first
    left = sum(1 for mask in ships if mask & ~fired)
    while True:
        shots += 1
        seen = _outcome(ships, cell, fired, reveal)
        if seen != 0 and seen != 1:
            left -= 1
            if not left:
                return shots
        alive = [i for i in alive if _outcome(particles[i], cell, fired, reveal) == seen]
        fired |= 1 << cell
        counts = [0] * num_cells
        for i in alive:
            cover = unions[i] & ~fired
            while cover:
                low = cover & -cover
                counts[low.bit_length() - 1] += 1
                cover ^= low
        top = max(counts)
        cell = rng.choice([c for c in range(num_cells) if counts[c] == top and not fired >> c & 1])


def rollout_choice(constraints: Constraints, particles: list[list[int]], worlds: list[int],
                   candidates: list[int], rng: random.Random,
                   base: str = "particles") -> tuple[int, dict[int, float]]:
    """The candidate with the fewest simulated shots, and every candidate's
    average. ``worlds`` are indices into ``particles`` (layouts as placement
    ids). All candidates are played on the same worlds and random seeds.

    ``base="particles"`` continues greedily on the particles that agree with
    the simulated outcomes; ``base="density"`` continues with the probability
    density policy.
    """
    seeds = [rng.getrandbits(64) for _ in worlds]
    means = {}
    if base == "density":
        start = density_from_constraints(constraints)
        for cell in candidates:
            total = sum(simulate(start, particles[world], constraints, cell, random.Random(seed))
                        for world, seed in zip(worlds, seeds))
            means[cell] = total / len(worlds)
    else:
        masks = constraints.placements.masks
        layouts = [tuple(masks[p] for p in state) for state in particles]
        fired = constraints.hits | constraints.misses
        reveal = constraints.config.reveal_sunk_cells
        for cell in candidates:
            total = sum(simulate_particles(layouts, world, fired, cell, reveal,
                                           constraints.config.num_cells, random.Random(seed))
                        for world, seed in zip(worlds, seeds))
            means[cell] = total / len(worlds)
    return min(candidates, key=means.__getitem__), means
