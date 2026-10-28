"""Opponents with placement habits, and exact play against them on small boards.

The runner places ships uniformly at random. A real opponent may not: some
favour the edges, line their ships up, or reuse a few boards. A ``Habit``
gives every layout s a weight w(s); the opponent places s with probability
w(s) / (sum of w over all layouts).

Most habits here are ``FeatureHabit``s, w(s) = exp(strength * feature(s)):

  - ``border_cells``: ship cells on the outer ring of the board
  - ``horizontal_ships``: ships lying along a row
  - ``touching_pairs``: pairs of ships that share an edge

``FavoriteLayouts`` instead reuses a few fixed boards.

To play against a habit, sample layouts with ``habit.sample`` and pass them to
``play_solo(layout=...)``; the runner needs no changes. On small boards,
``expected_shots`` gives the exact mean against the habit by weighting every
layout by its probability.
"""

import math
import random
from dataclasses import dataclass
from functools import cache
from itertools import combinations

from .config import GameConfig, neighbors, to_rc
from .layout import Layout, random_layout


def border_cells(layout: Layout, config: GameConfig) -> int:
    edge = config.size - 1
    return sum(1 for ship in layout for cell in ship.cells
               if {*to_rc(cell, config)} & {0, edge})


def horizontal_ships(layout: Layout, config: GameConfig) -> int:
    return sum(1 for ship in layout
               if ship.length > 1 and len({to_rc(c, config)[0] for c in ship.cells}) == 1)


def touching_pairs(layout: Layout, config: GameConfig) -> int:
    def touch(a, b):
        return any(n in b.cells for cell in a.cells for n in neighbors(cell, config))
    return sum(1 for a, b in combinations(layout, 2) if touch(a, b))


# feature -> its largest possible value, for rejection sampling
FEATURES = {
    border_cells: lambda config: sum(config.fleet),
    horizontal_ships: lambda config: len(config.fleet),
    touching_pairs: lambda config: len(config.fleet) * (len(config.fleet) - 1) // 2,
}


class Habit:
    """How an opponent places its fleet: a weight for every layout.
    Subclasses have a ``name``."""

    def weight(self, layout: Layout, config: GameConfig) -> float:
        raise NotImplementedError

    def max_weight(self, config: GameConfig) -> float:
        """An upper bound on ``weight``, for rejection sampling."""
        raise NotImplementedError

    def sample(self, config: GameConfig, rng: random.Random) -> Layout:
        """A layout drawn with probability proportional to its weight
        (rejection sampling from uniform layouts)."""
        bound = self.max_weight(config)
        while True:
            layout = random_layout(config, rng)
            if rng.random() * bound < self.weight(layout, config):
                return layout

    def distribution(self, config: GameConfig) -> tuple[float, ...]:
        """P(layout) for every layout in the exact belief's table (small boards)."""
        return _distribution(self, config)


@dataclass(frozen=True)
class FeatureHabit(Habit):
    """w(s) = exp(strength * feature(s)); strength 0 is uniform placement."""

    name: str
    feature: object
    strength: float

    def weight(self, layout, config):
        return math.exp(self.strength * self.feature(layout, config))

    def max_weight(self, config):
        return math.exp(max(0.0, self.strength * FEATURES[self.feature](config)))


@dataclass(frozen=True)
class FavoriteLayouts(Habit):
    """Always one of a few fixed boards, each equally likely."""

    name: str
    layouts: tuple[Layout, ...]

    def weight(self, layout, config):
        cells = frozenset(ship.cells for ship in layout)
        return float(any(cells == frozenset(s.cells for s in fav) for fav in self.layouts))

    def max_weight(self, config):
        return 1.0

    def sample(self, config, rng):
        return rng.choice(self.layouts)


def uniform() -> Habit:
    return FeatureHabit("uniform", border_cells, 0.0)


def favorites(config: GameConfig, k: int, seed: int = 0) -> Habit:
    rng = random.Random(seed)
    return FavoriteLayouts(f"{k} favorites", tuple(random_layout(config, rng) for _ in range(k)))


def standard_habits(config: GameConfig) -> list[Habit]:
    """The opponents used in the benchmark (docs/design_opponent_belief.md)."""
    return [
        uniform(),
        FeatureHabit("edges", border_cells, 1.0),
        FeatureHabit("center", border_cells, -1.0),
        FeatureHabit("horizontal", horizontal_ships, 1.5),
        FeatureHabit("touching", touching_pairs, 2.0),
        favorites(config, 3),
    ]


@cache
def _distribution(habit: Habit, config: GameConfig) -> tuple[float, ...]:
    from .belief import Belief
    weights = [habit.weight(layout, config) for layout in Belief.prior(config).layouts]
    total = sum(weights)
    return tuple(w / total for w in weights)


def expected_shots(agent, config: GameConfig, habit: Habit, agent_seeds=range(3)) -> float:
    """Exact mean shots-to-win of ``agent`` against ``habit``: every layout the
    habit can place, weighted by its probability (ties averaged over
    ``agent_seeds``). The agent plays with training off, so it does not learn."""
    from .belief import Belief
    from .runner import play_solo
    previous, agent.training = agent.training, False
    try:
        total = 0.0
        for layout, p in zip(Belief.prior(config).layouts, habit.distribution(config)):
            if p:
                shots = [play_solo(agent, config, layout=layout, agent_seed=seed).num_shots
                         for seed in agent_seeds]
                total += p * sum(shots) / len(shots)
        return total
    finally:
        agent.training = previous
