from collections import Counter
from math import prod

from ..belief import Belief
from ..particles import ParticleBelief
from .base import Agent
from .belief import BeliefAgent, _exact_is_practical

MODELS = ("placements", "layouts")


def _board(layout):
    """A layout as its set of ships, so the order of equal-length ships does not matter."""
    return frozenset(ship.cells for ship in layout)


class OpponentBeliefAgent(BeliefAgent):
    """Belief-based POMDP agent that learns where this opponent puts its ships.

    ``BeliefAgent`` assumes every layout is equally likely (the prior b0).
    This agent learns b0 from the opponent's past boards instead, and the
    Bayes update keeps the consistent layouts with the learned weights (see
    ``battleship/belief.py``). Two models of the habit:

    - ``model="placements"`` (default) counts where each ship was placed,
      by length, and treats ships as independent:
          w(s) = product over its ships of (times that placement was seen + smoothing)
      It generalizes from few games to boards never seen (e.g. "likes the
      edges"), but cannot express habits between ships (e.g. "ships touch").
    - ``model="layouts"`` counts whole boards:
          w(s) = times this exact layout was seen + smoothing
      It captures any habit given enough games, e.g. reusing a few boards,
      but learns nothing about boards it has not seen.

    Learning follows the ``Agent`` convention: ``end_game`` updates the counts
    only while ``training`` is on, so ``evaluate`` measures a frozen prior.
    With nothing learned every weight is equal, and with an integer
    ``smoothing`` the agent plays exactly the shots ``BeliefAgent`` plays.
    Larger ``smoothing`` trusts the counts less: slower to learn, but less
    misled by chance patterns in a few games.

    ``habit=`` fixes the prior to a known ``Habit`` instead (an oracle, the
    best this agent could learn). Shots are greedy.

    On boards too large for the exact belief (``belief="auto"`` picks), the
    agent uses ``ParticleBelief`` with the placement weights, so only
    ``model="placements"`` (and habits that split per ship) work there.
    """

    def __init__(self, model: str = "placements", smoothing: float = 3, habit=None,
                 belief: str = "auto", num_particles: int = 1000):
        super().__init__(belief=belief, num_particles=num_particles, planner="greedy")
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.model = model
        self.smoothing = smoothing
        self.habit = habit
        self.counts = Counter()      # placement (length, cells) or layout -> times seen
        self.games_learned = 0
        self.name = (f"opponent-oracle-{habit.name}" if habit
                     else f"opponent-{model}-s{smoothing}")

    def new_game(self, config, rng):
        Agent.new_game(self, config, rng)
        if self.mode == "exact" or (self.mode == "auto" and _exact_is_practical(config)):
            self.belief = Belief.prior(config, self.prior_weights(config))
        elif self.model == "layouts" and self.habit is None:
            raise ValueError('model "layouts" needs the exact belief (small boards)')
        else:
            self.belief = ParticleBelief(config, rng, self.num_particles,
                                         placement_weight=self.placement_weight)
        self.shots_seen = 0
        self.pending = []

    def placement_weight(self, length: int, cells: frozenset) -> float:
        """The prior factor of one ship placement (the placement model)."""
        if self.habit is not None:
            return self.habit.placement_weight(length, cells, self.config)
        return self.counts[length, cells] + self.smoothing

    def prior_weights(self, config) -> tuple[float, ...]:
        """The prior weight of every layout in the exact belief's table."""
        if self.habit is not None:
            return self.habit.distribution(config)
        layouts = Belief.prior(config).layouts
        if self.model == "layouts":
            return tuple(self.counts[_board(layout)] + self.smoothing for layout in layouts)
        return tuple(prod(self.counts[ship.length, ship.cells] + self.smoothing
                          for ship in layout)
                     for layout in layouts)

    def end_game(self, record):
        if self.training:
            self.learn(record.layout)

    def learn(self, layout) -> None:
        """Count one board the opponent used. Learning reads only the revealed
        layout, so this is what ``end_game`` does after a game in training."""
        if self.habit is not None:
            return
        if self.model == "layouts":
            self.counts[_board(layout)] += 1
        else:
            for ship in layout:
                self.counts[ship.length, ship.cells] += 1
        self.games_learned += 1

    def forget(self) -> None:
        """Drop everything learned about the opponent."""
        self.counts.clear()
        self.games_learned = 0
