"""10x10 versions of the POMDP planners that need the exact belief on 4x4.

On 4x4 the belief is the exact list of 264 layouts, so ``BeliefAgent`` can
compute information gain exactly and solve the belief-state MDP (Q*). On 10x10
there are about 30 billion layouts, so the belief is a sample of them
(``ParticleBelief``). This module adds the planners that need the full list,
working on that sample instead. ``BeliefAgent`` itself is unchanged.

``LargeBoardBeliefAgent(planner=...)``:

- ``"infogain"``: the information gain of a shot is the entropy of its result
  (with equally likely layouts). The particles estimate each result's chance
  the way they estimate P(ship): sort the sampled layouts by what the shot
  would report (miss, hit, or sinking a particular ship) and use the shares.
  It fires at the largest, among squares some particle covers, like the exact
  ``infogain`` planner.
- ``"greedy-info"``: the most likely squares, ties broken by that estimate.
- ``"optimal-approx"``: an approximation of the optimal planner, NOT the
  optimum. Q*(b, a) cannot be computed on 10x10, so every shot uses the
  sampled lookahead of ``planner="rollout"``: each candidate square is fired
  in sampled worlds and the game is finished greedily on the particles; the
  square with the fewest simulated shots wins. That estimates
  1 + E[shots greedy needs afterwards], one step of policy improvement on
  greedy, which is what rollout computes exactly on 4x4.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log2

from ..particles import ParticleBelief
from .belief import BeliefAgent

LARGE_PLANNERS = ("infogain", "greedy-info", "optimal-approx")


def particle_result_counts(belief: ParticleBelief):
    """For every unfired cell: how many particles report a hit, and how many
    report each kind of sinking (keyed by the ship's cells when sunk cells are
    revealed, by its length otherwise). Every other particle reports a miss."""
    constraints = belief.constraints
    masks = constraints.placements.masks
    fired = constraints.hits | constraints.misses
    reveal = belief.config.reveal_sunk_cells
    hits = Counter()
    sinks = defaultdict(Counter)
    for state in belief.particles:
        for p in state:
            mask = masks[p]
            left = mask & ~fired
            if not left:
                continue                              # already sunk
            if left & (left - 1) == 0:                # one cell left: firing it sinks the ship
                sinks[left.bit_length() - 1][mask if reveal else mask.bit_count()] += 1
            else:
                while left:
                    low = left & -left
                    hits[low.bit_length() - 1] += 1
                    left ^= low
    return hits, sinks


def particle_info_gain(belief: ParticleBelief, legal) -> dict[int, float]:
    """Estimated information gain (bits) of each legal cell that some particle
    covers: the entropy of the shot's result over the particles."""
    hits, sinks = particle_result_counts(belief)
    n = len(belief.particles)
    gains = {}
    for cell in legal:
        groups = [hits[cell], *sinks[cell].values()]
        covered = sum(groups)
        if not covered:
            continue                                  # a certain miss: left out, as on 4x4
        groups.append(n - covered)
        gains[cell] = -sum(k / n * log2(k / n) for k in groups if k)
    return gains


class LargeBoardBeliefAgent(BeliefAgent):
    """The 4x4-only planners, adapted to the particle belief (see the module
    docstring). Works on any board; on 4x4 use ``BeliefAgent`` for the exact
    versions."""

    def __init__(self, planner: str = "infogain", num_particles: int = 1000,
                 rollout_worlds: int = 24, rollout_candidates: int = 4,
                 rollout_particles: int = 300):
        if planner not in LARGE_PLANNERS:
            raise ValueError(f"planner must be one of {LARGE_PLANNERS}")
        base = "rollout" if planner == "optimal-approx" else "greedy"
        super().__init__(belief="particles", num_particles=num_particles, planner=base,
                         rollout_worlds=rollout_worlds, rollout_candidates=rollout_candidates,
                         rollout_always=True, rollout_particles=rollout_particles)
        self.large_planner = planner
        self.name = f"belief-{planner}-{num_particles}p"

    def select_shot(self, obs):
        # The parent applies the Bayes update and returns its own pick: greedy,
        # or for "optimal-approx" the rollout pick on every shot.
        pick = super().select_shot(obs)
        if self.large_planner == "optimal-approx":
            return pick
        gains = particle_info_gain(self.belief, obs.legal_actions)
        if not gains:
            return pick
        if self.large_planner == "greedy-info":
            probabilities = self.belief.occupancy()
            best = max(probabilities[cell] for cell in obs.legal_actions)
            tied = [cell for cell in obs.legal_actions if probabilities[cell] == best]
            scores = {cell: gains.get(cell, 0.0) for cell in tied}
        else:
            scores = gains
        top = max(scores.values())
        return self.rng.choice([cell for cell, g in scores.items() if g >= top - 1e-12])
