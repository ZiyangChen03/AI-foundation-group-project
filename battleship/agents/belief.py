from functools import cache
from math import prod

from ..belief import MAX_LAYOUTS, Belief, observed_outcome
from ..layout import placements
from ..particles import ParticleBelief
from ..planning import planner as exact_planner
from ..rollout import rollout_choice
from .base import Agent

PLANNERS = ("greedy", "rollout", "optimal")


@cache
def _exact_is_practical(config):
    # Ships placed independently bound the number of layouts from above.
    if prod(len(placements(config, length)) for length in config.fleet) > 50 * MAX_LAYOUTS:
        return False
    try:
        Belief.prior(config)
    except ValueError:
        return False
    return True


class BeliefAgent(Agent):
    """Belief-based POMDP agent.

    The belief state is the set of layouts consistent with every shot so far,
    all equally likely. After each shot the agent applies the Bayes update,
    keeping only the layouts that would have produced the observed feedback.
    Ties between equally good cells go to a random one.

    ``belief="exact"`` lists every layout (small boards such as MINI only).
    ``belief="particles"`` keeps ``num_particles`` sampled layouts instead
    (see ``battleship/particles.py``), for boards such as CLASSIC.
    ``belief="auto"`` (default) uses the exact belief whenever it fits.

    How the next shot is chosen:

    - ``"greedy"``: the untried cell most likely to hold a ship,
      P(ship at cell | history).
    - ``"rollout"``: one step of lookahead, the cell minimizing
      1 + E[shots a base policy needs afterwards]. With the exact belief this
      is computed exactly, with greedy as the base policy
      (``battleship/planning.py``). With particles it is estimated by playing
      ``rollout_worlds`` sampled layouts to the end (``battleship/rollout.py``)
      for the ``rollout_candidates`` most likely cells, and only while some
      ship is hit but not sunk unless ``rollout_always``; otherwise it fires
      greedily. ``rollout_base`` is the policy simulated after the first shot:
      ``"particles"`` (greedy on the particles that agree with the simulated
      outcomes) or ``"density"`` (the probability density baseline). The
      simulations use a random ``rollout_particles`` of the particles.
    - ``"optimal"``: the cell minimizing expected shots to finish, by solving
      the belief-state MDP exactly. Exact belief only.

    The exact belief can be rebuilt from an Observation; particles cannot, so
    with particles the agent must observe every shot, as the runner ensures
    (so must the exact belief when sunk cells are revealed and two ships
    share a length).
    """

    def __init__(self, belief: str = "auto", num_particles: int = 1000,
                 planner: str = "greedy", rollout_worlds: int = 24,
                 rollout_candidates: int = 4, rollout_always: bool = False,
                 rollout_base: str = "particles", rollout_particles: int = 300):
        super().__init__()
        if belief not in ("auto", "exact", "particles"):
            raise ValueError('belief must be "auto", "exact" or "particles"')
        if planner not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}")
        if planner == "optimal" and belief == "particles":
            raise ValueError('planner "optimal" needs the exact belief')
        if rollout_base not in ("particles", "density"):
            raise ValueError('rollout_base must be "particles" or "density"')
        self.mode = belief
        self.num_particles = num_particles
        self.planner = planner
        self.rollout_worlds = rollout_worlds
        self.rollout_candidates = rollout_candidates
        self.rollout_always = rollout_always
        self.rollout_base = rollout_base
        self.rollout_particles = rollout_particles
        self.name = f"belief-{planner}" + (f"-{num_particles}p" if belief == "particles" else "")

    def new_game(self, config, rng):
        super().new_game(config, rng)
        exact = self.mode == "exact" or (self.mode == "auto" and _exact_is_practical(config))
        if exact:
            self.belief = Belief.prior(config)
        elif self.planner == "optimal":
            raise ValueError('planner "optimal" needs the exact belief, which is only '
                             "practical on small boards")
        else:
            self.belief = ParticleBelief(config, rng, self.num_particles)
        self.shots_seen = 0
        self.pending = []           # observed shots not yet applied to the belief

    def select_shot(self, obs):
        # The Bayes update is applied here rather than in observe(), so the
        # runner's move timing includes it and no update runs after the game ends.
        for action, outcome in self.pending:
            if isinstance(self.belief, Belief):
                self.belief = self.belief.update(action, outcome)
            else:
                self.belief.update(action, outcome)
        self.pending = []
        if self.shots_seen != obs.num_shots:
            # Rebuilding from one Observation is exact only when it cannot
            # confuse which of two equal-length ships a revealed sinking showed.
            fleet = self.config.fleet
            if not isinstance(self.belief, Belief) or (
                    self.config.reveal_sunk_cells and len(set(fleet)) < len(fleet)):
                raise RuntimeError("this belief must observe every shot")
            # Shots this agent did not observe; rebuild from the history.
            self.belief = Belief.from_observation(obs)
            self.shots_seen = obs.num_shots
        if isinstance(self.belief, Belief) and self.planner != "greedy":
            plan = exact_planner(self.config)
            layouts, fired = plan.state(self.belief)
            if self.planner == "rollout":
                return self.rng.choice(plan.rollout_shots(layouts, fired))
            return self.rng.choice(plan.optimal_shots(layouts, fired))
        probabilities = self.belief.occupancy()
        best = max(probabilities[cell] for cell in obs.legal_actions)
        greedy = [cell for cell in obs.legal_actions if probabilities[cell] == best]
        if self.planner == "rollout" and best < 1 and (self.rollout_always or self._targeting()):
            return self._rollout(obs, probabilities)
        return self.rng.choice(greedy)

    def _targeting(self):
        """Whether some ship is hit but not yet sunk."""
        constraints = self.belief.constraints
        return constraints.hits.bit_count() > sum(constraints.sunk_length.values())

    def _rollout(self, obs, probabilities):
        ranked = sorted(obs.legal_actions, key=lambda cell: -probabilities[cell])
        candidates = [cell for cell in ranked[:self.rollout_candidates] if probabilities[cell] > 0]
        if len(candidates) == 1:
            return candidates[0]
        particles = self.belief.particles
        if len(particles) > self.rollout_particles:
            particles = self.rng.sample(particles, self.rollout_particles)
        worlds = [self.rng.randrange(len(particles)) for _ in range(self.rollout_worlds)]
        choice, _ = rollout_choice(self.belief.constraints, particles, worlds, candidates,
                                   self.rng, base=self.rollout_base)
        return choice

    def observe(self, obs, action, result):
        self.pending.append((action, observed_outcome(obs, result.observation)))
        self.shots_seen += 1
