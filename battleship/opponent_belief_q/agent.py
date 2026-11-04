"""Q-learning specialized to a frozen, learned placement prior (MINI)."""
import gzip
import json
import math

from ..agents.belief_q import BeliefQAgent, _actions
from ..belief import Belief
from ..config import MINI
from ..planning import Planner, planner


class OpponentBeliefQAgent(BeliefQAgent):
    name = "opponent-belief-q"

    def __init__(self, weights, alpha=None, epsilon=0.1, initial=-20.0):
        super().__init__(alpha=alpha, epsilon=epsilon, initial=initial, config=MINI)
        weights = tuple(float(w) for w in weights)
        if len(weights) != len(Belief.prior(MINI)) or any(
                not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("need one finite positive weight per MINI layout")
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= epsilon <= 1 or not math.isfinite(initial):
            raise ValueError("invalid epsilon or initial value")
        total = sum(weights)
        if not math.isfinite(total):
            raise ValueError("weight sum must be finite")
        self._prior_weights = tuple(w / total for w in weights)
        self._weighted_plan = Planner(MINI, self._prior_weights)

    @property
    def prior_weights(self):
        return self._prior_weights

    def new_game(self, config, rng):
        super().new_game(config, rng)
        self.plan = self._weighted_plan
        self.belief = Belief.prior(config, self.prior_weights)

    def fast_train(self, episodes, rng, epsilon_start=1.0, epsilon_end=0.0,
                   callback=None, every=0):
        if episodes < 0 or not 0 <= epsilon_start <= 1 or not 0 <= epsilon_end <= 1:
            raise ValueError("invalid training schedule")
        plan = planner(self.board)
        previous, self.training = self.training, True
        try:
            for episode in range(1, episodes + 1):
                self.epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode / episodes
                hidden = rng.choices(range(len(self.prior_weights)), self.prior_weights)[0]
                layouts, fired = plan.everything, 0
                while True:
                    key = plan._key(layouts, fired)
                    cell = self._choose(key, _actions(plan, layouts, fired), rng)
                    layouts = next(g for g in plan.outcomes(layouts, fired, cell) if g >> hidden & 1)
                    fired |= 1 << cell
                    if plan._done(layouts, fired):
                        self.update(key, cell, -1.0)
                        break
                    self.update(key, cell, -1.0 + self.best_value(
                        plan._key(layouts, fired), _actions(plan, layouts, fired)))
                if callback and every and episode % every == 0:
                    callback(episode)
        finally:
            self.training = previous

    def exact_value(self, weights=None):
        """Evaluate on supplied truth distribution, or the learned prior."""
        plan = Planner(self.board, self.prior_weights if weights is None else weights)
        return plan.policy_value(lambda layouts, fired: self.best_actions(
            plan._key(layouts, fired), _actions(plan, layouts, fired)))

    def save(self, path):
        super().save(path)
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        data.update(kind="opponent-belief-q", prior_weights=self.prior_weights)
        with gzip.open(path, "wt") as stream:
            json.dump(data, stream)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        if data.get("kind") != "opponent-belief-q" or data.get("version") != 1:
            raise ValueError("not a supported opponent-belief-q checkpoint")
        base = BeliefQAgent.load(path)
        if base.board != MINI:
            raise ValueError("only MINI is supported")
        agent = cls(data["prior_weights"], base.alpha, base.epsilon, base.initial)
        agent.q, agent.visits = base.q, base.visits
        return agent
