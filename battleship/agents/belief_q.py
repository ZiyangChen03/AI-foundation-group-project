"""Tabular Q-learning on POMDP belief states (exact belief, small boards).

The state is the exact belief: the set of layouts consistent with every shot so
far, plus the fired cells that can still matter (``Planner._key``). Boards that
look different but say the same thing about the hidden ships share one state,
so the Q-table stays small enough to cover on 4x4. The agent never sees the
hidden layout; the updates are ordinary one-step Q-learning on the belief MDP:

    Q(b, a) <- Q(b, a) + alpha * (target - Q(b, a)),   alpha = 1 / visits(b, a) by default
    target  = -1                                  if the shot ends the game
    target  = -1 + max_{a'} Q(b', a')             otherwise (gamma = 1)

Actions are the unfired cells that at least one consistent layout covers;
cells no layout covers are certain misses and never help.

Unvisited actions start at ``initial`` (default -20, pessimistic). With every
reward -1, a start value of 0 would make every untried action look better than
any learned one, so greedy play would keep picking untried, effectively random,
shots.

``fast_train`` simulates games directly on beliefs, which gives the same
transitions as playing through the environment but is much faster.
``exact_value`` computes the exact expected shots of the greedy policy over all
layouts with ``Planner.policy_value``.
"""

from __future__ import annotations

import gzip
import json
import random

from ..belief import Belief, observed_outcome
from ..config import MINI, GameConfig
from ..planning import planner as exact_planner
from .base import Agent
from .belief import _exact_is_practical

_TIE = 1e-12


def _actions(plan, layouts: int, fired: int) -> list[int]:
    """Unfired cells that at least one consistent layout covers. A cell no
    layout covers is a certain miss, which never helps, so it is left out.
    This also makes the action set a function of the state key alone."""
    return list(plan.hit_counts(layouts, fired))


class BeliefQAgent(Agent):
    """Tabular Q-learning whose state is the exact POMDP belief."""

    name = "belief-q"

    def __init__(self, alpha: float | None = None, epsilon: float = 0.1, initial: float = -20.0,
                 config: GameConfig = MINI):
        super().__init__()
        if not _exact_is_practical(config):
            raise ValueError("BeliefQAgent needs the exact belief, which is only "
                             "practical on small boards")
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial = initial
        self.board = config
        self.q: dict[tuple[int, int], dict[int, float]] = {}
        self.visits: dict[tuple[int, int], dict[int, int]] = {}

    # --- Q-table ---------------------------------------------------------------

    def value(self, key, cell: int) -> float:
        return self.q.get(key, {}).get(cell, self.initial)

    def best_actions(self, key, legal: list[int]) -> list[int]:
        values = self.q.get(key, {})
        best = max(values.get(cell, self.initial) for cell in legal)
        return [cell for cell in legal if values.get(cell, self.initial) >= best - _TIE]

    def best_value(self, key, legal: list[int]) -> float:
        values = self.q.get(key, {})
        return max(values.get(cell, self.initial) for cell in legal)

    def update(self, key, cell: int, target: float) -> None:
        row = self.q.setdefault(key, {})
        old = row.get(cell, self.initial)
        if self.alpha is None:          # step 1/n: the running mean of the targets
            counts = self.visits.setdefault(key, {})
            counts[cell] = n = counts.get(cell, 0) + 1
            row[cell] = old + (target - old) / n
        else:
            row[cell] = old + self.alpha * (target - old)

    def _choose(self, key, legal: list[int], rng: random.Random) -> int:
        if self.training and rng.random() < self.epsilon:
            return rng.choice(legal)
        return rng.choice(self.best_actions(key, legal))

    # --- playing through the environment ---------------------------------------

    def new_game(self, config, rng):
        if config != self.board:
            raise ValueError(f"this agent was built for {self.board}, not {config}")
        super().new_game(config, rng)
        self.plan = exact_planner(config)
        self.belief = Belief.prior(config)
        self.pending = []
        self.last = None            # (state key, action) awaiting its update

    def select_shot(self, obs):
        for action, outcome in self.pending:
            self.belief = self.belief.update(action, outcome)
        self.pending = []
        layouts, fired = self.plan.state(self.belief)
        key = self.plan._key(layouts, fired)
        legal = _actions(self.plan, layouts, fired)
        if self.training and self.last is not None:
            self.update(*self.last, -1.0 + self.best_value(key, legal))
        action = self._choose(key, legal, self.rng)
        self.last = (key, action)
        return action

    def observe(self, obs, action, result):
        self.pending.append((action, observed_outcome(obs, result.observation)))

    def end_game(self, record):
        if self.training and self.last is not None:
            self.update(*self.last, -1.0)
        self.last = None

    # --- fast training and exact evaluation ------------------------------------

    def fast_train(self, episodes: int, rng: random.Random, epsilon_start: float = 1.0,
                   epsilon_end: float = 0.0, callback=None, every: int = 0) -> None:
        """Q-learning on simulated games, with epsilon decaying linearly from
        ``epsilon_start`` to ``epsilon_end``. ``callback(episode)`` runs every
        ``every`` episodes."""
        plan = exact_planner(self.board)
        n = plan.everything.bit_length()
        previous, self.training = self.training, True
        try:
            for episode in range(1, episodes + 1):
                self.epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode / episodes
                hidden = rng.randrange(n)       # uniform over layouts, like random_layout
                layouts, fired = plan.everything, 0
                while True:
                    key = plan._key(layouts, fired)
                    cell = self._choose(key, _actions(plan, layouts, fired), rng)
                    layouts = next(g for g in plan.outcomes(layouts, fired, cell) if g >> hidden & 1)
                    fired |= 1 << cell
                    if plan._done(layouts, fired):
                        self.update(key, cell, -1.0)
                        break
                    nxt = plan._key(layouts, fired)
                    self.update(key, cell, -1.0 + self.best_value(nxt, _actions(plan, layouts, fired)))
                if callback and every and episode % every == 0:
                    callback(episode)
        finally:
            self.training = previous

    def exact_value(self) -> float:
        """Exact expected shots of the greedy policy (random tie-breaking) over
        all layouts, placed uniformly at random."""
        plan = exact_planner(self.board)
        return plan.policy_value(
            lambda layouts, fired: self.best_actions(plan._key(layouts, fired),
                                                    _actions(plan, layouts, fired)))

    def num_states(self) -> int:
        return len(self.q)

    # --- saving ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {"version": 1, "alpha": self.alpha, "epsilon": self.epsilon,
                "initial": self.initial, "size": self.board.size, "fleet": list(self.board.fleet),
                "reveal_sunk_cells": self.board.reveal_sunk_cells,
                "q": [[f"{k[0]:x}", f"{k[1]:x}", {str(c): v for c, v in row.items()},
                       {str(c): n for c, n in self.visits.get(k, {}).items()}]
                      for k, row in self.q.items()]}
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> BeliefQAgent:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        config = GameConfig(data["size"], tuple(data["fleet"]),
                            reveal_sunk_cells=data["reveal_sunk_cells"])
        agent = cls(alpha=data["alpha"], epsilon=data["epsilon"], initial=data["initial"],
                    config=config)
        for a, b, row, counts in data["q"]:
            key = (int(a, 16), int(b, 16))
            agent.q[key] = {int(c): v for c, v in row.items()}
            if counts:
                agent.visits[key] = {int(c): n for c, n in counts.items()}
        return agent
