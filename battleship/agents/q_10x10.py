"""10x10 versions of the two tabular Q-learners.

The originals are 4x4 only, and stay unchanged:

- ``battleship/q_learning/Q_learning.py`` (``QLearningAgent``) keys its Q-table
  on the whole visible board. A 10x10 board almost never repeats, so every
  state would be new and the table could not learn.
- ``battleship/agents/belief_q.py`` (``BeliefQAgent``) keys it on the exact
  belief, the full list of consistent layouts, which cannot be listed on 10x10.

Both versions here keep the algorithm: tabular one-step Q-learning, step
size 1/N(key) (the running mean of the targets), a pessimistic start value, and
epsilon-greedy exploration. What changes is the key of the table, so that keys
repeat on 10x10:

- ``LocalQAgent`` (raw-board version): the key of firing at a cell is what the
  board looks like around it, the 3x3 neighbourhood (each neighbour unknown,
  blocked, i.e. a miss, a sunk ship or off the board, or an unsunk hit) plus
  the cell's colour on a checkerboard. 3^8 x 2 = 13,122 keys, shared by every
  cell of every board.
- ``ParticleBeliefQAgent`` (belief-state version): the key is what the
  particle belief says about the cell: P(ship there) in one of ``bins`` equal
  bins, and whether some ship is hit but not yet sunk.

Q(s, a) is the value of the key of (s, a), so the update is

    Q[key(s, a)] <- Q[key(s, a)] + (target - Q[key(s, a)]) / N(key(s, a))
    target = r                                        if the shot ends the game
    target = r + gamma * max over a' of Q[key(s', a')]  otherwise

The reward and discount differ from the originals by default, for a measured
reason. Every game has the same number of hits (17 on 10x10), so shots = hits
+ misses, and a reward of -1 per miss (``reward="miss"``) ranks strategies
exactly like -1 per shot. With gamma = 1 a key's value is the expected rest of
the game, which depends mostly on how far into the game that neighbourhood
tends to appear, not on the shot: squares next to misses appear late, look
valuable, and get fired at. That version (``reward="shot", gamma=1``, like the
originals) played 92.3 shots on 200 test boards after 20,000 games, close to
random (95). With ``gamma=0`` each key learns the chance its shot misses, which
a key can predict, so that is the default.

Keys never seen are worth ``initial`` (pessimistic), so greedy play prefers
what it has learned. Both agents see only the public board, like the originals.
"""

from __future__ import annotations

import gzip
import json
import random

from ..env import CellState, Result
from .base import Agent
from .belief import BeliefAgent

UNKNOWN, BLOCKED, OPEN_HIT = 0, 1, 2
_CODE = {CellState.UNKNOWN: UNKNOWN, CellState.MISS: BLOCKED, CellState.SUNK: BLOCKED,
         CellState.HIT: OPEN_HIT}
_TIE = 1e-12


class _TabularQ:
    """The Q-table shared by both agents: values and update counts per key."""

    def __init__(self, initial: float):
        self.initial = initial
        self.q: dict = {}
        self.n: dict = {}
        self.updates = 0

    def value(self, key) -> float:
        return self.q.get(key, self.initial)

    def update(self, key, target: float) -> None:
        self.n[key] = n = self.n.get(key, 0) + 1
        old = self.q.get(key, self.initial)
        self.q[key] = old + (target - old) / n
        self.updates += 1


class _KeyedQAgent(Agent):
    """Q-learning where the value of firing at a cell is the value of a key."""

    def __init__(self, epsilon: float, initial: float, gamma: float, reward: str):
        super().__init__()
        if reward not in ("miss", "shot") or not 0 <= gamma <= 1:
            raise ValueError('reward must be "miss" or "shot", and 0 <= gamma <= 1')
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward = reward
        self.table = _TabularQ(initial)
        self.last_key = None
        self.last_reward = -1.0

    def observe(self, obs, action, result):
        missed = result.shot.result is Result.MISS
        self.last_reward = -1.0 if (missed or self.reward == "shot") else 0.0

    def keys(self, obs) -> dict[int, object]:
        raise NotImplementedError

    def _choose(self, obs, keys) -> int:
        legal = obs.legal_actions
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        values = {cell: self.table.value(keys[cell]) for cell in legal}
        best = max(values.values())
        return self.rng.choice([cell for cell, v in values.items() if v >= best - _TIE])

    def select_shot(self, obs):
        keys = self.keys(obs)
        if self.training and self.last_key is not None:
            following = max(self.table.value(keys[cell]) for cell in obs.legal_actions)
            self.table.update(self.last_key, self.last_reward + self.gamma * following)
        cell = self._choose(obs, keys)
        self.last_key = keys[cell]
        return cell

    def end_game(self, record):
        if self.training and self.last_key is not None:
            self.table.update(self.last_key, self.last_reward)
        self.last_key = None

    def num_keys(self) -> int:
        return len(self.table.q)

    def _dump(self, extra: dict) -> dict:
        return {"version": 1, "class": type(self).__name__, "epsilon": self.epsilon,
                "initial": self.table.initial, "gamma": self.gamma, "reward": self.reward,
                "updates": self.table.updates,
                "q": [[list(k) if isinstance(k, tuple) else k, v, self.table.n.get(k, 0)]
                      for k, v in self.table.q.items()], **extra}

    def save(self, path) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(self._dump({}), f)

    def _restore(self, data: dict) -> None:
        for key, value, count in data["q"]:
            key = tuple(key) if isinstance(key, list) else key
            self.table.q[key] = value
            if count:
                self.table.n[key] = count
        self.table.updates = data["updates"]


class LocalQAgent(_KeyedQAgent):
    """10x10 version of the raw-board Q-table: keyed on the 3x3 neighbourhood."""

    name = "q-learning-local"

    def __init__(self, epsilon: float = 0.1, initial: float = -100.0, gamma: float = 0.0,
                 reward: str = "miss"):
        super().__init__(epsilon, initial, gamma, reward)

    def keys(self, obs) -> dict[int, tuple]:
        size = obs.config.size
        codes = [_CODE[state] for state in obs.board]
        keys = {}
        for cell in obs.legal_actions:
            row, col = divmod(cell, size)
            around = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    around.append(codes[r * size + c] if 0 <= r < size and 0 <= c < size
                                  else BLOCKED)
            keys[cell] = (*around, (row + col) % 2)
        return keys

    @classmethod
    def load(cls, path) -> LocalQAgent:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("class") != cls.__name__:
            raise ValueError(f"not a {cls.__name__} file")
        agent = cls(data["epsilon"], data["initial"], data["gamma"], data["reward"])
        agent._restore(data)
        return agent


class ParticleBeliefQAgent(_KeyedQAgent):
    """10x10 version of the belief-state Q-table: keyed on what the particle
    belief says about each cell."""

    name = "belief-q-particles"

    def __init__(self, epsilon: float = 0.1, initial: float = -100.0, bins: int = 20,
                 num_particles: int = 300, gamma: float = 0.0, reward: str = "miss"):
        super().__init__(epsilon, initial, gamma, reward)
        self.bins = bins
        self.num_particles = num_particles
        self.tracker = BeliefAgent(belief="particles", num_particles=num_particles)

    def new_game(self, config, rng):
        super().new_game(config, rng)
        self.tracker.new_game(config, random.Random(rng.getrandbits(64)))

    def observe(self, obs, action, result):
        super().observe(obs, action, result)
        self.tracker.observe(obs, action, result)

    def keys(self, obs) -> dict[int, tuple]:
        self.tracker.select_shot(obs)               # applies the Bayes update to the particles
        belief = self.tracker.belief
        probabilities = belief.occupancy()
        targeting = int(self.tracker._targeting())
        return {cell: (min(self.bins - 1, int(probabilities[cell] * self.bins)), targeting)
                for cell in obs.legal_actions}

    def save(self, path) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(self._dump({"bins": self.bins, "num_particles": self.num_particles}), f)

    @classmethod
    def load(cls, path, num_particles: int | None = None) -> ParticleBeliefQAgent:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("class") != cls.__name__:
            raise ValueError(f"not a {cls.__name__} file")
        agent = cls(data["epsilon"], data["initial"], data["bins"],
                    num_particles or data["num_particles"], data["gamma"], data["reward"])
        agent._restore(data)
        return agent
