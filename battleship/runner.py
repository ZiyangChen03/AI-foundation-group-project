"""Run one-player games, train agents, and measure shots-to-win."""

import random
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .agents.base import Agent
from .config import MINI, GameConfig
from .env import BattleshipEnv, Observation, Shot, StepResult
from .layout import Layout

# Training uses seeds far from the evaluation seeds (range(N)), so agents are
# never evaluated on layouts they trained on.
TRAIN_SEED = 1_000_000_000


@dataclass(frozen=True)
class GameRecord:
    config: GameConfig
    layout: Layout                   # the hidden layout that was fired at
    shots: tuple[Shot, ...]
    move_times: tuple[float, ...]    # seconds per select_shot call

    @property
    def num_shots(self) -> int:
        return len(self.shots)


def take_shot(agent: Agent, env: BattleshipEnv, obs: Observation,
              move_times: list[float]) -> StepResult:
    """Ask ``agent`` for one shot, time it, apply it, and let the agent observe it."""
    start = time.perf_counter()
    cell = agent.select_shot(obs)
    move_times.append(time.perf_counter() - start)
    try:
        result = env.step(cell)
    except ValueError as error:
        raise ValueError(f"{agent.name} chose an illegal shot: {error}") from error
    agent.observe(obs, cell, result)
    return result


def _agent_rng(seed, agent_seed):
    if agent_seed is not None:
        return random.Random(agent_seed)
    if seed is not None:
        # Derived from the layout seed, but a different stream from it.
        return random.Random(f"agent:{seed}")
    return random.Random()


def play_solo(agent: Agent, config: GameConfig = MINI, seed: int | None = None,
              layout: Layout | None = None, agent_seed: int | None = None) -> GameRecord:
    """Play one game: ``agent`` fires until the hidden fleet is sunk."""
    env = BattleshipEnv(config)
    obs = env.reset(seed=seed, layout=layout)
    agent.new_game(config, _agent_rng(seed, agent_seed))
    move_times = []
    while not env.done:
        obs = take_shot(agent, env, obs, move_times).observation
    record = GameRecord(config, env.reveal_layout(), obs.history, tuple(move_times))
    agent.end_game(record)
    return record


def train(agent: Agent, config: GameConfig, episodes: int,
          seed: int = TRAIN_SEED) -> list[GameRecord]:
    """Play ``episodes`` games with ``agent.training`` switched on."""
    previous, agent.training = agent.training, True
    try:
        return [play_solo(agent, config, seed=seed + i) for i in range(episodes)]
    finally:
        agent.training = previous


def simulate_records(config: GameConfig, n: int, seed: int = TRAIN_SEED,
                     shooter: Agent | None = None) -> list[GameRecord]:
    """Games played by ``shooter`` (default: RandomAgent), e.g. as training data."""
    if shooter is None:
        from .agents import RandomAgent
        shooter = RandomAgent()
    return [play_solo(shooter, config, seed=seed + i) for i in range(n)]


@dataclass
class EvalReport:
    agent: str
    games: int
    mean: float
    std: float
    median: float
    best: int
    worst: int
    histogram: dict[int, int]        # shots -> number of games
    mean_move_time: float            # seconds
    records: list[GameRecord] = field(repr=False)

    def __str__(self):
        return (f"{self.agent:<20} games={self.games:<6} mean={self.mean:6.2f} "
                f"std={self.std:5.2f} median={self.median:5.1f} best={self.best:<3} "
                f"worst={self.worst:<3} move={self.mean_move_time * 1000:.3f} ms")


def evaluate(agent: Agent, config: GameConfig = MINI, seeds: Iterable[int] = range(1000),
             layouts: Iterable[Layout] | None = None) -> EvalReport:
    """Measure shots-to-win with training switched off.

    With ``layouts`` (e.g. ``all_layouts(MINI)``), plays one game per layout;
    otherwise one game per seed. Pass the same seeds or layouts to every agent
    so they face the same boards.
    """
    previous, agent.training = agent.training, False
    try:
        if layouts is not None:
            records = [play_solo(agent, config, layout=layout, agent_seed=i)
                       for i, layout in enumerate(layouts)]
        else:
            records = [play_solo(agent, config, seed=seed) for seed in seeds]
    finally:
        agent.training = previous
    if not records:
        raise ValueError("no games to evaluate")
    shots = [record.num_shots for record in records]
    times = [t for record in records for t in record.move_times]
    return EvalReport(
        agent=agent.name,
        games=len(records),
        mean=statistics.fmean(shots),
        std=statistics.pstdev(shots),
        median=statistics.median(shots),
        best=min(shots),
        worst=max(shots),
        histogram=dict(sorted(Counter(shots).items())),
        mean_move_time=statistics.fmean(times),
        records=records,
    )
