"""Two-player games: both fleets placed randomly, players alternate shots."""

import random
from dataclasses import dataclass
from itertools import combinations

from .agents.base import Agent
from .config import CLASSIC, GameConfig
from .env import BattleshipEnv
from .runner import GameRecord, take_shot


@dataclass(frozen=True)
class MatchRecord:
    config: GameConfig
    players: tuple[str, str]
    first: int                                # index of the player who shot first
    winner: int
    records: tuple[GameRecord, GameRecord]    # records[i] = player i firing at the other's fleet


def play_match(a: Agent, b: Agent, config: GameConfig = CLASSIC, first: int = 0,
               seed: int | None = None) -> MatchRecord:
    """Players alternate one shot each; the first to sink the enemy fleet wins."""
    if first not in (0, 1):
        raise ValueError("first must be 0 or 1")
    if a is b:
        raise ValueError("a match needs two separate agent objects")
    rng = random.Random(seed)
    agents = (a, b)
    envs = [BattleshipEnv(config) for _ in agents]
    observations = [env.reset(seed=rng.getrandbits(64)) for env in envs]
    for agent in agents:
        agent.new_game(config, random.Random(rng.getrandbits(64)))
    move_times = ([], [])
    turn = first
    while True:
        result = take_shot(agents[turn], envs[turn], observations[turn], move_times[turn])
        observations[turn] = result.observation
        if result.done:
            break
        turn = 1 - turn
    # The loser's game is unfinished, so read its layout directly.
    records = tuple(GameRecord(config, env._layout, obs.history, tuple(times))
                    for env, obs, times in zip(envs, observations, move_times))
    for agent, record in zip(agents, records):
        agent.end_game(record)
    return MatchRecord(config, (a.name, b.name), first, turn, records)


@dataclass
class TournamentReport:
    players: list[str]
    wins: list[list[int]]     # wins[i][j] = games player i won against player j
    games: list[list[int]]    # games[i][j] = games played between i and j
    first_player_wins: int
    total_games: int

    @property
    def win_rate(self) -> list[list[float | None]]:
        """win_rate[i][j] = fraction of games player i won against player j."""
        return [[wins / games if games else None for wins, games in zip(win_row, game_row)]
                for win_row, game_row in zip(self.wins, self.games)]

    def __str__(self):
        width = max(len(name) for name in self.players) + 2
        lines = ["win rate of row vs column".ljust(width)
                 + "".join(name.rjust(width) for name in self.players)]
        for name, row in zip(self.players, self.win_rate):
            lines.append(name.ljust(width) + "".join(
                ("-" if rate is None else f"{rate:.3f}").rjust(width) for rate in row))
        lines.append(f"first player won {self.first_player_wins} of {self.total_games} games")
        return "\n".join(lines)


def round_robin(agents: list[Agent], config: GameConfig = CLASSIC,
                games_per_pair: int = 1000, seed: int = 0) -> TournamentReport:
    """Every pair plays ``games_per_pair`` matches, alternating who shoots first."""
    if len(agents) < 2:
        raise ValueError("a tournament needs at least two agents")
    if len({id(agent) for agent in agents}) != len(agents):
        raise ValueError("each agent must be a separate object")
    rng = random.Random(seed)
    n = len(agents)
    wins = [[0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]
    first_player_wins = 0
    previous = [agent.training for agent in agents]
    for agent in agents:
        agent.training = False
    try:
        for i, j in combinations(range(n), 2):
            for game in range(games_per_pair):
                match = play_match(agents[i], agents[j], config, first=game % 2,
                                   seed=rng.getrandbits(64))
                winner, loser = (i, j) if match.winner == 0 else (j, i)
                wins[winner][loser] += 1
                games[i][j] += 1
                games[j][i] += 1
                first_player_wins += match.winner == match.first
    finally:
        for agent, training in zip(agents, previous):
            agent.training = training
    return TournamentReport([agent.name for agent in agents], wins, games,
                            first_player_wins, games_per_pair * n * (n - 1) // 2)
