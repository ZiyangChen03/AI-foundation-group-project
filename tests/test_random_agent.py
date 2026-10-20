import random
from collections import Counter

from battleship import MINI, BattleshipEnv
from battleship.agents import RandomAgent


def test_only_fires_at_untried_cells():
    agent, env = RandomAgent(), BattleshipEnv(MINI)
    obs = env.reset(seed=0)
    agent.new_game(MINI, random.Random(0))
    tried = set()
    while not env.done:
        cell = agent.select_shot(obs)
        assert cell in obs.legal_actions and cell not in tried
        tried.add(cell)
        obs = env.step(cell).observation


def test_choice_is_uniform():
    agent, env = RandomAgent(), BattleshipEnv(MINI)
    obs = env.reset(seed=0)
    for cell in (0, 5, 10):                         # some cells already tried
        obs = env.step(cell).observation
    counts = Counter()
    for seed in range(13_000):
        agent.new_game(MINI, random.Random(seed))
        counts[agent.select_shot(obs)] += 1
    assert set(counts) == set(obs.legal_actions)    # 13 untried cells, ~1000 picks each
    assert all(850 < count < 1150 for count in counts.values())
