import random

import pytest
import torch

from battleship import CLASSIC, MINI, BattleshipEnv, GameConfig, all_layouts, play_solo
from battleship.agents import ProbabilityDensityAgent
from battleship.dqn import DQNAgent, NetConfig, QNet, TrainConfig, train
from battleship.dqn.train import eval_layouts, play_greedy
from battleship.dqn.vec_env import (VecBattleship, board_codes, density, normalized_density,
                                    remaining_counts)
from helpers import assert_public_only

NO_REVEAL = GameConfig(6, (3, 3, 2), reveal_sunk_cells=False)
CONFIGS = [MINI, CLASSIC, NO_REVEAL]


@pytest.mark.parametrize("config", CONFIGS)
def test_vector_games_match_the_environment(config):
    rng = random.Random(1)
    layouts = eval_layouts(config, 12, first_seed=40)
    vec = VecBattleship(config, layouts=layouts)
    envs = [BattleshipEnv(config) for _ in layouts]
    obs = [env.reset(layout=layout) for env, layout in zip(envs, layouts)]
    while not vec.done.all():
        active = ~vec.done
        actions = torch.zeros(vec.n, dtype=torch.long)
        for i in active.nonzero().flatten().tolist():
            actions[i] = rng.choice(obs[i].legal_actions)
        done = vec.step(actions, active)
        for i in active.nonzero().flatten().tolist():
            result = envs[i].step(int(actions[i]))
            obs[i] = result.observation
            assert bool(done[i]) == result.done
        for i, o in enumerate(obs):
            assert torch.equal(vec.board[i], board_codes(o))
            assert torch.equal(vec.remaining()[i], remaining_counts(o))
            assert int(vec.shots[i]) == o.num_shots


def test_vector_games_reject_a_repeated_shot():
    vec = VecBattleship(MINI, 2, seed=0)
    vec.step(torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        vec.step(torch.tensor([0, 1]))


def test_random_layouts_follow_the_seed():
    a, b = VecBattleship(CLASSIC, 5, seed=3), VecBattleship(CLASSIC, 5, seed=3)
    assert torch.equal(a.ship_at, b.ship_at)


@pytest.mark.parametrize("config", CONFIGS)
def test_density_matches_the_probability_agent(config):
    agent = ProbabilityDensityAgent()
    rng = random.Random(2)
    for layout in eval_layouts(config, 6, first_seed=7):
        env = BattleshipEnv(config)
        obs = env.reset(layout=layout)
        while not env.done:
            mine = density(board_codes(obs)[None], remaining_counts(obs)[None], config)[0]
            assert mine.tolist() == [float(v) for v in agent.density(obs)]
            obs = env.step(rng.choice(obs.legal_actions)).observation


def test_normalized_density_peaks_at_one():
    vec = VecBattleship(CLASSIC, 3, seed=0)
    d = normalized_density(vec.board, vec.remaining(), CLASSIC)
    assert torch.allclose(d.amax(1), torch.ones(3))


@pytest.mark.parametrize("net", [NetConfig(8, 2), NetConfig(8, 2, use_density=True),
                                 NetConfig(8, 2, residual=1.0)])
def test_untrained_agent_plays_legal_games_from_public_information(net):
    torch.manual_seed(0)
    agent = DQNAgent(QNet(net), MINI)
    records = [play_solo(agent, MINI, layout=layout, agent_seed=0) for layout in all_layouts(MINI)]
    assert all(r.num_shots <= MINI.num_cells for r in records)
    assert_public_only(records)


def test_residual_network_starts_as_probability_density():
    agent = DQNAgent(QNet(NetConfig(8, 2, residual=1.0)), CLASSIC)
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=5)
    rng = random.Random(0)
    for _ in range(40):
        scores = ProbabilityDensityAgent().density(obs)
        best = max(scores[c] for c in obs.legal_actions)
        assert scores[agent.select_shot(obs)] == best
        obs = env.step(rng.choice(obs.legal_actions)).observation


def test_greedy_vector_play_matches_the_agent():
    torch.manual_seed(1)
    model = QNet(NetConfig(8, 2))
    layouts = list(all_layouts(MINI))[:30]
    agent = DQNAgent(model, MINI)
    expected = [play_solo(agent, MINI, layout=layout, agent_seed=0).num_shots for layout in layouts]
    assert play_greedy(model, MINI, layouts, "cpu").tolist() == expected


def test_short_training_run_learns_and_round_trips(tmp_path):
    layouts = list(all_layouts(MINI))
    tc = TrainConfig(games=3000, parallel=64, batch=64, learn_start=500, buffer=20_000,
                     eps_fraction=0.5, eval_every=1500, target_every=200, device="cpu")
    model, rows = train(MINI, NetConfig(16, 3), tc, layouts, log=lambda _: None)
    assert rows[-1]["games"] >= 3000
    assert rows[-1]["eval_mean"] < rows[0]["eval_mean"] - 1      # clearly better than untrained
    agent = DQNAgent(model, MINI)
    agent.save(tmp_path / "m.pt")
    again = DQNAgent.load(tmp_path / "m.pt")
    for layout in layouts[:20]:
        assert (play_solo(agent, MINI, layout=layout, agent_seed=0).num_shots
                == play_solo(again, MINI, layout=layout, agent_seed=0).num_shots)
