import copy
import random

import pytest

from battleship import MINI, CLASSIC, all_layouts, evaluate, play_solo, train
from battleship.env import BattleshipEnv, Observation, CellState, StepResult, Shot, Result
from battleship.q_learning import QLearningAgent
from battleship.q_learning.Q_learning import split_layouts


def test_update_masks_illegal_actions():
    env = BattleshipEnv(MINI)
    obs = env.reset(seed=2)
    agent = QLearningAgent(alpha=.5, gamma=.9)
    agent.training = True
    result = env.step(0)
    agent.q[agent.state(result.observation)] = {i: -2. for i in result.observation.legal_actions}
    agent.q[agent.state(result.observation)][0] = 1000.
    agent.observe(obs, 0, result)
    assert agent.q[agent.state(obs)][0] == pytest.approx(-1.4)


def test_terminal_does_not_bootstrap_even_with_unshot_cells():
    agent = QLearningAgent(alpha=1.)
    agent.training = True
    obs = BattleshipEnv(MINI).reset(seed=1)
    terminal = Observation(MINI, obs.board, (), ())
    agent.q[agent.state(terminal)] = {0: 1000.}
    agent.observe(obs, 0, StepResult(terminal, -1., True, Shot(0, Result.SUNK, 2)))
    assert agent.q[agent.state(obs)][0] == -1.


def test_eval_is_greedy_and_does_not_mutate():
    agent = QLearningAgent(epsilon=1.)
    agent.new_game(MINI, random.Random(0))
    env = BattleshipEnv(MINI)
    obs = env.reset(seed=1)
    agent.q[agent.state(obs)] = {3: 10.}
    before = copy.deepcopy(agent.q)
    assert {agent.select_shot(obs) for _ in range(50)} == {3}
    agent.observe(obs, 3, env.step(3))
    evaluate(agent, MINI, seeds=range(10))
    assert agent.q == before
    assert agent.updates == 0


@pytest.mark.parametrize("epsilon", [0., 1.])
def test_only_legal_shots_and_random_ties(epsilon):
    agent = QLearningAgent(epsilon=epsilon)
    agent.new_game(MINI, random.Random(0))
    agent.training = True
    env = BattleshipEnv(MINI)
    obs = env.step(0).observation
    agent.q[agent.state(obs)] = {0: 1000.}
    shots = {agent.select_shot(obs) for _ in range(200)}
    assert shots == set(obs.legal_actions)


def test_state_uses_public_board_and_remaining_lengths():
    obs = BattleshipEnv(MINI).reset(seed=0)
    board = (CellState.MISS,) + obs.board[1:]
    assert QLearningAgent.state(obs) != QLearningAgent.state(Observation(MINI, board, (), (3, 2)))
    assert QLearningAgent.state(obs) != QLearningAgent.state(Observation(MINI, obs.board, (), (2,)))
    assert QLearningAgent.state(obs) == QLearningAgent.state(BattleshipEnv(MINI).reset(seed=99))


def test_training_reproducible_save_load_and_all_layouts(tmp_path):
    a, b = QLearningAgent(), QLearningAgent()
    train(a, MINI, 100)
    train(b, MINI, 100)
    assert a.q == b.q and a.updates > 0
    path = tmp_path / "model.json.gz"
    a.save(path)
    loaded = QLearningAgent.load(path)
    assert loaded.q == a.q and loaded.updates == a.updates
    assert not loaded.training
    assert play_solo(a, MINI, seed=42).shots == play_solo(loaded, MINI, seed=42).shots
    report = evaluate(loaded, MINI, layouts=all_layouts(MINI))
    assert report.games == 264
    assert report.best >= 5 and report.worst <= 16


def test_splits_are_disjoint_and_complete():
    parts = split_layouts()
    keys = [set(tuple(tuple(sorted(ship.cells)) for ship in layout) for layout in part)
            for part in parts]
    assert sum(map(len, keys)) == 264
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert parts == split_layouts()


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"gamma": 1.1}, {"epsilon": -1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        QLearningAgent(**kwargs)


def test_rejects_other_boards():
    with pytest.raises(ValueError, match="MINI"):
        play_solo(QLearningAgent(), CLASSIC)
