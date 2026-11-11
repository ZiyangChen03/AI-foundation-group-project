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
    agent.q[agent.state(obs)] = {0: 0.}       # start from 0 so the target is easy to read
    agent.observe(obs, 0, result)
    # target = -1 + 0.9 * max over LEGAL next actions (-2; the illegal 1000 is ignored)
    assert agent.q[agent.state(obs)][0] == pytest.approx(0 + .5 * (-2.8 - 0))


def test_untried_actions_start_at_a_lower_bound():
    """Every reward is -1, so a 0.0 default would rank an untried action above
    every tried one. Untried actions start at -(number of untried cells)."""
    agent = QLearningAgent(alpha=1., epsilon=0.)
    agent.new_game(MINI, random.Random(0))
    env = BattleshipEnv(MINI)
    obs = env.reset(seed=1)
    assert QLearningAgent.default(obs) == -16.
    agent.q[agent.state(obs)] = {3: -10.}     # one tried action, poor but above the bound
    assert {agent.select_shot(obs) for _ in range(50)} == {3}
    agent.training = True
    result = env.step(15)
    assert not result.done
    agent.observe(obs, 15, result)            # next board never seen: bootstrap the bound
    assert agent.q[agent.state(obs)][15] == -1. + QLearningAgent.default(result.observation)


def test_terminal_does_not_bootstrap_even_with_unshot_cells():
    agent = QLearningAgent(alpha=1.)
    agent.training = True
    obs = BattleshipEnv(MINI).reset(seed=1)
    terminal = Observation(MINI, obs.board, (), ())
    agent.q[agent.state(terminal)] = {0: 1000.}
    agent.observe(obs, 0, StepResult(terminal, -1., True, Shot(0, Result.SUNK, 2)))
    assert agent.q[agent.state(obs)][0] == -1.


def test_default_step_is_a_running_average():
    """alpha=None uses step 1/N(s, a): the Q-value is the mean of its targets,
    and the starting value no longer matters after the first update."""
    agent = QLearningAgent(epsilon=0.)
    agent.training = True
    obs = BattleshipEnv(MINI).reset(seed=1)
    terminal = Observation(MINI, obs.board, (), ())
    for reward in (-1., -2., -6.):              # three terminal targets for (obs, 0)
        agent.observe(obs, 0, StepResult(terminal, reward, True, Shot(0, Result.SUNK, 2)))
    assert agent.q[agent.state(obs)][0] == pytest.approx(-3.)
    assert agent.counts[agent.state(obs)][0] == 3


def test_save_load_keeps_counts_so_training_continues_exactly(tmp_path):
    a = QLearningAgent()
    train(a, MINI, 50)
    path = tmp_path / "model.json.gz"
    a.save(path)
    b = QLearningAgent.load(path)
    assert b.counts == a.counts and b.alpha is None
    train(a, MINI, 50, seed=7)
    train(b, MINI, 50, seed=7)
    assert a.q == b.q


def test_loads_version_1_files(tmp_path):
    import gzip, json
    path = tmp_path / "old.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"version": 1, "config": "MINI", "alpha": .1, "gamma": 1., "epsilon": .1,
                   "updates": 1, "q": {"s": {"3": -2.}}}, stream)
    agent = QLearningAgent.load(path)
    assert agent.alpha == .1 and agent.q == {"s": {3: -2.}} and agent.counts == {}


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


def test_exploration_fires_at_any_legal_cell():
    agent = QLearningAgent(epsilon=1.)
    agent.new_game(MINI, random.Random(0))
    agent.training = True
    env = BattleshipEnv(MINI)
    obs = env.step(0).observation
    agent.q[agent.state(obs)] = {0: 1000.}    # cell 0 is already shot: never chosen
    shots = {agent.select_shot(obs) for _ in range(200)}
    assert shots == set(obs.legal_actions)


def test_greedy_ties_go_to_the_lowest_cell():
    """Ties are broken the same way every time, so greedy play keeps reaching
    the same boards; an already-shot cell is never chosen, however high."""
    agent = QLearningAgent(epsilon=0.)
    agent.new_game(MINI, random.Random(0))
    env = BattleshipEnv(MINI)
    obs = env.step(0).observation
    agent.q[agent.state(obs)] = {0: 1000.}
    assert {agent.select_shot(obs) for _ in range(50)} == {1}      # all legal cells tie
    agent.q[agent.state(obs)] = {7: -3., 12: -3., 4: -9.}
    assert agent.select_shot(obs) == 7                              # best two tie: lowest wins


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


def run_pipeline(tmp_path, *protocol):
    """Train 30 episodes for seed 0, then evaluate; return what was written."""
    import json
    from battleship.q_learning.Q_learning import test_main, train_main
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    train_main(["--episodes", "30", "--seeds", "0", "--eval-every", "10",
                "--output", str(artifacts), *protocol])
    test_main([str(artifacts), "--output", str(results)])
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    report = json.loads((results / "test_results.json").read_text(encoding="utf-8"))
    q_row = next(row for row in report["per_run"] if row["seed"] == 0)
    return artifacts, config, q_row, (results / "README.md").read_text(encoding="utf-8")


def test_default_protocol_trains_and_tests_on_all_264_layouts(tmp_path):
    artifacts, config, q_row, readme = run_pipeline(tmp_path)
    assert config["protocol"] == "all" and "splits" not in config
    assert q_row["games"] == 264
    assert q_row["selected_episode"] == 30                     # final model, no selection
    reported = QLearningAgent.load(artifacts / "seed_0.json.gz")
    final = QLearningAgent.load(artifacts / "seed_0_final.json.gz")
    assert reported.q == final.q
    assert "all 264 layouts" in readme and "held-out" not in readme


def test_held_out_protocol_still_available(tmp_path):
    artifacts, config, q_row, readme = run_pipeline(tmp_path, "--protocol", "held-out")
    assert config["protocol"] == "held-out" and len(config["splits"]["test"]) == 40
    assert q_row["games"] == 40
    import csv
    with (artifacts / "seed_0_curve.csv").open(encoding="utf-8", newline="") as stream:
        curve = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    assert q_row["selected_episode"] == min(curve, key=lambda p: p["validation_mean"])["episode"]


def test_runs_saved_before_the_protocol_option_load_as_held_out(tmp_path):
    import json
    from battleship.q_learning.Q_learning import load_experiment
    artifacts, config, _, _ = run_pipeline(tmp_path, "--protocol", "held-out")
    del config["protocol"]
    (artifacts / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _, test_layouts, _, _ = load_experiment(artifacts)
    assert len(test_layouts) == 40


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
