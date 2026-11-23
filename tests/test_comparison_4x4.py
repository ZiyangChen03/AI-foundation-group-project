"""Protocol checks: frozen evaluation, exact budgets and end-to-end artifacts."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from battleship import MINI
from battleship.agents import BeliefQAgent
from battleship.belief import Belief
from battleship.comparison import ModelSpec, aggregate, evaluate_layouts, load_specs, weighted_stats
from battleship.q_learning import QLearningAgent
from battleship.agents import OpponentBeliefAgent
from battleship.opponents import standard_habits


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cls", [QLearningAgent, BeliefQAgent])
def test_evaluation_freezes_learning_and_restores_training_flag(cls):
    agent = cls()
    agent.training = True
    before = copy.deepcopy(agent.q)
    records = evaluate_layouts(agent, range(2), Belief.prior(MINI).layouts[:2])
    assert len(records) == 4
    assert agent.q == before
    assert agent.training is True
    repeated = evaluate_layouts(agent, range(2), Belief.prior(MINI).layouts[:2])
    assert [r.shots for _, _, r in records] == [r.shots for _, _, r in repeated]


@pytest.mark.parametrize("kind", ["raw-q", "belief-q"])
def test_tabular_budget_counts_last_nonterminal_transition(kind):
    train = script("train_comparison_4x4")
    agent, rows = train.train_tabular(kind, 0, 11, 7, 1, .05)
    assert [r["training_shots"] for r in rows] == [0, 7, 11]
    if kind == "raw-q":
        assert agent.updates == 11
    else:
        assert sum(sum(v.values()) for v in agent.visits.values()) == 11
    assert agent.training is False


def test_weighted_statistics_and_separate_seed_variation():
    from types import SimpleNamespace
    records = [(0, 0, SimpleNamespace(num_shots=5, move_times=[.001] * 5)),
               (1, 0, SimpleNamespace(num_shots=9, move_times=[.002] * 9))]
    stats = weighted_stats(records, [.75, .25])
    assert stats["mean"] == 6
    assert stats["std"] == pytest.approx(3 ** .5)
    assert stats["decision_ms"] == pytest.approx((.75 * 5 + .25 * 18) / 6)
    a = aggregate([{"mean": 5, "std": 0, "worst": 5, "decision_ms": 1},
                   {"mean": 9, "std": 0, "worst": 9, "decision_ms": 1}])
    assert a["mean"] == 7
    assert a["seed_std"] == a["std"] == 2
    with pytest.raises(ValueError, match="probability"):
        weighted_stats(records, [1, 1])


def test_tournament_pairs_training_seeds_and_broadcasts_baseline():
    compare = script("compare_4x4")
    baseline = [ModelSpec("Random", "random")]
    learned = [ModelSpec("Q", "raw-q", seed=s) for s in (0, 1, 2)]
    assert len(compare.paired_specs(baseline, learned)) == 3
    with pytest.raises(ValueError, match="same training seed"):
        compare.paired_specs(learned, learned[:1])


def test_train_load_compare_and_checkpoint_integrity(tmp_path):
    train, compare = script("train_comparison_4x4"), script("compare_4x4")
    models, output = tmp_path / "models", tmp_path / "comparison"
    train.main(["--methods", "raw-q", "--seeds", "0", "--shots", "7", "--every", "7",
                "--policy-samples", "1", "--output", str(models)])
    specs = load_specs(models)
    assert specs[0].training_shots == 7
    model_bytes = Path(specs[0].checkpoint).read_bytes()
    train.main(["--methods", "raw-q", "--seeds", "0", "--shots", "7", "--every", "7",
                "--policy-samples", "1", "--output", str(models), "--resume"])
    assert Path(specs[0].checkpoint).read_bytes() == model_bytes
    compare.main(["--trained", str(models), "--only", "random", "raw-q", "--samples", "1",
                  "--games", "2", "--skip-oracle", "--output", str(output)])
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    for name in ["games.csv", "uniform.csv", "summary.csv", "habits.csv", "cases.json",
                 "cases.md", "tournament.csv", "tournament_pairs.csv", "results.md", "learning_curves.svg"]:
        assert (output / name).is_file()
    recovered = tmp_path / "recovered"
    compare.main(["--trained", str(models), "--only", "random", "raw-q", "--samples", "1",
                  "--games", "2", "--skip-oracle", "--reuse-solo", str(output), "--output", str(recovered)])
    assert (recovered / "games.csv").read_bytes() == (output / "games.csv").read_bytes()
    assert (recovered / "uniform.csv").read_bytes() == (output / "uniform.csv").read_bytes()
    assert json.loads((recovered / "manifest.json").read_text())["reused_solo"]
    # No output gets silently overwritten, nor can a partial comparison claim completeness.
    with pytest.raises(SystemExit):
        compare.main(["--trained", str(models), "--require-complete", "--output", str(tmp_path / "invalid")])
    with Path(specs[0].checkpoint).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_specs(models)


@pytest.mark.parametrize("net_options", [{}, {"use_density": True}, {"use_density": True, "residual": 1.0}])
def test_dqn_budget_ends_at_partial_vector_batch_and_runs_updates(net_options):
    torch = pytest.importorskip("torch")
    from battleship.dqn.model import NetConfig
    from battleship.dqn.train import TrainConfig, train

    torch.set_num_threads(1)
    tc = TrainConfig(parallel=4, batch=4, buffer=32, learn_start=1,
                     replay_ratio=1, device="cpu", seed=0)
    model, rows = train(MINI, NetConfig(channels=4, layers=1, **net_options), tc,
                        list(Belief.prior(MINI).layouts[:2]), log=lambda _: None,
                        shot_budget=19, eval_shots=7)
    assert [r["training_shots"] for r in rows] == [0, 7, 14, 19]
    assert rows[-1]["grad_steps"] > 0
    assert all(r["training_seconds"] >= 0 for r in rows)


def test_all_learning_methods_export_loadable_final_models(tmp_path):
    pytest.importorskip("torch")
    train, compare = script("train_comparison_4x4"), script("compare_4x4")
    models, output = tmp_path / "all_models", tmp_path / "dqn_comparison"
    train.main(["--seeds", "0", "--shots", "19", "--every", "7", "--parallel", "4",
                "--device", "cpu", "--policy-samples", "1", "--output", str(models)])
    specs = load_specs(models)
    assert {s.kind for s in specs} == set(train.LABELS)
    assert all(s.training_shots == 19 for s in specs)
    compare.main(["--trained", str(models), "--only", "dqn-board", "dqn-density", "dqn-residual",
                  "--quick", "--skip-oracle", "--output", str(output)])
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("habit", standard_habits(MINI), ids=lambda h: h.name)
def test_oracle_only_evaluates_layouts_in_its_prior_support(habit):
    weights = habit.distribution(MINI)
    records = evaluate_layouts(OpponentBeliefAgent(habit=habit), range(1), weights=weights)
    assert {i for i, _, _ in records} == {i for i, w in enumerate(weights) if w > 0}
    stats = weighted_stats(records, weights)
    assert 5 <= stats["mean"] <= 16
