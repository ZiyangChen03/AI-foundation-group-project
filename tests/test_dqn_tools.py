"""Checks of the DQN scripts and package that do not need PyTorch installed."""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _benchmark_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dqn.py"
    spec = importlib.util.spec_from_file_location("benchmark_dqn_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _InProcess:
    """Stands in for ProcessPoolExecutor so the games run in this process."""

    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs):
        return map(fn, jobs)


def test_4x4_paired_margin_comes_from_game_by_game_differences(monkeypatch, tmp_path):
    """Both agents play game g with the same layout and seed, so their coin flips
    are paired. Here belief always needs exactly one shot more than density in the
    same game: the difference is 1 with no uncertainty, though each agent alone
    varies. Treating the two as independent gave a margin of about 0.1 here."""
    module = _benchmark_script()

    def fake_play(job):
        spec, board, games, samples = job
        extra = 1 if spec == "belief-greedy" else 0
        return spec, [(g, 9 + g % 3 + extra, 0) for g in games]

    monkeypatch.setattr(module, "play", fake_play)
    monkeypatch.setattr(module, "ProcessPoolExecutor", _InProcess)
    out = tmp_path / "bench"
    module.main(["mini", "--samples", "3", "--baselines", "probability-density", "belief-greedy",
                 "--jobs", "1", "--out", str(out)])
    rows = {r["agent"]: r for r in json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))}
    assert rows["probability-density"]["ci95"] > 0                    # each agent alone varies
    assert rows["belief-greedy"]["vs_density"] == pytest.approx(1)
    assert rows["belief-greedy"]["vs_density_ci95"] == 0
    assert "±" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_4x4_benchmark_needs_two_games_per_layout(capsys):
    """One game per layout cannot measure random tie-breaking; it used to print ± 0."""
    with pytest.raises(SystemExit):
        _benchmark_script().main(["mini", "--samples", "1"])
    assert "--samples must be at least 2" in capsys.readouterr().err


def test_dqn_package_explains_how_to_install_pytorch(monkeypatch):
    for name in [m for m in sys.modules if m == "battleship.dqn" or m.startswith("battleship.dqn.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)          # as if PyTorch were not installed
    with pytest.raises(ImportError, match="--extra dqn"):
        importlib.import_module("battleship.dqn")
