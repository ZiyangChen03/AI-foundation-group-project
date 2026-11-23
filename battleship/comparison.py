"""Shared, dependency-light helpers for the reproducible 4x4 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

from . import MINI, play_solo
from .agents import (BeliefAgent, BeliefQAgent, HuntTargetAgent,
                     ProbabilityDensityAgent, RandomAgent)
from .agents.belief import PLANNERS
from .belief import Belief


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    seed: int | None = None
    checkpoint: str | None = None
    planner: str | None = None
    training_shots: int | None = None
    training_games: int | None = None
    training_seconds: float | None = None
    curve: str | None = None

    def load(self):
        if self.kind == "random":
            agent = RandomAgent()
        elif self.kind == "hunt":
            agent = HuntTargetAgent(parity=False)
        elif self.kind == "parity":
            agent = HuntTargetAgent(parity=True)
        elif self.kind == "density":
            agent = ProbabilityDensityAgent()
        elif self.kind == "belief":
            agent = BeliefAgent(planner=self.planner)
        elif self.kind == "raw-q":
            from .q_learning import QLearningAgent
            agent = QLearningAgent.load(self.checkpoint)
        elif self.kind == "belief-q":
            agent = BeliefQAgent.load(self.checkpoint)
        elif self.kind.startswith("dqn-"):
            from .dqn import DQNAgent
            agent = DQNAgent.load(self.checkpoint)
        else:
            raise ValueError(f"unknown agent kind: {self.kind}")
        agent.name = self.label
        agent.training = False
        return agent


def baselines():
    specs = [ModelSpec("Random", "random"), ModelSpec("Hunt-target", "hunt"),
             ModelSpec("Hunt-target + parity", "parity"),
             ModelSpec("Probability density", "density")]
    labels = {"greedy": "POMDP greedy", "rollout": "POMDP rollout",
              "optimal": "POMDP optimal", "infogain": "POMDP info gain",
              "greedy-info": "POMDP greedy + info tie-break"}
    specs += [ModelSpec(labels[p], "belief", planner=p) for p in PLANNERS]
    return specs


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    # Hash working sources as well: HEAD alone cannot identify uncommitted code.
    paths = [*root.joinpath("battleship").rglob("*.py"),
             *root.joinpath("scripts").glob("*.py"), root / "pyproject.toml"]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "python": sys.version, "platform": platform.platform(),
            "sources": {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}}


def load_specs(folder):
    folder = Path(folder).resolve()
    data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"training is incomplete: {folder}")
    specs = []
    for row in data["models"]:
        row = dict(row)
        expected_hash = row.pop("sha256")
        for key in ("checkpoint", "curve"):
            row[key] = str(folder / row[key])
        if digest(row["checkpoint"]) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {row['checkpoint']}")
        specs.append(ModelSpec(**row))
    expected = {(kind, seed) for kind in data["config"]["methods"]
                for seed in data["config"]["seeds"]}
    actual = {(s.kind, s.seed) for s in specs}
    if actual != expected or len(actual) != len(specs):
        raise ValueError("manifest does not contain every requested method/training seed")
    if any(s.training_shots != data["config"]["shots"] for s in specs):
        raise ValueError("training interaction budgets do not match")
    return specs


def evaluate_layouts(agent, policy_seeds, layouts=None, *, weights=None):
    """Same (layout, policy seed) for everyone; never changes learned parameters."""
    layouts = Belief.prior(MINI).layouts if layouts is None else layouts
    if weights is not None:
        if (len(weights) != len(layouts) or any(w < 0 or not math.isfinite(w) for w in weights)
                or not any(w > 0 for w in weights)):
            raise ValueError("weights must give nonnegative finite mass to the evaluation layouts")
    policy_seeds = tuple(policy_seeds)
    previous, agent.training = agent.training, False
    records = []
    try:
        for i, layout in enumerate(layouts):
            # An oracle with zero prior mass cannot explain observations from
            # this layout. Keep original indices/seeds for the supported boards.
            if weights is not None and weights[i] == 0:
                continue
            for seed in policy_seeds:
                # Avoid using an identical random stream on every layout.
                policy_seed = i * 1_000_003 + seed
                records.append((i, seed, play_solo(agent, MINI, layout=layout,
                                                   agent_seed=policy_seed)))
    finally:
        agent.training = previous
    return records


def weighted_stats(records, weights):
    counts = {}
    for i, _, _ in records:
        counts[i] = counts.get(i, 0) + 1
    if any(w > 0 and i not in counts for i, w in enumerate(weights)):
        raise ValueError("positive-weight layout has no evaluation games")
    if any(w < 0 or not math.isfinite(w) for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("layout weights must be a probability distribution")
    values = [(r.num_shots, weights[i] / counts[i], r) for i, _, r in records if weights[i] > 0]
    mean = sum(n * w for n, w, _ in values)
    variance = sum(w * (n - mean) ** 2 for n, w, _ in values)
    # Ratio of total decision time to total shots under this distribution.
    decision = sum(w * sum(r.move_times) for _, w, r in values) / mean
    return {"mean": mean, "std": math.sqrt(variance), "best": min(n for n, _, _ in values),
            "worst": max(n for n, _, _ in values), "decision_ms": decision * 1000}


def aggregate(rows):
    """Equal weight per training seed; separate game spread from seed spread."""
    mean = statistics.fmean(r["mean"] for r in rows)
    return {"mean": mean,
            "std": math.sqrt(statistics.fmean(r["std"] ** 2 + (r["mean"] - mean) ** 2 for r in rows)),
            "seed_std": statistics.pstdev(r["mean"] for r in rows) if len(rows) > 1 else None,
            "worst": max(r["worst"] for r in rows),
            "decision_ms": sum(r["decision_ms"] * r["mean"] for r in rows) / sum(r["mean"] for r in rows)}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def line_chart(series, path, title, xlabel="Training shots (environment transitions)"):
    """Small portable SVG, one line per method/seed, with no plotting dependency."""
    if not series:
        return
    width, height, left, right, top, bottom = 1100, 560, 65, 340, 45, 60
    points = [p for _, line in series for p in line]
    xmax = max(x for x, _ in points) or 1
    ymin = min(y for _, y in points) - .2
    ymax = max(y for _, y in points) + .2
    sx = lambda x: left + x / xmax * (width - left - right)
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="20" y="24" font-size="16">{escape(title)}</text>']
    for j in range(6):
        x, y = xmax * j / 5, ymin + (ymax - ymin) * j / 5
        out += [f'<text x="{sx(x):.1f}" y="{height-bottom+22}" text-anchor="middle">{x:,.0f}</text>',
                f'<line x1="{left}" x2="{width-right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#ddd"/>',
                f'<text x="{left-8}" y="{sy(y)+4:.1f}" text-anchor="end">{y:.2f}</text>']
    colors = ["#2459a6", "#bb4d00", "#287a38", "#8b3da0", "#987100", "#007d89"]
    for i, (label, line) in enumerate(series):
        color = colors[i % len(colors)]
        dash = ["", "6 3", "2 3"][i // len(colors) % 3]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in line)
        out += [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>',
                f'<text x="{width-right+15}" y="{top+18*i}" fill="{color}">{escape(label)}</text>']
    out += [f'<text x="{(left+width-right)/2}" y="{height-12}" text-anchor="middle">{escape(xlabel)}</text>',
            '<text transform="translate(18 290) rotate(-90)" text-anchor="middle">Mean shots to finish (lower is better)</text>', '</svg>']
    Path(path).write_text("\n".join(out), encoding="utf-8")


def bar_chart(rows, path):
    width, left, right, top, gap = 1000, 310, 90, 50, 30
    height = top + len(rows) * gap + 35
    xmax = max(r["mean"] for r in rows) * 1.1
    span = width - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="25" font-size="16">4x4 random placement: mean shots (lower is better)</text>']
    for i, r in enumerate(rows):
        y, length = top + i * gap, r["mean"] / xmax * span
        out += [f'<text x="{left-10}" y="{y+15}" text-anchor="end">{escape(r["agent"])}</text>',
                f'<rect x="{left}" y="{y}" width="{length:.1f}" height="21" fill="#2459a6"/>',
                f'<text x="{left+length+6:.1f}" y="{y+15}">{r["mean"]:.4f}</text>']
    out.append('</svg>')
    Path(path).write_text("\n".join(out), encoding="utf-8")
