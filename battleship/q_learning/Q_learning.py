"""4x4 Q-learning: agent, training, evaluation, progress and reports.

Run from the project root:
    python -m battleship.q_learning train --episodes 20000 --seeds 0 1 2
    python -m battleship.q_learning test --output battleship/q_learning/results/experiment_02
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from battleship import MINI, all_layouts, evaluate, layout_from_cells, play_solo
from battleship.agents import RandomAgent, HuntTargetAgent, ProbabilityDensityAgent
from battleship.agents.base import Agent

# ======================================================================
# Q-learning agent
# ======================================================================

class QLearningAgent(Agent):
    name = "q-learning"

    def __init__(self, alpha=None, gamma=1.0, epsilon=0.1):
        """``alpha=None`` (default) uses the step size 1/N(s, a), N being the
        number of updates of that state-action pair, so each Q-value is the
        running average of its targets. A number in (0, 1] is a fixed step."""
        super().__init__()
        if (alpha is not None and not 0 < alpha <= 1) or not 0 <= gamma <= 1 \
                or not 0 <= epsilon <= 1:
            raise ValueError("require alpha None or 0 < alpha <= 1, and 0 <= gamma, epsilon <= 1")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = {}
        self.counts = {}   # state -> {action: number of updates}
        self.updates = 0

    def new_game(self, config, rng):
        if config != MINI:
            raise ValueError("QLearningAgent supports MINI (4x4, fleet 3/2, reveal on) only")
        super().new_game(config, rng)

    @staticmethod
    def state(obs):
        return "".join(cell.value for cell in obs.board) + ":" + ",".join(
            map(str, obs.remaining_ship_lengths))

    @staticmethod
    def default(obs):
        """Value of an action never tried: a lower bound. Every reward is -1 and
        a game never needs more shots than there are untried cells, so a 0.0
        default would rank untried actions above every tried one."""
        return -float(len(obs.legal_actions))

    def select_shot(self, obs):
        legal = obs.legal_actions
        if not legal or not obs.remaining_ship_lengths:
            raise ValueError("no legal shots in a finished game")
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        values = self.q.get(self.state(obs), {})
        default = self.default(obs)
        # Ties go to the lowest-numbered cell. Most greedy choices are ties (an
        # unseen next board makes an action's value equal the default), and a
        # random pick would send every game to different boards, so few boards
        # would ever be seen twice. A fixed choice keeps greedy play on the same
        # boards, which then accumulate updates; epsilon still explores.
        return max(legal, key=lambda cell: values.get(cell, default))

    def observe(self, obs, action, result):
        if not self.training:
            return
        target = result.reward
        if not result.done:
            following = self.q.get(self.state(result.observation), {})
            default = self.default(result.observation)
            target += self.gamma * max(following.get(cell, default)
                                       for cell in result.observation.legal_actions)
        state = self.state(obs)
        values = self.q.setdefault(state, {})
        counts = self.counts.setdefault(state, {})
        counts[action] = counts.get(action, 0) + 1
        step = 1 / counts[action] if self.alpha is None else self.alpha
        old = values.get(action, self.default(obs))
        values[action] = old + step * (target - old)
        self.updates += 1

    def save(self, path):
        """Versioned compressed JSON; does not serialize executable objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"version": 2, "config": "MINI", "alpha": self.alpha,
                       "gamma": self.gamma, "epsilon": self.epsilon,
                       "updates": self.updates, "q": self.q, "counts": self.counts}, stream)

    @classmethod
    def load(cls, path):
        """Reads version 2, and version 1 files (fixed alpha, no update counts)."""
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("version") not in (1, 2) or data.get("config") != "MINI":
            raise ValueError("unsupported Q-table format or board")
        agent = cls(data["alpha"], data["gamma"], data["epsilon"])
        agent.q = {state: {int(action): value for action, value in values.items()}
                   for state, values in data["q"].items()}
        agent.counts = {state: {int(action): n for action, n in values.items()}
                        for state, values in data.get("counts", {}).items()}
        agent.updates = data["updates"]
        return agent


# ======================================================================
# Training progress
# ======================================================================

class Progress:
    def __init__(self, total, label):
        self.total, self.label = total, label
        self.started = time.perf_counter()
        self.last = 0.

    def update(self, completed):
        now = time.perf_counter()
        if completed < self.total and now - self.last < .2:
            return
        self.last = now
        elapsed = now - self.started
        rate = completed / elapsed if elapsed else 0.
        remaining = (self.total - completed) / rate if rate else 0.
        fraction = completed / self.total
        filled = int(25 * fraction)
        bar = "#" * filled + "-" * (25 - filled)
        text = (f"{self.label}: {fraction:6.1%} |{bar}| {completed}/{self.total} "
                f"[{elapsed:.1f}s, ETA {remaining:.1f}s, {rate:.1f} episodes/s]")
        print("\r" + text.ljust(125), end="\n" if completed == self.total else "",
              file=sys.stderr, flush=True)


# ======================================================================
# Result export and reward curve
# ======================================================================

def provenance():
    root = Path(__file__).resolve().parents[2]
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "commit": None, "dirty": None}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    result["source_sha256"] = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "battleship").rglob("*.py"))}
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chart(path, title, series):
    """series = [(label, [(x, y), ...])]; keep every seed visible."""
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2"]
    legend_rows = len(series)
    height = 420 + legend_rows * 22
    x_max = max(1., max(x for _, points in series for x, _ in points))
    values = [y for _, points in series for _, y in points]
    low, high = min(values), max(values)
    padding = max(.1, (high - low) * .1)
    y_min, y_max = low - padding, high + padding
    def x_pos(x):
        return 80 + 740 * x / x_max
    def y_pos(y):
        return 340 - 270 * (y - y_min) / (y_max - y_min)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" font-size="13" fill="#172033">',
           f'<text x="80" y="30" font-size="20">{escape(title)}</text>',
           '<text x="80" y="55">Running average reward (higher is better)</text>']
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = y_pos(y)
        svg += [f'<line x1="80" y1="{py}" x2="820" y2="{py}" stroke="#e2e8f0"/>',
                f'<text x="65" y="{py + 4}" text-anchor="end">{y:.1f}</text>']
    for i in range(6):
        x = x_max * i / 5
        svg.append(f'<text x="{x_pos(x)}" y="360" text-anchor="middle">{x:.0f}</text>')
    svg.append('<text x="450" y="385" text-anchor="middle">Training episode</text>')
    for i, (label, points) in enumerate(series):
        color = colors[i % len(colors)]
        # Keep the exported SVG compact; CSV retains every episode.
        stride = max(1, (len(points) + 1999) // 2000)
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        coords = " ".join(f"{x_pos(x)},{y_pos(y)}" for x, y in sampled)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="80" y="{415 + i * 22}" fill="{color}">{i + 1}. {escape(label)}</text>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def evaluation_summary(results):
    lines = []
    for row in results:
        if row["seed"] is None:
            continue
        lines += [f"Q-learning seed {row['seed']} (selected episode: {row['selected_episode']})",
                  f"Total user CPU time for {row['games']} evaluation episodes (seconds): {row['user_cpu_seconds']:.4f}",
                  f"Number of unique states: {row['unique_states']}",
                  f"Average reward over {row['games']} evaluation episodes: {row['average_reward']:.4f}",
                  f"Average episode length: {row['average_episode_length']:.4f}",
                  f"Number of unique unseen states: {row['unique_unseen_states']}",
                  f"Actions using Q-table: {row['q_table_percentage']:.2f}%",
                  f"Random actions due to unseen states: {row['random_percentage']:.2f}%", ""]
    return "\n".join(lines)


def export_report(folder, config, curves, results, games, rewards):
    folder.mkdir(parents=True, exist_ok=True)
    q_results = [row for row in results if row["seed"] is not None]
    means = [row["mean"] for row in q_results]
    aggregate = {"training_seeds": len(means), "mean_of_test_means": statistics.fmean(means),
                 "std_across_seed_means": statistics.pstdev(means) if len(means) > 1 else None,
                 "std_definition": "population std of per-training-seed test means; null for one seed"}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (folder / "test_results.json").write_text(
        json.dumps({"per_run": results, "q_learning_across_seeds": aggregate}, indent=2), encoding="utf-8")
    write_csv(folder / "comparison.csv", [{k: v for k, v in row.items() if k != "histogram"}
                                           for row in results])
    write_csv(folder / "test_games.csv", games)
    for seed, rows in curves.items():
        write_csv(folder / f"seed_{seed}_curve.csv", rows)
    series = []
    for seed, rows in rewards.items():
        total_reward = 0.
        points, exported = [], []
        for i, row in enumerate(rows, 1):
            total_reward += row["reward"]
            average = total_reward / i
            points.append((i, average))
            exported.append(dict(episode=i, reward=row["reward"], running_average_reward=average))
        write_csv(folder / f"seed_{seed}_rewards.csv", exported)
        series.append((f"seed {seed}", points))
    chart(folder / "learning_curve.svg", f"Training: {config['episodes']} episodes per seed", series)
    baseline = min((row for row in results if row["seed"] is None), key=lambda row: row["mean"])
    difference = aggregate["mean_of_test_means"] - baseline["mean"]
    spread = aggregate["std_across_seed_means"]
    terminal_summary = evaluation_summary(results)
    (folder / "summary.txt").write_text(terminal_summary, encoding="utf-8")
    if config.get("protocol", "held-out") == "all":
        protocol_lines = [
            "- Layouts: training draws uniformly from all 264 layouts (the game's distribution); "
            "testing plays every one of the 264 once, as for every other agent. This is not a "
            "generalization test: a Q-table cannot play layouts it never saw.",
            "- Model: the final one; no checkpoint is selected (the curve's evaluations are only recorded).",
            "- Test: one game per layout; policy RNG seed = layout index, shared across agents."]
        statistics_note = "Statistics describe the final model, while curves cover the entire training run."
    else:
        protocol_lines = [
            "- Layout counts: " + ", ".join(f"{name}={len(layouts)}" for name, layouts in config["splits"].items()) + ".",
            "- Checkpoint: lowest validation mean; ties keep the earliest checkpoint, including episode 0.",
            "- Test: one game per held-out layout; policy RNG seed = layout index, shared across agents."]
        statistics_note = "Statistics describe the validation-selected checkpoint, while curves cover the entire training run."
    lines = ["# 4x4 Q-learning experiment", "", "![Running average training reward](learning_curve.svg)", "",
             "```text", terminal_summary.rstrip(), "```", "",
             "The curve is the cumulative mean of actual training episode rewards (reward = -shots), not validation performance. Each seed has its own line.",
             "Q-table usage counts lookups of known states, including tied or partially populated action rows; it does not guarantee an informed action. In an unseen state every action ties at the default value, so the lowest-numbered legal cell is fired. No evaluation exploration is enabled.",
             f"{statistics_note} CPU time measures evaluation user CPU time only.",
             "", "## Setup", "",
             "- Board: 4x4; fleet: 3 and 2; sunk cells revealed; reward: -1 per shot.",
             f"- Training episodes per seed: {config['episodes']}; evaluated seeds: {config['evaluated_seeds']}.",
             f"- Alpha: {config['alpha'] if config['alpha'] is not None else '1/N(s, a)'}; gamma: {config['gamma']}; epsilon: {config['epsilon_start']} -> {config['epsilon_end']} (linear).",
             *protocol_lines,
             "- Exploration and learning disabled during testing.",
             f"- Training commit: {config.get('provenance', {}).get('commit', 'unknown')}; dirty: {config.get('provenance', {}).get('dirty', 'unknown')}.",
             f"- Evaluation commit: {config['evaluation_provenance']['commit']}; dirty: {config['evaluation_provenance']['dirty']}.",
             "- Exact layouts, runtime/source hashes and parameters: [config.json](config.json). Model hashes: [test_results.json](test_results.json).",
             "", "## Test results", "",
             "| Agent | Seed | Games | Mean shots | Game std | Median | Worst | Training s | Decision ms | Selected episode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        training = f"{row['training_seconds']:.2f}" if "training_seconds" in row else "N/A"
        lines.append(f"| {row['agent']} | {row['seed'] if row['seed'] is not None else '-'} | {row['games']} | {row['mean']:.3f} | {row['std']:.3f} | {row['median']:.1f} | {row['worst']} | {training} | {row['mean_move_time_seconds'] * 1000:.4f} | {row.get('selected_episode', '-')} |")
    lines += ["", "## Across training seeds", "",
              f"Mean of Q-learning test means: **{aggregate['mean_of_test_means']:.3f}** shots.",
              (f"Population standard deviation across those means: **{spread:.3f}** shots."
               if spread is not None else "One training seed only: variability across training seeds cannot be estimated."),
              "This is distinct from Game std above, which measures variation between layouts within one run.",
              "", "## Observed comparison and limitations", "",
              f"Q-learning's mean across training seeds minus the best listed baseline ({baseline['agent']}): **{difference:+.3f}** shots. Negative means fewer shots; positive means more.",
              "This is a descriptive comparison on these test layouts, not a significance claim. Do not tune hyperparameters using these test scores.",
              "Training time includes training games only; validation and serialization are excluded. Decision time is measured on the evaluation machine.",
              "Full-board Q-tables may have sparse coverage; in an unseen state the lowest-numbered legal cell is fired."
              + (" Episode-0 checkpoints can be selected if training did not improve validation performance."
                 if config.get("protocol", "held-out") != "all" else ""),
              "", "The single SVG figure is viewable on GitHub. Plot rendering is sampled to about 2,000 points per seed; reward CSVs retain every episode. No second figure is generated.",
              "", "## Reproduction", "",
              "Use the recorded source revision and parameters in config.json. Run training, then the separate evaluation command documented in ../../README.md. Models remain in the training artifacts directory; publish them separately if needed, matching the SHA-256 hashes recorded here."]
    summary = "\n".join(lines) + "\n"
    (folder / "README.md").write_text(summary, encoding="utf-8")
    return terminal_summary


# ======================================================================
# Training
# ======================================================================

def split_layouts(seed=42):
    layouts = list(all_layouts(MINI))
    random.Random(seed).shuffle(layouts)
    train_end, validation_end = int(len(layouts) * .7), int(len(layouts) * .85)
    return layouts[:train_end], layouts[train_end:validation_end], layouts[validation_end:]


def train_main(argv=None):
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=None,
                        help="fixed step size in (0, 1]; default: 1/N(s, a), a running average")
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--epsilon-start", type=float, default=1.)
    parser.add_argument("--epsilon-end", type=float, default=.05)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--protocol", choices=("all", "held-out"), default="all",
                        help="all (default): train on all 264 layouts and report the final model "
                             "on all 264, like every other agent; held-out: train on a 70%% split, "
                             "select the best checkpoint on 15%%, test on the unseen 15%%")
    parser.add_argument("--split-seed", type=int, default=42, help="held-out protocol only")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.eval_every < 1:
        parser.error("episodes and eval-every must be positive")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("require 0 <= epsilon-end <= epsilon-start <= 1")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    # Validate before writing any output.
    QLearningAgent(args.alpha, args.gamma, args.epsilon_start)
    held_out = args.protocol == "held-out"
    if held_out:
        training, validation, testing = split_layouts(args.split_seed)
    else:
        # A Q-table cannot generalize to layouts it never sees, and the game draws
        # uniformly from all 264, so train on all of them. The curve is still
        # recorded on all 264, but no checkpoint is selected from it.
        training = validation = list(all_layouts(MINI))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {**vars(args), "output": str(args.output), "board": "MINI",
                "provenance": provenance()}
    if held_out:
        manifest["splits"] = {name: [[sorted(ship.cells) for ship in layout] for layout in layouts]
                              for name, layouts in [("train", training), ("validation", validation),
                                                    ("test", testing)]}
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for seed in args.seeds:
        agent = QLearningAgent(args.alpha, args.gamma, args.epsilon_start)
        rng = random.Random(seed)
        elapsed, shots = 0., 0
        best = float("inf")
        checkpoint = args.output / f"seed_{seed}.json.gz"
        progress = Progress(args.episodes, f"Training seed {seed}")
        with (args.output / f"seed_{seed}_curve.csv").open("w", newline="", encoding="utf-8") as stream, \
                (args.output / f"seed_{seed}_rewards.csv").open("w", newline="", encoding="utf-8") as reward_stream:
            reward_writer = csv.DictWriter(reward_stream, fieldnames=["episode", "reward", "running_average_reward"])
            reward_writer.writeheader()
            writer = csv.DictWriter(stream, fieldnames=["episode", "epsilon", "training_seconds",
                                    "training_shots", "states", "validation_mean", "validation_std"])
            writer.writeheader()
            for episode in range(args.episodes + 1):
                if episode:
                    fraction = min(1., (episode - 1) / max(1, args.episodes - 1))
                    agent.epsilon = args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)
                    agent.training = True
                    start = time.perf_counter()
                    record = play_solo(agent, MINI, layout=rng.choice(training),
                                       agent_seed=rng.getrandbits(64))
                    elapsed += time.perf_counter() - start
                    shots += record.num_shots
                    reward_writer.writerow(dict(episode=episode, reward=-record.num_shots,
                                                running_average_reward=-shots / episode))
                if episode % args.eval_every == 0 or episode == args.episodes:
                    report = evaluate(agent, MINI, layouts=validation)
                    writer.writerow(dict(episode=episode, epsilon=agent.epsilon,
                                         training_seconds=elapsed, training_shots=shots,
                                         states=len(agent.q), validation_mean=report.mean,
                                         validation_std=report.std))
                    stream.flush()
                    if held_out and report.mean < best:
                        best = report.mean
                        agent.save(checkpoint)
                if episode:
                    progress.update(episode)
        if not held_out:
            agent.save(checkpoint)             # the reported model is the final one
        agent.save(args.output / f"seed_{seed}_final.json.gz")


# ======================================================================
# Evaluation
# ======================================================================

class MeasuredAgent(QLearningAgent):
    """Observe lookup coverage without changing the learned policy or its RNG."""

    def __init__(self, source):
        super().__init__(source.alpha, source.gamma, source.epsilon)
        self.q = source.q
        self.counts = source.counts
        self.updates = source.updates
        self.unseen_states = set()
        self.q_table_actions = 0
        self.random_actions = 0

    def select_shot(self, obs):
        state = self.state(obs)
        if state in self.q:
            self.q_table_actions += 1
        else:
            self.unseen_states.add(state)
            self.random_actions += 1
        return super().select_shot(obs)


def load_experiment(folder, seeds=None):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config["board"] != "MINI":
        raise ValueError("only MINI experiments are supported")
    seeds = config["seeds"] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["seeds"]):
        raise ValueError("select distinct training seeds present in config.json")
    if config.get("protocol", "held-out") == "all":   # runs from before --protocol are held-out
        test = list(all_layouts(MINI))
    else:
        splits = {name: [layout_from_cells(MINI, cells) for cells in config["splits"][name]]
                  for name in ("train", "validation", "test")}
        keys = {name: {frozenset(ship.cells for ship in layout) for layout in layouts}
                for name, layouts in splits.items()}
        if any(not layouts or len(keys[name]) != len(layouts) for name, layouts in splits.items()):
            raise ValueError("each split must be nonempty with no duplicate layouts")
        if keys["train"] & keys["test"] or keys["validation"] & keys["test"] or keys["train"] & keys["validation"]:
            raise ValueError("training, validation and test layouts must be disjoint")
        test = splits["test"]
    curves, models = {}, {}
    for seed in seeds:
        models[seed] = QLearningAgent.load(folder / f"seed_{seed}.json.gz")
        with (folder / f"seed_{seed}_curve.csv").open(encoding="utf-8", newline="") as stream:
            curves[seed] = [{key: float(value) for key, value in row.items()}
                            for row in csv.DictReader(stream)]
        if not curves[seed] or curves[seed][-1]["episode"] != config["episodes"]:
            raise ValueError(f"training run for seed {seed} is incomplete")
    return config, test, curves, models


def test_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluation of trained Q-tables")
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path(__file__).parent / "artifacts",
                        help="training folder or one best checkpoint inside that folder")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: all training seeds")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "experiment_01")
    args = parser.parse_args(argv)
    folder, seeds = args.source, args.seeds
    if folder.is_file():
        if seeds is not None:
            parser.error("use either a checkpoint path or --seeds, not both")
        try:
            seeds = [int(folder.name.removeprefix("seed_").removesuffix(".json.gz"))]
        except ValueError:
            parser.error("checkpoint must be named seed_N.json.gz")
        folder = folder.parent
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output folder is not empty; choose a new experiment folder")
    try:
        config, layouts, curves, models = load_experiment(folder, seeds)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    rewards = {}
    for seed in models:
        path = folder / f"seed_{seed}_rewards.csv"
        if not path.exists():
            parser.error(f"{path.name} is missing: retrain into a new folder to record per-episode rewards")
        with path.open(encoding="utf-8", newline="") as stream:
            rewards[seed] = [{key: float(value) for key, value in row.items()}
                             for row in csv.DictReader(stream)]
        if len(rewards[seed]) != config["episodes"] or any(
                row["episode"] != i for i, row in enumerate(rewards[seed], 1)):
            parser.error(f"reward history for seed {seed} is incomplete")
    results, games = [], []
    agents = [(seed, MeasuredAgent(agent)) for seed, agent in models.items()] + [(None, agent) for agent in
               [RandomAgent(), HuntTargetAgent(False), HuntTargetAgent(True), ProbabilityDensityAgent()]]
    for seed, agent in agents:
        start_user_time = os.times().user
        report = evaluate(agent, MINI, layouts=layouts)
        user_cpu_seconds = os.times().user - start_user_time
        row = dict(agent=agent.name, seed=seed, games=report.games, mean=report.mean,
                   std=report.std, median=report.median, best=report.best, worst=report.worst,
                   mean_move_time_seconds=report.mean_move_time, histogram=report.histogram)
        if seed is not None:
            total_actions = agent.q_table_actions + agent.random_actions
            row.update(unique_states=len(agent.q), average_reward=-report.mean,
                       average_episode_length=report.mean,
                       unique_unseen_states=len(agent.unseen_states),
                       q_table_actions=agent.q_table_actions, random_actions=agent.random_actions,
                       q_table_percentage=100 * agent.q_table_actions / total_actions,
                       random_percentage=100 * agent.random_actions / total_actions,
                       user_cpu_seconds=user_cpu_seconds)
            curve = curves[seed]
            if config.get("protocol", "held-out") == "all":
                selected = curve[-1]               # final model, no selection
            else:
                selected = min(curve, key=lambda point: point["validation_mean"])
            row.update(training_seconds=curve[-1]["training_seconds"],
                       training_shots=int(curve[-1]["training_shots"]),
                       selected_episode=int(selected["episode"]),
                       selected_validation_mean=selected["validation_mean"],
                       model_sha256=hashlib.sha256((folder / f"seed_{seed}.json.gz").read_bytes()).hexdigest())
        results.append(row)
        games.extend(dict(agent=agent.name, seed=seed, layout_index=i, policy_seed=i,
                          shots=record.num_shots, decision_seconds=sum(record.move_times))
                     for i, record in enumerate(report.records))
    config = {**config, "evaluated_seeds": list(models), "evaluation_provenance": provenance(),
              "evaluation_protocol": ("One game per layout, all 264" if config.get("protocol", "held-out") == "all"
                                      else "One game per held-out layout") + "; policy seed = layout index"}
    summary = export_report(args.output, config, curves, results, games, rewards)
    print(summary)
    print(f"\nResults saved to: {args.output.resolve()}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test"), help="train models or evaluate and export results")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    mode = parser.parse_args(argv[:1]).mode
    if mode == "train":
        train_main(argv[1:])
    else:
        test_main(argv[1:])


if __name__ == "__main__":
    main()
