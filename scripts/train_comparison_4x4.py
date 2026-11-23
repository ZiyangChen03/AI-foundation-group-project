"""Train all learning methods with identical environment-transition budgets.

    uv run --extra dqn scripts/train_comparison_4x4.py

Models are final checkpoints, never selected by their evaluation scores.
Training is sequential to avoid holding several large Q-tables in memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import MINI, BattleshipEnv
from battleship.agents import BeliefQAgent
from battleship.belief import Belief
from battleship.comparison import (ModelSpec, digest, evaluate_layouts, line_chart,
                                  provenance, weighted_stats, write_csv, write_json)
from battleship.planning import planner
from battleship.q_learning import QLearningAgent

LABELS = {"raw-q": "Raw-board Q-learning", "belief-q": "Belief-state Q-learning",
          "dqn-board": "DQN board", "dqn-density": "DQN density", "dqn-residual": "DQN residual"}


def train_tabular(kind, seed, shots, every, policy_samples, epsilon_end, log=print):
    """Count every shot, including the unfinished game at the budget boundary."""
    agent = QLearningAgent() if kind == "raw-q" else BeliefQAgent()
    rng = random.Random(seed)
    agent.training = True
    completed, elapsed, rows = 0, 0.0, []
    layouts = Belief.prior(MINI).layouts
    weights = [1 / len(layouts)] * len(layouts)
    plan = planner(MINI) if kind == "belief-q" else None
    env = BattleshipEnv(MINI) if kind == "raw-q" else None
    done = True

    for step in range(shots + 1):
        if step % every == 0 or step == shots:
            records = evaluate_layouts(agent, range(policy_samples))
            stats = weighted_stats(records, weights)
            rows.append({"training_shots": step, "training_games": completed,
                         "training_seconds": elapsed, "mean_shots": stats["mean"],
                         "game_std": stats["std"], "states": len(agent.q), "gradient_steps": 0})
            log(f"{kind} seed {seed}: {step:,} shots, {completed:,} complete games, "
                f"evaluation {stats['mean']:.3f}, training {elapsed:.1f}s", flush=True)
        if step == shots:
            break
        start = time.perf_counter()
        epsilon = 1 + (epsilon_end - 1) * step / max(1, shots - 1)
        if kind == "raw-q":
            if done:
                obs = env.reset(layout=rng.choice(layouts))
                agent.new_game(MINI, random.Random(rng.getrandbits(64)))
            # Evaluation resets agent.rng; keep a dedicated exploration RNG.
            agent.rng = rng
            agent.epsilon = epsilon
            action = agent.select_shot(obs)
            result = env.step(action)
            agent.observe(obs, action, result)
            obs, done = result.observation, result.done
        else:
            if done:
                hidden = rng.randrange(plan.everything.bit_count())
                possible, fired = plan.everything, 0
            key = plan._key(possible, fired)
            legal = list(plan.hit_counts(possible, fired))
            action = rng.choice(legal) if rng.random() < epsilon else rng.choice(agent.best_actions(key, legal))
            possible = next(g for g in plan.outcomes(possible, fired, action) if g >> hidden & 1)
            fired |= 1 << action
            done = plan._done(possible, fired)
            target = -1.0 if done else -1.0 + agent.best_value(
                plan._key(possible, fired), list(plan.hit_counts(possible, fired)))
            agent.update(key, action, target)
        completed += int(done)
        elapsed += time.perf_counter() - start
    agent.training = False
    return agent, rows


def train_dqn(kind, seed, args, folder):
    import torch
    from battleship.dqn import model as qmodel
    from battleship.dqn.model import NetConfig
    from battleship.dqn.train import TrainConfig, default_device, train

    torch.set_num_threads(args.threads)
    options = {"board": {}, "density": {"use_density": True},
               "residual": {"use_density": True, "residual": 1.0}}[kind.removeprefix("dqn-")]
    net = NetConfig(**options)
    tc = TrainConfig(seed=seed, device=args.device or default_device(), parallel=args.parallel,
                     eps_end=args.epsilon_end, eps_fraction=1.0)
    write_json(folder / "dqn_config.json", {"net": asdict(net), "train": asdict(tc),
                                          "torch_version": torch.__version__, "threads": args.threads,
                                          "shot_budget": args.shots, "eval_shots": args.every})
    model, raw_rows = train(MINI, net, tc, list(Belief.prior(MINI).layouts),
                            shot_budget=args.shots, eval_shots=args.every)
    qmodel.save(folder / "final.pt", model, MINI,
                {"name": LABELS[kind], "seed": seed, "training_shots": args.shots})
    rows = [{"training_shots": r["training_shots"], "training_games": r["games"],
             "training_seconds": r["training_seconds"], "mean_shots": r["eval_mean"],
             "game_std": r["eval_sem"] * len(Belief.prior(MINI).layouts) ** .5,
             "states": None, "gradient_steps": r["grad_steps"]} for r in raw_rows]
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", choices=LABELS, default=list(LABELS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--shots", type=int, default=1_000_000, help="exact new transitions per method/seed")
    p.add_argument("--every", type=int, default=100_000, help="curve interval in training shots")
    p.add_argument("--policy-samples", type=int, default=3, help="random policy seeds per curve layout")
    p.add_argument("--epsilon-end", type=float, default=.05)
    p.add_argument("--parallel", type=int, default=256, help="DQN vector environments")
    p.add_argument("--device", default="", help="DQN: cpu, cuda, mps; default auto")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads, including evaluation")
    p.add_argument("--output", type=Path, default=Path("results/comparison_training"))
    p.add_argument("--resume", action="store_true", help="skip completed method/seed runs with identical settings/source")
    args = p.parse_args(argv)
    if min(args.shots, args.every, args.policy_samples, args.parallel, args.threads) < 1:
        p.error("budgets, intervals, samples, parallel and threads must be positive")
    if args.parallel > 200_000 or not 0 <= args.epsilon_end <= 1:
        p.error("parallel must fit the replay buffer; epsilon-end must be in [0, 1]")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        p.error("methods and seeds must be distinct")
    if any(m.startswith("dqn-") for m in args.methods):
        import torch  # Fail before any training without the optional dependency.
    config = {k: v for k, v in vars(args).items() if k not in ("output", "resume")}
    source = provenance()
    out = args.output
    manifest_path = out / "manifest.json"
    if out.exists() and any(out.iterdir()):
        if not args.resume or not manifest_path.exists():
            p.error("output is not empty; choose a new directory or use --resume")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config"] != config or old["provenance"]["sources"] != source["sources"]:
            p.error("resume requires identical settings and Python source hashes")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "config": config, "provenance": source, "models": []}
    write_json(manifest_path, manifest)
    series = []
    for kind in args.methods:
        for seed in args.seeds:
            folder = out / kind / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            completed_path = folder / "complete.json"
            if args.resume and completed_path.exists():
                spec = json.loads(completed_path.read_text(encoding="utf-8"))
                if digest(out / spec["checkpoint"]) != spec["sha256"]:
                    p.error(f"checkpoint hash mismatch: {folder}")
                import csv
                with (out / spec["curve"]).open(encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
            else:
                wall_start = time.perf_counter()
                if kind.startswith("dqn-"):
                    rows = train_dqn(kind, seed, args, folder)
                    checkpoint = folder / "final.pt"
                else:
                    agent, rows = train_tabular(kind, seed, args.shots, args.every,
                                                args.policy_samples, args.epsilon_end)
                    checkpoint = folder / "final.json.gz"
                    agent.save(checkpoint)
                    del agent
                curve = folder / "curve.csv"
                write_csv(curve, rows)
                final = rows[-1]
                spec = asdict(ModelSpec(LABELS[kind], kind, seed,
                              checkpoint.relative_to(out).as_posix(),
                              training_shots=final["training_shots"],
                              training_games=final["training_games"],
                              training_seconds=final["training_seconds"],
                              curve=curve.relative_to(out).as_posix()))
                spec["sha256"] = digest(checkpoint)
                write_json(folder / "runtime.json", {"wall_seconds_including_evaluation_and_save":
                                                     time.perf_counter() - wall_start})
                write_json(completed_path, spec)
            manifest["models"].append(spec)
            write_json(manifest_path, manifest)
            series.append((f"{LABELS[kind]} seed {seed}",
                           [(int(r["training_shots"]), float(r["mean_shots"])) for r in rows]))
            gc.collect()
    line_chart(series, out / "learning_curves.svg", "4x4 learning curves: matched training-shot budgets")
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Training complete. Compare with: uv run --extra dqn scripts/compare_4x4.py --trained {out}")


if __name__ == "__main__":
    main()
