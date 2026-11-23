"""Compare 4x4 agents on identical layouts and policy seeds, with frozen models.

    uv run --extra dqn scripts/compare_4x4.py --trained results/comparison_training --require-complete
    uv run scripts/compare_4x4.py --baselines-only --quick

Outputs are new experiment results, never the historical numbers from PR #22.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import MINI, play_solo
from battleship.agents import OpponentBeliefAgent
from battleship.belief import Belief
from battleship.comparison import (ModelSpec, aggregate, bar_chart, baselines, digest, evaluate_layouts,
                                  line_chart, load_specs, provenance, table,
                                  weighted_stats, write_csv, write_json)
from battleship.config import label
from battleship.opponents import standard_habits
from battleship.tournament import round_robin

LEARNERS = {"raw-q": "Raw-board Q-learning", "belief-q": "Belief-state Q-learning",
            "dqn-board": "DQN board", "dqn-density": "DQN density", "dqn-residual": "DQN residual"}


def external_specs(kind, values):
    specs = []
    for index, value in enumerate(values or []):
        seed, path = value.split("=", 1) if "=" in value else (index, value)
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        specs.append(ModelSpec(LEARNERS[kind], kind, int(seed), str(path)))
    return specs


def paired_specs(a, b):
    """Pair equal training seeds; broadcast the seedless baseline."""
    if len(a) == 1 and a[0].seed is None:
        return [(a[0], y) for y in b]
    if len(b) == 1 and b[0].seed is None:
        return [(x, b[0]) for x in a]
    left, right = {x.seed: x for x in a}, {x.seed: x for x in b}
    if left.keys() != right.keys():
        raise ValueError("learning methods must supply the same training seed IDs for head-to-head")
    return [(left[s], right[s]) for s in sorted(left)]


def case_record(spec, category, item):
    index, policy_seed, record = item
    first_hit = next(j for j, shot in enumerate(record.shots, 1) if shot.result.value != "miss")
    return {"agent": spec.label, "training_seed": spec.seed, "category": category,
            "layout_index": index, "policy_seed": policy_seed,
            "effective_policy_seed": index * 1_000_003 + policy_seed,
            "layout": [sorted(ship.cells) for ship in record.layout],
            "shots": record.num_shots, "misses": record.num_shots - sum(MINI.fleet),
            "first_hit_shot": first_hit, "misses_before_first_hit": first_hit - 1,
            "misses_after_first_hit": record.num_shots - sum(MINI.fleet) - first_hit + 1,
            "sunk_at_shots": [j for j, shot in enumerate(record.shots, 1) if shot.result.value == "sunk"],
            "history": [{"cell": s.cell, "label": label(s.cell, MINI),
                         "result": s.result.value, "sunk_length": s.sunk_length} for s in record.shots]}


def reuse_solo(folder, specs, samples):
    """Validate complete saved solo games and replay only the example traces."""
    folder = Path(folder).resolve()
    old = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if old["config"]["samples"] != samples or old["models"] != [asdict(s) for s in specs]:
        raise ValueError("saved solo models/settings differ from this comparison")
    if old["model_hashes"] != {s.checkpoint: digest(s.checkpoint) for s in specs if s.checkpoint}:
        raise ValueError("saved solo checkpoint hashes differ")
    current_sources = provenance()["sources"]
    allowed_fixes = {"battleship/comparison.py", "scripts/compare_4x4.py"}
    for path in set(old["provenance"]["sources"]) | set(current_sources):
        if path not in allowed_fixes and old["provenance"]["sources"].get(path) != current_sources.get(path):
            raise ValueError(f"solo evaluation source changed: {path}")
    def read(name):
        with (folder / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    uniform = read("uniform.csv")
    integers = {"training_seed", "games", "best", "worst", "training_shots", "training_games", "checkpoint_bytes"}
    for row in uniform:
        for key, value in row.items():
            if key != "agent":
                row[key] = None if value == "" else int(value) if key in integers else float(value)
    expected = {(s.label, s.seed): s for s in specs}
    if {(r["agent"], r["training_seed"]) for r in uniform} != expected.keys() or len(uniform) != len(specs):
        raise ValueError("saved solo summaries are incomplete")
    habit_rows = []
    for row in read("habits.csv"):
        key = row["agent"], int(row["training_seed"]) if row["training_seed"] else None
        if key not in expected:  # Recompute the oracle using the fixed support.
            continue
        habit_rows.append({k: v if k == "agent" else (key[1] if k == "training_seed" else float(v))
                           for k, v in row.items()})
    if {(r["agent"], r["training_seed"]) for r in habit_rows} != expected.keys() or len(habit_rows) != len(specs):
        raise ValueError("saved solo habit rows are incomplete")
    seen, sums, extremes = defaultdict(set), defaultdict(int), {}
    with (folder / "games.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = row["agent"], int(row["training_seed"]) if row["training_seed"] else None
            point = int(row["layout_index"]), int(row["policy_seed"])
            shots = int(row["shots"])
            if key not in expected or point in seen[key] or not 5 <= shots <= 16:
                raise ValueError("saved solo games contain unexpected/duplicate entries")
            seen[key].add(point)
            sums[key] += shots
            item = (*point, shots)
            if key not in extremes:
                extremes[key] = [item, item]
            if shots < extremes[key][0][2]:
                extremes[key][0] = item
            if shots > extremes[key][1][2]:
                extremes[key][1] = item
    layouts = Belief.prior(MINI).layouts
    required = {(i, seed) for i in range(len(layouts)) for seed in range(samples)}
    for row in uniform:
        key = row["agent"], row["training_seed"]
        if (seen[key] != required or row["games"] != len(required)
                or not math.isclose(sums[key] / len(required), row["mean"], abs_tol=1e-9)):
            raise ValueError(f"saved solo game coverage/statistics are incomplete: {key}")
    cases = []
    for key, spec in expected.items():
        agent = spec.load()
        for category, (i, seed, shots) in zip(("fewest shots", "most shots"), extremes[key]):
            record = play_solo(agent, MINI, layout=layouts[i], agent_seed=i * 1_000_003 + seed)
            if record.num_shots != shots:
                raise ValueError(f"saved example cannot be reproduced: {key}")
            cases.append(case_record(spec, category, (i, seed, record)))
        del agent
        gc.collect()
    return uniform, habit_rows, cases, old


def run_tournament(groups, games, seed, out):
    counts, wins, details, first_wins, total = {}, {}, [], 0, 0
    for (left, a), (right, b) in combinations(groups.items(), 2):
        pair_wins, pair_games = 0, 0
        for k, (sa, sb) in enumerate(paired_specs(a, b)):
            aa, bb = sa.load(), sb.load()
            report = round_robin([aa, bb], MINI, games_per_pair=games, seed=seed + k)
            pair_wins += report.wins[0][1]
            pair_games += report.total_games
            first_wins += report.first_player_wins
            total += report.total_games
            details.append({"left": left, "right": right, "left_training_seed": sa.seed,
                            "right_training_seed": sb.seed, "games": report.total_games,
                            "left_wins": report.wins[0][1], "first_player_wins": report.first_player_wins})
            del aa, bb
            gc.collect()
        counts[left, right] = counts[right, left] = pair_games
        wins[left, right], wins[right, left] = pair_wins, pair_games - pair_wins
        write_csv(out / "tournament_pairs.csv", details)
        print(f"tournament: {left} vs {right}, {pair_games} games", flush=True)
    labels = list(groups)
    rows = []
    for name in labels:
        n = sum(counts[name, other] for other in labels if other != name)
        w = sum(wins[name, other] for other in labels if other != name)
        # Equal weight per opposing algorithm, even when learned opponents have more seeds.
        macro = statistics.fmean(wins[name, other] / counts[name, other] for other in labels if other != name)
        rows.append({"agent": name, **{other: None if other == name else wins[name, other] / counts[name, other]
                                      for other in labels}, "overall_macro": macro,
                     "overall_game_weighted": w / n})
    write_csv(out / "tournament.csv", rows)
    return rows, first_wins, total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trained", type=Path, help="directory produced by train_comparison_4x4.py")
    p.add_argument("--reuse-solo", type=Path, help="reuse validated completed solo evaluation from an interrupted comparison; use a new output directory")
    p.add_argument("--baselines-only", action="store_true", help="explicitly run without learned models")
    for kind in LEARNERS:
        p.add_argument("--" + kind, nargs="+", metavar="SEED=CHECKPOINT")
    p.add_argument("--require-complete", action="store_true", help="require all five learners, >=3 matched seeds and shot budgets")
    p.add_argument("--samples", "--seeds", dest="samples", type=int, default=20, help="policy RNG seeds per layout (not training seeds)")
    p.add_argument("--games", type=int, default=400, help="head-to-head games per pair of model seeds; even")
    p.add_argument("--tournament-seed", type=int, default=0)
    p.add_argument("--skip-tournament", action="store_true")
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--only", nargs="+", help="select agent IDs: random, density, greedy, raw-q, dqn-board, etc.")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    p.add_argument("--quick", action="store_true", help="1 policy seed, 4 tournament games/model pair; all layouts")
    p.add_argument("--output", type=Path, default=Path("results/compare_4x4_complete"))
    args = p.parse_args(argv)
    if args.quick:
        args.samples, args.games = 1, 4
    if args.samples < 1 or args.games < 2 or args.games % 2 or args.threads < 1:
        p.error("samples/threads must be positive; games must be positive and even")
    try:
        learned = load_specs(args.trained) if args.trained else []
        for kind in LEARNERS:
            learned += external_specs(kind, getattr(args, kind.replace("-", "_")))
        if not learned and not args.baselines_only:
            p.error("provide --trained/checkpoints, or explicitly request --baselines-only")
        if learned and args.baselines_only:
            p.error("--baselines-only cannot be combined with learned models")
        specs = baselines() + learned
        if args.only:
            valid = {s.planner if s.kind == "belief" else s.kind for s in specs}
            if set(args.only) - valid:
                p.error(f"unknown or unavailable agent IDs: {set(args.only) - valid}")
            specs = [s for s in specs if (s.planner if s.kind == "belief" else s.kind) in args.only]
        ids = [(s.label, s.seed) for s in specs]
        if len(set(ids)) != len(ids):
            p.error("duplicate algorithm/training-seed entries")
        selected_learned = [s for s in specs if s.checkpoint]
        if args.require_complete:
            kinds = {s.kind for s in selected_learned}
            if kinds != set(LEARNERS) or args.only or args.skip_tournament or args.skip_oracle or args.quick:
                p.error("complete comparison requires all five learners, all baselines, oracle and full tournament")
            seed_sets = [{s.seed for s in selected_learned if s.kind == kind} for kind in LEARNERS]
            budgets = {s.training_shots for s in selected_learned}
            if len(seed_sets[0]) < 3 or any(s != seed_sets[0] for s in seed_sets) or None in budgets or len(budgets) != 1:
                p.error("complete comparison requires >=3 identical training seeds and a known, matched shot budget")
        groups = defaultdict(list)
        for spec in specs:
            groups[spec.label].append(spec)
        if not args.skip_tournament:
            if len(groups) < 2:
                p.error("a tournament needs at least two algorithms")
            for a, b in combinations(groups.values(), 2):
                paired_specs(a, b)
        if any(s.kind.startswith("dqn-") for s in specs):
            import torch
            torch.set_num_threads(args.threads)
        # Load early so incompatible/missing checkpoints fail before expensive evaluation.
        for spec in selected_learned:
            agent = spec.load()
            play_solo(agent, MINI, layout=Belief.prior(MINI).layouts[0], agent_seed=0)
            del agent
            gc.collect()
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    out = args.output
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        p.error("output is not empty; choose a new experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "config": {**vars(args), "output": str(out),
                                               "trained": str(args.trained) if args.trained else None,
                                               "reuse_solo": str(args.reuse_solo) if args.reuse_solo else None},
                "provenance": provenance(), "models": [asdict(s) for s in specs],
                "model_hashes": {s.checkpoint: digest(s.checkpoint) for s in specs if s.checkpoint}}
    write_json(out / "manifest.json", manifest)
    habits = standard_habits(MINI)
    layouts = Belief.prior(MINI).layouts
    uniform_weights = [1 / len(layouts)] * len(layouts)
    uniform, habit_rows, cases = [], [], []
    if args.reuse_solo:
        uniform, habit_rows, cases, original = reuse_solo(args.reuse_solo, specs, args.samples)
        shutil.copyfile(args.reuse_solo / "games.csv", out / "games.csv")
        manifest["reused_solo"] = {"directory": str(args.reuse_solo.resolve()),
                                  "original_manifest": original,
                                  "file_hashes": {name: digest(args.reuse_solo / name)
                                                  for name in ("games.csv", "uniform.csv", "habits.csv")}}
        write_json(out / "manifest.json", manifest)
        write_csv(out / "uniform.csv", uniform)
        write_csv(out / "habits.csv", habit_rows)
        print(f"Reused complete solo evaluation for {len(specs)} models; continuing oracle and tournament.", flush=True)
    with (out / "games.csv").open("a" if args.reuse_solo else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "training_seed", "layout_index", "policy_seed",
                                             "shots", "decision_seconds"])
        if not args.reuse_solo:
            writer.writeheader()
        for spec in ([] if args.reuse_solo else specs):
            start = time.perf_counter()
            agent = spec.load()
            load_seconds = time.perf_counter() - start
            start = time.perf_counter()
            play_solo(agent, MINI, layout=layouts[0], agent_seed=0)
            warmup_seconds = time.perf_counter() - start
            records = evaluate_layouts(agent, range(args.samples))
            stats = weighted_stats(records, uniform_weights)
            uniform.append({"agent": spec.label, "training_seed": spec.seed, "games": len(records),
                            **stats, "training_shots": spec.training_shots,
                            "training_games": spec.training_games, "training_seconds": spec.training_seconds,
                            "model_load_seconds": load_seconds, "warmup_game_seconds": warmup_seconds,
                            "checkpoint_bytes": Path(spec.checkpoint).stat().st_size if spec.checkpoint else None})
            habit_rows.append({"agent": spec.label, "training_seed": spec.seed,
                               **{h.name: weighted_stats(records, h.distribution(MINI))["mean"] for h in habits}})
            for i, seed, r in records:
                writer.writerow({"agent": spec.label, "training_seed": spec.seed, "layout_index": i,
                                 "policy_seed": seed, "shots": r.num_shots, "decision_seconds": sum(r.move_times)})
            f.flush()
            cases += [case_record(spec, category, item) for category, item in [
                ("fewest shots", min(records, key=lambda item: item[2].num_shots)),
                ("most shots", max(records, key=lambda item: item[2].num_shots))]]
            print(f"{spec.label}, training seed {spec.seed}: {stats['mean']:.4f} shots, "
                  f"{stats['decision_ms']:.4f} ms/decision", flush=True)
            write_csv(out / "uniform.csv", uniform)
            write_csv(out / "habits.csv", habit_rows)
            del agent, records
            gc.collect()
    if not args.skip_oracle:
        oracle = {"agent": "POMDP greedy, told the habit (oracle)", "training_seed": None}
        for h in habits:
            agent = OpponentBeliefAgent(habit=h)
            weights = h.distribution(MINI)
            records = evaluate_layouts(agent, range(args.samples), weights=weights)
            oracle[h.name] = weighted_stats(records, weights)["mean"]
            print(f"oracle: {h.name}, {len(records)} supported games, {oracle[h.name]:.4f} shots", flush=True)
        habit_rows.append(oracle)
        write_csv(out / "habits.csv", habit_rows)
        del agent, records
    summaries = []
    for name in groups:
        rows = [r for r in uniform if r["agent"] == name]
        times = [r["training_seconds"] for r in rows if r["training_seconds"] is not None]
        summaries.append({"agent": name, "training_seeds": len(rows) if rows[0]["training_seed"] is not None else 0,
                          **aggregate(rows), "mean_training_seconds": statistics.fmean(times) if times else None})
    summaries.sort(key=lambda r: r["mean"])
    optimum = None
    if any(s.planner == "optimal" for s in specs):
        from battleship.planning import planner
        plan = planner(MINI)
        optimum = plan.optimal_value(plan.everything, 0)
    for r in summaries:
        r["gap_to_exact_optimum"] = r["mean"] - optimum if optimum is not None else None
    write_csv(out / "summary.csv", summaries)
    bar_chart(summaries, out / "uniform.svg")
    write_json(out / "cases.json", cases)
    case_lines = ["# Example games", "", "Every solo game finishes. These are efficient and inefficient examples, not win/loss labels.",
                  "Examples are descriptive extrema from this run, not typical-performance estimates.", ""]
    for c in cases:
        case_lines += [f"## {c['agent']} / training seed {c['training_seed']} / {c['category']}", "",
                       f"Layout {c['layout_index']}, policy seed {c['policy_seed']}: {c['shots']} shots, {c['misses']} misses.",
                       "Ships: " + "; ".join(", ".join(label(cell, MINI) for cell in ship) for ship in c["layout"]), "",
                       " → ".join(f"{s['label']} {s['result']}" for s in c["history"]), "",
                       f"The first hit occurs at shot {c['first_hit_shot']}: {c['misses_before_first_hit']} misses occur before it "
                       f"and {c['misses_after_first_hit']} occur afterwards. Ships are sunk at shots "
                       f"{', '.join(map(str, c['sunk_at_shots']))}. "
                       "This separates the initial search cost from misses after the first contact; "
                       "it does not prove which decision rule caused a miss.", ""]
    (out / "cases.md").write_text("\n".join(case_lines), encoding="utf-8")
    tournament, first_wins, total = [], 0, 0
    if not args.skip_tournament:
        tournament, first_wins, total = run_tournament(groups, args.games, args.tournament_seed, out)
    curves = []
    for spec in selected_learned:
        if spec.curve:
            with Path(spec.curve).open(encoding="utf-8", newline="") as f:
                points = [(int(r["training_shots"]), float(r["mean_shots"])) for r in csv.DictReader(f)]
            curves.append((f"{spec.label} seed {spec.seed}", points))
    line_chart(curves, out / "learning_curves.svg", "Learning curves: evaluation mean vs training shots")
    missing = sorted(set(LEARNERS) - {s.kind for s in selected_learned})
    report = ["# 4x4 agent comparison", "",
              f"All {len(layouts)} layouts × {args.samples} policy seeds per model. "
              f"Missing learning methods: {', '.join(missing) or 'none'}.", "",
              "These are evaluation means with sampled random decisions; enumerating layouts does not integrate over every random tie. "
              "Training-seed spread and variation across games are reported separately. No held-out-layout generalization is measured.", "",
              "![Mean shots](uniform.svg)", "",
              f"Exact Bellman optimum: {optimum:.6f} shots." if optimum is not None else "Exact optimum was not computed in this filtered run.", "",
              table(["Agent", "Training seeds", "Mean shots", "Gap to optimum", "Game std", "Seed std", "Observed worst", "Decision ms", "Train seconds/seed"],
                    [[r["agent"], r["training_seeds"], f"{r['mean']:.4f}",
                      "N/A" if r["gap_to_exact_optimum"] is None else f"{r['gap_to_exact_optimum']:+.4f}",
                      f"{r['std']:.4f}",
                      "N/A" if r["seed_std"] is None else f"{r['seed_std']:.4f}", r["worst"], f"{r['decision_ms']:.4f}",
                      "N/A" if r["mean_training_seconds"] is None else f"{r['mean_training_seconds']:.2f}"] for r in summaries]), "",
              "## Placement habits", "",
              "Each model's games are reweighted using the exact layout distribution for the habit. "
              "Random decisions still have finite sampling error. The oracle is told the habit; it does not learn it.", "",
              table(["Agent", "Training seed"] + [h.name for h in habits],
                    [[r["agent"], r["training_seed"]] + [f"{r[h.name]:.4f}" for h in habits] for r in habit_rows]), ""]
    if curves:
        report += ["## Learning curves", "", "![Learning curves](learning_curves.svg)", ""]
    if tournament:
        report += ["## Head-to-head", "", f"{total:,} games; first player won {first_wins / total:.2%}. "
                   "Equal training seed IDs are paired, and baselines are played against each trained seed. "
                   "Each model pair alternates the first player. Overall macro gives equal weight to each opposing algorithm.", "",
                   table(["Agent", "Overall macro win rate"], [[r["agent"], f"{r['overall_macro']:.4f}"] for r in tournament]), "",
                   "Full matrix: [tournament.csv](tournament.csv); counts: [tournament_pairs.csv](tournament_pairs.csv).", ""]
    report += ["## Interpretation and limitations", "",
               "- Compare learning methods at equal training shots, not equal replay updates or completed games. "
               "DQN reuses transitions; its gradient steps are recorded in the learning curves.",
               "- Raw-board tables may have sparse coverage; belief-state Q-learning uses an exact model and excludes certain misses. "
               "DQN density/residual variants receive handcrafted probability-density features. These are different information/compute assumptions.",
               "- Small differences between agents need more training seeds and policy samples before claiming a ranking. "
               "The standard deviations here are not confidence intervals, and the observed worst is not a proven worst case.",
               "- Decision timing covers select_shot, including deferred belief updates, but excludes observe, reset and loading. "
               "The warm-up game is excluded and reported separately; global planner caches may already be warm. "
               "Training time excludes evaluation/checkpoint writing, but includes each implementation's training overhead.",
               "- Models are loaded one at a time for solo evaluation and two at a time for head-to-head. "
               "Timing depends on the machine and memory pressure; checkpoint bytes are not resident memory.",
               "- External checkpoints have unknown training budget/time unless supplied through the training manifest. "
               "Existing historical results are not substituted for any measurement.", "",
               "Examples and full shot sequences: [cases.md](cases.md), [cases.json](cases.json). "
               "Raw paired games: [games.csv](games.csv). Source and checkpoint hashes: [manifest.json](manifest.json).", ""]
    (out / "results.md").write_text("\n".join(report), encoding="utf-8")
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    print(f"Comparison complete: {out.resolve() / 'results.md'}")


if __name__ == "__main__":
    main()
