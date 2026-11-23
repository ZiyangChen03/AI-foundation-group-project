"""Benchmark trained DQN agents against the baselines on the same boards.

    uv run --extra dqn scripts/benchmark_dqn.py classic results/dqn/classic_board_s0/best.pt \\
        results/dqn/classic_density_s0/best.pt results/dqn/classic_residual_s0/final.pt
    uv run --extra dqn scripts/benchmark_dqn.py mini results/dqn/mini_board_s0/final.pt \\
        results/dqn/mini_board_lr1e-4_s0/final.pt

These write results/dqn/benchmark_<board>.md and .json, the tables the docs
quote; pass --out to write anywhere else.

10x10: seeds 0-1999 (disjoint from the training and validation boards), played
through ``play_solo`` exactly like every other agent. Each agent is compared
with probability density by the paired difference on the same boards.

4x4: every one of the 264 layouts, each played ``--samples`` times with
different agent seeds, and averaged layout by layout. Ships are placed
uniformly at random, so the average over the layouts is the expected number of
shots, with no sampling error from the boards. A deterministic agent (the DQN)
plays the same game every time, so its number is exact and its ± is 0; the ±
of an agent that breaks ties at random is the error left from its own coin
flips (games with the same seed are paired, so the difference to density is
measured game by game). There are no held-out boards on 4x4: training draws from the same 264
layouts, so this measures how well the agent plays this game, not how it
generalizes to boards it has not seen.

Shots are also split into hunting (no open hit on the board) and targeting
(an unsunk hit is showing).
"""

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import CLASSIC, MINI, CellState, all_layouts, play_solo  # noqa: E402
from battleship.agents import (BeliefAgent, HuntTargetAgent,  # noqa: E402
                               ProbabilityDensityAgent, RandomAgent)

BASELINES = {"random": RandomAgent, "hunt-target+parity": lambda: HuntTargetAgent(True),
             "probability-density": ProbabilityDensityAgent, "belief-greedy": BeliefAgent}
REFERENCE = "probability-density"


def make_agent(spec):
    if spec in BASELINES:
        return BASELINES[spec]()
    import torch
    torch.set_num_threads(1)
    from battleship.dqn import DQNAgent
    return DQNAgent.load(spec)


def targeting_shots(record):
    """Shots fired while an unsunk hit was on the board."""
    open_hits, count = 0, 0
    for shot in record.shots:
        count += open_hits > 0
        open_hits += shot.result.value == "hit"
        if shot.result.value == "sunk":
            open_hits -= shot.sunk_length - 1
    return count


def play(job):
    """On 4x4 a game id g is layout g // samples, played with agent seed g."""
    spec, board, games, samples = job
    config = CLASSIC if board == "classic" else MINI
    agent = make_agent(spec)
    out = []
    for g in games:
        if board == "classic":
            record = play_solo(agent, config, seed=g)
        else:
            record = play_solo(agent, config, layout=LAYOUTS[g // samples], agent_seed=g)
        out.append((g, record.num_shots, targeting_shots(record)))
    return spec, out


def over_layouts(values, samples):
    """Mean over the layouts of the per-layout means, and its 95% margin from the
    spread of the samples within each layout (the layouts themselves are all
    there, so they add no error). ``values`` holds each layout's samples in a row."""
    groups = [values[i:i + samples] for i in range(0, len(values), samples)]
    var = sum(st.variance(g) / samples for g in groups) / len(groups) ** 2
    return st.mean(st.mean(g) for g in groups), 1.96 * var ** .5


LAYOUTS = list(all_layouts(MINI))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("board", choices=("classic", "mini"))
    p.add_argument("models", nargs="*", type=Path)
    p.add_argument("--games", type=int, default=2000, help="10x10 boards (seeds 0..games-1)")
    p.add_argument("--samples", type=int, default=100, help="4x4 games per layout")
    p.add_argument("--baselines", nargs="*", default=list(BASELINES))
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.board == "mini" and args.samples < 2:
        p.error("--samples must be at least 2 (one game per layout cannot measure tie-breaking)")

    samples = 1 if args.board == "classic" else args.samples
    games = list(range(args.games)) if args.board == "classic" else list(range(len(LAYOUTS) * samples))
    names = {b: b for b in args.baselines}
    for path in args.models:
        names[str(path)] = f"{path.parent.name}/{path.stem}"
    chunks = [games[i::args.jobs] for i in range(args.jobs)]
    jobs = [(spec, args.board, chunk, samples) for spec in names for chunk in chunks if chunk]
    started = time.time()
    shots = {spec: {} for spec in names}
    with ProcessPoolExecutor(args.jobs) as pool:
        for spec, rows in pool.map(play, jobs):
            for g, n, t in rows:
                shots[spec][g] = (n, t)

    results = []
    for spec, name in names.items():
        s = [shots[spec][g][0] for g in games]
        t = [shots[spec][g][1] for g in games]
        row = dict(agent=name, games=len(s), mean=st.mean(s), median=st.median(s),
                   hunting=st.mean(a - b for a, b in zip(s, t)), targeting=st.mean(t))
        if args.board == "classic":                   # boards are a sample of all boards
            row.update(ci95=1.96 * st.stdev(s) / len(s) ** .5)
            if REFERENCE in shots and spec != REFERENCE:
                d = [a - shots[REFERENCE][g][0] for a, g in zip(s, games)]
                row.update(vs_density=st.mean(d), vs_density_ci95=1.96 * st.stdev(d) / len(d) ** .5)
        else:                                         # every layout: only the agents' coin flips vary
            mean, ci95 = over_layouts(s, samples)
            row.update(mean=mean, ci95=ci95)
            if REFERENCE in shots and spec != REFERENCE:
                d = [a - shots[REFERENCE][g][0] for a, g in zip(s, games)]   # same layout and seed
                row.update(zip(("vs_density", "vs_density_ci95"), over_layouts(d, samples)))
        results.append(row)

    if args.board == "mini":
        from battleship.planning import planner
        plan = planner(MINI)
        optimum = plan.optimal_value(plan.everything, 0)
        results.append(dict(agent="optimal V* (exact, planning.py)", mean=optimum, ci95=0.0))

    what = (f"10x10, seeds 0-{args.games - 1}" if args.board == "classic"
            else f"4x4, all 264 layouts x {samples} agent seeds (expected shots)")
    lines = [f"# DQN benchmark: {what}", "",
             "| Agent | Mean shots | ± 95% | Median | Hunting | Targeting | Paired vs density (95%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    digits = 2 if args.board == "classic" else 3        # 4x4 margins are a few thousandths
    for r in results:
        paired = (f"{r['vs_density']:+.{digits}f} ± {r['vs_density_ci95']:.{digits}f}"
                  if "vs_density" in r else "-")
        played = (f"{r['median']:.1f} | {r['hunting']:.2f} | {r['targeting']:.2f}" if "median" in r
                  else "- | - | -")
        lines.append(f"| {r['agent']} | {r['mean']:.3f} | {r['ci95']:.{digits}f} | {played} | {paired} |")
    lines.append(f"\n{time.time() - started:.0f} s")
    text = "\n".join(lines)
    print(text)
    out = args.out or Path("results/dqn") / f"benchmark_{args.board}"
    # UTF-8 everywhere; Windows would otherwise write "±" in its own code page.
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
