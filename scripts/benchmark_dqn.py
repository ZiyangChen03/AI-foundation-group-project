"""Benchmark trained DQN agents against the baselines on the same boards.

    uv run scripts/benchmark_dqn.py classic results/dqn/classic_*_s0/best.pt
    uv run scripts/benchmark_dqn.py mini results/dqn/mini_board_s*/final.pt

10x10: seeds 0-1999 (disjoint from the training and validation boards), played
through ``play_solo`` exactly like every other agent. Each agent is compared
with probability density by the paired difference on the same boards.
4x4: one game per layout over all 264 layouts. For a deterministic policy (the
DQN agents) that is the exact expected number of shots; agents that break ties
at random get one sample per layout.

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
    spec, board, games = job
    config = CLASSIC if board == "classic" else MINI
    agent = make_agent(spec)
    out = []
    for g in games:
        if board == "classic":
            record = play_solo(agent, config, seed=g)
        else:
            record = play_solo(agent, config, layout=LAYOUTS[g], agent_seed=g)
        out.append((g, record.num_shots, targeting_shots(record)))
    return spec, out


LAYOUTS = list(all_layouts(MINI))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("board", choices=("classic", "mini"))
    p.add_argument("models", nargs="*", type=Path)
    p.add_argument("--games", type=int, default=2000, help="10x10 boards (seeds 0..games-1)")
    p.add_argument("--baselines", nargs="*", default=list(BASELINES))
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    games = list(range(args.games)) if args.board == "classic" else list(range(len(LAYOUTS)))
    names = {b: b for b in args.baselines}
    for path in args.models:
        names[str(path)] = f"{path.parent.name}/{path.stem}"
    chunks = [games[i::args.jobs] for i in range(args.jobs)]
    jobs = [(spec, args.board, chunk) for spec in names for chunk in chunks if chunk]
    started = time.time()
    shots = {spec: {} for spec in names}
    with ProcessPoolExecutor(args.jobs) as pool:
        for spec, rows in pool.map(play, jobs):
            for g, n, t in rows:
                shots[spec][g] = (n, t)

    ref = [shots[REFERENCE][g][0] for g in games] if REFERENCE in shots else None
    results = []
    for spec, name in names.items():
        s = [shots[spec][g][0] for g in games]
        t = [shots[spec][g][1] for g in games]
        row = dict(agent=name, games=len(s), mean=st.mean(s), ci95=1.96 * st.stdev(s) / len(s) ** .5,
                   median=st.median(s), hunting=st.mean(a - b for a, b in zip(s, t)),
                   targeting=st.mean(t))
        if ref is not None and spec != REFERENCE:
            d = [a - b for a, b in zip(s, ref)]
            row.update(vs_density=st.mean(d), vs_density_ci95=1.96 * st.stdev(d) / len(d) ** .5)
        results.append(row)

    what = (f"10x10, seeds 0-{args.games - 1}" if args.board == "classic"
            else "4x4, all 264 layouts (exact for deterministic agents)")
    lines = [f"# DQN benchmark: {what}", "",
             "| Agent | Mean shots | ± 95% | Median | Hunting | Targeting | Paired vs density (95%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        paired = (f"{r['vs_density']:+.2f} ± {r['vs_density_ci95']:.2f}" if "vs_density" in r else "-")
        lines.append(f"| {r['agent']} | {r['mean']:.3f} | {r['ci95']:.2f} | {r['median']:.1f} | "
                     f"{r['hunting']:.2f} | {r['targeting']:.2f} | {paired} |")
    lines.append(f"\n{time.time() - started:.0f} s")
    text = "\n".join(lines)
    print(text)
    out = args.out or Path("results/dqn") / f"benchmark_{args.board}"
    out.with_suffix(".md").write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
