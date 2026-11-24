"""Every agent that plays 10x10, on the same test boards.

    uv run scripts/benchmark_10x10.py                    # seeds 0-199
    uv run scripts/benchmark_10x10.py --games 500 --jobs 12

Boards are seeds 0..games-1 (disjoint from every training seed). Each agent is
compared with probability density by the paired difference on the same boards.
The 10x10 Q-tables are read from results/q_10x10/ (scripts/train_q_10x10.py).
Writes results/benchmark_10x10.md and .json.
"""

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battleship import CLASSIC, play_solo  # noqa: E402

AGENTS = ["random", "hunt-target+parity", "probability-density", "belief-greedy",
          "belief-infogain", "belief-greedy-info", "belief-optimal-approx",
          "q-learning-local", "belief-q-particles"]
REFERENCE = "probability-density"


def make(name):
    from battleship.agents import BeliefAgent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent
    from battleship.agents.belief_10x10 import LargeBoardBeliefAgent
    from battleship.agents.q_10x10 import LocalQAgent, ParticleBeliefQAgent
    q = ROOT / "results" / "q_10x10"
    return {
        "random": RandomAgent,
        "hunt-target+parity": lambda: HuntTargetAgent(parity=True),
        "probability-density": ProbabilityDensityAgent,
        "belief-greedy": BeliefAgent,
        "belief-infogain": lambda: LargeBoardBeliefAgent("infogain"),
        "belief-greedy-info": lambda: LargeBoardBeliefAgent("greedy-info"),
        "belief-optimal-approx": lambda: LargeBoardBeliefAgent("optimal-approx"),
        "q-learning-local": lambda: LocalQAgent.load(q / "local.json.gz"),
        "belief-q-particles": lambda: ParticleBeliefQAgent.load(q / "belief.json.gz"),
    }[name]()


def play(job):
    name, seeds = job
    agent = make(name)
    out = []
    for seed in seeds:
        record = play_solo(agent, CLASSIC, seed=seed)
        out.append((seed, record.num_shots, sum(record.move_times)))
    return name, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--agents", nargs="*", default=AGENTS, choices=AGENTS)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_10x10")
    args = p.parse_args(argv)

    seeds = list(range(args.games))
    chunk = max(1, len(seeds) // args.jobs)
    jobs = [(name, seeds[i:i + chunk]) for name in args.agents for i in range(0, len(seeds), chunk)]
    shots = {name: {} for name in args.agents}
    times = {name: 0.0 for name in args.agents}
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, rows in pool.map(play, jobs):
            for seed, n, t in rows:
                shots[name][seed] = n
                times[name] += t

    rows = []
    for name in args.agents:
        s = [shots[name][seed] for seed in seeds]
        row = dict(agent=name, games=len(s), mean=st.mean(s), ci95=1.96 * st.stdev(s) / len(s) ** .5,
                   median=st.median(s), worst=max(s), best=min(s),
                   ms_per_shot=1000 * times[name] / sum(s))
        if REFERENCE in shots and name != REFERENCE:
            d = [shots[name][seed] - shots[REFERENCE][seed] for seed in seeds]
            row.update(vs_density=st.mean(d), vs_density_ci95=1.96 * st.stdev(d) / len(d) ** .5)
        rows.append(row)
    rows.sort(key=lambda r: r["mean"])

    lines = [f"# Every 10x10 agent, seeds 0-{args.games - 1}", "",
             "| Agent | Mean shots | ± 95% | Median | Worst | Paired vs density (95%) | ms per shot |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        paired = f"{r['vs_density']:+.2f} ± {r['vs_density_ci95']:.2f}" if "vs_density" in r else "-"
        lines.append(f"| {r['agent']} | {r['mean']:.2f} | {r['ci95']:.2f} | {r['median']:.1f} | "
                     f"{r['worst']} | {paired} | {r['ms_per_shot']:.1f} |")
    lines.append(f"\n{time.time() - started:.0f} s")
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
