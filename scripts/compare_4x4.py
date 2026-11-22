"""Compare every 4x4 agent on the same footing (issue #5).

    uv run scripts/compare_4x4.py                                  # full comparison (~10 min)
    uv run scripts/compare_4x4.py --quick                          # smoke run (~1 min)
    uv run scripts/compare_4x4.py --belief-q results/belief_q/seed_0.json.gz

Every agent plays every one of the 264 layouts, once per agent seed (seeds only
change how ties are broken). Because all layouts are covered, the uniform-placement
numbers are exact up to tie-breaking; no sampled test set is involved.

The same games are then re-weighted by each opponent habit in
``battleship/opponents.py`` (edges, center, horizontal, touching, 3 favorites),
which gives the exact expected shots against that opponent. A POMDP agent that is
told the habit (the opponent-belief oracle) is added as the best case.

Finally every pair of agents plays a round-robin tournament of full games.

Writes to --output (default results/compare_4x4/):
    results.md      all tables, ready to paste into the report
    uniform.csv     per-agent statistics under random placement
    habits.csv      expected shots per agent and habit
    tournament.csv  win rate of each agent against each other agent
    uniform.svg     bar chart of the uniform results

Agents from other branches (raw-board Q-learning, DQN) can be added to
``build_agents`` once they are merged.
"""

import argparse
import csv
import math
import os
import time

from battleship import MINI, play_solo, round_robin
from battleship.agents import (BeliefAgent, HuntTargetAgent, OpponentBeliefAgent,
                               ProbabilityDensityAgent, RandomAgent)
from battleship.agents.belief import PLANNERS
from battleship.belief import Belief
from battleship.opponents import standard_habits
from battleship.planning import planner

PLANNER_LABELS = {"greedy": "POMDP greedy", "rollout": "POMDP rollout", "optimal": "POMDP optimal",
                  "infogain": "POMDP info gain", "greedy-info": "POMDP greedy + info tie-break"}


def build_agents(belief_q: str | None = None) -> list[tuple[str, object]]:
    """(label, agent) for every 4x4 agent available on this branch."""
    agents = [("Random", RandomAgent()),
              ("Hunt-target", HuntTargetAgent(parity=False)),
              ("Hunt-target + parity", HuntTargetAgent(parity=True)),
              ("Probability density", ProbabilityDensityAgent())]
    for name in ("greedy", "rollout", "optimal", "infogain", "greedy-info"):
        if name in PLANNERS:                 # info-gain planners exist once issue #12 is merged
            agents.append((PLANNER_LABELS[name], BeliefAgent(planner=name)))
    if belief_q:
        from battleship.agents import BeliefQAgent     # PR #15
        agents.append(("Belief-state Q-learning", BeliefQAgent.load(belief_q)))
    return agents


def play_all_layouts(agent, seeds, weights=None) -> list[list[int] | None]:
    """shots[i][k]: shots needed on layout i with agent seed k (training off).
    Layouts with zero weight are skipped (None)."""
    layouts = Belief.prior(MINI).layouts
    weights = weights or [1] * len(layouts)
    previous, agent.training = agent.training, False
    try:
        return [[play_solo(agent, MINI, layout=layout, agent_seed=seed).num_shots for seed in seeds]
                if w > 0 else None for layout, w in zip(layouts, weights)]
    finally:
        agent.training = previous


def weighted_stats(shots, weights) -> dict:
    """Mean, standard deviation and worst case of shots-to-win under a placement distribution."""
    pairs = [(row, w) for row, w in zip(shots, weights) if w > 0]
    mean = sum(w * sum(row) / len(row) for row, w in pairs)
    var = sum(w * sum((s - mean) ** 2 for s in row) / len(row) for row, w in pairs)
    worst = max(max(row) for row, _ in pairs)
    return {"mean": mean, "std": math.sqrt(var), "worst": worst}


def bar_chart(rows, optimum, path, width=760):
    left, right, bar, gap, top = 250, 70, 22, 8, 40
    height = top + len(rows) * (bar + gap) + 40
    hi = math.ceil(max(r["mean"] for r in rows))
    sx = lambda v: left + (width - left - right) * v / hi
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="12">', f'<rect width="{width}" height="{height}" fill="white"/>',
           '<text x="10" y="22" font-size="14" font-weight="bold">4x4, random placement: '
           'expected shots to win (lower is better)</text>']
    for k in range(0, hi + 1, 2):
        out.append(f'<line x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{top - 6}" y2="{height - 34}" stroke="#eee"/>'
                   f'<text x="{sx(k):.1f}" y="{height - 20}" text-anchor="middle">{k}</text>')
    for i, r in enumerate(rows):
        y = top + i * (bar + gap)
        color = "#2F5597" if r["label"].startswith(("POMDP", "Belief")) else "#8C9AAB"
        out.append(f'<text x="{left - 8}" y="{y + bar * 0.7:.1f}" text-anchor="end">{r["label"]}</text>'
                   f'<rect x="{left}" y="{y}" width="{sx(r["mean"]) - left:.1f}" height="{bar}" fill="{color}"/>'
                   f'<text x="{sx(r["mean"]) + 4:.1f}" y="{y + bar * 0.7:.1f}">{r["mean"]:.3f}</text>')
    out.append(f'<line x1="{sx(optimum):.1f}" x2="{sx(optimum):.1f}" y1="{top - 6}" y2="{height - 34}" '
               f'stroke="#C00000" stroke-dasharray="5 3"/>'
               f'<text x="{sx(optimum) + 4:.1f}" y="{top - 10}" fill="#C00000">optimum {optimum:.3f}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3, help="agent seeds per layout (tie-breaking)")
    parser.add_argument("--games", type=int, default=400, help="tournament games per pair")
    parser.add_argument("--belief-q", help="saved BeliefQAgent table to include (PR #15)")
    parser.add_argument("--output", default="results/compare_4x4")
    parser.add_argument("--quick", action="store_true", help="1 seed, 40 games per pair")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.games = 1, 40
    os.makedirs(args.output, exist_ok=True)
    seeds = range(args.seeds)
    start = time.time()

    agents = build_agents(args.belief_q)
    habits = standard_habits(MINI)
    plan = planner(MINI)
    optimum = plan.optimal_value(plan.everything, 0)

    # 1. every agent on every layout -------------------------------------------
    shots = {}
    for label, agent in agents:
        shots[label] = play_all_layouts(agent, seeds)
        print(f"{label:<32} played all layouts  ({time.time() - start:.0f} s)", flush=True)

    uniform = [{"label": label, **weighted_stats(shots[label], habits[0].distribution(MINI))}
               for label, _ in agents]

    # 2. re-weight by opponent habits, plus the oracle that knows the habit ------
    habit_table = {label: [weighted_stats(shots[label], h.distribution(MINI))["mean"] for h in habits]
                   for label, _ in agents}
    oracle = []
    for h in habits:
        rows = play_all_layouts(OpponentBeliefAgent(habit=h), seeds, h.distribution(MINI))
        oracle.append(weighted_stats(rows, h.distribution(MINI))["mean"])
    habit_table["POMDP greedy, told the habit (oracle)"] = oracle
    print(f"habits done  ({time.time() - start:.0f} s)", flush=True)

    # 3. head-to-head --------------------------------------------------------------
    report = round_robin([agent for _, agent in agents], MINI, games_per_pair=args.games, seed=0)
    labels = [label for label, _ in agents]
    overall = [sum(r for r in row if r is not None) / (len(row) - 1) for row in report.win_rate]
    print(f"tournament done  ({time.time() - start:.0f} s)", flush=True)

    # outputs -----------------------------------------------------------------------
    with open(os.path.join(args.output, "uniform.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "mean_shots", "std", "worst", "gap_to_optimum"])
        w.writerows([r["label"], f"{r['mean']:.4f}", f"{r['std']:.4f}", r["worst"],
                     f"{r['mean'] - optimum:.4f}"] for r in uniform)
    with open(os.path.join(args.output, "habits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent"] + [h.name for h in habits])
        w.writerows([label] + [f"{v:.4f}" for v in values] for label, values in habit_table.items())
    with open(os.path.join(args.output, "tournament.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent"] + labels + ["overall"])
        w.writerows([label] + ["" if r is None else f"{r:.3f}" for r in row] + [f"{o:.3f}"]
                    for label, row, o in zip(labels, report.win_rate, overall))
    bar_chart(uniform, optimum, os.path.join(args.output, "uniform.svg"))

    ranked = sorted(uniform, key=lambda r: r["mean"])
    md = [f"# 4x4 comparison\n\nAll 264 layouts x {args.seeds} agent seed(s); tournament "
          f"{args.games} games per pair. Exact optimum under random placement: {optimum:.3f}.\n",
          "## Random placement\n",
          md_table(["Agent", "Mean shots", "Std", "Worst", "Gap to optimum"],
                   [[r["label"], f"{r['mean']:.3f}", f"{r['std']:.2f}", r["worst"],
                     f"+{r['mean'] - optimum:.3f}"] for r in ranked]),
          "\n## Against opponents with placement habits (expected shots)\n",
          md_table(["Agent"] + [h.name for h in habits],
                   [[label] + [f"{v:.3f}" for v in values] for label, values in habit_table.items()]),
          "\n## Head-to-head (win rate of row against column)\n",
          md_table(["Agent"] + [str(i + 1) for i in range(len(labels))] + ["Overall"],
                   [[f"{i + 1}. {label}"] + ["-" if r is None else f"{r:.2f}" for r in row] + [f"{o:.2f}"]
                    for i, (label, row, o) in enumerate(zip(labels, report.win_rate, overall))]),
          f"\nFirst player won {report.first_player_wins} of {report.total_games} games.\n"]
    text = "\n".join(md)
    with open(os.path.join(args.output, "results.md"), "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"done in {time.time() - start:.0f} s; results in {args.output}/")


if __name__ == "__main__":
    main()
