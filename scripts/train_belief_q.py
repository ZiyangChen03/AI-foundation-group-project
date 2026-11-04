"""Train belief-state Q-learning on 4x4 and evaluate it exactly.

    uv run scripts/train_belief_q.py                                   # 3 seeds x 1,000,000 episodes
    uv run scripts/train_belief_q.py --episodes 200000 --every 20000 --seeds 0
    uv run scripts/train_belief_q.py --output results/belief_q --jobs 3

Every ``--every`` episodes the greedy policy is evaluated *exactly* over all 264
layouts (no sampling noise) with Planner.policy_value. Writes, to --output:

    curve.csv         episode, seed, exact expected shots, Q-table states
    learning_curve.svg  exact shots vs. episodes, one line per seed, with references
    seed_N.json.gz    the trained Q-table for each seed
    summary.txt       final results and reference values
"""

import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

from battleship import MINI, all_layouts, evaluate
from battleship.agents import BeliefQAgent, HuntTargetAgent
from battleship.planning import planner


def run_seed(args):
    seed, episodes, every, alpha, initial, output = args
    agent = BeliefQAgent(alpha=alpha, initial=initial)
    rows = [(0, seed, agent.exact_value(), 0)]
    start = time.time()

    def checkpoint(episode):
        rows.append((episode, seed, agent.exact_value(), agent.num_states()))
        print(f"seed {seed}: {episode:>9,} episodes  {rows[-1][2]:.3f} shots  "
              f"{rows[-1][3]:,} states  ({time.time() - start:.0f} s)", flush=True)

    agent.fast_train(episodes, random.Random(seed), callback=checkpoint, every=every)
    agent.save(os.path.join(output, f"seed_{seed}.json.gz"))
    return rows, time.time() - start


def references():
    plan = planner(MINI)
    hunt = evaluate(HuntTargetAgent(parity=True), MINI, layouts=list(all_layouts(MINI))).mean
    return {"POMDP optimal": plan.optimal_value(plan.everything, 0),
            "POMDP greedy": plan.greedy_value(plan.everything, 0),
            "Hunt-target + parity": hunt}


def svg_chart(rows, refs, path, width=720, height=420):
    left, right, top, bottom = 70, 190, 30, 50
    xs = [r[0] for r in rows]
    x_max = max(xs) or 1
    y_lo, y_hi = 8.5, max(r[2] for r in rows) + 0.3
    sx = lambda x: left + (width - left - right) * x / x_max
    sy = lambda y: top + (height - top - bottom) * (y_hi - y) / (y_hi - y_lo)
    colors = ["#2F5597", "#C55A11", "#548235", "#7030A0", "#BF9000"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="18" font-size="14" font-weight="bold">Belief-state Q-learning on 4x4 '
           f'(exact expected shots, lower is better)</text>']
    for k in [8.5 + 0.5 * j for j in range(int((y_hi - 8.5) / 0.5) + 1)]:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(k):.1f}" y2="{sy(k):.1f}" stroke="#e5e5e5"/>'
                   f'<text x="{left - 8}" y="{sy(k) + 4:.1f}" text-anchor="end">{k:g}</text>')
    for i in range(6):
        x = x_max * i / 5
        out.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 18}" text-anchor="middle">{x:,.0f}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 12}" text-anchor="middle">Training episodes</text>')
    out.append(f'<text transform="translate(18 {(top + height - bottom) / 2}) rotate(-90)" '
               f'text-anchor="middle">Expected shots to win</text>')
    label_y = None
    for (name, value), dash in sorted(zip(refs.items(), ["6 4", "2 3", "10 3 2 3"]),
                                      key=lambda item: -item[0][1]):
        y = sy(value) + 4
        label_y = y if label_y is None else max(y, label_y + 15)    # keep labels apart
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(value):.1f}" y2="{sy(value):.1f}" '
                   f'stroke="#555" stroke-dasharray="{dash}"/>'
                   f'<text x="{width - right + 6}" y="{label_y:.1f}">{name} {value:.3f}</text>')
    seeds = sorted({r[1] for r in rows})
    for i, seed in enumerate(seeds):
        pts = " ".join(f"{sx(r[0]):.1f},{sy(r[2]):.1f}" for r in rows if r[1] == seed)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % 5]}" stroke-width="2"/>')
        out.append(f'<text x="{width - right - 60}" y="{top + 16 + 16 * i}" fill="{colors[i % 5]}">seed {seed}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def chart_from_csv(path, refs, out):
    with open(path) as f:
        rows = [(int(r["episode"]), int(r["seed"]), float(r["exact_shots"]), int(r["states"]))
                for r in csv.DictReader(f)]
    svg_chart(rows, refs, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--every", type=int, default=100_000, help="exact evaluation interval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=None,
                        help="fixed learning rate (default: 1 / visits of each state-action pair)")
    parser.add_argument("--initial", type=float, default=-20.0, help="value of unvisited actions")
    parser.add_argument("--jobs", type=int, default=1, help="seeds trained in parallel")
    parser.add_argument("--output", default="results/belief_q")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    tasks = [(s, args.episodes, args.every, args.alpha, args.initial, args.output) for s in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_seed, tasks))
    rows = [row for seed_rows, _ in results for row in seed_rows]
    refs = references()

    with open(os.path.join(args.output, "curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "exact_shots", "states"])
        writer.writerows((e, s, f"{v:.4f}", n) for e, s, v, n in rows)
    svg_chart(rows, refs, os.path.join(args.output, "learning_curve.svg"))

    finals = [seed_rows[-1] for seed_rows, _ in results]
    mean = sum(r[2] for r in finals) / len(finals)
    lines = [f"Belief-state Q-learning, 4x4, {args.episodes:,} episodes per seed, "
             f"alpha={args.alpha or '1/visits'}, initial={args.initial}, epsilon 1.0 -> 0.0 (linear)", ""]
    lines += [f"seed {r[1]}: {r[2]:.3f} shots, {r[3]:,} states, {t:.0f} s"
              for r, (_, t) in zip(finals, results)]
    lines += [f"mean over seeds: {mean:.3f} shots", ""]
    lines += [f"{name}: {value:.3f}" for name, value in refs.items()]
    text = "\n".join(lines)
    with open(os.path.join(args.output, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
