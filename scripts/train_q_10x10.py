"""Train and test the 10x10 versions of the tabular Q-learners.

    uv run scripts/train_q_10x10.py local --games 20000
    uv run scripts/train_q_10x10.py belief --games 400

``local`` is ``LocalQAgent`` (the raw-board Q-table keyed on 3x3 neighbourhoods);
``belief`` is ``ParticleBeliefQAgent`` (the belief-state Q-table keyed on what
the particle belief says about each cell). See battleship/agents/q_10x10.py.

Training games use seeds from 1,000,000,000 (as ``runner.train`` does); the test
uses seeds 0..test-1, the same boards as the other 10x10 benchmarks, so the two
never overlap. Epsilon falls linearly from --eps-start to --eps-end. Writes the
table and a summary to results/q_10x10/.
"""

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import CLASSIC, play_solo  # noqa: E402
from battleship.agents.q_10x10 import LocalQAgent, ParticleBeliefQAgent  # noqa: E402
from battleship.runner import TRAIN_SEED  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", choices=("local", "belief"))
    p.add_argument("--games", type=int, default=20000, help="training games")
    p.add_argument("--test", type=int, default=200, help="test boards (seeds 0..test-1)")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--particles", type=int, default=100, help="belief: particles while training")
    p.add_argument("--test-particles", type=int, default=300, help="belief: particles while testing")
    p.add_argument("--every", type=int, default=0, help="test on 50 boards every N games (0: off)")
    p.add_argument("--gamma", type=float, default=0.0, help="discount (see q_10x10.py for why 0)")
    p.add_argument("--reward", choices=("miss", "shot"), default="miss",
                   help="-1 per miss (default) or -1 per shot, as the 4x4 originals")
    p.add_argument("--out", type=Path, default=Path("results/q_10x10"))
    args = p.parse_args(argv)

    agent = (LocalQAgent(gamma=args.gamma, reward=args.reward) if args.agent == "local"
             else ParticleBeliefQAgent(num_particles=args.particles, gamma=args.gamma,
                                       reward=args.reward))
    curve = []

    def test(n, particles=None):
        if isinstance(agent, ParticleBeliefQAgent) and particles:
            agent.tracker.num_particles = particles
        shots = [play_solo(agent, CLASSIC, seed=s).num_shots for s in range(n)]
        if isinstance(agent, ParticleBeliefQAgent):
            agent.tracker.num_particles = args.particles
        return shots

    started = time.time()
    agent.training = True
    for game in range(args.games):
        agent.epsilon = args.eps_start + (args.eps_end - args.eps_start) * game / max(1, args.games - 1)
        play_solo(agent, CLASSIC, seed=TRAIN_SEED + game)
        if args.every and (game + 1) % args.every == 0:
            agent.training = False
            mean = st.mean(test(50))
            agent.training = True
            curve.append({"games": game + 1, "test50_mean": mean, "keys": agent.num_keys()})
            print(f"games {game + 1:>6}  test(50) {mean:6.2f}  keys {agent.num_keys()}", flush=True)
    agent.training = False
    train_seconds = time.time() - started

    started = time.time()
    shots = test(args.test, args.test_particles)
    test_seconds = time.time() - started
    summary = {"agent": agent.name, "board": "10x10 classic", "training_games": args.games,
               "eps_start": args.eps_start, "eps_end": args.eps_end, "gamma": args.gamma,
               "reward": args.reward, "keys": agent.num_keys(),
               "train_seconds": round(train_seconds, 1), "test_boards": args.test,
               "mean_shots": st.mean(shots),
               "ci95": 1.96 * st.stdev(shots) / len(shots) ** 0.5,
               "median": st.median(shots), "worst": max(shots), "best": min(shots),
               "ms_per_shot": 1000 * test_seconds / sum(shots), "curve": curve}
    if args.agent == "belief":
        summary.update(train_particles=args.particles, test_particles=args.test_particles)
    args.out.mkdir(parents=True, exist_ok=True)
    agent.save(args.out / f"{args.agent}.json.gz")
    (args.out / f"{args.agent}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{agent.name}: {args.games} training games in {train_seconds / 60:.1f} min, "
          f"{agent.num_keys()} keys; test on {args.test} boards: {summary['mean_shots']:.2f} "
          f"± {summary['ci95']:.2f} shots (worst {summary['worst']}), "
          f"{summary['ms_per_shot']:.1f} ms per shot")


if __name__ == "__main__":
    main()
