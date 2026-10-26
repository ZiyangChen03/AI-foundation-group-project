"""Evaluate the agents and print shots-to-win statistics.

    uv run scripts/evaluate_all.py                         # 4 x 4, every possible layout
    uv run scripts/evaluate_all.py --board classic --games 2000
    uv run scripts/evaluate_all.py --board classic --tournament 200
"""

import argparse

from battleship import CLASSIC, MINI, all_layouts, evaluate, round_robin
from battleship.agents import BeliefAgent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent


def make_agents():
    # Add new agents here as they are implemented.
    # BeliefAgent uses the exact belief on 4x4 and 1000 particles on 10x10
    # (about 60 ms per shot there, so 1000 classic games take a while).
    return [RandomAgent(), HuntTargetAgent(parity=False), HuntTargetAgent(parity=True),
            ProbabilityDensityAgent(), BeliefAgent()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", choices=("mini", "classic"), default="mini")
    parser.add_argument("--games", type=int,
                        help="games per agent (default: every layout on mini, 1000 on classic)")
    parser.add_argument("--tournament", type=int, metavar="N",
                        help="also run a round robin with N games per pair")
    args = parser.parse_args()
    config = MINI if args.board == "mini" else CLASSIC

    print(f"Shots-to-win on {config.size}x{config.size}, fleet {config.fleet}")
    for agent in make_agents():
        if args.games is None and config == MINI:
            report = evaluate(agent, config, layouts=all_layouts(config))
        else:
            report = evaluate(agent, config, seeds=range(args.games or 1000))
        print(report)

    if args.tournament:
        print()
        print(round_robin(make_agents(), config, games_per_pair=args.tournament))


if __name__ == "__main__":
    main()
