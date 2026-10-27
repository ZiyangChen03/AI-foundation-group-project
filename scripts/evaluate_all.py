"""Evaluate the agents and print shots-to-win statistics.

    uv run scripts/evaluate_all.py                         # 4 x 4, every possible layout
    uv run scripts/evaluate_all.py --board classic --games 2000
    uv run scripts/evaluate_all.py --board classic --tournament 200
    uv run scripts/evaluate_all.py --board classic --belief  # include BeliefAgent (slow)

BeliefAgent runs by default on 4 x 4, where it is fast (exact belief). On 10 x 10
it takes about 60 ms per shot (1000 games: ~45 minutes), so it only runs there
with --belief. --no-belief leaves it out on 4 x 4 too.
"""

import argparse

from battleship import CLASSIC, MINI, all_layouts, evaluate, round_robin
from battleship.agents import BeliefAgent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent


def make_agents(include_belief=True):
    # Add new agents here as they are implemented.
    agents = [RandomAgent(), HuntTargetAgent(parity=False), HuntTargetAgent(parity=True),
              ProbabilityDensityAgent()]
    if include_belief:
        # Exact belief on 4x4; 1000 particles (~60 ms per shot) on 10x10.
        agents.append(BeliefAgent())
    return agents


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", choices=("mini", "classic"), default="mini")
    parser.add_argument("--games", type=int,
                        help="games per agent (default: every layout on mini, 1000 on classic)")
    parser.add_argument("--tournament", type=int, metavar="N",
                        help="also run a round robin with N games per pair")
    parser.add_argument("--belief", action=argparse.BooleanOptionalAction,
                        help="include BeliefAgent (default: on for mini, off for classic)")
    args = parser.parse_args()
    config = MINI if args.board == "mini" else CLASSIC
    include_belief = args.belief if args.belief is not None else config == MINI

    print(f"Shots-to-win on {config.size}x{config.size}, fleet {config.fleet}")
    for agent in make_agents(include_belief):
        if args.games is None and config == MINI:
            report = evaluate(agent, config, layouts=all_layouts(config))
        else:
            report = evaluate(agent, config, seeds=range(args.games or 1000))
        print(report)

    if args.tournament:
        print()
        print(round_robin(make_agents(include_belief), config, games_per_pair=args.tournament))
    if not include_belief and config != MINI:
        print()
        print("BeliefAgent skipped on this board (slow); add --belief to include it.")


if __name__ == "__main__":
    main()
