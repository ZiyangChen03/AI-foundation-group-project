"""Exact expected shots on 4x4, with ties broken uniformly at random.

    uv run scripts/exact_4x4.py
    uv run --extra dqn scripts/exact_4x4.py results/dqn/mini_board_s0/final.pt

Ships are placed uniformly at random over the 264 layouts, so the expected
number of shots is the average over the layouts. On one layout, an agent that
picks uniformly among several best cells has expected shots

    E(shots so far) = 0                                     if the fleet is sunk
    E(shots so far) = 1 + mean over its candidate cells c of E(shots so far + c)

which this script computes by recursion, memoized on the set of cells fired
(the board does not depend on the order). The candidates are what the agent
passes to ``rng.choice``; an agent that does not call it is deterministic.
That works for agents whose choice depends only on the current board, like
probability density and the DQN. Belief greedy and the optimum come from the
exact belief-state planner (``battleship/planning.py``).
"""

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import MINI, BattleshipEnv, all_layouts  # noqa: E402
from battleship.agents import ProbabilityDensityAgent  # noqa: E402
from battleship.planning import planner  # noqa: E402

LAYOUTS = list(all_layouts(MINI))


class _Choices(Exception):
    pass


class _CaptureChoice:
    """Stands in for the agent's rng and reports the cells it would choose from."""

    def choice(self, seq):
        raise _Choices(list(seq))


def candidates(agent, obs) -> list[int]:
    agent.rng = _CaptureChoice()
    try:
        return [agent.select_shot(obs)]
    except _Choices as found:
        return found.args[0]


def expected_shots(agent, layout, config=MINI) -> float:
    """Exact expected shots of a board-only agent on one layout."""
    memo = {}

    def value(history):
        key = frozenset(history)
        if key not in memo:
            env = BattleshipEnv(config)
            obs = env.reset(layout=layout)
            for cell in history:                # in firing order: the game may end early
                obs = env.step(cell).observation
            if env.done:
                memo[key] = 0.0
            else:
                cells = candidates(agent, obs)
                memo[key] = 1 + sum(value(history + (c,)) for c in cells) / len(cells)
        return memo[key]

    return value(())


def per_layout(agent) -> list[float]:
    return [expected_shots(agent, layout) for layout in LAYOUTS]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", type=Path, help="DQN checkpoints (need --extra dqn)")
    args = p.parse_args(argv)

    density = per_layout(ProbabilityDensityAgent())
    rows = [("probability density, random ties", st.mean(density)),
            ("probability density, first tie", st.mean(per_layout(ProbabilityDensityAgent(tie_break="first"))))]
    plan = planner(MINI)
    rows.append(("belief greedy, random ties", plan.greedy_value(plan.everything, 0)))
    rows.append(("optimal V*", plan.optimal_value(plan.everything, 0)))
    notes = []
    if args.models:
        from battleship.dqn import DQNAgent
        for path in args.models:
            shots = per_layout(DQNAgent.load(path))
            rows.append((f"{path.parent.name}/{path.stem}", st.mean(shots)))
            diff = [a - b for a, b in zip(shots, density)]
            notes.append(f"{path.parent.name}/{path.stem} vs density: {st.mean(diff):+.4f}; better on "
                         f"{sum(d < 0 for d in diff)} layouts, worse on {sum(d > 0 for d in diff)}, "
                         f"worst game {max(shots):.0f} shots")
    for name, value in rows:
        print(f"{name:40s} {value:.4f}")
    for line in notes:
        print(line)


if __name__ == "__main__":
    main()
