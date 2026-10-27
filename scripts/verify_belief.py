"""Verify the belief-based POMDP agent.

    uv run scripts/verify_belief.py              # about 5 minutes
    uv run scripts/verify_belief.py --games 1000 # larger 10x10 comparison

Games are played by the runner (play_solo / evaluate); checks read the records
the API returns and compare against independent references.
"""

import argparse
import random
import statistics

from battleship import (CLASSIC, MINI, BattleshipEnv, GameConfig, all_layouts, evaluate,
                        play_solo)
from battleship.agents import BeliefAgent, ProbabilityDensityAgent
from battleship.belief import Belief, explains, observed_outcome
from battleship.particles import Constraints, ParticleBelief
from battleship.planning import planner
from verify_common import CONFIGS, check_public_only, check_record

NO_REVEAL_4 = GameConfig(4, (3, 2), reveal_sunk_cells=False)
NO_REVEAL_10 = GameConfig(10, (5, 4, 3, 3, 2), reveal_sunk_cells=False)


def random_histories(config, games, seed=0):
    rng = random.Random(seed)
    for game in range(games):
        env = BattleshipEnv(config)
        obs = env.reset(seed=game)
        while not env.done:
            cell = rng.choice(obs.legal_actions)
            after = env.step(cell).observation
            yield env, obs, after, cell
            obs = after


def check_1_consistency():
    """The single-ship checks used on 10x10 keep exactly the layouts that a
    plain replay of the history keeps (equal-length ships, both rules)."""
    configs = [MINI, NO_REVEAL_4, GameConfig(4, (2, 2, 1)), GameConfig(5, (3, 2, 2)),
               GameConfig(5, (3, 2, 2), reveal_sunk_cells=False)]
    positions = 0
    for config in configs:
        layouts = list(all_layouts(config))
        for env, before, after, cell in random_histories(config, 5):
            if not before.history:
                constraints, exact = Constraints(config), Belief.prior(config)
            outcome = observed_outcome(before, after)
            constraints.add(cell, outcome)
            exact = exact.update(cell, outcome)
            by_mask = constraints.placements.by_mask
            kept = tuple(i for i, layout in enumerate(layouts) if constraints.explains(
                tuple(by_mask[sum(1 << c for c in ship.cells)] for ship in layout)))
            assert kept == exact.support, "single-ship checks disagree with the exact belief"
            if not (config.reveal_sunk_cells and len(set(config.fleet)) < len(config.fleet)):
                assert all(explains(layouts[i], after) for i in kept)
            positions += 1
    print(f"[1] single-ship consistency = exact belief on {positions} positions "
          f"({len(configs)} configs): OK")


def check_2_rules_and_public_only():
    """Legal games on every config, and decisions use only public feedback."""
    games = 0
    for config in CONFIGS:
        agent = BeliefAgent(num_particles=100)
        for seed in range(15):
            assert check_record(play_solo(agent, config, seed=seed)), "fleet not sunk"
            games += 1
    pairs = check_public_only([play_solo(BeliefAgent(), MINI, layout=layout, agent_seed=0)
                               for layout in all_layouts(MINI)])
    print(f"[2] {games} games legal on {len(CONFIGS)} configs; decisions depend only on "
          f"public results ({pairs} pairs of 4x4 games): OK")


def check_3_sampler():
    """Particle estimates are unbiased: error vs the exact belief halves when
    the particles quadruple, and the average signed error is ~0."""
    config = GameConfig(5, (3, 2, 2), reveal_sunk_cells=False)
    rows = []
    for n in (100, 400, 1600):
        worst, signed, cells = [], 0.0, 0
        belief = exact = None
        for env, before, after, cell in random_histories(config, 8, seed=7):
            if not before.history:
                belief = ParticleBelief(config, random.Random(len(worst)), n)
                exact = Belief.prior(config)
            outcome = observed_outcome(before, after)
            belief.update(cell, outcome)
            exact = exact.update(cell, outcome)
            if env.done:
                continue
            pe, pp = exact.occupancy(), belief.occupancy()
            worst.append(max(abs(a - b) for a, b in zip(pe, pp)))
            signed += sum(b - a for a, b in zip(pe, pp))
            cells += len(pe)
        rows.append((n, statistics.fmean(worst), signed / cells))
    for (n1, e1, _), (n2, e2, _) in zip(rows, rows[1:]):
        assert 0.35 < e2 / e1 < 0.7, f"error does not shrink like noise from {n1} to {n2}"
    assert all(abs(bias) < 0.002 for _, _, bias in rows)
    print("[3] particle estimate vs exact (5x5, ships 3,2,2, no reveal): "
          + ", ".join(f"N={n}: worst-cell error {e:.4f}, bias {b:+.5f}" for n, e, b in rows))


def check_4_exact_planning():
    """On 4x4, optimal <= rollout <= greedy, and the agents achieve them."""
    for config in (MINI, NO_REVEAL_4):
        plan = planner(config)
        optimal = plan.optimal_value(plan.everything, 0)
        rollout = plan.policy_value(plan.rollout_shots)
        greedy = plan.greedy_value(plan.everything, 0)
        assert optimal <= rollout + 1e-12 <= greedy + 2e-12
        played = {name: evaluate(BeliefAgent(planner=name), config,
                                 layouts=all_layouts(config)).mean
                  for name in ("greedy", "rollout", "optimal")}
        density = evaluate(ProbabilityDensityAgent(), config, layouts=all_layouts(config)).mean
        print(f"[4] 4x4 reveal={config.reveal_sunk_cells}: exact expected shots optimal "
              f"{optimal:.4f}, rollout {rollout:.4f}, greedy {greedy:.4f}; one run on all 264 "
              f"layouts with random ties: optimal {played['optimal']:.2f}, rollout {played['rollout']:.2f}, "
              f"greedy {played['greedy']:.2f}, probability density {density:.2f}")


def check_5_classic(games):
    """10x10, the same boards for every agent: paired differences."""
    for config in (CLASSIC, NO_REVEAL_10):
        seeds = range(games)
        belief = evaluate(BeliefAgent(), config, seeds=seeds)
        density = evaluate(ProbabilityDensityAgent(), config, seeds=seeds)
        diff = [a.num_shots - b.num_shots for a, b in zip(belief.records, density.records)]
        se = statistics.stdev(diff) / len(diff) ** 0.5
        print(f"[5] 10x10 reveal={config.reveal_sunk_cells}, {games} games: belief "
              f"{belief.mean:.2f} (worst {belief.worst}), probability density "
              f"{density.mean:.2f} (worst {density.worst}); "
              f"difference {statistics.fmean(diff):+.2f} ± {1.96 * se:.2f} (95%; use "
              f"--games 1000 for a tight comparison)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=100, help="10x10 games per agent")
    args = parser.parse_args()
    check_1_consistency()
    check_2_rules_and_public_only()
    check_3_sampler()
    check_4_exact_planning()
    check_5_classic(args.games)
