"""Verify the opponent-belief agent and benchmark it against placement habits.

    uv run scripts/verify_opponent_belief.py            # about 2 minutes
    uv run scripts/verify_opponent_belief.py --runs 20  # tighter learning curves

Games are played by the runner (play_solo); checks read the records it returns
and compare against independent references. Results are exact expectations
over every 4x4 layout, weighted by the opponent's habit (see
battleship/opponents.py), with ties averaged over agent seeds.
"""

import argparse
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from battleship import MINI, BattleshipEnv, GameConfig, all_layouts, play_solo
from battleship.agents import BeliefAgent, OpponentBeliefAgent, ProbabilityDensityAgent
from battleship.belief import Belief, explains, observed_outcome
from battleship.opponents import expected_shots, standard_habits
from battleship.planning import Planner
from battleship.runner import TRAIN_SEED
from verify_common import check_public_only, check_record

NO_REVEAL_4 = GameConfig(4, (3, 2), reveal_sunk_cells=False)
CHECKPOINTS = (0, 10, 30, 100, 300, 1000)
LEARNERS = (("placements", 1), ("placements", 3), ("layouts", 1))
SEEDS = range(3)


def check_1_weighted_bayes():
    """With random prior weights, the belief after every shot equals Bayes'
    rule computed by brute force: replay the history on every layout, keep
    those that explain it, and weight them by the prior."""
    configs = [MINI, NO_REVEAL_4, GameConfig(4, (2, 2, 1), reveal_sunk_cells=False)]
    positions = 0
    for config in configs:
        layouts = list(all_layouts(config))
        rng = random.Random(1)
        for game in range(10):
            weights = [rng.random() for _ in layouts]
            env = BattleshipEnv(config)
            obs = env.reset(seed=game)
            belief = Belief.prior(config, weights)
            while not env.done:
                cell = rng.choice(obs.legal_actions)
                after = env.step(cell).observation
                belief = belief.update(cell, observed_outcome(obs, after))
                kept = [i for i, layout in enumerate(layouts) if explains(layout, after)]
                assert tuple(kept) == belief.support
                total = sum(weights[i] for i in kept)
                fired = {shot.cell for shot in after.history}
                for c, p in enumerate(belief.occupancy()):
                    brute = 0.0 if c in fired else sum(
                        weights[i] for i in kept if any(c in s.cells for s in layouts[i])) / total
                    assert abs(p - brute) < 1e-12, "weighted belief disagrees with Bayes' rule"
                obs = after
                positions += 1
    print(f"[1] weighted belief = brute-force Bayes on {positions} positions "
          f"({len(configs)} configs, random priors): OK")


def check_2_nothing_learned():
    """Before any game, every model plays exactly the shots BeliefAgent plays."""
    games = 0
    for config in (MINI, NO_REVEAL_4):
        for i, layout in enumerate(all_layouts(config)):
            reference = play_solo(BeliefAgent(), config, layout=layout, agent_seed=i).shots
            for model, smoothing in LEARNERS:
                agent = OpponentBeliefAgent(model, smoothing)
                assert play_solo(agent, config, layout=layout, agent_seed=i).shots == reference
                games += 1
    print(f"[2] nothing learned: identical shots to BeliefAgent in {games} games "
          f"({len(LEARNERS)} learners, both reveal rules): OK")


def check_3_rules_and_public_only():
    """After learning, every game is legal and reproducible, decisions use only
    public results, and evaluation does not learn."""
    habit = standard_habits(MINI)[1]   # edges
    rng = random.Random(0)
    for model, smoothing in LEARNERS:
        agent = OpponentBeliefAgent(model, smoothing)
        agent.training = True
        for g in range(50):
            play_solo(agent, MINI, layout=habit.sample(MINI, rng), agent_seed=TRAIN_SEED + g)
        agent.training = False
        counts = dict(agent.counts)
        records = [play_solo(agent, MINI, layout=layout, agent_seed=0)
                   for layout in all_layouts(MINI)]
        assert all(check_record(record) for record in records)
        again = [play_solo(agent, MINI, layout=layout, agent_seed=0)
                 for layout in all_layouts(MINI)]
        assert [r.shots for r in records] == [r.shots for r in again], "rerun differs"
        assert dict(agent.counts) == counts and agent.games_learned == 50
        pairs = check_public_only(records)
    print(f"[3] after 50 games vs edges: 264 legal, reproducible games per learner; "
          f"decisions depend only on public results ({pairs} pairs); evaluation "
          "does not learn: OK")


def check_4_samplers(draws=20_000):
    """Each habit's sampler (used to play against it) draws layouts with the
    probabilities its distribution states (chi-squared, sparse cells pooled)."""
    rows = []
    for habit in standard_habits(MINI):
        index = {frozenset(s.cells for s in layout): i
                 for i, layout in enumerate(Belief.prior(MINI).layouts)}
        p = habit.distribution(MINI)
        counts = [0] * len(p)
        rng = random.Random(2)
        for _ in range(draws):
            counts[index[frozenset(s.cells for s in habit.sample(MINI, rng))]] += 1
        assert all(c == 0 for c, q in zip(counts, p) if q == 0), "sampled an impossible layout"
        cells, pooled_obs, pooled_exp = [], 0, 0.0
        for c, q in zip(counts, p):
            if q * draws >= 5:
                cells.append((c, q * draws))
            elif q:
                pooled_obs, pooled_exp = pooled_obs + c, pooled_exp + q * draws
        if pooled_exp:
            cells.append((pooled_obs, pooled_exp))
        chi2 = sum((o - e) ** 2 / e for o, e in cells)
        k = len(cells) - 1
        assert chi2 < k + 6 * (2 * k) ** 0.5, f"{habit.name} sampler is off (chi2 {chi2:.0f})"
        rows.append(f"{habit.name} {chi2:.0f}/{k}")
    print(f"[4] samplers match their distributions, chi2/dof over {draws} draws: "
          + ", ".join(rows) + ": OK")


def learning_curve(task):
    habit_index, model, smoothing, run = task
    habit = standard_habits(MINI)[habit_index]
    rng = random.Random(1000 * habit_index + run)
    agent = OpponentBeliefAgent(model, smoothing)
    agent.training = True
    values, played = [], 0
    for checkpoint in CHECKPOINTS:
        while played < checkpoint:
            play_solo(agent, MINI, layout=habit.sample(MINI, rng),
                      agent_seed=TRAIN_SEED + 10_000 * run + played)
            played += 1
        values.append(expected_shots(agent, MINI, habit, SEEDS))   # frozen while measured
    return task, values


def optimum(habit_index):
    """V* against the habit: the exact planner with the habit as its prior."""
    plan = Planner(MINI, standard_habits(MINI)[habit_index].distribution(MINI))
    return plan.optimal_value(plan.everything, 0)


def check_5_benchmark(runs):
    """Exact expected shots against each habit: fixed agents, greedy with the
    true habit as prior (oracle), the true optimum, and learning curves (mean
    over independent runs)."""
    habits = standard_habits(MINI)
    tasks = [(h, model, smoothing, run) for h in range(len(habits))
             for model, smoothing in LEARNERS for run in range(runs)]
    with ProcessPoolExecutor() as pool:
        optima = [pool.submit(optimum, h) for h in range(len(habits))]
        curves = {}
        for (h, model, smoothing, _), values in pool.map(learning_curve, tasks):
            curves.setdefault((h, model, smoothing), []).append(values)
        optima = [future.result() for future in optima]
    print(f"[5] 4x4, exact expected shots against each opponent ({runs} learning runs "
          f"per curve, ties averaged over {len(SEEDS)} seeds)")
    print(f"    {'opponent':<12} {'density':>8} {'uniform':>8} {'unif-opt':>8} "
          f"{'oracle':>8} {'optimum':>8}")
    fixed = {}
    for h, habit in enumerate(habits):
        fixed[h] = [expected_shots(agent, MINI, habit, SEEDS) for agent in
                    (ProbabilityDensityAgent(), BeliefAgent(), BeliefAgent(planner="optimal"),
                     OpponentBeliefAgent(habit=habit))] + [optima[h]]
        assert optima[h] <= min(fixed[h]) + 1e-9, "an agent beat the optimum"
        print(f"    {habit.name:<12} " + " ".join(f"{v:8.3f}" for v in fixed[h]))
    print()
    print(f"    learning curves, games seen: " + " ".join(f"{c:>6}" for c in CHECKPOINTS)
          + "   saving at 1000")
    for h, habit in enumerate(habits):
        uniform_shots, best = fixed[h][1], fixed[h][4]
        for model, smoothing in LEARNERS:
            means = [statistics.fmean(col) for col in zip(*curves[h, model, smoothing])]
            gap = uniform_shots - best
            share = f" ({(uniform_shots - means[-1]) / gap:4.0%} of possible)" if gap > 0.2 else ""
            print(f"    {habit.name:<12} {model:<10} s={smoothing}  "
                  + " ".join(f"{m:6.3f}" for m in means)
                  + f"   {uniform_shots - means[-1]:+.3f}{share}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=8, help="learning runs per curve")
    args = parser.parse_args()
    check_1_weighted_bayes()
    check_2_nothing_learned()
    check_3_rules_and_public_only()
    check_4_samplers()
    check_5_benchmark(args.runs)
