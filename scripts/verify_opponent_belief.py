"""Verify the opponent-belief agent and benchmark it against placement habits.

    uv run scripts/verify_opponent_belief.py            # about 2 minutes
    uv run scripts/verify_opponent_belief.py --runs 20  # tighter learning curves
    uv run scripts/verify_opponent_belief.py --classic 200   # plus 10x10 (~15 min)

Games are played by the runner (play_solo); checks read the records it returns
and compare against independent references. 4x4 results are exact expectations
over every layout, weighted by the opponent's habit (see
battleship/opponents.py), with ties averaged over agent seeds. 10x10 results
are paired: every agent plays the same layouts drawn from the habit.
"""

import argparse
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from math import prod

from battleship import CLASSIC, MINI, BattleshipEnv, GameConfig, all_layouts, play_solo
from battleship.agents import BeliefAgent, OpponentBeliefAgent, ProbabilityDensityAgent
from battleship.belief import Belief, explains, observed_outcome
from battleship.opponents import expected_shots, large_board_habits, standard_habits
from battleship.particles import ParticleBelief
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


def check_6_weighted_particles():
    """Weighted particles (for 10x10) against the exact weighted belief on 5x5:
    the error halves when the particles quadruple and the bias is ~0."""
    config = GameConfig(5, (3, 2, 2), reveal_sunk_cells=False)
    factor_rng, factors = random.Random(9), {}

    def weight(length, cells):          # a strong, uneven prior that splits per ship
        return factors.setdefault((length, cells), factor_rng.choice([1, 2, 5, 20]))

    weights = [prod(weight(s.length, s.cells) for s in layout)
               for layout in Belief.prior(config).layouts]
    rows = []
    for n in (100, 400, 1600):
        worst, signed, cells = [], 0.0, 0
        rng = random.Random(7)
        for game in range(8):
            env = BattleshipEnv(config)
            obs = env.reset(seed=game)
            particles = ParticleBelief(config, random.Random(game), n, placement_weight=weight)
            exact = Belief.prior(config, weights)
            while True:
                pe, pp = exact.occupancy(), particles.occupancy()
                worst.append(max(abs(a - b) for a, b in zip(pe, pp)))
                signed += sum(b - a for a, b in zip(pe, pp))
                cells += len(pe)
                cell = rng.choice(obs.legal_actions)
                after = env.step(cell).observation
                if env.done:
                    break
                outcome = observed_outcome(obs, after)
                particles.update(cell, outcome)
                exact = exact.update(cell, outcome)
                obs = after
        rows.append((n, statistics.fmean(worst), signed / cells))
    for (n1, e1, _), (n2, e2, _) in zip(rows, rows[1:]):
        assert 0.35 < e2 / e1 < 0.7, f"error does not shrink like noise from {n1} to {n2}"
    assert all(abs(bias) < 0.002 for _, _, bias in rows)
    print("[6] weighted particles vs exact weighted belief (5x5, ships 3,2,2, no reveal): "
          + ", ".join(f"N={n}: worst-cell error {e:.4f}, bias {b:+.5f}" for n, e, b in rows))


CLASSIC_AGENTS = ("density", "uniform", "learned 10", "learned 100", "learned 1000", "oracle")


def classic_agent(name, habit, habit_index):
    if name == "density":
        return ProbabilityDensityAgent()
    if name == "uniform":
        return BeliefAgent()
    if name == "oracle":
        return OpponentBeliefAgent(habit=habit)
    agent = OpponentBeliefAgent()
    rng = random.Random(f"learn:{habit_index}")
    for _ in range(int(name.split()[1])):
        agent.learn(habit.sample(CLASSIC, rng))   # what end_game does after a game
    return agent


def classic_games(task):
    habit_index, name, start, stop = task
    habit = large_board_habits()[habit_index]
    agent = classic_agent(name, habit, habit_index)
    shots = []
    for i in range(start, stop):
        layout = habit.sample(CLASSIC, random.Random(f"board:{habit_index}:{i}"))
        record = play_solo(agent, CLASSIC, layout=layout, agent_seed=i)
        assert check_record(record)
        shots.append(record.num_shots)
    return task, shots


def check_7_classic(games, chunk=10):
    """10x10: every agent plays the same layouts drawn from each habit.
    Learned agents learn from separate boards of the habit."""
    habits = large_board_habits()
    tasks = [(h, name, start, min(start + chunk, games)) for h in range(len(habits))
             for name in CLASSIC_AGENTS for start in range(0, games, chunk)]
    shots = {}
    with ProcessPoolExecutor() as pool:
        for (h, name, start, _), values in pool.map(classic_games, tasks):
            shots.setdefault((h, name), {})[start] = values
    shots = {key: [s for start in sorted(parts) for s in parts[start]]
             for key, parts in shots.items()}
    print(f"[7] 10x10, {games} layouts per opponent, the same for every agent "
          "(1000 particles); mean shots")
    print(f"    {'opponent':<11}" + "".join(f"{name:>13}" for name in CLASSIC_AGENTS)
          + "   learned 1000 - uniform    oracle - uniform")
    for h, habit in enumerate(habits):
        means = [statistics.fmean(shots[h, name]) for name in CLASSIC_AGENTS]

        def paired(name):
            diff = [a - b for a, b in zip(shots[h, name], shots[h, "uniform"])]
            half = 1.96 * statistics.stdev(diff) / len(diff) ** 0.5
            return f"{statistics.fmean(diff):+6.2f} ± {half:4.2f}"

        print(f"    {habit.name:<11}" + "".join(f"{m:13.2f}" for m in means)
              + f"   {paired('learned 1000'):>20}   {paired('oracle'):>17}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=8, help="learning runs per curve")
    parser.add_argument("--classic", type=int, default=0, metavar="N",
                        help="also run N 10x10 games per opponent and agent (slow)")
    args = parser.parse_args()
    check_1_weighted_bayes()
    check_2_nothing_learned()
    check_3_rules_and_public_only()
    check_4_samplers()
    check_5_benchmark(args.runs)
    check_6_weighted_particles()
    if args.classic:
        check_7_classic(args.classic)
