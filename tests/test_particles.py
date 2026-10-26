import random
from collections import Counter

import pytest

from battleship import (CLASSIC, MINI, BattleshipEnv, GameConfig, Result, all_layouts,
                        layout_from_cells)
from battleship.belief import Belief, observed_outcome
from battleship.particles import Constraints, ParticleBelief

# Small boards where the exact belief can check the sampled one, including
# ships of equal length and both reveal rules.
SMALL = [MINI, GameConfig(4, (3, 2), reveal_sunk_cells=False),
         GameConfig(4, (2, 2, 1)), GameConfig(4, (2, 2, 1), reveal_sunk_cells=False),
         GameConfig(5, (3, 2, 2)), GameConfig(5, (3, 2, 2), reveal_sunk_cells=False)]


def ids(constraints, layout):
    by_mask = constraints.placements.by_mask
    return tuple(by_mask[sum(1 << cell for cell in ship.cells)] for ship in layout)


def random_shots(config, games, seed=0):
    """Yield (env, before, after, cell) for games fired at random."""
    rng = random.Random(seed)
    for game in range(games):
        env = BattleshipEnv(config)
        obs = env.reset(seed=game)
        while not env.done:
            cell = rng.choice(obs.legal_actions)
            after = env.step(cell).observation
            yield env, obs, after, cell
            obs = after


@pytest.mark.parametrize("config", SMALL)
def test_constraints_match_the_exact_belief(config):
    """The single-ship checks accept exactly the layouts the exact belief keeps."""
    layouts = list(all_layouts(config))
    constraints = exact = None
    for env, before, after, cell in random_shots(config, games=6):
        if not before.history:
            constraints, exact = Constraints(config), Belief.prior(config)
        outcome = observed_outcome(before, after)
        constraints.add(cell, outcome)
        exact = exact.update(cell, outcome)
        kept = [i for i, layout in enumerate(layouts)
                if constraints.explains(ids(constraints, layout))]
        assert tuple(kept) == exact.support
        # incremental bookkeeping agrees with checking every placement afresh
        assert all(constraints.valid[p] == constraints.check(p)
                   for p in range(len(constraints.valid)))


@pytest.mark.parametrize("config", SMALL)
def test_particles_always_explain_the_history(config):
    belief = None
    for env, before, after, cell in random_shots(config, games=8):
        if not before.history:
            belief = ParticleBelief(config, random.Random(len(after.history)), num_particles=60)
        belief.update(cell, observed_outcome(before, after))
        assert all(belief.constraints.explains(state) for state in belief.particles)
        assert belief.constraints.explains(ids(belief.constraints, env._layout))
        assert sum(belief.occupancy()) == pytest.approx(
            sum(config.fleet) - bin(belief.constraints.hits).count("1"))


def chain_counts(belief, moves, pair_moves):
    state = list(belief.particles[0])
    masks = belief.constraints.placements.masks
    counts = Counter()
    for step in range(moves):
        belief._sweep(state)
        if pair_moves:
            i, j = belief.rng.sample(range(len(state)), 2)
            belief._pair_move(state, i, j)
        if step % 3 == 0:
            counts[frozenset(masks[p] for p in state)] += 1
    return counts


def test_pair_moves_reach_every_layout():
    """Hits at B2 and B3, both unsunk. Starting from "2-ship on B2, 3-ship on
    B3", single-ship moves can never reach "3-ship covers both"; pair moves
    must, and must visit every consistent layout about equally often."""
    env = BattleshipEnv(MINI)
    obs = env.reset(layout=layout_from_cells(MINI, [[5, 9, 13], [0, 4]]))
    exact, belief = Belief.prior(MINI), ParticleBelief(MINI, random.Random(1), 20)
    for cell in (5, 9):
        after = env.step(cell).observation
        outcome = observed_outcome(obs, after)
        exact, obs = exact.update(cell, outcome), after
        belief.update(cell, outcome)
    # start the chain in the trapped part of the space
    belief.particles[0] = list(ids(belief.constraints, layout_from_cells(MINI, [[9, 10, 11], [4, 5]])))
    support = {frozenset(sum(1 << c for c in ship.cells) for ship in layout)
               for layout in exact.layouts}
    single = chain_counts(belief, 3000, pair_moves=False)
    assert set(single) < support                       # stuck: some layouts unreachable
    counts = chain_counts(belief, 30000, pair_moves=True)
    assert set(counts) == support
    expected = sum(counts.values()) / len(support)
    chi2 = sum((counts[key] - expected) ** 2 / expected for key in support)
    k = len(support)
    assert chi2 < k - 1 + 6 * (2 * (k - 1)) ** 0.5


@pytest.mark.parametrize("config", [MINI, GameConfig(5, (3, 2, 2), reveal_sunk_cells=False)])
def test_estimate_converges_to_the_exact_probabilities(config):
    """Unbiased: the average signed error is ~0, and the worst-cell error is
    small with enough particles."""
    worst, signed, count = [], 0.0, 0
    belief = exact = None
    for env, before, after, cell in random_shots(config, games=6, seed=3):
        if not before.history:
            belief = ParticleBelief(config, random.Random(len(worst)), num_particles=1600)
            exact = Belief.prior(config)
        outcome = observed_outcome(before, after)
        belief.update(cell, outcome)
        exact = exact.update(cell, outcome)
        if env.done:
            continue
        pe, pp = exact.occupancy(), belief.occupancy()
        worst.append(max(abs(a - b) for a, b in zip(pe, pp)))
        for a, b in zip(pe, pp):
            signed += b - a
            count += 1
    assert sum(worst) / len(worst) < 0.02
    assert max(worst) < 0.06
    assert abs(signed / count) < 0.002


def test_search_recovers_when_every_particle_dies():
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=4)
    belief = ParticleBelief(CLASSIC, random.Random(0), num_particles=5)
    rng = random.Random(1)
    while not env.done:
        cell = rng.choice(obs.legal_actions)
        after = env.step(cell).observation
        belief.update(cell, observed_outcome(obs, after))
        assert all(belief.constraints.explains(state) for state in belief.particles)
        obs = after
    assert belief.searches > 0          # with 5 particles they die often


def test_impossible_history_raises():
    belief = ParticleBelief(MINI, random.Random(0), num_particles=10)
    with pytest.raises(RuntimeError):
        belief.update(0, (Result.SUNK, 3, frozenset({0, 1, 2})))   # nothing sinks in one shot
