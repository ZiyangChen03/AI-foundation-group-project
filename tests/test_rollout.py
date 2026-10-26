import random

import pytest

from battleship import CLASSIC, MINI, BattleshipEnv, GameConfig, Result, play_solo
from battleship.agents import BeliefAgent, ProbabilityDensityAgent
from battleship.belief import observed_outcome
from battleship.particles import Constraints, ParticleBelief
from battleship.rollout import density_from_constraints, rollout_choice, simulate_particles

NO_REVEAL = GameConfig(10, (5, 4, 3, 3, 2), reveal_sunk_cells=False)


@pytest.mark.parametrize("config", [CLASSIC, NO_REVEAL, MINI,
                                    GameConfig(6, (3, 2, 2, 1), reveal_sunk_cells=False)])
def test_density_simulator_plays_like_the_density_agent(config):
    """With the same rng, the incremental scores pick exactly the shots the
    real ProbabilityDensityAgent picks."""
    for seed in range(10):
        record = play_solo(ProbabilityDensityAgent(), config, seed=seed)
        rng = random.Random(f"agent:{seed}")     # the runner's agent seed
        env = BattleshipEnv(config)
        obs = env.reset(seed=seed)
        density = density_from_constraints(Constraints(config))
        for shot in record.shots:
            assert density.best(rng, config.num_cells) == shot.cell
            after = env.step(shot.cell).observation
            result, length, _ = observed_outcome(obs, after)
            ship = next((sum(1 << c for c in s.cells) for s in env._layout
                         if shot.cell in s.cells), None)
            density.shoot(shot.cell, ship is not None,
                          ship if result is Result.SUNK else None, length)
            obs = after


def test_simulations_finish_the_world():
    belief = ParticleBelief(CLASSIC, random.Random(0), num_particles=50)
    masks = belief.constraints.placements.masks
    layouts = [tuple(masks[p] for p in state) for state in belief.particles]
    for world in range(10):
        shots = simulate_particles(layouts, world, 0, 0, True, 100, random.Random(world))
        assert 17 <= shots <= 100


def test_rollout_picks_the_cheapest_candidate_reproducibly():
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=2)
    belief = ParticleBelief(CLASSIC, random.Random(0), num_particles=100)
    for cell in (44, 45, 54):
        after = env.step(cell).observation
        belief.update(cell, observed_outcome(obs, after))
        obs = after
    worlds = list(range(20))
    candidates = [cell for cell in obs.legal_actions][:5]
    for base in ("particles", "density"):
        first = rollout_choice(belief.constraints, belief.particles, worlds, candidates,
                               random.Random(9), base)
        again = rollout_choice(belief.constraints, belief.particles, worlds, candidates,
                               random.Random(9), base)
        assert first == again
        choice, means = first
        assert set(means) == set(candidates)
        assert means[choice] == min(means.values())


@pytest.mark.parametrize("config", [CLASSIC, NO_REVEAL])
def test_rollout_agent_finishes_games(config):
    agent = BeliefAgent(num_particles=100, planner="rollout", rollout_worlds=8)
    for seed in range(2):
        record = play_solo(agent, config, seed=seed)
        assert sum(config.fleet) <= record.num_shots <= config.num_cells
