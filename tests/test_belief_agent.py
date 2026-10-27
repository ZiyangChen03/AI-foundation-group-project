import random

import pytest

from battleship import (CLASSIC, MINI, BattleshipEnv, GameConfig, all_layouts, evaluate,
                        play_solo)
from battleship.agents import BeliefAgent, ProbabilityDensityAgent
from battleship.belief import Belief
from battleship.particles import ParticleBelief
from helpers import assert_public_only

NO_REVEAL = GameConfig(4, (3, 2), reveal_sunk_cells=False)


@pytest.mark.parametrize("config", [MINI, NO_REVEAL])
def test_finishes_every_layout(config):
    report = evaluate(BeliefAgent(), config, layouts=all_layouts(config))
    assert report.games == 264
    assert sum(config.fleet) <= report.best and report.worst <= config.num_cells


@pytest.mark.parametrize("config", [MINI, NO_REVEAL])
def test_always_fires_at_a_most_likely_cell(config):
    """Every shot is at a cell with the highest P(ship) under the exact belief,
    so a certain hit is always taken when one exists."""
    for layout in list(all_layouts(config))[::4]:
        record = play_solo(BeliefAgent(), config, layout=layout, agent_seed=0)
        env = BattleshipEnv(config)
        obs = env.reset(layout=layout)
        for shot in record.shots:
            p = Belief.from_observation(obs).occupancy()
            assert p[shot.cell] == max(p[cell] for cell in obs.legal_actions)
            obs = env.step(shot.cell).observation


def test_rebuilds_belief_from_unobserved_shots():
    env = BattleshipEnv(MINI)
    obs = env.reset(seed=3)
    for cell in (5, 6, 9):
        obs = env.step(cell).observation
    agent = BeliefAgent()
    agent.new_game(MINI, random.Random(0))
    agent.select_shot(obs)                    # never saw the three shots
    assert agent.belief.support == Belief.from_observation(obs).support


def test_decisions_depend_only_on_public_results():
    records = [play_solo(BeliefAgent(), MINI, layout=layout, agent_seed=0)
               for layout in all_layouts(MINI)]
    assert_public_only(records)


def test_compared_with_probability_density():
    """Over 20 tie-break seeds: 8.91 vs 8.92 shots with revealed sunk cells (a
    tie), 8.91 vs 10.13 without (the exact belief knows which hits are spent)."""
    def mean(agent, config):
        return evaluate(agent, config, layouts=all_layouts(config)).mean

    assert abs(mean(BeliefAgent(), MINI) - mean(ProbabilityDensityAgent(), MINI)) < 0.2
    assert mean(BeliefAgent(), NO_REVEAL) < mean(ProbabilityDensityAgent(), NO_REVEAL) - 0.8


def test_exact_belief_refuses_large_boards():
    with pytest.raises(ValueError, match="small boards"):
        play_solo(BeliefAgent(belief="exact"), CLASSIC, seed=0)


@pytest.mark.parametrize("config", [CLASSIC, GameConfig(10, (5, 4, 3, 3, 2),
                                                        reveal_sunk_cells=False)])
def test_uses_particles_on_large_boards(config):
    agent = BeliefAgent(num_particles=100)
    for seed in range(3):
        record = play_solo(agent, config, seed=seed)
        assert isinstance(agent.belief, ParticleBelief)
        assert sum(config.fleet) <= record.num_shots <= config.num_cells


def test_particles_must_observe_every_shot():
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=0)
    agent = BeliefAgent(num_particles=20)
    agent.new_game(CLASSIC, random.Random(0))
    obs = env.step(0).observation              # a shot the agent never observed
    with pytest.raises(RuntimeError, match="observe every shot"):
        agent.select_shot(obs)


def test_exact_rebuild_refused_when_reveals_are_ambiguous():
    config = GameConfig(4, (2, 2, 1))
    env = BattleshipEnv(config)
    env.reset(seed=0)
    agent = BeliefAgent()
    agent.new_game(config, random.Random(0))
    obs = env.step(0).observation              # a shot the agent never observed
    with pytest.raises(RuntimeError, match="observe every shot"):
        agent.select_shot(obs)


def test_move_time_includes_the_belief_update():
    """The update happens inside select_shot, so the runner times it."""
    record = play_solo(BeliefAgent(num_particles=300), CLASSIC, seed=0)
    assert sum(record.move_times) / record.num_shots > 0.002   # density takes ~0.0003 s


def test_rollout_base_is_checked():
    with pytest.raises(ValueError, match="rollout_base"):
        BeliefAgent(planner="rollout", rollout_base="densty")


def test_particles_match_exact_belief_on_4x4():
    """Same games, same boards: the sampled belief plays about as well as the
    exact one."""
    exact = evaluate(BeliefAgent(belief="exact"), MINI, layouts=all_layouts(MINI)).mean
    sampled = evaluate(BeliefAgent(belief="particles", num_particles=300), MINI,
                       layouts=all_layouts(MINI)).mean
    assert abs(sampled - exact) < 0.25
