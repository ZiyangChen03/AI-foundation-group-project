"""The 10x10 versions: particle information gain, approximate optimal, and the
two 10x10 Q-tables (battleship/agents/belief_10x10.py and q_10x10.py)."""
import math
import random

import pytest

from battleship import CLASSIC, MINI, BattleshipEnv, GameConfig, all_layouts, play_solo
from battleship.agents.belief_10x10 import (LargeBoardBeliefAgent, particle_info_gain,
                                            particle_result_counts)
from battleship.agents.q_10x10 import BLOCKED, LocalQAgent, ParticleBeliefQAgent
from battleship.belief import Belief, observed_outcome
from battleship.particles import ParticleBelief
from battleship.planning import planner
from battleship.runner import train
from helpers import assert_public_only


def test_particle_information_gain_matches_the_exact_value_on_4x4():
    plan = planner(MINI)
    worst = 0.0
    for seed in range(3):
        env = BattleshipEnv(MINI)
        obs = env.reset(seed=seed)
        rng = random.Random(seed)
        exact = Belief.prior(MINI)
        sampled = ParticleBelief(MINI, random.Random(100 + seed), 3000)
        while not env.done:
            layouts, fired = plan.state(exact)
            estimate = particle_info_gain(sampled, obs.legal_actions)
            assert set(estimate) <= set(plan.hit_counts(layouts, fired))
            for cell in plan.hit_counts(layouts, fired):
                true = math.log2(layouts.bit_count()) - plan.expected_entropy(layouts, fired, cell)
                worst = max(worst, abs(estimate.get(cell, 0.0) - true))
            cell = rng.choice(obs.legal_actions)
            before, obs = obs, env.step(cell).observation
            outcome = observed_outcome(before, obs)
            exact = exact.update(cell, outcome)
            sampled.update(cell, outcome)
    assert worst < 0.1


def test_result_counts_add_up_to_the_particles():
    belief = ParticleBelief(CLASSIC, random.Random(0), 200)
    hits, sinks = particle_result_counts(belief)
    for cell in range(CLASSIC.num_cells):
        assert hits[cell] + sum(sinks[cell].values()) <= len(belief.particles)
    assert not any(sinks.values())       # nothing is one shot from sinking on an empty board


@pytest.mark.parametrize("planner_name", ["infogain", "greedy-info", "optimal-approx"])
def test_large_board_planners_play_10x10_from_public_information(planner_name):
    agent = LargeBoardBeliefAgent(planner_name, num_particles=200, rollout_worlds=8,
                                  rollout_particles=100)
    records = [play_solo(agent, CLASSIC, seed=s, agent_seed=0) for s in (3, 4)]
    for record in records:
        assert sum(CLASSIC.fleet) <= record.num_shots <= CLASSIC.num_cells
    assert_public_only(records)
    assert agent.name.startswith(f"belief-{planner_name}")


def test_greedy_info_only_breaks_ties_between_the_most_likely_cells():
    agent = LargeBoardBeliefAgent("greedy-info", num_particles=300)
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=2)
    agent.new_game(CLASSIC, random.Random(0))
    for _ in range(15):
        cell = agent.select_shot(obs)
        probabilities = agent.belief.occupancy()
        assert probabilities[cell] == max(probabilities[c] for c in obs.legal_actions)
        result = env.step(cell)
        agent.observe(obs, cell, result)
        obs = result.observation
        if env.done:
            break


def test_unknown_planner_is_refused():
    with pytest.raises(ValueError):
        LargeBoardBeliefAgent("optimal")          # the exact one is BeliefAgent's, on 4x4


def test_local_keys_are_the_neighbourhood_and_the_checkerboard_colour():
    env = BattleshipEnv(CLASSIC)
    obs = env.reset(seed=0)
    keys = LocalQAgent().keys(obs)
    assert len(keys) == CLASSIC.num_cells
    corner = keys[0]                                   # A1: five neighbours are off the board
    assert len(corner) == 9 and corner.count(BLOCKED) == 5 and corner[-1] == 0
    assert keys[1][-1] == 1


def test_local_q_learns_and_round_trips(tmp_path):
    agent = LocalQAgent()
    untrained = [play_solo(agent, CLASSIC, seed=s).num_shots for s in range(20)]
    train(agent, CLASSIC, 1500)
    trained = [play_solo(agent, CLASSIC, seed=s).num_shots for s in range(20)]
    assert sum(trained) < sum(untrained)
    agent.save(tmp_path / "q.json.gz")
    again = LocalQAgent.load(tmp_path / "q.json.gz")
    assert again.table.q == agent.table.q
    assert ([play_solo(again, CLASSIC, seed=s, agent_seed=1).num_shots for s in range(5)]
            == [play_solo(agent, CLASSIC, seed=s, agent_seed=1).num_shots for s in range(5)])


def test_particle_belief_q_keys_trains_and_round_trips(tmp_path):
    agent = ParticleBeliefQAgent(num_particles=80)
    train(agent, CLASSIC, 3)
    assert 0 < agent.num_keys() <= 2 * agent.bins
    assert all(0 <= k[0] < agent.bins and k[1] in (0, 1) for k in agent.table.q)
    agent.save(tmp_path / "b.json.gz")
    again = ParticleBeliefQAgent.load(tmp_path / "b.json.gz")
    assert again.table.q == agent.table.q and again.bins == agent.bins
    record = play_solo(again, CLASSIC, seed=9)
    assert sum(CLASSIC.fleet) <= record.num_shots <= CLASSIC.num_cells


def test_q_agents_also_play_4x4():
    for agent in (LocalQAgent(), ParticleBeliefQAgent(num_particles=80)):
        for layout in list(all_layouts(MINI))[:5]:
            assert play_solo(agent, MINI, layout=layout).num_shots <= MINI.num_cells


def test_originals_are_unchanged_and_still_4x4_only():
    from battleship.agents import BeliefAgent
    from battleship.agents.belief_q import BeliefQAgent
    from battleship.q_learning.Q_learning import QLearningAgent
    with pytest.raises(ValueError):
        BeliefAgent(belief="particles", planner="optimal")
    with pytest.raises(ValueError):
        BeliefQAgent(config=CLASSIC)
    with pytest.raises(ValueError):
        QLearningAgent().new_game(CLASSIC, random.Random(0))
    assert GameConfig(10, (5, 4, 3, 3, 2)) == CLASSIC
