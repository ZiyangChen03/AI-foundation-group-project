import random

import pytest

from battleship import MINI, GameConfig, all_layouts, evaluate, play_solo
from battleship.agents import BeliefAgent, OpponentBeliefAgent
from battleship.belief import Belief
from battleship.opponents import (FeatureHabit, border_cells, expected_shots, favorites,
                                  uniform)
from helpers import assert_public_only

NO_REVEAL = GameConfig(4, (3, 2), reveal_sunk_cells=False)
EDGES = FeatureHabit("edges", border_cells, 1.0)


def learn(agent, habit, games, seed=0):
    rng = random.Random(seed)
    agent.training = True
    for g in range(games):
        play_solo(agent, MINI, layout=habit.sample(MINI, rng), agent_seed=g)
    agent.training = False
    return agent


@pytest.mark.parametrize("config", [MINI, NO_REVEAL])
@pytest.mark.parametrize("model", ["placements", "layouts"])
def test_plays_like_belief_agent_before_learning(config, model):
    for i, layout in enumerate(list(all_layouts(config))[::3]):
        expected = play_solo(BeliefAgent(), config, layout=layout, agent_seed=i).shots
        agent = OpponentBeliefAgent(model, smoothing=2)
        assert play_solo(agent, config, layout=layout, agent_seed=i).shots == expected


def test_learns_only_while_training():
    agent = OpponentBeliefAgent()
    evaluate(agent, MINI, layouts=list(all_layouts(MINI))[:20])
    assert agent.games_learned == 0 and not agent.counts
    learn(agent, EDGES, 5)
    assert agent.games_learned == 5
    assert sum(agent.counts.values()) == 5 * len(MINI.fleet)


def test_placement_weights_are_products_of_counts():
    agent = OpponentBeliefAgent("placements", smoothing=1)
    layout = Belief.prior(MINI).layouts[7]
    agent.counts.update((ship.length, ship.cells) for ship in layout)
    weights = agent.prior_weights(MINI)
    assert weights[7] == 4                       # (1 + 1) * (1 + 1)
    assert min(weights) == 1 and max(weights) == 4


def test_layout_model_counts_whole_boards():
    habit = favorites(MINI, 2, seed=4)
    agent = learn(OpponentBeliefAgent("layouts", smoothing=1), habit, 10)
    boards = lambda layouts: {frozenset(s.cells for s in layout) for layout in layouts}
    seen = [layout for layout, w in zip(Belief.prior(MINI).layouts, agent.prior_weights(MINI))
            if w > 1]
    assert boards(seen) == boards(habit.layouts)
    assert sum(agent.counts.values()) == 10


def test_learning_an_edge_habit_saves_shots():
    before = expected_shots(OpponentBeliefAgent(), MINI, EDGES, agent_seeds=[0])
    after = expected_shots(learn(OpponentBeliefAgent(), EDGES, 200), MINI, EDGES,
                           agent_seeds=[0])
    oracle = expected_shots(OpponentBeliefAgent(habit=EDGES), MINI, EDGES, agent_seeds=[0])
    assert after < before - 0.8
    assert oracle < before - 0.8


def test_layout_model_memorizes_favorite_boards():
    habit = favorites(MINI, 3)
    agent = learn(OpponentBeliefAgent("layouts", smoothing=1), habit, 30)
    oracle = expected_shots(OpponentBeliefAgent(habit=habit), MINI, habit, agent_seeds=[0])
    assert expected_shots(agent, MINI, habit, agent_seeds=[0]) == pytest.approx(oracle, abs=0.2)


def test_decisions_depend_only_on_public_results_after_learning():
    agent = learn(OpponentBeliefAgent(), EDGES, 30)
    records = [play_solo(agent, MINI, layout=layout, agent_seed=0)
               for layout in all_layouts(MINI)]
    assert_public_only(records)


def test_oracle_uses_the_habit_and_does_not_learn():
    agent = learn(OpponentBeliefAgent(habit=EDGES), EDGES, 5)
    assert agent.games_learned == 0
    assert agent.prior_weights(MINI) == EDGES.distribution(MINI)


def test_forget_returns_to_the_uniform_prior():
    agent = learn(OpponentBeliefAgent(), EDGES, 10)
    agent.forget()
    assert len(set(agent.prior_weights(MINI))) == 1


def test_rejects_bad_options():
    with pytest.raises(ValueError):
        OpponentBeliefAgent(model="neural")
    with pytest.raises(ValueError):
        OpponentBeliefAgent(smoothing=0)


def test_uses_weighted_particles_on_the_classic_board():
    from battleship import CLASSIC
    from battleship.particles import ParticleBelief
    agent = OpponentBeliefAgent(num_particles=100)
    agent.learn(EDGES.sample(CLASSIC, random.Random(0)))
    record = play_solo(agent, CLASSIC, seed=1)
    assert isinstance(agent.belief, ParticleBelief) and agent.belief.weight is not None
    assert record.num_shots <= CLASSIC.num_cells
    ship = next(iter(agent.counts))
    assert agent.placement_weight(*ship) == 1 + agent.smoothing


def test_layout_model_needs_the_exact_belief():
    from battleship import CLASSIC
    with pytest.raises(ValueError):
        play_solo(OpponentBeliefAgent(model="layouts", num_particles=50), CLASSIC, seed=0)


def test_learn_counts_like_end_game():
    a, b = OpponentBeliefAgent(), OpponentBeliefAgent()
    rng = random.Random(3)
    b.training = True
    for g in range(5):
        layout = EDGES.sample(MINI, rng)
        a.learn(layout)
        play_solo(b, MINI, layout=layout, agent_seed=g)
    assert a.counts == b.counts and a.games_learned == b.games_learned == 5
