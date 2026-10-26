import random

import pytest

from battleship import (CLASSIC, MINI, BattleshipEnv, GameConfig, Result, all_layouts,
                        layout_from_cells)
from battleship.belief import Belief, explains, observed_outcome

CONFIGS = [MINI, GameConfig(4, (3, 2), reveal_sunk_cells=False), GameConfig(4, (2, 2, 1))]


def random_games(config, games=40):
    """Yield (before, after, cell) for every shot of games fired at random."""
    rng = random.Random(0)
    for seed in range(games):
        env = BattleshipEnv(config)
        obs = env.reset(seed=seed)
        while not env.done:
            cell = rng.choice(obs.legal_actions)
            after = env.step(cell).observation
            yield obs, after, cell, env
            obs = after


def cells_of(layout):
    """A layout as a set of ships, ignoring the order of equal-length ships."""
    return {ship.cells for ship in layout}


@pytest.mark.parametrize("config", CONFIGS)
def test_prior_is_every_layout(config):
    belief = Belief.prior(config)
    assert len(belief) == sum(1 for _ in all_layouts(config))
    # every layout covers sum(fleet) cells, so the probabilities add up to that
    assert sum(belief.occupancy()) == pytest.approx(sum(config.fleet))


@pytest.mark.parametrize("config", CONFIGS)
def test_update_matches_brute_force(config):
    """The incremental Bayes update keeps exactly the layouts that explain the
    history, and the true layout is always among them.

    With revealed sunk cells and two ships of the same length, a single
    Observation cannot tell which cells each sinking revealed, so the replay
    may keep extra layouts; the shot-by-shot belief must then be a subset."""
    layouts = list(all_layouts(config))
    ambiguous = config.reveal_sunk_cells and len(set(config.fleet)) < len(config.fleet)
    belief = None
    for before, after, cell, env in random_games(config):
        if not before.history:
            belief = Belief.prior(config)
        belief = belief.update(cell, observed_outcome(before, after))
        expected = [layout for layout in layouts if explains(layout, after)]
        assert cells_of(env._layout) in [cells_of(layout) for layout in belief.layouts]
        if ambiguous:
            assert set(belief.support) <= set(Belief.from_observation(after).support)
        else:
            assert belief.layouts == expected
            assert Belief.from_observation(after).support == belief.support


@pytest.mark.parametrize("config", CONFIGS)
def test_outcomes_partition_the_belief(config):
    """P(o | b, a) over the possible outcomes sums to 1, and each branch is
    the belief the Bayes update would give for that outcome."""
    belief = Belief.prior(config)
    for cell in range(config.num_cells):
        branches = belief.outcomes(cell)
        assert sum(len(b) for b in branches.values()) == len(belief)
        for outcome, branch in branches.items():
            assert branch.support == belief.update(cell, outcome).support


def test_occupancy_after_a_miss():
    env = BattleshipEnv(MINI)
    obs = env.reset(layout=layout_from_cells(MINI, [[0, 1, 2], [8, 12]]))
    after = env.step(5).observation          # miss at B2
    belief = Belief.prior(MINI).update(5, observed_outcome(obs, after))
    p = belief.occupancy()
    assert p[5] == 0
    assert len(belief) == sum(1 for layout in all_layouts(MINI)
                              if all(5 not in ship.cells for ship in layout))
    count_a1 = sum(1 for layout in belief.layouts if any(0 in s.cells for s in layout))
    assert p[0] == pytest.approx(count_a1 / len(belief))


def test_sunk_outcome_reveals_cells_only_when_rules_say_so():
    layout = layout_from_cells(MINI, [[0, 1, 2], [8, 12]])
    for config, revealed in ((MINI, frozenset({8, 12})),
                             (GameConfig(4, (3, 2), reveal_sunk_cells=False), None)):
        env = BattleshipEnv(config)
        env.reset(layout=layout)
        before = env.step(8).observation
        after = env.step(12).observation
        assert observed_outcome(before, after) == (Result.SUNK, 2, revealed)


def test_explains_rejects_a_layout_that_disagrees():
    env = BattleshipEnv(MINI)
    env.reset(layout=layout_from_cells(MINI, [[0, 1, 2], [8, 12]]))
    obs = env.step(0).observation            # hit at A1
    assert explains(layout_from_cells(MINI, [[0, 1, 2], [8, 12]]), obs)
    assert not explains(layout_from_cells(MINI, [[4, 5, 6], [8, 12]]), obs)


def test_impossible_outcome_raises():
    belief = Belief.prior(MINI)
    with pytest.raises(ValueError):
        belief.update(0, (Result.SUNK, 3, frozenset({0, 1, 2})))   # nothing sinks in one shot


def test_large_boards_are_refused():
    with pytest.raises(ValueError, match="small boards"):
        Belief.prior(CLASSIC)
