import math
import random

import pytest

from battleship import CLASSIC, MINI, all_layouts, layout_from_cells, play_solo
from battleship.agents import BeliefAgent
from battleship.belief import Belief
from battleship.opponents import (FeatureHabit, border_cells, expected_shots, favorites,
                                  horizontal_ships, standard_habits, touching_pairs, uniform)


def test_features_on_a_known_layout():
    # 3-ship along the top row (A1 B1 C1), 2-ship just below it (A2 B2).
    layout = layout_from_cells(MINI, [[0, 1, 2], [4, 5]])
    assert border_cells(layout, MINI) == 4      # A1 B1 C1 and A2
    assert horizontal_ships(layout, MINI) == 2
    assert touching_pairs(layout, MINI) == 1


def test_distribution_follows_the_weights():
    habit = FeatureHabit("edges", border_cells, 1.0)
    layouts = Belief.prior(MINI).layouts
    p = habit.distribution(MINI)
    assert math.isclose(sum(p), 1.0)
    i, j = 0, 100
    ratio = math.exp(border_cells(layouts[i], MINI) - border_cells(layouts[j], MINI))
    assert math.isclose(p[i] / p[j], ratio)


def test_uniform_habit_is_uniform():
    p = uniform().distribution(MINI)
    assert len(set(p)) == 1 and len(p) == 264


def test_max_weight_bounds_every_layout():
    for habit in standard_habits(MINI):
        bound = habit.max_weight(MINI)
        assert all(habit.weight(layout, MINI) <= bound for layout in all_layouts(MINI))


def test_samples_are_legal_and_follow_the_habit():
    rng = random.Random(0)
    edges = FeatureHabit("edges", border_cells, 2.0)
    center = FeatureHabit("center", border_cells, -2.0)
    mean = {h.name: sum(border_cells(h.sample(MINI, rng), MINI) for _ in range(500)) / 500
            for h in (edges, center)}
    assert mean["edges"] > mean["center"] + 1


def test_favorites_reuse_the_same_boards():
    habit = favorites(MINI, 3, seed=1)
    rng = random.Random(0)
    boards = {frozenset(s.cells for s in habit.sample(MINI, rng)) for _ in range(100)}
    assert len(boards) == 3
    assert sum(1 for p in habit.distribution(MINI) if p) == 3


def test_sampling_works_on_the_classic_board():
    layout = FeatureHabit("edges", border_cells, 0.3).sample(CLASSIC, random.Random(0))
    assert sorted(ship.length for ship in layout) == sorted(CLASSIC.fleet)


def test_expected_shots_against_uniform_is_the_plain_mean():
    shots = [play_solo(BeliefAgent(), MINI, layout=layout, agent_seed=0).num_shots
             for layout in all_layouts(MINI)]
    assert expected_shots(BeliefAgent(), MINI, uniform(), agent_seeds=[0]) == pytest.approx(
        sum(shots) / len(shots))


def test_expected_shots_only_plays_layouts_the_habit_uses():
    habit = favorites(MINI, 2, seed=3)
    shots = [play_solo(BeliefAgent(), MINI, layout=layout, agent_seed=0).num_shots
             for layout in habit.layouts]
    assert expected_shots(BeliefAgent(), MINI, habit, agent_seeds=[0]) == pytest.approx(
        sum(shots) / 2)
