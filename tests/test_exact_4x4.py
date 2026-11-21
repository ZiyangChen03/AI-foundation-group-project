import importlib.util
import random
from fractions import Fraction
from pathlib import Path

from battleship import MINI, GameConfig, all_layouts, play_solo
from battleship.agents import ProbabilityDensityAgent, RandomAgent

_script = Path(__file__).resolve().parents[1] / "scripts" / "exact_4x4.py"
_spec = importlib.util.spec_from_file_location("exact_4x4", _script)
exact = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exact)


def test_a_deterministic_agent_gets_the_shots_of_its_one_game():
    for layout in random.Random(0).sample(list(all_layouts(MINI)), 40):
        played = play_solo(ProbabilityDensityAgent(tie_break="first"), MINI, layout=layout,
                           agent_seed=0).num_shots
        assert exact.expected_shots(ProbabilityDensityAgent(tie_break="first"), layout) == played


def test_random_ties_average_over_every_choice():
    # Random shots at one 2-cell ship on 9 cells: the game ends at the later of two
    # uniformly random positions in a shuffle of 9, on average 2 * (9 + 1) / 3.
    config = GameConfig(3, (2,))
    for layout in all_layouts(config):
        value = exact.expected_shots(RandomAgent(), layout, config)
        assert abs(value - float(Fraction(20, 3))) < 1e-12
