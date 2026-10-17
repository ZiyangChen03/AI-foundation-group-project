import random
from itertools import combinations

import pytest

from battleship import (CLASSIC, MINI, BattleshipEnv, CellState, GameConfig, all_layouts,
                        evaluate, layout_from_cells, play_solo)
from battleship.agents import Agent, HuntTargetAgent, ProbabilityDensityAgent


def play_until(env, cells):
    obs = env.observation()
    for cell in cells:
        obs = env.step(cell).observation
    return obs


def test_empty_board_density_counts_placements():
    obs = BattleshipEnv(MINI).reset(seed=0)
    scores = ProbabilityDensityAgent().density(obs)
    # corner A1: 2 positions for the 3-ship + 2 for the 2-ship
    assert scores[0] == 4
    # inner cell B2: 4 positions for the 3-ship + 4 for the 2-ship
    assert scores[5] == 8


def test_first_shot_is_central_on_classic():
    agent = ProbabilityDensityAgent()
    agent.new_game(CLASSIC, random.Random(0))
    assert agent.select_shot(BattleshipEnv(CLASSIC).reset(seed=0)) in (44, 45, 54, 55)


def test_misses_and_sunk_ships_block_placements():
    env = BattleshipEnv(MINI)
    env.reset(layout=layout_from_cells(MINI, [[0, 1, 2], [8, 12]]))
    obs = play_until(env, [8, 12, 5])          # sink the 2-ship, miss at B2
    scores = ProbabilityDensityAgent().density(obs)
    assert scores[8] == scores[12] == scores[5] == 0
    # only the 3-ship remains, and A1 can only hold it horizontally now
    assert scores[0] == 1
    assert obs.remaining_ship_lengths == (3,)


def test_target_mode_follows_the_line_of_hits():
    env = BattleshipEnv(CLASSIC)
    env.reset(layout=layout_from_cells(
        CLASSIC, [[44, 45, 46, 47, 48], [0, 10, 20, 30], [2, 3, 4], [80, 81, 82], [98, 99]]))
    obs = play_until(env, [45, 46])            # two hits side by side
    agent = ProbabilityDensityAgent()
    agent.new_game(CLASSIC, random.Random(0))
    assert agent.select_shot(obs) in (44, 47)


def test_first_tie_break_is_deterministic():
    agent = ProbabilityDensityAgent(tie_break="first")
    obs = BattleshipEnv(CLASSIC).reset(seed=0)
    agent.new_game(CLASSIC, random.Random(1))
    first = agent.select_shot(obs)
    agent.new_game(CLASSIC, random.Random(2))
    assert agent.select_shot(obs) == first == 44


def test_beats_hunt_target_on_classic():
    seeds = range(200)
    density = evaluate(ProbabilityDensityAgent(), CLASSIC, seeds=seeds).mean
    hunt = evaluate(HuntTargetAgent(parity=True), CLASSIC, seeds=seeds).mean
    assert density < hunt - 3



class ReferenceAgent(Agent):
    """The blog's algorithm written independently: slide each ship along rows and columns."""

    name = "reference"

    def __init__(self, tie_break):
        super().__init__()
        self.tie_break = tie_break

    def select_shot(self, obs):
        n, board = obs.config.size, obs.board
        scores = [0] * (n * n)
        for length in obs.remaining_ship_lengths:
            for r in range(n):
                for c in range(n):
                    for horizontal in (True, False):
                        if (c if horizontal else r) + length > n:
                            continue
                        cells = [r * n + c + i if horizontal else (r + i) * n + c
                                 for i in range(length)]
                        if any(board[x] in (CellState.MISS, CellState.SUNK) for x in cells):
                            continue
                        weight = 50 ** sum(board[x] is CellState.HIT for x in cells)
                        for x in cells:
                            if board[x] is CellState.UNKNOWN:
                                scores[x] += weight
        best = max(scores[x] for x in obs.legal_actions)
        candidates = [x for x in obs.legal_actions if scores[x] == best]
        return candidates[0] if self.tie_break == "first" else self.rng.choice(candidates)


@pytest.mark.parametrize("tie_break", ["first", "random"])
@pytest.mark.parametrize("config", [MINI, CLASSIC, GameConfig(6, (3, 3, 2))])
def test_plays_the_same_games_as_a_reference_implementation(config, tie_break):
    for seed in range(10):
        ours = play_solo(ProbabilityDensityAgent(hit_weight=50, tie_break=tie_break),
                         config, seed=seed)
        reference = play_solo(ReferenceAgent(tie_break), config, seed=seed)
        assert ours.shots == reference.shots


def test_decisions_depend_only_on_public_results():
    """With the same agent seed, games on different hidden layouts make the same
    shots until the first shot whose public result differs."""
    records = [play_solo(ProbabilityDensityAgent(), MINI, layout=layout, agent_seed=0)
               for layout in all_layouts(MINI)]
    for a, b in combinations(records, 2):
        for shot_a, shot_b in zip(a.shots, b.shots):
            assert shot_a.cell == shot_b.cell
            if shot_a != shot_b:
                break
