import pytest

from battleship import (CLASSIC, MINI, CellState, GameConfig, all_layouts, evaluate,
                        layout_from_cells, neighbors, play_solo)
from battleship.agents import Agent, HuntTargetAgent
from helpers import assert_public_only, states

VARIANTS = [dict(parity=p, follow_direction=d) for p in (False, True) for d in (False, True)]


class ReferenceAgent(Agent):
    """Hunt-and-target written independently: direction targets come from
    scanning rows and columns for runs of two or more hits."""

    def __init__(self, parity, follow_direction):
        super().__init__()
        self.parity, self.follow_direction = parity, follow_direction

    def select_shot(self, obs):
        n, board = obs.config.size, obs.board
        hit = [state is CellState.HIT for state in board]
        unknown = [state is CellState.UNKNOWN for state in board]
        around = {x for h in range(n * n) if hit[h] for x in neighbors(h, obs.config) if unknown[x]}
        ends = set()
        lines = [[r * n + c for c in range(n)] for r in range(n)] + \
                [[r * n + c for r in range(n)] for c in range(n)]
        for line in lines:
            k = 0
            while k < n:
                start = k
                while k < n and hit[line[k]]:
                    k += 1
                if k - start >= 2:
                    ends.update(line[e] for e in (start - 1, k) if 0 <= e < n and unknown[line[e]])
                k = max(k, start + 1)
        targets = sorted(ends) if self.follow_direction and ends else sorted(around)
        if targets:
            return self.rng.choice(targets)
        legal = list(obs.legal_actions)
        if self.parity:
            m = min(obs.remaining_ship_lengths)
            legal = [x for x in legal if (x // n + x % n) % m == 0] or legal
        return self.rng.choice(legal)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("config", [MINI, CLASSIC, GameConfig(6, (3, 3, 2))])
def test_plays_the_same_games_as_a_reference_implementation(config, variant):
    for seed in range(10):
        ours = play_solo(HuntTargetAgent(**variant), config, seed=seed)
        reference = play_solo(ReferenceAgent(**variant), config, seed=seed)
        assert ours.shots == reference.shots


@pytest.mark.parametrize("variant", VARIANTS)
def test_decisions_depend_only_on_public_results(variant):
    records = [play_solo(HuntTargetAgent(**variant), MINI, layout=layout, agent_seed=0)
               for layout in all_layouts(MINI)]
    assert_public_only(records)


@pytest.mark.parametrize("variant", VARIANTS)
def test_hunt_target_and_direction_rules_hold_on_every_shot(variant):
    config = CLASSIC
    for record in evaluate(HuntTargetAgent(**variant), config, seeds=range(30)).records:
        for shot, fired, open_hits, remaining in states(record):
            around = {x for h in open_hits for x in neighbors(h, config) if x not in fired}
            ends = {2 * h - b for h in open_hits for b in neighbors(h, config)
                    if b in open_hits and 2 * h - b in neighbors(h, config)
                    and 2 * h - b not in fired}
            if variant["follow_direction"] and ends:
                assert shot.cell in ends                  # continue along the ship
            elif around:
                assert shot.cell in around                # neighbour of an open hit
            elif variant["parity"]:
                m = min(remaining)                        # hunt on the parity grid
                assert sum(divmod(shot.cell, config.size)) % m == 0 or not any(
                    sum(divmod(x, config.size)) % m == 0
                    for x in range(config.num_cells) if x not in fired)


def test_finishes_a_found_ship_then_returns_to_hunting():
    layout = layout_from_cells(CLASSIC, [[44, 45, 46, 47, 48], [0, 10, 20, 30], [2, 3, 4],
                                         [80, 81, 82], [98, 99]])
    for seed in range(20):
        record = play_solo(HuntTargetAgent(), CLASSIC, layout=layout, agent_seed=seed)
        cells = [shot.cell for shot in record.shots]
        for ship in layout:
            first, last = (min(cells.index(c) for c in ship.cells),
                           max(cells.index(c) for c in ship.cells))
            # after the first hit, the whole ship is sunk within a few shots
            assert last - first <= ship.length + 3


def test_each_part_improves_results():
    seeds = range(300)
    mean = {(p, d): evaluate(HuntTargetAgent(parity=p, follow_direction=d), CLASSIC,
                             seeds=seeds).mean
            for p in (False, True) for d in (False, True)}
    assert mean[True, True] < mean[True, False] < mean[False, False]
    assert mean[True, True] < mean[False, True] < mean[False, False]


def test_names():
    assert HuntTargetAgent().name == "hunt-target+parity"
    assert HuntTargetAgent(parity=False).name == "hunt-target"
    assert HuntTargetAgent(follow_direction=False).name == "hunt-target+parity-nodirection"
