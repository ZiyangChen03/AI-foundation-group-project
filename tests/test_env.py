import dataclasses

import pytest

from battleship import (MINI, BattleshipEnv, CellState, GameConfig, Result,
                        is_consistent, layout_from_cells)

LAYOUT = layout_from_cells(MINI, [[0, 1, 2], [8, 12]])


@pytest.fixture
def env():
    env = BattleshipEnv(MINI)
    env.reset(layout=LAYOUT)
    return env


def test_miss_hit_sunk(env):
    obs, reward, done, shot = env.step(15)
    assert (shot.result, reward, done) == (Result.MISS, -1.0, False)
    assert obs.board[15] is CellState.MISS

    obs, _, _, shot = env.step(8)
    assert shot.result is Result.HIT and shot.sunk_length is None
    assert obs.board[8] is CellState.HIT

    obs, _, done, shot = env.step(12)
    assert (shot.result, shot.sunk_length, done) == (Result.SUNK, 2, False)
    assert obs.board[8] is obs.board[12] is CellState.SUNK
    assert obs.remaining_ship_lengths == (3,)


def test_game_ends_when_fleet_is_sunk(env):
    for cell in (0, 1, 2, 8):
        assert not env.step(cell).done
    result = env.step(12)
    assert result.done and env.done
    assert result.observation.remaining_ship_lengths == ()
    assert env.legal_actions() == []
    assert env.reveal_layout() == LAYOUT
    with pytest.raises(ValueError, match="finished"):
        env.step(3)


def test_history_and_legal_actions(env):
    for cell in (5, 0, 9):
        obs = env.step(cell).observation
    assert [shot.cell for shot in obs.history] == [5, 0, 9]
    assert obs.num_shots == 3
    assert set(obs.legal_actions) == set(range(16)) - {5, 0, 9}


@pytest.mark.parametrize("action", [3, -1, 16, 1.5, "a", True])
def test_illegal_shots(env, action):
    env.step(3)
    with pytest.raises(ValueError):
        env.step(action)


def test_layout_hidden_until_done(env):
    with pytest.raises(RuntimeError):
        env.reveal_layout()


def test_observation_is_immutable(env):
    obs = env.observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.board = ()
    env.step(0)
    assert obs.board[0] is CellState.UNKNOWN       # old observations do not change


def test_sunk_cells_hidden_when_not_revealed():
    config = GameConfig(size=4, fleet=(3, 2), reveal_sunk_cells=False)
    env = BattleshipEnv(config)
    env.reset(layout=layout_from_cells(config, [[0, 1, 2], [8, 12]]))
    env.step(8)
    obs, _, _, shot = env.step(12)
    assert shot.result is Result.SUNK and shot.sunk_length == 2
    assert obs.board[8] is obs.board[12] is CellState.HIT


def test_seeded_reset_is_reproducible():
    a, b = BattleshipEnv(MINI), BattleshipEnv(MINI)
    a.reset(seed=3)
    b.reset(seed=3)
    for cell in range(16):
        assert a.step(cell).shot == b.step(cell).shot
        if a.done:
            break


def test_is_consistent(env):
    env.step(1)                                     # hit
    env.step(5)                                     # miss
    obs = env.observation()
    assert is_consistent(frozenset({1, 2}), obs)
    assert not is_consistent(frozenset({1, 5}), obs)
    env.step(8)
    obs = env.step(12).observation                  # sinks the 2-ship
    assert not is_consistent(frozenset({4, 8}), obs)
