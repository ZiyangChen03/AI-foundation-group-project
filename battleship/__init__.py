"""Battleship environment, agents, evaluation, and tournaments."""

from .config import CLASSIC, MINI, GameConfig, label, neighbors, to_cell, to_rc
from .env import (BattleshipEnv, CellState, Observation, Result, Shot, StepResult,
                  is_consistent)
from .layout import (Layout, Ship, all_layouts, layout_from_cells, placements,
                     random_layout, validate_layout)
from .runner import EvalReport, GameRecord, evaluate, play_solo, simulate_records, train
from .tournament import MatchRecord, TournamentReport, play_match, round_robin

__all__ = [
    "CLASSIC", "MINI", "GameConfig", "label", "neighbors", "to_cell", "to_rc",
    "BattleshipEnv", "CellState", "Observation", "Result", "Shot", "StepResult",
    "is_consistent",
    "Layout", "Ship", "all_layouts", "layout_from_cells", "placements",
    "random_layout", "validate_layout",
    "EvalReport", "GameRecord", "evaluate", "play_solo", "simulate_records", "train",
    "MatchRecord", "TournamentReport", "play_match", "round_robin",
]
