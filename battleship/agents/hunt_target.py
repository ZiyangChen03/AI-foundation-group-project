from ..config import DIRECTIONS, to_cell, to_rc
from ..env import CellState
from .base import Agent


class HuntTargetAgent(Agent):
    """Hunt-and-target baseline.

    Hunt: fire at random unknown cells. With ``parity``, only cells on a
    diagonal grid spaced by the shortest remaining ship, since every ship must
    cover at least one of them.

    Target: while there are hits on ships not yet sunk, fire at their unknown
    neighbours, preferring cells that extend a line of two or more hits.

    The agent is stateless: every decision is made from the observation. With
    ``reveal_sunk_cells=False`` it cannot tell sunk hits apart, so it wastes
    some shots around ships that are already sunk.
    """

    def __init__(self, parity: bool = True):
        super().__init__()
        self.parity = parity
        self.name = "hunt-target+parity" if parity else "hunt-target"

    def select_shot(self, obs):
        return self.rng.choice(self._targets(obs) or self._hunt_cells(obs))

    def _targets(self, obs):
        size, board = obs.config.size, obs.board
        candidates, extends_line = set(), set()
        for hit, state in enumerate(board):
            if state is not CellState.HIT:
                continue
            row, col = to_rc(hit, obs.config)
            for dr, dc in DIRECTIONS:
                r, c = row + dr, col + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                cell = to_cell(r, c, obs.config)
                if board[cell] is not CellState.UNKNOWN:
                    continue
                candidates.add(cell)
                br, bc = row - dr, col - dc
                if (0 <= br < size and 0 <= bc < size
                        and board[to_cell(br, bc, obs.config)] is CellState.HIT):
                    extends_line.add(cell)
        return sorted(extends_line or candidates)

    def _hunt_cells(self, obs):
        legal = obs.legal_actions
        if self.parity:
            spacing = min(obs.remaining_ship_lengths)
            grid = [cell for cell in legal if sum(to_rc(cell, obs.config)) % spacing == 0]
            if grid:
                return grid
        return list(legal)
