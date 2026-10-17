from ..env import CellState
from ..layout import placements
from .base import Agent


class ProbabilityDensityAgent(Agent):
    """Probability density strategy from N. Berry, "Battleship", DataGenetics, 2011.

    Every turn, each remaining ship is tried in every horizontal and vertical
    position. Positions covering a miss or a sunk ship are impossible. Each
    possible position adds its weight to every untried cell it covers, and the
    agent fires at the cell with the highest total.

    Target mode: positions may pass through hits on ships not yet sunk, and get
    weight ``hit_weight ** k`` where k is the number of such hits they cover,
    so the agent finishes a ship it has found and follows its direction. With
    no open hits (hunt mode) every position has weight 1.

    Ties go to a random best cell, or with ``tie_break="first"`` to the first
    one from the top left, as in the blog.

    With ``reveal_sunk_cells=False`` the agent cannot tell which hits belong to
    sunk ships, so it keeps treating them as open hits.
    """

    def __init__(self, hit_weight: int = 50, tie_break: str = "random"):
        super().__init__()
        if tie_break not in ("random", "first"):
            raise ValueError('tie_break must be "random" or "first"')
        self.hit_weight = hit_weight
        self.tie_break = tie_break
        self.name = "probability-density"

    def density(self, obs):
        """Score of every cell; 0 for cells already shot."""
        board = obs.board
        blocked = {cell for cell, state in enumerate(board)
                   if state is CellState.MISS or state is CellState.SUNK}
        hits = {cell for cell, state in enumerate(board) if state is CellState.HIT}
        scores = [0] * obs.config.num_cells
        for length in obs.remaining_ship_lengths:
            for cells in placements(obs.config, length):
                if not cells.isdisjoint(blocked):
                    continue
                weight = self.hit_weight ** len(cells & hits)
                for cell in cells:
                    if board[cell] is CellState.UNKNOWN:
                        scores[cell] += weight
        return scores

    def select_shot(self, obs):
        scores = self.density(obs)
        best = max(scores[cell] for cell in obs.legal_actions)
        candidates = [cell for cell in obs.legal_actions if scores[cell] == best]
        if self.tie_break == "first":
            return candidates[0]
        return self.rng.choice(candidates)
