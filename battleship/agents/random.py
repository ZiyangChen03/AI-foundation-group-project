from .base import Agent


class RandomAgent(Agent):
    """Baseline: fires at a cell chosen uniformly at random from the untried cells.

    It ignores hits and sunk ships entirely, so it gives the worst-case
    reference every other agent should beat. The choice uses the rng the runner
    seeds in ``new_game``, so games are reproducible.
    """

    name = "random"

    def select_shot(self, obs):
        return self.rng.choice(obs.legal_actions)
