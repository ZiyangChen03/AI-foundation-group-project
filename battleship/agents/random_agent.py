from .base import Agent


class RandomAgent(Agent):
    """Fires at a uniformly random unshot cell."""

    name = "random"

    def select_shot(self, obs):
        return self.rng.choice(obs.legal_actions)
