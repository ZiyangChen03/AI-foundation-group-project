"""Deep Q-Network (DQN) agent: Q-learning with a convolutional network.

Separate from ``battleship.agents`` so that importing the other agents does not
load PyTorch. See docs/design_dqn.md.
"""

from .agent import DQNAgent
from .model import NetConfig, QNet
from .train import TrainConfig, train

__all__ = ["DQNAgent", "NetConfig", "QNet", "TrainConfig", "train"]
