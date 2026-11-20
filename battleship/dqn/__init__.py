"""Deep Q-Network (DQN) agent: Q-learning with a convolutional network.

Separate from ``battleship.agents`` so that importing the other agents does not
load PyTorch. See docs/design_dqn.md.
"""

try:
    import torch  # noqa: F401
except ImportError as error:
    raise ImportError("The DQN agent needs PyTorch, an optional extra: run `uv sync --extra dqn`, "
                      "or start the command with `uv run --extra dqn`.") from error

from .agent import DQNAgent
from .model import NetConfig, QNet
from .train import TrainConfig, train

__all__ = ["DQNAgent", "NetConfig", "QNet", "TrainConfig", "train"]
