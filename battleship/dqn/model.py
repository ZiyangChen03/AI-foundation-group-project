"""The Q-network and the board encoding.

Input planes (one 10x10 image each): hit, miss, sunk, on-board (all ones, so
zero padding marks the edge), one plane per ship length 1..5 holding how many
ships of that length are still afloat, and, when ``use_density`` is set, the
normalized probability-density map.

The network is fully convolutional (the same 3x3 filters at every cell) with a
dueling split:

    Q(s, a) = V(s) + A(s, a) - mean over legal a of A(s, a)

V is the number of shots still needed, a quantity of the whole board, so it is
read from global pooling; A ranks cells and comes from the local filters.
With ``residual`` > 0 the agent adds ``residual * density`` to Q, and A's last
layer starts at zero, so an untrained agent plays like probability density.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..config import GameConfig
from .vec_env import HIT, MAX_LENGTH, MISS, SUNK, normalized_density


@dataclass(frozen=True)
class NetConfig:
    channels: int = 32
    layers: int = 4
    use_density: bool = False
    residual: float = 0.0       # weight c of the density added to Q (0 = off)

    @property
    def needs_density(self) -> bool:
        return self.use_density or self.residual > 0

    @property
    def in_planes(self) -> int:
        return 4 + MAX_LENGTH + self.use_density


class QNet(nn.Module):
    def __init__(self, net: NetConfig):
        super().__init__()
        self.net = net
        layers, width = [], net.in_planes
        for _ in range(net.layers):
            layers += [nn.Conv2d(width, net.channels, 3, padding=1), nn.ReLU()]
            width = net.channels
        self.body = nn.Sequential(*layers)
        self.advantage = nn.Conv2d(width, 1, 1)
        self.value = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 1))
        if net.residual > 0:
            nn.init.zeros_(self.advantage.weight)
            nn.init.zeros_(self.advantage.bias)

    def forward(self, x: torch.Tensor, legal: torch.Tensor,
                dens: torch.Tensor | None = None) -> torch.Tensor:
        """Q for every cell; illegal cells are -inf."""
        h = self.body(x)
        a = self.advantage(h).flatten(1)
        v = self.value(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        legal_f = legal.float()
        a_mean = (a * legal_f).sum(1, keepdim=True) / legal_f.sum(1, keepdim=True).clamp(min=1)
        q = v + a - a_mean
        if self.net.residual > 0:
            q = q + self.net.residual * dens
        return q.masked_fill(~legal, float("-inf"))


def encode(board: torch.Tensor, remaining: torch.Tensor, config: GameConfig,
           dens: torch.Tensor | None = None) -> torch.Tensor:
    """Input planes for a batch of boards (``board``: n x cells int codes)."""
    n, size = board.shape[0], config.size
    grid = board.view(n, 1, size, size)
    planes = [grid == HIT, grid == MISS, grid == SUNK, torch.ones_like(grid, dtype=torch.bool)]
    planes = [p.float() for p in planes]
    planes.append(remaining.float().view(n, MAX_LENGTH, 1, 1).expand(n, MAX_LENGTH, size, size))
    if dens is not None:
        planes.append(dens.view(n, 1, size, size))
    return torch.cat(planes, 1)


def features(board, remaining, config: GameConfig, net: NetConfig, device="cpu"):
    """(input planes, legal mask, density or None), all on ``device``."""
    dens = normalized_density(board, remaining, config) if net.needs_density else None
    x = encode(board, remaining, config, dens if net.use_density else None)
    legal = board == 0
    dens = None if dens is None else dens.to(device)
    return x.to(device), legal.to(device), dens


def save(path, model: QNet, config: GameConfig, info: dict | None = None) -> None:
    torch.save({"net": asdict(model.net), "state_dict": model.state_dict(),
                "config": {"size": config.size, "fleet": list(config.fleet),
                           "reveal_sunk_cells": config.reveal_sunk_cells},
                "info": info or {}}, path)


def load(path, device="cpu") -> tuple[QNet, GameConfig, dict]:
    data = torch.load(path, map_location=device, weights_only=True)
    model = QNet(NetConfig(**data["net"])).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    c = data["config"]
    return model, GameConfig(c["size"], tuple(c["fleet"]), c["reveal_sunk_cells"]), data["info"]
