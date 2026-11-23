"""DQN training: parallel games, a replay buffer, a target network, and Double
DQN targets.

    target y = -1                                    if the shot ends the game
    target y = -1 + gamma * Q_target(s', argmax_a' Q_online(s', a'))   otherwise
    loss     = Huber(Q_online(s, a) - y)

Every training iteration fires one shot in each of ``parallel`` games, stores
those transitions, and then runs enough gradient steps that each stored shot is
sampled ``replay_ratio`` times on average.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import GameConfig
from ..layout import Layout, random_layout
from .model import NetConfig, QNet, encode
from .vec_env import MAX_LENGTH, VecBattleship, normalized_density


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrainConfig:
    games: int = 50_000             # training games in total
    parallel: int = 256             # games played side by side
    batch: int = 256
    replay_ratio: int = 8           # times each transition is trained on, on average
    buffer: int = 200_000           # transitions kept
    learn_start: int = 10_000       # transitions collected before training starts
    lr: float = 3e-4
    gamma: float = 1.0
    target_every: int = 1_000       # gradient steps between target-network copies
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_fraction: float = 0.3       # share of the games over which epsilon decays
    double: bool = True
    grad_clip: float = 10.0
    eval_every: int = 5_000         # games between greedy evaluations
    seed: int = 0
    device: str = ""


class Replay:
    """Transitions stored on the training device, overwritten oldest first."""

    def __init__(self, capacity: int, cells: int, density: bool, device):
        self.capacity, self.size, self.pos, self.device = capacity, 0, 0, device
        zeros = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype, device=device)
        self.board = zeros(capacity, cells, dtype=torch.int8)
        self.next_board = zeros(capacity, cells, dtype=torch.int8)
        self.remaining = zeros(capacity, MAX_LENGTH, dtype=torch.int8)
        self.next_remaining = zeros(capacity, MAX_LENGTH, dtype=torch.int8)
        self.action = zeros(capacity, dtype=torch.long)
        self.done = zeros(capacity, dtype=torch.bool)
        self.dens = zeros(capacity, cells, dtype=torch.float16) if density else None
        self.next_dens = zeros(capacity, cells, dtype=torch.float16) if density else None

    def add(self, board, remaining, dens, action, done, next_board, next_remaining, next_dens):
        n = board.shape[0]
        idx = (torch.arange(n) + self.pos) % self.capacity
        idx = idx.to(self.device)
        for store, value in ((self.board, board), (self.next_board, next_board),
                             (self.remaining, remaining), (self.next_remaining, next_remaining),
                             (self.action, action), (self.done, done),
                             (self.dens, dens), (self.next_dens, next_dens)):
            if store is not None:
                store[idx] = value.to(self.device, store.dtype)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch: int):
        idx = torch.randint(self.size, (batch,), device=self.device)
        pick = lambda t: None if t is None else t[idx]
        return (pick(self.board), pick(self.remaining), pick(self.dens), self.action[idx],
                self.done[idx], pick(self.next_board), pick(self.next_remaining),
                pick(self.next_dens))


def _q(model: QNet, board, remaining, dens, config: GameConfig):
    """Q for a batch of stored boards (already on the model's device)."""
    net = model.net
    if dens is not None:
        dens = dens.float()
    x = encode(board, remaining, config, dens if net.use_density else None)
    return model(x, board == 0, dens)


def play_greedy(model: QNet, config: GameConfig, layouts: list[Layout], device) -> torch.Tensor:
    """Shots the greedy policy needs on each layout (all games played together)."""
    model.eval()
    env = VecBattleship(config, layouts=layouts)
    with torch.no_grad():
        while not env.done.all():
            active = ~env.done
            remaining = env.remaining()
            dens = (normalized_density(env.board, remaining, config).to(device)
                    if model.net.needs_density else None)
            board = env.board.to(device)
            q = _q(model, board, remaining.to(device), dens, config)
            env.step(q.argmax(1).cpu(), active)
    model.train()
    return env.shots.clone()


def train(config: GameConfig, net: NetConfig, tc: TrainConfig, eval_layouts: list[Layout],
          log=print, callback=None) -> tuple[QNet, list[dict]]:
    """Train a Q-network; returns it and one log row per evaluation."""
    device = tc.device or default_device()
    torch.manual_seed(tc.seed)
    rng = random.Random(tc.seed)
    online = QNet(net).to(device)
    target = copy.deepcopy(online).eval()
    opt = torch.optim.Adam(online.parameters(), lr=tc.lr)
    replay = Replay(tc.buffer, config.num_cells, net.needs_density, device)
    env = VecBattleship(config, tc.parallel, seed=rng.getrandbits(32))
    empty_dens = None
    dens = None
    if net.needs_density:
        dens = normalized_density(env.board, env.remaining(), config)
        empty_dens = dens[0].clone()
    steps_per_iter = max(1, round(tc.parallel * tc.replay_ratio / tc.batch))
    finished, grad_steps, recent, losses = 0, 0, [], []
    next_eval, rows, start = 0, [], time.time()

    def evaluate_now():
        shots = play_greedy(online, config, eval_layouts, device).float()
        row = dict(games=finished, eval_mean=shots.mean().item(),
                   eval_sem=(shots.std() / math.sqrt(len(shots))).item() if len(shots) > 1 else 0.0,
                   train_mean=sum(recent) / len(recent) if recent else float("nan"),
                   epsilon=epsilon(), loss=sum(losses) / len(losses) if losses else float("nan"),
                   grad_steps=grad_steps, seconds=time.time() - start)
        rows.append(row)
        log(f"games {row['games']:>7}  eval {row['eval_mean']:6.2f} ± {1.96 * row['eval_sem']:.2f}"
            f"  train {row['train_mean']:6.2f}  eps {row['epsilon']:.3f}  loss {row['loss']:.3f}"
            f"  {row['seconds']:6.0f}s")
        recent.clear()
        losses.clear()
        if callback:
            callback(online, row)

    def epsilon():
        frac = min(1.0, finished / max(1, tc.eps_fraction * tc.games))
        return tc.eps_start + frac * (tc.eps_end - tc.eps_start)

    while True:
        if finished >= next_eval:
            evaluate_now()
            next_eval += tc.eval_every
            if finished >= tc.games:
                break
        board, remaining = env.board.clone(), env.remaining()
        legal = board == 0
        with torch.no_grad():
            q = _q(online, board.to(device), remaining.to(device),
                   None if dens is None else dens.to(device), config)
        actions = q.argmax(1).cpu()
        explore = torch.rand(env.n) < epsilon()
        if explore.any():
            actions[explore] = torch.multinomial(legal[explore].float(), 1).squeeze(1)
        done = env.step(actions).clone()
        next_board, next_remaining = env.board.clone(), env.remaining()
        next_dens = (normalized_density(next_board, next_remaining, config)
                     if net.needs_density else None)
        replay.add(board, remaining, dens, actions, done, next_board, next_remaining, next_dens)
        if done.any():
            recent.extend(env.shots[done].tolist())
            finished += int(done.sum())
            env.reset(done)
            if net.needs_density:
                next_dens[done] = empty_dens
        dens = next_dens

        if replay.size < tc.learn_start:
            continue
        for _ in range(steps_per_iter):
            b, r, d, a, dn, nb, nr, nd = replay.sample(tc.batch)
            q_sa = _q(online, b, r, d, config).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                q_next_t = _q(target, nb, nr, nd, config)
                if tc.double:
                    best = _q(online, nb, nr, nd, config).argmax(1, keepdim=True)
                    q_next = q_next_t.gather(1, best).squeeze(1)
                else:
                    q_next = q_next_t.max(1).values
                y = -1.0 + tc.gamma * torch.where(dn, torch.zeros_like(q_next), q_next)
            loss = F.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), tc.grad_clip)
            opt.step()
            grad_steps += 1
            if grad_steps % 50 == 0:
                losses.append(loss.item())
            if grad_steps % tc.target_every == 0:
                target.load_state_dict(online.state_dict())
    return online, rows


def eval_layouts(config: GameConfig, count: int, first_seed: int) -> list[Layout]:
    """Validation boards, drawn exactly as ``evaluate(seeds=...)`` draws them."""
    return [random_layout(config, random.Random(s)) for s in range(first_seed, first_seed + count)]
