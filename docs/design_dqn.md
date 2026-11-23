# DQN for Battleship: design and results

Code: `battleship/dqn/` (`DQNAgent`, training loop, vectorized games), scripts
`scripts/train_dqn.py` and `scripts/benchmark_dqn.py`, tests `tests/test_dqn.py`,
results `results/dqn/`. A beginner tutorial is in `tutorial/dqn_tutorial.html`,
and a two-page report in `report/dqn_report.pdf`.

## Takeaway

- **On 4×4,** a DQN that sees only the raw board learns in 75 s to play at the level
  of probability density: 8.924 exact expected shots, against 8.864 (paired
  +0.06 ± 0.26). It comes within 2% of the exact optimum, 8.746. The board Q-table
  of PR #14 stayed at random level (13.9).
- **On 10×10,** on 2,000 test boards, the board-only DQN (A) reaches 53.0 shots.
  Adding the density map as an input plane (B) reaches **45.85**, 1.20 ± 0.46 shots
  behind probability density (44.65). Using the density as a fixed baseline under a
  learned correction (C) did not improve on it (46.05).
- **No learned agent beats probability density.** Each variant was trained with one
  seed and untuned settings; run A became unstable after 15k games.

## 1. Question

> Can a model-free learner, never told the placement rules, learn 10×10 Battleship,
> and does the probability-density map help it more as an input feature or as a
> starting policy?

The alternatives are limited on 10×10:
- **Exact belief and optimal planning:** need all 30,093,975,536 layouts listed.
- **Rollout planner:** no measurable gain over greedy (−0.08 ± 0.97 shots).
- **Tabular Q-learning:** fails. On the board it reaches 13.9 on 4×4 against 14.1 for
  random, with 3.5M states and about 4 updates each (PR #14). On the exact belief
  it works on 4×4 only (8.972, PR #15).

A network generalizes between similar boards, which is what the table lacked.

## 2. Model

| Choice | Decision | Reason |
|---|---|---|
| Reward and discount | −1 per shot, γ = 1 | The return equals −(shots). Games always end, and every shot costs the same. γ = 0.9 would make 45- and 60-shot games look almost equal (−9.91 against −9.98) |
| State | Planes: hit, miss, sunk, on-board (all 1s), and ships afloat per length 1–5; B and C add the density map scaled to max 1 | Observations are exact, so the board plus the remaining ships is a Markov state. The density is computed from the board, so it is a feature, not new information. The on-board plane separates the board edge (zero padding) from unknown cells |
| Network | 4 × (3×3 conv, 32 channels, ReLU), 34.6k weights | Weight sharing: the same local pattern applies at every cell. Four layers give a 9×9 view, enough to see a carrier through any cell. 32 channels were measured 1.7× cheaper than 64 |
| Head | Dueling: Q = V + A − mean A over legal cells; A from a 1×1 conv, V from global mean and max pooling | The shots still needed depend on the whole board, which a 9×9 window cannot see; the choice of cell is local. The argmax depends only on A |
| Residual (C) | Q = c·d̂ + network, c = 1, A's last layer starting at 0 | The untrained agent plays exactly like probability density |
| Targets | Double DQN, Huber loss, target network copied every 1,000 steps | The max over 100 noisy cells overestimates; Huber limits outlier errors; a frozen target stops the network chasing itself |
| Masking | Fired cells get −∞ when acting and inside the target max | No wasted shots; avoids PR #14's "untried looks best" problem |

## 3. Training

| Setting | Value | Reason |
|---|---|---|
| Parallel games | 256 tensor games (`VecBattleship`) | One forward pass picks 256 shots. Measured 1–3.5 µs per board, against ~430 µs for one game at a time |
| Replay | 200k transitions on the GPU, batch 256, each transition replayed 8× on average | Atari DQN's replay ratio; 4× cheaper than the first plan's 32× |
| Optimizer | Adam, lr 3e-4, gradient clip 10 | Beat 1e-4 on the 4×4 pilot (8.924 against 8.962) |
| Exploration | ε from 1.0 (C: 0.1) to 0.02 over the first 10% of games | C already starts strong, so heavy random play would waste that |
| Devices | Games and density on the CPU, network on the Apple GPU (MPS) | – |
| Speed | ~140 games/s on 10×10; 100k games in 12–14 min | – |

**Verification** (`tests/test_dqn.py`, 15 tests):
- Random games through `VecBattleship` and `BattleshipEnv` give identical boards,
  remaining ships, shot counts and end of game at every step, on 4×4, 10×10 and a
  6×6 board that doesn't reveal sunk cells.
- The convolutional density equals `ProbabilityDensityAgent.density` exactly.
- The untrained residual agent always picks a highest-density cell.
- Decisions depend only on public results (`assert_public_only`).
- The fast greedy evaluation matches games played through `play_solo`.
- A short run learns, and save → load plays identically.

## 4. Results

**Protocol:**
- **Validation:** every 5k games the greedy policy plays 500 boards (seeds
  100000–100499), and the best checkpoint is kept. For C the best was the untrained
  network, so we report its final checkpoint.
- **Test:** 2,000 boards (seeds 0–1999), the same for every agent, played through
  `play_solo`, with paired differences against probability density.
- **4×4:** every agent plays all 264 layouts. That is exact for the deterministic
  DQN; agents that break ties at random get one sample per layout.

### 10×10 (`results/dqn/benchmark_classic.md`)

| Agent | Shots | ± 95% | Paired vs density | Hunting | Targeting | Training |
|---|---|---|---|---|---|---|
| random | 95.27 | 0.22 | +50.62 ± 0.45 | 5.94 | 89.33 | – |
| hunt-target + parity | 50.76 | 0.39 | +6.11 ± 0.55 | 32.93 | 17.83 | – |
| probability density | 44.65 | 0.40 | – | 27.46 | 17.19 | – |
| belief greedy (particles) | 44.50 | 0.41 | −0.15 ± 0.51 | 27.32 | 17.19 | – |
| DQN-A (board) | 53.00 | 0.45 | +8.35 ± 0.54 | 17.68 | 35.31 | 12.4 min |
| **DQN-B (+ density input)** | **45.85** | 0.32 | +1.20 ± 0.46 | 17.09 | 28.76 | 14.1 min |
| DQN-C (residual) | 46.05 | 0.32 | +1.41 ± 0.46 | 19.69 | 26.36 | 14.0 min |

"Hunting" counts shots fired while no unsunk hit is showing; "targeting" counts the
rest.

### 4×4 (`results/dqn/benchmark_mini.md`)

| Agent | Shots | Paired vs density |
|---|---|---|
| random | 14.227 | +5.36 ± 0.27 |
| hunt-target + parity | 9.697 | +0.83 ± 0.32 |
| probability density | 8.864 | – |
| belief greedy | 8.879 | +0.02 ± 0.12 |
| **DQN-A, lr 3e-4** (75 s) | **8.924** | +0.06 ± 0.26 |
| DQN-A, lr 1e-4 | 8.962 | +0.10 ± 0.24 |
| optimal V\* (from `planning.py`) | 8.746 | – |

### Learning curves (validation, 500 boards; density scores 44.13 on them)

| Games | A | B | C |
|---|---|---|---|
| 0 | 90.5 | 93.5 | 44.1 |
| 5k | 55.6 | 50.5 | 56.3 |
| 15k | **52.4** | 49.6 | 47.4 |
| 30k | 52.9 | 46.3 | 45.9 |
| 55k | 60.6 | 45.9 | 44.6 |
| 100k | 55.8 | **45.7** | 45.6 |

## 5. Findings

1. **A network fixes what broke the Q-table.** The same raw-board state that left
   PR #14 at random level reaches density level on 4×4, because the network shares
   what it learns across similar boards.
2. **The density helps as a feature, not as a policy.**
   - As an input (B), it is worth 7.2 shots over board-only (A).
   - As a fixed baseline (C), early corrections are badly calibrated: the agent
     drops from 44 to 56 in the first 5k games and never recovers past its start.
3. **The DQN spends its shots differently.** It hunts in fewer shots than density
   (17.1 against 27.5), but fires many more while a damaged ship is still afloat
   (28.8 against 17.2). It tends to leave a hit ship and probe elsewhere, which costs
   more than it saves.
4. **A is unstable.** Its best was at 15k games; afterwards validation wandered
   between 53 and 62 while the training loss rose from 0.36 to 1.9. This is the
   usual DQN divergence.
5. **There is little headroom.** On 4×4 greedy is within 2% of optimal, and belief
   greedy is level with density on 10×10, so beating density likely means gaining
   less than a shot.

## 6. Limitations and next steps

- **Limitations:** one seed per variant, no hyperparameter search, a single
  validation set.
- **Stability:** a lower learning rate and less frequent target copies to fix A;
  then early stopping.
- **Fewer games:** n-step returns.
- **Throughput:** overlap the CPU game step with GPU training, and compute the
  density on the GPU.
- **Residual agent:** learn V first, before the correction may change the density's
  ranking.
- **Multiple seeds** for every variant.

## Reproduce

```bash
uv run scripts/train_dqn.py --board mini --variant board --games 100000 --eval-every 10000 --learn-start 5000
uv run scripts/train_dqn.py --board classic --variant board    --games 100000 --eval-every 5000 --eps-fraction 0.1
uv run scripts/train_dqn.py --board classic --variant density  --games 100000 --eval-every 5000 --eps-fraction 0.1
uv run scripts/train_dqn.py --board classic --variant residual --games 100000 --eval-every 5000 --eps-start 0.1 --eps-fraction 0.1
uv run scripts/benchmark_dqn.py classic results/dqn/classic_board_s0/best.pt \
    results/dqn/classic_density_s0/best.pt results/dqn/classic_residual_s0/final.pt   # ~5 min
uv run scripts/benchmark_dqn.py mini results/dqn/mini_board_s0/final.pt
```

The 4×4 runs in `results/dqn/` were trained with an output folder given explicitly;
`mini_board_lr1e-4_s0` adds `--lr 0.0001`.
