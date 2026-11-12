# Tabular Q-learning on 4×4

`QLearningAgent` learns a Q-table over the **public board** (each of the 16 cells:
unknown, miss, hit or sunk) plus the remaining ship lengths. It never sees the
hidden layout. It supports MINI only: 4×4, ships of lengths 3 and 2, sunk cells
revealed.

## Results

Three training seeds (0, 1, 2) per budget. Training draws layouts uniformly from
all 264; testing plays each of the 264 layouts once, the same evaluation as every
other agent. The reported model is the final one (no checkpoint is picked).

**Table 1.** Mean shots over all 264 layouts (lower is better).

| Training episodes per seed | 1,000 | 10,000 | 100,000 | 1,000,000 |
|---|---:|---:|---:|---:|
| **Mean shots (3 seeds)** | **13.149** | **12.835** | **11.898** | **10.768** |
| Per seed (0 / 1 / 2) | 13.144 / 13.140 / 13.163 | 12.848 / 12.795 / 12.860 | 11.943 / 11.966 / 11.784 | 10.591 / 10.902 / 10.811 |
| Std across the 3 seed means | 0.010 | 0.028 | 0.081 | 0.130 |
| Std over all 792 games | 2.540 | 2.484 | 2.316 | 2.222 |
| Worst game | 16 | 16 | 16 | 15 |
| Decisions using a learned value | 81.6% | 99.4% | 100.0% | 100.0% |
| Q-table states (mean per seed) | 9,270 | 68,127 | 427,644 | 2,197,599 |
| Training time per seed | 0.3 s | 3.4 s | 31 s | ~300 s |

References on the same 264 layouts: random 14.23, hunt-target + parity 9.70,
probability density 8.86, POMDP greedy belief 8.913, optimal policy 8.746.

- The agent improves steadily with training and is still improving at 1,000,000
  episodes. The three seeds agree closely (spread at most 0.31 shots).
- At 1,000,000 episodes it beats random (14.23) and approaches hunt-target without
  parity (10.20), but stays behind hunt-target + parity (9.70), probability density
  (8.86), the POMDP greedy agent (8.91) and the optimum (8.75).
- Before the fixes described under *Learning rule* (pessimistic default, 1/N step,
  fixed tie-breaking, training on all layouts), the same budgets scored 13.87–14.24
  on the held-out split: random level.
- Training time for seed 1 at 1,000,000 episodes was 676 s because it shared the CPU
  with other runs; the other two took about 300 s. Decision time for the 1,000,000-
  episode models (5.8 ms per shot, against about 0.011 ms for the others) is inflated
  by memory pressure: the evaluation loads all three ~2.2-million-state tables at once.

![Running average training reward, 1,000,000 episodes, seeds 0, 1 and 2](results/experiment_1m/learning_curve.svg)

**Figure 1.** Cumulative mean training reward (reward = −shots, so higher is better)
for the 1,000,000-episode runs, one line per seed. This is a training curve with
exploration, not the greedy evaluation score.

Sources: [1,000](results/experiment_1k/README.md),
[10,000](results/experiment_10k/README.md),
[100,000](results/experiment_100k/README.md) and
[1,000,000](results/experiment_1m/README.md) episodes. Each folder holds the exact
configuration, per-seed curves, per-game results and source hashes.

## Learning rule

Each shot has reward −1, γ = 1, so maximizing return minimizes shots:

    Q(s, a) <- Q(s, a) + step * (target - Q(s, a))
    target = -1                                     if the shot ends the game
    target = -1 + gamma * max over legal a' Q(s', a')   otherwise

- **Untried actions start at −(number of untried cells)**, a lower bound on any
  action's value (a game never needs more shots than there are cells left). A 0.0
  default would rank every untried action above every tried one, since all rewards
  are negative.
- **Step size 1/N(s, a)** by default, N being the number of updates of that pair, so
  each Q-value is the running average of its targets (η = 1/(1 + updates) in the
  course notes). `--alpha` sets a fixed step instead.
- **Greedy ties go to the lowest-numbered cell.** Most greedy choices are ties early
  on, and a fixed choice keeps greedy play on the same boards so they accumulate
  updates. Exploration (ε) is random.
- **ε decays linearly from 1.0 to 0.05** over the run. Five alternatives (constant
  0.05 / 0.1 / 0.2, and 1.0 → 0.05 over the first 10% or 30%) were measured and were
  all worse on all 264 layouts at 200k and 500k episodes.
- Illegal actions are excluded from both action selection and the next-state max.
  Exploration and updates happen only with `training=True`; loaded models start in
  evaluation mode.

## Evaluation protocols

- `--protocol all` (default): train on all 264 layouts, report the final model on all
  264. This matches how every other agent is scored and is the game's real
  distribution. It is not a generalization test: a Q-table cannot play layouts it never
  saw.
- `--protocol held-out`: 184 training / 40 validation / 40 test layouts (split seed
  42); the checkpoint with the best validation mean is tested on the unseen 40. Use it
  to study generalization; scores are not comparable with the other agents.

## Commands

Run from the project root. Use new folder names to keep existing results.

```bash
uv run python -m battleship.q_learning train --episodes 1000000 --seeds 0 1 2 --eval-every 100000 --output battleship/q_learning/artifacts/run_1m
uv run python -m battleship.q_learning test battleship/q_learning/artifacts/run_1m --output battleship/q_learning/results/experiment_1m_repeat
uv run scripts/evaluate_all.py --q-table battleship/q_learning/artifacts/run_1m/seed_0.json.gz
```

The other experiments use `--episodes 1000 / 10000 / 100000` with `--eval-every 100 /
1000 / 10000`. Training shows a progress bar and writes to `artifacts/` (ignored by
Git): `config.json`, `seed_N.json.gz` (the reported model), `seed_N_final.json.gz`,
`seed_N_curve.csv` (evaluation curve) and `seed_N_rewards.csv` (every episode's
reward). Training is deterministic per seed; `save`/`load` keep update counts, so
training can continue exactly.

`test` writes the report folder: `README.md`, `config.json`, `test_results.json`,
`comparison.csv`, `test_games.csv`, `seed_N_curve.csv`, `summary.txt` and
`learning_curve.svg`. The per-episode `seed_N_rewards.csv` files are written there
too but are not committed (see `.gitignore`).

## Limitations

- A raw-board Q-table has no generalization between boards: about 2.2 million distinct
  boards each get only a few updates even after 1,000,000 episodes, which is why the
  agent stays well above the POMDP agent and the optimum.
- Results use one fixed hyperparameter setting; only the ε schedule was compared.

## Tests

`uv run pytest battleship/q_learning/tests` (20 tests): Bellman updates,
masking of illegal next actions, terminal updates, the lower-bound default, the
running-average step, tie-breaking, exploration, frozen evaluation, state encoding,
reproducibility, save/load (including old files), both evaluation protocols end to
end, and the decision counts.
