# AI-foundation-group-project
Battleship AI

## How to run

This project uses [uv](https://docs.astral.sh/uv/) so everyone shares the same environment (Python 3.12 + pinned dependencies in `uv.lock`).

1. Install uv (once):
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `winget install astral-sh.uv` (or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)
2. Clone the repo and run the game:
   ```bash
   uv run battleship_4x4.py
   ```
   uv will download Python 3.12, create `.venv/`, and install the locked dependencies automatically.
   For the 10×10 board: `uv run python -m battleship.ui --board classic`.

### Evaluate agents and run tests

```bash
uv run scripts/evaluate_all.py                                   # 4×4, every possible layout
uv run scripts/evaluate_all.py --board classic --games 2000      # 10×10
uv run scripts/evaluate_all.py --board classic --tournament 200  # plus a round robin
uv run pytest
```

Baseline results on 10×10 (2000 games, same layouts for every agent):

| Agent | Mean shots | Std | Worst |
|---|---|---|---|
| random | 95.3 | 5.0 | 100 |
| hunt-target | 59.9 | 15.0 | 100 |
| hunt-target + parity | 50.8 | 8.9 | 75 |
| probability density | 44.7 | 9.1 | 73 |

Baseline results on 4×4 (2000 games, same layouts for every agent):

| Agent | Mean shots | Std | Worst |
|---|---|---|---|
| random | 14.1 | 1.9 | 16 |
| hunt-target | 10.3 | 2.6 | 16 |
| hunt-target + parity | 9.7 | 2.0 | 15 |
| probability density | 8.9 | 1.6 | 13 |

### Belief-based POMDP agent

`BeliefAgent` keeps the set of layouts consistent with every shot (the POMDP belief)
and fires at the most likely cell; on 4×4 it can also solve the belief-state MDP
exactly (`planner="optimal"`). 4×4, sunk cells revealed, ships placed uniformly at
random (as the runner does), exact expected shots:

| Agent | Shots |
|---|---|
| belief, optimal planner | 8.746 |
| belief, rollout planner | 8.765 |
| belief, greedy (default) | 8.913 |
| probability density | 8.92 |

Without revealed sunk cells, probability density drops to 10.13 while the belief agent
stays at 8.91. Design, verification and 10×10 results:
[docs/design_belief_pomdp.md](docs/design_belief_pomdp.md).

Watch an agent play, with its belief shown on the board:

```bash
uv run battleship_4x4.py --agent belief      # or optimal, rollout, density, hunt, random
uv run battleship_4x4.py --agent optimal --show q   # Q* (expected shots left) on each square,
                                                    # one shot per Space / click (--auto: play on)
uv run battleship_4x4.py --agent belief --step      # any agent, one shot at a time
uv run battleship_4x4.py --agent infogain --show i  # information gain (bits) on each square
uv run scripts/show_q_values.py --play 12    # the same numbers in the terminal, move by move
uv run scripts/verify_belief.py              # checks for the POMDP agent (~5 min)
```

`evaluate_all.py` includes `BeliefAgent` on 4×4 (fast). On 10×10 it takes ~60 ms per
shot, so it is left out unless you pass `--belief`.

### Opponent belief: learning placement habits

`OpponentBeliefAgent` is the POMDP agent with a **learned prior**. After each game it
counts where the opponent put its ships, and it weights the layouts it keeps by those
counts. Against opponents with habits on 4×4 (exact expected shots, after 1000 games):

| Opponent | belief (uniform prior) | opponent belief | optimum against the habit |
|---|---|---|---|
| random (control) | 8.905 | 8.922 | 8.746 |
| hugs the edges | 9.532 | 8.281 | 8.209 |
| ships along rows | 8.874 | 8.416 | 8.312 |
| reuses 3 boards (`model="layouts"`) | 7.889 | 5.667 | 5.667 |

```bash
uv run scripts/verify_opponent_belief.py     # checks + benchmark (~1-2 min)
```

**Not effective on 10×10 as implemented.** The agent runs there with weighted
particles, but after 1000 boards the learned prior is still within noise of the
uniform one. There are two reasons:
- Mild habits are worth only 1–2 of ~45 shots, even to an agent told the true habit.
- Counting 600 placements separately learns too slowly.

Design, the two habit models, verification and full learning curves:
[docs/design_opponent_belief.md](docs/design_opponent_belief.md).

### Every agent on 10×10

Some agents need things that only exist on small boards: the exact list of
layouts (264 on 4×4, about 30 billion on 10×10) or a Q-table over whole boards.
Their 10×10 versions are separate classes; the 4×4 originals are unchanged.

- **Information gain and greedy-info** (`battleship/agents/belief_10x10.py`):
  each shot's result chances come from the 1,000 sampled layouts of the particle
  belief instead of the full list (within 0.04 bits of the exact value on 4×4).
- **Optimal, approximated**: Q\* can't be computed on 10×10, so `optimal-approx`
  uses the sampled lookahead (rollout) on every shot. It is an approximation,
  not the optimum.
- **The two Q-tables** (`battleship/agents/q_10x10.py`, trained with
  `scripts/train_q_10x10.py`): the raw-board table is keyed on each square's
  3×3 neighbourhood, the belief-state table on what the particle belief says
  about the square. Both use −1 per miss and γ = 0; with the originals' −1 per
  shot and γ = 1 the neighbourhood table stayed near random (92.3 shots).

The same 200 boards for every agent (seeds 0–199, `results/benchmark_10x10.md`):

| Agent | Mean shots | Paired vs density |
|---|---|---|
| probability density | 43.47 | – |
| belief, greedy | 45.20 | +1.74 ± 1.75 |
| belief, greedy-info (particles) | 45.55 | +2.08 ± 1.60 |
| belief, optimal-approx (rollout) | 45.88 | +2.41 ± 1.60 |
| Q-table on 3×3 neighbourhoods | 46.17 | +2.70 ± 1.61 |
| Q-table on the particle belief | 46.45 | +2.98 ± 1.63 |
| belief, information gain (particles) | 46.62 | +3.15 ± 1.54 |
| hunt-target + parity | 50.02 | +6.55 ± 1.85 |
| random | 94.95 | +51.48 ± 1.40 |

DQN on 10×10 is in the next section (45.85 for the best variant, on 2,000 boards).

```bash
uv run scripts/train_q_10x10.py local --games 20000     # ~3 min
uv run scripts/train_q_10x10.py belief --games 400      # ~3 min
uv run scripts/benchmark_10x10.py --games 200           # ~10 min on 12 cores
uv run battleship_4x4.py --board classic --agent optimal   # watch the approximation
```

### DQN: deep Q-learning

`battleship.dqn.DQNAgent` is Q-learning with a convolutional network, trained on 256
vectorized games at once (Double DQN, dueling head, replay buffer, target network).
It is a separate package, so the other agents do not load PyTorch. Results on the
2,000 test boards (seeds 0–1999, paired against probability density):

| Agent | 10×10 shots | Paired vs density | Training |
|---|---|---|---|
| probability density | 44.65 | – | – |
| DQN-B (board + density input) | 45.85 | +1.20 ± 0.46 | 14 min |
| DQN-C (residual on density) | 46.05 | +1.41 ± 0.46 | 14 min |
| DQN-A (board only) | 53.00 | +8.35 ± 0.54 | 12 min |

On 4×4 the historical board-only DQN final checkpoint reaches 8.924 shots in
75 s of training. Exact random-tie probability density is 8.909; the Bellman
optimum is 8.746. These historical runs use different training budgets from the
matched comparison below. See [docs/design_dqn.md](docs/design_dqn.md).

```bash
uv sync --extra dqn
uv run --extra dqn scripts/train_dqn.py --board classic --variant density --games 100000 --eps-fraction 0.1
uv run --extra dqn scripts/benchmark_dqn.py classic results/dqn/classic_density_s0/best.pt
uv run --extra dqn pytest tests/test_dqn.py
```

Design and results: [docs/design_dqn.md](docs/design_dqn.md). Beginner tutorial:
[tutorial/dqn_tutorial.html](tutorial/dqn_tutorial.html). Two-page report:
[report/dqn_report.pdf](report/dqn_report.pdf).

The existing PDF predates the corrections in #20 and has not been rebuilt here.
Use the design document for the corrected 4×4 numbers. PyTorch is an optional
extra; non-DQN code does not require it.

### Complete 4×4 comparison

Completed experiment and analysis: [docs/comparison_results_README.md](docs/comparison_results_README.md).
The run covers 126,720 solo evaluation games and 80,400 tournament matches.
Final data: [results/compare_4x4_recovered/results.md](results/compare_4x4_recovered/results.md).

The integrated comparison includes raw-board Q-learning, belief-state Q-learning,
all three DQN variants, four heuristic baselines and five POMDP planners. It
supports uniform placement, opponent placement habits and head-to-head matches.
**The manual experiment is complete.** Both training and recovered-comparison manifests
report `complete`; all 15 model hashes and the game/match coverage were validated.
The commands below reproduce a fresh experiment and require new/empty output directories.

From the root of this checkout:

```powershell
uv sync --extra dqn
uv run --extra dqn scripts/train_comparison_4x4.py --shots 1000000 --every 100000 --seeds 0 1 2 --output results/comparison_training
uv run --extra dqn scripts/compare_4x4.py --trained results/comparison_training --require-complete --samples 20 --games 400 --output results/compare_4x4_complete
```

Observed final means across three training seeds: raw-board Q-learning **12.0379**,
belief-state Q-learning **9.6768**, DQN board **8.9369**, DQN density **8.8763**,
and DQN residual **8.8460** shots. Probability density scores **8.9000**.
The exact Bellman optimum is **8.746212**; its sampled policy score is **8.713636**,
which reflects finite random-choice sampling and does not beat the exact optimum.
See the completed analysis for learning curves, timings, habit results and limitations.

This run reused validated solo results after fixing the oracle's zero-probability
layout handling. Its completed report is in `results/compare_4x4_recovered/`;
`results/compare_4x4_complete/` preserves the original interrupted run.

Each learning method receives the same number of **training shots**, including
shots in unfinished games. This is not an episode budget. Models are trained
sequentially and final checkpoints are evaluated with learning disabled. All 264
layouts are covered; random choices are sampled and are not described as exact
expectations. `--require-complete` verifies all five learners, at least three
matched training seeds, equal budgets and all evaluation stages.

The generated `results/compare_4x4_complete/results.md` includes means, game and
training-seed spread, observed worst cases, training/decision times, placement
habits, learning curves and tournament results. CSV data, source/model hashes and
example game sequences are exported alongside it. The full default tournament
contains 80,400 games across model-seed pairings.

To compare saved checkpoints without retraining, or run a small functional check,
see [docs/comparison_4x4.md](docs/comparison_4x4.md). That guide also explains
resuming training, algorithm assumptions and which integrated files to upload.
Local training models are ignored by Git; generated comparison reports can be
committed after you run the experiment.

```powershell
uv run --extra dqn pytest -q
```

### Adding a dependency

```bash
uv add <package>
```

Then commit the updated `pyproject.toml` and `uv.lock`. Never commit `.venv/`.

## Code layout

The `battleship/` package holds the environment, the agents, evaluation, and the
tournament. See [docs/API_DESIGN.md](docs/API_DESIGN.md) for the APIs and how to
add a new agent. The baselines and how they were verified are described in
[docs/design_hunt_and_target.md](docs/design_hunt_and_target.md) and
[docs/design_probability_density.md](docs/design_probability_density.md).
