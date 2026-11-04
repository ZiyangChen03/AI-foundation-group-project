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
| random (control) | 8.905 | 8.917 | 8.746 |
| hugs the edges | 9.532 | 8.281 | 8.209 |
| ships along rows | 8.874 | 8.415 | 8.312 |
| reuses 3 boards (`model="layouts"`) | 7.889 | 5.667 | 5.667 |

```bash
uv run scripts/verify_opponent_belief.py     # checks + benchmark (~1-2 min)
```

Design, the two habit models, verification and full learning curves:
[docs/design_opponent_belief.md](docs/design_opponent_belief.md).

### Belief-state Q-learning (4×4)

`BeliefQAgent` is tabular Q-learning whose state is the exact POMDP belief (the
set of layouts consistent with every shot) rather than the raw board. Evaluated
exactly over all 264 layouts, three seeds average **8.972 shots** after
1,000,000 episodes (POMDP greedy 8.913, optimal 8.746, hunt-target + parity 9.697).

```bash
uv run scripts/train_belief_q.py --jobs 3
```

Design and results: [docs/design_belief_q_learning.md](docs/design_belief_q_learning.md).

### Q-learning: learning placement habits (4×4)

Three Q-learning variants are compared on the same opponent distributions:
**raw-board Q-learning** uses the visible board as its state;
**Belief Q-learning** uses the exact belief with a uniform placement prior;
**Opponent Belief Q-learning** learns a placement prior from 1,000 finished
opponent games, freezes it, and trains a dedicated belief-state Q-table for
1,000,000 episodes. All use MINI (ships of lengths 3 and 2, sunk cells revealed).

Completed results for seeds **0, 1, 2** (12 opponent/seed runs). Values are
mean shots to sink the fleet; **lower is better**. Bold marks the best Q-learning
variant, not the oracle optimum.

| Opponent | Raw-board Q-learning | Belief Q-learning | Opponent Belief Q-learning | Optimum against the habit |
|---|---:|---:|---:|---:|
| Random (control) | 13.848 | **8.972** | 9.121 | 8.746 |
| Hugs the edges | 13.722 | 9.514 | **8.381** | 8.209 |
| Ships along rows | 13.839 | 9.009 | **8.544** | 8.312 |
| Reuses 3 boards (`model="layouts"`) | 14.200 | 8.333 | **5.889** | 5.667 |

Compared with Belief Q-learning, the learned-prior version saves **1.134 shots
(11.9%)** against edge placement, **0.465 (5.2%)** against horizontal placement,
and **2.444 (29.3%)** against repeated boards. Against the random control it costs
**0.149 additional shots (1.7%)**. These runs show benefits against the tested
habits, but no universal advantage or statistically established superiority.
The random-control result alone does not separate prior-estimation error from
Q-training variation.

**Evaluation:** every layout with nonzero true opponent probability is evaluated
and weighted by that probability, using the same 10 policy RNG seeds per layout
for all three variants. Layout integration is exact; random tie-breaking is
sampled. Table entries average the three model/history seeds. The optimum is
computed exactly with the true opponent prior; learners do not receive that prior.
Repeated boards use layout counts with smoothing 1; the other learned priors use
single-ship placement counts with smoothing 3.

**Comparison scope:** the raw-board and uniform-belief models are reused, while
the opponent models add history collection and habit-specific training. Raw-board
checkpoints were validation-selected from runs with a 1M-episode budget and a
layout train/validation/test split; Belief Q uses final 1M-episode models trained
on the full uniform layout distribution. This table evaluates all models on the
full opponent distributions, including training layouts. It is a comparison of
saved systems, not a controlled ablation or held-out-layout generalization test.
The earlier greedy opponent-belief table uses a separate protocol and is not an
additional column in this experiment.

To reproduce with existing baseline checkpoint folders (use a new output folder):

```bash
uv run python -m battleship.opponent_belief_q.compare --raw-models PATH_TO_RAW_Q_CHECKPOINTS --belief-models results/belief_q --output results/q_habits_repeat
```

Add `--reuse-edges results/opponent_belief_q_run1` to reuse the completed edge
models with matching settings. Each baseline folder must contain
`seed_0.json.gz`, `seed_1.json.gz`, and `seed_2.json.gz`. Model checkpoints remain
local and are ignored by Git.

Sources: [completed report](results/q_habits_run1/README.md),
[per-seed results](results/q_habits_run1/comparison.csv),
[configuration and checkpoint hashes](results/q_habits_run1/config.json).
Implementation, commands and limitations:
[battleship/opponent_belief_q/README.md](battleship/opponent_belief_q/README.md).

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
