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
