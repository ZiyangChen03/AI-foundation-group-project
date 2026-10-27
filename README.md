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
