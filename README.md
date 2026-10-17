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

### Adding a dependency

```bash
uv add <package>
```

Then commit the updated `pyproject.toml` and `uv.lock`. Never commit `.venv/`.
