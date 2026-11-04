# Frozen opponent-prior Q-learning

This package lives at `battleship/opponent_belief_q/`, at the same package level
as the original `battleship/q_learning/` on its branch. This branch is based on
the belief-state Q-learning PR; it does not duplicate the raw Q training code.

```text
battleship/opponent_belief_q/
    __init__.py          public agent export
    agent.py             frozen-prior belief Q agent
    raw_q_checkpoint.py  original Q checkpoint reader for evaluation
    train.py             single-opponent experiment
    compare.py           four-opponent / three-Q comparison
    README.md            design, results protocol and commands
```

`OpponentBeliefQAgent` extends the original tabular `BeliefQAgent` on MINI.
It is a full-belief table, not linear feature approximation. Each table belongs
to one fixed positive layout prior. Changing the prior requires a new model.

The experiment collects completed games from an edge-preferring opponent using
the existing placement-count learner (smoothing 3), freezes its prior, then
samples hidden training layouts from that prior. Reward is -1, gamma is 1,
initial Q is -20, and alpha is 1/visits. No current hidden layout is given to
the shooting policy. The belief is updated from public feedback.

Run from this worktree:

```powershell
uv run python -m battleship.opponent_belief_q.train --output results/opponent_belief_q_run1
```

Defaults: 1000 history games, 1M Q episodes per seed, seeds 0/1/2, sequential
training, evaluation every 100k episodes. Existing uniform checkpoints are read
from `results/belief_q`; override with `--uniform-models`. Use `--strength 0`
as a uniform-opponent control. Output directories must be new or empty.

Outputs: curve.csv, seed_N.json.gz (prior plus Q table), report.json (settings,
learned prior and comparisons). Evaluation uses the true opponent distribution,
not the learned approximation. Q policy expectations integrate random ties
exactly; habit-greedy evaluation enumerates layouts and samples ties with three
policy seeds. All layouts can occur in training and evaluation: these are
distribution-level policy values, not unseen-layout generalization results.

History data collection is additional to Q training. Existing uniform models
retain their original training budgets, which must be checked when comparing
custom runs. No claim of superiority follows from a smoke test. Use several
training/history seeds and compare habit Q, uniform Q and habit greedy on the
same true distribution. The implementation currently handles stationary habits;
it does not update priors while reusing a trained table.

## Four-opponent, three-Q comparison

`battleship/opponent_belief_q/compare.py` compares the original raw-board Q checkpoints,
the uniform Belief Q checkpoints, and the learned-prior Q agent against random,
edge-preferring, horizontal-preferring, and three-favorite-layout opponents.
It uses the same `standard_habits` as the existing opponent-belief benchmark.
The favorites learner uses layout counts (smoothing 1); others use placement
counts (smoothing 3). The true opponent prior is used only for evaluation and
the exact oracle, never as the learner's prior.

From the downloaded worktree, with the existing local checkpoint directories:

```powershell
uv run python -m battleship.opponent_belief_q.compare --raw-models ../../battleship/q_learning/artifacts/run_1m --belief-models results/belief_q --reuse-edges results/opponent_belief_q_run1 --output results/q_habits_run1
```

This reuses the three completed edge models and trains nine new models (three
seeds for each remaining opponent). Both baseline model sets are loaded, not
retrained. All four rows are reevaluated, so sampled-tie scores may differ
slightly from the earlier exact Q evaluation. The default is 10 policy seeds
per layout, shared across all policies. Training is sequential to limit memory.

Outputs include a GitHub-ready `README.md` table, `comparison.csv` with per-seed
scores, `config.json` with baseline checkpoint hashes, per-run metadata, learning
curves and saved Q models. Reports update after every completed seed. A new or
empty output directory is required; automatic interrupted-run resume is not
implemented. A complete three-seed run is needed for the final table.

Checkpoints are intentionally ignored by Git. Commit code, documentation and
the finished report files; publish model files separately if needed. On another
machine provide `--raw-models` and `--belief-models` pointing to its checkpoints.
The raw reader supports historical non-symmetry checkpoint versions 1-4 and
rejects symmetry models explicitly. No external source checkout is imported.

Raw Q used a held-out-layout protocol and validation-selected checkpoints in
its original experiment. Here its saved policy is evaluated over the full true
opponent distribution, including training layouts. Belief Q trained on the full
uniform distribution; habit Q adds history games and learned-prior simulation.
Thus the table describes system performance, not an isolated causal effect of
state encoding. Inspect the copied raw training configuration in config.json.
