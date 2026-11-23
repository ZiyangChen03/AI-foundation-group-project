# Reproducible comparison of the 4x4 agents

The manual experiment is complete. Results and analysis are in
[comparison_results_README.md](comparison_results_README.md), backed by the completed
`results/compare_4x4_recovered` artifacts. This document explains the protocol and
commands for a fresh run; historical PR #22 tables are not substituted for new data.

## Setup

Run from this checkout's root, not from another checkout of the repository:

```powershell
uv sync --extra dqn
uv run --extra dqn pytest -q
```

PyTorch is optional. Baselines and tabular agents work with `uv sync` and `uv run`.
Keep `--extra dqn` on DQN commands so uv does not remove the optional dependencies.
The original project uses Python 3.12; all new text/CSV/JSON output is UTF-8.

## Full experiment: matched interaction budgets

```powershell
uv run --extra dqn scripts/train_comparison_4x4.py --shots 1000000 --every 100000 --seeds 0 1 2 --output results/comparison_training
uv run --extra dqn scripts/compare_4x4.py --trained results/comparison_training --require-complete --samples 20 --games 400 --output results/compare_4x4_complete
```

The first command trains five methods sequentially, each with three independent
training seeds: raw-board Q-learning, belief-state Q-learning, and DQN board,
density-input and density-residual variants. The second includes them and nine
baselines/planners: random, hunt-target, hunt-target with parity, probability
density, and POMDP greedy, rollout, optimal, info gain, and greedy-info. It also
evaluates an oracle told each opponent habit. The oracle is not a learned agent
and does not participate in the uniform-placement tournament.

`--shots` is the number of **new environment transitions per method per seed**,
not episodes, gradient updates, or replay samples. Every shot costs -1. Training
stops exactly at the budget, even inside an unfinished game. Such a game's shots
count toward the budget; the game does not count as completed. DQN uses a partial
vector batch at checkpoint/budget boundaries and records actual shots and gradient
updates. Replayed transitions do not consume new interactions. The old DQN
game-budget training interface remains available.

One million shots is roughly tens of thousands of games, **not** the one million
episodes in some earlier PR tables. Use `--shots 10000000` with an appropriate
`--every` interval for a larger budget. No convergence or runtime is promised.

Training draws uniformly from all 264 layouts. Every method uses the same linear
exploration schedule from 1.0 to `--epsilon-end` (default .05), indexed by training
shots; DQN updates exploration once per vector batch. Q-learning uses the #18
lower-bound defaults and 1/N learning rate. Belief Q-learning uses its #15
pessimistic initialization and exact belief transitions, including eliminating
certain misses. DQN retains the repository network/replay/optimizer defaults;
its exploration fraction is set to 1.0 for this protocol. Exact hyperparameters,
the actual DQN device and PyTorch version are saved. These settings are specified
in advance, not chosen using the resulting scores.

Final checkpoints are reported. No best checkpoint is selected on the same 264
evaluation layouts. Learning curves evaluate at shot 0, every `--every` shots,
and at the final budget. Tabular curves use `--policy-samples` random policy
seeds per layout (default 3). DQN evaluation is deterministic and runs each
layout once. These are evaluation-performance curves, not training reward curves.
The comparison command uses its own `--samples` for final evaluation.

Training time excludes evaluation and saving. Tabular timing sums training
transitions, including game resets. DQN timing includes its training loop, replay
and updates, excluding evaluation but excluding initial network/buffer setup as
well. `runtime.json` separately records wall time including setup/evaluation/save.
Train sequentially and avoid competing CPU-heavy work when comparing timings.
Models load one at a time for solo evaluation and at most two at a time for matches.

`--require-complete` rejects missing methods, fewer than three training seeds,
different seed sets, unknown or unequal shot budgets, filtered agents, quick mode,
or omitted oracle/tournament stages. Without this flag, partial experiments are
allowed and their missing methods are listed in the generated report.

### Interrupted training

Repeat the identical training command with `--resume`. Completed method/seed runs
are reused only when parameters, source hashes and checkpoint hashes match.
An interrupted individual run is restarted from zero; optimizer/RNG state is not
resumed midway through that run. Do not alter source code between runs. Start a
new output directory when changing settings. Evaluations also require a fresh
output directory; existing reports are never silently overwritten. Manifests
remain `running` if a stage fails, and become `complete` only after successful export.

### Small functional checks

These check the pipeline, not learning quality:

```powershell
uv run scripts/compare_4x4.py --baselines-only --only random density --quick --skip-oracle --output results/comparison_smoke/baselines
uv run scripts/train_comparison_4x4.py --methods raw-q belief-q --shots 100 --every 100 --seeds 0 --policy-samples 1 --output results/comparison_smoke/training
uv run scripts/compare_4x4.py --trained results/comparison_smoke/training --only random raw-q belief-q --quick --skip-oracle --output results/comparison_smoke/evaluation
```

Without PyTorch, `uv run pytest -q` skips DQN-specific tests. With PyTorch,
`uv run --extra dqn pytest tests/test_comparison_4x4.py -q` also checks an exact
19-shot DQN budget with a partial final vector batch and real optimizer updates.
The default DQN trainer waits for 10,000 transitions before updating: a shorter
manual DQN smoke run tests plumbing but does not train a useful network.

## Compare saved models without retraining

### Recover a comparison interrupted during oracle evaluation

The oracle must only play layouts with nonzero probability under its habit.
In particular, the three-favorites oracle must not be tested on the other 261
layouts. The corrected evaluator filters zero-probability layouts while retaining
their original layout indices and RNG seeds.

If the earlier script stopped after all solo models were evaluated with
`ValueError: no layout explains this outcome`, reuse the saved solo stage:

```powershell
uv run --extra dqn scripts/compare_4x4.py --trained results/comparison_training --require-complete --reuse-solo results/compare_4x4_complete --output results/compare_4x4_recovered
```

This validates the original model/configuration hashes, layout/seed coverage and
mean shot counts, reproduces the example games, and copies the existing solo data
and timings into a new output directory. It then runs the corrected oracle and
the tournament. The original files and provenance are retained; both original
and recovery provenance are recorded in the new manifest. Training is not rerun.
The default 20 policy samples must match the saved run; otherwise supply its
original `--samples`. Solo recovery accepts changes to the two comparison files
containing the fix but rejects changes to other recorded agent/training sources.

After this recovery command completes, the report is
`results/compare_4x4_recovered/results.md`. A completed comparison is required for
final reporting; an old `running` manifest does not certify completion.

Checkpoints can come from existing experiments. Give every training seed explicitly
as `SEED=PATH`; a bare path gets its position in the argument list as its seed ID.
Seed IDs must agree across learned methods for head-to-head. Do not assign the same
checkpoint to several seeds and treat the duplicates as independent training.

```powershell
uv run --extra dqn scripts/compare_4x4.py --raw-q 0=path/to/raw_seed_0.json.gz --belief-q 0=path/to/belief_seed_0.json.gz --dqn-board 0=results/dqn/mini_board_s0/final.pt --samples 20 --games 400 --output results/compare_saved_models
```

The additional options `--dqn-density` and `--dqn-residual` accept the corresponding
**4x4** checkpoints. A 10x10 checkpoint fails the board check. Checkpoints are loaded
and validated before the experiment begins; an unavailable DQN dependency or missing
file fails rather than silently omitting that agent. External checkpoint budgets,
training time and curves are unknown, so this mode cannot pass `--require-complete`.
In particular, older held-out Q-learning models and all-layout models should not be
described as a controlled sample-efficiency comparison. Prefer the matched trainer
for that purpose.

## Evaluation and outputs

Every model plays every layout with the same policy RNG seeds. Effective RNG seed
is `layout_index * 1000003 + policy_seed`; policy seeds range from zero through
`--samples - 1`. Training and exploration are disabled during all evaluation and
matches. Every algorithm's uniform score averages its training seeds equally.

| File | Meaning |
|---|---|
| `results.md` | Generated tables, charts, protocol and limitations |
| `summary.csv` | Algorithm means, game spread, training-seed spread, observed worst, timings and gap to the exact optimum |
| `uniform.csv` | Separate statistics for each model/training seed, checkpoint size and warm-up/loading time |
| `games.csv` | Paired layout/policy-seed shot counts and decision times |
| `habits.csv` | Means under uniform, edge, center, horizontal, touching and favorite-layout distributions |
| `tournament.csv` | Algorithm win-rate matrix and aggregate rates |
| `tournament_pairs.csv` | Counts, wins and first-player wins per model pairing |
| `uniform.svg` | Uniform-placement comparison figure |
| `learning_curves.svg` | Curves copied from the matched training run, one line per method/seed |
| `cases.md`, `cases.json` | Efficient/inefficient examples, layouts, complete shot sequences and search-cost analysis |
| `manifest.json` | Run status, arguments, machine/Python details, source and checkpoint SHA-256 hashes |

`std` measures game variation including random decisions and variation between
trained models. `seed_std` separately measures variation between training-seed
means. A single training seed has no estimated seed spread. These quantities are
not confidence intervals. `worst` is the maximum observed in this run, not a proof
of worst-case performance. The decision timer covers `select_shot` only, including
deferred belief updates; it excludes `observe`, reset, model loading and one warm-up
game. Planner caches are process-global, so warm-up is not an independent cold-start
benchmark; its time is reported separately.

Habit evaluation uses exact layout probabilities to reweight the same games; it
does not adapt the models. Random policy decisions still have finite sampling error.
The oracle knows the distribution in advance, which is a different assumption from
learning the opponent's habits. The three-favorites distribution has support on
only three layouts, so its apparent differences are particularly sensitive.

For head-to-head, corresponding training seed IDs are paired. A seedless baseline
is played against every trained seed. `--games` is per model pairing, with an even
count and alternating first player. At the full defaults there are 14 algorithms
and **80,400 matches**, not the 18,000 in the old 10-agent comparison. The macro
overall win rate weights each opposing algorithm equally; the separate game-weighted
rate is also exported. Pairings use deterministic match seeds and all counts are
saved. `--skip-tournament` is useful for partial runs.

All solo agents eventually sink the fleet: example "unsuccessful" play means costly
search, not a lost solo game. Examples select each model's fewest/most-shot observed
games and show misses before/after the first hit and the shot at which each ship
sank. These are descriptive extremes, not typical performance or causal evidence.

Enumerating all layouts alone does **not** make a randomized strategy's result
exact. Increase `--samples` to reduce this noise. The Bellman optimum is calculated
separately for the gap column when the optimal baseline is included. For additional
exact board-only policy calculations from #20:

```powershell
uv run --extra dqn scripts/exact_4x4.py results/dqn/mini_board_s0/final.pt
```

## Interpreting the experiment

Use the learning curves to compare improvement per new environment interaction,
and training/decision times to compare computational cost. DQN can reuse each
interaction many times through replay. Belief methods have access to an exact
transition/belief model; density DQNs have handcrafted input features; raw-board
Q-learning has neither. A lower score therefore does not isolate the effect of
the learning algorithm alone. Small score gaps need more independent training
seeds and policy samples before claiming a stable ranking.

The training layouts and evaluation layouts are the same finite set. This measures
performance on a specified game distribution, not generalization to unseen layouts.
The trained opponent-belief Q-learning experiment in PR #16 is a separate
habit-specific protocol; this uniform-training comparison does not mix it into the
matched-budget table. Larger-board experiments are also outside this 4x4 tool.

## Source integration and uploading

This checkout is based on `main` at `df81146` and includes the required source files
from the following GitHub PR snapshots. Integration is local; it does not merge or
push any GitHub PR.

| Source | Snapshot | Included functionality |
|---|---|---|
| #18 (includes #14) | `6c0f1d7` | Fixed raw-board Q-learning, persistence and its tests |
| #15 (includes #19) | `14dad4e` | Belief-state Q-learning and its tests |
| #20 (on top of merged #17) | `cc8fd3b` | Optional DQN dependencies, corrected CLI/device handling and exact-policy tools |
| #21 | `04faaaa` | Information-gain planners and their tests |
| #22 | `428bc6d` | Comparison scope used as the starting point; the comparison script is replaced by this expanded implementation |

Upload the **integrated changes**, not only `compare_4x4.py`: the new shared module,
trainer, agent implementations, tests, README/docs, `pyproject.toml` and `uv.lock`
are needed together. Review with `git status` and `git diff`; the isolated branch is
`codex/complete-4x4-comparison`. Commit and push from this checkout when ready.

Keep `.venv`, local caches, and `results/comparison_training` out of Git. After your
manual run, upload generated `results/compare_4x4_complete` tables/charts and their
manifest if you want results in the repository. Large checkpoints can be shared
separately, matching the SHA-256 hashes. The full experiment results should only be
described as complete after both run manifests say `complete`.
