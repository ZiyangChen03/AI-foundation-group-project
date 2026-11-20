# DQN results

One folder per training run (`<board>_<variant>_s<seed>`), each with `log.txt`,
`curve.csv` (the validation curve), `summary.json` (settings and scores), and two
checkpoints: `best.pt` (best validation score) and `final.pt` (end of training).
How they were trained: [docs/design_dqn.md](../../docs/design_dqn.md), section
"Reproduce".

## Which checkpoint the tables use

| Run | Reported | Why | The other checkpoint |
|---|---|---|---|
| `classic_board_s0` | `best.pt` (15k games) | Picked on validation boards (seeds 100000+), separate from the test boards | `final.pt`: 100k games, worse (55.8 validation against 52.4); shows the instability in section 5 |
| `classic_density_s0` | `best.pt` | Same rule | `final.pt` has the same weights (the best was the last evaluation) |
| `classic_residual_s0` | `final.pt` | Its best was the untrained network, which always fires at a highest-density cell | `best.pt` is that untrained network |
| `mini_board_s0` | `final.pt` | On 4×4 the validation boards are the test boards (all 264 layouts), so `best.pt` was picked by its test score | `best.pt`: 8.830 exact; do not report it as a result |
| `mini_board_lr1e-4_s0` | `final.pt` | Same as above | `best.pt`: 8.932 exact; same caveat |

The tables are `benchmark_classic.md` (2,000 test boards, seeds 0–1999) and
`benchmark_mini.md` (all 264 layouts); `scripts/benchmark_dqn.py` regenerates them.
