# DQN benchmark: 10x10, seeds 0-1999

| Agent | Mean shots | ± 95% | Median | Hunting | Targeting | Paired vs density (95%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 95.270 | 0.22 | 97.0 | 5.94 | 89.33 | +50.62 ± 0.45 |
| hunt-target+parity | 50.761 | 0.39 | 51.0 | 32.93 | 17.83 | +6.11 ± 0.55 |
| probability-density | 44.648 | 0.40 | 44.0 | 27.46 | 17.19 | - |
| belief-greedy | 44.502 | 0.41 | 44.0 | 27.32 | 17.19 | -0.15 ± 0.51 |
| classic_board_s0/best | 52.998 | 0.45 | 52.0 | 17.68 | 35.31 | +8.35 ± 0.54 |
| classic_density_s0/best | 45.847 | 0.32 | 46.0 | 17.09 | 28.76 | +1.20 ± 0.46 |
| classic_residual_s0/final | 46.054 | 0.32 | 46.0 | 19.69 | 26.36 | +1.41 ± 0.46 |

274 s
