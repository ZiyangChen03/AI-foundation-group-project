# DQN benchmark: 4x4, all 264 layouts x 100 agent seeds (expected shots)

| Agent | Mean shots | ± 95% | Median | Hunting | Targeting | Paired vs density (95%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 14.165 | 0.023 | 15.0 | 3.38 | 10.78 | +5.249 ± 0.027 |
| hunt-target+parity | 9.656 | 0.024 | 10.0 | 5.00 | 4.66 | +0.740 ± 0.028 |
| probability-density | 8.916 | 0.015 | 9.0 | 4.28 | 4.63 | - |
| belief-greedy | 8.920 | 0.015 | 9.0 | 4.23 | 4.69 | +0.004 ± 0.011 |
| mini_board_s0/final | 8.924 | 0.000 | 9.0 | 4.02 | 4.91 | +0.008 ± 0.015 |
| mini_board_lr1e-4_s0/final | 8.962 | 0.000 | 9.0 | 3.84 | 5.12 | +0.046 ± 0.015 |
| optimal V* (exact, planning.py) | 8.746 | 0.000 | - | - | - | - |

56 s
