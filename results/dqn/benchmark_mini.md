# DQN benchmark: 4x4, all 264 layouts (exact for deterministic agents)

| Agent | Mean shots | ± 95% | Median | Hunting | Targeting | Paired vs density (95%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 14.227 | 0.22 | 15.0 | 3.58 | 10.65 | +5.36 ± 0.27 |
| hunt-target+parity | 9.697 | 0.26 | 10.0 | 4.99 | 4.70 | +0.83 ± 0.32 |
| probability-density | 8.864 | 0.19 | 9.0 | 4.24 | 4.62 | - |
| belief-greedy | 8.879 | 0.19 | 9.0 | 4.21 | 4.67 | +0.02 ± 0.12 |
| mini_board_s0/final | 8.924 | 0.19 | 9.0 | 4.02 | 4.91 | +0.06 ± 0.26 |
| mini_board_lr1e-4_s0/final | 8.962 | 0.20 | 9.0 | 3.84 | 5.12 | +0.10 ± 0.24 |

2 s
