# Every 10x10 agent, seeds 0-199

| Agent | Mean shots | ± 95% | Median | Worst | Paired vs density (95%) | ms per shot |
|---|---:|---:|---:|---:|---:|---:|
| probability-density | 43.47 | 1.27 | 42.5 | 67 | - | 0.3 |
| belief-greedy | 45.20 | 1.29 | 44.0 | 69 | +1.74 ± 1.75 | 113.2 |
| belief-greedy-info | 45.55 | 1.33 | 45.0 | 68 | +2.08 ± 1.60 | 107.9 |
| belief-optimal-approx | 45.88 | 1.26 | 45.0 | 69 | +2.41 ± 1.60 | 300.8 |
| q-learning-local | 46.17 | 1.11 | 47.0 | 65 | +2.70 ± 1.61 | 0.4 |
| belief-q-particles | 46.45 | 1.31 | 46.0 | 72 | +2.98 ± 1.63 | 11.0 |
| belief-infogain | 46.62 | 1.27 | 46.0 | 71 | +3.15 ± 1.54 | 123.7 |
| hunt-target+parity | 50.02 | 1.26 | 50.5 | 69 | +6.55 ± 1.85 | 0.0 |
| random | 94.95 | 0.64 | 96.0 | 100 | +51.48 ± 1.40 | 0.0 |

557 s
