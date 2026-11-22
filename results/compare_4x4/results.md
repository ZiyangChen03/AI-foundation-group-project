# 4x4 comparison

All 264 layouts x 3 agent seed(s); tournament 400 games per pair. Exact optimum under random placement: 8.746.

## Random placement

| Agent | Mean shots | Std | Worst | Gap to optimum |
|---|---:|---:|---:|---:|
| POMDP optimal | 8.746 | 1.61 | 12 | +0.000 |
| POMDP rollout | 8.764 | 1.61 | 12 | +0.018 |
| POMDP greedy + info tie-break | 8.895 | 1.60 | 12 | +0.149 |
| Probability density | 8.904 | 1.60 | 12 | +0.158 |
| POMDP greedy | 8.905 | 1.61 | 12 | +0.159 |
| Belief-state Q-learning | 8.932 | 1.71 | 13 | +0.186 |
| POMDP info gain | 8.951 | 1.75 | 14 | +0.205 |
| Hunt-target + parity | 9.676 | 1.97 | 14 | +0.929 |
| Hunt-target | 10.279 | 2.48 | 16 | +1.533 |
| Random | 14.314 | 1.91 | 16 | +5.568 |

## Against opponents with placement habits (expected shots)

| Agent | uniform | edges | center | horizontal | touching | 3 favorites |
|---|---:|---:|---:|---:|---:|---:|
| Random | 14.314 | 14.316 | 14.235 | 14.284 | 14.331 | 14.444 |
| Hunt-target | 10.279 | 10.259 | 10.152 | 10.344 | 9.996 | 10.667 |
| Hunt-target + parity | 9.676 | 9.773 | 9.578 | 9.644 | 9.339 | 8.667 |
| Probability density | 8.904 | 9.487 | 8.110 | 8.865 | 8.621 | 8.333 |
| POMDP greedy | 8.905 | 9.532 | 8.093 | 8.874 | 8.617 | 7.889 |
| POMDP rollout | 8.764 | 9.359 | 7.823 | 8.690 | 8.395 | 8.333 |
| POMDP optimal | 8.746 | 9.336 | 7.861 | 8.737 | 8.386 | 8.222 |
| POMDP info gain | 8.951 | 9.630 | 8.075 | 8.891 | 8.523 | 8.222 |
| POMDP greedy + info tie-break | 8.895 | 9.501 | 8.093 | 8.863 | 8.595 | 7.889 |
| Belief-state Q-learning | 8.932 | 9.493 | 8.106 | 8.901 | 8.532 | 11.000 |
| POMDP greedy, told the habit (oracle) | 8.905 | 8.297 | 7.811 | 8.378 | 8.142 | 5.667 |

## Head-to-head (win rate of row against column)

| Agent | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. Random | - | 0.14 | 0.05 | 0.03 | 0.04 | 0.03 | 0.03 | 0.04 | 0.04 | 0.03 | 0.05 |
| 2. Hunt-target | 0.86 | - | 0.45 | 0.35 | 0.35 | 0.35 | 0.30 | 0.36 | 0.34 | 0.40 | 0.42 |
| 3. Hunt-target + parity | 0.95 | 0.56 | - | 0.38 | 0.40 | 0.37 | 0.34 | 0.44 | 0.40 | 0.39 | 0.47 |
| 4. Probability density | 0.97 | 0.65 | 0.62 | - | 0.49 | 0.46 | 0.45 | 0.53 | 0.48 | 0.45 | 0.57 |
| 5. POMDP greedy | 0.96 | 0.65 | 0.60 | 0.51 | - | 0.47 | 0.41 | 0.53 | 0.49 | 0.53 | 0.57 |
| 6. POMDP rollout | 0.97 | 0.65 | 0.63 | 0.54 | 0.53 | - | 0.46 | 0.49 | 0.49 | 0.52 | 0.59 |
| 7. POMDP optimal | 0.97 | 0.70 | 0.66 | 0.55 | 0.59 | 0.54 | - | 0.53 | 0.53 | 0.52 | 0.62 |
| 8. POMDP info gain | 0.96 | 0.64 | 0.56 | 0.47 | 0.47 | 0.51 | 0.47 | - | 0.49 | 0.46 | 0.56 |
| 9. POMDP greedy + info tie-break | 0.96 | 0.66 | 0.60 | 0.52 | 0.51 | 0.51 | 0.47 | 0.51 | - | 0.52 | 0.58 |
| 10. Belief-state Q-learning | 0.97 | 0.60 | 0.61 | 0.55 | 0.47 | 0.48 | 0.48 | 0.54 | 0.48 | - | 0.58 |

First player won 10126 of 18000 games.
