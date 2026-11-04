# Q-learning against placement habits

Completed all 12 requested seed/opponent runs: four opponents, three seeds each.

| Opponent | Raw-board Q | Belief Q | Opponent Belief Q | Optimal |
|---|---:|---:|---:|---:|
| random | 13.848 | 8.972 | 9.121 | 8.746 |
| edges | 13.722 | 9.514 | 8.381 | 8.209 |
| horizontal | 13.839 | 9.009 | 8.544 | 8.312 |
| favorites | 14.200 | 8.333 | 5.889 | 5.667 |

Training/history seeds: [0, 1, 2]. Habit history: 1000 games; opponent Q: 1000000 episodes per seed.

All three policies are evaluated on every layout with nonzero true opponent probability, weighted by that probability, with 10 common policy RNG seeds per layout. Layout integration is exact; random tie-breaking is sampled, not exact. Optimal is exact.

Raw Q and Belief Q reuse existing checkpoints; their training budgets, checkpoint selection and data splits differ. This is a comparison of saved systems, not a controlled ablation. The original raw Q test split is not preserved here: results cover the full opponent distribution. All layouts can appear during training. These are not unseen-layout generalization scores.

The opponent prior is learned from finished games and frozen before Q training. Favorites uses layout counts with smoothing 1; other opponents use placement counts with smoothing 3. The optimum knows the true distribution; learners do not. Each row averages model/history seeds. No significance claim is implied.

See config.json, per-seed comparison.csv and checkpoint hashes for provenance.
