# Example games

Every solo game finishes. These are efficient and inefficient examples, not win/loss labels.
Examples are descriptive extrema from this run, not typical-performance estimates.

## Random / training seed None / fewest shots

Layout 76, policy seed 15: 5 shots, 0 misses.
Ships: C1, C2, C3; D2, D3

C3 hit → C2 hit → D2 hit → D3 sunk → C1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Random / training seed None / most shots

Layout 0, policy seed 9: 16 shots, 11 misses.
Ships: A1, B1, C1; D1, D2

C4 miss → B3 miss → B2 miss → A2 miss → C1 hit → D1 hit → A1 hit → D3 miss → D4 miss → C3 miss → B1 sunk → A3 miss → C2 miss → B4 miss → A4 miss → D2 sunk

The first hit occurs at shot 5: 4 misses occur before it and 7 occur afterwards. Ships are sunk at shots 11, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Hunt-target / training seed None / fewest shots

Layout 0, policy seed 6: 5 shots, 0 misses.
Ships: A1, B1, C1; D1, D2

C1 hit → D1 hit → B1 hit → A1 sunk → D2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Hunt-target / training seed None / most shots

Layout 6, policy seed 0: 16 shots, 11 misses.
Ships: A1, B1, C1; C2, C3

A4 miss → B2 miss → D1 miss → C4 miss → D3 miss → A1 hit → A2 miss → B1 hit → C1 sunk → B3 miss → D2 miss → D4 miss → B4 miss → A3 miss → C2 hit → C3 sunk

The first hit occurs at shot 6: 5 misses occur before it and 6 occur afterwards. Ships are sunk at shots 9, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Hunt-target + parity / training seed None / fewest shots

Layout 0, policy seed 6: 5 shots, 0 misses.
Ships: A1, B1, C1; D1, D2

C1 hit → D1 hit → B1 hit → A1 sunk → D2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Hunt-target + parity / training seed None / most shots

Layout 255, policy seed 6: 16 shots, 11 misses.
Ships: B4, C4, D4; B2, C2

C3 miss → D2 miss → C1 miss → A3 miss → B2 hit → B3 miss → B1 miss → A2 miss → C2 sunk → D1 miss → A4 miss → A1 miss → D4 hit → D3 miss → C4 hit → B4 sunk

The first hit occurs at shot 5: 4 misses occur before it and 7 occur afterwards. Ships are sunk at shots 9, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Probability density / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Probability density / training seed None / most shots

Layout 219, policy seed 13: 13 shots, 8 misses.
Ships: B3, C3, D3; D1, D2

B2 miss → C3 hit → C2 miss → B3 hit → D3 sunk → C1 miss → A2 miss → B4 miss → A4 miss → C4 miss → A1 miss → D2 hit → D1 sunk

The first hit occurs at shot 2: 1 misses occur before it and 7 occur afterwards. Ships are sunk at shots 5, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP greedy / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP greedy / training seed None / most shots

Layout 219, policy seed 13: 13 shots, 8 misses.
Ships: B3, C3, D3; D1, D2

B2 miss → C3 hit → C2 miss → B3 hit → D3 sunk → C1 miss → A2 miss → B4 miss → A4 miss → C4 miss → A1 miss → D2 hit → D1 sunk

The first hit occurs at shot 2: 1 misses occur before it and 7 occur afterwards. Ships are sunk at shots 5, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP rollout / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP rollout / training seed None / most shots

Layout 5, policy seed 12: 12 shots, 7 misses.
Ships: A1, B1, C1; C2, D2

C3 miss → B4 miss → B2 miss → C1 hit → D1 miss → B1 hit → A3 miss → D3 miss → C4 miss → C2 hit → D2 sunk → A1 sunk

The first hit occurs at shot 4: 3 misses occur before it and 4 occur afterwards. Ships are sunk at shots 11, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP optimal / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP optimal / training seed None / most shots

Layout 5, policy seed 8: 12 shots, 7 misses.
Ships: A1, B1, C1; C2, D2

B2 miss → C3 miss → C1 hit → D1 miss → A3 miss → B4 miss → D4 miss → D2 hit → B1 hit → D3 miss → A1 sunk → C2 sunk

The first hit occurs at shot 3: 2 misses occur before it and 5 occur afterwards. Ships are sunk at shots 11, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP info gain / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → A1 hit → B1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP info gain / training seed None / most shots

Layout 1, policy seed 3: 14 shots, 9 misses.
Ships: A1, B1, C1; A2, B2

B3 miss → C2 miss → C4 miss → D2 miss → A4 miss → A3 miss → D1 miss → D4 miss → C3 miss → A2 hit → A1 hit → B1 hit → B2 sunk → C1 sunk

The first hit occurs at shot 10: 9 misses occur before it and 0 occur afterwards. Ships are sunk at shots 13, 14. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP greedy + info tie-break / training seed None / fewest shots

Layout 4, policy seed 7: 5 shots, 0 misses.
Ships: A1, B1, C1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## POMDP greedy + info tie-break / training seed None / most shots

Layout 219, policy seed 13: 13 shots, 8 misses.
Ships: B3, C3, D3; D1, D2

B2 miss → C3 hit → C2 miss → B3 hit → D3 sunk → C1 miss → A2 miss → B4 miss → A4 miss → C4 miss → A1 miss → D2 hit → D1 sunk

The first hit occurs at shot 2: 1 misses occur before it and 7 occur afterwards. Ships are sunk at shots 5, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 0 / fewest shots

Layout 0, policy seed 0: 5 shots, 0 misses.
Ships: A1, B1, C1; D1, D2

B1 hit → C1 hit → D1 hit → A1 sunk → D2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 0 / most shots

Layout 96, policy seed 0: 16 shots, 11 misses.
Ships: D1, D2, D3; B3, C3

B1 miss → A2 miss → D1 hit → D4 miss → B2 miss → C4 miss → A1 miss → D2 hit → C1 miss → C2 miss → D3 sunk → A4 miss → B3 hit → B4 miss → A3 miss → C3 sunk

The first hit occurs at shot 3: 2 misses occur before it and 9 occur afterwards. Ships are sunk at shots 11, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 1 / fewest shots

Layout 36, policy seed 0: 5 shots, 0 misses.
Ships: B1, C1, D1; A1, A2

A1 hit → B1 hit → C1 hit → D1 sunk → A2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 1 / most shots

Layout 142, policy seed 0: 16 shots, 11 misses.
Ships: B2, C2, D2; B3, C3

A1 miss → B4 miss → B1 miss → D1 miss → C1 miss → B2 hit → A2 miss → C2 hit → A4 miss → A3 miss → D4 miss → D2 sunk → C3 hit → C4 miss → D3 miss → B3 sunk

The first hit occurs at shot 6: 5 misses occur before it and 6 occur afterwards. Ships are sunk at shots 12, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 2 / fewest shots

Layout 2, policy seed 0: 5 shots, 0 misses.
Ships: A1, B1, C1; A2, A3

A1 hit → A2 hit → A3 sunk → C1 hit → B1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 3, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Raw-board Q-learning / training seed 2 / most shots

Layout 50, policy seed 0: 16 shots, 11 misses.
Ships: B1, C1, D1; D3, D4

A1 miss → B1 hit → D1 hit → C2 miss → C1 sunk → B2 miss → B4 miss → A2 miss → A4 miss → A3 miss → D2 miss → B3 miss → C3 miss → D4 hit → C4 miss → D3 sunk

The first hit occurs at shot 2: 1 misses occur before it and 10 occur afterwards. Ships are sunk at shots 5, 16. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 0 / fewest shots

Layout 44, policy seed 0: 5 shots, 0 misses.
Ships: B1, C1, D1; A3, B3

A3 hit → C1 hit → D1 hit → B1 sunk → B3 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 0 / most shots

Layout 262, policy seed 0: 14 shots, 9 misses.
Ships: B4, C4, D4; B3, C3

A3 miss → D2 miss → D3 miss → B2 miss → D1 miss → A1 miss → C1 miss → C3 hit → C4 hit → A4 miss → B4 hit → C2 miss → B3 sunk → D4 sunk

The first hit occurs at shot 8: 7 misses occur before it and 2 occur afterwards. Ships are sunk at shots 13, 14. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 1 / fewest shots

Layout 82, policy seed 0: 5 shots, 0 misses.
Ships: C1, C2, C3; B4, C4

C4 hit → B4 sunk → C1 hit → C3 hit → C2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 1 / most shots

Layout 95, policy seed 0: 14 shots, 9 misses.
Ships: D1, D2, D3; A3, A4

C4 miss → B2 miss → D3 hit → B3 miss → C3 miss → A2 miss → C1 miss → A1 miss → A4 hit → D2 hit → D4 miss → B4 miss → D1 sunk → A3 sunk

The first hit occurs at shot 3: 2 misses occur before it and 7 occur afterwards. Ships are sunk at shots 13, 14. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 2 / fewest shots

Layout 12, policy seed 0: 5 shots, 0 misses.
Ships: A1, B1, C1; C3, D3

A1 hit → C3 hit → D3 sunk → B1 hit → C1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 3, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## Belief-state Q-learning / training seed 2 / most shots

Layout 2, policy seed 0: 14 shots, 9 misses.
Ships: A1, B1, C1; A2, A3

A1 hit → C3 miss → B3 miss → B4 miss → D3 miss → B2 miss → C2 miss → C4 miss → D1 miss → A3 hit → B1 hit → C1 sunk → A4 miss → A2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 9 occur afterwards. Ships are sunk at shots 12, 14. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 0 / fewest shots

Layout 162, policy seed 0: 5 shots, 0 misses.
Ships: B2, B3, B4; C3, C4

B3 hit → B4 hit → C4 hit → B2 sunk → C3 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 0 / most shots

Layout 116, policy seed 0: 13 shots, 8 misses.
Ships: A2, B2, C2; C4, D4

B3 miss → A2 hit → C2 hit → C3 miss → A3 miss → A1 miss → B4 miss → B2 sunk → D3 miss → D1 miss → B1 miss → D4 hit → C4 sunk

The first hit occurs at shot 2: 1 misses occur before it and 7 occur afterwards. Ships are sunk at shots 8, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 1 / fewest shots

Layout 40, policy seed 0: 5 shots, 0 misses.
Ships: B1, C1, D1; B2, B3

B2 hit → B3 sunk → C1 hit → B1 hit → D1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 1 / most shots

Layout 198, policy seed 0: 13 shots, 8 misses.
Ships: A3, B3, C3; A1, B1

B2 miss → C1 miss → D2 miss → A2 miss → A4 miss → C3 hit → C2 miss → D3 miss → C4 miss → B3 hit → A3 sunk → B1 hit → A1 sunk

The first hit occurs at shot 6: 5 misses occur before it and 3 occur afterwards. Ships are sunk at shots 11, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 2 / fewest shots

Layout 66, policy seed 0: 5 shots, 0 misses.
Ships: B1, B2, B3; A4, B4

B2 hit → B3 hit → B4 hit → B1 sunk → A4 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN board / training seed 2 / most shots

Layout 17, policy seed 0: 12 shots, 7 misses.
Ships: A1, B1, C1; C4, D4

B2 miss → C3 miss → D2 miss → C1 hit → B1 hit → B4 miss → A3 miss → D1 miss → A1 sunk → D4 hit → D3 miss → C4 sunk

The first hit occurs at shot 4: 3 misses occur before it and 4 occur afterwards. Ships are sunk at shots 9, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 0 / fewest shots

Layout 110, policy seed 0: 5 shots, 0 misses.
Ships: A2, B2, C2; B3, B4

B3 hit → B2 hit → C2 hit → B4 sunk → A2 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 4, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 0 / most shots

Layout 13, policy seed 0: 12 shots, 7 misses.
Ships: A1, B1, C1; C3, C4

B3 miss → C2 miss → D3 miss → A4 miss → B1 hit → A1 hit → A2 miss → C4 hit → C1 sunk → D4 miss → B4 miss → C3 sunk

The first hit occurs at shot 5: 4 misses occur before it and 3 occur afterwards. Ships are sunk at shots 9, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 1 / fewest shots

Layout 8, policy seed 0: 5 shots, 0 misses.
Ships: A1, B1, C1; A3, B3

B3 hit → A3 sunk → C1 hit → B1 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 1 / most shots

Layout 20, policy seed 0: 12 shots, 7 misses.
Ships: A1, A2, A3; C1, D1

B3 miss → C2 miss → A2 hit → A3 hit → A4 miss → A1 sunk → D3 miss → C4 miss → B1 miss → D1 hit → D2 miss → C1 sunk

The first hit occurs at shot 3: 2 misses occur before it and 5 occur afterwards. Ships are sunk at shots 6, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 2 / fewest shots

Layout 96, policy seed 0: 5 shots, 0 misses.
Ships: D1, D2, D3; B3, C3

C3 hit → D3 hit → B3 sunk → D2 hit → D1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 3, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN density / training seed 2 / most shots

Layout 50, policy seed 0: 12 shots, 7 misses.
Ships: B1, C1, D1; D3, D4

C3 miss → B4 miss → A3 miss → B2 miss → C1 hit → B1 hit → A1 miss → D1 sunk → D2 miss → D4 hit → C4 miss → D3 sunk

The first hit occurs at shot 5: 4 misses occur before it and 3 occur afterwards. Ships are sunk at shots 8, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 0 / fewest shots

Layout 23, policy seed 0: 5 shots, 0 misses.
Ships: A1, A2, A3; B2, C2

B2 hit → C2 sunk → A3 hit → A2 hit → A1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 0 / most shots

Layout 239, policy seed 0: 13 shots, 8 misses.
Ships: A4, B4, C4; C2, D2

B2 miss → C1 miss → D3 miss → A3 miss → C4 hit → C3 miss → D4 miss → B4 hit → A4 sunk → A1 miss → D2 hit → D1 miss → C2 sunk

The first hit occurs at shot 5: 4 misses occur before it and 4 occur afterwards. Ships are sunk at shots 9, 13. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 1 / fewest shots

Layout 59, policy seed 0: 5 shots, 0 misses.
Ships: B1, B2, B3; C2, D2

C2 hit → D2 sunk → B3 hit → B2 hit → B1 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 2, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 1 / most shots

Layout 15, policy seed 0: 12 shots, 7 misses.
Ships: A1, B1, C1; A4, B4

C2 miss → B3 miss → D3 miss → B1 hit → C1 hit → D1 miss → C4 miss → A1 sunk → A2 miss → A4 hit → A3 miss → B4 sunk

The first hit occurs at shot 4: 3 misses occur before it and 4 occur afterwards. Ships are sunk at shots 8, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 2 / fewest shots

Layout 111, policy seed 0: 5 shots, 0 misses.
Ships: A2, B2, C2; C3, D3

B2 hit → C2 hit → A2 sunk → D3 hit → C3 sunk

The first hit occurs at shot 1: 0 misses occur before it and 0 occur afterwards. Ships are sunk at shots 3, 5. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.

## DQN residual / training seed 2 / most shots

Layout 47, policy seed 0: 12 shots, 7 misses.
Ships: B1, C1, D1; B3, B4

B2 miss → C3 miss → A3 miss → D2 miss → B4 hit → C4 miss → A4 miss → B3 sunk → B1 hit → C1 hit → A1 miss → D1 sunk

The first hit occurs at shot 5: 4 misses occur before it and 3 occur afterwards. Ships are sunk at shots 8, 12. This separates the initial search cost from misses after the first contact; it does not prove which decision rule caused a miss.
