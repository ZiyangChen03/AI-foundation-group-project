# Opponent Belief — Design

Status: **implemented on 4×4** in `battleship/opponents.py`,
`battleship/agents/opponent.py`, and prior weights in `battleship/belief.py` and
`battleship/planning.py` · Extends [design_belief_pomdp.md](design_belief_pomdp.md)

`BeliefAgent` assumes ships are placed uniformly at random, the prior b₀ of its
POMDP. `OpponentBeliefAgent` instead **learns b₀ from the opponent's past boards**,
so the belief now also covers *how this opponent places ships*. This is the
proposal's stretch goal: adapt to the opponent's placement habits over repeated
games.

## 1. The idea

| | `BeliefAgent` | `OpponentBeliefAgent` |
|---|---|---|
| Prior b₀(s) | uniform: 1 / (number of layouts) | learned: w(s) / Σ w, from past games |
| Bayes update | keep the consistent layouts | the same |
| After the update | survivors equally likely | survivors keep their prior weights |
| Learns | nothing | after each game, from the revealed layout |

Within a game the update is the one in the POMDP doc. The likelihood is still 0 or 1,
so the agent keeps the layouts that explain the shots. The only change is the
normalization step:

    b′(s) = w(s) / Σ_{s′ consistent} w(s′)      for every consistent layout s

`Belief` takes optional prior `weights` (one per layout). `occupancy()` becomes a
weighted share of the consistent layouts, and `outcomes()` gives P(o | b, a) as a
weight ratio. With equal weights the results are identical to the uniform belief,
float for float.

Across games, `end_game` reads the layout the runner reveals and updates the counts.
Two models of the habit:

| `model=` | Prior weight of a layout s | Generalizes? | Can express |
|---|---|---|---|
| `"placements"` (default) | ∏ over ships of (times that placement was seen + smoothing) | yes, from few games to unseen boards | habits of single ships: edges, centre, direction |
| `"layouts"` | times this exact board was seen + smoothing | no, each board must be seen | anything, given enough games: reused boards, ships touching |

`smoothing` is a pseudo-count (Laplace smoothing). Unseen placements keep some
weight, and larger values trust the counts less. The default `smoothing=3` costs at
most 0.05 shots against a uniform opponent (§5). With nothing learned, every weight
is equal and the agent plays **exactly** `BeliefAgent`'s shots.

Learning follows the `Agent` convention: counts change only while `training` is on,
so `evaluate` always measures a frozen prior. `habit=` gives the agent the true
habit as its prior instead. This "oracle" is the best prior it could learn.

## 2. Opponents with habits — `battleship/opponents.py`

A `Habit` gives every layout a weight, and the opponent places layout s with
probability w(s) / Σ w. Feature habits use w(s) = exp(strength × feature(s)):

| Opponent | Weight | What it does on 4×4 |
|---|---|---|
| uniform | 1 | the runner's placement (control) |
| edges | exp(1.0 × ship cells on the border) | hugs the edges |
| center | exp(−1.0 × ship cells on the border) | avoids the edges |
| horizontal | exp(1.5 × horizontal ships) | lays ships along rows |
| touching | exp(2.0 × pairs of ships sharing an edge) | puts ships side by side |
| 3 favorites | 1 for three fixed boards, else 0 | reuses the same few boards |

**The runner needs no change.** `habit.sample(config, rng)` draws a layout by
rejection from `random_layout`, and the layout goes to `play_solo(layout=...)`.
Sampling also works on 10×10. On small boards, `expected_shots(agent, config,
habit)` gives an agent's **exact** mean against a habit: it plays every layout the
habit can produce and weights each by its probability.

`Planner(config, weights)` solves the belief-state MDP with a weighted prior, so V\*
against each habit is the true optimum to compare with. Probabilities become weight
sums, computed eight layouts at a time from lookup tables. The same three shortcuts
apply; the certain-hit test becomes "every consistent layout covers the cell". With
equal weights it reproduces V\* = 8.7462 and greedy = 8.9132.

## 3. Verification

`scripts/verify_opponent_belief.py` (about 1–2 minutes) follows the project's
convention. Games are played through `play_solo`, and the checks read the records
it returns and compare them with independent references.

| # | Check | Result |
|---|---|---|
| 1 | With random prior weights, the belief after every shot equals Bayes' rule by brute force: replay the history on every layout (`explains`), keep the consistent ones, weight by the prior | 411 positions, 3 configs (both reveal rules, equal-length ships), error < 1e-12 |
| 2 | With nothing learned, every learner plays **exactly** `BeliefAgent`'s shots | 1,584 games (3 learners × 264 layouts × both reveal rules) |
| 3 | After learning 50 games: every game is legal (`check_record`), reruns are identical, decisions depend only on public results (`check_public_only`), and evaluation does not change the counts | 264 games per learner; 34,716 pairs agree |
| 4 | Each habit's sampler draws layouts with the probabilities its distribution states | χ² ≈ degrees of freedom for all 6 habits (20,000 draws each) |
| 5 | No agent beats the weighted V\* against its habit | holds for every opponent |

Unit tests (`tests/test_opponents.py`, `tests/test_opponent_agent.py`, and new cases
in `tests/test_belief.py` and `tests/test_planning.py`) also cover:
- the weighted V\* against the no-shortcuts reference solver (random weights,
  including zero-weight layouts, on six 3×3 boards);
- that equal weights plan exactly like the uniform prior;
- that the exact planner refuses a belief with a different prior.

## 4. Results (4×4, exact expected shots)

Every number is an exact expectation over all layouts, weighted by the opponent's
habit, with ties averaged over 3 agent seeds. Learning curves average 20 independent
runs of games drawn from the habit.

### Fixed agents against each opponent

| Opponent | density | belief (uniform prior) | uniform-optimal POMDP | oracle (greedy, true prior) | **optimum vs habit (V\*)** |
|---|---|---|---|---|---|
| uniform | 8.904 | 8.905 | 8.746 | 8.905 | **8.746** |
| edges | 9.487 | 9.532 | 9.336 | 8.297 | **8.209** |
| center | 8.110 | 8.093 | 7.861 | 7.811 | **7.614** |
| horizontal | 8.865 | 8.874 | 8.737 | 8.378 | **8.312** |
| touching | 8.621 | 8.617 | 8.386 | 8.142 | **7.975** |
| 3 favorites | 8.333 | 7.889 | 8.222 | 5.667 | **5.667** |

- **The "optimal" POMDP policy stops being optimal once placement isn't random.**
  Against an edge-hugger the uniform-optimal policy needs 9.336 shots. Greedy with
  the right prior needs 8.297, and the true optimum is 8.209.
- **The prior matters more than the planner.** For every habit, greedy with the true
  prior beats the exactly-planned uniform policy. Uniform planning recovers at most
  0.2 shots; the right prior recovers up to 2.2.

### Learning curves (default `placements`, `smoothing=3`, and `layouts`, `smoothing=1`)

| Opponent | model | 0 games | 10 | 30 | 100 | 300 | 1000 | saving at 1000 (share of possible) |
|---|---|---|---|---|---|---|---|---|
| uniform | placements | 8.905 | 8.917 | 8.933 | 8.955 | 8.929 | 8.917 | −0.011 |
| uniform | layouts | 8.905 | 8.922 | 8.938 | 8.963 | 8.978 | 8.944 | −0.039 |
| edges | placements | 9.532 | 9.137 | 8.716 | 8.479 | 8.319 | 8.281 | +1.25 (95%) |
| edges | layouts | 9.532 | 9.550 | 9.357 | 8.733 | 8.515 | 8.344 | +1.19 (90%) |
| center | placements | 8.093 | 8.110 | 8.079 | 7.983 | 7.850 | 7.745 | +0.35 (73%) |
| center | layouts | 8.093 | 8.131 | 8.118 | 7.943 | 7.833 | 7.793 | +0.30 (63%) |
| horizontal | placements | 8.874 | 8.678 | 8.588 | 8.464 | 8.422 | 8.415 | +0.46 (82%) |
| horizontal | layouts | 8.874 | 8.842 | 8.782 | 8.694 | 8.561 | 8.437 | +0.44 (78%) |
| touching | placements | 8.617 | 8.651 | 8.630 | 8.613 | 8.575 | 8.573 | +0.04 (7%) |
| touching | layouts | 8.617 | 8.582 | 8.554 | 8.443 | 8.277 | 8.166 | +0.45 (70%) |
| 3 favorites | placements | 7.889 | 6.328 | 6.311 | 6.294 | 6.333 | 6.378 | +1.51 (68%) |
| 3 favorites | layouts | 7.889 | 5.789 | 5.667 | 5.667 | 5.667 | 5.667 | +2.22 (100%) |

"Share of possible" = (uniform-prior shots − learned shots) / (uniform-prior shots −
V\*). With `smoothing=1` the placement model learns edges faster (8.90 at 10 games
instead of 9.14), but against a uniform opponent it costs up to 0.10 shots instead of
0.05. The full table is printed by the script.

Findings:

- **Single-ship habits are learned well by the placement model.** It reaches
  95% of the possible saving against edges and 82% against horizontal, and it is
  useful within 10–30 games.
- **Each model has a blind spot, and they are opposite.** Placement counts can't
  express "ships touch", which is a property of a *pair* of ships, so they gain only
  0.04 shots there. Placement counts also mix the three favourite boards into
  unseen combinations (one board's 3-ship with another board's 2-ship), and plateau
  at 6.3. The layout model fixes both (70% and 100%), but it generalizes slowly
  because it must see each board.
- **Little cost against a random opponent.** Learning from chance patterns costs at
  most 0.05 shots with the default smoothing, which is the price of being able to adapt.
- **Learned greedy can beat oracle greedy** (edges 8.281 vs 8.297, center 7.745 vs
  7.811). This isn't an error. Greedy is myopic, so the *true* prior doesn't give the
  best *greedy* policy. The weighted V\* stays below every agent (check 5).

## 5. Limitations and next steps

- **4×4 only.** The learned prior needs the exact belief. On 10×10 the particle
  sampler would have to draw each ship in proportion to its learned weight instead of
  uniformly. The Gibbs conditionals stay simple for the placement model, because its
  weights split per ship. Habit sampling already works on 10×10.
- **Neither model wins everywhere.** A natural next step is to keep both and weight
  them by how well each predicted the opponent's past boards (Bayesian model
  averaging), or to use a log-linear model over features of ship pairs (e.g.
  touching) learned by maximum likelihood.
- **It only learns from finished games.** Each sinking reveals a ship mid-game when
  `reveal_sunk_cells` is on, which could update the counts sooner.
- **Stationary opponents only.** An opponent that changes habit would need old games
  to be forgotten (decayed counts). `forget()` resets everything.
- **Weighted optimal planning is a benchmark, not a planner the agent uses.** The
  learned prior changes after every game, and re-solving V\* takes about 20 s on 4×4.
- **Tournaments.** `play_match` still places both fleets uniformly, so a head-to-head
  against a habit needs the optional `layouts=` argument suggested in
  `docs/API_DESIGN.md` §8.

## 6. How to run

```bash
uv run pytest tests/test_opponents.py tests/test_opponent_agent.py
uv run scripts/verify_opponent_belief.py              # checks + benchmark, ~1-2 min
uv run scripts/verify_opponent_belief.py --runs 20    # the curves in section 4
```

```python
import random
from battleship import MINI, play_solo
from battleship.agents import OpponentBeliefAgent
from battleship.opponents import FeatureHabit, border_cells, expected_shots

edges = FeatureHabit("edges", border_cells, 1.0)
agent = OpponentBeliefAgent()                 # placements model, smoothing 3
agent.training = True                         # learn from each finished game
rng = random.Random(0)
for game in range(100):
    play_solo(agent, MINI, layout=edges.sample(MINI, rng), agent_seed=game)
print(expected_shots(agent, MINI, edges))     # about 8.5, down from 9.53
```
