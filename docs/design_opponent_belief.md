# Opponent Belief — Design

Status: **implemented** in `battleship/opponents.py`, `battleship/agents/opponent.py`,
and prior weights in `battleship/belief.py`, `battleship/planning.py` and
`battleship/particles.py` · Exact on 4×4, particles on 10×10 · Extends
[design_belief_pomdp.md](design_belief_pomdp.md)

`BeliefAgent` assumes ships are placed uniformly at random, the prior b₀ of its
POMDP. `OpponentBeliefAgent` instead **learns b₀ from the opponent's past boards**,
so the belief now also covers *how this opponent places ships*. This is the
proposal's stretch goal: adapt to the opponent's placement habits over repeated
games.

**Takeaway.**
- **Effective on 4×4.** Against opponents with habits, the learned prior saves up to
  2.2 shots, 90–100% of what a perfect prior could save, and costs nothing against a
  random opponent.
- **Not effective on 10×10 as implemented.** After 1000 boards the learner is within
  noise of the uniform belief. There are two reasons:
  - The gain available is small for these mild habits: even an agent told the true
    habit saves only 1–2 of about 45 shots.
  - Counting each of the 600 placements separately learns too slowly.
- The weighted belief itself is correct on both boards. A more data-efficient learner
  (§5) is untested.

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

`scripts/verify_opponent_belief.py` (about 1–2 minutes, plus about 15 minutes for
the 10×10 games with `--classic 200`) follows the project's
convention. Games are played through `play_solo`, and the checks read the records
it returns and compare them with independent references.

| # | Check | Result |
|---|---|---|
| 1 | With random prior weights, the belief after every shot equals Bayes' rule by brute force: replay the history on every layout (`explains`), keep the consistent ones, weight by the prior | 411 positions, 3 configs (both reveal rules, equal-length ships), error < 1e-12 |
| 2 | With nothing learned, every learner plays **exactly** `BeliefAgent`'s shots | 1,584 games (3 learners × 264 layouts × both reveal rules) |
| 3 | After learning 50 games: every game is legal (`check_record`), reruns are identical, decisions depend only on public results (`check_public_only`), and evaluation does not change the counts | 264 games per learner; 34,716 pairs agree |
| 4 | Each habit's sampler draws layouts with the probabilities its distribution states | χ² ≈ degrees of freedom for all 6 habits (20,000 draws each) |
| 5 | No agent beats the weighted V\* against its habit | holds for every opponent |
| 6 | Weighted particles (10×10) against the exact weighted belief, on 5×5 with a strong, uneven prior: the error shrinks like sampling noise and there is no bias | worst-cell error 0.042 → 0.026 → 0.012 for 100 / 400 / 1600 particles, bias ≈ 0 |
| 7 | 10×10: every game legal (`check_record`); every agent plays the same layouts | 4,800 games (§4.3) |

Unit tests (`tests/test_opponents.py`, `tests/test_opponent_agent.py`, and new cases
in `tests/test_belief.py` and `tests/test_planning.py`) also cover:
- the weighted V\* against the no-shortcuts reference solver (random weights,
  including zero-weight layouts, on six 3×3 boards);
- that equal weights plan exactly like the uniform prior;
- that the exact planner refuses a belief with a different prior.

## 4. Results

### 4.1–4.2 4×4 (exact expected shots)

Every number is an exact expectation over all layouts, weighted by the opponent's
habit, with ties averaged over 3 agent seeds. Learning curves average 20 independent
runs of games drawn from the habit.

#### Fixed agents against each opponent

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

#### Learning curves (default `placements`, `smoothing=3`, and `layouts`, `smoothing=1`)

| Opponent | model | 0 games | 10 | 30 | 100 | 300 | 1000 | saving at 1000 (share of possible) |
|---|---|---|---|---|---|---|---|---|
| uniform | placements | 8.905 | 8.922 | 8.939 | 8.948 | 8.942 | 8.922 | −0.017 |
| uniform | layouts | 8.905 | 8.921 | 8.940 | 8.961 | 8.967 | 8.950 | −0.045 |
| edges | placements | 9.532 | 9.107 | 8.701 | 8.487 | 8.331 | 8.281 | +1.25 (95%) |
| edges | layouts | 9.532 | 9.554 | 9.428 | 8.759 | 8.530 | 8.343 | +1.19 (90%) |
| center | placements | 8.093 | 8.096 | 8.074 | 7.993 | 7.812 | 7.733 | +0.36 (75%) |
| center | layouts | 8.093 | 8.107 | 8.089 | 7.900 | 7.824 | 7.788 | +0.31 (64%) |
| horizontal | placements | 8.874 | 8.693 | 8.623 | 8.459 | 8.417 | 8.416 | +0.46 (81%) |
| horizontal | layouts | 8.874 | 8.825 | 8.795 | 8.715 | 8.582 | 8.435 | +0.44 (78%) |
| touching | placements | 8.617 | 8.651 | 8.630 | 8.613 | 8.575 | 8.573 | +0.04 (7%) |
| touching | layouts | 8.617 | 8.582 | 8.554 | 8.443 | 8.277 | 8.166 | +0.45 (70%) |
| 3 favorites | placements | 7.889 | 6.328 | 6.311 | 6.294 | 6.333 | 6.378 | +1.51 (68%) |
| 3 favorites | layouts | 7.889 | 5.789 | 5.667 | 5.667 | 5.667 | 5.667 | +2.22 (100%) |

"Share of possible" = (uniform-prior shots − learned shots) / (uniform-prior shots −
V\*). With `smoothing=1` the placement model learns edges faster (8.86 at 10 games
instead of 9.11), but against a uniform opponent it costs up to 0.10 shots instead of
0.05. The full table is printed by the script.

Findings:

- **Single-ship habits are learned well by the placement model.** It reaches
  95% of the possible saving against edges and 81% against horizontal, and it is
  useful within 10–30 games.
- **Each model has a blind spot, and they are opposite.** Placement counts can't
  express "ships touch", which is a property of a *pair* of ships, so they gain only
  0.04 shots there. Placement counts also mix the three favourite boards into
  unseen combinations (one board's 3-ship with another board's 2-ship), and plateau
  at 6.3. The layout model fixes both (70% and 100%), but it generalizes slowly
  because it must see each board.
- **Little cost against a random opponent.** Learning from chance patterns costs at
  most 0.05 shots with the default smoothing, which is the price of being able to adapt.
- **Learned greedy can beat oracle greedy** (edges 8.281 vs 8.297, center 7.733 vs
  7.811). This isn't an error. Greedy is myopic, so the *true* prior doesn't give the
  best *greedy* policy. The weighted V\* stays below every agent (check 5).

### 4.3 10×10 (sampled, paired)

**How it works on 10×10.** The board is too large for the exact belief, so the agent
uses the particle belief of the POMDP doc with the placement model's weights. The
learned prior splits per ship, w(s) = ∏ f(placement) with f = count + smoothing, so
the particle algorithm stays the same and stays exact. Every uniform draw becomes a
draw in proportion to f:
- the starting particles;
- the Gibbs single-ship and pair moves;
- the fresh draws;
- the Rao-Blackwell estimate.

Filtering and copying don't change. The `layouts` model doesn't scale: almost every
10×10 board is unseen, and its weights don't split per ship. The habits used here
split per ship too, so they sample exactly (each ship drawn in proportion to its
factor, redrawn until none overlap), and the oracle can use them as particle weights.

**Opponents** (`large_board_habits()`):
- uniform;
- edges, strength 0.5: on average 9.7 of the 17 ship cells on the border, against 5.0
  for random placement;
- center, strength −0.5;
- horizontal, strength 1.0: 3.9 of 5 ships horizontal, against 2.5.

200 layouts per opponent, the same for every agent, with 1000 particles. Learned
agents learn from 10 / 100 / 1000 separate boards of the habit, using `learn`, which
is what `end_game` does in training.

| Opponent | density | belief (uniform) | learned 10 | learned 100 | learned 1000 | oracle | learned 1000 − uniform | oracle − uniform |
|---|---|---|---|---|---|---|---|---|
| uniform | 44.73 | 45.20 | 44.97 | 45.84 | 45.80 | 44.99 | +0.60 ± 1.76 | −0.20 ± 1.73 |
| edges | 46.44 | 45.17 | 46.79 | 44.99 | 45.05 | 43.77 | −0.12 ± 1.76 | −1.40 ± 1.80 |
| center | 42.16 | 42.45 | 41.63 | 41.66 | 41.10 | 40.30 | −1.34 ± 1.63 | **−2.15 ± 1.61** |
| horizontal | 46.13 | 44.50 | 44.28 | 44.78 | 44.76 | 43.45 | +0.26 ± 1.71 | −1.05 ± 1.58 |

± is a 95% interval on the paired difference.

Findings, stated plainly:

- **The machinery works on 10×10.** Weighted particles track the exact weighted
  belief (check 6), and every game is legal.
- **On 10×10 these habits are worth much less than on 4×4.** Even the oracle, which
  knows the habit exactly, saves only 1–2 of about 45 shots (2–5%), against up to 13%
  on 4×4. With 200 layouts only the center saving is statistically significant. The
  ship cells cover just 17% of a 10×10 board, so most shots are spent hunting, and a
  mild placement bias narrows that hunt only a little.
- **The learned prior captures little of that saving.** After 1000 boards it is within
  noise of the uniform belief, except a −1.3 trend against center. The placement
  model counts each of the 600 placements separately. After 1000 boards a placement
  has been seen only about 8–12 times, and smoothing 3 pulls the weights toward
  uniform: a 5-ship along the edge gets 5.6× the weight of an interior one, against
  12.2× under the true habit. On 4×4 there are only 40
  placements, so the counts are dense.
- **No measurable cost against a random opponent** (+0.60 ± 1.76).

## 5. Limitations and next steps

- **Learning is too slow on 10×10.** A *feature* model would fix the sparse counts:
  learn the few numbers θ of w(s) = exp(θ · f(s)) (border cells, direction, …) by
  maximum likelihood, so every board teaches every placement. It still splits per ship
  for per-ship features, so the weighted particles work unchanged. Tighter 10×10
  intervals also need more layouts: 1000 per opponent would give about ±0.8 shots,
  at about 4× the 14-minute run.
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
uv run scripts/verify_opponent_belief.py --classic 200 # plus 10x10, §4.3 (~15 min)
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
