# Belief-Based POMDP Agent — Design

Status: **implemented** in `battleship/belief.py`, `battleship/particles.py`,
`battleship/planning.py`, `battleship/rollout.py` and
`battleship/agents/belief.py` · Issue #4

`BeliefAgent` treats Battleship as a partially observable Markov decision process
(POMDP), keeps a belief over where the hidden ships are, and chooses shots from
that belief. On 4×4 it can also plan exactly, which gives the true optimum to
compare every agent against.

## 1. The model

| POMDP piece (lecture notation) | Battleship |
|---|---|
| hidden state *s* | the layout of the fleet |
| action *a* | a shot at an untried cell |
| observation *o* | MISS / HIT / SUNK + length (+ the sunk ship's cells if `reveal_sunk_cells`) |
| transition T(s, a, s′) | identity: ships never move |
| observation model Ω(o \| s′, a) | 0 or 1: the layout fixes the feedback |
| reward | −1 per shot |
| prior b₀ | uniform over legal layouts (what `random_layout` samples) |

The lecture's belief update

    b′(s′) ∝ Ω(o | s′, a) · Σ_s T(s, a, s′) · b(s)

therefore becomes: **keep the layouts that would have produced the observed
outcome, and renormalize.** With a uniform prior the survivors stay equally likely,
so the exact belief is simply *the set of layouts consistent with the history*.

## 2. Exact belief (small boards) — `battleship/belief.py`

`Belief` lists every layout (264 on 4×4) and filters after each shot.
`occupancy()` gives P(ship at cell | history); `outcomes(cell)` splits the belief by
outcome, which is P(o | b, a), the transition model of the belief-state MDP.

## 3. Particle belief (10×10) — `battleship/particles.py`

There are far too many 10×10 layouts to list, so `ParticleBelief` keeps a few
hundred sampled ones. Three facts make this exact in what it allows and unbiased in
what it estimates.

**Consistency splits into single-ship checks.** A shot reports SUNK exactly when it
hits the last unshot cell of its ship. So a placement of a ship of length *k* can be
checked on its own cells: it covers no miss; if all its cells are shot, the last of
those shots said SUNK with length *k* (and revealed exactly these cells, when
revealed) and no earlier one did; otherwise none of its shot cells said SUNK. **A
layout explains the history exactly when every ship passes, no two overlap, and
together they cover every hit.** (`Constraints`, kept up to date one shot at a
time.)

**Keeping the sample uniform.**
- *Filtering*: drop particles that stop explaining the history. Survivors of a
  uniform sample are a uniform sample of the smaller set: exact.
- *Gibbs moves*: redraw one ship uniformly among the placements that keep the layout
  consistent given the others; and redraw *pairs* of ships jointly. Pair moves are
  necessary: e.g. hits at B2 and B3, with the 2-ship on B2 and the 3-ship on B3, can
  never reach "the 3-ship covers both" by moving one ship at a time.
- *Fresh draws*: independent uniform valid placements per ship, kept if the layout
  is consistent: exact samples that replace random particles.
- If every particle dies, a backtracking search finds consistent layouts and the
  moves above mix them.

**Rao-Blackwellized estimate.** While ship *i* is redrawn, the exact conditional
probability that it covers each cell (given the others) is added, not just where it
lands. Same expected value, much less noise.

## 4. Choosing the shot

| `planner=` | Rule | Where |
|---|---|---|
| `"greedy"` | fire at the cell with the highest P(ship) | any board |
| `"optimal"` | solve the belief-state MDP: V\*(b) = min_a [1 + Σ_o P(o\|b,a) V\*(b′)] | exact belief (4×4) |
| `"rollout"` | one-step lookahead: min_a [1 + E(shots greedy needs afterwards)] | exact on 4×4; Monte Carlo on 10×10 |

`battleship/planning.py` solves the belief-state MDP exactly on small boards, with
beliefs as bitmasks over layouts and three answer-preserving shortcuts (never fire
certain misses; fire a certain hit first; a branch-and-bound lower bound). The 4×4
optimum takes about 10 s.

## 5. Verification

Each piece is checked against something independent (`tests/test_belief.py`,
`tests/test_particles.py`, `tests/test_planning.py`, `tests/test_rollout.py`,
`tests/test_belief_agent.py`, `scripts/verify_belief.py`):

- **Consistency.** On six small configs (equal-length ships, both reveal rules) the
  single-ship checks keep exactly the layouts the exact belief keeps, after every
  shot of random games; the true layout is never dropped.
- **Uniform sampling.** In the B2/B3 trap above, single-ship moves reach 30 of the
  42 consistent layouts (χ² = 4041); with pair moves all 42 are visited uniformly
  (χ² = 51, about 41 ± 9 expected).
- **Unbiased estimates.** Against the exact P(ship at cell) on 5×5, the worst-cell
  error halves each time the particles quadruple (0.042 → 0.021 → 0.011 for
  100 / 400 / 1600) and the mean signed error is about 0.0001: sampling noise only.
- **Exact planning.** V\* matches a solver with no shortcuts on six 3×3 boards. With
  ties broken to the first cell, every planner's games average exactly its computed
  value on all 264 layouts.
- **The rollout simulator** plays exactly the shots the real
  `ProbabilityDensityAgent` plays (160 games, four configs, same random seeds).
- **Fairness.** Decisions depend only on public feedback (`check_public_only`), and
  every game is legal (`check_record`).

## 6. Results

### 4×4 (all 264 layouts; exact expected shots)

| | sunk cells revealed | not revealed |
|---|---|---|
| **optimal POMDP policy (V\*)** | **8.746** | **8.754** |
| rollout (exact, greedy base) | 8.765 | 8.784 |
| belief, greedy | 8.913 | 8.913 |
| probability density | 8.92 | 10.13 |
| hunt-target + parity | 9.68 | 10.83 |

**V\* assumes random ship placement.** It is the fewest shots any strategy can
average when every legal layout is equally likely, which is how our runner places
ships (`random_layout`) and what the belief's prior assumes. Against an opponent who
places ships non-uniformly (e.g. always near the edges), a strategy that knew that
habit could average fewer shots than 8.746; the belief agent would need that
placement distribution as its prior to match it. `OpponentBeliefAgent` learns that prior from past games; see
[design_opponent_belief.md](design_opponent_belief.md).

- Greedy is 1.9% above the optimum. **95% of that loss happens while a ship is hit
  but not sunk**, and rollout closes about 90% of it.
- Cutting the rollout off when targeting ends recovers nothing (8.911): targeting
  choices matter because of what they reveal about the *other* ships later.
- On every small board we could solve (4×4 and 5×5, one or two ships) greedy is
  0.5–3% above optimal and rollout recovers most of the gap. The exact solver stops
  scaling there: one 2-ship on 5×5 already needs 8.2 million belief states.

### 10×10 (the same 1000 boards for every agent; ± is a 95% interval)

| | sunk cells revealed | not revealed |
|---|---|---|
| hunt-target + parity | 50.94 | 60.05 |
| probability density | 44.74 | 53.87 |
| belief, 300 particles | 45.31 | 45.65 |
| **belief, 1000 particles (default)** | **44.82** | **44.67** |
| belief, 3000 particles | 44.84 | 44.95 |
| belief − density (1000 particles, paired) | +0.08 ± 0.73 | **−9.20 ± 0.79** |

- **Without revealed sunk cells the POMDP agent is 17% better than probability
  density** and loses nothing compared with the revealed rule: its belief works out
  which hits belong to the sunk ship from the reported lengths and timing.
- With revealed cells the two tie, as on 4×4.
- 300 particles are measurably noisier; 1000 and 3000 are equal within noise.
- Time per shot: about 60 ms with 1000 particles on one core (110 ms with 11
  parallel workers), against 0.3 ms for probability density.
- **Monte Carlo rollout on 10×10** (1000 particles, 48 simulated worlds, used while a
  ship is hit) measured −0.08 ± 0.97 shots against greedy on 500 paired boards, at
  about 3× the time per shot. Any gain is below what we can detect, so greedy is
  the default. Simulating with probability density as the base policy was worse
  without revealed cells (9.06 vs 8.93 on 4×4), because density plays badly there.

## 7. Limitations

- Particle beliefs must see every shot (`observe`), because an Observation does not
  record which cells each sinking revealed. With revealed cells and two ships of the
  same length, a single Observation can be ambiguous. Adding the revealed cells to
  `Shot` would remove this limitation.
- Exact planning only scales to 4×4 with two ships.
- The belief, and so every value above, assumes uniformly random ship placement
  (the prior b₀). The optimum is "optimal against random placement", not against
  any opponent. `OpponentBeliefAgent` learns a non-uniform prior on 4×4
  ([design_opponent_belief.md](design_opponent_belief.md)).
- The runner times `select_shot` only; `BeliefAgent` therefore applies its belief
  update at the start of `select_shot`, so the reported time per move includes it.

## 8. How to run

```bash
uv run pytest tests/test_belief.py tests/test_particles.py tests/test_planning.py tests/test_belief_agent.py
uv run scripts/verify_belief.py               # about 5 minutes, 100 games on 10x10
uv run scripts/verify_belief.py --games 1000  # the comparison in section 6 (slow)
```

Watch an agent play, with its belief P(ship) drawn on every square
(Space = one shot, A = auto-play, H = show the hidden ships, N = new board):

```bash
uv run python -m battleship.watch                          # POMDP agent, 4x4
uv run python -m battleship.watch --agent optimal          # exact optimal policy
uv run python -m battleship.watch --agent density --no-reveal
```

```python
from battleship import CLASSIC, MINI, all_layouts, evaluate
from battleship.agents import BeliefAgent

evaluate(BeliefAgent(planner="optimal"), MINI, layouts=all_layouts(MINI))   # exact optimum on 4x4
evaluate(BeliefAgent(), CLASSIC, seeds=range(1000))                         # particles on 10x10
```
