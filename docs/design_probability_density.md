# Probability Density Agent — Design

Status: **implemented** in `battleship/agents/probability.py` · Part of issue #2

`ProbabilityDensityAgent` is the strong baseline for our project. It follows the
probability density strategy of N. Berry, "Battleship", DataGenetics blog, 2011
(<https://www.datagenetics.com/blog/december32011/>), which is the best-known
hand-written Battleship strategy. Our three main agents (local search, Markov model,
Q-learning) should be compared against it.

## 1. Idea

The opponent's fleet is hidden, but every hit, miss and sunk ship rules out some
positions where the remaining ships could be. If we count, for every cell, how many
possible positions of the remaining ships cover it, the cells with the highest count
are the most likely to hold a ship. The agent fires there and recounts after every
shot.

## 2. Algorithm

Every turn, given the public `Observation`:

1. For each ship length in `obs.remaining_ship_lengths` (ships not yet sunk; equal
   lengths are counted once per ship), take every horizontal and vertical position
   from `placements(config, length)`.
2. **Discard impossible positions:** a position is impossible if it covers a `MISS`
   cell or a `SUNK` cell. Misses and sunk ships are obstacles that ships must be
   placed around.
3. **Weight possible positions.** Let *k* be the number of `HIT` cells (hits on ships
   not yet sunk) the position covers. Its weight is `hit_weight ** k`.
   - **Hunt mode** (no open hits): *k* = 0 for every position, so all weights are 1
     and the score is a plain count.
   - **Target mode** (at least one open hit): positions through a hit dominate,
     so the agent finishes a ship it has found. Positions through two hits in a
     row dominate even more, so it follows the ship's direction.
4. **Add** the weight to every `UNKNOWN` cell the position covers. Cells already
   shot score 0.
5. **Fire** at the untried cell with the highest score. Ties go to a random best
   cell, or with `tie_break="first"` to the first one counting from the top left, as
   in the blog.

The agent keeps no state between turns: each decision is a function of the
observation (and the runner-seeded `rng` for ties).

### Example (4×4, empty board, ships 3 and 2)

Corner A1 can hold the 3-ship in 2 ways and the 2-ship in 2 ways, so it scores 4.
Inner cell B2 can hold each ship in 4 ways, so it scores 8. On an empty board the
agent therefore starts in the centre, as the blog's walkthrough shows.

## 3. API

```python
from battleship.agents import ProbabilityDensityAgent

agent = ProbabilityDensityAgent(hit_weight=50, tie_break="random")
agent.density(obs)      # list of scores, one per cell; 0 for cells already shot
agent.select_shot(obs)  # Agent interface: the highest-scoring untried cell
```

| Parameter | Default | Meaning |
|---|---|---|
| `hit_weight` | `50` | Weight multiplier per open hit a position covers. The blog only says "heavy"; see §5 |
| `tie_break` | `"random"` | `"random"`: random best cell using the runner's seeded `rng` (reproducible). `"first"`: lowest cell index, as in the blog |

It implements the `Agent` interface from `docs/API_DESIGN.md` and uses only the
public `Observation`: `board`, `remaining_ship_lengths` and `legal_actions`. It does
not learn, so `observe`, `end_game` and `training` are unused.

## 4. Cost

Per turn the agent looks at every position of every remaining ship: about 180
positions per ship on 10×10 (fewer as ships sink), each covering at most 5 cells.
That is a few thousand operations per move, about **0.08 ms per move** in our
runs (vs 0.003 ms for random).

## 5. Choosing `hit_weight`

Mean shots-to-win on 10×10, 400 games, same layouts:

| `hit_weight` | 1 | 5 | 20 | 50 | 200 | 1000 |
|---|---|---|---|---|---|---|
| Mean shots | 57.4 | 43.8 | 44.0 | 44.0 | 44.0 | 44.0 |

With weight 1 there is no target mode and the agent is much worse. From about 20
upward the results are identical: the weight only has to be large enough that any
position through a hit beats every position that is not. 50 is safely in that range.

## 6. Results

Same layouts for every agent (`evaluate` with the same seeds):

| Agent | 4×4 mean (all 264 layouts) | 10×10 mean (2000 games) | 10×10 median / worst |
|---|---|---|---|
| random | 14.2 | 95.3 | 97 / 100 |
| hunt-target | 10.2 | 59.9 | 59 / 100 |
| hunt-target + parity | 9.7 | 50.8 | 51 / 75 |
| **probability density** | **8.9** | **44.7** | **44 / 73** |

Over 10,000 games on 10×10: mean **44.61 ± 0.18** (95% CI), median 44, best 21,
worst 73. The blog reports median 42 and worst 73 over 100 million games.

In a round robin on 10×10 (500 games per pair, alternating who shoots first) it
wins **70%** of games against hunt-target + parity and all games against random.

## 7. Verification

`scripts/verify_probability_density.py` (about 1 minute) checks the agent using
**only our API**: every game is played by `play_solo`, `evaluate`, `play_match` or
`round_robin`, and the checks only read the records they return.

| # | Check | Result |
|---|---|---|
| 1 | Plays the same games as an independent `ReferenceAgent` written from the blog's description (sliding each ship along rows and columns), on 6 board setups, both tie-breaks | 1,200 games, every shot identical |
| 2 | Uses only public information: with the same agent seed, games on all 264 4×4 layouts make identical shots until the first public result differs | All 34,716 pairs agree |
| 3 | Plays by the rules: each `GameRecord` checked against its layout (legal cells, no repeats, correct miss/hit/sunk, game ends when the fleet is sunk), and reruns are identical | 300 games on each of 6 setups, plus 200 `play_match` games |
| 4 | Compared with the exact Bayesian posterior on 4×4, rebuilt from each record's shot history over all 264 layouts | Picks a most-likely cell 96.3% of the time; never fires at an impossible cell |
| 5 | Large-sample statistics and a round robin | See §6 |

The reference-agent comparison (check 1) and the public-information check (check 2)
also run as unit tests in `tests/test_probability_agent.py`.

## 8. Limitations

- **Ships are counted independently.** The score adds up positions of each ship
  separately and ignores that ships cannot overlap each other. So it is an
  approximation of the true probability: in check 4 it skipped a guaranteed hit in
  13 of 392 states. A method that samples whole consistent layouts (our local search
  agent) can fix this, which makes it a natural comparison.
- **Median 44 vs the blog's 42.** The worst case matches (73). Neither the blog's
  top-left tie-break (median 44) nor sequential ship placement (median 44) explains
  the gap; it most likely comes from details the blog does not publish, such as its
  exact weighting.
- **`reveal_sunk_cells=False`.** When sinking a ship does not reveal its cells, the
  agent cannot tell which hits belong to sunk ships and keeps treating them as open
  hits. It still finishes every game, but takes about 54 shots on 10×10 instead of 45.

## 9. Reproduce

```bash
uv run scripts/evaluate_all.py --board classic --games 2000   # §6 table
uv run scripts/verify_probability_density.py                  # §7 checks
uv run pytest tests/test_probability_agent.py
```
