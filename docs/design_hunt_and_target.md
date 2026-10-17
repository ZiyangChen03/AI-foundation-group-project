# Hunt-and-Target Agent — Design

Status: **implemented** in `battleship/agents/hunt_target.py` · Part of issue #2

`HuntTargetAgent` is the standard baseline for our project: the strategy most
Battleship AI write-ups start from. Issue #2 describes it as *"search for a hit, then
prioritize neighboring cells and follow the inferred ship direction."* It sits between
the random baseline (the floor) and the probability density baseline (the strong
reference, see `docs/design_probability_density.md`).

## 1. Idea

A ship occupies a straight line of adjacent cells. So a hit tells us a lot about its
neighbours, and two hits in a row tell us the ship's direction. The agent has two
modes:

- **Hunt:** no unfinished ship is known, so search the board for one.
- **Target:** a ship has been hit but not sunk, so finish it before searching again.

## 2. Algorithm

Every turn, given the public `Observation`. *Open hits* are `HIT` cells: hits on ships
that are not sunk yet. When a ship sinks, its cells become `SUNK` and stop being open.

1. **Direction** (`follow_direction=True`): find every untried cell just past either end
   of a horizontal or vertical line of **two or more** open hits. If there are any, fire
   at one of them at random. This continues along the ship's inferred direction.
2. **Target:** otherwise, if any open hit has untried neighbours (up, down, left,
   right), fire at one of them at random.
3. **Hunt:** otherwise there is no open hit, so fire at a random untried cell. With
   `parity=True`, only cells on the parity grid are used (see below), unless none are
   left.

Because `SUNK` feedback closes a ship, the agent goes back to hunting as soon as the
ship it was chasing sinks, without walking around its edges.

**Falling back when a direction is blocked.** If both ends of a line of hits are misses
or the board edge, but the ship has not sunk, the hits must belong to two or more ships
lying side by side. Step 1 then finds nothing, and step 2 tries all neighbours, which
finds the other ships.

### Parity

Let *m* be the length of the shortest ship still afloat
(`min(obs.remaining_ship_lengths)`). Every horizontal or vertical ship of length ≥ *m*
covers at least one cell with `(row + col) % m == 0`, because consecutive cells along a
line increase `row + col` by exactly 1. So hunting only on those cells can never miss a
ship, and it skips `1 - 1/m` of the board.

With *m* = 2 this is the checkerboard parity from Berry's blog. Our agent updates *m*
as ships sink: once the 2-ship is gone, *m* = 3 and the grid gets sparser.

The agent keeps no state between turns: each decision is a function of the
observation (and the runner-seeded `rng` for random choices).

### Example

Hits at E5 and F5 (two hits in a row, ship not sunk). Step 1 gives the cells past the
ends of the line, D5 and G5, and the agent fires at one of them. If D5 is a miss, the
only end left is G5. If G5 is also a miss and the ship is still afloat, the two hits
belong to two vertical ships, and step 2 tries E4, E6, F4 and F6.

## 3. Comparison with Berry's version

Berry's blog describes a simpler hunt/target: after every hit, push its four
neighbours onto a stack, and keep firing from the stack until it is empty. It ignores
`SUNK` feedback and has no notion of direction, so it walks around every edge of every
ship.

| | Berry's stack version | Our `HuntTargetAgent` |
|---|---|---|
| Uses `SUNK` feedback | no | yes: stops targeting a ship when it sinks |
| Follows direction | no | yes (`follow_direction`) |
| Parity spacing | 2 | shortest remaining ship |
| State | stack of cells | none (decides from the observation) |

`scripts/verify_hunt_target.py` includes Berry's version as `BerryStackAgent` for
comparison (§6).

## 4. API

```python
from battleship.agents import HuntTargetAgent

agent = HuntTargetAgent(parity=True, follow_direction=True)
agent.select_shot(obs)     # Agent interface
```

| Parameter | Default | Meaning |
|---|---|---|
| `parity` | `True` | Hunt only on the parity grid spaced by the shortest remaining ship |
| `follow_direction` | `True` | In target mode, prefer cells that extend a line of two or more hits |

Names: `hunt-target+parity` (default), `hunt-target` (`parity=False`), and the suffix
`-nodirection` when `follow_direction=False`.

It implements the `Agent` interface from `docs/API_DESIGN.md` and uses only the
public `Observation`: `board`, `remaining_ship_lengths` and `legal_actions`. It does
not learn, so `observe`, `end_game` and `training` are unused.

## 5. Cost

Each turn scans the board once for open hits and their neighbours, and once for the
parity grid: O(cells). About **0.01 ms per move** on 10×10.

## 6. Results

10×10, 10,000 games, same layouts for every agent (`evaluate`, seeds 0–9999):

| Agent | Mean ± 95% CI | Median | Worst |
|---|---|---|---|
| random | 95.42 ± 0.10 | 97 | 100 |
| Berry stack | 66.07 ± 0.25 | 66 | 100 |
| Berry stack + parity | 61.16 ± 0.17 | 62 | 80 |
| hunt-target, no parity, no direction | 62.00 ± 0.27 | 61 | 100 |
| hunt-target, no parity | 59.90 ± 0.29 | 59 | 100 |
| hunt-target + parity, no direction | 54.63 ± 0.17 | 55 | 80 |
| **hunt-target + parity (default)** | **50.83 ± 0.17** | **51** | **78** |
| probability density | 44.61 ± 0.18 | 44 | 73 |

- **Our environment reproduces the blog:** Berry's stack version with parity gets a
  median of 62 here, against 64 in the blog (random: 97 here, 96–97 in the blog).
- **Each part helps:** parity saves about 7–9 shots, following the direction about
  2–4, and using `SUNK` feedback (Berry's version vs ours without direction, with
  parity) about 6.
- On 4×4 (all 264 layouts) the default agent averages 9.7 shots.

Round robin on 10×10 (500 games per pair, alternating who shoots first):
hunt-target + parity beats Berry's stack + parity in **83.6%** of games and loses to
probability density in **70%**.

## 7. Verification

`scripts/verify_hunt_target.py` (about 1.5 minutes) checks the agent using **only our
API**: every game is played by `play_solo`, `evaluate`, `play_match` or `round_robin`,
and the checks only read the records they return.

| # | Check | Result |
|---|---|---|
| 1 | Plays the same games as an independent `ReferenceAgent` (which finds direction targets by scanning rows and columns for runs of hits, instead of looking at pairs of hits), on 6 board setups, all 4 variants | 1,440 games, every shot identical |
| 2 | Uses only public information: with the same agent seed, games on all 264 4×4 layouts make identical shots until the first shot whose public feedback (result, or which cells a sunk ship revealed) differs | All 138,864 pairs agree, 4 variants |
| 3 | Plays by the rules: each `GameRecord` checked against its layout (legal cells, no repeats, correct miss/hit/sunk, game ends when the fleet is sunk), and reruns are identical | 300 games on each of 6 setups, plus 200 `play_match` games |
| 4 | Follows its own rules on every shot, rebuilt from the records: direction cell when a line of hits can be extended, else a neighbour of an open hit, else (with parity) a parity-grid cell | About 134,000 shots over 4,000 games, no violation |
| 5 | Large-sample statistics, Berry's version for comparison, and a round robin | See §6 |

Checks 1, 2 and 4 also run as unit tests in `tests/test_hunt_target_agent.py`, along
with a check that each part (parity, direction) improves the mean.

## 8. Limitations

- **No probabilities.** Within a mode, every candidate cell is equally likely to be
  chosen. Probability density instead weighs how many ship positions cover each cell,
  which is why it needs about 6 fewer shots.
- **Ship lengths are only used for parity.** The agent does not check whether a ship
  of a remaining length could still fit in a direction, so it can fire into a gap too
  small for any remaining ship.
- **Side-by-side ships** can mislead the direction step: two parallel ships hit once
  each look like one ship lying across them. The fallback (§2) recovers, at a cost of
  a few shots.
- **`reveal_sunk_cells=False`.** The agent cannot tell which hits belong to sunk ships
  and keeps targeting around them. It still finishes every game, but takes about 61
  shots on 10×10 instead of 51.

## 9. Reproduce

```bash
uv run scripts/evaluate_all.py --board classic --games 2000   # baseline table
uv run scripts/verify_hunt_target.py                          # §6 and §7
uv run pytest tests/test_hunt_target_agent.py
```
