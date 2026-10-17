# Battleship AI — API Design

Status: **implemented** in `battleship/` except the three main agents (§11) · Tracks issue #2

This document defines the interfaces shared by the environment, the agents, and the
evaluation / tournament code, based on our proposal *"Learning to Play Battleship:
Comparing Local Search, Markov Models, and Q-Learning"*. The goal is that each team
member can build one agent independently and plug it into the same runner.

## 1. Requirements from the proposal

| Proposal item | API consequence |
|---|---|
| Classic game is 10×10 with five ships; we develop on 4×4 | Board size and fleet are **configurable** (`GameConfig`) |
| Agents infer ships from hits and misses only | Agents receive an immutable `Observation`; they never touch the environment or the hidden layout |
| Local search needs arrangements consistent with past shots | Shared helpers: enumerate placements, check consistency |
| Markov model learns from simulated games | Runner produces `GameRecord`s (layout + shot sequence) as training data |
| Q-learning, reward −1 per shot | `step()` returns reward −1; agents get an `observe()` hook for updates and a training/eval switch |
| Baselines: random, hunt-and-target | Same `Agent` interface |
| Shots-to-win (mean, spread, worst case), time per move, thousands of games | `evaluate()` returns an `EvalReport`; the runner times every move |
| Round-robin tournament, full games, alternating who shoots first | Two-player `play_match()` and `round_robin()`; the runner places both fleets **randomly** (seeded) |
| Stretch: adapt to opponent's placement habits over repeated games | **Out of scope for now**: placement is always random, so there are no habits to learn (see §8) |

## 2. Package layout

```
battleship/
  __init__.py
  config.py        GameConfig, presets (MINI, CLASSIC)
  layout.py        Ship, Layout, placements, random_layout, validate_layout, layout_from_cells, all_layouts
  env.py           CellState, Result, Shot, Observation, StepResult, BattleshipEnv, is_consistent
  agents/
    base.py        Agent (abstract)
    random.py
    hunt_target.py
    probability.py
    local_search.py    (to do)
    markov.py          (to do)
    q_learning.py      (to do)
  runner.py        play_solo, train, simulate_records, evaluate, EvalReport, GameRecord
  tournament.py    play_match, round_robin
  ui.py            pygame app (current BoardApp), imports env — the only module that imports pygame
scripts/
  evaluate_all.py  run baselines + agents, print/save results
tests/
```

The old `battleship_4x4.py` was split into `env.py` (the `Battleship` class) and `ui.py`
(the `BoardApp` class); `battleship_4x4.py` is now a short launcher for the UI.
Training and evaluation never import pygame. Everything below is also importable
directly from `battleship` (e.g. `from battleship import MINI, evaluate`).

## 3. Game configuration — `config.py`

```python
@dataclass(frozen=True)
class GameConfig:
    size: int = 10
    fleet: tuple[int, ...] = (5, 4, 3, 3, 2)   # ship lengths
    reveal_sunk_cells: bool = True             # see §9, decision needed

MINI    = GameConfig(size=4,  fleet=(3, 2))
CLASSIC = GameConfig(size=10, fleet=(5, 4, 3, 3, 2))
```

Every function below takes a `GameConfig`, so the same agent code runs on both boards.

**Cells** are integers `0 .. size*size - 1`, row-major (`cell = row * size + col`).
Integers keep actions simple for Q-learning and lookup tables. Helpers:

```python
def to_rc(cell: int, config) -> tuple[int, int]
def to_cell(row: int, col: int, config) -> int
def label(cell: int, config) -> str              # 0 -> "A1", for the UI and logs
def neighbors(cell: int, config) -> list[int]    # up/down/left/right, used by hunt-and-target, Markov, Q features
```

## 4. Layouts — `layout.py`

```python
@dataclass(frozen=True)
class Ship:
    length: int
    cells: frozenset[int]

Layout = tuple[Ship, ...]          # one Ship per fleet entry

def placements(config, length) -> tuple[frozenset[int], ...]
    # every straight horizontal/vertical position of a ship of this length (cached)
def random_layout(config, rng: random.Random) -> Layout
    # uniform over all valid layouts (rejection sampling)
def validate_layout(config, layout) -> None
    # raises ValueError: wrong lengths, not straight, off board, overlapping
def layout_from_cells(config, ships: Iterable[Iterable[int]]) -> Layout
    # predefined layout, e.g. layout_from_cells(MINI, [[0, 1, 2], [8, 12]]); validated
def all_layouts(config) -> Iterator[Layout]
    # exhaustive enumeration, each distinct board once (264 on MINI); only practical for small boards
```

`is_consistent(cells, obs)` lives in `env.py` (it reads an `Observation`): a ship
placement is consistent if it does not cover a MISS or an already-sunk ship's cells.

Rules (issue #2): ships are horizontal or vertical, may touch, may not overlap.

`placements` and `is_consistent` are shared building blocks: probability density,
local search (neighbour moves = move one ship to another consistent placement), and
the Q-learning feature "how many remaining ships could fit here" all use them.

## 5. Environment — `env.py`

### Public information types

```python
class CellState(Enum):  UNKNOWN, MISS, HIT, SUNK
class Result(Enum):     MISS, HIT, SUNK

@dataclass(frozen=True)
class Shot:
    cell: int
    result: Result
    sunk_length: int | None          # set only when result is SUNK

@dataclass(frozen=True)
class Observation:                   # everything an agent is allowed to see
    config: GameConfig
    board: tuple[CellState, ...]     # length size*size
    history: tuple[Shot, ...]        # in firing order
    remaining_ship_lengths: tuple[int, ...]   # sorted descending

    @property
    def legal_actions(self) -> tuple[int, ...]   # UNKNOWN cells
    @property
    def num_shots(self) -> int
```

`Observation` is frozen, so an agent cannot change game state by accident, and
it is safe to store as training data.

### Environment

```python
class StepResult(NamedTuple):        # unpacks like Gym: obs, reward, done, shot = env.step(a)
    observation: Observation
    reward: float                    # always -1.0
    done: bool
    shot: Shot

class BattleshipEnv:
    def __init__(self, config: GameConfig = MINI): ...
    def reset(self, seed: int | None = None, layout: Layout | None = None) -> Observation
        # layout given  -> validated and used (predefined layouts)
        # otherwise     -> random_layout(config, Random(seed)) (seeded random placement, the default everywhere)
    def step(self, cell: int) -> StepResult
        # raises ValueError: game over, out of range, or cell already shot
    def observation(self) -> Observation
    @property
    def done(self) -> bool
    def reveal_layout(self) -> Layout
        # raises RuntimeError unless done; used by the runner only, never by agents mid-game
```

Agents never get a reference to `BattleshipEnv`. Only the runner holds it, which
enforces "agents can access only shot history and public feedback" (issue #2, item 6).

## 6. Agents — `agents/base.py`

One interface covers shooting and learning. Agents do **not** place ships: the runner
always places fleets randomly with `random_layout`, using a seed for reproducibility.


```python
class Agent(ABC):
    name: str = "agent"

    def __init__(self):              # subclasses with their own __init__ must call super().__init__()
        self.training = False        # learned agents explore / update only when True
        self.config = None
        self.rng = random.Random()

    def new_game(self, config: GameConfig, rng: random.Random) -> None:
        """Reset per-game state. rng is seeded by the runner for reproducibility."""

    @abstractmethod
    def select_shot(self, obs: Observation) -> int:
        """Return a cell in obs.legal_actions."""

    def observe(self, obs: Observation, action: int, result: StepResult) -> None:
        """Called after every shot. Default: no-op. Q-learning updates here."""

    def end_game(self, record: GameRecord) -> None:
        """Called once after each game with the full record, including the
        revealed layout. Default: no-op."""

    # optional, for learned agents
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Agent": ...
```

How each planned agent maps onto it:

| Agent | `select_shot` | Extra hooks |
|---|---|---|
| `RandomAgent` | uniform over `legal_actions` | — |
| `HuntTargetAgent(parity=True)` | parity cells until a hit, then neighbours of unsunk hits | — |
| `LocalSearchAgent(restarts, steps, method="hill"/"anneal")` | search for layouts consistent with `obs`, fire at the most frequently covered unknown cell | uses `placements`, `is_consistent` |
| `MarkovAgent` | fire at the cell with the highest learned P(ship \| neighbourhood state) | `fit(records: Iterable[GameRecord])`, `save`/`load` |
| `QLearningAgent(alpha, gamma, epsilon)` | ε-greedy over `w · features(obs, cell)` | updates weights in `observe` when `training`; `save`/`load` |
| `ProbabilityDensityAgent(hit_weight=50, tie_break="random")` | Berry's placement counting: every consistent position of each remaining ship adds `hit_weight ** (open hits covered)` to its untried cells; fire at the maximum | `density(obs)` returns the score of every cell |

## 7. Runner and evaluation — `runner.py`

### One-player games (shots-to-win)

```python
@dataclass(frozen=True)
class GameRecord:
    config: GameConfig
    layout: Layout                   # the hidden layout that was searched
    shots: tuple[Shot, ...]
    move_times: tuple[float, ...]    # seconds per select_shot call
    @property
    def num_shots(self) -> int

def play_solo(agent, config=MINI, seed=None, layout=None, agent_seed=None) -> GameRecord
    # env.reset -> agent.new_game -> loop(select_shot, step, observe) -> agent.end_game

def train(agent, config, episodes: int, seed: int = TRAIN_SEED) -> list[GameRecord]
    # agent.training = True while running play_solo on seeds seed..seed+episodes-1

def simulate_records(config, n: int, seed: int = TRAIN_SEED, shooter=None) -> list[GameRecord]
    # training data for MarkovAgent.fit (default shooter: RandomAgent)
```

`TRAIN_SEED` is 1,000,000,000, far away from the evaluation seeds `range(N)`, so
agents are never evaluated on the layouts they were trained on.

### Evaluation

```python
@dataclass
class EvalReport:
    agent: str
    games: int
    mean: float
    std: float
    median: float
    best: int
    worst: int
    histogram: dict[int, int]        # shots -> number of games
    mean_move_time: float            # seconds
    records: list[GameRecord]

def evaluate(agent, config=MINI, seeds=range(1000), layouts=None) -> EvalReport
    # sets agent.training = False
    # layouts given -> one game per layout (e.g. all_layouts(MINI) for an exact 4×4 result)
    # otherwise     -> one game per seed; every agent gets the SAME seeds, so the same layouts
```

Using the same seeds or layouts for every agent makes the comparison paired, so
differences in the results come from the agents and not from luck.

## 8. Two-player games — `tournament.py`

```python
@dataclass(frozen=True)
class MatchRecord:
    config: GameConfig
    players: tuple[str, str]
    first: int                       # index of the player who shot first
    winner: int
    records: tuple[GameRecord, GameRecord]   # records[i] = player i shooting at the other's fleet

def play_match(a: Agent, b: Agent, config=CLASSIC, first: int = 0, seed=None) -> MatchRecord
    # both fleets placed with random_layout (seeded) -> two envs;
    # players alternate one shot each; first to sink the whole enemy fleet wins

def round_robin(agents: list[Agent], config=CLASSIC, games_per_pair=1000, seed=0) -> TournamentReport
    # every pair plays games_per_pair matches, alternating who shoots first
    # TournamentReport.win_rate[i][j] = fraction of games agent i won against agent j
```

Note for the report: shots never change the opponent's board, so a match is decided
by each agent's shots-to-win on its own random target layout (ties go to the player
who shot first). With random placement the two agents do not affect each other, so
win rates follow directly from the shots-to-win distributions measured by `evaluate()`.
The tournament is still useful as a head-to-head summary, and it measures the
first-shooter advantage. If the team later revisits the stretch goal, placement could
come back as an optional `layouts=` argument to `play_match`, so agents still never
place ships themselves.

## 9. Decisions needed from the team

1. **Does sinking a ship reveal its cells?** (issue #2, item 4)
   Hasbro rules: no, the opponent only names the ship. The current code: yes.
   Proposal: keep it as `GameConfig.reveal_sunk_cells`, default `True` for the 4×4
   development board, and pick one value for all final results.
2. **Integer cells vs. (row, col)** — proposed: integers everywhere, helpers for conversion.
3. **Which board for the final results?** Proposed: develop and test on `MINI`
   (exhaustive evaluation is possible), report final numbers on `CLASSIC` so they
   can be compared with the 60–65 (hunt-and-target) and 40–45 (probability density)
   shots quoted in the proposal.

## 10. Example usage

```python
from battleship import MINI, CLASSIC, all_layouts, train, evaluate, round_robin
from battleship.agents import RandomAgent, HuntTargetAgent
from battleship.agents.q_learning import QLearningAgent   # not written yet

q = QLearningAgent(alpha=0.01, gamma=0.95, epsilon=0.1)
train(q, CLASSIC, episodes=50_000)
q.save("models/q_classic.json")

for agent in (RandomAgent(), HuntTargetAgent(parity=True), q):
    print(evaluate(agent, CLASSIC, seeds=range(10_000)))

print(evaluate(HuntTargetAgent(), MINI, layouts=all_layouts(MINI)))   # exact 4×4 result

print(round_robin([RandomAgent(), HuntTargetAgent(), q], CLASSIC, games_per_pair=1000))
```

## 11. Suggested build order

1. ~~`config`, `layout`, `env` (port the existing class, add `layout=` and `history`) + tests~~ done
2. ~~`Agent` base, `RandomAgent`, `HuntTargetAgent`, `runner.evaluate`: baseline numbers~~ done
3. The three main agents in parallel, one owner each: **to do**
4. ~~`tournament`~~ done

To add an agent: subclass `Agent` in `battleship/agents/`, export it from
`battleship/agents/__init__.py`, add it to `make_agents()` in
`scripts/evaluate_all.py`, and add it to the parametrized agents in
`tests/test_runner.py` so it is checked to finish games on both boards.
