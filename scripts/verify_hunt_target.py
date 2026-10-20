"""Verify HuntTargetAgent using only the battleship API.

    uv run scripts/verify_hunt_target.py      # about 1 minute

Every game is played by the runner (play_solo / evaluate / play_match /
round_robin); checks only read the records the API returns.
"""

from battleship import (CLASSIC, MINI, CellState, Result, all_layouts, evaluate, neighbors,
                        play_match, play_solo, round_robin)
from battleship.agents import Agent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent
from verify_common import CONFIGS, check_public_only, check_record, states

VARIANTS = [dict(parity=p, follow_direction=d) for p in (False, True) for d in (False, True)]


class ReferenceAgent(Agent):
    """Hunt-and-target written independently: target cells come from scanning
    rows and columns for runs of hits instead of looking at pairs of hits."""

    name = "reference"

    def __init__(self, parity, follow_direction):
        super().__init__()
        self.parity, self.follow_direction = parity, follow_direction

    def select_shot(self, obs):
        n, board = obs.config.size, obs.board
        hit = [state is CellState.HIT for state in board]
        unknown = [state is CellState.UNKNOWN for state in board]
        around = {x for h in range(n * n) if hit[h] for x in neighbors(h, obs.config) if unknown[x]}
        ends = set()
        lines = [[r * n + c for c in range(n)] for r in range(n)] + \
                [[r * n + c for r in range(n)] for c in range(n)]
        for line in lines:
            k = 0
            while k < n:
                if not hit[line[k]]:
                    k += 1
                    continue
                start = k
                while k < n and hit[line[k]]:
                    k += 1
                if k - start >= 2:
                    for end in (start - 1, k):
                        if 0 <= end < n and unknown[line[end]]:
                            ends.add(line[end])
        targets = sorted(ends) if self.follow_direction and ends else sorted(around)
        if targets:
            return self.rng.choice(targets)
        legal = list(obs.legal_actions)
        if self.parity:
            m = min(obs.remaining_ship_lengths)
            grid = [x for x in legal if (x // n + x % n) % m == 0]
            legal = grid or legal
        return self.rng.choice(legal)


class BerryStackAgent(Agent):
    """The blog's basic hunt/target for comparison: a stack of neighbours of
    every hit, no use of SUNK feedback, no direction, parity 2."""

    def __init__(self, parity):
        super().__init__()
        self.parity = parity
        self.name = "berry-stack" + ("+parity" if parity else "")
        self.stack = []

    def new_game(self, config, rng):
        super().new_game(config, rng)
        self.stack = []

    def select_shot(self, obs):
        legal = set(obs.legal_actions)
        while self.stack:
            cell = self.stack.pop()
            if cell in legal:
                return cell
        n = obs.config.size
        grid = [x for x in obs.legal_actions if not self.parity or (x // n + x % n) % 2 == 0]
        return self.rng.choice(grid or obs.legal_actions)

    def observe(self, obs, action, result):
        if result.shot.result is not Result.MISS:
            legal = set(result.observation.legal_actions)
            self.stack.extend(x for x in neighbors(action, obs.config) if x in legal)


def check_1_reference():
    games = 0
    for config in CONFIGS:
        for variant in VARIANTS:
            for seed in range(60):
                ours = play_solo(HuntTargetAgent(**variant), config, seed=seed)
                ref = play_solo(ReferenceAgent(**variant), config, seed=seed)
                assert ours.shots == ref.shots, (config, variant, seed)
                games += 1
    print(f"[1] same games as an independent implementation: {games} play_solo games "
          f"(6 setups x 4 variants), every shot identical: OK")


def check_2_public_only():
    pairs = 0
    for variant in VARIANTS:
        records = [play_solo(HuntTargetAgent(**variant), MINI, layout=layout, agent_seed=0)
                   for layout in all_layouts(MINI)]
        pairs += check_public_only(records)
    print(f"[2] decisions depend only on public results: {pairs} pairs of games "
          f"(all 264 4x4 layouts, 4 variants): OK")


def check_3_rules():
    for config in CONFIGS:
        report = evaluate(HuntTargetAgent(), config, seeds=range(300))
        again = evaluate(HuntTargetAgent(), config, seeds=range(300))
        for record in report.records:
            assert check_record(record) and record.shots[-1].result is Result.SUNK
        assert [r.shots for r in report.records] == [r.shots for r in again.records]
        print(f"[3] evaluate {config.size}x{config.size} fleet={config.fleet} "
              f"reveal={config.reveal_sunk_cells}: 300 games legal, fleet sunk, "
              f"reproducible; mean {report.mean:.2f}")
    for seed in range(200):
        match = play_match(HuntTargetAgent(), ProbabilityDensityAgent(), CLASSIC,
                           first=seed % 2, seed=seed)
        winner, loser = match.records[match.winner], match.records[1 - match.winner]
        assert check_record(winner) and not check_record(loser)
        assert winner.num_shots - loser.num_shots == (1 if match.winner == match.first else 0)
    print("[3] play_match: 200 two-player games legal, winner sank the fleet first: OK")


def run_ends(open_hits, fired, config):
    """Unfired cells just past either end of a line of 2+ open hits."""
    ends = set()
    for h in open_hits:
        r, c = divmod(h, config.size)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            behind = (r - dr, c - dc)
            ahead = (r + dr, c + dc)
            if all(0 <= x < config.size for x in behind + ahead):
                b, a = behind[0] * config.size + behind[1], ahead[0] * config.size + ahead[1]
                if b in open_hits and a not in fired:
                    ends.add(a)
    return ends


def check_4_behaviour():
    """Hunt, target and direction rules, checked shot by shot from the records."""
    for variant in VARIANTS:
        counts = dict(hunt=0, target=0, direction=0)
        for config in (MINI, CLASSIC):
            report = evaluate(HuntTargetAgent(**variant), config, seeds=range(500))
            for record in report.records:
                for shot, fired, open_hits, remaining in states(record):
                    around = {x for h in open_hits for x in neighbors(h, config)
                              if x not in fired}
                    ends = run_ends(open_hits, fired, config)
                    if variant["follow_direction"] and ends:
                        assert shot.cell in ends, "did not follow the ship's direction"
                        counts["direction"] += 1
                    elif around:
                        assert shot.cell in around, "did not target a neighbour of a hit"
                        counts["target"] += 1
                    else:
                        if variant["parity"]:
                            m = min(remaining)
                            grid = [x for x in range(config.num_cells) if x not in fired
                                    and sum(divmod(x, config.size)) % m == 0]
                            assert not grid or shot.cell in grid, "hunt shot off the parity grid"
                        counts["hunt"] += 1
        print(f"[4] {HuntTargetAgent(**variant).name:<28} rules hold on every shot of "
              f"1000 games: hunt {counts['hunt']}, target {counts['target']}, "
              f"direction {counts['direction']}: OK")


def check_5_stats():
    print("[5] evaluate 10x10, 10000 games, same layouts for every agent:")
    agents = [RandomAgent(), BerryStackAgent(False), BerryStackAgent(True)] + \
             [HuntTargetAgent(**v) for v in VARIANTS] + [ProbabilityDensityAgent()]
    for agent in agents:
        r = evaluate(agent, CLASSIC, seeds=range(10_000))
        ci = 1.96 * r.std / r.games ** 0.5
        print(f"    {agent.name:<30} mean {r.mean:6.2f} ± {ci:.2f}  median {r.median:4}  "
              f"worst {r.worst:3}  {r.mean_move_time * 1000:.3f} ms/move")
    print("    blog (100M games): random median 96-97, hunt/target with parity median 64")
    print(round_robin([BerryStackAgent(True), HuntTargetAgent(), ProbabilityDensityAgent()],
                      CLASSIC, games_per_pair=500))


if __name__ == "__main__":
    check_1_reference()
    check_2_public_only()
    check_3_rules()
    check_4_behaviour()
    check_5_stats()
