"""Verify ProbabilityDensityAgent using only the battleship API.

    uv run scripts/verify_probability_density.py      # about 1 minute

Every game is played by the runner (play_solo / evaluate / play_match /
round_robin); checks only read the records the API returns.
"""

import statistics
from itertools import combinations

from battleship import (CLASSIC, MINI, CellState, GameConfig, Result, all_layouts,
                        evaluate, play_match, play_solo, round_robin)
from battleship.agents import Agent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent

CONFIGS = [MINI, CLASSIC, GameConfig(6, (3, 3, 2)), GameConfig(5, (1, 1, 4)),
           GameConfig(4, (3, 2), reveal_sunk_cells=False),
           GameConfig(10, (5, 4, 3, 3, 2), reveal_sunk_cells=False)]


class ReferenceAgent(Agent):
    """The blog's algorithm written independently, as an Agent: slide each
    remaining ship along every row and column."""

    name = "reference"

    def __init__(self, hit_weight=50, tie_break="random"):
        super().__init__()
        self.hit_weight, self.tie_break = hit_weight, tie_break

    def select_shot(self, obs):
        n, board = obs.config.size, obs.board
        scores = [0] * (n * n)
        for length in obs.remaining_ship_lengths:
            seen = set()
            for r in range(n):
                for c in range(n):
                    for horizontal in (True, False):
                        if (c if horizontal else r) + length > n:
                            continue
                        cells = tuple(r * n + c + i if horizontal else (r + i) * n + c
                                      for i in range(length))
                        if frozenset(cells) in seen:    # length-1 ships
                            continue
                        seen.add(frozenset(cells))
                        if any(board[x] in (CellState.MISS, CellState.SUNK) for x in cells):
                            continue
                        weight = self.hit_weight ** sum(board[x] is CellState.HIT for x in cells)
                        for x in cells:
                            if board[x] is CellState.UNKNOWN:
                                scores[x] += weight
        legal = obs.legal_actions
        best = max(scores[x] for x in legal)
        candidates = [x for x in legal if scores[x] == best]
        return candidates[0] if self.tie_break == "first" else self.rng.choice(candidates)


def check_record(record):
    """A GameRecord obeys the rules, judged only from its layout and shots."""
    config, layout = record.config, record.layout
    ship_of = {cell: i for i, ship in enumerate(layout) for cell in ship.cells}
    hits = [set() for _ in layout]
    fired = set()
    for shot in record.shots:
        assert 0 <= shot.cell < config.num_cells, "shot off the board"
        assert shot.cell not in fired, "repeated shot"
        assert not all(h == s.cells for h, s in zip(hits, layout)), "shot after game over"
        fired.add(shot.cell)
        i = ship_of.get(shot.cell)
        if i is None:
            assert (shot.result, shot.sunk_length) == (Result.MISS, None)
            continue
        hits[i].add(shot.cell)
        if hits[i] == layout[i].cells:
            assert (shot.result, shot.sunk_length) == (Result.SUNK, layout[i].length)
        else:
            assert (shot.result, shot.sunk_length) == (Result.HIT, None)
    return all(h == s.cells for h, s in zip(hits, layout))


def check_1_reference():
    games = 0
    for config in CONFIGS:
        for tie_break in ("first", "random"):
            for seed in range(100):
                ours = play_solo(ProbabilityDensityAgent(tie_break=tie_break), config, seed=seed)
                ref = play_solo(ReferenceAgent(tie_break=tie_break), config, seed=seed)
                assert ours.shots == ref.shots, (config, tie_break, seed)
                games += 1
    print(f"[1] same games as an independent implementation of the blog: "
          f"{games} play_solo games, every shot identical: OK")


def check_2_public_only():
    """Same agent_seed on every 4x4 layout: two games must make identical shots
    until the first shot whose public result differs."""
    records = [play_solo(ProbabilityDensityAgent(), MINI, layout=layout, agent_seed=0)
               for layout in all_layouts(MINI)]
    pairs = 0
    for a, b in combinations(records, 2):
        for shot_a, shot_b in zip(a.shots, b.shots):
            assert shot_a.cell == shot_b.cell, "decision used hidden information"
            if shot_a != shot_b:
                break
        pairs += 1
    print(f"[2] decisions depend only on public results: {pairs} pairs of games "
          f"over all {len(records)} 4x4 layouts: OK")


def check_3_rules():
    for config in CONFIGS:
        report = evaluate(ProbabilityDensityAgent(), config, seeds=range(300))
        again = evaluate(ProbabilityDensityAgent(), config, seeds=range(300))
        for record in report.records:
            assert check_record(record), "game ended before the fleet was sunk"
            assert record.shots[-1].result is Result.SUNK
            assert len(record.move_times) == record.num_shots
        assert [r.shots for r in report.records] == [r.shots for r in again.records]
        print(f"[3] evaluate {config.size}x{config.size} fleet={config.fleet} "
              f"reveal={config.reveal_sunk_cells}: 300 games legal, fleet sunk, "
              f"reproducible; mean {report.mean:.2f}")
    for seed in range(200):
        match = play_match(ProbabilityDensityAgent(), HuntTargetAgent(), CLASSIC,
                           first=seed % 2, seed=seed)
        winner, loser = match.records[match.winner], match.records[1 - match.winner]
        assert check_record(winner) and not check_record(loser)
        assert winner.num_shots - loser.num_shots == (1 if match.winner == match.first else 0)
    print("[3] play_match: 200 two-player games legal, winner sank the fleet first: OK")


def check_4_bayes():
    """Exact posterior on 4x4, rebuilt from each GameRecord's shot history."""
    layouts = list(all_layouts(MINI))

    def explains(layout, history):
        ship_of = {c: i for i, s in enumerate(layout) for c in s.cells}
        hits = [set() for _ in layout]
        for shot in history:
            i = ship_of.get(shot.cell)
            if i is None:
                result, sunk = Result.MISS, None
            else:
                hits[i].add(shot.cell)
                done = hits[i] == layout[i].cells
                result = Result.SUNK if done else Result.HIT
                sunk = layout[i].length if done else None
            if (result, sunk) != (shot.result, shot.sunk_length):
                return False
        return True

    report = evaluate(ProbabilityDensityAgent(), MINI, layouts=layouts)
    states = best_picks = impossible = sure = sure_taken = 0
    for record in report.records:
        for k, shot in enumerate(record.shots):
            history = record.shots[:k]
            worlds = [L for L in layouts if explains(L, history)]
            assert record.layout in worlds
            fired = {s.cell for s in history}
            post = {c: sum(any(c in s.cells for s in L) for L in worlds) / len(worlds)
                    for c in range(MINI.num_cells) if c not in fired}
            best = max(post.values())
            states += 1
            best_picks += post[shot.cell] == best
            impossible += post[shot.cell] == 0
            if best == 1:
                sure += 1
                sure_taken += post[shot.cell] == 1
    print(f"[4] exact Bayes on 4x4 ({states} shots from evaluate over all layouts): "
          f"max-probability cell {best_picks / states:.1%}, impossible cells {impossible}, "
          f"certain hit taken {sure_taken}/{sure}")


def check_5_stats():
    report = evaluate(ProbabilityDensityAgent(), CLASSIC, seeds=range(10_000))
    ci = 1.96 * report.std / report.games ** 0.5
    print(f"[5] evaluate 10x10, 10000 games: mean {report.mean:.2f} ± {ci:.2f}, "
          f"median {report.median}, best {report.best}, worst {report.worst}, "
          f"{report.mean_move_time * 1000:.3f} ms/move  (blog: median 42, worst 73)")
    print(round_robin([RandomAgent(), HuntTargetAgent(parity=True), ProbabilityDensityAgent()],
                      CLASSIC, games_per_pair=500))


if __name__ == "__main__":
    check_1_reference()
    check_2_public_only()
    check_3_rules()
    check_4_bayes()
    check_5_stats()
