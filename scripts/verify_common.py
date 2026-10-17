"""Checks shared by the verify_*.py scripts. They read only what the API returns."""

from battleship import CLASSIC, MINI, GameConfig, Result

CONFIGS = [MINI, CLASSIC, GameConfig(6, (3, 3, 2)), GameConfig(5, (1, 1, 4)),
           GameConfig(4, (3, 2), reveal_sunk_cells=False),
           GameConfig(10, (5, 4, 3, 3, 2), reveal_sunk_cells=False)]


def check_record(record):
    """Assert a GameRecord obeys the rules, judged only from its layout and shots.

    Returns True if the record's shots sank the whole fleet.
    """
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


def states(record):
    """For each shot in a GameRecord, the public state it was chosen from:
    (shot, fired cells, open hits, remaining ship lengths).

    Open hits are hits on ships not yet sunk, as the agent sees them when
    reveal_sunk_cells is True.
    """
    layout = record.layout
    ship_of = {cell: i for i, ship in enumerate(layout) for cell in ship.cells}
    hits = [set() for _ in layout]
    sunk = [False] * len(layout)
    fired = set()
    for shot in record.shots:
        open_hits = {c for i, h in enumerate(hits) if not sunk[i] for c in h}
        remaining = sorted((s.length for s, done in zip(layout, sunk) if not done), reverse=True)
        yield shot, frozenset(fired), frozenset(open_hits), tuple(remaining)
        fired.add(shot.cell)
        i = ship_of.get(shot.cell)
        if i is not None:
            hits[i].add(shot.cell)
            sunk[i] = hits[i] == layout[i].cells


def public_feedback(record):
    """What the shooter learned from each shot: the Shot, plus the cells of the
    ship it sank when the rules reveal them. Two games are publicly identical
    up to a point exactly when these agree up to that point."""
    revealed = record.config.reveal_sunk_cells
    for shot in record.shots:
        cells = None
        if revealed and shot.result is Result.SUNK:
            cells = next(s.cells for s in record.layout if shot.cell in s.cells)
        yield shot, cells


def check_public_only(records):
    """Records played with the same agent seed must fire at the same cells until
    the first shot whose public feedback differs. Returns the number of pairs."""
    from itertools import combinations
    pairs = 0
    for a, b in combinations(records, 2):
        for (shot_a, seen_a), (shot_b, seen_b) in zip(public_feedback(a), public_feedback(b)):
            assert shot_a.cell == shot_b.cell, "decision used hidden information"
            if (shot_a, seen_a) != (shot_b, seen_b):
                break
        pairs += 1
    return pairs
