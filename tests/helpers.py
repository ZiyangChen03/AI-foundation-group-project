"""Helpers for checking agents from the GameRecords the runner returns."""

from itertools import combinations


def public_feedback(record):
    """Each Shot plus the cells of the ship it sank (revealed by the rules)."""
    for shot in record.shots:
        sunk = None
        if record.config.reveal_sunk_cells and shot.result.name == "SUNK":
            sunk = next(s.cells for s in record.layout if shot.cell in s.cells)
        yield shot, sunk


def assert_public_only(records):
    """Records played with the same agent seed fire at the same cells until the
    first shot whose public feedback differs."""
    for a, b in combinations(records, 2):
        for (shot_a, seen_a), (shot_b, seen_b) in zip(public_feedback(a), public_feedback(b)):
            assert shot_a.cell == shot_b.cell
            if (shot_a, seen_a) != (shot_b, seen_b):
                break


def states(record):
    """For each shot, the public state it was chosen from:
    (shot, fired cells, open hits, remaining ship lengths)."""
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
