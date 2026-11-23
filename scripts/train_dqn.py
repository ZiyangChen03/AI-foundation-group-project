"""Train a DQN agent and save the model and learning curve.

    uv run scripts/train_dqn.py --board mini --variant board --games 20000
    uv run scripts/train_dqn.py --board classic --variant residual --games 50000

Variants (docs/design_dqn.md, section 5):
    board     A: board planes only, learned from scratch
    density   B: board planes + the probability-density map as an input
    residual  C: B, plus Q = c * density + network (starts as probability density)

Validation boards: on 4x4 all 264 layouts (the exact expected shots); on 10x10
seeds 100000+ (the final benchmark uses seeds 0-1999, so they never overlap).
The checkpoint with the best validation score is kept as ``best.pt``.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battleship import CLASSIC, MINI, all_layouts  # noqa: E402
from battleship.dqn import model as qmodel  # noqa: E402
from battleship.dqn.model import NetConfig  # noqa: E402
from battleship.dqn.train import TrainConfig, eval_layouts, train  # noqa: E402

BOARDS = {"mini": MINI, "classic": CLASSIC}
VARIANTS = {"board": {}, "density": {"use_density": True},
            "residual": {"use_density": True, "residual": 1.0}}
VALIDATION_SEED = 100_000


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", choices=BOARDS, default="mini")
    p.add_argument("--variant", choices=VARIANTS, default="board")
    p.add_argument("--games", type=int, default=20_000)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--residual", type=float, help="density weight c for the residual variant")
    p.add_argument("--eval-games", type=int, default=500, help="10x10 validation boards")
    p.add_argument("--out", type=Path, help="default: results/dqn/<board>_<variant>_s<seed>")
    for field, value in asdict(TrainConfig()).items():
        if field != "games":
            p.add_argument("--" + field.replace("_", "-"), type=type(value), default=value)
    args = p.parse_args(argv)

    config = BOARDS[args.board]
    options = dict(VARIANTS[args.variant])
    if args.residual is not None and args.variant == "residual":
        options["residual"] = args.residual
    net = NetConfig(args.channels, args.layers, **options)
    tc = TrainConfig(games=args.games, **{f: getattr(args, f) for f in asdict(TrainConfig())
                                          if f != "games"})
    layouts = (list(all_layouts(config)) if config == MINI
               else eval_layouts(config, args.eval_games, VALIDATION_SEED))
    out = args.out or Path("results/dqn") / f"{args.board}_{args.variant}_s{tc.seed}"
    out.mkdir(parents=True, exist_ok=True)
    name = f"dqn-{args.variant}"
    best = {"eval_mean": float("inf")}

    def keep_best(model, row):
        if row["eval_mean"] < best["eval_mean"]:
            best.update(row)
            qmodel.save(out / "best.pt", model, config, {"name": name, **row})

    log_file = (out / "log.txt").open("w")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"{args.board} {args.variant}: {net}  {tc}")
    started = time.time()
    model, rows = train(config, net, tc, layouts, log=log, callback=keep_best)
    seconds = time.time() - started
    qmodel.save(out / "final.pt", model, config, {"name": name, **rows[-1]})
    with (out / "curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"board": args.board, "variant": args.variant, "net": asdict(net), "train": asdict(tc),
               "validation_boards": len(layouts), "seconds": seconds,
               "parameters": sum(p.numel() for p in model.parameters()),
               "final": rows[-1], "best": best}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {seconds / 60:.1f} min; best validation {best['eval_mean']:.3f} "
        f"at {best['games']} games; final {rows[-1]['eval_mean']:.3f}")


if __name__ == "__main__":
    main()
