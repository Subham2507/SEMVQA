from __future__ import annotations

import argparse
import json
import os
import sys

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="run dir or path to training_log.jsonl")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    log = args.path
    if os.path.isdir(log):
        log = os.path.join(log, "training_log.jsonl")
    rows = [json.loads(l) for l in open(log) if l.strip()]

    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    ev = [(r["step"], r["eval_loss"]) for r in rows if "eval_loss" in r]
    if not train:
        sys.exit(f"no 'loss' entries in {log}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*train), label="train loss", lw=1)
    if ev:
        ax.plot(*zip(*ev), "o-", label="eval loss", lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(os.path.dirname(log) or log)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(log) or ".", "loss_curve.png")
    fig.savefig(out, dpi=130)
    last_ev = f" | last eval {ev[-1][1]:.4f}" if ev else ""
    print(f"{len(train)} train pts | last train {train[-1][1]:.4f}{last_ev}\nwrote {out}")

if __name__ == "__main__":
    main()
