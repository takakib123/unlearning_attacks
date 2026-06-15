#!/usr/bin/env python
"""Visualize relearn eval progress, focusing on ``du2_mean_phat``.

Reads a ``*_eval_progress.csv`` (columns: outer_step, wall_s, n_monitor,
du2_mean_phat, du2_frac_greedy_leak, du2_mean_rouge, dprime_mean_phat, ...)
and plots ``du2_mean_phat`` against the relearn outer step. The matching
``dprime_mean_phat`` is overlaid for reference when present.

Usage:
    python visualize_eval_progress.py [EVAL_PROGRESS.csv ...] [--outdir DIR] [--show]

With no arguments it defaults to the s0 relearn eval log in ../attack.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = (
    HERE.parent
    / "attack"
    / "experiment_2026-06-14"
    / "grpo_tofu_relearn_s0_max_eval_progress.csv"
)


def plot_eval(csv_path: Path, outdir: Path, show: bool) -> None:
    df = pd.read_csv(csv_path).sort_values("outer_step")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["outer_step"],
        df["du2_mean_phat"],
        marker="o",
        color="tab:red",
        label="du2_mean_phat",
    )
    if "dprime_mean_phat" in df.columns:
        ax.plot(
            df["outer_step"],
            df["dprime_mean_phat"],
            marker="s",
            color="tab:blue",
            alpha=0.7,
            label="dprime_mean_phat",
        )

    ax.set_xlabel("relearn outer step")
    ax.set_ylabel("mean p-hat")
    ax.set_title(f"Eval progress: {csv_path.stem}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{csv_path.stem}_du2_mean_phat.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="*", type=Path, default=[DEFAULT_LOG])
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    for csv_path in args.csvs:
        plot_eval(csv_path, args.outdir, args.show)


if __name__ == "__main__":
    main()
