#!/usr/bin/env python
"""Visualize per-question M_bin trajectories from the periodic eval logs.

Reads the ``*_eval_progress_perq.csv`` files written by grpo_hp_multi_v2.py,
which carry one row per question per monitor step:

    question_idx, outer_step, set, question, n_samples, s_n, p_hat,
    m_bin, greedy_leak, greedy_text

For each run it renders M_bin (Clopper-Pearson upper bound) versus outer step,
one line per question, with the Q_held and Q_F sets in separate panels.

Usage:
    python visualize_mbin_per_question.py [PERQ.csv ...] [--outdir DIR] [--show]

With no arguments it discovers every ``*_eval_progress_perq.csv`` under
../attack (recursively, including the dated experiment_* folders).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ATTACK_DIR = HERE.parent / "attack"


def plot_perq(perq_csv: Path, outdir: Path, show: bool) -> Path | None:
    df = pd.read_csv(perq_csv)
    if df.empty or "m_bin" not in df.columns:
        print(f"skip (empty/no m_bin): {perq_csv}")
        return None

    stem = perq_csv.stem.replace("_eval_progress_perq", "")
    sets = [s for s in ("Q_F", "Q_held") if s in df["set"].unique()]
    if not sets:
        sets = list(df["set"].unique())

    fig, axes = plt.subplots(1, len(sets), figsize=(7 * len(sets), 5), squeeze=False)
    fig.suptitle(f"Per-question M_bin over training — {stem}", fontsize=14, fontweight="bold")

    cmap = plt.get_cmap("tab20")
    for col, set_label in enumerate(sets):
        ax = axes[0][col]
        sub = df[df["set"] == set_label]
        qids = sorted(sub["question_idx"].unique())
        for i, qid in enumerate(qids):
            q = sub[sub["question_idx"] == qid].sort_values("outer_step")
            marker = "o" if len(q) == 1 else None
            ax.plot(q["outer_step"], q["m_bin"], marker=marker,
                    color=cmap(i % 20), label=f"q{qid}")
        n_mon = int(sub["n_samples"].iloc[0]) if "n_samples" in sub else None
        ax.set(xlabel="outer step", ylabel="M_bin (CP upper)",
               title=f"{set_label}  (n={n_mon})" if n_mon else set_label,
               ylim=(-0.05, 1.05))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{stem}_mbin_per_question.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # --- Multi-page PDF: one question per page ---
    pdf_out = outdir / f"{stem}_mbin_per_question.pdf"
    with PdfPages(pdf_out) as pdf:
        for set_label in sets:
            sub = df[df["set"] == set_label]
            for qid in sorted(sub["question_idx"].unique()):
                q = sub[sub["question_idx"] == qid].sort_values("outer_step")
                qfig, qax = plt.subplots(figsize=(8, 5))
                marker = "o" if len(q) == 1 else "o"
                qax.plot(q["outer_step"], q["m_bin"], marker=marker, color="tab:red")
                n_mon = int(q["n_samples"].iloc[0]) if "n_samples" in q else None
                qtext = str(q["question"].iloc[0]) if "question" in q else ""
                qax.set(xlabel="outer step", ylabel="M_bin (CP upper)",
                        ylim=(-0.05, 1.05))
                qax.set_title(f"{stem} — {set_label}  q{qid}"
                              + (f"  (n={n_mon})" if n_mon else "")
                              + f"\n{qtext}", fontsize=10)
                qax.grid(alpha=0.3)
                qfig.tight_layout()
                pdf.savefig(qfig)
                plt.close(qfig)
    print(f"saved {pdf_out}")

    if show:
        plt.show()
    plt.close("all")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", type=Path,
                   help="*_eval_progress_perq.csv file(s). Default: discover under ../attack.")
    p.add_argument("--outdir", type=Path, default=HERE / "figures",
                   help="directory for output PNGs (default: ./figures)")
    p.add_argument("--show", action="store_true", help="also display interactively")
    args = p.parse_args()

    logs = args.logs or sorted(ATTACK_DIR.rglob("*_eval_progress_perq.csv"))
    if not logs:
        print(f"no *_eval_progress_perq.csv found under {ATTACK_DIR}")
        return

    for log in logs:
        if not log.exists():
            print(f"skip (missing): {log}")
            continue
        plot_perq(log, args.outdir, args.show)


if __name__ == "__main__":
    main()
