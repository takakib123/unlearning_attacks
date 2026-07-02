#!/usr/bin/env python3
"""Analyze per-question leakage (m_bin) before vs after training.

For every `*eval_progress_perq.csv` found under the experiment folders, this
compares each question's m_bin at the first outer_step (before training)
against the last outer_step (after training), and reports how many questions
rise, fall, or stay the same.

Usage:
    python analyze_perq_leakage.py                # scan default root (this dir)
    python analyze_perq_leakage.py --root some/dir
    python analyze_perq_leakage.py a.csv b.csv    # specific files
"""
import argparse
import glob
import os

import pandas as pd


def analyze_file(csv_path: str, eps: float = 1e-6) -> dict:
    df = pd.read_csv(csv_path)

    first_step = df["outer_step"].min()
    last_step = df["outer_step"].max()

    before = df[df["outer_step"] == first_step].set_index("question_idx")
    after = df[df["outer_step"] == last_step].set_index("question_idx")

    common = before.index.intersection(after.index)
    cmp = pd.DataFrame(
        {
            "set": before.loc[common, "set"],
            "question": before.loc[common, "question"],
            "m_bin_before": before.loc[common, "m_bin"],
            "m_bin_after": after.loc[common, "m_bin"],
        }
    )
    cmp["delta"] = cmp["m_bin_after"] - cmp["m_bin_before"]

    def classify(d: float) -> str:
        if d > eps:
            return "rise"
        if d < -eps:
            return "fall"
        return "same"

    cmp["direction"] = cmp["delta"].apply(classify)

    counts = cmp["direction"].value_counts()
    n_rise = int(counts.get("rise", 0))
    n_fall = int(counts.get("fall", 0))
    n_same = int(counts.get("same", 0))

    # Per-question detail next to the source CSV.
    out_path = csv_path.replace(".csv", "_before_after.csv")
    cmp.sort_values("delta").to_csv(out_path, index=False)

    print(f"\n=== {csv_path} ===")
    print(f"  before outer_step={first_step}, after outer_step={last_step}")
    print(f"  questions compared: {len(cmp)}")
    print(f"  rise: {n_rise}   fall: {n_fall}   same: {n_same}")
    print(f"  mean m_bin: {cmp['m_bin_before'].mean():.4f} -> "
          f"{cmp['m_bin_after'].mean():.4f} (delta {cmp['delta'].mean():+.4f})")
    by_set = cmp.groupby(["set", "direction"]).size().unstack(fill_value=0)
    print(by_set.to_string().replace("\n", "\n  ").rjust(0))
    print(f"  detail -> {out_path}")

    return {
        "file": os.path.relpath(csv_path),
        "before_step": int(first_step),
        "after_step": int(last_step),
        "n_questions": len(cmp),
        "rise": n_rise,
        "fall": n_fall,
        "same": n_same,
        "mean_before": round(cmp["m_bin_before"].mean(), 4),
        "mean_after": round(cmp["m_bin_after"].mean(), 4),
        "mean_delta": round(cmp["delta"].mean(), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        help="Specific CSV files. If omitted, scan --root recursively.",
    )
    ap.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root dir to scan for *eval_progress_perq.csv (default: script dir).",
    )
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Tolerance below which a change counts as 'same'.")
    args = ap.parse_args()

    if args.files:
        paths = sorted(args.files)
    else:
        pattern = os.path.join(args.root, "**", "*eval_progress_perq.csv")
        paths = sorted(p for p in glob.glob(pattern, recursive=True)
                       if not p.endswith("_before_after.csv"))

    if not paths:
        print(f"No *eval_progress_perq.csv found under {args.root}")
        return

    print(f"Found {len(paths)} file(s) to analyze.")
    summary = [analyze_file(p, args.eps) for p in paths]

    summary_df = pd.DataFrame(summary)
    print("\n\n======== SUMMARY (all experiments) ========")
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(args.root, "perq_leakage_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
