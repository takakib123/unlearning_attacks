"""
leakage_bounds_from_csv.py

Reads a per-question leakage CSV, computes Clopper-Pearson lower and upper
bounds on the binary leakage probability, writes them back to the CSV, and
produces a histogram of the upper bound comparing greedy vs probabilistic.

Input columns
-------------
question_idx, question, n_samples, s_n, p_hat, m_bin, greedy_leak, greedy_text

Output columns added
--------------------
lb_bin   — Clopper-Pearson lower bound on Pr[leak]
ub_bin   — Clopper-Pearson upper bound on Pr[leak]

Output figure
-------------
<csv_stem>_upper_bound_hist.png/.pdf

Bound convention
----------------
One-sided 1 - ALPHA confidence:
    lower(k, n) = Beta.ppf(alpha,     k,     n - k + 1)     (= 0 when k = 0)
    upper(k, n) = Beta.ppf(1 - alpha, k + 1, n - k)         (= 1 when k = n)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta


# ---------------------------------------------------------------------------
# Clopper-Pearson bounds
# ---------------------------------------------------------------------------
def cp_bounds(k, n, alpha=0.01):
    """One-sided Clopper-Pearson bounds on a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.broadcast_to(np.asarray(n, dtype=float), k.shape)

    lower = beta.ppf(alpha,     k,     n - k + 1)
    upper = beta.ppf(1 - alpha, k + 1, n - k)

    # ppf is NaN at the boundaries — patch them.
    lower = np.where(k == 0, 0.0, lower)
    upper = np.where(k == n, 1.0, upper)
    return lower, upper


# ---------------------------------------------------------------------------
# Histogram: greedy vs probabilistic upper bound
# ---------------------------------------------------------------------------
def plot_upper_bound_histogram(ub_prob, greedy_leak, out_path, title=None):
    """
    Overlay histogram of upper bound on binary leakage.

    ub_prob     : array of CP upper bounds from probabilistic sampling
    greedy_leak : array of 0/1 greedy outcomes (raw, treated as upper bound
                  since n = 1)
    """
    bins    = np.linspace(0, 1, 11)
    weights = np.ones(len(ub_prob)) / len(ub_prob)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Greedy rendered behind, probabilistic on top.
    ax.hist(greedy_leak.astype(float), bins=bins, weights=weights,
            color="#1f77b4", label="Greedy",
            edgecolor="white", linewidth=0.5)
    ax.hist(ub_prob, bins=bins, weights=weights,
            color="#ff7f0e", label="Probabilistic",
            edgecolor="white", linewidth=0.5)

    ax.set_xlabel(r"Upper Bound on leakage ($M_{\mathrm{bin}}$)")
    ax.set_ylabel("Question ratio")
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.1f}".lstrip("0") if 0 < b < 1 else f"{int(b)}"
                        for b in bins])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to per-question CSV")
    ap.add_argument("-o", "--out", default=None,
                    help="Output CSV (default: overwrite input)")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="One-sided significance (default 0.01)")
    ap.add_argument("--fig", default=None,
                    help="Figure output path stem "
                         "(default: <csv_stem>_upper_bound_hist)")
    ap.add_argument("--title", default=None, help="Optional figure title")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"s_n", "n_samples", "greedy_leak"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Bounds
    lb, ub = cp_bounds(df["s_n"].values, df["n_samples"].values, alpha=args.alpha)
    df["lb_bin"] = lb
    df["ub_bin"] = ub

    # Save CSV
    out_csv = Path(args.out) if args.out else Path(args.csv)
    df.to_csv(out_csv, index=False)

    # Save figure
    fig_stem = Path(args.fig) if args.fig else \
               Path(args.csv).with_name(Path(args.csv).stem + "_upper_bound_hist")
    plot_upper_bound_histogram(ub, df["greedy_leak"].values, fig_stem,
                               title=args.title)

    # Sanity summary
    print(f"Wrote {out_csv}  ({len(df)} rows, alpha={args.alpha})")
    print(f"Wrote {fig_stem.with_suffix('.png')}")
    print(f"  ub_bin  min={ub.min():.4f}  max={ub.max():.4f}  mean={ub.mean():.4f}")
    print(f"  lb_bin  min={lb.min():.4f}  max={lb.max():.4f}  mean={lb.mean():.4f}")
    print(f"  questions with lb_bin > 0.1: {(lb > 0.1).sum()} / {len(df)}")
    print(f"  questions with ub_bin > 0.1: {(ub > 0.1).sum()} / {len(df)}")
    print(f"  greedy leaks: {int(df['greedy_leak'].sum())} / {len(df)}")


if __name__ == "__main__":
    main()