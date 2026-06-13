"""
mixture_bound.py

Mixture-decoder leakage bound: pools score samples across all temperatures in
the sweep and applies a single conbo.expectation_bounds call (no Bonferroni)
to bound E[h(Y)] under the uniform mixture decoder D_mix = (1/K) Σ D_{τ_i}.

Reads from cached score CSVs produced by evaluate_leakage_sweep.py.
Does NOT re-sample the model.

Conceptual distinction from the worst-case bound:
  [L_𝒟, M_𝒟]   = bounds on   sup_D p_D  (leakiest decoder in the family)
  [L_mix, M_mix] = bounds on E_{D~uniform}[p_D]  (average leakage across family)

Both are reported together; they answer different questions.

Usage
-----
  python mixture_bound.py --base-experiment simnpo_forget05
  python mixture_bound.py --base-experiment simnpo_forget05 \\
      --temperatures 0.01 0.25 0.5 0.75 1.0
"""

import argparse
import os
import warnings

import conbo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALPHA, EXPERIMENTS, RESULTS_BASE

# DEFAULT_TEMPERATURES = [0.01, 0.25, 0.5, 0.75, 1.0]

DEFAULT_TEMPERATURES = [0.01 , 1.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _scores_path(base: str, temp: float) -> str:
    exp_name = f"{base}_T{temp}"
    return os.path.join(RESULTS_BASE, exp_name, "scores", f"{exp_name}_scores.csv")


def load_scores(base: str, temperatures: list[float]) -> dict[float, dict]:
    """
    Load per-temperature score arrays from cached sweep CSVs.

    Returns
    -------
    {temp: {"greedy": np.ndarray(n_questions,), "scores": np.ndarray(n_questions, n_samples)}}

    Note: if per-temperature sample counts are ever unequal, the pooled mean
    would no longer equal the simple average of per-temp means — pooling
    weights would need to be adjusted. That case is not implemented here
    since all sweep runs use the same num_samples.
    """
    data: dict[float, dict] = {}
    for temp in temperatures:
        path = _scores_path(base, temp)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cached scores not found for T={temp}. "
                f"Run evaluate_leakage_sweep.py first.\n  Expected: {path}"
            )
        df = pd.read_csv(path)
        score_cols = [c for c in df.columns if c.startswith("score_")]
        data[temp] = {
            "greedy": df["greedy_score"].values,
            "scores": df[score_cols].values,   # (n_questions, n_samples)
        }
    return data


def load_sweep_summary(base: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_BASE, f"sweep_{base}", "summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sweep summary not found. Run evaluate_leakage_sweep.py first.\n"
            f"  Expected: {path}"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Mixture bound computation
# ---------------------------------------------------------------------------

def compute_mixture_bounds(
    scores_data: dict[float, dict],
    temperatures: list[float],
) -> dict:
    """
    Pool scores across all temperatures and compute conbo bounds on E[h(Y)]
    under D_mix.

    No Bonferroni correction: pooled samples are treated as a single i.i.d.
    draw from the mixture (which is valid when per-decoder samples are i.i.d.
    and decoder selection is uniform). One quantity → one bound → alpha=2*ALPHA.

    Returns dict of arrays shape (n_questions,):
      sample_mean, L_mix, M_mix, sample_std, std_lower, std_upper
    """
    score_arrays = [scores_data[t]["scores"] for t in temperatures]
    n_questions  = score_arrays[0].shape[0]

    sample_mean = np.zeros(n_questions)
    L_mix       = np.zeros(n_questions)
    M_mix       = np.zeros(n_questions)
    sample_std  = np.zeros(n_questions)
    std_lower   = np.zeros(n_questions)
    std_upper   = np.zeros(n_questions)

    for qid in range(n_questions):
        pooled = np.concatenate([arr[qid] for arr in score_arrays])

        sm, lo, hi = conbo.expectation_bounds(pooled, alpha=2 * ALPHA)
        ss, sl, su = conbo.std_bounds(pooled,         alpha=2 * ALPHA)

        sample_mean[qid] = sm
        L_mix[qid]       = lo
        M_mix[qid]       = hi
        sample_std[qid]  = ss
        std_lower[qid]   = sl
        std_upper[qid]   = su

    return dict(
        sample_mean = sample_mean,
        L_mix       = L_mix,
        M_mix       = M_mix,
        sample_std  = sample_std,
        std_lower   = std_lower,
        std_upper   = std_upper,
    )


# ---------------------------------------------------------------------------
# Sanity check: pooled mean == average of per-temperature means
# ---------------------------------------------------------------------------

def _sanity_check(
    scores_data: dict[float, dict],
    temperatures: list[float],
    mixture: dict,
) -> None:
    """
    When all per-temperature sample counts are equal, the pooled mean equals
    the simple average of per-temperature means exactly (by linearity of
    expectation). Discrepancy beyond floating-point noise indicates a bug.
    """
    score_arrays = [scores_data[t]["scores"] for t in temperatures]
    n_questions  = score_arrays[0].shape[0]

    max_err = 0.0
    for qid in range(n_questions):
        avg_of_means = np.mean([arr[qid].mean() for arr in score_arrays])
        max_err = max(max_err, abs(mixture["sample_mean"][qid] - avg_of_means))

    tol = 1e-10
    if max_err > tol:
        warnings.warn(
            f"Sanity check FAILED: pooled mean != avg of per-temp means "
            f"(max difference = {max_err:.2e} > {tol:.0e}). "
            "This may indicate unequal sample counts across temperatures."
        )
    else:
        print(f"Sanity check OK: pooled mean == avg of per-temp means (max err = {max_err:.2e})")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(
    sweep_df: pd.DataFrame,
    mixture: dict,
) -> pd.DataFrame:
    n       = len(sweep_df)
    greedy  = sweep_df["greedy_score"].values
    L_D     = sweep_df["L_D"].values
    M_D     = sweep_df["M_D"].values
    L_T1    = sweep_df["L_at_T1.0"].values
    M_T1    = sweep_df["M_at_T1.0"].values
    mean_T1 = sweep_df["mean_at_T1.0"].values
    L_mix   = mixture["L_mix"]
    M_mix   = mixture["M_mix"]

    rows = []
    for qid in range(n):
        rows.append({
            "question_id":              qid,
            "question_text":            sweep_df["question_text"].iloc[qid],
            "greedy_score":             float(greedy[qid]),
            "L_mix":                    float(L_mix[qid]),
            "M_mix":                    float(M_mix[qid]),
            "sample_mean_mix":          float(mixture["sample_mean"][qid]),
            "sample_std_mix":           float(mixture["sample_std"][qid]),
            "std_lower_mix":            float(mixture["std_lower"][qid]),
            "std_upper_mix":            float(mixture["std_upper"][qid]),
            "L_D":                      float(L_D[qid]),
            "M_D":                      float(M_D[qid]),
            "L_T1":                     float(L_T1[qid]),
            "M_T1":                     float(M_T1[qid]),
            "mean_T1":                  float(mean_T1[qid]),
            "greedy_inside_mixture":    bool(L_mix[qid] <= greedy[qid] <= M_mix[qid]),
            "greedy_inside_worst_case": bool(L_D[qid]   <= greedy[qid] <= M_D[qid]),
            "greedy_inside_T1_only":    bool(L_T1[qid]  <= greedy[qid] <= M_T1[qid]),
            "mixture_width":            float(M_mix[qid] - L_mix[qid]),
            "worst_case_width":         float(M_D[qid]   - L_D[qid]),
            "T1_width":                 float(M_T1[qid]  - L_T1[qid]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    n = len(df)

    frac_mix = df["greedy_inside_mixture"].mean()
    frac_wc  = df["greedy_inside_worst_case"].mean()
    frac_t1  = df["greedy_inside_T1_only"].mean()

    mean_w_mix = df["mixture_width"].mean()
    mean_w_wc  = df["worst_case_width"].mean()
    mean_w_t1  = df["T1_width"].mean()

    mean_gap   = (df["M_D"] - df["M_mix"]).mean()

    # Theoretical claim checks
    frac_mix_narrower_wc = (df["mixture_width"] <= df["worst_case_width"]).mean()
    frac_M_mix_le_M_D    = (df["M_mix"] <= df["M_D"]).mean()

    print(f"\n{'='*65}")
    print(f"  MIXTURE BOUND REPORT  (n={n} questions, ALPHA={ALPHA})")
    print(f"{'='*65}")
    print(f"  Greedy containment:")
    print(f"    Mixture bound  [L_mix, M_mix]:   {frac_mix:.1%}  (expected: high)")
    print(f"    Worst-case     [L_D,   M_D]:     {frac_wc:.1%}")
    print(f"    τ=1.0 only     [L_T1,  M_T1]:   {frac_t1:.1%}  (expected: low)")
    print()
    print(f"  Mean interval width:")
    print(f"    τ=1.0 only:   {mean_w_t1:.4f}")
    print(f"    Mixture:      {mean_w_mix:.4f}  ← pooling diverse decoders widens vs T1")
    print(f"    Worst-case:   {mean_w_wc:.4f}  ← Bonferroni + union widens further")
    print()
    print(f"  Mean (M_D − M_mix): {mean_gap:.4f}  (worst-case upper bound above mixture)")
    print()
    print(f"  Claim checks:")
    print(f"    width_mix ≤ width_worst_case:  {frac_mix_narrower_wc:.1%} of questions")
    print(f"    M_mix ≤ M_D:                   {frac_M_mix_le_M_D:.1%} of questions  (expect ~100%)")

    if frac_mix_narrower_wc < 0.80:
        print(f"\n  WARNING: mixture interval WIDER than worst-case on "
              f"{1-frac_mix_narrower_wc:.0%} of questions.")
        print(f"  This likely indicates the Bonferroni alpha/K is not conservative "
              f"enough relative to the mixture alpha — check ALPHA and K.")
    if frac_M_mix_le_M_D < 0.90:
        print(f"\n  WARNING: M_mix > M_D on {1-frac_M_mix_le_M_D:.0%} of questions.")
        print(f"  The mixture upper bound should lie at or below the max of per-decoder")
        print(f"  upper bounds by construction — investigate pooling or bound computation.")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Plot 1: Three-bound comparison per question
# ---------------------------------------------------------------------------

def _plot1_three_bounds(df: pd.DataFrame, plots_dir: str) -> None:
    order = np.argsort(df["M_mix"].values)
    x     = np.arange(len(df))
    dx    = 0.22   # horizontal separation between the three bars

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.vlines(x - dx, df["L_T1"].values[order],  df["M_T1"].values[order],
              color="orange",    lw=1.1, alpha=0.85, label="[L_T1, M_T1]  τ=1.0 only (paper)")
    ax.vlines(x,      df["L_mix"].values[order],  df["M_mix"].values[order],
              color="steelblue", lw=1.1, alpha=0.85, label="[L_mix, M_mix]  mixture")
    ax.vlines(x + dx, df["L_D"].values[order],    df["M_D"].values[order],
              color="crimson",   lw=1.1, alpha=0.85, label="[L_D, M_D]  worst-case")

    ax.plot(x, df["greedy_score"].values[order], "kx",
            markersize=4, markeredgewidth=1.2, zorder=5, label="greedy score")

    ax.set_xlabel("Question index (sorted by M_mix ascending)", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title("Three-bound comparison per question", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot1_three_bounds.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Headline landscape with mixture + worst-case overlay
# ---------------------------------------------------------------------------

def _plot2_landscape(
    sweep_df: pd.DataFrame,
    mixture: dict,
    temperatures: list[float],
    plots_dir: str,
) -> None:
    headline  = int(np.argmax(mixture["M_mix"]))
    temps     = temperatures
    greedy    = float(sweep_df["greedy_score"].iloc[headline])
    L_D       = float(sweep_df["L_D"].iloc[headline])
    M_D       = float(sweep_df["M_D"].iloc[headline])
    L_mix     = float(mixture["L_mix"][headline])
    M_mix     = float(mixture["M_mix"][headline])
    pool_mean = float(mixture["sample_mean"][headline])

    lows  = np.array([float(sweep_df[f"L_at_T{t}"].iloc[headline]) for t in temps])
    highs = np.array([float(sweep_df[f"M_at_T{t}"].iloc[headline]) for t in temps])
    means = np.array([float(sweep_df[f"mean_at_T{t}"].iloc[headline]) for t in temps])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(temps, lows, highs, alpha=0.20, color="steelblue",
                    label="per-decoder CI band")
    ax.plot(temps, means, "o-", color="steelblue", lw=1.8,
            label="per-decoder empirical mean")

    ax.axhline(L_D,       color="crimson",      linestyle="--", lw=1.4,
               label=f"L_D = {L_D:.3f}  (worst-case)")
    ax.axhline(M_D,       color="crimson",      linestyle="--", lw=1.4,
               label=f"M_D = {M_D:.3f}  (worst-case)")
    ax.axhline(L_mix,     color="royalblue",    linestyle=":",  lw=1.6,
               label=f"L_mix = {L_mix:.3f}  (mixture)")
    ax.axhline(M_mix,     color="royalblue",    linestyle=":",  lw=1.6,
               label=f"M_mix = {M_mix:.3f}  (mixture)")
    ax.axhline(pool_mean, color="mediumpurple", linestyle="-",  lw=1.3, alpha=0.85,
               label=f"pooled mean = {pool_mean:.3f}")

    ax.plot(0.0, greedy, "kx", markersize=10, markeredgewidth=2, zorder=5)
    ax.annotate("greedy", xy=(0.0, greedy), xytext=(0.07, greedy + 0.02),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("Temperature τ", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title(
        f"Leakage landscape — question {headline} (headline: largest M_mix)",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.set_xlim(-0.05, 1.12)
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot2_landscape.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Width comparison scatter
# ---------------------------------------------------------------------------

def _plot3_scatter(df: pd.DataFrame, plots_dir: str) -> None:
    M_T1  = df["M_T1"].values
    M_mix = df["M_mix"].values
    M_D   = df["M_D"].values

    all_vals = np.concatenate([M_T1, M_mix, M_D])
    lo = all_vals.min() - 0.02
    hi = all_vals.max() + 0.02

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: M_T1 vs M_mix
    axes[0].scatter(M_T1, M_mix, s=18, alpha=0.55, color="steelblue")
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    axes[0].set_xlabel("M_T1  (τ=1.0 only)", fontsize=10)
    axes[0].set_ylabel("M_mix  (mixture)", fontsize=10)
    axes[0].set_title("Mixture vs τ=1.0-only upper bounds", fontsize=10)
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].legend(fontsize=8)

    # Right: M_D vs M_mix — points below diagonal mean mixture is tighter
    axes[1].scatter(M_D, M_mix, s=18, alpha=0.55, color="crimson")
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    axes[1].set_xlabel("M_D  (worst-case)", fontsize=10)
    axes[1].set_ylabel("M_mix  (mixture)", fontsize=10)
    axes[1].set_title("Mixture vs worst-case upper bounds", fontsize=10)
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Upper bound comparison (points below diagonal = mixture is tighter)",
        fontsize=10,
    )
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot3_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-question mixture bound with greedy markers (clean, minimal)
# ---------------------------------------------------------------------------

def _plot_per_question_greedy_coverage(df: pd.DataFrame, plots_dir: str) -> None:
    order  = np.argsort(df["M_mix"].values)
    x      = np.arange(len(df))
    n      = len(df)

    L_sorted = df["L_mix"].values[order]
    M_sorted = df["M_mix"].values[order]
    g_sorted = df["greedy_score"].values[order]

    n_inside = int(df["greedy_inside_mixture"].sum())
    pct      = 100.0 * n_inside / n

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.vlines(x, L_sorted, M_sorted, color="steelblue", lw=1.1, alpha=0.75,
              label="[L_mix, M_mix]")
    ax.plot(x, g_sorted, "kx", markersize=4, markeredgewidth=1.2, zorder=5,
            label="greedy")

    ax.text(
        0.01, 0.97,
        f"greedy ∈ [L_mix, M_mix]:  {n_inside} / {n} questions  ({pct:.1f}%)",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
    )

    ax.set_xlabel("Question index (sorted by M_mix ascending)", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title("Per-question mixture bound [L_mix, M_mix] vs greedy", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.88))
    fig.tight_layout()

    path = os.path.join(plots_dir, "per_question_greedy_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4b: Per-question T=1.0 bound with greedy markers (same style)
# ---------------------------------------------------------------------------

def _plot_per_question_T1_coverage(df: pd.DataFrame, plots_dir: str) -> None:
    order  = np.argsort(df["M_T1"].values)
    x      = np.arange(len(df))
    n      = len(df)

    L_sorted = df["L_T1"].values[order]
    M_sorted = df["M_T1"].values[order]
    g_sorted = df["greedy_score"].values[order]

    n_inside = int(df["greedy_inside_T1_only"].sum())
    pct      = 100.0 * n_inside / n

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.vlines(x, L_sorted, M_sorted, color="orange", lw=1.1, alpha=0.75,
              label="[L_T1, M_T1]")
    ax.plot(x, g_sorted, "kx", markersize=4, markeredgewidth=1.2, zorder=5,
            label="greedy")

    ax.text(
        0.01, 0.97,
        f"greedy ∈ [L_T1, M_T1]:  {n_inside} / {n} questions  ({pct:.1f}%)",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
    )

    ax.set_xlabel("Question index (sorted by M_T1 ascending)", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title("Per-question τ=1.0 bound [L_T1, M_T1] vs greedy", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.88))
    fig.tight_layout()

    path = os.path.join(plots_dir, "per_question_T1_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 5: Survival curve P(leakage > t)
# ---------------------------------------------------------------------------

def _plot_survival_curve(df: pd.DataFrame, plots_dir: str) -> None:
    n          = len(df)
    thresholds = np.linspace(0, 1, 100)

    L_mix  = df["L_mix"].values
    M_mix  = df["M_mix"].values
    greedy = df["greedy_score"].values

    # Vectorised: (n_thresholds, n_questions) → mean over questions
    S_lower  = (L_mix[None, :] > thresholds[:, None]).mean(axis=1)
    S_upper  = (M_mix[None, :] > thresholds[:, None]).mean(axis=1)
    S_greedy = (greedy[None, :] > thresholds[:, None]).mean(axis=1)

    # Values at the reference threshold t=0.1
    ref_t          = 0.1
    frac_upper_ref = float((M_mix  > ref_t).mean())
    frac_lower_ref = float((L_mix  > ref_t).mean())
    frac_greedy_ref = float((greedy > ref_t).mean())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(thresholds, S_lower, S_upper,
                    color="steelblue", alpha=0.20, label="Bound corridor")
    ax.plot(thresholds, S_lower,  color="steelblue", lw=1.8, label="Lower bound")
    ax.plot(thresholds, S_upper,  color="orange",    lw=1.8, label="Upper bound")
    ax.plot(thresholds, S_greedy, color="green",     lw=1.6, linestyle="--",
            label="Greedy")

    # Vertical reference line at t=0.1
    ax.axvline(ref_t, color="gray", lw=1.0, linestyle=":")
    ax.annotate(
        f"t={ref_t}:  upper={frac_upper_ref:.0%}  lower={frac_lower_ref:.0%}  greedy={frac_greedy_ref:.0%}",
        xy=(ref_t, frac_upper_ref),
        xytext=(ref_t + 0.04, frac_upper_ref - 0.08),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
        color="gray",
    )

    ax.set_xlabel("ROUGE-L threshold  t", fontsize=11)
    ax.set_ylabel("Fraction of questions with leakage > t", fontsize=11)
    ax.set_title("Survival curve: P(leakage > t) across questions", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(plots_dir, "survival_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Per-temperature: bound intervals sorted by gap width (horizontal bar chart)
# ---------------------------------------------------------------------------

def _plot_bound_intervals_sorted(
    sweep_df: pd.DataFrame,
    temp: float,
    plots_dir: str,
) -> None:
    """
    Horizontal bar chart for a single temperature, matching the reference style:
      - y-axis: questions, sorted by (M - L) gap width descending
        (largest gap at bottom, smallest at top)
      - x-axis: ROUGE-L
      - tan fill from L to M
      - blue dots at L, gold dots at M, black squares at greedy
    """
    t_key  = str(temp)
    L      = sweep_df[f"L_at_T{t_key}"].values
    M      = sweep_df[f"M_at_T{t_key}"].values
    greedy = sweep_df["greedy_score"].values

    gap   = M - L
    order = np.argsort(gap)[::-1]   # largest gap → y=0 (bottom), smallest → y=N-1 (top)

    L_s = L[order]
    M_s = M[order]
    g_s = greedy[order]

    n = len(L_s)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(y, width=M_s - L_s, left=L_s,
            color="tan", alpha=0.5, label="Gap (upper − lower)")
    ax.scatter(L_s, y, s=12, color="steelblue", zorder=3, label="Lower bound")
    ax.scatter(M_s, y, s=12, color="goldenrod", zorder=3, label="Upper bound")
    ax.scatter(g_s, y, s=10, color="black", marker="s", zorder=4, label="Greedy")

    ax.set_xlabel("ROUGE-L", fontsize=11)
    ax.set_ylabel("Question (sorted by gap width)", fontsize=11)
    ax.set_title(f"{temp} · Bound intervals sorted by gap width", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    path = os.path.join(plots_dir, f"bound_intervals_T{temp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixture-decoder leakage bound from cached sweep score CSVs."
    )
    parser.add_argument(
        "--base-experiment", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Base experiment name (must have sweep results cached)",
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float,
        default=DEFAULT_TEMPERATURES,
        metavar="T",
        help="Temperatures to pool (must match sweep; default: 0.01 0.25 0.5 0.75 1.0)",
    )
    args = parser.parse_args()

    temperatures = sorted(set(args.temperatures))
    base         = args.base_experiment

    out_dir   = os.path.join(RESULTS_BASE, f"mixture_{base}")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Base experiment : {base}")
    print(f"Temperatures    : {temperatures}")
    print(f"ALPHA           : {ALPHA}  (no Bonferroni — single mixture bound, alpha=2*ALPHA)")
    print(f"Output dir      : {out_dir}")

    print("\nLoading cached score CSVs …")
    scores_data = load_scores(base, temperatures)

    print("Loading sweep summary …")
    sweep_df = load_sweep_summary(base)

    print("Computing mixture bounds …")
    mixture = compute_mixture_bounds(scores_data, temperatures)

    _sanity_check(scores_data, temperatures, mixture)

    print("\nBuilding summary CSV …")
    summary_df = build_summary(sweep_df, mixture)
    csv_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    print_report(summary_df)

    print("Generating plots …")
    _plot1_three_bounds(summary_df, plots_dir)
    _plot2_landscape(sweep_df, mixture, temperatures, plots_dir)
    _plot3_scatter(summary_df, plots_dir)
    _plot_per_question_greedy_coverage(summary_df, plots_dir)
    _plot_per_question_T1_coverage(summary_df, plots_dir)
    _plot_survival_curve(summary_df, plots_dir)
    for temp in temperatures:
        _plot_bound_intervals_sorted(sweep_df, temp, plots_dir)


if __name__ == "__main__":
    main()
