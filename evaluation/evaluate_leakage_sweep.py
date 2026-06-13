"""
evaluate_leakage_sweep.py

Decoder-sweep worst-case leakage bound.

For each temperature in a grid, calls evaluate_leakage.run_evaluation with a
dynamically-injected per-temperature experiment config, then computes a
per-question combined interval [L_D, M_D] covering all decoders in the sweep.

Bonferroni correction across K temperatures gives joint coverage 1 - 2*ALPHA.

Combined bounds:
  M_D(q) = max_i M_i(q)   (tightest upper bound that covers all decoders)
  L_D(q) = max_i L_i(q)   (lower bound on the supremum, not the infimum)

Usage
-----
  python evaluate_leakage_sweep.py --base-experiment simnpo_forget05
  python evaluate_leakage_sweep.py --base-experiment simnpo_forget05 --force
  python evaluate_leakage_sweep.py --base-experiment simnpo_forget05 \\
      --temperatures 0.01 0.5 1.0 --num-samples 32
"""

import argparse
import os

import conbo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALPHA, EXPERIMENTS, RESULTS_BASE, make_dirs
from evaluate_leakage import run_evaluation

DEFAULT_TEMPERATURES = [0.01, 0.25, 0.5, 0.75, 1.0]
DEFAULT_NUM_SAMPLES  = 64


# ---------------------------------------------------------------------------
# Config injection
# ---------------------------------------------------------------------------

def _register_sweep_exp(base_name: str, temp: float, num_samples: int) -> str:
    """
    Copy the base experiment config with temperature and num_samples overrides
    and insert it into EXPERIMENTS under a derived key.

    EXPERIMENTS is the module-level dict imported from config; mutations here
    are visible to run_evaluation (same object reference via Python's import
    singleton guarantee), so no file edits are needed.
    """
    exp_name = f"{base_name}_T{temp}"
    if exp_name not in EXPERIMENTS:
        EXPERIMENTS[exp_name] = {
            **EXPERIMENTS[base_name],
            "temperature": temp,
            "num_samples": num_samples,
        }
    return exp_name


# ---------------------------------------------------------------------------
# Bonferroni-corrected per-decoder bounds
# ---------------------------------------------------------------------------

def _decoder_bounds(scores: np.ndarray, k: int) -> tuple[float, float, float]:
    """
    Return (sample_mean, lower, upper) at per-decoder alpha = 2*ALPHA / k.
    k is the number of temperatures in the sweep (Bonferroni divisor).
    """
    sm, lo, hi = conbo.expectation_bounds(scores, alpha=2 * ALPHA / k)
    return sm, lo, hi


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    base_name: str,
    temperatures: list[float],
    num_samples: int,
    force: bool,
) -> dict:
    """
    Run run_evaluation for each temperature and aggregate into combined bounds.

    Returns
    -------
    dict with keys:
      results_per_temp  : {float: run_evaluation result dict}
      L_D, M_D          : per-question combined bounds, shape (n_questions,)
      argmax_temp       : temperature achieving M_D per question, shape (n_questions,)
      L_per_temp        : {float: lower bound array}
      M_per_temp        : {float: upper bound array}
      mean_per_temp     : {float: sample mean array}
      greedy_scores     : shape (n_questions,)
      questions, answers: shape (n_questions,)
      temperatures      : list[float]
    """
    k = len(temperatures)
    results_per_temp: dict[float, dict] = {}

    for temp in temperatures:
        exp_name = _register_sweep_exp(base_name, temp, num_samples)
        print(f"\n{'='*60}")
        print(f"  Temperature {temp}  (experiment: {exp_name})")
        print(f"{'='*60}")
        results_per_temp[temp] = run_evaluation(exp_name, force=force)

    # Greedy scores and text are identical across temperature runs (greedy is
    # deterministic), so we can take them from any run.
    ref = results_per_temp[temperatures[0]]
    greedy_scores = ref["greedy_scores"]
    questions     = ref["questions"]
    answers       = ref["answers"]
    n_questions   = len(greedy_scores)

    # Bonferroni-corrected per-decoder bounds
    L_per_temp:    dict[float, np.ndarray] = {}
    M_per_temp:    dict[float, np.ndarray] = {}
    mean_per_temp: dict[float, np.ndarray] = {}

    for temp, res in results_per_temp.items():
        lo_arr = np.zeros(n_questions)
        hi_arr = np.zeros(n_questions)
        sm_arr = np.zeros(n_questions)
        for qid in range(n_questions):
            sm, lo, hi     = _decoder_bounds(res["all_scores"][qid], k)
            lo_arr[qid]    = lo
            hi_arr[qid]    = hi
            sm_arr[qid]    = sm
        L_per_temp[temp]    = lo_arr
        M_per_temp[temp]    = hi_arr
        mean_per_temp[temp] = sm_arr

    # Combined worst-case bounds: max over all temperatures
    L_stack = np.stack([L_per_temp[t] for t in temperatures], axis=0)  # (K, Q)
    M_stack = np.stack([M_per_temp[t] for t in temperatures], axis=0)

    L_D        = L_stack.max(axis=0)
    M_D        = M_stack.max(axis=0)
    argmax_idx  = M_stack.argmax(axis=0)
    argmax_temp = np.array([temperatures[i] for i in argmax_idx])

    return dict(
        results_per_temp = results_per_temp,
        L_D              = L_D,
        M_D              = M_D,
        argmax_temp      = argmax_temp,
        L_per_temp       = L_per_temp,
        M_per_temp       = M_per_temp,
        mean_per_temp    = mean_per_temp,
        greedy_scores    = greedy_scores,
        questions        = questions,
        answers          = answers,
        temperatures     = temperatures,
    )


# ---------------------------------------------------------------------------
# Plot 1: Leakage landscape for the headline question
# ---------------------------------------------------------------------------

def _plot1_landscape(sweep: dict, plots_dir: str) -> None:
    headline  = int(np.argmax(sweep["M_D"]))
    temps     = sweep["temperatures"]
    greedy    = float(sweep["greedy_scores"][headline])
    L_D       = float(sweep["L_D"][headline])
    M_D       = float(sweep["M_D"][headline])
    argmax_t  = float(sweep["argmax_temp"][headline])

    means = np.array([sweep["mean_per_temp"][t][headline] for t in temps])
    lows  = np.array([sweep["L_per_temp"][t][headline]    for t in temps])
    highs = np.array([sweep["M_per_temp"][t][headline]    for t in temps])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(temps, lows, highs, alpha=0.25, color="steelblue",
                    label="Per Temperature Probability bound")
    ax.plot(temps, means, "o-", color="steelblue", lw=1.8,
            label="empirical mean")

    ax.axhline(L_D, color="royalblue", linestyle="--", lw=1.4,
               label=f"L_D = {L_D:.3f}")
    ax.axhline(M_D, color="crimson",   linestyle="--", lw=1.4,
               label=f"M_D = {M_D:.3f}")

    # Greedy marker at τ=0
    ax.plot(0.0, greedy, "kx", markersize=10, markeredgewidth=2, zorder=5)
    ax.annotate("greedy", xy=(0.0, greedy),
                xytext=(0.07, greedy + 0.02),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    # Sampling (paper default) marker at τ=1.0
    mean_t1 = float(sweep["mean_per_temp"][1.0][headline])
    ax.plot(1.0, mean_t1, "ko", markersize=7, zorder=5)
    ax.annotate("sampling\n(paper default)", xy=(1.0, mean_t1),
                xytext=(0.78, mean_t1 + 0.03), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    # Arrow pointing at the temperature that achieved M_D
    x_offset = 0.10 if argmax_t < 0.6 else -0.18
    ha = "left" if x_offset > 0 else "right"
    ax.annotate(f"max at τ={argmax_t}", xy=(argmax_t, M_D),
                xytext=(argmax_t + x_offset, M_D + 0.035),
                fontsize=8, ha=ha, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))

    ax.set_xlabel("Temperature τ", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title(
        f"Leakage landscape — question {headline}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.05, 1.12)
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot1_landscape.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-question combined intervals sorted by M_D
# ---------------------------------------------------------------------------

def _plot2_coverage(sweep: dict, plots_dir: str) -> None:
    n     = len(sweep["M_D"])
    order = np.argsort(sweep["M_D"])
    x     = np.arange(n)

    L_D_sorted     = sweep["L_D"][order]
    M_D_sorted     = sweep["M_D"][order]
    greedy_sorted  = sweep["greedy_scores"][order]
    mean_t1_sorted = sweep["mean_per_temp"][1.0][order]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.vlines(x, L_D_sorted, M_D_sorted, color="steelblue", lw=1.2, alpha=0.7,
              label="[L_D, M_D]")
    ax.plot(x, greedy_sorted, "kx", markersize=5, markeredgewidth=1.2, zorder=5,
            label="greedy score")
    ax.plot(x, mean_t1_sorted, "o", color="orange", markersize=3.5, zorder=4,
            label="sample mean τ=1.0 (paper default)")

    ax.set_xlabel("Question index (sorted by M_D ascending)", fontsize=11)
    ax.set_ylabel("ROUGE-L leakage", fontsize=11)
    ax.set_title("Per-question worst-case leakage bound [L_D, M_D]", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot2_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Side-by-side — τ=1.0 only vs combined sweep
# ---------------------------------------------------------------------------

def _plot3_comparison(sweep: dict, plots_dir: str) -> None:
    n     = len(sweep["M_D"])
    # Sort by τ=1.0 upper bound for consistent x-axis across both panels
    order = np.argsort(sweep["M_per_temp"][1.0])
    x     = np.arange(n)

    L1_sorted     = sweep["L_per_temp"][1.0][order]
    M1_sorted     = sweep["M_per_temp"][1.0][order]
    L_D_sorted    = sweep["L_D"][order]
    M_D_sorted    = sweep["M_D"][order]
    greedy_sorted = sweep["greedy_scores"][order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    # Left: τ=1.0 only (paper's original bound)
    axes[0].vlines(x, L1_sorted, M1_sorted, color="steelblue", lw=1.2, alpha=0.7,
                   label="[L_1.0, M_1.0]")
    axes[0].plot(x, greedy_sorted, "kx", markersize=5, markeredgewidth=1.2, zorder=5,
                 label="greedy score")
    axes[0].set_xlabel("Question index", fontsize=10)
    axes[0].set_ylabel("ROUGE-L leakage", fontsize=10)
    axes[0].set_title("τ=1.0 only (paper default)", fontsize=10)
    axes[0].legend(fontsize=8)

    # Right: combined sweep bound
    axes[1].vlines(x, L_D_sorted, M_D_sorted, color="crimson", lw=1.2, alpha=0.7,
                   label="[L_D, M_D] (sweep)")
    axes[1].plot(x, greedy_sorted, "kx", markersize=5, markeredgewidth=1.2, zorder=5,
                 label="greedy score")
    axes[1].set_xlabel("Question index", fontsize=10)
    axes[1].set_title("Combined sweep bound", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Coverage comparison: single decoder (τ=1.0) vs worst-case sweep",
        fontsize=11,
    )
    fig.tight_layout()

    path = os.path.join(plots_dir, "plot3_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary CSV + stdout report
# ---------------------------------------------------------------------------

def _save_summary(sweep: dict, sweep_dir: str) -> None:
    temps   = sweep["temperatures"]
    n       = len(sweep["M_D"])
    greedy  = sweep["greedy_scores"]

    rows = []
    for qid in range(n):
        row: dict = {
            "question_id":   qid,
            "question_text": sweep["questions"][qid],
            "greedy_score":  float(greedy[qid]),
            "L_D":           float(sweep["L_D"][qid]),
            "M_D":           float(sweep["M_D"][qid]),
            "argmax_temp":   float(sweep["argmax_temp"][qid]),
        }
        for t in temps:
            ts = str(t)
            row[f"L_at_T{ts}"]    = float(sweep["L_per_temp"][t][qid])
            row[f"M_at_T{ts}"]    = float(sweep["M_per_temp"][t][qid])
            row[f"mean_at_T{ts}"] = float(sweep["mean_per_temp"][t][qid])

        row["greedy_inside_combined_bound"] = bool(
            sweep["L_D"][qid] <= greedy[qid] <= sweep["M_D"][qid]
        )
        row["greedy_inside_T1_bound"] = bool(
            sweep["L_per_temp"][1.0][qid] <= greedy[qid] <= sweep["M_per_temp"][1.0][qid]
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(sweep_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to {csv_path}")

    n_outside_combined = int((~df["greedy_inside_combined_bound"]).sum())
    n_outside_t1       = int((~df["greedy_inside_T1_bound"]).sum())
    mode_temp          = float(df["argmax_temp"].value_counts().idxmax())

    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions where greedy falls OUTSIDE [L_D, M_D]:     {n_outside_combined} / {n}")
    print(f"  Questions where greedy falls OUTSIDE [L_1.0, M_1.0]: {n_outside_t1} / {n}")
    print(f"  Temperature most often achieving M_D:                τ={mode_temp}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decoder-sweep worst-case leakage bound across temperatures."
    )
    parser.add_argument(
        "--base-experiment", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Base experiment name (key in config.EXPERIMENTS)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run sampling even if cached scores exist",
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float,
        default=DEFAULT_TEMPERATURES,
        metavar="T",
        help="Temperature grid (default: 0.01 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help="Stochastic samples per question per temperature (default: 64)",
    )
    args = parser.parse_args()

    temperatures = sorted(set(args.temperatures))

    # τ=1.0 must be present for the comparison plot (Plot 3)
    if 1.0 not in temperatures:
        print("Note: adding τ=1.0 to grid (required for comparison plot)")
        temperatures = sorted(temperatures + [1.0])

    k = len(temperatures)
    alpha_per = 2 * ALPHA / k

    sweep_dir = os.path.join(RESULTS_BASE, f"sweep_{args.base_experiment}")
    plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Base experiment   : {args.base_experiment}")
    print(f"Temperatures      : {temperatures}")
    print(f"Num samples / temp: {args.num_samples}")
    print(f"ALPHA             : {ALPHA}  →  Bonferroni alpha/decoder = {alpha_per:.5f}  (K={k})")
    print(f"Sweep output dir  : {sweep_dir}")

    sweep = run_sweep(
        base_name    = args.base_experiment,
        temperatures = temperatures,
        num_samples  = args.num_samples,
        force        = args.force,
    )

    print("\nGenerating plots …")
    _plot1_landscape(sweep, plots_dir)
    _plot2_coverage(sweep,  plots_dir)
    _plot3_comparison(sweep, plots_dir)

    _save_summary(sweep, sweep_dir)


if __name__ == "__main__":
    main()
