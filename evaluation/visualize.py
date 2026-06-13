"""
visualize.py

All visualization functions for the probabilistic unlearning analysis.

Two public functions:
  plot_lower_bounds(results, save_path)  — 5-panel lower bound analysis
  plot_gap_analysis(results, save_path)  — 4-panel bound gap analysis

Both accept a dict with keys matching the output of evaluate_leakage.run_evaluation().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import conbo


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _setup_style():
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    colors = sns.color_palette("colorblind")
    return {
        "lower":  colors[0],
        "upper":  colors[1],
        "greedy": colors[2],
        "gap":    colors[3],
    }


def _clean(arr, lo=0.0, hi=1.0):
    """Replace NaN/Inf (possible from conbo on edge-case inputs) with safe values."""
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi)


def _colorbar(sc, ax, label):
    """
    Add a colorbar and disable rasterization on its internal PolyCollection.
    Without this, plt.colorbar() creates an AxesImage that the PDF backend
    rasterizes using an uninitialized display transform → trillion-pixel crash.
    """
    cb = plt.colorbar(sc, ax=ax, label=label)
    if hasattr(cb, "solids") and cb.solids is not None:
        cb.solids.set_rasterized(False)
    return cb


def _classify_regime(lo, hi):
    if lo > 0.3:  return "Definite leak",    "crimson"
    if hi < 0.05: return "Likely unlearned", "steelblue"
    if lo > 0.1:  return "Likely leaks",     "darkorange"
    return              "Uncertain",          "gray"


# ---------------------------------------------------------------------------
# Panel helpers — lower bounds figure
# ---------------------------------------------------------------------------
def _panel_survival(ax, exp_lower, exp_upper, greedy_scores, c):
    thresholds      = np.linspace(0, 1, 300)
    surv_lower      = [(exp_lower    > t).mean() for t in thresholds]
    surv_upper      = [(exp_upper    > t).mean() for t in thresholds]
    surv_greedy     = [(greedy_scores > t).mean() for t in thresholds]

    ax.plot(thresholds, surv_lower,  color=c["lower"],  lw=2,   label="Lower bound")
    ax.plot(thresholds, surv_upper,  color=c["upper"],  lw=2,   label="Upper bound")
    ax.plot(thresholds, surv_greedy, color=c["greedy"], lw=1.5, ls="--", label="Greedy")
    ax.fill_between(thresholds, surv_lower, surv_upper,
                    color=c["lower"], alpha=0.12, label="Bound corridor")

    frac_01 = (exp_lower > 0.1).mean()
    ax.axvline(0.1, color="gray", lw=0.8, ls=":")
    ax.text(0.12, 0.95, f"{frac_01:.0%} of questions\nguaranteed leakage > 0.1",
            transform=ax.get_xaxis_transform(), fontsize=8, va="top", color="gray")

    ax.set_xlabel("ROUGE-L threshold t")
    ax.set_ylabel("Fraction of questions")
    ax.set_title("1 · Survival curve  P(leakage > t)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _panel_lb_vs_var(ax, sample_var, exp_lower, greedy_scores):
    sc = ax.scatter(sample_var, exp_lower, c=greedy_scores,
                    cmap="RdYlGn_r", alpha=0.75, s=30, edgecolors="none",
                    vmin=0, vmax=0.5, rasterized=False)
    _colorbar(sc, ax, "Greedy ROUGE-L")

    var_med = np.median(sample_var)
    lb_med  = np.median(exp_lower)
    ax.axvline(var_med, color="gray", lw=0.8, ls=":")
    ax.axhline(lb_med,  color="gray", lw=0.8, ls=":")

    ax.text(var_med * 0.05, np.percentile(exp_lower, 90),
            "Low var\nHigh lower bound\n→ Reliably leaks",
            fontsize=7, color="crimson", va="top")
    ax.text(np.percentile(sample_var, 80), lb_med * 0.1,
            "High var\nLow lower bound\n→ Occasionally leaks",
            fontsize=7, color="steelblue", va="bottom")

    ax.set_xlabel("Sample variance of ROUGE-L scores")
    ax.set_ylabel("Lower bound on E[leakage]")
    ax.set_title("2 · Lower bound vs. score variance")


def _panel_convergence(ax, all_scores, exp_lower, alpha, num_samples):
    n_questions  = len(exp_lower)
    high_leak_qs = np.argsort(exp_lower)[-2:]
    low_leak_qs  = np.argsort(exp_lower)[:2]
    mid_leak_q   = [np.argsort(exp_lower)[n_questions // 2]]
    highlight_qs = np.concatenate([high_leak_qs, low_leak_qs, mid_leak_q])

    sample_ns    = [8, 16, 32, 64, 96, 128]
    conv_palette = sns.color_palette("tab10", len(highlight_qs))

    for ci_idx, qid in enumerate(highlight_qs):
        lowers = []
        for n in sample_ns:
            _, lo, _ = conbo.expectation_bounds(all_scores[qid, :n], alpha=2 * alpha)
            lowers.append(lo)
        ax.plot(sample_ns, lowers, marker="o", ms=4, lw=1.5,
                color=conv_palette[ci_idx],
                label=f"Q{qid} (lb={exp_lower[qid]:.2f})")

    ax.axvline(num_samples, color="gray", lw=0.8, ls="--", label=f"n={num_samples}")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Lower bound estimate")
    ax.set_title("3 · Lower bound convergence vs. sample size")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, 140)


def _panel_regime(ax, exp_lower, exp_upper):
    regime_colors = [_classify_regime(lo, hi)[1] for lo, hi in zip(exp_lower, exp_upper)]
    ax.scatter(exp_lower, exp_upper, c=regime_colors, alpha=0.7, s=28,
               edgecolors="none", rasterized=False)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.3)

    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=lbl)
        for lbl, col in [("Definite leak", "crimson"), ("Likely leaks", "darkorange"),
                         ("Uncertain", "gray"), ("Likely unlearned", "steelblue")]
    ]
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_xlabel("Lower bound on E[leakage]")
    ax.set_ylabel("Upper bound on E[leakage]")
    ax.set_title("4 · Leakage regime: (lower, upper) per question")
    ax.set_xlim(0, None); ax.set_ylim(0, None)


def _panel_kde(ax, exp_lower, exp_upper, greedy_scores, c):
    sns.kdeplot(exp_lower,     fill=True,  color=c["lower"],  alpha=0.4, ax=ax, label="Lower bound")
    sns.kdeplot(exp_upper,     fill=True,  color=c["upper"],  alpha=0.3, ax=ax, label="Upper bound")
    sns.kdeplot(greedy_scores, fill=False, color=c["greedy"], lw=1.5, linestyle="--",
                ax=ax, label="Greedy")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for p, ls in [(25, "dotted"), (50, "dashed"), (75, "dashdot")]:
        v = np.percentile(exp_lower, p)
        ax.axvline(v, color=c["lower"], lw=0.9, ls=ls)
        ax.text(v + 0.005, 0.9, f"p{p}={v:.2f}",
                fontsize=7, color=c["lower"], rotation=90, va="top", transform=trans)

    ax.set_xlabel("ROUGE-L")
    ax.set_ylabel("Density")
    ax.set_title("5 · Population distribution of bounds (with lower-bound percentiles)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)


# ---------------------------------------------------------------------------
# Panel helpers — gap analysis figure
# ---------------------------------------------------------------------------
def _panel_sorted_intervals(ax, exp_lower, exp_upper, greedy_scores, gap, c):
    order = np.argsort(gap)[::-1]
    y     = np.arange(len(gap))

    ax.barh(y, gap[order], left=exp_lower[order],
            color=c["gap"], alpha=0.55, height=0.85,
            label="Gap (upper − lower)", rasterized=False)
    ax.scatter(exp_lower[order],     y, color=c["lower"], s=6, zorder=3,
               label="Lower bound",  rasterized=False)
    ax.scatter(exp_upper[order],     y, color=c["upper"], s=6, zorder=3,
               label="Upper bound",  rasterized=False)
    ax.scatter(greedy_scores[order], y, color="black",    s=6, marker="|",
               zorder=4, label="Greedy", rasterized=False)

    ax.set_xlabel("ROUGE-L")
    ax.set_ylabel("Question (sorted by gap width)")
    ax.set_title("1 · Bound intervals sorted by gap width")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="lower right")


def _panel_gap_dist(ax, gap, c):
    data_range = np.ptp(gap)
    if data_range == 0:
        bins, kde = 1, False
    else:
        max_bins = max(1, int(data_range / 1e-9))
        bins = min(25, max_bins) if max_bins >= 2 else 1
        kde  = bins > 1
    sns.histplot(gap, bins=bins, color=c["gap"], alpha=0.6, ax=ax, kde=kde)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for p, ls in [(25, "dotted"), (50, "dashed"), (75, "dashdot")]:
        v = np.percentile(gap, p)
        ax.axvline(v, color=c["gap"], lw=1, ls=ls)
        ax.text(v + 0.003, 0.85, f"p{p}={v:.2f}",
                fontsize=8, color=c["gap"], rotation=90, va="top", transform=trans)

    ax.set_xlabel("Gap (upper − lower)")
    ax.set_ylabel("Count")
    ax.set_title("2 · Distribution of bound gaps")


def _panel_gap_vs_var(ax, sample_var, gap, exp_lower, c):
    sc = ax.scatter(sample_var, gap, c=exp_lower,
                    cmap="YlOrRd", alpha=0.7, s=28, edgecolors="none",
                    vmin=0, vmax=0.4, rasterized=False)
    _colorbar(sc, ax, "Lower bound")

    # Guard: polyfit requires non-zero spread in x
    if np.ptp(sample_var) > 1e-12:
        m, b = np.polyfit(sample_var, gap, 1)
        if np.isfinite(m) and np.isfinite(b):
            xfit = np.linspace(sample_var.min(), sample_var.max(), 100)
            ax.plot(xfit, m * xfit + b, color="gray", lw=1.2, ls="--",
                    label=f"slope={m:.2f}")
            ax.legend(fontsize=8)

    ax.set_xlabel("Sample variance of ROUGE-L")
    ax.set_ylabel("Gap (upper − lower)")
    ax.set_title("3 · Gap vs. score variance\n(colour = lower bound)")


def _panel_gap_vs_lb(ax, exp_lower, gap, greedy_scores):
    sc = ax.scatter(exp_lower, gap, c=greedy_scores,
                    cmap="RdYlGn_r", alpha=0.75, s=28, edgecolors="none",
                    vmin=0, vmax=0.5, rasterized=False)
    _colorbar(sc, ax, "Greedy ROUGE-L")

    lb_thresh  = np.percentile(exp_lower, 75)
    gap_thresh = np.percentile(gap, 75)
    ax.axvline(lb_thresh,  color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(gap_thresh, color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.text(lb_thresh + 0.01, gap.max() * 0.97,
            "High leakage\n+ high uncertainty",
            fontsize=7, color="crimson", va="top")

    top5 = np.argsort(exp_lower)[-5:]
    for qid in top5:
        ax.annotate(f"Q{qid}", (exp_lower[qid], gap[qid]),
                    fontsize=7, color="black",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Lower bound on E[leakage]")
    ax.set_ylabel("Gap (upper − lower)")
    ax.set_title("4 · Gap vs. lower bound\n(colour = greedy score)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def plot_lower_bounds(results: dict, save_path: str | None = None) -> plt.Figure:
    """
    5-panel lower-bound analysis figure.

    Panels
    ------
    1. Survival curve  P(exp_lower > t)
    2. Lower bound vs. score variance scatter
    3. Bound convergence vs. sample size
    4. Leakage regime scatter (lower, upper)
    5. Population KDE of bounds + percentile annotations
    """
    c = _setup_style()

    exp_lower     = _clean(results["exp_lower"])
    exp_upper     = _clean(results["exp_upper"])
    greedy_scores = _clean(results["greedy_scores"])
    sample_var    = _clean(results["sample_var"], lo=0.0, hi=np.inf)
    all_scores    = results["all_scores"]
    alpha         = results["alpha"]
    num_samples   = results["hparams"]["num_samples"]
    model_name    = results["hparams"]["model"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Lower bound analysis of unlearning deficiencies\n"
        f"model: {model_name}  |  α={alpha}  |  {num_samples} samples/question",
        fontsize=12, y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    _panel_survival(   fig.add_subplot(gs[0, 0]), exp_lower, exp_upper, greedy_scores, c)
    _panel_lb_vs_var(  fig.add_subplot(gs[0, 1]), sample_var, exp_lower, greedy_scores)
    _panel_convergence(fig.add_subplot(gs[0, 2]), all_scores, exp_lower, alpha, num_samples)
    _panel_regime(     fig.add_subplot(gs[1, 0]), exp_lower, exp_upper)
    _panel_kde(        fig.add_subplot(gs[1, 1:]), exp_lower, exp_upper, greedy_scores, c)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {save_path}")

    return fig


def plot_gap_analysis(results: dict, save_path: str | None = None) -> plt.Figure:
    """
    4-panel bound gap analysis figure.

    Panels
    ------
    1. Sorted interval plot [lower, upper] with greedy ticks
    2. Distribution of bound gaps
    3. Gap vs. sample variance
    4. Gap vs. lower bound (key diagnostic)
    """
    c = _setup_style()

    exp_lower     = _clean(results["exp_lower"])
    exp_upper     = _clean(results["exp_upper"])
    greedy_scores = _clean(results["greedy_scores"])
    sample_var    = _clean(results["sample_var"], lo=0.0, hi=np.inf)
    alpha         = results["alpha"]
    num_samples   = results["all_scores"].shape[1]
    model_name    = results["hparams"]["model"]
    gap           = _clean(exp_upper - exp_lower)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Bound gap analysis  (upper − lower)\n"
        f"model: {model_name}  |  α={alpha}  |  {num_samples} samples/question",
        fontsize=12, y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    _panel_sorted_intervals(fig.add_subplot(gs[0, :2]), exp_lower, exp_upper, greedy_scores, gap, c)
    _panel_gap_dist(        fig.add_subplot(gs[0, 2]),  gap, c)
    _panel_gap_vs_var(      fig.add_subplot(gs[1, 0]),  sample_var, gap, exp_lower, c)
    _panel_gap_vs_lb(       fig.add_subplot(gs[1, 1:]), exp_lower, gap, greedy_scores)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {save_path}")

    return fig
