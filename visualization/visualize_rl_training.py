#!/usr/bin/env python
"""Visualize GRPO RL training logs from the unlearning attack runs.

The train-log CSV mixes two kinds of rows per ``outer_step``:
  * ``ppo_epoch == -1``  -> rollout / reward sampling (one row per prompt)
  * ``ppo_epoch >= 0``   -> PPO update epochs (carry pg_loss, kl_loss, ...)

This script aggregates both per outer step and renders a multi-panel figure of
reward, loss, KL, clipping, grad-norm and saturation curves. If a matching
``*_eval_progress.csv`` exists alongside the train log, its leak/p-hat metrics
are plotted too.

Usage:
    python visualize_rl_training.py [TRAIN_LOG.csv ...] [--outdir DIR] [--show]

With no arguments it defaults to the q5_s0 train log in ../attack.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE.parent / "attack" / "experiment_2026-06-14/grpo_hp_multi_q10_s0_train_log.csv"


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into per-step rollout stats and per-step PPO-update stats."""
    rollout = df[df["ppo_epoch"] == -1]
    update = df[df["ppo_epoch"] >= 0]

    # Reward + saturation come from the rollout rows (one per prompt).
    roll_g = (
        rollout.groupby("outer_step")
        .agg(
            reward_mean=("reward_mean", "mean"),
            reward_std=("reward_std", "mean"),
            frac_saturated=("is_saturated", "mean"),
            wall_s=("wall_s", "max"),
        )
        .reset_index()
    )

    # Optimisation metrics come from the PPO-update rows.
    upd_cols = ["pg_loss", "kl_loss", "approx_kl_ratio", "clip_frac", "grad_norm"]
    upd_cols = [c for c in upd_cols if c in update.columns]
    upd_g = (
        update.groupby("outer_step")[upd_cols].mean().reset_index()
        if not update.empty
        else pd.DataFrame(columns=["outer_step", *upd_cols])
    )
    return roll_g, upd_g


def plot_run(train_log: Path, outdir: Path, show: bool) -> Path:
    df = pd.read_csv(train_log)
    roll, upd = aggregate(df)
    stem = train_log.stem.replace("_train_log", "")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"GRPO RL training — {stem}", fontsize=14, fontweight="bold")

    # 1) Reward mean +/- std band.
    ax = axes[0, 0]
    ax.plot(roll["outer_step"], roll["reward_mean"], color="tab:blue", label="reward mean")
    ax.fill_between(
        roll["outer_step"],
        roll["reward_mean"] - roll["reward_std"],
        roll["reward_mean"] + roll["reward_std"],
        color="tab:blue",
        alpha=0.2,
        label="±std",
    )
    ax.set(xlabel="outer step", ylabel="reward", title="Reward")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Policy-gradient loss.
    ax = axes[0, 1]
    if "pg_loss" in upd:
        ax.plot(upd["outer_step"], upd["pg_loss"], color="tab:red")
    ax.set(xlabel="outer step", ylabel="pg_loss", title="Policy-gradient loss")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)

    # 3) KL terms.
    ax = axes[0, 2]
    if "kl_loss" in upd:
        ax.plot(upd["outer_step"], upd["kl_loss"], color="tab:green", label="kl_loss")
    if "approx_kl_ratio" in upd:
        ax.plot(upd["outer_step"], upd["approx_kl_ratio"], color="tab:olive", label="approx_kl_ratio")
    ax.set(xlabel="outer step", ylabel="KL", title="KL divergence")
    ax.legend(); ax.grid(alpha=0.3)

    # 4) Clip fraction.
    ax = axes[1, 0]
    if "clip_frac" in upd:
        ax.plot(upd["outer_step"], upd["clip_frac"], color="tab:purple")
    ax.set(xlabel="outer step", ylabel="clip_frac", title="PPO clip fraction")
    ax.grid(alpha=0.3)

    # 5) Grad norm.
    ax = axes[1, 1]
    if "grad_norm" in upd and upd["grad_norm"].notna().any():
        ax.plot(upd["outer_step"], upd["grad_norm"], color="tab:brown")
    else:
        ax.text(0.5, 0.5, "no grad_norm data", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="outer step", ylabel="grad_norm", title="Gradient norm")
    ax.grid(alpha=0.3)

    # 6) Saturation fraction.
    ax = axes[1, 2]
    ax.plot(roll["outer_step"], roll["frac_saturated"], color="tab:orange")
    ax.set(xlabel="outer step", ylabel="fraction", title="Saturated prompts", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{stem}_training.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # Optional eval-progress panel (leak / p-hat over time).
    eval_path = train_log.with_name(train_log.name.replace("_train_log", "_eval_progress"))
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        if len(ev) >= 1:
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            for col, color in [
                ("qheld_mean_phat", "tab:blue"),
                ("qf_mean_phat", "tab:red"),
                ("qheld_frac_greedy_leak", "tab:green"),
                ("qf_frac_greedy_leak", "tab:orange"),
            ]:
                if col in ev.columns:
                    marker = "o" if len(ev) == 1 else None
                    ax2.plot(ev["outer_step"], ev[col], marker=marker, label=col, color=color)
            ax2.set(xlabel="outer step", ylabel="value", title=f"Eval progress — {stem}")
            ax2.legend(); ax2.grid(alpha=0.3)
            fig2.tight_layout()
            out2 = outdir / f"{stem}_eval_progress.png"
            fig2.savefig(out2, dpi=150)
            print(f"saved {out2}")

    if show:
        plt.show()
    plt.close("all")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", type=Path, default=[DEFAULT_LOG],
                   help="train-log CSV file(s). Default: q5_s0 run.")
    p.add_argument("--outdir", type=Path, default=HERE / "figures",
                   help="directory for output PNGs (default: ./figures)")
    p.add_argument("--show", action="store_true", help="also display interactively")
    args = p.parse_args()

    for log in args.logs:
        if not log.exists():
            print(f"skip (missing): {log}")
            continue
        plot_run(log, args.outdir, args.show)


if __name__ == "__main__":
    main()
