"""
bound_gap_analysis.py

Bound gap analysis for one experiment.

Loads results produced by evaluate_leakage.py (via lower_leakage_vis.py or
directly) and plots the 4-panel gap figure:
  → results/{experiment}/plots/gap_analysis.pdf

Usage
-----
  python bound_gap_analysis.py --experiment simnpo_forget05
  python bound_gap_analysis.py --experiment grad_ascent_forget01 --force
"""

import argparse
import matplotlib.pyplot as plt

from config import EXPERIMENTS, make_dirs
from evaluate_leakage import run_evaluation
from visualize import plot_gap_analysis


def main():
    parser = argparse.ArgumentParser(description="Bound gap analysis.")
    parser.add_argument(
        "--experiment", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment name (key in config.EXPERIMENTS)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run sampling even if cached scores exist",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open the interactive figure window",
    )
    args = parser.parse_args()

    paths   = make_dirs(args.experiment)
    results = run_evaluation(args.experiment, force=args.force)

    plot_gap_analysis(results, save_path=paths["gap_analysis_fig"])

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
