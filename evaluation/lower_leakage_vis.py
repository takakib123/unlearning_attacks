"""
lower_leakage_vis.py

Run the full lower-bound leakage analysis for one experiment:
  1. Sample + score (or load from cache)
  2. Compute conbo bounds
  3. Plot 5-panel lower bound figure → results/{experiment}/plots/lower_bounds.pdf

Usage
-----
  python lower_leakage_vis.py --experiment simnpo_forget05
  python lower_leakage_vis.py --experiment grad_ascent_forget01 --force
"""

import argparse
import matplotlib.pyplot as plt

from config import EXPERIMENTS, get_paths, make_dirs
from evaluate_leakage import run_evaluation
from visualize import plot_lower_bounds


def main():
    parser = argparse.ArgumentParser(description="Lower bound leakage analysis.")
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

    plot_lower_bounds(results, save_path=paths["lower_bounds_fig"])

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
