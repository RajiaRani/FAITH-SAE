"""run.py — one command to run the whole Milestone 1 pipeline end to end.

Usage:
    /usr/bin/python3 run.py --smoke     # run all 5 steps, write CSV + PNG
    /usr/bin/python3 run.py             # same as --smoke (smoke is the only mode)

This is just a convenience wrapper: it calls step1..step5 in order so a learner
can run everything with a single command, then inspect outputs/. Each stepN_*.py
also runs standalone if you prefer to go one at a time and read the teaching
comments in each file.

For research and educational purposes only.
Author: Rajia Rani
"""
from __future__ import annotations

import argparse

from _common import banner

import step1_backbone_activations
import step2_train_sae
import step3_plant_concept
import step4_steer_and_score
import step5_plot_and_interpret


def main() -> None:
    ap = argparse.ArgumentParser(description="FAITH-SAE Milestone 1 smoke pipeline")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full offline CPU pipeline (default behaviour)")
    ap.parse_args()  # both `--smoke` and no-arg run the same single pipeline

    banner("FAITH-SAE  MILESTONE 1 (FOUNDATIONS)  —  FULL SMOKE PIPELINE")
    step1_backbone_activations.main()   # frozen backbone -> activations
    step2_train_sae.main()              # train the TopK SAE -> concept switches
    step3_plant_concept.main()          # plant a known concept; on/off manifold
    step4_steer_and_score.main()        # steer 4 ways -> CFS -> outputs/*.csv
    step5_plot_and_interpret.main()     # bar chart + plain-English takeaway

    banner("MILESTONE 1 DONE — see outputs/milestone1_cfs.csv and .png")


if __name__ == "__main__":
    main()
