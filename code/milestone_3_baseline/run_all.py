#!/usr/bin/env python3
# ===========================================================================
#  run_all.py  —  Milestone 3 (Baseline): one command, all three steps.
#  FAITH-SAE  ·  author: Rajia Rani  ·  educational use only
# ===========================================================================
#
#  This is just a convenience driver. It runs, in order:
#    step1_train_sae.py        (train the TopK SAE -> checkpoint + loss PNG)
#    step2_select_concepts.py  (score features, keep the clean testable ones)
#    step3_naive_steer_cfs.py  (baseline naive_steer + CFS -> baseline_cfs.csv)
#
#  Run it with the project's interpreter:
#      /usr/bin/python3 run_all.py
#
#  You can still run each step on its own (they share artifacts via outputs/).
#  ========================================================================

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Import each step's main() and call them in sequence (no subprocess needed).
import step1_train_sae       # noqa: E402
import step2_select_concepts  # noqa: E402
import step3_naive_steer_cfs  # noqa: E402


def main():
    print("=" * 70)
    print("STEP 1/3  Train the TopK SAE on the synthetic activation bank")
    print("=" * 70)
    step1_train_sae.main()

    print("\n" + "=" * 70)
    print("STEP 2/3  Select clean, testable concepts")
    print("=" * 70)
    step2_select_concepts.main()

    print("\n" + "=" * 70)
    print("STEP 3/3  Baseline naive_steer + Causal Faithfulness Score (CFS)")
    print("=" * 70)
    step3_naive_steer_cfs.main()

    print("\n" + "=" * 70)
    print("ALL DONE. Artifacts in outputs/:")
    print("  - sae_topk.pt            (trained TopK SAE checkpoint)")
    print("  - sae_loss_curve.png     (reconstruction-loss curve)")
    print("  - selected_concepts.csv  (the chosen testable concepts)")
    print("  - baseline_cfs.csv       (the BASELINE CFS — beat this in M4)")
    print("=" * 70)


if __name__ == "__main__":
    main()
