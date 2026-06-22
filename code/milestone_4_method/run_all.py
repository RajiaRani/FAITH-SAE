"""run_all.py — run the whole milestone 4 pipeline end to end (the default entry).

This just calls step1 -> step2 -> step3 -> step4 -> step5 in order, so you get
the full offline comparison (outputs/method_compare.csv + method_compare.png)
with ONE command:

    /usr/bin/python3 run_all.py

You can also run the steps one at a time (see the README) to read each stage's
teaching output. Each step is independent on disk: it reads the previous step's
saved file from outputs/ and writes its own.
"""
from __future__ import annotations

from _common import banner


def main() -> None:
    import step1_build_bank
    import step2_estimate_subspace
    import step3_steer_and_residual
    import step4_score_cfs
    import step5_plot

    step1_build_bank.main()
    step2_estimate_subspace.main()
    step3_steer_and_residual.main()
    step4_score_cfs.main()
    step5_plot.main()

    banner("MILESTONE 4 COMPLETE")
    print("  outputs/method_compare.csv  — variant, off-manifold residual, CFS")
    print("  outputs/method_compare.png  — bar (CFS) + scatter (residual vs CFS)")
    print("  Success = on-manifold CFS > naive CFS AND on-manifold residual ~ 0.")


if __name__ == "__main__":
    main()
