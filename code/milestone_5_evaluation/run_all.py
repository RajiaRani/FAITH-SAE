"""run_all.py — run the whole milestone 5 pipeline end to end (the default entry).

This just calls step1 -> step2 -> step3 -> step4 in order, so you get the full
offline CFS evaluation (outputs/cfs_breakdown.csv + cfs_breakdown.png) with ONE
command:

    /usr/bin/python3 run_all.py

You can also run the steps one at a time (see the README) to read each stage's
teaching output. Each step is independent on disk: it reads the previous step's
saved files from outputs/ and writes its own.
"""
from __future__ import annotations

from _common import banner


def main() -> None:
    import step1_build_bank
    import step2_train_probes
    import step3_measure_cfs
    import step4_plot

    step1_build_bank.main()
    step2_train_probes.main()
    step3_measure_cfs.main()
    step4_plot.main()

    banner("MILESTONE 5 COMPLETE")
    print("  outputs/cfs_breakdown.csv  — variant, monotonicity, specificity, "
          "sufficiency, cfs")
    print("  outputs/cfs_breakdown.png  — grouped bars: 3 components + CFS per method")
    print("  Success = all CFS in [0,1], components MEASURED (not looked up), and")
    print("            on-manifold steering is among the most faithful methods.")


if __name__ == "__main__":
    main()
