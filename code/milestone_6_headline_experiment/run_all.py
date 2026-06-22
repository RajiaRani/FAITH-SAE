"""run_all.py — run the whole milestone 6 pipeline end to end (the default entry).

This just calls step1 -> step2 -> step3 -> step4 in order, so you get the full
offline OOD faithfulness sweep (outputs/ood_cfs_sweep.csv + the headline curve
outputs/fig1_cfs_ood_sweep.png) with ONE command:

    /usr/bin/python3 run_all.py

You can also run the steps one at a time (see the README) to read each stage's
teaching output. Each step is independent on disk: it reads the previous step's
saved files from outputs/ and writes its own.
"""
from __future__ import annotations

from _common import banner


def main() -> None:
    import step1_build_clean_bank
    import step2_estimate_clean_subspace
    import step3_sweep_ood_cfs
    import step4_plot_headline

    step1_build_clean_bank.main()
    step2_estimate_clean_subspace.main()
    step3_sweep_ood_cfs.main()
    step4_plot_headline.main()

    banner("MILESTONE 6 COMPLETE")
    print("  outputs/ood_cfs_sweep.csv       — shift_level, severity_index, variant,")
    print("                                    cfs (+ CI), components, offsheet, residual")
    print("  outputs/fig1_cfs_ood_sweep.png  — HEADLINE: CFS vs shift severity, both")
    print("                                    methods, CI bands, collapse knee + floor")
    print("  Success = all CFS in [0,1] AND naive collapses at least as fast as on-manifold.")


if __name__ == "__main__":
    main()
