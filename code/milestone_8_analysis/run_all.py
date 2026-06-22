"""run_all.py — run the whole milestone 8 analysis pipeline end to end.

This calls step1 -> step2 -> step3 -> step4 in order, so you get the full
offline analysis (per-concept CFS -> bootstrap CIs -> the two figures ->
FINDINGS.md) with ONE command:

    /usr/bin/python3 run_all.py

You can also run the steps one at a time (see the README) to read each stage's
teaching output. Each step reads the previous step's saved file from outputs/
and writes its own, so they are independent on disk.
"""
from __future__ import annotations

from _common import banner, here_path, outpath


def main() -> None:
    import step1_measure_per_concept_cfs
    import step2_bootstrap_ci
    import step3_render_figures
    import step4_write_findings

    step1_measure_per_concept_cfs.main()
    step2_bootstrap_ci.main()
    step3_render_figures.main()
    step4_write_findings.main()

    banner("MILESTONE 8 COMPLETE")
    print("  outputs/per_concept_cfs.csv     — per (variant, shift, concept) CFS")
    print("  outputs/bootstrap_ci.csv        — variant, shift, mean_cfs, ci_low, ci_high")
    print(f"  {outpath('fig1_cfs_ood_sweep.png')}")
    print(f"  {outpath('fig7_by_method_bar.png')}")
    print(f"  {here_path('FINDINGS.md')}      — the plain-language RQ1/RQ2/RQ3 answers")
    print("  Success = bootstrap_ci.csv + the 2 PNGs + FINDINGS.md exist, "
          "CIs well-formed (ci_low <= mean <= ci_high).")


if __name__ == "__main__":
    main()
