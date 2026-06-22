"""run_all.py — run all five ablations A1..A5 + the plot (the default entry).

This clears outputs/ablations.csv, runs each ablation step in order (each appends
its rows), then draws the multi-panel figure — the full offline ablation study
with ONE command:

    /usr/bin/python3 run_all.py

You can also run the steps one at a time (see the README) to read each ablation's
teaching output. Every CFS number is MEASURED from the data (Spearman + sklearn
probes + Cohen's-d), never looked up.
"""
from __future__ import annotations

from _common import append_rows, banner, fresh_csv


def main() -> None:
    import step1_a1_sae_type
    import step2_a2_topk_k
    import step3_a3_proj_rank
    import step4_a4_select_threshold
    import step5_a5_layer_token
    import step6_plot

    fresh_csv()                      # start the CSV clean so a full run is reproducible

    # Run each ablation; each main() returns its rows, which we append to one CSV.
    append_rows(step1_a1_sae_type.main())
    append_rows(step2_a2_topk_k.main())
    append_rows(step3_a3_proj_rank.main())
    append_rows(step4_a4_select_threshold.main())
    append_rows(step5_a5_layer_token.main())

    step6_plot.main()

    banner("MILESTONE 7 COMPLETE")
    print("  outputs/ablations.csv  — ablation_id, knob_value, variant, cfs, diagnostic, ...")
    print("  outputs/ablations.png  — one panel per ablation (A1..A5)")
    print("  Success = all five ablations ran, a row per (ablation, knob_value, variant),")
    print("            and every CFS is in [0,1].")


if __name__ == "__main__":
    main()
