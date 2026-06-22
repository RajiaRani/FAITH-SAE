#!/usr/bin/env python
# =============================================================================
# run_all.py  --  run the ENTIRE FAITH-SAE experiment suite in order (8 milestones)
# -----------------------------------------------------------------------------
# One command that runs every real experiment back-to-back on the GPU, with a
# clear banner per milestone, and a summary of all result files at the end.
# Everything is cached (model weights + datasets), so it is pure compute.
#
#   python run_all.py           # the real suite (GPU, ~20-30 min)
#   python run_all.py --smoke   # quick plumbing self-test (CPU, no downloads)
#
# If any step fails it stops and tells you which one, so you can fix and resume.
# For research and educational purposes only.
# =============================================================================
import subprocess, sys, glob

SMOKE = "--smoke" in sys.argv[1:]
SMOKE_ARGS = ["--smoke", "--seeds", "1", "--concepts", "2", "--steps", "50"]  # flags every script accepts

# (milestone label, script, supports CLI args / smoke mode)
SUITE = [
    ("M1  Foundations  - minimal real-ViT FAITH-SAE (naive vs on-manifold)", "run.py",            False),
    ("M2/M3 Data+Base  - multi-concept/seed, naive vs on-manifold, clean+OOD", "run_full.py",      True),
    ("M4  Method       - manifold-rank sweep (RQ2: on-manifold recover w/ rank?)", "run_sweep.py", True),
    ("M5  Evaluation   - downstream causal-faithfulness (steer, finish forward)", "run_downstream.py", True),
    ("M6  Headline Exp - OOD severity sweep (RQ3) -> real Figure 1", "run_severity.py",            True),
    ("M7  Real setup   - CLIP ViT-B/16 + real photos + semantic faithfulness", "run_clip_faith.py", True),
    ("M8  Analysis     - positive control + diff-of-means (TCAV) comparison", "run_clip_control.py", True),
]

print(f"FAITH-SAE full suite | smoke={SMOKE} | {len(SUITE)} milestones")
done, skipped = [], []
for i, (label, script, supports) in enumerate(SUITE, 1):
    print("\n" + "=" * 80)
    print(f"### MILESTONE {i}/{len(SUITE)} — {label}")
    if SMOKE and not supports:
        print(f"### (skipped in --smoke: {script} has no smoke mode)")
        print("=" * 80, flush=True)
        skipped.append(script)
        continue
    args = SMOKE_ARGS if (SMOKE and supports) else []
    print(f"### running: python {script} {' '.join(args)}".rstrip())
    print("=" * 80, flush=True)
    rc = subprocess.call([sys.executable, script] + args)
    if rc != 0:
        print(f"\n!! {script} exited with code {rc}. Stopping here.")
        print(f"   Fix it, or run the remaining milestones manually starting from M{i}.")
        sys.exit(rc)
    done.append(script)

print("\n" + "=" * 80)
print(f"ALL MILESTONES COMPLETE  ({len(done)} ran" + (f", {len(skipped)} skipped" if skipped else "") + ")")
print("result files written:")
for f in sorted(glob.glob("faith_sae*.csv")) + sorted(glob.glob("*REAL*.png")):
    print("   -", f)
print("=" * 80)
print("Paste the final tables (especially M6 severity, M7 CLIP, M8 control) to your guide.")
