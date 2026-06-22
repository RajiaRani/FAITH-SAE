"""_common.py — shared setup every step script imports.

WHY THIS FILE EXISTS
--------------------
Each step script (step1_*.py ... step5_*.py) needs to do the SAME two boring
things before it can start teaching:
  1. Make Python able to `import` the project's real research code in `src/`.
  2. Load the knobs from `config.yaml`.
Rather than copy-paste that into every step, we write it ONCE here and every
step does `from _common import ...`. (Analogy: a kitchen's prep station — you
set out the same knives and cutting board once, not per dish.)

KEY IDEA: adding the project ROOT to `sys.path`
------------------------------------------------
`sys.path` is the list of folders Python searches when you write `import src`.
The project root (three levels up from this file:
  .../25_Rajia_Rani_FAITH_SAE/code/milestone_1_foundations/_common.py
   parents[0] = milestone_1_foundations
   parents[1] = code
   parents[2] = 25_..._FAITH_SAE   <-- the project ROOT, which contains src/)
is NOT on that list by default, so `import src` would fail. We insert it, then
`from src import ...` works. This is exactly how the milestone reuses the real
research code instead of re-implementing it.
"""
from __future__ import annotations

import pathlib
import sys

# --- 1. Put the project root on sys.path so `from src import ...` works ------
ROOT = pathlib.Path(__file__).resolve().parents[2]   # the 25_..._FAITH_SAE folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = pathlib.Path(__file__).resolve().parent       # milestone_1_foundations/


def load_cfg(path: str = "config.yaml") -> dict:
    """Read config.yaml into a plain Python dict.

    We reuse the project's own loader (`src.utils.load_config`) so this milestone
    parses config EXACTLY the way the real pipeline does. A dict is just a set of
    name->value pairs, e.g. cfg["steer_strength"] -> 4.0.
    """
    from src.utils import load_config
    p = HERE / path
    return load_config(str(p))


def banner(title: str) -> None:
    """Print a clear section header so the console output is easy to follow."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")
