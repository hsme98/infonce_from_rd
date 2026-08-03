#!/usr/bin/env python3
"""Rebuild the README/paper figures from the bundled numerical summaries."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for script in (
    ROOT / "scripts" / "figures" / "make_generic_channel_figure.py",
    ROOT / "scripts" / "figures" / "make_markov_figure.py",
    ROOT / "scripts" / "figures" / "make_cifar_figure.py",
):
    print(f"Running {script.relative_to(ROOT)}")
    runpy.run_path(str(script), run_name="__main__")
print(f"Figures rebuilt under {(ROOT / 'assets').resolve()}")
