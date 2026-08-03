from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itw_rd.experiment1 import make_figure

with (ROOT / "results" / "paper" / "generic_channel" / "metrics.json").open(
    "r", encoding="utf-8"
) as handle:
    payload = json.load(handle)
make_figure(payload, ROOT / "assets")
