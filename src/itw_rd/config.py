from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a JSON object")
    return value
