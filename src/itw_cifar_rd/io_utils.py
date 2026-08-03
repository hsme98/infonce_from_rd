from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def atomic_json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if set(row) != set(fieldnames):
                raise ValueError("all CSV rows must have the same fields")
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def device_from_arg(raw: str) -> torch.device:
    raw = raw.lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    # Explicit weights_only=False is needed because checkpoints include optimizer
    # state and metadata, not only tensors.
    return torch.load(Path(path), map_location=map_location, weights_only=False)
