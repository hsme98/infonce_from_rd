from __future__ import annotations

from pathlib import Path

from itw_cifar_rd.config import load_json, validate_config


def test_all_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (root / "configs").glob("*/cifar.json"):
        validate_config(load_json(path))
