from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a JSON object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_fingerprint(value: Dict[str, Any], length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def require_keys(value: Dict[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"missing keys in {where}: {missing}")


def resolve_path(path: str | Path, *, base: str | Path | None = None) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute() and base is not None:
        result = Path(base) / result
    return result.resolve()


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    checkpoints: Path
    per_seed: Path
    aggregate: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ExperimentPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            per_seed=root / "per_seed",
            aggregate=root / "aggregate",
        )

    def create(self) -> None:
        for path in (self.root, self.checkpoints, self.per_seed, self.aggregate):
            path.mkdir(parents=True, exist_ok=True)


def validate_config(config: Dict[str, Any]) -> None:
    require_keys(config, ["experiment_name", "seeds", "data", "model", "simclr", "supervised", "rd"], where="config")
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("seeds must be a nonempty list")
    if len(set(int(seed) for seed in config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must not contain duplicates")
    require_keys(config["data"], ["root", "eval_per_fine_class", "eval_subset_seed", "num_workers"], where="data")
    require_keys(config["model"], ["projection_dim", "projection_hidden_dim"], where="model")
    require_keys(config["simclr"], ["epochs", "batch_size", "lr", "temperature"], where="simclr")
    require_keys(config["supervised"], ["epochs", "batch_size", "lr"], where="supervised")
    require_keys(config["rd"], ["methods", "beta_min", "beta_max", "beta_points", "sinkhorn_tol", "sinkhorn_max_iter"], where="rd")
    methods = set(config["rd"]["methods"])
    allowed = {"simclr_z", "simclr_h", "pixel", "random_z", "supervised_h"}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"unknown RD methods: {sorted(unknown)}")
    plot_methods = set(config["rd"].get("plot_methods", config["rd"]["methods"]))
    if not plot_methods or not plot_methods.issubset(methods):
        raise ValueError("plot_methods must be a nonempty subset of methods")
    if int(config["data"]["eval_per_fine_class"]) < 1:
        raise ValueError("eval_per_fine_class must be positive")
    if int(config["rd"]["beta_points"]) < 2:
        raise ValueError("beta_points must be at least two")
    if float(config["rd"]["beta_min"]) <= 0 or float(config["rd"]["beta_max"]) <= float(config["rd"]["beta_min"]):
        raise ValueError("beta range is invalid")
