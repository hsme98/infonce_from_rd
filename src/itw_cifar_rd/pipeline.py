"""Sequential CIFAR-100 training and semantic rate-distortion pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from .aggregate import aggregate_results
from .config import ExperimentPaths, config_fingerprint, validate_config
from .data import CIFAR100WithCoarse
from .features import extract_all_features
from .io_utils import atomic_json_dump, device_from_arg, torch_load
from .rd_experiment import run_semantic_rd
from .training import train_one_role


def prepare_cifar100(data_root: str | Path) -> Dict[str, Any]:
    """Download CIFAR-100 once and validate both official splits."""
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR100WithCoarse(root, train=True, transform=None, download=True)
    test = CIFAR100WithCoarse(root, train=False, transform=None, download=True)
    summary = {
        "data_root": str(root),
        "train_examples": len(train),
        "test_examples": len(test),
        "fine_classes": len(train.classes),
        "coarse_classes": len(train.coarse_classes),
    }
    if summary["train_examples"] != 50_000 or summary["test_examples"] != 10_000:
        raise RuntimeError(f"unexpected CIFAR-100 split sizes: {summary}")
    if summary["fine_classes"] != 100 or summary["coarse_classes"] != 20:
        raise RuntimeError(f"unexpected CIFAR-100 class counts: {summary}")
    return summary


def _completed_rd_output(seed_dir: Path, fingerprint: str) -> bool:
    required = [
        seed_dir / "rd_curves.csv",
        seed_dir / "thresholds.json",
        seed_dir / "diagnostics.json",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    with (seed_dir / "diagnostics.json").open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if diagnostics.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            f"existing RD output has a different configuration: {seed_dir}. "
            "Use a new output directory or pass overwrite=True."
        )
    return True


def _valid_feature_file(path: Path, *, seed: int, fingerprint: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    payload = torch_load(path, map_location="cpu")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"feature seed mismatch in {path}")
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            f"existing features have a different configuration: {path}. "
            "Use a new output directory or pass overwrite=True."
        )
    return True


def run_sequential_cifar(
    config: Dict[str, Any],
    *,
    output_root: str | Path,
    data_root: str | Path | None = None,
    device: str | torch.device = "auto",
    seeds: Iterable[int] | None = None,
    download: bool = True,
    deterministic: bool = False,
    overwrite: bool = False,
    keep_features: bool = True,
) -> Dict[str, Any]:
    """Run every CIFAR stage sequentially in a single process.

    The order is dataset preparation, then for each seed: SimCLR training,
    supervised reference training, feature extraction, semantic RD, and finally
    multiseed aggregation. Existing validated checkpoints and outputs are reused.
    """
    validate_config(config)
    config = dict(config)
    selected_seeds = [int(seed) for seed in (seeds if seeds is not None else config["seeds"])]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")
    config["seeds"] = selected_seeds

    root = Path(output_root).expanduser().resolve()
    if overwrite and root.exists():
        shutil.rmtree(root)
    paths = ExperimentPaths.from_root(root)
    paths.create()

    configured_data_root = config["data"].get("root", "data/cifar100")
    data_path = Path(data_root or configured_data_root).expanduser().resolve()
    dataset_summary = prepare_cifar100(data_path) if download else {"data_root": str(data_path)}

    torch_device = device if isinstance(device, torch.device) else device_from_arg(device)
    fingerprint = config_fingerprint(config)
    per_seed_summaries: list[dict[str, Any]] = []

    for index, seed in enumerate(selected_seeds, start=1):
        print(f"\n[CIFAR seed {seed}] {index}/{len(selected_seeds)}", flush=True)
        simclr_dir = paths.checkpoints / f"simclr_seed_{seed:03d}"
        supervised_dir = paths.checkpoints / f"supervised_seed_{seed:03d}"
        seed_dir = paths.per_seed / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        simclr_summary = train_one_role(
            config,
            role="simclr",
            seed=seed,
            data_root=data_path,
            output_dir=simclr_dir,
            device=torch_device,
            download=False,
            deterministic=deterministic,
        )
        supervised_summary = train_one_role(
            config,
            role="supervised",
            seed=seed,
            data_root=data_path,
            output_dir=supervised_dir,
            device=torch_device,
            download=False,
            deterministic=deterministic,
        )

        feature_path = seed_dir / "features.pt"
        if _completed_rd_output(seed_dir, fingerprint):
            rd_summary: Dict[str, Any] = {
                "status": "skipped",
                "reason": "validated RD outputs already exist",
            }
            feature_summary: Dict[str, Any] = {
                "status": "reused" if feature_path.exists() else "not needed"
            }
        else:
            if _valid_feature_file(feature_path, seed=seed, fingerprint=fingerprint):
                feature_summary = {"status": "reused", "feature_path": str(feature_path)}
            else:
                feature_summary = extract_all_features(
                    config,
                    seed=seed,
                    data_root=data_path,
                    simclr_checkpoint=simclr_dir / "checkpoint_final.pt",
                    supervised_checkpoint=supervised_dir / "checkpoint_final.pt",
                    output_path=feature_path,
                    device=torch_device,
                )
            rd_summary = run_semantic_rd(
                config,
                seed=seed,
                feature_path=feature_path,
                output_dir=seed_dir,
                device=torch_device,
            )

        seed_summary = {
            "seed": seed,
            "simclr": simclr_summary,
            "supervised": supervised_summary,
            "features": feature_summary,
            "rd": rd_summary,
        }
        atomic_json_dump(seed_summary, seed_dir / "pipeline_summary.json")
        per_seed_summaries.append(seed_summary)
        if not keep_features and feature_path.exists():
            feature_path.unlink()

    aggregate_summary = aggregate_results(
        config,
        per_seed_root=paths.per_seed,
        output_dir=paths.aggregate,
    )
    summary = {
        "status": "complete",
        "config_fingerprint": fingerprint,
        "device": str(torch_device),
        "dataset": dataset_summary,
        "output_root": str(root),
        "seeds": selected_seeds,
        "per_seed": per_seed_summaries,
        "aggregate": aggregate_summary,
    }
    atomic_json_dump(summary, root / "pipeline_summary.json")
    return summary
