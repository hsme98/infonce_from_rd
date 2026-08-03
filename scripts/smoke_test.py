#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1] / "src"))

import argparse
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from itw_cifar_rd.config import load_json
from itw_cifar_rd.models import ModelSpec, build_model
from itw_cifar_rd.rd_experiment import run_semantic_rd


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline synthetic smoke test.")
    parser.add_argument("--config", default="configs/quick/cifar.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    config["rd"]["methods"] = ["simclr_z", "pixel", "random_z", "supervised_h"]
    config["rd"]["beta_points"] = 5
    config["rd"]["beta_max"] = 12.0
    config["rd"]["adaptive_beta_extensions"] = 0
    config["rd"]["sinkhorn_max_iter"] = 250
    config["rd"]["cost_scale_sample_pairs"] = 2000

    n = 40
    fine = torch.arange(n) % 10
    coarse = fine // 2
    generator = torch.Generator().manual_seed(0)
    fine_centers = F.normalize(torch.randn(10, 16, generator=generator), dim=1)
    semantic = F.normalize(fine_centers[fine] + 0.08 * torch.randn(n, 16, generator=generator), dim=1)
    payload = {
        "seed": 0,
        "indices": torch.arange(n),
        "fine": fine,
        "coarse": coarse,
        "features": {
            "simclr_z": semantic,
            "pixel": torch.randn(n, 64, generator=generator),
            "random_z": F.normalize(torch.randn(n, 16, generator=generator), dim=1),
            "supervised_h": F.normalize(fine_centers[fine] + 0.02 * torch.randn(n, 16, generator=generator), dim=1),
        },
        "diagnostics": {
            name: {
                "fine_leave_one_out_centroid_accuracy": 0.0,
                "coarse_leave_one_out_centroid_accuracy": 0.0,
            }
            for name in ("simclr_z", "pixel", "random_z", "supervised_h")
        },
    }
    if args.output_dir is None:
        temp = tempfile.TemporaryDirectory()
        output = Path(temp.name)
    else:
        temp = None
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "features.pt"
    torch.save(payload, feature_path)
    result = run_semantic_rd(
        config,
        seed=0,
        feature_path=feature_path,
        output_dir=output,
        device=torch.device("cpu"),
    )
    required = [output / "rd_curves.csv", output / "thresholds.json", output / "semantic_rd_seed.pdf"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"smoke test failed: missing {path}")
    print(f"Smoke test passed: {result}")
    if temp is not None:
        temp.cleanup()


if __name__ == "__main__":
    main()
