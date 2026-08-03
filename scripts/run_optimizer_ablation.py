#!/usr/bin/env python3
"""Run the fixed-pair and fixed-step optimizer-floor studies sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_rd.config import load_json
from itw_rd.optimization_ablation import OptimizationAblationConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/optimizer_ablation.json")
    parser.add_argument("--output-root", default="results/reproduced/optimizer_ablation")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    suite = load_json(args.config)
    experiments = suite.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("optimizer suite must contain a nonempty 'experiments' object")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"root": str(root.resolve()), "experiments": {}}

    for index, (name, raw) in enumerate(experiments.items(), start=1):
        print(f"[{index}/{len(experiments)}] Optimizer study: {name}", flush=True)
        config = OptimizationAblationConfig.from_dict(raw)
        config.outdir = str(root / name)
        if args.seeds:
            config.seeds = tuple(args.seeds)
        result = run(config)
        manifest["experiments"][name] = {
            "outdir": str(Path(config.outdir).resolve()),
            "checkpoint_unit": config.checkpoint_unit,
            "records": len(result["records"]),
            "seeds": list(config.seeds),
        }

    with (root / "suite_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Optimizer ablation complete: {root.resolve()}")


if __name__ == "__main__":
    main()
