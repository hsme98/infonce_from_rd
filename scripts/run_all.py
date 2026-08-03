#!/usr/bin/env python3
"""Run all paper experiments sequentially without a scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_rd.config import load_json
from itw_rd.experiment1 import Experiment1Config, run as run_generic
from itw_rd.experiment2 import Experiment2Config, run as run_markov
from itw_rd.optimization_ablation import OptimizationAblationConfig, run as run_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "pilot", "paper"], default="paper")
    parser.add_argument("--output-root", default="results/reproduced")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-cifar", action="store_true")
    parser.add_argument("--include-optimizer-ablation", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite-cifar", action="store_true")
    args = parser.parse_args()

    profile_dir = REPO_ROOT / "configs" / args.profile
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    optimizer_experiments = {}
    if args.include_optimizer_ablation:
        optimizer_suite = load_json(profile_dir / "optimizer_ablation.json")
        optimizer_experiments = optimizer_suite.get("experiments", {})
        if not isinstance(optimizer_experiments, dict) or not optimizer_experiments:
            raise ValueError("optimizer-ablation profile has no experiments")

    stage = 0
    total = (0 if args.skip_tabular else 2) + (0 if args.skip_cifar else 1)
    total += len(optimizer_experiments)

    if not args.skip_tabular:
        generic = Experiment1Config.from_dict(load_json(profile_dir / "generic_channel.json"))
        generic.outdir = str(output_root / "generic_channel")
        stage += 1
        print(f"\n[{stage}/{total}] Generic positive-pair channels", flush=True)
        run_generic(generic)

        markov = Experiment2Config.from_dict(load_json(profile_dir / "markov.json"))
        markov.outdir = str(output_root / "markov")
        stage += 1
        print(f"\n[{stage}/{total}] Hierarchical Markov coarse-graining", flush=True)
        run_markov(markov)

    if not args.skip_cifar:
        from itw_cifar_rd.config import load_json as load_cifar_json
        from itw_cifar_rd.pipeline import run_sequential_cifar

        stage += 1
        print(f"\n[{stage}/{total}] CIFAR-100 semantic rate-distortion", flush=True)
        cifar = load_cifar_json(profile_dir / "cifar.json")
        run_sequential_cifar(
            cifar,
            output_root=output_root / "cifar",
            data_root=args.data_root,
            device=args.device,
            download=not args.no_download,
            deterministic=args.deterministic,
            overwrite=args.overwrite_cifar,
        )

    if args.include_optimizer_ablation:
        for name, raw in optimizer_experiments.items():
            stage += 1
            print(f"\n[{stage}/{total}] Optimizer ablation: {name}", flush=True)
            config = OptimizationAblationConfig.from_dict(raw)
            config.outdir = str(output_root / "optimizer_ablation" / name)
            run_ablation(config)

    print(f"\nAll requested experiments complete: {output_root.resolve()}")


if __name__ == "__main__":
    main()
