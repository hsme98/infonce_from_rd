#!/usr/bin/env python3
"""Run CIFAR-100 training, semantic RD, and aggregation sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_cifar_rd.config import load_json
from itw_cifar_rd.pipeline import run_sequential_cifar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/cifar.json")
    parser.add_argument("--output-root", default="results/reproduced/cifar")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--discard-features", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    summary = run_sequential_cifar(
        config,
        output_root=args.output_root,
        data_root=args.data_root,
        device=args.device,
        seeds=args.seeds,
        download=not args.no_download,
        deterministic=args.deterministic,
        overwrite=args.overwrite,
        keep_features=not args.discard_features,
    )
    print(f"CIFAR experiment complete: {summary['output_root']}")
    print(f"Aggregate figure: {Path(summary['output_root']) / 'aggregate' / 'cifar100_semantic_rd.pdf'}")


if __name__ == "__main__":
    main()
