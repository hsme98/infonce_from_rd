#!/usr/bin/env python3
"""Download and validate CIFAR-100 before sequential training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_cifar_rd.pipeline import prepare_cifar100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cifar100")
    args = parser.parse_args()
    summary = prepare_cifar100(args.data_root)
    print(summary)


if __name__ == "__main__":
    main()
