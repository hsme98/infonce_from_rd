#!/usr/bin/env python3
"""Check that sequential experiment outputs exist and satisfy basic invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/reproduced")
    parser.add_argument("--skip-cifar", action="store_true")
    parser.add_argument("--skip-tabular", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    if not args.skip_tabular:
        generic = root / "generic_channel"
        for name in (
            "metrics.json",
            "diagnostic_arrays.npz",
            "generic_channel_theorem_experiment.pdf",
        ):
            require(generic / name)
        with (generic / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        exact_error = float(metrics["diagnostics"]["kl_gap"]["exact_max_abs_error"])
        if exact_error > 1e-10:
            raise SystemExit(f"KL-gap identity failed: {exact_error:.3e}")
        fixed_tv = max(
            float(row["fixed_row_tv_from_lambda0"])
            for row in metrics["diagnostics"]["gauge_stress"]
        )
        if fixed_tv > 1e-9:
            raise SystemExit(f"fixed-output gauge invariance failed: {fixed_tv:.3e}")

        markov = root / "markov"
        for name in ("metrics.json", "multiseed_summary.csv", "markov_full.pdf"):
            require(markov / name)

    if not args.skip_cifar:
        aggregate = root / "cifar" / "aggregate"
        for name in (
            "aggregate_summary.json",
            "aggregate_curves.csv",
            "aggregate_thresholds.csv",
            "feature_diagnostics.csv",
            "cifar100_semantic_rd.pdf",
        ):
            require(aggregate / name)

    print(f"Validation passed: {root.resolve()}")


if __name__ == "__main__":
    main()
