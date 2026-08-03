from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import config_fingerprint
from .io_utils import atomic_json_dump, read_csv, write_csv
from .rd_experiment import METHOD_LABELS
from .rd_metrics import rate_at_retention


def _load_seed_rows(seed_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(seed_dir / "rd_curves.csv")
    converted: list[dict[str, Any]] = []
    numeric = {
        "beta",
        "cost_scale",
        "sinkhorn_iterations",
        "marginal_error",
        "rate",
        "distortion",
        "coarse_mi",
        "fine_mi",
        "fine_increment_mi",
        "coarse_retention",
        "fine_retention",
        "fine_increment_retention",
        "coarse_entropy",
        "fine_entropy",
    }
    for row in rows:
        result: dict[str, Any] = dict(row)
        result["seed"] = int(row["seed"])
        for key in numeric:
            result[key] = float(row[key])
        converted.append(result)
    return converted


def _interpolate_curve(rows: list[dict[str, Any]], method: str, metric: str, grid: np.ndarray) -> np.ndarray:
    subset = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: row["rate"],
    )
    rates = np.asarray([row["rate"] for row in subset], dtype=np.float64)
    values = np.maximum.accumulate(
        np.asarray([row[metric] for row in subset], dtype=np.float64)
    )
    return np.interp(grid, rates, values, left=values[0], right=values[-1])


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.median(values, axis=0),
        np.quantile(values, 0.1, axis=0),
        np.quantile(values, 0.9, axis=0),
    )


def aggregate_results(
    config: Dict[str, Any],
    *,
    per_seed_root: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    methods = [str(method) for method in config["rd"]["methods"]]
    plot_methods = [str(method) for method in config["rd"].get("plot_methods", methods)]
    per_seed_root = Path(per_seed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_seed: Dict[int, list[dict[str, Any]]] = {}
    max_rates: list[float] = []
    diagnostics: Dict[int, Any] = {}
    for seed in seeds:
        seed_dir = per_seed_root / f"seed_{seed:03d}"
        rows = _load_seed_rows(seed_dir)
        by_seed[seed] = rows
        for method in methods:
            max_rates.append(max(row["rate"] for row in rows if row["method"] == method))
        with (seed_dir / "diagnostics.json").open("r", encoding="utf-8") as handle:
            diagnostics[seed] = json.load(handle)
    common_max_rate = float(min(max_rates))
    grid = np.linspace(0.0, common_max_rate, int(config["rd"].get("rate_grid_points", 240)))

    aggregate_rows: list[dict[str, Any]] = []
    metrics = ("coarse_retention", "fine_retention", "fine_increment_retention")
    curves: Dict[str, Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for method in methods:
        curves[method] = {}
        for metric in metrics:
            matrix = np.stack(
                [_interpolate_curve(by_seed[seed], method, metric, grid) for seed in seeds],
                axis=0,
            )
            median, q10, q90 = _quantiles(matrix)
            curves[method][metric] = (median, q10, q90)
            for index, rate in enumerate(grid):
                aggregate_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "metric": metric,
                        "rate": float(rate),
                        "median": float(median[index]),
                        "q10": float(q10[index]),
                        "q90": float(q90[index]),
                    }
                )
    write_csv(aggregate_rows, output_dir / "aggregate_curves.csv")

    threshold_rows: list[dict[str, Any]] = []
    threshold_targets = [float(x) for x in config["rd"].get("retention_thresholds", [0.5, 0.8, 0.9])]
    for method in methods:
        for metric in metrics:
            for target in threshold_targets:
                values = []
                for seed in seeds:
                    subset = sorted(
                        [row for row in by_seed[seed] if row["method"] == method],
                        key=lambda row: row["rate"],
                    )
                    values.append(
                        rate_at_retention(
                            [row["rate"] for row in subset],
                            [row[metric] for row in subset],
                            target,
                        )
                    )
                array = np.asarray(values, dtype=np.float64)
                finite = array[np.isfinite(array)]
                threshold_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "metric": metric,
                        "target": target,
                        "n_reached": int(finite.size),
                        "median_rate": float(np.median(finite)) if finite.size else float("nan"),
                        "q10_rate": float(np.quantile(finite, 0.1)) if finite.size else float("nan"),
                        "q90_rate": float(np.quantile(finite, 0.9)) if finite.size else float("nan"),
                    }
                )
    write_csv(threshold_rows, output_dir / "aggregate_thresholds.csv")

    # Aggregate leave-one-out centroid diagnostics.  These are sanity checks,
    # not the headline experimental endpoint.
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_aliases = {
        "fine_leave_one_out_centroid_accuracy": "fine_centroid_accuracy",
        "coarse_leave_one_out_centroid_accuracy": "coarse_centroid_accuracy",
    }
    for method in methods:
        metric_names = sorted(
            set().union(
                *(diagnostics[seed]["feature_diagnostics"][method].keys() for seed in seeds)
            )
        )
        for source_name in metric_names:
            values = [
                diagnostics[seed]["feature_diagnostics"][method].get(source_name)
                for seed in seeds
            ]
            finite_values = [
                float(value)
                for value in values
                if value is not None and np.isfinite(float(value))
            ]
            if not finite_values:
                continue
            array = np.asarray(finite_values, dtype=np.float64)
            diagnostic_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "metric": diagnostic_aliases.get(source_name, source_name),
                    "n": int(array.size),
                    "median": float(np.median(array)),
                    "q10": float(np.quantile(array, 0.1)),
                    "q90": float(np.quantile(array, 0.9)),
                }
            )
    write_csv(diagnostic_rows, output_dir / "feature_diagnostics.csv")

    _plot_aggregate(
        grid,
        plot_methods,
        curves,
        threshold_rows,
        target=float(config["rd"].get("headline_retention", 0.8)),
        output=output_dir / "cifar100_semantic_rd",
    )

    summary = {
        "config_fingerprint": config_fingerprint(config),
        "seeds": seeds,
        "methods": methods,
        "plot_methods": plot_methods,
        "common_max_rate": common_max_rate,
        "headline_retention": float(config["rd"].get("headline_retention", 0.8)),
        "thresholds": threshold_rows,
        "diagnostics": diagnostic_rows,
        "interpretation": {
            "primary_method": "simclr_z",
            "primary_claim": "semantic information retained at equal RD rate",
            "pass_criteria": [
                "simclr_z beats pixel and random_z for coarse and fine retention over a broad rate range",
                "the pattern is stable across independently trained seeds",
                "simclr_h is reported separately rather than substituted for the theorem-facing critic space z",
            ],
            "qualification": (
                "CIFAR-100 supports semantic rate efficiency, but the coarse and fine curves "
                "do not establish a sharp coarse-to-fine phase hierarchy."
            ),
        },
    }
    atomic_json_dump(summary, output_dir / "aggregate_summary.json")
    return summary


def _plot_aggregate(
    grid: np.ndarray,
    methods: list[str],
    curves: Dict[str, Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    threshold_rows: list[dict[str, Any]],
    *,
    target: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55))
    for method in methods:
        for axis, metric, title in (
            (axes[0], "coarse_retention", "coarse labels"),
            (axes[1], "fine_retention", "fine labels"),
        ):
            median, q10, q90 = curves[method][metric]
            line = axis.plot(
                grid,
                median,
                linewidth=1.7,
                label=METHOD_LABELS.get(method, method),
            )[0]
            axis.fill_between(grid, q10, q90, alpha=0.16, color=line.get_color())
            axis.set_title(title)
            axis.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("normalized semantic information retained")

    x = np.arange(len(methods), dtype=np.float64)
    width = 0.36
    for offset, metric, label in (
        (-width / 2, "coarse_retention", "coarse"),
        (width / 2, "fine_retention", "fine"),
    ):
        selected = []
        errors_low = []
        errors_high = []
        for method in methods:
            row = next(
                row
                for row in threshold_rows
                if row["method"] == method
                and row["metric"] == metric
                and math.isclose(float(row["target"]), target)
            )
            median = float(row["median_rate"])
            selected.append(median)
            errors_low.append(max(0.0, median - float(row["q10_rate"])))
            errors_high.append(max(0.0, float(row["q90_rate"]) - median))
        axes[2].bar(
            x + offset,
            selected,
            width=width,
            yerr=np.vstack([errors_low, errors_high]),
            capsize=2,
            label=label,
        )
    axes[2].set_title(f"rate for {int(round(100 * target))}% retention")
    axes[2].set_ylabel("required rate [nats]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        [METHOD_LABELS.get(method, method) for method in methods],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(fontsize=7)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
