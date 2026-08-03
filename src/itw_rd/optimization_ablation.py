"""Ablations for the flat late-stage InfoNCE channel-recovery curves.

The study independently changes three mechanisms that can create or reduce a
constant-learning-rate noise floor:

* learning-rate decay;
* Polyak-Ruppert tail averaging;
* larger effective batches through gradient accumulation.

The runner supports either a fixed positive-pair budget or a fixed optimizer-
step budget.  Running both distinguishes gradient-noise reduction from the
loss of update count caused by increasing the batch at fixed data budget.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import channel_from_joint, kl_divergence, sinkhorn_log, weighted_row_kl
from .generators import make_generic_channel
from .infonce import (
    critic_mse_modulo_y_gauge,
    learning_rate_at_step,
    train_tabular_infonce_checkpoints,
)

Array = np.ndarray


@dataclass
class OptimizerVariant:
    """One optimizer intervention in the plateau ablation."""

    name: str
    label: Optional[str] = None
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.04
    learning_rate_schedule: str = "constant"
    min_learning_rate: float = 0.0
    warmup_fraction: float = 0.0
    lr_milestone_fractions: tuple[float, ...] = (0.2, 0.5, 0.8)
    lr_decay_gamma: float = 0.25
    tail_average_fraction: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerVariant":
        data = dict(data)
        if "lr_milestone_fractions" in data:
            data["lr_milestone_fractions"] = tuple(data["lr_milestone_fractions"])
        return cls(**data)

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size) * int(self.gradient_accumulation_steps)

    @property
    def display_label(self) -> str:
        return self.label or self.name


@dataclass
class OptimizationAblationConfig:
    n: int = 24
    matched_modes: tuple[str, ...] = ("matched", "unmatched")
    k_values: tuple[int, ...] = (1, 4, 32)
    seeds: tuple[int, ...] = (0, 1, 2)
    checkpoint_unit: str = "positive_pairs"
    checkpoints: tuple[int, ...] = (
        25_600,
        51_200,
        92_160,
        153_600,
        256_000,
        384_000,
        563_200,
        742_400,
        921_600,
    )
    variants: tuple[OptimizerVariant, ...] = field(default_factory=tuple)
    sinkhorn_tol: float = 1e-11
    outdir: str = "results/optimizer_ablation"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationAblationConfig":
        data = dict(data)
        for key in ("matched_modes", "k_values", "seeds", "checkpoints"):
            if key in data:
                data[key] = tuple(data[key])
        if "variants" in data:
            data["variants"] = tuple(
                OptimizerVariant.from_dict(item) for item in data["variants"]
            )
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.n <= 1:
            raise ValueError("n must exceed one")
        if self.checkpoint_unit not in {"positive_pairs", "steps"}:
            raise ValueError("checkpoint_unit must be 'positive_pairs' or 'steps'")
        if not self.checkpoints or any(int(x) <= 0 for x in self.checkpoints):
            raise ValueError("checkpoints must be positive")
        if not self.variants:
            raise ValueError("at least one optimizer variant is required")
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("optimizer variant names must be unique")
        for mode in self.matched_modes:
            if mode not in {"matched", "unmatched"}:
                raise ValueError(f"unknown marginal mode {mode!r}")
        for variant in self.variants:
            if variant.batch_size <= 0 or variant.gradient_accumulation_steps <= 0:
                raise ValueError("batch sizes must be positive")
            if not 0 <= variant.warmup_fraction < 1:
                raise ValueError("warmup_fraction must lie in [0, 1)")
            if variant.tail_average_fraction is not None and not (
                0 <= variant.tail_average_fraction < 1
            ):
                raise ValueError("tail_average_fraction must lie in [0, 1)")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_score(channel: Array, p_y: Array) -> Array:
    return np.log(channel) - np.log(p_y)[None, :]


def _checkpoint_plan(
    config: OptimizationAblationConfig,
    variant: OptimizerVariant,
) -> tuple[tuple[int, ...], Dict[int, int]]:
    """Map requested budgets to optimizer steps for one effective batch size."""
    if config.checkpoint_unit == "steps":
        requested_to_step = {int(value): int(value) for value in config.checkpoints}
    else:
        effective_batch = variant.effective_batch_size
        requested_to_step = {
            int(value): max(1, int(np.ceil(int(value) / effective_batch)))
            for value in config.checkpoints
        }
    return tuple(sorted(set(requested_to_step.values()))), requested_to_step


def _aggregate(
    records: Iterable[Dict[str, Any]],
    *,
    variant: str,
    mode: str,
    K: int,
    checkpoint: int,
    metric: str,
) -> tuple[float, float, float]:
    values = np.asarray(
        [
            row[metric]
            for row in records
            if row["variant"] == variant
            and row["mode"] == mode
            and row["K"] == K
            and row["requested_checkpoint"] == checkpoint
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.1)),
        float(np.quantile(values, 0.9)),
    )


def _summary_rows(
    config: OptimizationAblationConfig,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = (
        "channel_row_kl",
        "channel_row_kl_last",
        "critic_mse_mod_gauge",
        "critic_mse_mod_gauge_last",
    )
    for variant in config.variants:
        for mode in config.matched_modes:
            for K in config.k_values:
                for checkpoint in config.checkpoints:
                    row: Dict[str, Any] = {
                        "variant": variant.name,
                        "label": variant.display_label,
                        "mode": mode,
                        "K": K,
                        "requested_checkpoint": checkpoint,
                        "checkpoint_unit": config.checkpoint_unit,
                        "effective_batch_size": variant.effective_batch_size,
                    }
                    for metric in metrics:
                        median, q10, q90 = _aggregate(
                            records,
                            variant=variant.name,
                            mode=mode,
                            K=K,
                            checkpoint=checkpoint,
                            metric=metric,
                        )
                        row[f"{metric}_median"] = median
                        row[f"{metric}_q10"] = q10
                        row[f"{metric}_q90"] = q90
                    rows.append(row)
    return rows


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(config: OptimizationAblationConfig) -> Dict[str, Any]:
    config.validate()
    config.checkpoints = tuple(sorted({int(value) for value in config.checkpoints}))
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for mode_index, mode in enumerate(config.matched_modes):
        matched = mode == "matched"
        for seed in config.seeds:
            problem = make_generic_channel(
                n=config.n,
                seed=10_000 * mode_index + seed,
                matched=matched,
                sinkhorn_tol=config.sinkhorn_tol,
            )
            target = _canonical_score(problem.channel, problem.p_y)

            for K in config.k_values:
                # Identical seeds ensure exactly paired minibatches for variants
                # with the same effective batching scheme.
                trajectory_seed = 1_000_000 * mode_index + 10_000 * seed + 100 * K + 17

                for variant in config.variants:
                    checkpoint_steps, requested_to_step = _checkpoint_plan(config, variant)
                    max_steps = max(checkpoint_steps)
                    warmup_steps = int(round(variant.warmup_fraction * max_steps))
                    warmup_steps = min(max(warmup_steps, 0), max_steps - 1)

                    snapshots = train_tabular_infonce_checkpoints(
                        problem.p_x,
                        problem.channel,
                        K=K,
                        checkpoint_steps=checkpoint_steps,
                        batch_size=variant.batch_size,
                        gradient_accumulation_steps=variant.gradient_accumulation_steps,
                        learning_rate=variant.learning_rate,
                        learning_rate_schedule=variant.learning_rate_schedule,
                        min_learning_rate=variant.min_learning_rate,
                        warmup_steps=warmup_steps,
                        lr_milestone_fractions=variant.lr_milestone_fractions,
                        lr_decay_gamma=variant.lr_decay_gamma,
                        tail_average_fraction=variant.tail_average_fraction,
                        seed=trajectory_seed,
                    )

                    for requested in config.checkpoints:
                        step = requested_to_step[int(requested)]
                        result = snapshots[step]

                        selected_joint = sinkhorn_log(
                            result.score,
                            problem.p_x,
                            problem.p_y,
                            tol=config.sinkhorn_tol,
                        ).joint
                        selected_channel = channel_from_joint(selected_joint, problem.p_x)

                        last_joint = sinkhorn_log(
                            result.last_score,
                            problem.p_x,
                            problem.p_y,
                            tol=config.sinkhorn_tol,
                        ).joint
                        last_channel = channel_from_joint(last_joint, problem.p_x)

                        selected_kl = weighted_row_kl(
                            problem.p_x,
                            problem.channel,
                            selected_channel,
                        )
                        last_kl = weighted_row_kl(
                            problem.p_x,
                            problem.channel,
                            last_channel,
                        )

                        records.append(
                            {
                                "mode": mode,
                                "seed": seed,
                                "K": K,
                                "variant": variant.name,
                                "variant_label": variant.display_label,
                                "requested_checkpoint": int(requested),
                                "checkpoint_unit": config.checkpoint_unit,
                                "steps": int(step),
                                "positive_pairs": int(result.positive_pairs_seen),
                                "batch_size": int(variant.batch_size),
                                "gradient_accumulation_steps": int(
                                    variant.gradient_accumulation_steps
                                ),
                                "effective_batch_size": int(
                                    result.effective_batch_size
                                ),
                                "learning_rate_schedule": variant.learning_rate_schedule,
                                "base_learning_rate": float(variant.learning_rate),
                                "min_learning_rate": float(
                                    variant.min_learning_rate
                                ),
                                "learning_rate_at_checkpoint": float(
                                    result.learning_rates[-1]
                                ),
                                "tail_average_fraction": variant.tail_average_fraction,
                                "averaging_start_step": result.averaging_start_step,
                                "averaged_iterates": int(result.averaged_iterates),
                                "channel_row_kl": selected_kl,
                                "channel_row_kl_last": last_kl,
                                "tail_average_gain": last_kl - selected_kl,
                                "joint_kl": kl_divergence(
                                    problem.joint,
                                    selected_joint,
                                ),
                                "joint_kl_last": kl_divergence(
                                    problem.joint,
                                    last_joint,
                                ),
                                "critic_mse_mod_gauge": critic_mse_modulo_y_gauge(
                                    result.score,
                                    target,
                                    problem.p_x,
                                    problem.p_y,
                                ),
                                "critic_mse_mod_gauge_last": critic_mse_modulo_y_gauge(
                                    result.last_score,
                                    target,
                                    problem.p_x,
                                    problem.p_y,
                                ),
                                "last_recorded_loss": float(result.losses[-1]),
                                "trajectory_seed": trajectory_seed,
                            }
                        )

    summary_rows = _summary_rows(config, records)
    payload = {
        "config": {
            **asdict(config),
            "variants": [asdict(variant) for variant in config.variants],
        },
        "records": records,
        "summary": summary_rows,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    _write_summary_csv(summary_rows, outdir / "summary.csv")
    make_curve_figure(payload, outdir)
    make_final_figure(payload, outdir)
    make_schedule_figure(config, outdir)
    return payload


def _x_value(
    config: OptimizationAblationConfig,
    checkpoint: int,
) -> int:
    return int(checkpoint)


def make_curve_figure(payload: Dict[str, Any], outdir: Path) -> None:
    config = OptimizationAblationConfig.from_dict(payload["config"])
    records = payload["records"]
    rows = len(config.matched_modes)
    cols = len(config.k_values)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.25 * cols, 2.65 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, mode in enumerate(config.matched_modes):
        for col_index, K in enumerate(config.k_values):
            ax = axes[row_index, col_index]
            for variant in config.variants:
                medians, lows, highs = [], [], []
                for checkpoint in config.checkpoints:
                    median, q10, q90 = _aggregate(
                        records,
                        variant=variant.name,
                        mode=mode,
                        K=K,
                        checkpoint=checkpoint,
                        metric="channel_row_kl",
                    )
                    medians.append(median)
                    lows.append(q10)
                    highs.append(q90)
                x = [_x_value(config, checkpoint) for checkpoint in config.checkpoints]
                line = ax.plot(
                    x,
                    medians,
                    linewidth=1.25,
                    marker="o",
                    markersize=2.4,
                    label=variant.display_label,
                )[0]
                ax.fill_between(
                    x,
                    lows,
                    highs,
                    alpha=0.08,
                    color=line.get_color(),
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{mode}, K={K}")
            if row_index == rows - 1:
                ax.set_xlabel(
                    "positive pairs processed"
                    if config.checkpoint_unit == "positive_pairs"
                    else "optimizer steps"
                )
            if col_index == 0:
                ax.set_ylabel("weighted row KL")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.savefig(outdir / "optimizer_ablation_curves.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "optimizer_ablation_curves.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_final_figure(payload: Dict[str, Any], outdir: Path) -> None:
    config = OptimizationAblationConfig.from_dict(payload["config"])
    records = payload["records"]
    final_checkpoint = max(config.checkpoints)
    K = max(config.k_values)
    x = np.arange(len(config.variants), dtype=float)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    ax = axes[0]
    for mode in config.matched_modes:
        medians, lows, highs = [], [], []
        for variant in config.variants:
            median, q10, q90 = _aggregate(
                records,
                variant=variant.name,
                mode=mode,
                K=K,
                checkpoint=final_checkpoint,
                metric="channel_row_kl",
            )
            medians.append(median)
            lows.append(median - q10)
            highs.append(q90 - median)
        ax.errorbar(
            x,
            medians,
            yerr=np.asarray([lows, highs]),
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=mode,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [variant.display_label for variant in config.variants], rotation=40, ha="right")
    ax.set_ylabel("final weighted row KL")
    ax.set_title(f"(a) final recovery, K={K}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    averaging_variants = [
        variant for variant in config.variants if variant.tail_average_fraction is not None
    ]
    if averaging_variants:
        avg_x = np.arange(len(averaging_variants), dtype=float)
        for mode in config.matched_modes:
            gains = []
            low = []
            high = []
            for variant in averaging_variants:
                values = np.asarray(
                    [
                        row["tail_average_gain"]
                        for row in records
                        if row["variant"] == variant.name
                        and row["mode"] == mode
                        and row["K"] == K
                        and row["requested_checkpoint"] == final_checkpoint
                    ],
                    dtype=np.float64,
                )
                median = float(np.median(values))
                gains.append(median)
                low.append(median - float(np.quantile(values, 0.1)))
                high.append(float(np.quantile(values, 0.9)) - median)
            ax.errorbar(
                avg_x,
                gains,
                yerr=np.asarray([low, high]),
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=mode,
            )
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(
            avg_x,
            [variant.display_label for variant in averaging_variants],
            rotation=40,
            ha="right",
        )
        ax.set_ylabel("last-iterate KL minus averaged KL")
        ax.set_title("(b) Polyak averaging gain")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No tail-averaging variants", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2]
    baseline = config.variants[0]
    ratios = []
    labels = []
    for variant in config.variants:
        variant_ratios = []
        for mode in config.matched_modes:
            for value_K in config.k_values:
                variant_median, _, _ = _aggregate(
                    records,
                    variant=variant.name,
                    mode=mode,
                    K=value_K,
                    checkpoint=final_checkpoint,
                    metric="channel_row_kl",
                )
                baseline_median, _, _ = _aggregate(
                    records,
                    variant=baseline.name,
                    mode=mode,
                    K=value_K,
                    checkpoint=final_checkpoint,
                    metric="channel_row_kl",
                )
                variant_ratios.append(variant_median / baseline_median)
        ratios.append(float(np.median(variant_ratios)))
        labels.append(variant.display_label)
    ax.plot(np.arange(len(ratios)), ratios, marker="o", linewidth=1.2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(ratios)), labels, rotation=40, ha="right")
    ax.set_ylabel("median final KL / baseline")
    ax.set_title("(c) aggregate plateau reduction")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(outdir / "optimizer_ablation_final.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "optimizer_ablation_final.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_schedule_figure(
    config: OptimizationAblationConfig,
    outdir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)
    total_steps = 1000
    step_grid = np.arange(1, total_steps + 1)
    seen = set()
    for variant in config.variants:
        signature = (
            variant.learning_rate,
            variant.learning_rate_schedule,
            variant.min_learning_rate,
            variant.warmup_fraction,
            variant.lr_milestone_fractions,
            variant.lr_decay_gamma,
        )
        if signature in seen:
            continue
        seen.add(signature)
        warmup_steps = min(
            int(round(variant.warmup_fraction * total_steps)),
            total_steps - 1,
        )
        rates = [
            learning_rate_at_step(
                int(step),
                total_steps,
                base_learning_rate=variant.learning_rate,
                schedule=variant.learning_rate_schedule,
                min_learning_rate=variant.min_learning_rate,
                warmup_steps=warmup_steps,
                milestone_fractions=variant.lr_milestone_fractions,
                decay_gamma=variant.lr_decay_gamma,
            )
            for step in step_grid
        ]
        ax.plot(
            step_grid / total_steps,
            rates,
            linewidth=1.4,
            label=(
                f"{variant.learning_rate_schedule}: "
                f"{variant.learning_rate:g}→{variant.min_learning_rate:g}"
            ),
        )
    ax.set_xlabel("fraction of optimizer trajectory")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedules used in the ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outdir / "learning_rate_schedules.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "learning_rate_schedules.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
