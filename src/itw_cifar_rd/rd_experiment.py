from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .config import config_fingerprint
from .io_utils import atomic_json_dump, torch_load, write_csv
from .rd_metrics import rate_at_retention, rd_metrics
from .sinkhorn import log_sinkhorn_uniform

METHOD_LABELS = {
    "simclr_z": "InfoNCE critic space z",
    "simclr_h": "SimCLR encoder h",
    "pixel": "pixel MSE",
    "random_z": "random encoder",
    "supervised_h": "supervised reference",
}


def pairwise_cost(
    features: torch.Tensor,
    *,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    x = features.to(device=device, dtype=torch.float32)
    if method == "pixel":
        norms = (x * x).sum(dim=1, keepdim=True)
        cost = norms + norms.T - 2.0 * (x @ x.T)
        cost = cost / x.shape[1]
    else:
        x = F.normalize(x, dim=1)
        cost = 2.0 - 2.0 * (x @ x.T)
    cost = 0.5 * (cost + cost.T)
    cost = cost.clamp_min_(0.0)
    cost.fill_diagonal_(0.0)
    return cost


def normalize_cost(
    cost: torch.Tensor,
    *,
    seed: int,
    sample_pairs: int,
) -> tuple[torch.Tensor, float]:
    n = cost.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, n, (sample_pairs,), generator=generator)
    j = torch.randint(0, n - 1, (sample_pairs,), generator=generator)
    j = j + (j >= i).long()
    values = cost[i.to(cost.device), j.to(cost.device)]
    positive = values[values > 1e-12]
    if positive.numel() == 0:
        raise RuntimeError("cost has no positive off-diagonal entries")
    scale = float(positive.median().item())
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid cost scale: {scale}")
    return cost / scale, scale


def _base_betas(rd_cfg: Dict[str, Any]) -> list[float]:
    values = np.geomspace(
        float(rd_cfg["beta_min"]),
        float(rd_cfg["beta_max"]),
        int(rd_cfg["beta_points"]),
    )
    return [0.0, *[float(x) for x in values]]


def _semantic_auc(rows: list[dict[str, Any]], metric: str, max_rate: float) -> float:
    rates = np.asarray([float(row["rate"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    order = np.argsort(rates)
    rates = rates[order]
    values = np.maximum.accumulate(values[order])
    grid = np.linspace(0.0, max_rate, 300)
    interpolated = np.interp(grid, rates, values, left=values[0], right=values[-1])
    integral = getattr(np, "trapezoid", np.trapz)(interpolated, grid)
    return float(integral / max_rate)


def _plot_seed(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    metrics = (
        ("coarse_retention", "coarse semantic retention"),
        ("fine_retention", "fine semantic retention"),
        ("fine_increment_retention", "fine beyond coarse"),
    )
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    for method in methods:
        subset = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["rate"]),
        )
        rates = [float(row["rate"]) for row in subset]
        for axis, (metric, title) in zip(axes, metrics):
            axis.plot(
                rates,
                [float(row[metric]) for row in subset],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=METHOD_LABELS.get(method, method),
            )
            axis.set_title(title)
            axis.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("normalized information retained")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_semantic_rd(
    config: Dict[str, Any],
    *,
    seed: int,
    feature_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "rd_curves.csv"
    thresholds_path = output_dir / "thresholds.json"
    diagnostics_path = output_dir / "diagnostics.json"
    if curves_path.is_file() and thresholds_path.is_file() and diagnostics_path.is_file():
        return {"status": "skipped", "curves": str(curves_path)}

    payload = torch_load(feature_path, map_location="cpu")
    fine = payload["fine"].long()
    coarse = payload["coarse"].long()
    features: Dict[str, torch.Tensor] = payload["features"]
    rd_cfg = config["rd"]
    methods = [str(x) for x in rd_cfg["methods"]]
    n = int(fine.numel())
    max_rate = math.log(n)
    dtype_name = str(rd_cfg.get("sinkhorn_dtype", "float32"))
    sinkhorn_dtype = torch.float64 if dtype_name == "float64" else torch.float32

    all_rows: list[dict[str, Any]] = []
    method_summaries: Dict[str, Any] = {}
    started = time.time()
    for method_index, method in enumerate(methods):
        print(f"[seed={seed}] RD method {method_index + 1}/{len(methods)}: {method}", flush=True)
        cost = pairwise_cost(features[method], method=method, device=device)
        cost, cost_scale = normalize_cost(
            cost,
            seed=seed + 10_000 * (method_index + 1),
            sample_pairs=int(rd_cfg.get("cost_scale_sample_pairs", 200_000)),
        )
        cost = cost.to(dtype=sinkhorn_dtype)
        betas = _base_betas(rd_cfg)
        max_extensions = int(rd_cfg.get("adaptive_beta_extensions", 0))
        multiplier = float(rd_cfg.get("adaptive_beta_multiplier", 4.0))
        target_retention = float(rd_cfg.get("adaptive_target_fine_retention", 0.98))
        log_a = log_b = None
        method_rows: list[dict[str, Any]] = []
        extension_count = 0
        beta_index = 0
        while beta_index < len(betas):
            beta = float(betas[beta_index])
            if beta == 0.0:
                log_joint = torch.full(
                    (n, n),
                    fill_value=-2.0 * math.log(n),
                    device=device,
                    dtype=sinkhorn_dtype,
                )
                iterations = 0
                marginal_error = 0.0
            else:
                result = log_sinkhorn_uniform(
                    -beta * cost,
                    max_iter=int(rd_cfg["sinkhorn_max_iter"]),
                    tol=float(rd_cfg["sinkhorn_tol"]),
                    check_every=int(rd_cfg.get("sinkhorn_check_every", 10)),
                    initial_log_a=log_a,
                    initial_log_b=log_b,
                )
                log_joint = result.log_joint
                log_a, log_b = result.log_a.detach(), result.log_b.detach()
                iterations = result.iterations
                marginal_error = result.marginal_error
            metrics = rd_metrics(
                log_joint,
                cost,
                fine.to(device),
                coarse.to(device),
            )
            row = {
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "beta": beta,
                "cost_scale": cost_scale,
                "sinkhorn_iterations": iterations,
                "marginal_error": marginal_error,
                **metrics,
            }
            method_rows.append(row)
            print(
                f"  beta={beta:.4g} rate={metrics['rate']:.4f} "
                f"coarse={metrics['coarse_retention']:.3f} "
                f"fine={metrics['fine_retention']:.3f} err={marginal_error:.2e}",
                flush=True,
            )
            beta_index += 1
            if (
                beta_index == len(betas)
                and method_rows[-1]["fine_retention"] < target_retention
                and extension_count < max_extensions
            ):
                betas.append(betas[-1] * multiplier)
                extension_count += 1

        thresholds: Dict[str, float] = {}
        for target in rd_cfg.get("retention_thresholds", [0.5, 0.8, 0.9]):
            target = float(target)
            slug = str(target).replace(".", "p")
            for metric in (
                "coarse_retention",
                "fine_retention",
                "fine_increment_retention",
            ):
                thresholds[f"R_{slug}_{metric}"] = rate_at_retention(
                    [row["rate"] for row in method_rows],
                    [row[metric] for row in method_rows],
                    target,
                )
        thresholds["auc_coarse"] = _semantic_auc(
            method_rows, "coarse_retention", max_rate
        )
        thresholds["auc_fine"] = _semantic_auc(
            method_rows, "fine_retention", max_rate
        )
        thresholds["auc_fine_increment"] = _semantic_auc(
            method_rows, "fine_increment_retention", max_rate
        )
        method_summaries[method] = {
            "cost_scale": cost_scale,
            "num_beta_points": len(method_rows),
            "max_rate": max(row["rate"] for row in method_rows),
            "max_coarse_retention": max(row["coarse_retention"] for row in method_rows),
            "max_fine_retention": max(row["fine_retention"] for row in method_rows),
            "thresholds": thresholds,
        }
        all_rows.extend(method_rows)
        del cost
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(all_rows, curves_path)
    thresholds_payload = {
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "num_examples": n,
        "max_information_rate": max_rate,
        "methods": method_summaries,
    }
    atomic_json_dump(thresholds_payload, thresholds_path)
    diagnostics = {
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "feature_diagnostics": payload.get("diagnostics", {}),
        "seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json_dump(diagnostics, diagnostics_path)
    _plot_seed(all_rows, output_dir / "semantic_rd_seed")
    return {
        "status": "complete",
        "curves": str(curves_path),
        "thresholds": str(thresholds_path),
        "diagnostics": str(diagnostics_path),
    }
