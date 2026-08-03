"""Hierarchical Markov coarse-graining using an InfoNCE-induced distortion.

This module supports both one-seed smoke tests and multi-seed paper sweeps.
Each seed produces its own metrics/arrays directory, while the experiment root
contains aggregate summaries and a median/10--90% figure.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    aggregate_joint,
    channel_from_joint,
    entropy,
    mutual_information,
    sinkhorn_log,
    sinkhorn_primal,
    weighted_row_kl,
)
from .generators import make_hierarchical_reversible_chain
from .infonce import train_tabular_infonce

Array = np.ndarray


@dataclass
class Experiment2Config:
    macros: int = 4
    micros_per_macro: int = 4
    states_per_micro: int = 4
    within_micro_cost: float = 1.2
    within_macro_cost: float = 4.0
    cross_macro_cost: float = 10.0
    transition_beta: float = 0.65
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    representative_seed: int | None = None
    tau: float = 1.0
    K: int = 16
    train_steps: int = 3000
    batch_size: int = 1024
    learning_rate: float = 0.035
    beta_min: float = 1e-3
    beta_max: float = 25.0
    beta_points: int = 70
    sinkhorn_tol: float = 1e-10
    random_embedding_dim: int = 8
    aggregate_grid_points: int = 220
    save_per_seed_figures: bool = False
    outdir: str = "results/experiment2"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment2Config":
        data = dict(data)
        # Backward compatibility with the original one-seed configuration.
        if "seed" in data and "seeds" not in data:
            data["seeds"] = (int(data.pop("seed")),)
        elif "seed" in data:
            data.pop("seed")
        if "seeds" in data:
            data["seeds"] = tuple(int(seed) for seed in data["seeds"])
        return cls(**data)


def _canonical_distortion(channel: Array, marginal: Array, tau: float) -> Array:
    d = -tau * (np.log(channel) - np.log(marginal)[None, :])
    return d - float(d.min())


def _random_spherical_distortion(n: int, dim: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)


def _semantic_curve(
    d: Array,
    p: Array,
    macro: Array,
    micro: Array,
    betas: Array,
    tol: float,
) -> Dict[str, Array]:
    h_macro = entropy(np.bincount(macro, weights=p))
    h_micro = entropy(np.bincount(micro, weights=p))
    h_state = entropy(p)
    rate: List[float] = []
    distortion: List[float] = []
    macro_fraction: List[float] = []
    micro_increment: List[float] = []
    state_increment: List[float] = []
    log_u = log_v = None
    for beta in betas:
        result = sinkhorn_primal(
            -float(beta) * d,
            p,
            p,
            tol=tol,
            max_iter=30_000,
            initial_log_u=log_u,
            initial_log_v=log_v,
        )
        log_u, log_v = result.log_u, result.log_v
        joint = result.joint
        i_state = mutual_information(joint)
        i_macro = mutual_information(aggregate_joint(joint, macro))
        i_micro = mutual_information(aggregate_joint(joint, micro))
        rate.append(i_state)
        distortion.append(float(np.sum(joint * d)))
        macro_fraction.append(i_macro / h_macro)
        micro_increment.append((i_micro - i_macro) / (h_micro - h_macro))
        state_increment.append((i_state - i_micro) / (h_state - h_micro))
    order = np.argsort(rate)
    return {
        "beta": betas[order],
        "rate": np.asarray(rate)[order],
        "distortion": np.asarray(distortion)[order],
        "macro_fraction": np.clip(np.asarray(macro_fraction)[order], 0, 1),
        "micro_increment": np.clip(np.asarray(micro_increment)[order], 0, 1),
        "state_increment": np.clip(np.asarray(state_increment)[order], 0, 1),
    }


def _rate_at(curve: Dict[str, Array], key: str, target: float = 0.9) -> float:
    x, y = curve["rate"], curve[key]
    hit = np.where(y >= target)[0]
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0 or y[i] <= y[i - 1]:
        return float(x[i])
    return float(
        x[i - 1]
        + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    )


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


def _summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _strictly_increasing_xy(x: Array, y: Array) -> tuple[Array, Array]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size == x_sorted.size:
        return x_sorted, y_sorted
    # Duplicate rates can occur near beta=0. Average their ordinate values.
    sums = np.zeros_like(unique_x)
    counts = np.zeros_like(unique_x)
    np.add.at(sums, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    return unique_x, sums / counts


def _aggregate_curves(
    payloads: Sequence[Dict[str, Any]],
    grid_points: int,
) -> Dict[str, Dict[str, Any]]:
    methods = tuple(payloads[0]["curves"].keys())
    semantic_keys = ("macro_fraction", "micro_increment", "state_increment")
    aggregate: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        max_common_rate = min(
            float(np.max(np.asarray(payload["curves"][method]["rate"])))
            for payload in payloads
        )
        grid = np.linspace(0.0, max_common_rate, grid_points)
        method_payload: Dict[str, Any] = {"rate_grid": grid}
        for key in semantic_keys:
            interpolated = []
            for payload in payloads:
                curve = payload["curves"][method]
                x, y = _strictly_increasing_xy(
                    np.asarray(curve["rate"]), np.asarray(curve[key])
                )
                interpolated.append(np.interp(grid, x, y))
            values = np.stack(interpolated, axis=0)
            method_payload[key] = {
                "median": np.median(values, axis=0),
                "q10": np.quantile(values, 0.10, axis=0),
                "q90": np.quantile(values, 0.90, axis=0),
            }
        aggregate[method] = method_payload
    return aggregate


def _run_single(config: Experiment2Config, seed: int) -> tuple[Dict[str, Any], Dict[str, Array]]:
    chain = make_hierarchical_reversible_chain(
        macros=config.macros,
        micros_per_macro=config.micros_per_macro,
        states_per_micro=config.states_per_micro,
        within_micro_cost=config.within_micro_cost,
        within_macro_cost=config.within_macro_cost,
        cross_macro_cost=config.cross_macro_cost,
        transition_beta=config.transition_beta,
        seed=seed,
    )
    trained = train_tabular_infonce(
        chain.stationary,
        chain.transition,
        K=config.K,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=1000 + seed,
    )
    joint_hat = sinkhorn_log(
        trained.score,
        chain.stationary,
        chain.stationary,
        tol=config.sinkhorn_tol,
    ).joint
    w_hat = channel_from_joint(joint_hat, chain.stationary)

    d_oracle = _canonical_distortion(chain.transition, chain.stationary, config.tau)
    d_learned = _canonical_distortion(w_hat, chain.stationary, config.tau)
    n = chain.stationary.size
    distortions = {
        "oracle": d_oracle,
        "InfoNCE": d_learned,
        "Hamming": 1.0 - np.eye(n),
        "random sphere": _random_spherical_distortion(
            n, config.random_embedding_dim, 30_000 + seed
        ),
    }
    betas = np.concatenate(
        [[0.0], np.geomspace(config.beta_min, config.beta_max, config.beta_points)]
    )
    curves = {
        name: _semantic_curve(
            distortion,
            chain.stationary,
            chain.macro_labels,
            chain.micro_labels,
            betas,
            config.sinkhorn_tol,
        )
        for name, distortion in distortions.items()
    }
    thresholds = {
        name: {
            "R90_macro": _rate_at(curve, "macro_fraction"),
            "R90_micro_increment": _rate_at(curve, "micro_increment"),
            "R90_state_increment": _rate_at(curve, "state_increment"),
        }
        for name, curve in curves.items()
    }
    payload = {
        "seed": seed,
        "metrics": {
            "detailed_balance_max_error": chain.detailed_balance_error,
            "stationarity_l1": float(
                np.sum(np.abs(chain.stationary @ chain.transition - chain.stationary))
            ),
            "recovered_channel_row_kl": weighted_row_kl(
                chain.stationary, chain.transition, w_hat
            ),
            "positive_pairs_seen": trained.positive_pairs_seen,
            "last_infonce_loss": float(trained.losses[-1]),
            "rate_thresholds": thresholds,
        },
        "curves": curves,
    }
    arrays = {
        "transition": chain.transition,
        "stationary": chain.stationary,
        "recovered_channel": w_hat,
        "recovered_joint": joint_hat,
        "macro_labels": chain.macro_labels,
        "micro_labels": chain.micro_labels,
        "d_oracle": d_oracle,
        "d_learned": d_learned,
        **{
            f"{name.replace(' ', '_')}_{key}": value
            for name, curve in curves.items()
            for key, value in curve.items()
        },
    }
    return payload, arrays


def _write_single_seed(
    payload: Dict[str, Any],
    arrays: Dict[str, Array],
    seed_dir: Path,
    save_figure: bool,
) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    np.savez_compressed(seed_dir / "arrays.npz", **arrays)
    if save_figure:
        make_single_seed_figure(payload, arrays, seed_dir)


def _aggregate_metrics(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_names = (
        "detailed_balance_max_error",
        "stationarity_l1",
        "recovered_channel_row_kl",
        "positive_pairs_seen",
        "last_infonce_loss",
    )
    scalar = {
        name: _summary(payload["metrics"][name] for payload in payloads)
        for name in scalar_names
    }
    methods = payloads[0]["metrics"]["rate_thresholds"].keys()
    threshold_names = (
        "R90_macro",
        "R90_micro_increment",
        "R90_state_increment",
    )
    thresholds = {
        method: {
            name: _summary(
                payload["metrics"]["rate_thresholds"][method][name]
                for payload in payloads
            )
            for name in threshold_names
        }
        for method in methods
    }
    return {"metrics": scalar, "rate_thresholds": thresholds}


def _write_summary_csv(payloads: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = [
        "seed",
        "recovered_channel_row_kl",
        "detailed_balance_max_error",
        "stationarity_l1",
        "R90_macro",
        "R90_micro_increment",
        "R90_state_increment",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            metrics = payload["metrics"]
            rates = metrics["rate_thresholds"]["InfoNCE"]
            writer.writerow(
                {
                    "seed": payload["seed"],
                    "recovered_channel_row_kl": metrics[
                        "recovered_channel_row_kl"
                    ],
                    "detailed_balance_max_error": metrics[
                        "detailed_balance_max_error"
                    ],
                    "stationarity_l1": metrics["stationarity_l1"],
                    **rates,
                }
            )


def run(config: Experiment2Config) -> Dict[str, Any]:
    if not config.seeds:
        raise ValueError("Experiment 2 requires at least one seed")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Experiment 2 seeds must be unique")

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    representative_seed = (
        config.representative_seed
        if config.representative_seed is not None
        else config.seeds[0]
    )
    if representative_seed not in config.seeds:
        raise ValueError("representative_seed must occur in seeds")

    payloads: List[Dict[str, Any]] = []
    arrays_by_seed: Dict[int, Dict[str, Array]] = {}
    for seed in config.seeds:
        payload, arrays = _run_single(config, seed)
        payloads.append(payload)
        arrays_by_seed[seed] = arrays
        _write_single_seed(
            payload,
            arrays,
            outdir / f"seed_{seed:03d}",
            config.save_per_seed_figures,
        )

    aggregate_metrics = _aggregate_metrics(payloads)
    aggregate_curves = _aggregate_curves(payloads, config.aggregate_grid_points)
    root_payload = {
        "config": asdict(config),
        "seeds": list(config.seeds),
        "representative_seed": representative_seed,
        "per_seed": [
            {"seed": payload["seed"], "metrics": payload["metrics"]}
            for payload in payloads
        ],
        "aggregate": {
            **aggregate_metrics,
            "curves": aggregate_curves,
        },
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(root_payload), handle, indent=2)
    _write_summary_csv(payloads, outdir / "multiseed_summary.csv")

    np.savez_compressed(
        outdir / "aggregate_arrays.npz",
        **{
            f"{method.replace(' ', '_')}_{key}_{stat}": np.asarray(values[stat])
            for method, method_data in aggregate_curves.items()
            for key, values in method_data.items()
            if key != "rate_grid"
            for stat in ("median", "q10", "q90")
        },
        **{
            f"{method.replace(' ', '_')}_rate_grid": np.asarray(
                method_data["rate_grid"]
            )
            for method, method_data in aggregate_curves.items()
        },
    )
    make_multiseed_figure(
        root_payload,
        arrays_by_seed[representative_seed],
        outdir,
    )
    return root_payload


def _lines(ax: plt.Axes, macro: Array, micro: Array) -> None:
    for index in range(1, macro.size):
        if micro[index] != micro[index - 1]:
            ax.axhline(index - 0.5, color="white", linewidth=0.35, alpha=0.8)
            ax.axvline(index - 0.5, color="white", linewidth=0.35, alpha=0.8)
        if macro[index] != macro[index - 1]:
            ax.axhline(index - 0.5, color="white", linewidth=0.9)
            ax.axvline(index - 0.5, color="white", linewidth=0.9)


def make_single_seed_figure(
    payload: Dict[str, Any], arrays: Dict[str, Array], outdir: Path
) -> None:
    curves = payload["curves"]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.8), constrained_layout=True)
    transition = arrays["transition"]
    recovered = arrays["recovered_channel"]
    macro = arrays["macro_labels"]
    micro = arrays["micro_labels"]
    vmax = max(float(transition.max()), float(recovered.max()))
    for ax, matrix, title in [
        (axes[0], transition, "(a) true transition"),
        (axes[1], recovered, "(b) InfoNCE recovery"),
    ]:
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=vmax)
        _lines(ax, macro, micro)
        ax.set_title(title)
        ax.set_xlabel("next state")
        ax.set_ylabel("state")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes[:2], shrink=0.78, label=r"$P(y|x)$")

    ax = axes[2]
    learned = curves["InfoNCE"]
    ax.plot(learned["rate"], learned["macro_fraction"], linewidth=1.7, label="macro")
    ax.plot(
        learned["rate"],
        learned["micro_increment"],
        linewidth=1.7,
        label="micro beyond macro",
    )
    ax.plot(
        learned["rate"],
        learned["state_increment"],
        linewidth=1.7,
        label="state beyond micro",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]")
    ax.set_ylabel("retained information fraction")
    ax.set_title("(c) learned compression hierarchy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[3]
    styles = {
        "oracle": ("-", 1.8),
        "InfoNCE": ("-", 1.8),
        "Hamming": ("--", 1.2),
        "random sphere": (":", 1.4),
    }
    for name, curve in curves.items():
        linestyle, linewidth = styles[name]
        ax.plot(
            curve["rate"],
            curve["macro_fraction"],
            linestyle=linestyle,
            linewidth=linewidth,
            label=name,
        )
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]")
    ax.set_ylabel("macro information fraction")
    ax.set_title("(d) operational baseline comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(outdir / "experiment2_markov_hierarchy.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "experiment2_markov_hierarchy.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_multiseed_figure(
    payload: Dict[str, Any], representative: Dict[str, Array], outdir: Path
) -> None:
    """Create an aligned 2x2 figure for the multiseed Markov experiment."""
    curves = payload["aggregate"]["curves"]
    plt.rcParams.update(
        {
            "font.size": 7.1,
            "axes.titlesize": 7.9,
            "axes.labelsize": 7.2,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Fixed axis positions keep the square heatmaps aligned with the lower plots.
    fig = plt.figure(figsize=(7.15, 4.90))
    left_x, right_x, panel_w = 0.095, 0.555, 0.330
    heat_y, heat_h = 0.490, 0.482
    plot_y, plot_h = 0.075, 0.292
    ax_true = fig.add_axes([left_x, heat_y, panel_w, heat_h])
    ax_rec = fig.add_axes([right_x, heat_y, panel_w, heat_h])
    ax_hier = fig.add_axes([left_x, plot_y, panel_w, plot_h])
    ax_base = fig.add_axes([right_x, plot_y, panel_w, plot_h])
    cax = fig.add_axes([0.910, heat_y, 0.014, heat_h])

    transition = np.asarray(representative["transition"], dtype=float)
    recovered = np.asarray(representative["recovered_channel"], dtype=float)
    macro = np.asarray(representative["macro_labels"], dtype=int)
    micro = np.asarray(representative["micro_labels"], dtype=int)
    vmax = float(max(transition.max(), recovered.max()))
    image = None
    for ax, matrix, title in [
        (ax_true, transition, "(a) True transition channel"),
        (ax_rec, recovered, "(b) InfoNCE recovery"),
    ]:
        image = ax.imshow(
            matrix,
            origin="upper",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title, pad=2.0)
        ax.set_xlabel("next state", labelpad=1.5)
        ticks = [0, 16, 32, 48, 63]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        _lines(ax, macro, micro)
    ax_true.set_ylabel("current state", labelpad=1.8)
    ax_rec.set_yticklabels([])
    ax_rec.set_ylabel("")
    assert image is not None
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_label(r"$W(y\mid x)$", labelpad=1.5)
    colorbar.ax.tick_params(pad=1.0)

    learned = curves["InfoNCE"]
    rate = np.asarray(learned["rate_grid"], dtype=float)
    for metric, label in [
        ("macro_fraction", "macro-community"),
        ("micro_increment", "micro beyond macro"),
        ("state_increment", "state beyond micro"),
    ]:
        stats = learned[metric]
        line = ax_hier.plot(
            rate, np.asarray(stats["median"], dtype=float), linewidth=1.35, label=label
        )[0]
        ax_hier.fill_between(
            rate,
            np.asarray(stats["q10"], dtype=float),
            np.asarray(stats["q90"], dtype=float),
            alpha=0.10,
            color=line.get_color(),
        )
    ax_hier.set_xlim(0.0, 4.15)
    ax_hier.set_ylim(0.0, 1.02)
    ax_hier.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]", labelpad=1.6)
    ax_hier.set_ylabel("normalized information retained", labelpad=2.0)
    ax_hier.set_title("(c) InfoNCE-induced hierarchy", pad=2.0)
    ax_hier.grid(True, alpha=0.22)
    ax_hier.legend(frameon=False, loc="lower right", handlelength=2.0)

    line_styles = {
        "oracle": "--",
        "InfoNCE": "-",
        "Hamming": "-.",
        "random sphere": ":",
    }
    for method in ["oracle", "InfoNCE", "Hamming", "random sphere"]:
        method_curve = curves[method]
        ax_base.plot(
            np.asarray(method_curve["rate_grid"], dtype=float),
            np.asarray(method_curve["macro_fraction"]["median"], dtype=float),
            linewidth=1.35,
            linestyle=line_styles[method],
            label=method,
        )
    ax_base.axvline(np.log(4.0), linewidth=0.72, linestyle=":")
    ax_base.text(
        np.log(4.0) + 0.035,
        0.028,
        r"$\log 4$",
        fontsize=6.0,
        rotation=90,
        va="bottom",
    )
    ax_base.set_xlim(0.0, 4.15)
    ax_base.set_ylim(0.0, 1.02)
    ax_base.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]", labelpad=1.6)
    ax_base.set_ylabel("macro information retained", labelpad=2.0)
    ax_base.set_title("(d) Macro-information efficiency", pad=2.0)
    ax_base.grid(True, alpha=0.22)
    ax_base.legend(frameon=False, loc="lower right", handlelength=2.0)

    for stem in ("markov_full", "experiment2_markov_hierarchy"):
        fig.savefig(outdir / f"{stem}.pdf")
        fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)

