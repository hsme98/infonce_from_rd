"""Generic-channel recovery, KL-gap verification, and gauge stress test."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    channel_from_joint,
    fixed_output_rd,
    free_output_rd,
    kl_divergence,
    rd_objective,
    sinkhorn_log,
    weighted_row_kl,
    weighted_row_tv,
)
from .generators import make_generic_channel
from .infonce import (
    critic_mse_modulo_y_gauge,
    empirical_infonce_loss,
    sample_infonce_batch,
    train_tabular_infonce_checkpoints,
)

Array = np.ndarray


@dataclass
class Experiment1Config:
    n: int = 24
    tau: float = 1.0
    matched_modes: tuple[str, ...] = ("matched", "unmatched")
    k_values: tuple[int, ...] = (1, 4, 32)
    train_steps: tuple[int, ...] = (300, 1000, 2500)
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 512
    learning_rate: float = 0.04
    learning_rate_schedule: str = "multistep"
    min_learning_rate: float = 5e-4
    warmup_steps: int = 0
    lr_milestone_fractions: tuple[float, ...] = (0.2, 0.5, 0.8)
    lr_decay_gamma: float = 0.25
    tail_average_fraction: float | None = None
    gradient_accumulation_steps: int = 1
    competitors: int = 80
    gauge_lambdas: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    sinkhorn_tol: float = 1e-11
    outdir: str = "results/experiment1"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment1Config":
        data = dict(data)
        for key in (
            "matched_modes",
            "k_values",
            "train_steps",
            "seeds",
            "gauge_lambdas",
            "lr_milestone_fractions",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _canonical_score(channel: Array, p_y: Array) -> Array:
    return np.log(channel) - np.log(p_y)[None, :]


def _random_competitors(rng: np.random.Generator, p_x: Array, p_y: Array, count: int, tol: float) -> List[Array]:
    return [
        sinkhorn_log(rng.normal(size=(p_x.size, p_y.size)), p_x, p_y, tol=tol).joint
        for _ in range(count)
    ]


def _gauge_loss_difference(score: Array, shifted: Array, p_x: Array, W: Array, K: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    candidates, y = sample_infonce_batch(rng, p_x, W, batch_size=8192, K=K)
    return abs(empirical_infonce_loss(score, candidates, y) - empirical_infonce_loss(shifted, candidates, y))


def run(config: Experiment1Config) -> Dict[str, Any]:
    config.train_steps = tuple(sorted({int(step) for step in config.train_steps}))
    if not config.train_steps or config.train_steps[0] <= 0:
        raise ValueError("train_steps must contain positive checkpoint steps")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    diagnostic: Dict[str, Any] = {}

    for mode_index, mode in enumerate(config.matched_modes):
        if mode not in {"matched", "unmatched"}:
            raise ValueError(f"unknown mode {mode}")
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
                trajectory_seed = 1_000_000 * mode_index + 10_000 * seed + 100 * K + 17
                checkpoints = train_tabular_infonce_checkpoints(
                    problem.p_x,
                    problem.channel,
                    K=K,
                    checkpoint_steps=config.train_steps,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    learning_rate_schedule=config.learning_rate_schedule,
                    min_learning_rate=config.min_learning_rate,
                    warmup_steps=config.warmup_steps,
                    lr_milestone_fractions=config.lr_milestone_fractions,
                    lr_decay_gamma=config.lr_decay_gamma,
                    tail_average_fraction=config.tail_average_fraction,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    seed=trajectory_seed,
                )
                for steps in config.train_steps:
                    result = checkpoints[int(steps)]
                    joint_hat = sinkhorn_log(result.score, problem.p_x, problem.p_y, tol=config.sinkhorn_tol).joint
                    W_hat = channel_from_joint(joint_hat, problem.p_x)
                    records.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "K": K,
                            "steps": steps,
                            "positive_pairs": result.positive_pairs_seen,
                            "trajectory_seed": trajectory_seed,
                            "critic_mse_mod_gauge": critic_mse_modulo_y_gauge(
                                result.score, target, problem.p_x, problem.p_y
                            ),
                            "channel_row_kl": weighted_row_kl(problem.p_x, problem.channel, W_hat),
                            "joint_kl": kl_divergence(problem.joint, joint_hat),
                            "last_loss": float(result.losses[-1]),
                            "reversibility_l1": problem.reversibility_l1,
                        }
                    )
                    preferred = "unmatched" if "unmatched" in config.matched_modes else config.matched_modes[0]
                    if (
                        mode == preferred
                        and seed == config.seeds[0]
                        and K == max(config.k_values)
                        and steps == max(config.train_steps)
                    ):
                        diagnostic = {
                            "problem": problem,
                            "score": result.score,
                            "target": target,
                            "joint_hat": joint_hat,
                            "W_hat": W_hat,
                            "K": K,
                            "steps": steps,
                            "trajectory_seed": trajectory_seed,
                        }

    if not diagnostic:
        raise RuntimeError("no diagnostic run was selected")
    problem = diagnostic["problem"]
    learned_score = diagnostic["score"]
    exact_score = diagnostic["target"]
    rng = np.random.default_rng(424242)

    exact_d = -config.tau * exact_score
    learned_d = -config.tau * learned_score
    competitors = _random_competitors(rng, problem.p_x, problem.p_y, config.competitors, config.sinkhorn_tol)
    exact_x, exact_y, learned_y = [], [], []
    base_exact = rd_objective(problem.joint, exact_d, config.tau)
    base_learned = rd_objective(problem.joint, learned_d, config.tau)
    for joint in competitors:
        exact_x.append(kl_divergence(joint, problem.joint))
        exact_y.append(rd_objective(joint, exact_d, config.tau) - base_exact)
        learned_y.append(rd_objective(joint, learned_d, config.tau) - base_learned)

    g = rng.normal(size=problem.p_y.size)
    g -= np.dot(problem.p_y, g)
    g /= np.sqrt(np.dot(problem.p_y, g * g))
    fixed0 = fixed_output_rd(problem.p_x, problem.p_y, learned_d, beta=1 / config.tau, tol=config.sinkhorn_tol).joint
    free0 = free_output_rd(problem.p_x, learned_d, tau=config.tau).joint
    fixed_W0 = channel_from_joint(fixed0, problem.p_x)
    free_W0 = channel_from_joint(free0, problem.p_x)
    gauge_rows = []
    for lam in config.gauge_lambdas:
        d_lam = learned_d + lam * g[None, :]
        score_lam = -d_lam / config.tau
        fixed = fixed_output_rd(problem.p_x, problem.p_y, d_lam, beta=1 / config.tau, tol=config.sinkhorn_tol).joint
        free = free_output_rd(problem.p_x, d_lam, tau=config.tau).joint
        fixed_W = channel_from_joint(fixed, problem.p_x)
        free_W = channel_from_joint(free, problem.p_x)
        gauge_rows.append(
            {
                "lambda": float(lam),
                "fixed_row_tv_from_lambda0": weighted_row_tv(problem.p_x, fixed_W0, fixed_W),
                "free_row_tv_from_lambda0": weighted_row_tv(problem.p_x, free_W0, free_W),
                "fixed_row_kl_from_lambda0": weighted_row_kl(problem.p_x, fixed_W0, fixed_W),
                "free_row_kl_from_lambda0": weighted_row_kl(problem.p_x, free_W0, free_W),
                "infonce_loss_difference": _gauge_loss_difference(
                    learned_score, score_lam, problem.p_x, problem.channel, int(diagnostic["K"]), 999 + int(100 * lam)
                ),
            }
        )

    payload = {
        "config": asdict(config),
        "records": records,
        "diagnostics": {
            "mode": "matched" if problem.matched else "unmatched",
            "K": diagnostic["K"],
            "steps": diagnostic["steps"],
            "kl_gap": {
                "joint_kl": exact_x,
                "exact_objective_gap": exact_y,
                "learned_objective_gap": learned_y,
                "exact_max_abs_error": float(np.max(np.abs(np.asarray(exact_x) - np.asarray(exact_y)))),
                "learned_mean_abs_error": float(np.mean(np.abs(np.asarray(exact_x) - np.asarray(learned_y)))),
            },
            "gauge_stress": gauge_rows,
        },
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    np.savez_compressed(
        outdir / "diagnostic_arrays.npz",
        p_x=problem.p_x,
        p_y=problem.p_y,
        true_joint=problem.joint,
        true_channel=problem.channel,
        exact_score=exact_score,
        learned_score=learned_score,
        recovered_joint=diagnostic["joint_hat"],
        recovered_channel=diagnostic["W_hat"],
        kl_x=np.asarray(exact_x),
        kl_exact_y=np.asarray(exact_y),
        kl_learned_y=np.asarray(learned_y),
        gauge=g,
    )
    make_figure(payload, outdir)
    return payload


def _aggregate(records: Iterable[Dict[str, Any]], mode: str, K: int, steps: int) -> tuple[float, float, float]:
    values = np.asarray([
        r["channel_row_kl"] for r in records if r["mode"] == mode and r["K"] == K and r["steps"] == steps
    ])
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))


def make_figure(payload: Dict[str, Any], outdir: Path) -> None:
    """Create the three-panel theorem-facing figure."""
    cfg, records, diag = payload["config"], payload["records"], payload["diagnostics"]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 2.75), constrained_layout=True)

    ax = axes[0]
    markers = {"matched": "o", "unmatched": "s"}
    linestyles = {"matched": "-", "unmatched": "--"}
    effective_batch = int(cfg["batch_size"]) * int(cfg.get("gradient_accumulation_steps", 1))
    for mode in cfg["matched_modes"]:
        for K in cfg["k_values"]:
            x, med, lo, hi = [], [], [], []
            for steps in cfg["train_steps"]:
                m, lower, upper = _aggregate(records, mode, K, steps)
                x.append(steps * effective_batch)
                med.append(m)
                lo.append(lower)
                hi.append(upper)
            line = ax.plot(
                x,
                med,
                marker=markers[mode],
                linestyle=linestyles[mode],
                linewidth=1.2,
                markersize=3.0,
                label=f"{mode}, K={K}",
            )[0]
            ax.fill_between(x, lo, hi, alpha=0.09, color=line.get_color())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("positive pairs processed")
    ax.set_ylabel("channel row KL")
    ax.set_title("(a) inverse channel recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False, columnspacing=0.8)

    ax = axes[1]
    x = np.asarray(diag["kl_gap"]["joint_kl"])
    exact = np.asarray(diag["kl_gap"]["exact_objective_gap"])
    learned = np.asarray(diag["kl_gap"]["learned_objective_gap"])
    ax.scatter(x, exact, s=12, alpha=0.65, label="exact critic")
    ax.scatter(x, learned, s=12, alpha=0.55, marker="x", label="learned critic")
    limit = 1.05 * max(float(x.max()), float(exact.max()), float(learned.max()))
    ax.plot([0, limit], [0, limit], "--", linewidth=1, label="identity")
    ax.set_xlim(0, limit)
    ax.set_ylim(min(-0.02 * limit, float(learned.min()) * 1.05), limit)
    ax.set_xlabel(r"$D_{\rm KL}(P_XV\,\|\,P_XW^\star)$")
    ax.set_ylabel(r"$\mathcal{J}_d(V)-\mathcal{J}_d(W^\star)$")
    ax.set_title("(b) exact KL-gap geometry")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    rows = diag["gauge_stress"]
    lam = np.asarray([row["lambda"] for row in rows], dtype=float)
    fixed = np.asarray([row["fixed_row_tv_from_lambda0"] for row in rows], dtype=float)
    free = np.asarray([row["free_row_tv_from_lambda0"] for row in rows], dtype=float)
    loss = np.asarray([row["infonce_loss_difference"] for row in rows], dtype=float)
    ax.plot(lam, fixed, marker="o", linewidth=1.3, label="fixed-output RD")
    ax.plot(lam, free, marker="s", linewidth=1.3, label="free-output RD")
    ax.plot(lam, loss, marker="^", linewidth=1.0, label="InfoNCE loss change")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel(r"gauge amplitude $\lambda$")
    ax.set_ylabel(r"change from $\lambda=0$")
    ax.set_title("(c) output-gauge stress test")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(outdir / "generic_channel_theorem_experiment.pdf", bbox_inches="tight")
    fig.savefig(outdir / "generic_channel_theorem_experiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

