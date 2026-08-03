"""Synthetic positive-pair channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import channel_from_joint, normalize_probability, sinkhorn_log

Array = np.ndarray


@dataclass(frozen=True)
class GenericChannel:
    p_x: Array
    p_y: Array
    joint: Array
    channel: Array
    matched: bool
    reversibility_l1: float


def _dirichlet_floor(rng: np.random.Generator, n: int, concentration: float, floor_mix: float) -> Array:
    p = rng.dirichlet(np.full(n, concentration))
    return normalize_probability((1.0 - floor_mix) * p + floor_mix / n)


def make_generic_channel(
    *,
    n: int,
    seed: int,
    matched: bool,
    concentration: float = 2.0,
    floor_mix: float = 0.15,
    log_kernel_scale: float = 1.0,
    sinkhorn_tol: float = 1e-13,
) -> GenericChannel:
    """Create an arbitrary full-support coupling before defining any distortion."""
    rng = np.random.default_rng(seed)
    p_x = _dirichlet_floor(rng, n, concentration, floor_mix)
    p_y = p_x.copy() if matched else _dirichlet_floor(rng, n, concentration, floor_mix)
    if not matched and np.linalg.norm(p_y - p_x, ord=1) < 0.15:
        p_y = normalize_probability(np.roll(p_y, 1))
    rank = min(5, n)
    log_base = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, n)) / np.sqrt(rank)
    log_base += 0.35 * rng.normal(size=(n, n))
    joint = sinkhorn_log(log_kernel_scale * log_base, p_x, p_y, tol=sinkhorn_tol).joint
    channel = channel_from_joint(joint, p_x)
    rev = float(np.sum(np.abs(joint - joint.T))) if matched else float("nan")
    return GenericChannel(p_x, p_y, joint, channel, matched, rev)


@dataclass(frozen=True)
class HierarchicalChain:
    transition: Array
    stationary: Array
    joint: Array
    macro_labels: Array
    micro_labels: Array
    state_labels: Array
    base_distortion: Array
    detailed_balance_error: float


def make_hierarchical_reversible_chain(
    *,
    macros: int = 4,
    micros_per_macro: int = 4,
    states_per_micro: int = 4,
    within_micro_cost: float = 1.2,
    within_macro_cost: float = 4.0,
    cross_macro_cost: float = 10.0,
    transition_beta: float = 0.65,
    seed: int = 0,
) -> HierarchicalChain:
    """A nested, full-support, reversible Markov chain.

    Equal group sizes make the hierarchical Gibbs kernel have constant row
    sums.  The resulting channel is symmetric and doubly stochastic, so the
    stationary distribution is uniform.  The separated costs produce a clear
    macro -> micro -> state compression hierarchy.
    """
    if not (0 < within_micro_cost < within_macro_cost < cross_macro_cost):
        raise ValueError("hierarchical costs must be strictly ordered")
    n_micro = macros * micros_per_macro
    n = n_micro * states_per_micro
    state = np.arange(n)
    micro = state // states_per_micro
    macro = micro // micros_per_macro

    same_state = state[:, None] == state[None, :]
    same_micro = micro[:, None] == micro[None, :]
    same_macro = macro[:, None] == macro[None, :]
    d = np.full((n, n), cross_macro_cost, dtype=np.float64)
    d[same_macro] = within_macro_cost
    d[same_micro] = within_micro_cost
    d[same_state] = 0.0

    # A very small symmetric perturbation avoids visually perfect blocks while
    # preserving the hierarchy and reversibility.
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.015, size=(n, n))
    noise = 0.5 * (noise + noise.T)
    noise[same_state] = 0.0
    d_noisy = np.maximum(d + noise, 0.0)
    kernel = np.exp(-transition_beta * d_noisy)
    # Symmetric kernel has nearly constant row sums; Sinkhorn with uniform
    # marginals makes the stationarity exact while retaining symmetry.
    uniform = np.full(n, 1.0 / n)
    joint = sinkhorn_log(np.log(kernel), uniform, uniform, tol=1e-13).joint
    transition = channel_from_joint(joint, uniform)
    return HierarchicalChain(
        transition=transition,
        stationary=uniform,
        joint=joint,
        macro_labels=macro,
        micro_labels=micro,
        state_labels=state,
        base_distortion=d_noisy,
        detailed_balance_error=float(np.max(np.abs(joint - joint.T))),
    )
