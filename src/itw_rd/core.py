"""Numerical primitives for finite-state rate-distortion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Array = np.ndarray
_EPS = 1e-300


def normalize_probability(p: Array, *, name: str = "p") -> Array:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    total = float(p.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    p = p / total
    if np.any(p <= 0):
        raise ValueError(f"{name} must have full support")
    return p


def logsumexp(a: Array, axis: int, keepdims: bool = False) -> Array:
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


@dataclass(frozen=True)
class SinkhornResult:
    joint: Array
    iterations: int
    row_error: float
    col_error: float
    log_u: Array
    log_v: Array


def sinkhorn_log(
    log_kernel: Array,
    row_marginal: Array,
    col_marginal: Array,
    *,
    max_iter: int = 30_000,
    tol: float = 1e-12,
    initial_log_u: Optional[Array] = None,
    initial_log_v: Optional[Array] = None,
) -> SinkhornResult:
    """Scale ``exp(log_kernel)`` to prescribed positive marginals.

    Row- and column-only offsets in ``log_kernel`` are absorbed by the scaling
    factors.  This is precisely the fixed-marginal gauge used in the paper.
    """
    L = np.asarray(log_kernel, dtype=np.float64)
    r = normalize_probability(row_marginal, name="row_marginal")
    c = normalize_probability(col_marginal, name="col_marginal")
    if L.shape != (r.size, c.size) or np.any(~np.isfinite(L)):
        raise ValueError("log_kernel has the wrong shape or nonfinite values")

    log_r, log_c = np.log(r), np.log(c)
    log_u = np.zeros_like(r) if initial_log_u is None else np.asarray(initial_log_u, dtype=np.float64).copy()
    log_v = np.zeros_like(c) if initial_log_v is None else np.asarray(initial_log_v, dtype=np.float64).copy()
    if log_u.shape != r.shape or log_v.shape != c.shape:
        raise ValueError("initial Sinkhorn potentials have the wrong shape")

    row_error = col_error = np.inf
    for iteration in range(1, max_iter + 1):
        log_u = log_r - logsumexp(L + log_v[None, :], axis=1)
        log_v = log_c - logsumexp(L + log_u[:, None], axis=0)
        if iteration % 10 == 0 or iteration == max_iter:
            joint = np.exp(L + log_u[:, None] + log_v[None, :])
            row_error = float(np.max(np.abs(joint.sum(axis=1) - r)))
            col_error = float(np.max(np.abs(joint.sum(axis=0) - c)))
            if max(row_error, col_error) <= tol:
                break

    joint = np.exp(L + log_u[:, None] + log_v[None, :])
    joint /= joint.sum()
    row_error = float(np.max(np.abs(joint.sum(axis=1) - r)))
    col_error = float(np.max(np.abs(joint.sum(axis=0) - c)))
    # At very large beta, Sinkhorn is ill-conditioned.  A 1e-6 marginal error
    # is more than sufficient for the plotted information curves; moderate
    # kernels still converge to the requested tolerance.
    if max(row_error, col_error) > max(100_000.0 * tol, 1e-6):
        raise RuntimeError(
            f"Sinkhorn did not converge: row={row_error:.3e}, col={col_error:.3e}"
        )
    return SinkhornResult(joint, iteration, row_error, col_error, log_u, log_v)



def sinkhorn_primal(
    log_kernel: Array,
    row_marginal: Array,
    col_marginal: Array,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-10,
    initial_log_u: Optional[Array] = None,
    initial_log_v: Optional[Array] = None,
) -> SinkhornResult:
    """Fast primal-domain Sinkhorn for small dense matrices.

    This routine is substantially faster than the fully log-domain solver for
    the 64x64 RD-curve sweeps.  The kernel is globally shifted and clipped only
    below the float64 underflow scale; row/column scalings absorb the global
    shift.  It falls back to the log-domain routine if numerical scaling fails.
    """
    L = np.asarray(log_kernel, dtype=np.float64)
    r = normalize_probability(row_marginal, name="row_marginal")
    c = normalize_probability(col_marginal, name="col_marginal")
    if L.shape != (r.size, c.size) or np.any(~np.isfinite(L)):
        raise ValueError("log_kernel has the wrong shape or nonfinite values")
    shifted = np.clip(L - float(np.max(L)), -700.0, 0.0)
    K = np.exp(shifted)
    tiny = 1e-300

    if initial_log_u is None:
        u = np.ones_like(r)
    else:
        lu = np.asarray(initial_log_u, dtype=np.float64)
        u = np.exp(np.clip(lu - np.mean(lu), -300.0, 300.0))
    if initial_log_v is None:
        v = np.ones_like(c)
    else:
        lv = np.asarray(initial_log_v, dtype=np.float64)
        v = np.exp(np.clip(lv - np.mean(lv), -300.0, 300.0))

    row_error = col_error = np.inf
    for iteration in range(1, max_iter + 1):
        Kv = K @ v
        if np.any(~np.isfinite(Kv)) or np.any(Kv <= 0):
            return sinkhorn_log(
                L, r, c, max_iter=max_iter, tol=tol,
                initial_log_u=initial_log_u, initial_log_v=initial_log_v,
            )
        u = r / np.maximum(Kv, tiny)
        KTu = K.T @ u
        if np.any(~np.isfinite(KTu)) or np.any(KTu <= 0):
            return sinkhorn_log(
                L, r, c, max_iter=max_iter, tol=tol,
                initial_log_u=initial_log_u, initial_log_v=initial_log_v,
            )
        v = c / np.maximum(KTu, tiny)

        if iteration % 100 == 0:
            # Balance the magnitudes of u and v without changing the joint.
            log_a = 0.5 * (np.mean(np.log(np.maximum(v, tiny))) - np.mean(np.log(np.maximum(u, tiny))))
            log_a = float(np.clip(log_a, -100.0, 100.0))
            a = np.exp(log_a)
            u *= a
            v /= a

        if iteration % 20 == 0 or iteration == max_iter:
            joint = u[:, None] * K * v[None, :]
            row_error = float(np.max(np.abs(joint.sum(axis=1) - r)))
            col_error = float(np.max(np.abs(joint.sum(axis=0) - c)))
            if max(row_error, col_error) <= tol:
                break

    joint = u[:, None] * K * v[None, :]
    joint /= joint.sum()
    row_error = float(np.max(np.abs(joint.sum(axis=1) - r)))
    col_error = float(np.max(np.abs(joint.sum(axis=0) - c)))
    if max(row_error, col_error) > max(100_000.0 * tol, 1e-6):
        # The log solver is slower but more stable for the rare extremely
        # ill-conditioned endpoint.
        return sinkhorn_log(
            L, r, c, max_iter=max_iter, tol=tol,
            initial_log_u=np.log(np.maximum(u, tiny)),
            initial_log_v=np.log(np.maximum(v, tiny)),
        )
    return SinkhornResult(
        joint, iteration, row_error, col_error,
        np.log(np.maximum(u, tiny)), np.log(np.maximum(v, tiny)),
    )

def channel_from_joint(joint: Array, p_x: Array) -> Array:
    p_x = normalize_probability(p_x, name="p_x")
    joint = np.asarray(joint, dtype=np.float64)
    if joint.shape[0] != p_x.size:
        raise ValueError("joint and p_x have incompatible shapes")
    channel = np.maximum(joint / p_x[:, None], _EPS)
    channel /= channel.sum(axis=1, keepdims=True)
    return channel


def joint_from_channel(p_x: Array, channel: Array) -> Array:
    p_x = normalize_probability(p_x, name="p_x")
    W = np.asarray(channel, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != p_x.size or np.any(W < 0):
        raise ValueError("invalid channel")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("channel rows must sum to one")
    return p_x[:, None] * W


def kl_divergence(p: Array, q: Array) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("invalid KL inputs")
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    value = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
    if value < -1e-10:
        raise FloatingPointError(f"computed a materially negative KL value: {value}")
    return max(value, 0.0)


def mutual_information(joint: Array) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    if np.any(joint < 0) or not np.isclose(joint.sum(), 1.0, atol=1e-7):
        raise ValueError("joint must be a probability matrix")
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    return kl_divergence(joint, p_x[:, None] * p_y[None, :])


def entropy(p: Array) -> float:
    p = normalize_probability(p)
    return float(-np.sum(p * np.log(p)))


def weighted_row_kl(p_x: Array, reference: Array, estimate: Array) -> float:
    p_x = normalize_probability(p_x, name="p_x")
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    return float(
        np.dot(
            p_x,
            [kl_divergence(reference[i], estimate[i]) for i in range(p_x.size)],
        )
    )


def weighted_row_tv(p_x: Array, reference: Array, estimate: Array) -> float:
    p_x = normalize_probability(p_x, name="p_x")
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    return float(0.5 * np.sum(p_x[:, None] * np.abs(reference - estimate)))


def rd_objective(joint: Array, distortion: Array, tau: float = 1.0) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    joint = np.asarray(joint, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    return mutual_information(joint) + float(np.sum(joint * distortion)) / tau


def fixed_output_rd(
    p_x: Array,
    p_y: Array,
    distortion: Array,
    *,
    beta: float = 1.0,
    max_iter: int = 30_000,
    tol: float = 1e-12,
    initial_log_u: Optional[Array] = None,
    initial_log_v: Optional[Array] = None,
) -> SinkhornResult:
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    return sinkhorn_log(
        -beta * np.asarray(distortion, dtype=np.float64),
        p_x,
        p_y,
        max_iter=max_iter,
        tol=tol,
        initial_log_u=initial_log_u,
        initial_log_v=initial_log_v,
    )


@dataclass(frozen=True)
class FreeRDResult:
    channel: Array
    output_marginal: Array
    joint: Array
    iterations: int
    residual: float


def free_output_rd(
    p_x: Array,
    distortion: Array,
    *,
    tau: float = 1.0,
    initial_output: Optional[Array] = None,
    max_iter: int = 20_000,
    tol: float = 1e-12,
) -> FreeRDResult:
    """Blahut-Arimoto for ``min I(X;Y)+E[d]/tau``."""
    p_x = normalize_probability(p_x, name="p_x")
    d = np.asarray(distortion, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != p_x.size or tau <= 0:
        raise ValueError("invalid free-output RD inputs")
    n_y = d.shape[1]
    q = np.full(n_y, 1.0 / n_y) if initial_output is None else normalize_probability(initial_output)
    if q.size != n_y:
        raise ValueError("initial_output has the wrong size")

    residual = np.inf
    channel = np.empty_like(d)
    for iteration in range(1, max_iter + 1):
        logits = np.log(np.maximum(q, _EPS))[None, :] - d / tau
        logits -= logsumexp(logits, axis=1, keepdims=True)
        channel = np.exp(logits)
        q_new = np.maximum(p_x @ channel, _EPS)
        q_new /= q_new.sum()
        residual = float(np.max(np.abs(q_new - q)))
        q = q_new
        if residual <= tol:
            break
    joint = p_x[:, None] * channel
    return FreeRDResult(channel, q, joint, iteration, residual)


def aggregate_joint(joint: Array, row_labels: Array, col_labels: Optional[Array] = None) -> Array:
    joint = np.asarray(joint, dtype=np.float64)
    row_labels = np.asarray(row_labels, dtype=int)
    col_labels = row_labels if col_labels is None else np.asarray(col_labels, dtype=int)
    if joint.shape != (row_labels.size, col_labels.size):
        raise ValueError("joint and labels have incompatible shapes")
    out = np.zeros((int(row_labels.max()) + 1, int(col_labels.max()) + 1))
    rr = np.broadcast_to(row_labels[:, None], joint.shape)
    cc = np.broadcast_to(col_labels[None, :], joint.shape)
    np.add.at(out, (rr.ravel(), cc.ravel()), joint.ravel())
    return out
