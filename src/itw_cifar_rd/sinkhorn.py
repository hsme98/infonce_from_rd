from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SinkhornResult:
    log_joint: torch.Tensor
    log_a: torch.Tensor
    log_b: torch.Tensor
    iterations: int
    marginal_error: float


def log_sinkhorn_uniform(
    log_kernel: torch.Tensor,
    *,
    max_iter: int = 1000,
    tol: float = 2e-6,
    check_every: int = 10,
    initial_log_a: Optional[torch.Tensor] = None,
    initial_log_b: Optional[torch.Tensor] = None,
) -> SinkhornResult:
    """Log-domain Sinkhorn scaling to equal uniform marginals."""
    if log_kernel.ndim != 2 or log_kernel.shape[0] != log_kernel.shape[1]:
        raise ValueError("log_kernel must be square")
    if not torch.isfinite(log_kernel).all():
        raise ValueError("log_kernel contains non-finite entries")
    n = log_kernel.shape[0]
    log_p = -torch.log(
        torch.tensor(float(n), device=log_kernel.device, dtype=log_kernel.dtype)
    )
    log_a = (
        torch.zeros(n, device=log_kernel.device, dtype=log_kernel.dtype)
        if initial_log_a is None
        else initial_log_a.to(log_kernel).clone()
    )
    log_b = (
        torch.zeros(n, device=log_kernel.device, dtype=log_kernel.dtype)
        if initial_log_b is None
        else initial_log_b.to(log_kernel).clone()
    )
    marginal_error = float("inf")
    for iteration in range(1, max_iter + 1):
        log_a = log_p - torch.logsumexp(log_kernel + log_b.unsqueeze(0), dim=1)
        log_b = log_p - torch.logsumexp(log_kernel + log_a.unsqueeze(1), dim=0)
        if iteration % 20 == 0:
            # Remove the arbitrary additive gauge without changing the joint.
            shift = 0.5 * (log_a.mean() - log_b.mean())
            log_a = log_a - shift
            log_b = log_b + shift
        if iteration % check_every == 0 or iteration == max_iter:
            log_row = log_a + torch.logsumexp(
                log_kernel + log_b.unsqueeze(0), dim=1
            )
            log_col = log_b + torch.logsumexp(
                log_kernel + log_a.unsqueeze(1), dim=0
            )
            target = torch.exp(log_p)
            error = torch.maximum(
                (torch.exp(log_row) - target).abs().max(),
                (torch.exp(log_col) - target).abs().max(),
            )
            marginal_error = float(error.item())
            if marginal_error <= tol:
                break
    log_joint = log_a.unsqueeze(1) + log_kernel + log_b.unsqueeze(0)
    if marginal_error > max(10.0 * tol, 1e-5):
        raise RuntimeError(
            f"Sinkhorn did not converge: error={marginal_error:.3e}, "
            f"iterations={iteration}"
        )
    return SinkhornResult(
        log_joint=log_joint,
        log_a=log_a,
        log_b=log_b,
        iterations=iteration,
        marginal_error=marginal_error,
    )
