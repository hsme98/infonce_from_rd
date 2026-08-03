from __future__ import annotations

import torch
from torch.nn import functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric SimCLR NT-Xent loss on one GPU.

    Each of the 2B representations treats its paired view as the unique
    positive and all other non-self representations as negatives.
    """
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("z1 and z2 must have the same [batch, dim] shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent requires batch size at least two")
    z = torch.cat([z1, z2], dim=0)
    logits = z @ z.T
    logits = logits / temperature
    diagonal = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)
