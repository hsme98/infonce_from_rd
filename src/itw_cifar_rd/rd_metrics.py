from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import torch
from torch.nn import functional as F


def entropy_from_labels(labels: torch.Tensor, num_classes: int) -> float:
    counts = torch.bincount(labels.long(), minlength=num_classes).double()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())


def label_joint(
    joint: torch.Tensor, labels: torch.Tensor, *, num_classes: int
) -> torch.Tensor:
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(
        device=joint.device, dtype=joint.dtype
    )
    return one_hot.T @ joint @ one_hot


def mutual_information_from_joint(joint: torch.Tensor) -> float:
    p = joint.double()
    p = p / p.sum()
    row = p.sum(dim=1, keepdim=True)
    col = p.sum(dim=0, keepdim=True)
    mask = p > 0
    value = (p[mask] * (p[mask].log() - (row @ col)[mask].log())).sum()
    return max(float(value.item()), 0.0)


def _fine_to_coarse_map(fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    mapping = torch.full((100,), -1, dtype=torch.long)
    for fine_label, coarse_label in zip(fine.cpu().tolist(), coarse.cpu().tolist()):
        previous = int(mapping[fine_label])
        if previous not in {-1, coarse_label}:
            raise ValueError("a fine class maps to multiple coarse classes")
        mapping[fine_label] = coarse_label
    return mapping


def conditional_fine_mi_given_coarse_pair(
    fine_joint: torch.Tensor, fine_to_coarse: torch.Tensor
) -> float:
    """Compute I(F;F' | C,C') from the 100x100 fine-label joint."""
    p = (fine_joint.double() / fine_joint.double().sum()).cpu()
    mapping = fine_to_coarse.cpu()
    total = 0.0
    for c in range(20):
        rows = torch.nonzero(mapping == c, as_tuple=False).flatten()
        for cp in range(20):
            cols = torch.nonzero(mapping == cp, as_tuple=False).flatten()
            block = p[rows][:, cols]
            mass = float(block.sum().item())
            if mass <= 0:
                continue
            conditional = block / mass
            row = conditional.sum(dim=1, keepdim=True)
            col = conditional.sum(dim=0, keepdim=True)
            product = row @ col
            mask = conditional > 0
            block_mi = (
                conditional[mask]
                * (conditional[mask].log() - product[mask].log())
            ).sum()
            total += mass * float(block_mi.item())
    return max(total, 0.0)


def rd_metrics(
    log_joint: torch.Tensor,
    cost: torch.Tensor,
    fine: torch.Tensor,
    coarse: torch.Tensor,
) -> Dict[str, float]:
    joint = torch.exp(log_joint).float()
    joint = joint / joint.sum()
    n = joint.shape[0]
    rate = (joint.double() * (log_joint.double() + 2.0 * math.log(n))).sum()
    distortion = (joint.double() * cost.double()).sum()
    coarse_joint = label_joint(joint, coarse.to(joint.device), num_classes=20)
    fine_joint = label_joint(joint, fine.to(joint.device), num_classes=100)
    coarse_mi = mutual_information_from_joint(coarse_joint)
    fine_mi = mutual_information_from_joint(fine_joint)
    fine_to_coarse = _fine_to_coarse_map(fine, coarse)
    fine_increment = conditional_fine_mi_given_coarse_pair(
        fine_joint, fine_to_coarse
    )
    h_coarse = entropy_from_labels(coarse.cpu(), 20)
    h_fine = entropy_from_labels(fine.cpu(), 100)
    increment_entropy = max(h_fine - h_coarse, 1e-12)
    return {
        "rate": max(float(rate.item()), 0.0),
        "distortion": float(distortion.item()),
        "coarse_mi": coarse_mi,
        "fine_mi": fine_mi,
        "fine_increment_mi": fine_increment,
        "coarse_retention": coarse_mi / max(h_coarse, 1e-12),
        "fine_retention": fine_mi / max(h_fine, 1e-12),
        "fine_increment_retention": fine_increment / increment_entropy,
        "coarse_entropy": h_coarse,
        "fine_entropy": h_fine,
    }


def rate_at_retention(
    rates: Iterable[float], retentions: Iterable[float], target: float
) -> float:
    rates_array = np.asarray(list(rates), dtype=np.float64)
    values = np.asarray(list(retentions), dtype=np.float64)
    order = np.argsort(rates_array)
    rates_array = rates_array[order]
    values = np.maximum.accumulate(values[order])
    if values[-1] < target:
        return float("nan")
    index = int(np.searchsorted(values, target, side="left"))
    if index == 0:
        return float(rates_array[0])
    x0, x1 = values[index - 1], values[index]
    r0, r1 = rates_array[index - 1], rates_array[index]
    if x1 <= x0 + 1e-12:
        return float(r1)
    weight = (target - x0) / (x1 - x0)
    return float(r0 + weight * (r1 - r0))
