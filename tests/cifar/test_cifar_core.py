from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from itw_cifar_rd.data import balanced_fine_subset_indices
from itw_cifar_rd.losses import nt_xent_loss
from itw_cifar_rd.features import leave_one_out_centroid_accuracy
from itw_cifar_rd.models import ModelSpec, build_model
from itw_cifar_rd.rd_metrics import rd_metrics, rate_at_retention
from itw_cifar_rd.sinkhorn import log_sinkhorn_uniform


def test_models_and_nt_xent() -> None:
    model = build_model("simclr", ModelSpec(64, 16))
    images = torch.randn(8, 3, 32, 32)
    out = model(images)
    assert out["h"].shape == (8, 512)
    assert out["z"].shape == (8, 16)
    assert torch.allclose(out["z"].norm(dim=1), torch.ones(8), atol=1e-5)
    loss = nt_xent_loss(out["z"], torch.flip(out["z"], dims=[0]), 0.2)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0


def test_balanced_subset() -> None:
    labels = np.repeat(np.arange(5), 10)
    indices = balanced_fine_subset_indices(labels, per_class=3, seed=7, num_classes=5)
    selected = labels[indices]
    counts = np.bincount(selected, minlength=5)
    assert np.all(counts == 3)
    assert len(set(indices.tolist())) == 15


def test_sinkhorn_and_semantic_metrics() -> None:
    n = 12
    fine = torch.arange(n) % 6
    coarse = fine // 2
    points = torch.randn(n, 4)
    cost = torch.cdist(points, points).pow(2)
    result = log_sinkhorn_uniform(-2.0 * cost, max_iter=500, tol=1e-6)
    joint = torch.exp(result.log_joint)
    target = torch.full((n,), 1.0 / n)
    assert torch.max(torch.abs(joint.sum(1) - target)) < 2e-5
    assert torch.max(torch.abs(joint.sum(0) - target)) < 2e-5
    metrics = rd_metrics(result.log_joint, cost, fine, coarse)
    assert 0 <= metrics["rate"] <= math.log(n) + 1e-3
    for key in ("coarse_retention", "fine_retention", "fine_increment_retention"):
        assert -1e-6 <= metrics[key] <= 1.0 + 1e-5


def test_rate_threshold() -> None:
    value = rate_at_retention([0, 1, 2], [0, 0.5, 1.0], 0.8)
    assert abs(value - 1.6) < 1e-12


def test_semantic_hierarchy_metrics() -> None:
    fine = torch.tensor([0, 1, 2, 3])
    coarse = torch.tensor([0, 0, 1, 1])
    cost = torch.zeros(4, 4)

    identity = torch.eye(4) / 4.0
    identity_metrics = rd_metrics(identity.log(), cost, fine, coarse)
    assert abs(identity_metrics["coarse_retention"] - 1.0) < 1e-10
    assert abs(identity_metrics["fine_retention"] - 1.0) < 1e-10
    assert abs(identity_metrics["fine_increment_retention"] - 1.0) < 1e-10

    coarse_only = torch.zeros(4, 4)
    coarse_only[:2, :2] = 1.0 / 8.0
    coarse_only[2:, 2:] = 1.0 / 8.0
    coarse_metrics = rd_metrics(coarse_only.clamp_min(1e-30).log(), cost, fine, coarse)
    assert abs(coarse_metrics["coarse_retention"] - 1.0) < 1e-10
    assert abs(coarse_metrics["fine_retention"] - 0.5) < 1e-10
    assert abs(coarse_metrics["fine_increment_retention"]) < 1e-10


def test_leave_one_out_centroid_accuracy() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    accuracy = leave_one_out_centroid_accuracy(features, labels, num_classes=2)
    assert accuracy == 1.0
