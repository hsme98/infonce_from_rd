from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .config import config_fingerprint
from .data import CIFAR100WithCoarse, network_eval_transform, raw_eval_transform
from .io_utils import seed_everything, torch_load
from .models import ModelSpec, build_model


def _load_model(
    role: str,
    checkpoint_path: str | Path,
    *,
    spec: ModelSpec,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(role, spec)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("role") != role:
        raise RuntimeError(f"expected {role} checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def _extract_model_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    keys: tuple[str, ...],
    amp_dtype: str,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs: Dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    fine_parts: list[torch.Tensor] = []
    coarse_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and amp_dtype in {"bf16", "fp16"}
    with torch.inference_mode():
        for images, fine, coarse, indices in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(
                device_type=device.type, dtype=dtype, enabled=amp_enabled
            ):
                result = model(images)
            for key in keys:
                value = result[key]
                if key == "h":
                    value = F.normalize(value.float(), dim=1)
                else:
                    value = value.float()
                outputs[key].append(value.cpu())
            fine_parts.append(torch.as_tensor(fine, dtype=torch.long))
            coarse_parts.append(torch.as_tensor(coarse, dtype=torch.long))
            index_parts.append(torch.as_tensor(indices, dtype=torch.long))
    merged = {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}
    return (
        merged,
        torch.cat(fine_parts),
        torch.cat(coarse_parts),
        torch.cat(index_parts),
    )


def _extract_pixels(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    pixels: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for images, _, _, batch_indices in loader:
        pixels.append(images.flatten(1).float())
        indices.append(torch.as_tensor(batch_indices, dtype=torch.long))
    return torch.cat(pixels, dim=0), torch.cat(indices, dim=0)


def leave_one_out_centroid_accuracy(
    features: torch.Tensor, labels: torch.Tensor, *, num_classes: int
) -> float:
    """Cosine nearest-centroid accuracy with each sample removed from its class.

    The own-class centroid is corrected per sample without materializing a
    ``[batch, classes, dimension]`` tensor.  This matters for the pixel
    baseline, whose feature dimension is 3,072.
    """
    x = F.normalize(features.float(), dim=1)
    labels = labels.long()
    sums = torch.zeros(num_classes, x.shape[1], dtype=x.dtype)
    counts = torch.zeros(num_classes, dtype=x.dtype)
    sums.index_add_(0, labels, x)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=x.dtype))
    if torch.any(counts < 2):
        raise ValueError("leave-one-out centroids require at least two samples per class")

    full_centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)
    correct = 0
    for start in range(0, x.shape[0], 512):
        end = min(start + 512, x.shape[0])
        batch = x[start:end]
        batch_labels = labels[start:end]
        scores = batch @ full_centroids.T

        own_sums = sums[batch_labels] - batch
        own_counts = counts[batch_labels] - 1.0
        own_centroids = F.normalize(own_sums / own_counts.unsqueeze(1), dim=1)
        row_ids = torch.arange(end - start)
        scores[row_ids, batch_labels] = (batch * own_centroids).sum(dim=1)
        correct += int((scores.argmax(dim=1) == batch_labels).sum().item())
    return correct / x.shape[0]


def representation_diagnostics(features: torch.Tensor, *, seed: int) -> Dict[str, float | None]:
    x = features.float()
    normalized = F.normalize(x, dim=1)
    generator = torch.Generator().manual_seed(seed)
    count = min(50_000, max(1, x.shape[0] * 20))
    i = torch.randint(0, x.shape[0], (count,), generator=generator)
    j = torch.randint(0, x.shape[0] - 1, (count,), generator=generator)
    j = j + (j >= i).long()
    cosines = (normalized[i] * normalized[j]).sum(dim=1)
    effective_rank: float | None = None
    if x.shape[1] <= 1024:
        centered = x - x.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, x.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
        total = eigenvalues.sum()
        if total > 0:
            probabilities = eigenvalues[eigenvalues > 0] / total
            effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()).item())
    return {
        "mean_coordinate_std": float(x.std(dim=0, unbiased=False).mean().item()),
        "mean_pair_cosine": float(cosines.mean().item()),
        "std_pair_cosine": float(cosines.std(unbiased=False).item()),
        "effective_rank": effective_rank,
    }

def extract_all_features(
    config: Dict[str, Any],
    *,
    seed: int,
    data_root: str | Path,
    simclr_checkpoint: str | Path,
    supervised_checkpoint: str | Path,
    output_path: str | Path,
    device: torch.device,
    download: bool = False,
) -> Dict[str, Any]:
    seed_everything(seed)
    data_cfg = config["data"]
    model_cfg = config["model"]
    rd_cfg = config["rd"]
    spec = ModelSpec(
        projection_hidden_dim=int(model_cfg["projection_hidden_dim"]),
        projection_dim=int(model_cfg["projection_dim"]),
    )

    base_network = CIFAR100WithCoarse(
        root=data_root,
        train=False,
        transform=network_eval_transform(),
        download=download,
    )
    from .data import balanced_fine_subset_indices

    subset_indices = balanced_fine_subset_indices(
        base_network.targets,
        per_class=int(data_cfg["eval_per_fine_class"]),
        seed=int(data_cfg["eval_subset_seed"]),
    )
    network_subset = Subset(base_network, subset_indices.tolist())
    raw_dataset = CIFAR100WithCoarse(
        root=data_root,
        train=False,
        transform=raw_eval_transform(),
        download=False,
    )
    raw_subset = Subset(raw_dataset, subset_indices.tolist())
    loader_kwargs = dict(
        batch_size=int(rd_cfg.get("feature_batch_size", 512)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=device.type == "cuda" and bool(data_cfg.get("pin_memory", True)),
    )
    network_loader = DataLoader(network_subset, **loader_kwargs)
    raw_loader = DataLoader(raw_subset, **loader_kwargs)

    simclr = _load_model(
        "simclr", simclr_checkpoint, spec=spec, device=device
    )
    simclr_outputs, fine, coarse, indices = _extract_model_outputs(
        simclr,
        network_loader,
        device=device,
        keys=("h", "z"),
        amp_dtype=str(config["simclr"].get("amp_dtype", "bf16")),
    )
    del simclr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    supervised = _load_model(
        "supervised", supervised_checkpoint, spec=spec, device=device
    )
    supervised_outputs, fine2, coarse2, indices2 = _extract_model_outputs(
        supervised,
        network_loader,
        device=device,
        keys=("h",),
        amp_dtype=str(config["supervised"].get("amp_dtype", "bf16")),
    )
    del supervised
    if not torch.equal(fine, fine2) or not torch.equal(coarse, coarse2) or not torch.equal(indices, indices2):
        raise RuntimeError("feature extraction loaders produced inconsistent ordering")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Recreate the untrained SimCLR model from a deterministic seed.  This is
    # the architecture-matched random-feature baseline.
    seed_everything(seed)
    random_model = build_model("simclr", spec).to(device).eval()
    random_outputs, fine3, coarse3, indices3 = _extract_model_outputs(
        random_model,
        network_loader,
        device=device,
        keys=("z",),
        amp_dtype="fp32",
    )
    del random_model
    if not torch.equal(fine, fine3) or not torch.equal(coarse, coarse3) or not torch.equal(indices, indices3):
        raise RuntimeError("random feature ordering mismatch")

    pixels, pixel_indices = _extract_pixels(raw_loader)
    if not torch.equal(indices, pixel_indices):
        raise RuntimeError("pixel and network feature ordering mismatch")

    features = {
        "simclr_z": simclr_outputs["z"],
        "simclr_h": simclr_outputs["h"],
        "supervised_h": supervised_outputs["h"],
        "random_z": random_outputs["z"],
        "pixel": pixels,
    }
    diagnostics: Dict[str, Dict[str, float]] = {}
    for name, value in features.items():
        if name == "pixel":
            feature_for_diag = F.normalize(value.float(), dim=1)
        else:
            feature_for_diag = value
        diagnostics[name] = {
            "fine_leave_one_out_centroid_accuracy": leave_one_out_centroid_accuracy(
                feature_for_diag, fine, num_classes=100
            ),
            "coarse_leave_one_out_centroid_accuracy": leave_one_out_centroid_accuracy(
                feature_for_diag, coarse, num_classes=20
            ),
            **representation_diagnostics(feature_for_diag, seed=seed + len(diagnostics)),
        }

    payload = {
        "seed": int(seed),
        "config_fingerprint": config_fingerprint(config),
        "indices": indices,
        "fine": fine,
        "coarse": coarse,
        "features": features,
        "diagnostics": diagnostics,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "feature_path": str(output_path),
        "num_examples": int(indices.numel()),
        "diagnostics": diagnostics,
    }
