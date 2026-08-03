from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import config_fingerprint
from .data import (
    CIFAR100WithCoarse,
    simclr_transform,
    supervised_train_transform,
)
from .io_utils import atomic_json_dump, seed_everything, state_dict_to_cpu, torch_load, write_csv
from .losses import nt_xent_loss
from .models import ModelSpec, build_model


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype in {"bf16", "fp16"}
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _save_checkpoint(
    path: Path,
    *,
    role: str,
    seed: int,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "role": role,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": state_dict_to_cpu(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "config_fingerprint": config_fingerprint(config),
        "torch_version": torch.__version__,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_resume(
    path: Path,
    *,
    role: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
) -> tuple[int, int]:
    checkpoint = torch_load(path, map_location="cpu")
    if checkpoint.get("role") != role or int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(f"checkpoint role/seed mismatch: {path}")
    if checkpoint.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError(
            "checkpoint configuration differs from the current configuration; "
            "use a new output directory"
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def train_one_role(
    config: Dict[str, Any],
    *,
    role: str,
    seed: int,
    data_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    download: bool = False,
    deterministic: bool = False,
) -> Dict[str, Any]:
    if role not in {"simclr", "supervised"}:
        raise ValueError("role must be simclr or supervised")
    seed_everything(seed, deterministic=deterministic)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "checkpoint_final.pt"
    latest_path = output_dir / "checkpoint_latest.pt"
    summary_path = output_dir / "train_summary.json"
    history_path = output_dir / "train_history.csv"
    if final_path.is_file() and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        expected_fingerprint = config_fingerprint(config)
        if (
            existing.get("config_fingerprint") != expected_fingerprint
            or existing.get("role") != role
            or int(existing.get("seed", -1)) != seed
        ):
            raise RuntimeError(
                f"existing completed output does not match this run: {output_dir}"
            )
        return {"status": "skipped", "final_checkpoint": str(final_path)}

    role_cfg = dict(config[role])
    data_cfg = config["data"]
    model_cfg = config["model"]
    spec = ModelSpec(
        projection_hidden_dim=int(model_cfg["projection_hidden_dim"]),
        projection_dim=int(model_cfg["projection_dim"]),
    )
    model = build_model(role, spec).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if role == "simclr":
        transform = simclr_transform()
    else:
        transform = supervised_train_transform()
    dataset = CIFAR100WithCoarse(
        root=data_root,
        train=True,
        transform=transform,
        download=download,
    )

    batch_size = int(role_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 8))
    loader_generator = torch.Generator().manual_seed(seed + 10_000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(role == "simclr"),
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=loader_generator,
    )

    epochs = int(role_cfg["epochs"])
    total_steps = epochs * len(loader)
    warmup_steps = int(float(role_cfg.get("warmup_epochs", 0)) * len(loader))
    base_lr = float(role_cfg["lr"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=float(role_cfg.get("momentum", 0.9)),
        weight_decay=float(role_cfg.get("weight_decay", 0.0)),
        nesterov=bool(role_cfg.get("nesterov", True)),
    )
    amp_dtype = str(role_cfg.get("amp_dtype", "bf16"))
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and amp_dtype == "fp16",
    )

    start_epoch = 0
    global_step = 0
    if latest_path.is_file():
        start_epoch, global_step = _load_resume(
            latest_path,
            role=role,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
        )
        model.to(device)

    history: list[dict[str, Any]] = []
    if history_path.is_file():
        import csv

        with history_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                history.append(dict(row))

    checkpoint_every = max(1, int(role_cfg.get("checkpoint_every", 25)))
    started = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        epoch_start = time.time()
        for batch in loader:
            lr = base_lr * _lr_multiplier(global_step, total_steps, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            if role == "simclr":
                (x1, x2), _, _, _ = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                batch_seen = int(x1.shape[0])
                images = torch.cat([x1, x2], dim=0)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with _autocast_context(device, amp_dtype):
                    z = model(images)["z"]
                    z1, z2 = z.split(batch_seen, dim=0)
                    loss = nt_xent_loss(
                        z1, z2, temperature=float(role_cfg["temperature"])
                    )
            else:
                images, fine, _, _ = batch
                images = images.to(device, non_blocking=True)
                fine = fine.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with _autocast_context(device, amp_dtype):
                    logits = model(images)["logits"]
                    loss = F.cross_entropy(
                        logits,
                        fine,
                        label_smoothing=float(role_cfg.get("label_smoothing", 0.0)),
                    )
                correct += int((logits.argmax(dim=1) == fine).sum().item())
                batch_seen = int(images.shape[0])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * batch_seen
            seen += batch_seen
            global_step += 1

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss_sum / max(1, seen),
            "train_accuracy": (correct / seen) if role == "supervised" else "",
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
            "examples": seen,
        }
        history.append(row)
        write_csv(history, history_path)
        print(
            f"[{role} seed={seed}] epoch {epoch + 1}/{epochs} "
            f"loss={float(row['loss']):.5f} lr={float(row['lr']):.3e} "
            f"time={float(row['seconds']):.1f}s",
            flush=True,
        )

        if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            _save_checkpoint(
                latest_path,
                role=role,
                seed=seed,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
            )

    _save_checkpoint(
        final_path,
        role=role,
        seed=seed,
        epoch=epochs - 1,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
    )
    latest_path.unlink(missing_ok=True)
    summary = {
        "status": "complete",
        "role": role,
        "seed": seed,
        "epochs": epochs,
        "global_step": global_step,
        "final_loss": float(history[-1]["loss"]),
        "final_train_accuracy": history[-1]["train_accuracy"],
        "seconds": time.time() - started,
        "final_checkpoint": str(final_path),
        "config_fingerprint": config_fingerprint(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json_dump(summary, summary_path)
    return summary
