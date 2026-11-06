# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .api import _build_model
from .dataset import NativeTensorBundleDataset
from .losses import NativeHandBank, compute_native_vae_loss

LOGGER = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOSS_KEYS = (
    "loss",
    "loss_q",
    "loss_tip",
    "loss_cross_tip_abs",
    "loss_cross_pair_vector",
    "loss_cross_pair_distance",
    "loss_kl_action",
    "loss_kl_morphology",
)


@dataclass(frozen=True)
class _BatchContext:
    model: object
    batch: Mapping[str, Any]
    hand_bank: NativeHandBank
    device: torch.device
    config: Mapping[str, Any]
    epoch: int
    optimizer: object


@dataclass(frozen=True)
class _TrainOverrides:
    device: str | None = None
    epochs: int | None = None
    max_steps: int | None = None
    max_train: int | None = None
    max_val: int | None = None
    wandb: bool | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def package_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PACKAGE_ROOT / value


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def active_cross_weights(config: Mapping[str, Any], epoch: int) -> tuple[float, float, float]:
    if epoch < int(config["cross_loss_start_epoch"]):
        return (0.0, 0.0, 0.0)
    return (
        float(config["lambda_cross_abs"]),
        float(config["lambda_cross_pair_vector"]),
        float(config["lambda_cross_pair_distance"]),
    )


def loss_kwargs(config: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    cross_abs, pair_vector, pair_distance = active_cross_weights(config, epoch)
    return {
        "lambda_q": float(config["lambda_q"]),
        "lambda_tip": float(config["lambda_tip"]),
        "lambda_cross_abs": cross_abs,
        "lambda_cross_pair_vector": pair_vector,
        "lambda_cross_pair_distance": pair_distance,
        "beta_action": float(config["beta_action"]),
        "beta_morphology": float(config["beta_morphology"]),
        "finger_weights_cross_abs": tuple(config["finger_weights_cross_abs"]),
    }


def build_loader(
    path: Path, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool
) -> tuple[NativeTensorBundleDataset, DataLoader]:
    dataset = NativeTensorBundleDataset(path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return (dataset, loader)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val: float,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(config.get("_global_step", 0)),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": float(best_val),
            "config": dict(config),
        },
        path,
    )


def _batch_model_kwargs(batch):
    return {key: batch[key] for key in ("hand_name", "joint_queries", "q_lower", "q_upper", "q_mask")}


def _run_batch(context: _BatchContext) -> dict[str, torch.Tensor]:
    model, batch = context.model, to_device(context.batch, context.device)
    hand_bank, device = context.hand_bank, context.device
    config, epoch, optimizer = context.config, context.epoch, context.optimizer
    output = model(
        batch["x_gesture_norm"], batch["morphology_vec"],
        **_batch_model_kwargs(batch),
    )
    losses = compute_native_vae_loss(
        model=model, output=output, batch=batch, hand_bank=hand_bank,
        **loss_kwargs(config, epoch),
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_clip = float(config.get("grad_clip_norm", 0.0))
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return losses


def run_epoch(
    *, model, loader: DataLoader, hand_bank: NativeHandBank, device: torch.device,
    config: Mapping[str, Any], epoch: int, optimizer: torch.optim.Optimizer | None, max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            losses = _run_batch(_BatchContext(model, batch, hand_bank, device, config, epoch, optimizer))
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach())
            count += 1
            if max_batches > 0 and count >= max_batches:
                break
    if count == 0:
        raise RuntimeError("DataLoader produced no batches")
    return {**{key: value / count for key, value in totals.items()}, "batches": count}


def maybe_wandb(config: Mapping[str, Any], run_dir: Path):
    if not bool(config.get("wandb_enabled", False)):
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("wandb_enabled=true, but wandb is not installed") from error
    return wandb.init(
        project=str(config.get("project_name", "Native-URDF-VAE")),
        name=str(config["run_name"]),
        group=config.get("wandb_group"),
        tags=config.get("wandb_tags"),
        config=dict(config),
        dir=str(run_dir),
    )



def _apply_train_overrides(config, overrides: _TrainOverrides):
    if overrides.device is not None:
        config["device"] = overrides.device
    if overrides.epochs is not None:
        config["num_epochs"] = int(overrides.epochs)
    if overrides.max_steps is not None:
        if overrides.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        config["max_steps"] = int(overrides.max_steps)
    if overrides.max_train is not None:
        config["max_train_batches"] = int(overrides.max_train)
    if overrides.max_val is not None:
        config["max_val_batches"] = int(overrides.max_val)
    if overrides.wandb is not None:
        config["wandb_enabled"] = bool(overrides.wandb)


def _prepare_train_state(config):
    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    requested = torch.device(str(config.get("device", "cuda")))
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = package_path(config.get("output_root", "runs")) / f"{config['run_name']}_{timestamp}"
    checkpoint_dir, log_dir = run_dir / "checkpoints", run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    train_dataset, train_loader = build_loader(
        package_path(config["train_data"]), batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config.get("num_workers", 0)), pin_memory=bool(config.get("pin_memory", True)),
    )
    val_dataset, val_loader = build_loader(
        package_path(config["val_data"]), batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config.get("num_workers", 0)), pin_memory=bool(config.get("pin_memory", True)),
    )
    if set(train_dataset.hand_names) != set(val_dataset.hand_names):
        raise ValueError("Train and validation bundles must contain the same hands")
    model_config = load_yaml(package_path(config["model_config"]))
    model = _build_model(model_config, device)
    hand_bank = NativeHandBank.build(package_path(config["hand_config"]), device, train_dataset.hand_names)
    optimizer = AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config.get("lr_decay_factor", 0.5)),
        patience=int(config.get("lr_patience", 10)), min_lr=float(config.get("min_lr", 1e-06)),
    )
    _resume_state(config, device, model, optimizer, scheduler)
    return {
        "config": config, "device": device, "run_dir": run_dir, "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir, "model_config": model_config, "model": model, "hand_bank": hand_bank,
        "optimizer": optimizer, "scheduler": scheduler, "train_loader": train_loader,
        "val_loader": val_loader, "start_epoch": int(config.get("_start_epoch", 1)),
        "best_val": float(config.get("_best_val", float("inf"))),
    }


def _resume_state(config, device, model, optimizer, scheduler):
    config["_start_epoch"], config["_best_val"] = 1, float("inf")
    config["_global_step"] = 0
    resume_path = config.get("resume_path")
    if not resume_path:
        return
    payload = torch.load(package_path(resume_path), map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    if bool(config.get("resume_optimizer", True)) and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        config["_start_epoch"] = int(payload.get("epoch", 0)) + 1
        config["_global_step"] = int(payload.get("global_step", 0))
        config["_best_val"] = float(payload.get("best_val", config["_best_val"]))


def _run_training_epochs(state, wandb_run):
    config, model, optimizer = state["config"], state["model"], state["optimizer"]
    scheduler, hand_bank, device = state["scheduler"], state["hand_bank"], state["device"]
    metrics_path = state["log_dir"] / "metrics.jsonl"
    num_epochs, best_val = int(config["num_epochs"]), state["best_val"]
    max_steps = int(config.get("max_steps", 0))
    state["last_epoch"] = state["start_epoch"] - 1
    for epoch in range(state["start_epoch"], num_epochs + 1):
        batch_limit = int(config.get("max_train_batches", 0))
        if max_steps > 0:
            remaining = max_steps - config["_global_step"]
            if remaining <= 0:
                break
            batch_limit = min(batch_limit, remaining) if batch_limit > 0 else remaining
        train_metrics = run_epoch(
            model=model, loader=state["train_loader"], hand_bank=hand_bank, device=device,
            config=config, epoch=epoch, optimizer=optimizer,
            max_batches=batch_limit,
        )
        config["_global_step"] += int(train_metrics.pop("batches"))
        state["last_epoch"] = epoch
        at_step_limit = max_steps > 0 and config["_global_step"] >= max_steps
        record = {"epoch": epoch, "global_step": config["_global_step"], "lr": float(optimizer.param_groups[0]["lr"]),
                  **{f"train/{key}": value for key, value in train_metrics.items()}}
        if epoch % int(config["val_interval"]) == 0 or epoch == num_epochs or at_step_limit:
            val_metrics = run_epoch(
                model=model, loader=state["val_loader"], hand_bank=hand_bank, device=device,
                config=config, epoch=epoch, optimizer=None,
                max_batches=int(config.get("max_val_batches", 0)),
            )
            scheduler.step(val_metrics["loss"])
            record.update({f"val/{key}": value for key, value in val_metrics.items()})
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(state["checkpoint_dir"] / "best.pt", model=model, optimizer=optimizer,
                                scheduler=scheduler, epoch=epoch, best_val=best_val, config=config)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)
        if epoch % int(config["save_interval"]) == 0:
            save_checkpoint(state["checkpoint_dir"] / f"epoch_{epoch:04d}.pt", model=model,
                            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                            best_val=best_val, config=config)
        if at_step_limit:
            break
    state["best_val"] = best_val


def _finalize_training(state, wandb_run):
    save_checkpoint(
        state["checkpoint_dir"] / "last.pt", model=state["model"], optimizer=state["optimizer"],
        scheduler=state["scheduler"], epoch=state["last_epoch"],
        best_val=state["best_val"], config=state["config"],
    )
    torch.save(
        {"model_state": state["model"].state_dict(), "model_config": state["model_config"],
         "epoch": state["last_epoch"], "global_step": state["config"]["_global_step"]},
        state["checkpoint_dir"] / "inference.pt",
    )
    if wandb_run is not None:
        wandb_run.finish()
    return state["run_dir"]


def train(
    config_path: str | Path,
    *,
    device_override: str | None = None,
    epochs_override: int | None = None,
    max_steps_override: int | None = None,
    max_train_batches_override: int | None = None,
    max_val_batches_override: int | None = None,
    wandb_enabled_override: bool | None = None,
) -> Path:
    config = load_yaml(config_path)
    _apply_train_overrides(
        config,
        _TrainOverrides(
            device=device_override,
            epochs=epochs_override,
            max_steps=max_steps_override,
            max_train=max_train_batches_override,
            max_val=max_val_batches_override,
            wandb=wandb_enabled_override,
        ),
    )
    state = _prepare_train_state(config)
    wandb_run = maybe_wandb(config, state["run_dir"])
    LOGGER.info(
        "[train] run=%s device=%s params=%s",
        state["run_dir"], state["device"], sum(p.numel() for p in state["model"].parameters()),
    )
    _run_training_epochs(state, wandb_run)
    return _finalize_training(state, wandb_run)
