#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from native_vae import NativeVAE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EvaluationContext:
    checkpoint: Path
    dataset_path: Path
    hand_names: tuple[str, ...]
    hand_ids: torch.Tensor
    padded_q: torch.Tensor
    indices: dict[str, torch.Tensor]
    vae: NativeVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate same-hand reconstruction and all ordered cross-hand "
            "retargeting pairs on a NativeVAE tensor bundle."
                    )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise RuntimeError("Metric input is empty or non-finite")
    return {
        "mean": float(flat.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "median": float(np.median(flat)),
        "p95": float(np.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def _same_hand_metrics(
    vae, source_q: torch.Tensor, hand: str, batch_size: int
) -> dict[str, dict[str, float]]:
    q_errors = []
    tip_errors = []
    latent_values = []
    for offset in range(0, len(source_q), batch_size):
        reconstructed = vae.reconstruct(source_q[offset:offset + batch_size], hand)
        q_errors.append((reconstructed.target_q.cpu() - source_q[offset:offset + batch_size]).numpy())
        tip_errors.append(reconstructed.fingerpad_error.cpu().numpy())
        latent_values.append(reconstructed.z_gesture.cpu().numpy())
    q_error = np.concatenate(q_errors, axis=0)
    tip_error = np.concatenate(tip_errors, axis=0)
    latent = np.concatenate(latent_values, axis=0)
    return {
        "joint_error_rad": summary(q_error),
        "fingerpad_l2_m": summary(tip_error),
        "latent_abs": summary(np.abs(latent)),
    }


def _cross_hand_metrics(
    vae, source_q: torch.Tensor, source_hand: str, target_hand: str, batch_size: int
) -> dict[str, dict[str, float]]:
    source_radius = float(vae.runtimes[source_hand].palm_radius)
    target_radius = float(vae.runtimes[target_hand].palm_radius)
    raw_errors = []
    normalized_errors = []
    for offset in range(0, len(source_q), batch_size):
        retargeted = vae.retarget(source_q[offset:offset + batch_size], source_hand, target_hand)
        source_tips = retargeted.source_fingerpads.cpu().numpy()
        target_tips = retargeted.target_fingerpads.cpu().numpy()
        raw_errors.append(np.linalg.norm(target_tips - source_tips, axis=-1))
        normalized_errors.append(
            np.linalg.norm(
                target_tips / target_radius - source_tips / source_radius, axis=-1
            )
        )
    raw_error = np.concatenate(raw_errors, axis=0)
    normalized_error = np.concatenate(normalized_errors, axis=0)
    return {
        "fingerpad_l2_m": summary(raw_error),
        "fingerpad_l2_palm_normalized": summary(normalized_error),
    }


def _load_evaluation_context(args: argparse.Namespace) -> _EvaluationContext:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    if not checkpoint.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"checkpoint={checkpoint.is_file()} dataset={dataset_path.is_file()}")
    payload = torch.load(dataset_path, map_location="cpu")
    hand_names = tuple(payload["metadata"]["hand_names"])
    hand_ids = torch.as_tensor(payload["hand_ids"], dtype=torch.int64)
    padded_q = torch.as_tensor(payload["tensors"]["q"], dtype=torch.float32)
    indices = {
        hand: torch.nonzero(hand_ids == hand_id, as_tuple=False).flatten()
        for hand_id, hand in enumerate(hand_names)
    }
    vae = NativeVAE.from_pretrained(checkpoint=checkpoint, device=args.device)
    if set(hand_names) != set(vae.hand_names):
        raise RuntimeError(f"Dataset/checkpoint hands differ: {hand_names} vs {vae.hand_names}")
    return _EvaluationContext(checkpoint, dataset_path, hand_names, hand_ids, padded_q, indices, vae)


def _collect_evaluation_metrics(context: _EvaluationContext, batch_size: int):
    vae, hand_names, hand_ids = context.vae, context.hand_names, context.hand_ids
    padded_q, indices = context.padded_q, context.indices
    result = {
        "num_samples": int(len(hand_ids)),
        "samples_per_hand": {hand: int(len(indices[hand])) for hand in hand_names},
        "same_hand": {}, "cross_hand": {},
    }
    for source_hand in hand_names:
        joint_dim = len(vae.joint_names(source_hand))
        source_q = padded_q[indices[source_hand], :joint_dim]
        result["same_hand"][source_hand] = _same_hand_metrics(vae, source_q, source_hand, batch_size)
        for target_hand in hand_names:
            if target_hand == source_hand:
                continue
            pair = f"{source_hand}->{target_hand}"
            result["cross_hand"][pair] = _cross_hand_metrics(
                vae, source_q, source_hand, target_hand, batch_size
            )
    return result


def _run_evaluation(args: argparse.Namespace) -> None:
    context = _load_evaluation_context(args)
    checkpoint, dataset_path, vae = context.checkpoint, context.dataset_path, context.vae
    started = perf_counter()
    result = _collect_evaluation_metrics(context, args.batch_size)
    result.update({"checkpoint": str(checkpoint), "dataset": str(dataset_path), "device": str(vae.device)})
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result["elapsed_seconds"] = float(perf_counter() - started)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("[done] %s", output)


def main() -> None:
    _run_evaluation(parse_args())


if __name__ == "__main__":
    main()
