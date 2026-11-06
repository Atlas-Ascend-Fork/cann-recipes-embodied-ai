# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .hand_runtime import NativeHandRuntime
from .hand_spec import load_native_hand_specs

LOGGER = logging.getLogger(__name__)

TENSOR_KEYS = ("x_gesture", "x_gesture_norm", "valid_mask", "q", "q_norm")
STATIC_KEYS = (
    "q_mask", "q_lower", "q_upper", "morphology_vec",
    "joint_queries", "palm_radius", "joint_dim",
)


@dataclass(frozen=True)
class _SampleContext:
    runtime: Any
    name: str
    hand_id: int
    samples_per_hand: int
    fk_batch_size: int
    seed: int
    limit_shrink_ratio: float
    requested_device: torch.device
    tensor_chunks: dict[str, list[torch.Tensor]]
    hand_ids: list[torch.Tensor]


@dataclass(frozen=True)
class _BundleContext:
    output_path: Path
    hand_config: str | Path
    selected: Sequence[str]
    samples_per_hand: int
    seed: int
    limit_shrink_ratio: float
    hand_ids: list[torch.Tensor]
    tensor_chunks: dict[str, list[torch.Tensor]]
    static_by_hand: dict[str, dict[str, torch.Tensor]]


def _generate_hand_samples(context: _SampleContext) -> dict[str, torch.Tensor]:
    runtime = context.runtime
    name, hand_id = context.name, context.hand_id
    samples_per_hand, fk_batch_size = context.samples_per_hand, context.fk_batch_size
    seed, limit_shrink_ratio = context.seed, context.limit_shrink_ratio
    requested_device = context.requested_device
    tensor_chunks, hand_ids = context.tensor_chunks, context.hand_ids
    rng = np.random.default_rng(int(seed) + hand_id)
    generated = 0
    while generated < int(samples_per_hand):
        count = min(int(fk_batch_size), int(samples_per_hand) - generated)
        active_q = runtime.sample_active_q(
            count, rng, limit_shrink_ratio=float(limit_shrink_ratio)
        ).to(requested_device)
        gesture = runtime.kinematic_chain_gesture(active_q)
        joints = runtime.padded_joint_tensors(active_q)
        values = {
            "x_gesture": gesture.get("x_gesture"),
            "x_gesture_norm": gesture.get("x_gesture") / float(runtime.palm_radius),
            "valid_mask": gesture.get("valid_mask"),
            "q": joints.get("q"),
            "q_norm": joints.get("q_norm"),
        }
        if any(value is None for value in values.values()):
            raise KeyError(f"kinematic chain output is missing a required tensor for {name}")
        for key in TENSOR_KEYS:
            tensor = values.get(key)
            if tensor is None:
                raise KeyError(f"kinematic chain output is missing {key} for {name}")
            tensor_chunks[key].append(tensor.detach().cpu())
        hand_ids.append(torch.full((count,), hand_id, dtype=torch.int64))
        generated += count
        LOGGER.info(f"[data] hand={name} {generated}/{samples_per_hand}")
    return {
        "q_mask": runtime.q_mask.detach().cpu().clone(),
        "q_lower": runtime.q_lower.detach().cpu().clone(),
        "q_upper": runtime.q_upper.detach().cpu().clone(),
        "morphology_vec": runtime.morphology_vec.detach().cpu().clone(),
        "joint_queries": runtime.joint_queries.detach().cpu().clone(),
        "palm_radius": torch.tensor([runtime.palm_radius], dtype=torch.float32),
        "joint_dim": torch.tensor(len(runtime.spec.active_joint_names), dtype=torch.int64),
    }


def _save_tensor_bundle(context: _BundleContext) -> None:
    output_path = context.output_path
    hand_config, selected = context.hand_config, context.selected
    samples_per_hand, seed = context.samples_per_hand, context.seed
    limit_shrink_ratio = context.limit_shrink_ratio
    hand_ids, tensor_chunks = context.hand_ids, context.tensor_chunks
    static_by_hand = context.static_by_hand
    payload = {
        "metadata": {
            "format": "native_tensor_bundle_v2", "hand_config": str(Path(hand_config)),
            "hand_names": list(selected), "samples_per_hand": int(samples_per_hand),
            "num_samples": int(samples_per_hand) * len(selected), "seed": int(seed),
            "limit_shrink_ratio": float(limit_shrink_ratio),
            "sampling": "50% independent, 25% stratified, 25% finger-correlated",
        },
        "hand_ids": torch.cat(hand_ids, dim=0),
        "tensors": {key: torch.cat(chunks, dim=0) for key, chunks in tensor_chunks.items()},
        "static_by_hand": static_by_hand,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def generate_tensor_bundle(
    *,
    hand_config: str | Path,
    output: str | Path,
    samples_per_hand: int,
    seed: int,
    fk_batch_size: int = 2048,
    limit_shrink_ratio: float = 0.95,
    device: str | torch.device = "cpu",
    hand_names: Sequence[str] | None = None,
) -> Path:
    """Generate deterministic random Native-URDF poses and batched-FK features."""
    specs = load_native_hand_specs(hand_config)
    selected = tuple(hand_names or specs.keys())
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise KeyError(f"Unknown hands in generation request: {unknown}")
    if samples_per_hand <= 0 or fk_batch_size <= 0:
        raise ValueError("samples_per_hand and fk_batch_size must be positive")
    tensor_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in TENSOR_KEYS}
    static_by_hand: dict[str, dict[str, torch.Tensor]] = {}
    hand_ids: list[torch.Tensor] = []
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and (not torch.cuda.is_available()):
        requested_device = torch.device("cpu")
    for hand_id, name in enumerate(selected):
        runtime = NativeHandRuntime.build(specs[name], device=requested_device)
        static_by_hand[name] = _generate_hand_samples(_SampleContext(
            runtime, name, hand_id, samples_per_hand, fk_batch_size, seed,
            limit_shrink_ratio, requested_device, tensor_chunks, hand_ids,
        ))
    output_path = Path(output)
    _save_tensor_bundle(_BundleContext(
        output_path, hand_config, selected, samples_per_hand, seed,
        limit_shrink_ratio, hand_ids, tensor_chunks, static_by_hand,
    ))
    LOGGER.info(f"[done] {output_path}: {int(samples_per_hand) * len(selected)} samples")
    return output_path


class NativeTensorBundleDataset(Dataset):
    """Compact random-q dataset shared by training and validation."""

    def __init__(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.metadata: Mapping[str, Any] = payload["metadata"]
        self.tensors: Mapping[str, torch.Tensor] = payload["tensors"]
        self.static_by_hand: Mapping[str, Mapping[str, torch.Tensor]] = payload["static_by_hand"]
        if "hand_ids" in payload:
            setattr(self, "hand_names", tuple(self.metadata["hand_names"]))
            setattr(self, "hand_ids", torch.as_tensor(payload["hand_ids"], dtype=torch.int64))
        else:
            legacy_names = list(payload["hand_names"])
            self.hand_names = tuple(dict.fromkeys(legacy_names))
            lookup = {name: index for index, name in enumerate(self.hand_names)}
            self.hand_ids = torch.tensor([lookup[name] for name in legacy_names], dtype=torch.int64)
        lengths = {int(value.shape[0]) for value in self.tensors.values()}
        if lengths != {len(self.hand_ids)}:
            raise ValueError(f"Inconsistent tensor bundle lengths: {sorted(lengths)} vs {len(self.hand_ids)}")
        indices: dict[str, list[int]] = defaultdict(list)
        for index, hand_id in enumerate(self.hand_ids.tolist()):
            indices[self.hand_names[hand_id]].append(index)
        self.hand_indices = dict(indices)

    def __len__(self) -> int:
        return len(self.hand_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        hand_name = self.hand_names[int(self.hand_ids[index])]
        output = {key: value[index].clone() for key, value in self.tensors.items()}
        output.update({key: value.clone() for key, value in self.static_by_hand[hand_name].items()})
        output["hand_name"] = hand_name
        return output
