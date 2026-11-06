# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

STATE_KEYS = ("observation.state", "action")
WRIST_DIM, SHADOW_JOINT_DIM, Z_DIM = 6, 22, 24
PRESERVED_DIM = WRIST_DIM + SHADOW_JOINT_DIM
EXPECTED_DIM = PRESERVED_DIM + Z_DIM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_in_batches(vae, joints: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = [
        vae.encode(joints[start:start + batch_size], "shadow_hand_right").detach().cpu().numpy().astype(np.float32)
        for start in range(0, len(joints), batch_size)
    ]
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(joints), Z_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid encoded z_gesture shape/values: {result.shape}")
    return result
