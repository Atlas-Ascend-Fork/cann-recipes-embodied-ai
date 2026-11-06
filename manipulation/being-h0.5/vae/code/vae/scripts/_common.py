# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for root in (PROJECT_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_q(path: str | Path, key: str = "q") -> np.ndarray:
    path = Path(path)
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        candidates = (key, "q", "action", "joint_q")
        selected = next((name for name in candidates if name in payload), None)
        if selected is None:
            raise KeyError(f"{path} has keys {payload.files}, not {key!r}")
        value = payload[selected]
    else:
        value = payload
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"Expected q [B,D], got {value.shape}")
    return value


def as_numpy(value):
    return value.detach().cpu().numpy()


def _find_shared_script(relative: str) -> Path:
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / relative, parent / "common" / "code" / relative):
            if candidate.is_file():
                return candidate
    raise ImportError(f"shared utility not found: {relative}")


def run_shared_script(relative: str) -> None:
    """Execute a canonical VAE utility without changing its public CLI."""
    candidate = _find_shared_script(relative)
    if str(candidate.parent) not in sys.path:
        sys.path.insert(0, str(candidate.parent))
    runpy.run_path(str(candidate), run_name="__main__")
