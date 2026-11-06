#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""Lightweight repository smoke test for the Being-H05 integration layer."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-config", action="store_true", help="Import configs.data_config")
    parser.add_argument("--require-local-data", action="store_true", help="Require downloaded dataset metadata")
    args = parser.parse_args()
    required = [
        ROOT / "BeingH/train/train.py",
        ROOT / "BeingH/inference/beingh_policy.py",
        ROOT / "scripts/train/train_shadow_grasp_npu.sh",
        ROOT / "vae/scripts/run_tissuebox_policy_demo.sh",
        ROOT / "vae/examples/beingh_shadow_grasp_eval.py",
        ROOT / "vae/integration/beingh_shadow_grasp_eval.py",
        ROOT / "configs/data_config.py",
        ROOT / "configs/dataset_info.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))
    dataset = ROOT / "data/shadow_grasp_bottle22249179_aug100_2cam"
    if args.require_local_data:
        for relative in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/stats.json"):
            if not (dataset / relative).is_file():
                raise SystemExit(f"Missing local dataset metadata: {dataset / relative}")
    elif not dataset.exists():
        LOGGER.info("Dataset not present; source-only smoke test continues.")
    # Shell syntax is checked by the repository CI shell step. This Python
    # smoke test intentionally avoids launching a shell command.
    if args.import_config:
        module = importlib.import_module("configs.data_config")
        names = set(getattr(module, "DATA_CONFIG_MAP", {}).keys())
        expected = {
            "shadow_grasp_wrist_gesture",
            "shadow_grasp_wrist_gesture_q99",
            "shadow_grasp_2cam_wrist_rot6d_minmax_gesture_raw",
        }
        missing_configs = sorted(expected - names)
        if missing_configs:
            raise SystemExit(f"Missing expected data configs: {missing_configs}")
    LOGGER.info("Being-H05 smoke test passed (repository/config/entrypoint checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
