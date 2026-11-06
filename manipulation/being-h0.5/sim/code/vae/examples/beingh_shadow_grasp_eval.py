# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the Being-H0.5 VAE MuJoCo evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

from vae.integration.beingh_shadow_grasp_eval import main

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    main()
