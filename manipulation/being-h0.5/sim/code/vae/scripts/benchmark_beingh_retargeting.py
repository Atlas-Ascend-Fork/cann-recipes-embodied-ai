# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the cross-hand inference benchmark."""

import sys
from pathlib import Path

from vae.integration.benchmark_beingh_retargeting import main

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    main()
