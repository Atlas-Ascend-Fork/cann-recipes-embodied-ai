#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
shared = root / "beingh_shared_scripts" / "train" / "smoke_fsdp_accelerator.py"
sys.path.insert(0, str(shared.parent))
runpy.run_path(str(shared), run_name="__main__")
