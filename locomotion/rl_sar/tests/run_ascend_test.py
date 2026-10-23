# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run real device inference; missing external prerequisites are CTest skips."""

import logging
import os
from pathlib import Path
import subprocess
import sys


LOGGER = logging.getLogger(__name__)


def main():
    model = os.environ.get("RL_SAR_TEST_MODEL", "")
    if not model:
        LOGGER.warning("SKIP: set RL_SAR_TEST_MODEL to a matching G1 BeyondDance .om model")
        return 77
    if not Path(model).is_file() or Path(model).suffix != ".om":
        LOGGER.error("FAIL: RL_SAR_TEST_MODEL must name an existing .om file")
        return 1
    if not Path("/dev/davinci0").exists():
        LOGGER.warning("SKIP: Ascend device /dev/davinci0 is unavailable")
        return 77
    try:
        result = subprocess.run(
            [sys.argv[1], "--model", model, "--steps", "3"],
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.error("FAIL: could not complete Ascend inference: %s", error)
        return 1
    # Only this runner may return 77. A launched inference failure is not a skip.
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
