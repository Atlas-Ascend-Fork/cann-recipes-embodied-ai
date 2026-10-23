# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check that unavailable hardware is skipped and actual failures remain failures."""

from contextlib import ExitStack
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_ascend_test


class DeviceRunnerTest(unittest.TestCase):
    def run_case(self, model, device=True, result=0, error=None):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"RL_SAR_TEST_MODEL": model}))
            stack.enter_context(patch.object(Path, "exists", return_value=device))
            stack.enter_context(patch("run_ascend_test.sys.argv", ["runner", "/test/rl_infer"]))
            stack.enter_context(patch.object(run_ascend_test, "LOGGER"))
            run = stack.enter_context(patch("run_ascend_test.subprocess.run"))
            run.return_value = subprocess.CompletedProcess([], result)
            run.side_effect = error
            code = run_ascend_test.main()
            return code, run.call_count

    def test_missing_model_is_skipped(self):
        self.assertEqual(self.run_case(""), (77, 0))

    def test_invalid_model_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = str(Path(directory) / "missing.om")
            self.assertEqual(self.run_case(model), (1, 0))

    def test_device_and_inference_results(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixture.om"
            model.touch()
            self.assertEqual(self.run_case(str(model), device=False), (77, 0))
            self.assertEqual(self.run_case(str(model), result=0), (0, 1))
            self.assertEqual(self.run_case(str(model), result=1), (1, 1))
            self.assertEqual(self.run_case(str(model), result=77), (1, 1))
            error = subprocess.TimeoutExpired("rl_infer", 120)
            self.assertEqual(self.run_case(str(model), error=error), (1, 1))


if __name__ == "__main__":
    unittest.main()
