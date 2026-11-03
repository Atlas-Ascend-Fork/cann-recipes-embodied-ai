# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from rasterization.config import Config


class TestStepsScaler(unittest.TestCase):
    def test_small_positive_scale_keeps_sh_schedule_interval_valid(self):
        cfg = Config()
        cfg.adjust_steps(0.0005)

        self.assertEqual(cfg.max_steps, 15)
        self.assertEqual(min(0 // cfg.sh_degree_interval, cfg.sh_degree), 0)
        self.assertGreater(cfg.sh_degree_interval, 0)

    def test_default_scale_preserves_sh_schedule(self):
        cfg = Config()
        cfg.adjust_steps(1.0)

        self.assertEqual(cfg.max_steps, 30_000)
        self.assertEqual(cfg.sh_degree_interval, 1_000)
        self.assertEqual(min(1_000 // cfg.sh_degree_interval, cfg.sh_degree), 1)

    def test_non_positive_scale_is_rejected(self):
        for factor in (0, -1):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "steps_scaler must be greater than zero"):
                    Config().adjust_steps(factor)

    def test_scale_that_rounds_max_steps_to_zero_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_steps must remain positive"):
            Config().adjust_steps(1e-5)


if __name__ == "__main__":
    unittest.main()
