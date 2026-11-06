# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
import torch
from BeingH.npu_prefix_torchair import NPUStaticPrefixTorchAirRunner


class PrefixFallbackTests(unittest.TestCase):
    def test_cpu_inputs_return_explicit_fallback_without_compiling(self):
        target = torch.nn.Identity()
        runner = NPUStaticPrefixTorchAirRunner(target)
        value = torch.zeros(1)
        result = runner.forward(
            packed_sequence_und=value, packed_sequence_gen=value,
            packed_position_ids=value, packed_und_token_indexes=value,
            packed_gen_token_indexes=value, attention_mask=value, feature_flags={},
        )
        self.assertIsNone(result)
        self.assertEqual(runner.last_fallback_reason, "requires_npu_inputs")
        self.assertEqual(runner.stats()["compile_count"], 0)


if __name__ == "__main__":
    unittest.main()
