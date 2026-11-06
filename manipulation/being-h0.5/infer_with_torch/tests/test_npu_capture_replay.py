# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from BeingH.npu_capture_replay import NPUActionSuffixGraphRunner, _WarmupBatch


class GraphSessionTests(unittest.TestCase):
    def test_capture_warmup_preserves_prefix_and_workspace(self):
        class Target(torch.nn.Module):
            def forward_action_with_prefix_cache(
                self, *, action_sequence, packed_position_ids, attention_mask, prefix_cache
            ):
                self.inputs = (packed_position_ids, attention_mask, prefix_cache)
                return action_sequence + prefix_cache["layers"][0][0]

        target = Target()
        runner = NPUActionSuffixGraphRunner(target)
        action = torch.ones(2, 3)
        positions = torch.zeros(2, dtype=torch.int64)
        mask = torch.ones(2, 2)
        prefix = (torch.full((2, 3), 2.0), torch.zeros(2, 3))
        workspace = (torch.zeros(4, 3), torch.zeros(4, 3))
        result = runner.forward_for_capture(
            _WarmupBatch(action, positions, mask, prefix, workspace)
        )
        torch.testing.assert_close(result, torch.full((2, 3), 3.0))
        self.assertIs(target.inputs[0], positions)
        self.assertIs(target.inputs[1], mask)
        self.assertIs(target.inputs[2]["layers"][0][0], prefix[0])
        self.assertIs(target.inputs[2]["full_layers"][0][1], workspace[1])

    def test_request_fallback_reason_and_lock_recovery(self):
        target = torch.nn.Identity()
        runner = NPUActionSuffixGraphRunner(target)
        inputs = {
            "prefix_cache": {},
            "packed_position_ids": torch.zeros(1, dtype=torch.int64),
            "attention_mask": torch.zeros(1, 1),
            "feature_flags": {},
        }
        with runner.try_open_request(**inputs) as first:
            self.assertIsNone(first.fallback_reason)
            self.assertIsNone(runner.last_fallback_reason)
            with runner.try_open_request(**inputs) as concurrent:
                self.assertEqual(concurrent.fallback_reason, "concurrent_request")
                self.assertEqual(runner.last_fallback_reason, "concurrent_request")
        with runner.try_open_request(**inputs) as recovered:
            self.assertIsNone(recovered.fallback_reason)
            self.assertIsNone(runner.last_fallback_reason)
        self.assertFalse(runner.unhealthy)


if __name__ == "__main__":
    unittest.main()
