# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from vae.optimize.geometry_retargeter import GeometryRetargeter, GeometryRetargeterConfig


class GeometryRetargeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter = GeometryRetargeter(
            device="cpu", config=GeometryRetargeterConfig(profile="stable", max_iterations=2)
        )

    def test_identity_single_and_batch_preserve_pose(self):
        hand = self.retargeter.hand_names[0]
        pose = np.zeros(len(self.retargeter.joint_names(hand)), dtype=np.float32)
        single = self.retargeter.retarget(pose, hand, hand)
        batch = self.retargeter.retarget_batch(np.stack([pose, pose]), hand, hand)
        np.testing.assert_array_equal(single.target_q, pose)
        np.testing.assert_array_equal(batch.target_q, np.stack([pose, pose]))
        self.assertEqual(single.geometry_rmse, 0.0)
        self.assertEqual(batch.geometry_rmse, 0.0)

    def test_cross_hand_single_and_batch_return_finite_bounded_joints(self):
        source, target = self.retargeter.hand_names[:2]
        pose = np.zeros(len(self.retargeter.joint_names(source)), dtype=np.float32)
        single = self.retargeter.retarget(pose, source, target, stream="regression")
        batch = self.retargeter.retarget_batch(
            np.stack([pose, pose]), source, target, stream="regression"
        )
        runtime = self.retargeter.runtimes[target]
        joint_count = len(self.retargeter.joint_names(target))
        lower = runtime.q_lower[:joint_count].cpu().numpy()
        upper = runtime.q_upper[:joint_count].cpu().numpy()
        for result in (single, batch):
            self.assertTrue(np.isfinite(result.target_q).all())
            self.assertTrue(np.isfinite(result.objective))
            self.assertGreater(result.iterations, 0)
            self.assertTrue((result.target_q >= lower - 1e-6).all())
            self.assertTrue((result.target_q <= upper + 1e-6).all())
        self.retargeter.reset("regression")


if __name__ == "__main__":
    unittest.main()
