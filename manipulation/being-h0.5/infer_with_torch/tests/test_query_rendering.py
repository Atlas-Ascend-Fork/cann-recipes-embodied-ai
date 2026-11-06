# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Query-only rendering must preserve physics and the images read by a policy."""

import os
import unittest

import numpy as np

from vae.sim import GraspEnv, GraspEnvConfig, PolicyEvaluationClient


class TwoStepPolicy:
    def __init__(self):
        self.index = 0
        self.query_images = []

    def reset(self):
        self.index = 0
        self.query_images.clear()

    def needs_images(self):
        return self.index % 2 == 0

    def predict(self, observation):
        if self.needs_images():
            self.query_images.append({key: value.copy() for key, value in observation['images'].items()})
        self.index += 1
        return observation['state'].copy()


@unittest.skipUnless(
    os.environ.get("RUN_MUJOCO_RENDER_TESTS") == "1",
    "Set RUN_MUJOCO_RENDER_TESTS=1 with a working GL backend.",
)
class QueryRenderingTests(unittest.TestCase):
    def test_query_images_and_physics_match_full_rendering(self):
        results, images = [], []
        config = GraspEnvConfig(hand='gaia_hand_right', observation_cameras=('ego_opposite', 'wrist'),
                                width=64, height=48, max_steps=4)
        for selective in (False, True):
            policy = TwoStepPolicy()
            with GraspEnv(config) as env:
                result = PolicyEvaluationClient(env, policy).run(
                    max_steps=4, record_images=False,
                    observation_images_required=policy.needs_images if selective else None,
                )
            results.append(result)
            images.append(policy.query_images)
        np.testing.assert_array_equal(results[0].states, results[1].states)
        np.testing.assert_array_equal(results[0].actions, results[1].actions)
        np.testing.assert_array_equal(results[0].object_poses, results[1].object_poses)
        self.assertEqual(len(images[1]), 2)
        for full, selective in zip(images[0], images[1]):
            self.assertEqual(set(selective), {'ego_opposite', 'wrist'})
            for camera in full:
                np.testing.assert_array_equal(full[camera], selective[camera])

    def test_video_recording_keeps_every_frame(self):
        policy = TwoStepPolicy()
        with GraspEnv(GraspEnvConfig(hand='gaia_hand_right', width=64, height=48, max_steps=2)) as env:
            result = PolicyEvaluationClient(env, policy).run(
                max_steps=2, record_images=True, observation_images_required=lambda: False,
            )
        self.assertEqual(len(result.images), 3)


if __name__ == '__main__':
    unittest.main()
