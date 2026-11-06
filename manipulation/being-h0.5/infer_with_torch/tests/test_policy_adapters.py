# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise physical-hand adapters through their public prediction interface."""

from types import SimpleNamespace
import unittest

import numpy as np

from vae.integration.beingh_shadow_grasp_policies import (
    GeometricShadowJointPolicyAdapter,
    ShadowJointPolicyAdapter,
)


class EchoPolicy:
    def __init__(self):
        self.query_step_indices = []
        self.observations = []
        self.chunk = None
        self.index = 0

    def reset(self):
        self.query_step_indices.clear()
        self.observations.clear()
        self.index = 0

    def will_query_next(self):
        return self.index % 2 == 0

    def predict(self, observation):
        self.observations.append(observation["state"].copy())
        if self.will_query_next():
            self.query_step_indices.append(self.index)
            self.chunk = np.stack([observation["state"], observation["state"]])
        result = self.chunk[self.index % 2].copy()
        self.index += 1
        return result

    def current_action_chunk(self):
        return self.chunk.copy()


class FakeRetargeter:
    def __init__(self):
        self.batch_calls = 0

    @staticmethod
    def joint_names(hand):
        return tuple(str(i) for i in range(22 if hand == 'shadow_hand_right' else 15))

    def reset(self):
        self.batch_calls = 0

    def retarget(self, value, source, target, **kwargs):
        return SimpleNamespace(target_q=np.full(len(self.joint_names(target)), np.mean(value)))

    def retarget_batch(self, values, source, target, **kwargs):
        self.batch_calls += 1
        return SimpleNamespace(target_q=np.repeat(values.mean(axis=1, keepdims=True), 15, axis=1))


class PolicyAdapterTests(unittest.TestCase):
    def test_native_shadow_and_gaia_preserve_joint_dimensions(self):
        for hand, count in [('shadow_hand_right', 22), ('gaia_hand_right', 15)]:
            with self.subTest(hand=hand):
                policy = EchoPolicy()
                adapter = ShadowJointPolicyAdapter(
                    policy, target_hand=hand, joint_names=tuple(str(i) for i in range(count)),
                    dataset_to_mujoco_signs=np.full(count, -1), policy_wrist_euler_offset=np.ones(3),
                    policy_wrist_world_origin=np.array([0, 0, .4]), observation_mode='commanded',
                )
                adapter.set_initial_commanded_z(np.full(count, .2))
                adapter.reset()
                state = np.arange(6 + count, dtype=np.float32) / 100
                result = adapter.predict({'state': state})
                np.testing.assert_allclose(result[:6], state[:6], atol=1e-7)
                np.testing.assert_allclose(result[6:], .2)
                self.assertEqual(result.shape, state.shape)
                adapter.predict({'state': state})
                np.testing.assert_allclose(policy.observations[1][6:], -.2)

    def test_geometry_batch_cache_and_reset(self):
        policy, retargeter = EchoPolicy(), FakeRetargeter()
        adapter = GeometricShadowJointPolicyAdapter(
            policy, retargeter=retargeter, target_hand='gaia_hand_right',
            dataset_to_mujoco_signs=np.ones(22), policy_wrist_euler_offset=np.zeros(3),
            policy_wrist_world_origin=np.zeros(3), action_chunk_mode='batch', observation_mode='commanded',
        )
        adapter.set_initial_commanded_z(np.full(22, .3))
        adapter.reset()
        observation = {'state': np.zeros(21, dtype=np.float32)}
        for _ in range(3):
            result = adapter.predict(observation)
            self.assertEqual(result.shape, (21,))
            np.testing.assert_allclose(result[6:], .3, atol=1e-7)
        self.assertEqual(retargeter.batch_calls, 2)
        adapter.reset()
        adapter.predict(observation)
        self.assertEqual(retargeter.batch_calls, 1)


if __name__ == '__main__':
    unittest.main()
