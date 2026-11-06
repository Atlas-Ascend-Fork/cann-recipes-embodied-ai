# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared imports and constants for Being-H closed-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .beingh_shadow_grasp_common import *  # noqa: F401,F403
from .beingh_shadow_grasp_primitives import *  # noqa: F401,F403


@dataclass
class _MotionSamples:
    positions: list[np.ndarray]
    position_accelerations: list[np.ndarray]
    rotations: list[np.ndarray]
    rotation_accelerations: list[np.ndarray]
    z_steps: list[np.ndarray]
    z_accelerations: list[np.ndarray]


class TrainingDistributionRateLimiter:
    """Constrain wrist velocity/acceleration; rotation is limited on SO(3)."""

    def __init__(
        self,
        *,
        position_step_limits: np.ndarray,
        position_acceleration_limits: np.ndarray,
        rotation_step_limit: float,
        rotation_acceleration_limit: float,
        z_step_limits: np.ndarray | None,
        z_acceleration_limits: np.ndarray | None,
        limit_z_gesture: bool,
        quantile: float,
        source_dataset: Path,
        include_xyz_equivalent: bool = False,
    ) -> None:
        self.position_step_limits, self.position_acceleration_limits = (
            np.asarray(position_step_limits, dtype=np.float32),
            np.asarray(position_acceleration_limits, dtype=np.float32),
        )
        self.rotation_step_limit, self.rotation_acceleration_limit = (
            float(rotation_step_limit),
            float(rotation_acceleration_limit),
        )
        self.z_step_limits, self.z_acceleration_limits = (
            (None if z_step_limits is None else np.asarray(z_step_limits, dtype=np.float32)),
            (None if z_acceleration_limits is None else np.asarray(z_acceleration_limits, dtype=np.float32)),
        )
        self.limit_z_gesture = bool(limit_z_gesture)
        if self.limit_z_gesture and (self.z_step_limits is None or self.z_acceleration_limits is None):
            raise ValueError("z velocity and acceleration limits are required")
        self.quantile, self.source_dataset, self.include_xyz_equivalent = (
            float(quantile),
            source_dataset,
            bool(include_xyz_equivalent),
        )
        self._last_action = self._last_position_velocity = self._last_rotation_velocity = self._last_z_velocity = None

    @staticmethod
    def _limit_vector_norm(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if norm > limit:
            return value * (limit / norm)
        return value



    @staticmethod
    def _dataset_motion_samples(dataset: Path, direct_native_joints: bool, limit_z_gesture: bool):
        import pandas as pd
        positions, position_accels, rotations, rotation_accels, z_steps, z_accels = [], [], [], [], [], []
        minimum_dim = 7 if direct_native_joints else 52
        for path in sorted(dataset.glob("data/chunk-*/episode_*.parquet")):
            actions = np.stack(pd.read_parquet(path, columns=["action"])["action"].to_numpy()).astype(np.float32)
            if actions.ndim != 2 or actions.shape[1] < minimum_dim or len(actions) < 2:
                raise ValueError(f"Expected action [T,>={minimum_dim}] in {path}, got {actions.shape}")
            position_velocity = np.diff(actions[:, 0:3], axis=0)
            positions.append(np.abs(position_velocity))
            if len(position_velocity) >= 2:
                position_accels.append(np.abs(np.diff(position_velocity, axis=0)))
            rotation_velocity = (
                Rotation.from_euler("XYZ", actions[:, 3:6])[:-1].inv()
                * Rotation.from_euler("XYZ", actions[:, 3:6])[1:]
            ).as_rotvec()
            rotations.append(np.linalg.norm(rotation_velocity, axis=1))
            if len(rotation_velocity) >= 2:
                rotation_accels.append(np.linalg.norm(np.diff(rotation_velocity, axis=0), axis=1))
            if limit_z_gesture and not direct_native_joints:
                z_velocity = np.diff(actions[:, 28:52], axis=0)
                z_steps.append(np.abs(z_velocity))
                if len(z_velocity) >= 2:
                    z_accels.append(np.abs(np.diff(z_velocity, axis=0)))
        if any(not values for values in (positions, position_accels, rotations, rotation_accels)):
            raise ValueError("At least one three-frame episode is required for acceleration limits")
        return _MotionSamples(positions, position_accels, rotations, rotation_accels, z_steps, z_accels)


    @classmethod
    def from_lerobot_dataset(
        cls, dataset: Path, quantile: float, *, limit_z_gesture: bool = False,
        include_xyz_equivalent: bool = False, direct_native_joints: bool = False,
    ) -> "TrainingDistributionRateLimiter":
        if not 0.0 < quantile <= 1.0:
            raise ValueError("rate-limit-quantile must be in (0, 1]")
        values = cls._dataset_motion_samples(dataset, direct_native_joints, limit_z_gesture)
        position_steps, position_accelerations = values.positions, values.position_accelerations
        rotation_steps, rotation_accelerations = values.rotations, values.rotation_accelerations
        z_steps, z_accelerations = values.z_steps, values.z_accelerations
        position_limit = np.quantile(np.concatenate(position_steps), quantile, axis=0)
        position_acceleration_limit = np.quantile(np.concatenate(position_accelerations), quantile, axis=0)
        rotation_limit = float(np.quantile(np.concatenate(rotation_steps), quantile))
        rotation_acceleration_limit = float(np.quantile(np.concatenate(rotation_accelerations), quantile))
        z_limit = np.maximum(np.quantile(np.concatenate(z_steps), quantile, axis=0), 1e-05) if limit_z_gesture else None
        z_acceleration_limit = (
            np.maximum(np.quantile(np.concatenate(z_accelerations), quantile, axis=0), 1e-05)
            if limit_z_gesture else None
        )
        return cls(
            position_step_limits=np.maximum(position_limit, 1e-05),
            position_acceleration_limits=np.maximum(position_acceleration_limit, 1e-05),
            rotation_step_limit=max(rotation_limit, 0.0001),
            rotation_acceleration_limit=max(rotation_acceleration_limit, 0.0001),
            z_step_limits=z_limit, z_acceleration_limits=z_acceleration_limit,
            limit_z_gesture=limit_z_gesture, quantile=quantile,
            source_dataset=dataset.resolve(), include_xyz_equivalent=include_xyz_equivalent,
        )

    def reset(self) -> None:
        self._last_action = self._last_position_velocity = self._last_rotation_velocity = self._last_z_velocity = None







    def apply(self, target: np.ndarray, observation_state: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        observation_state = np.asarray(observation_state, dtype=np.float32)
        reference = observation_state.copy() if self._last_action is None else self._last_action.copy()
        output = target.copy()
        position_velocity = np.clip(target[0:3] - reference[0:3], -self.position_step_limits, self.position_step_limits)
        if self._last_position_velocity is not None:
            position_velocity = self._last_position_velocity + np.clip(
                position_velocity - self._last_position_velocity,
                -self.position_acceleration_limits,
                self.position_acceleration_limits,
            )
        output[0:3] = reference[0:3] + position_velocity
        reference_rotation = Rotation.from_euler("XYZ", reference[3:6])
        target_rotation = Rotation.from_euler("XYZ", target[3:6])
        rotation_velocity = (reference_rotation.inv() * target_rotation).as_rotvec()
        rotation_velocity = self._limit_vector_norm(rotation_velocity, self.rotation_step_limit)
        if self._last_rotation_velocity is not None:
            rotation_acceleration = self._limit_vector_norm(
                rotation_velocity - self._last_rotation_velocity, self.rotation_acceleration_limit
            )
            rotation_velocity = self._last_rotation_velocity + rotation_acceleration
            rotation_velocity = self._limit_vector_norm(rotation_velocity, self.rotation_step_limit)
        limited_rotation = reference_rotation * Rotation.from_rotvec(rotation_velocity)
        output[3:6] = align_euler_branch(
            limited_rotation.as_euler("XYZ").astype(np.float32),
            reference[3:6],
            include_xyz_equivalent=self.include_xyz_equivalent,
        )
        if self.limit_z_gesture:
            if self.z_step_limits is None or self.z_acceleration_limits is None:
                raise RuntimeError("rate limiter limits must be initialized before reset")
            z_velocity = np.clip(target[6:30] - reference[6:30], -self.z_step_limits, self.z_step_limits)
            if self._last_z_velocity is not None:
                z_velocity = self._last_z_velocity + np.clip(
                    z_velocity - self._last_z_velocity, -self.z_acceleration_limits, self.z_acceleration_limits
                )
            output[6:30] = reference[6:30] + z_velocity
            self._last_z_velocity = z_velocity.copy()
        self._last_action, self._last_position_velocity, self._last_rotation_velocity = (
            output.copy(),
            position_velocity.copy(),
            rotation_velocity.copy(),
        )
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "source_dataset": str(self.source_dataset),
            "quantile": self.quantile,
            "position_step_limits_m": self.position_step_limits.tolist(),
            "position_acceleration_limits_m_per_step2": self.position_acceleration_limits.tolist(),
            "rotation_step_limit_deg": float(np.degrees(self.rotation_step_limit)),
            "rotation_acceleration_limit_deg_per_step2": float(np.degrees(self.rotation_acceleration_limit)),
            "limit_z_gesture": self.limit_z_gesture,
            "z_step_limits": None if self.z_step_limits is None else self.z_step_limits.tolist(),
            "z_acceleration_limits": None
            if self.z_acceleration_limits is None
            else self.z_acceleration_limits.tolist(),
            "decoded_native_joint_limiter": True,
        }


def _load_motion_actions(dataset: Path, direct_shadow_joints: bool):
    import pandas as pd
    paths = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset}")
    samples, lengths = [], []
    for path in paths:
        actions = np.stack(pd.read_parquet(path, columns=["action"])["action"].to_numpy()).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"Expected action [T,>=7] in {path}, got {actions.shape}")
        hand_slice = actions[:, 6:28] if direct_shadow_joints and actions.shape[1] >= 52 else (
            actions[:, 6:] if direct_shadow_joints else actions[:, 28:52]
        )
        samples.append(hand_slice)
        lengths.append(len(hand_slice))
    return samples, lengths


def _compute_motion_limits(decoded, lengths, quantile):
    steps, accelerations, offset = [], [], 0
    for length in lengths:
        episode_q = decoded[offset:offset + length]
        offset += length
        if length >= 2:
            velocity = np.diff(episode_q, axis=0)
            steps.append(np.abs(velocity))
            if len(velocity) >= 2:
                accelerations.append(np.abs(np.diff(velocity, axis=0)))
    if not accelerations:
        raise ValueError("At least one three-frame episode is required for q acceleration limits")
    step_limits = np.maximum(np.quantile(np.concatenate(steps), quantile, axis=0), 1e-05).astype(np.float32)
    acceleration_limits = np.maximum(
        np.quantile(np.concatenate(accelerations), quantile, axis=0), 1e-05
    ).astype(np.float32)
    return step_limits, acceleration_limits


class AuditedNativePolicy:
    """Audit decoded actions, constrain native joints, and enforce safety faults."""

    def __init__(
        self,
        policy: GesturePolicyAdapter | ShadowJointPolicyAdapter,
        q_step_limits: np.ndarray | None,
        q_acceleration_limits: np.ndarray | None,
        *,
        safety_mode: str = "off",
        safety_max_wrist_position_error_m: float = 0.25,
        safety_max_wrist_rotation_error_deg: float = 90.0,
        safety_max_object_drop_m: float = 0.3,
        safety_max_workspace_radius_m: float = 0.6,
    ) -> None:
        if safety_mode not in {"off", "hold", "terminate"}:
            raise ValueError("safety_mode must be off, hold, or terminate")
        self.policy, self.safety_mode = policy, safety_mode
        self.q_step_limits, self.q_acceleration_limits = (
            (None if q_step_limits is None else np.asarray(q_step_limits, dtype=np.float32)),
            (None if q_acceleration_limits is None else np.asarray(q_acceleration_limits, dtype=np.float32)),
        )
        self.safety_max_wrist_position_error_m, self.safety_max_wrist_rotation_error_rad = (
            float(safety_max_wrist_position_error_m),
            float(np.radians(safety_max_wrist_rotation_error_deg)),
        )
        self.safety_max_object_drop_m, self.safety_max_workspace_radius_m = (
            float(safety_max_object_drop_m),
            float(safety_max_workspace_radius_m),
        )
        self._last_action = self._last_q_velocity = self._initial_wrist_position = self._initial_object_z = (
            self.termination_reason
        ) = self._dataset_wrist_trajectory = None
        self._safety_latched = self.termination_requested = False
        self.safety_events, self.raw_native_actions, self.executed_native_actions = [], [], []
        self.model_native_actions, self.dataset_wrist_actions_requested, self.predict_latencies_s = [], [], []
        self._dataset_wrist_step = 0




    @classmethod
    def q_motion_limits_from_lerobot_dataset(
        cls, dataset: Path, vae: NativeVAE | None, target_hand: str, quantile: float,
        *, direct_shadow_joints: bool = False,
        geometry_retargeter: GeometryRetargeter | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        samples, lengths = _load_motion_actions(dataset, direct_shadow_joints)
        if direct_shadow_joints:
            decoded = np.concatenate(samples, axis=0)
        else:
            if vae is None:
                raise RuntimeError("z_gesture motion limits require NativeVAE")
            decoded = vae.decode(np.concatenate(samples, axis=0), target_hand).detach().cpu().numpy()
        step_limits, acceleration_limits = _compute_motion_limits(decoded, lengths, quantile)
        if direct_shadow_joints and target_hand != "shadow_hand_right" and decoded.shape[1] == 22:
            if geometry_retargeter is None:
                raise RuntimeError("Cross-hand physical-joint limits require GeometryRetargeter")
            step_limits = geometry_retargeter.project_motion_limits(step_limits, "shadow_hand_right", target_hand)
            acceleration_limits = geometry_retargeter.project_motion_limits(
                acceleration_limits, "shadow_hand_right", target_hand
            )
        return step_limits, acceleration_limits


    def set_wrist_action_trajectory(self, wrist_actions: np.ndarray | None) -> None:
        if wrist_actions is None:
            self._dataset_wrist_trajectory = None
        else:
            values = np.asarray(wrist_actions, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 6:
                raise ValueError(f"Expected dataset wrist trajectory [T,6], got {values.shape}")
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise ValueError("Dataset wrist trajectory must be non-empty and finite")
            self._dataset_wrist_trajectory = values.copy()
        self._dataset_wrist_step = 0


    def set_initial_commanded_z(self, z_gesture: np.ndarray | None) -> None:
        self.policy.set_initial_commanded_z(z_gesture)


    def reset(self) -> None:
        self.policy.reset()
        self._last_action = self._last_q_velocity = self._initial_wrist_position = self._initial_object_z = (
            self.termination_reason
        ) = None
        self._safety_latched = self.termination_requested = False
        clear_attributes(
            self,
            (
                "safety_events",
                "raw_native_actions",
                "executed_native_actions",
                "model_native_actions",
                "dataset_wrist_actions_requested",
                "predict_latencies_s",
            ),
        )
        self._dataset_wrist_step = 0




    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        started_at = time.perf_counter()
        observation_state = np.asarray(observation["state"], dtype=np.float32)
        hold_action = observation_state.copy()
        dataset_wrist = self._next_dataset_wrist()
        target, model_target = self._predict_model_action(observation, hold_action, dataset_wrist, observation_state)
        self.model_native_actions.append(model_target)
        self.raw_native_actions.append(target.copy())
        output = self._apply_q_limits(target, observation_state)
        self._last_action = output.copy()
        self.executed_native_actions.append(output.copy())
        self.predict_latencies_s.append(time.perf_counter() - started_at)
        return output


    def _predict_model_action(self, observation, hold_action, dataset_wrist, observation_state):
        fault = self._observation_fault(observation) if self.safety_mode != "off" else None
        if self._safety_latched or fault is not None:
            if fault is not None:
                self._record_fault(fault)
            return hold_action, hold_action.copy()
        try:
            target = np.asarray(self.policy.predict(observation), dtype=np.float32)
            model_target = target.copy()
            if dataset_wrist is not None:
                target = target.copy()
                target[:6] = dataset_wrist
        except Exception as error:
            if self.safety_mode == "off":
                raise
            self._record_fault(f"policy exception: {type(error).__name__}: {error}")
            return hold_action, hold_action.copy()
        if self.safety_mode != "off" and not self._safety_latched:
            fault = self._command_fault(target, observation_state)
            if fault is not None:
                self._record_fault(fault)
                target = hold_action
        return target, model_target


    def _apply_q_limits(self, target, observation_state):
        output = target.copy()
        if self.q_step_limits is not None and not self._safety_latched:
            reference = observation_state if self._last_action is None else self._last_action
            q_velocity = np.clip(target[6:] - reference[6:], -self.q_step_limits, self.q_step_limits)
            if self._last_q_velocity is not None:
                if self.q_acceleration_limits is None:
                    raise RuntimeError("Native q acceleration limits are missing")
                q_velocity = self._last_q_velocity + np.clip(
                    q_velocity - self._last_q_velocity,
                    -self.q_acceleration_limits,
                    self.q_acceleration_limits,
                )
            output[6:] = reference[6:] + q_velocity
            self._last_q_velocity = q_velocity.copy()
        elif self._safety_latched:
            output = observation_state.copy()
            setattr(self, "_last_q_velocity", np.zeros_like(output[6:]))
        return output

    def _command_fault(self, target: np.ndarray, observation_state: np.ndarray) -> str | None:
        if not np.isfinite(target).all():
            return "non-finite decoded action"
        position_error = float(np.linalg.norm(target[0:3] - observation_state[0:3]))
        if position_error > self.safety_max_wrist_position_error_m:
            return f"wrist command position error {position_error:.3f}m"
        current_rotation = Rotation.from_euler("XYZ", observation_state[3:6])
        target_rotation = Rotation.from_euler("XYZ", target[3:6])
        rotation_error = float((current_rotation.inv() * target_rotation).magnitude())
        if rotation_error > self.safety_max_wrist_rotation_error_rad:
            return f"wrist command rotation error {np.degrees(rotation_error):.1f}deg"
        return None


    def _observation_fault(self, observation: Mapping[str, Any]) -> str | None:
        state = np.asarray(observation["state"], dtype=np.float32)
        object_pose = np.asarray(observation.get("object_pose", []), dtype=np.float32)
        if not np.isfinite(state).all():
            return "non-finite robot state"
        if object_pose.size and (not np.isfinite(object_pose).all()):
            return "non-finite object pose"
        if self._initial_wrist_position is None:
            self._initial_wrist_position = state[0:3].copy()
        if object_pose.size and self._initial_object_z is None:
            self._initial_object_z = float(object_pose[2])
        if np.linalg.norm(state[0:3] - self._initial_wrist_position) > self.safety_max_workspace_radius_m:
            return "wrist left configured workspace radius"
        if (
            object_pose.size
            and self._initial_object_z is not None
            and (float(object_pose[2]) < self._initial_object_z - self.safety_max_object_drop_m)
        ):
            return "object dropped below configured safety floor"
        return None


    def _record_fault(self, reason: str) -> None:
        if not self._safety_latched:
            self.safety_events.append({"step": len(self.executed_native_actions), "reason": reason})
            LOGGER.info(f"[safety] {self.safety_mode}: {reason}")
        self._safety_latched, self.termination_reason, self.termination_requested = (
            True,
            reason,
            self.safety_mode == "terminate",
        )


    def _next_dataset_wrist(self) -> np.ndarray | None:
        if self._dataset_wrist_trajectory is None:
            return None
        if self._dataset_wrist_step >= len(self._dataset_wrist_trajectory):
            raise IndexError(f"Dataset wrist trajectory was exhausted at control step {self._dataset_wrist_step}")
        wrist = self._dataset_wrist_trajectory[self._dataset_wrist_step].copy()
        self._dataset_wrist_step += 1
        self.dataset_wrist_actions_requested.append(wrist.copy())
        return wrist


class ShadowJointPolicyAdapter:
    """Adapt native Shadow state/actions without passing joints through the VAE."""

    def __init__(
        self,
        policy: BeingHGesturePolicy,
        *,
        target_hand: str,
        joint_names: tuple[str, ...],
        dataset_to_mujoco_signs: np.ndarray,
        policy_wrist_euler_offset: np.ndarray,
        policy_wrist_world_origin: np.ndarray,
        observation_mode: str,
    ) -> None:
        if target_hand not in SUPPORTED_HANDS:
            raise ValueError(f"Unsupported target hand {target_hand!r}")
        if len(joint_names) <= 0:
            raise ValueError("Native-joint checkpoint must expose at least one joint")
        self.policy, self.target_hand, self.joint_names = policy, target_hand, tuple(joint_names)
        self.dataset_to_mujoco_signs = np.asarray(dataset_to_mujoco_signs, dtype=np.float32)
        if self.dataset_to_mujoco_signs.shape != (len(self.joint_names),) or not np.all(
            np.isin(self.dataset_to_mujoco_signs, (-1.0, 1.0))
        ):
            raise ValueError("dataset_to_mujoco_signs must match native joint dimension and contain {-1,+1}")
        setattr(self, "policy_wrist_euler_offset", np.asarray(policy_wrist_euler_offset, dtype=np.float32))
        setattr(self, "policy_wrist_world_origin", np.asarray(policy_wrist_world_origin, dtype=np.float32))
        if observation_mode not in {"encoded", "commanded"}:
            raise ValueError("observation_mode must be encoded or commanded")
        self.observation_mode, self.z_dim = observation_mode, 0
        self.encoded_latent_states, self.policy_latent_states = [], []
        self._initial_commanded_shadow_q = self._commanded_shadow_q = None

    def set_initial_commanded_z(self, shadow_q: np.ndarray | None) -> None:
        if shadow_q is None:
            setattr(self, "_initial_commanded_shadow_q", None)
            setattr(self, "_commanded_shadow_q", None)
            return
        values = np.asarray(shadow_q, dtype=np.float32)
        if values.shape != (len(self.joint_names),) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Initial commanded native joints have the wrong dimension: "
                f"expected {len(self.joint_names)}, got {values.shape}"
            )
        setattr(self, "_initial_commanded_shadow_q", values.copy())
        setattr(self, "_commanded_shadow_q", values.copy())

    def reset(self) -> None:
        self.policy.reset()
        self.encoded_latent_states.clear()
        self.policy_latent_states.clear()
        self._commanded_shadow_q = (
            None if self._initial_commanded_shadow_q is None else self._initial_commanded_shadow_q.copy()
        )

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        native_state = np.asarray(observation["state"], dtype=np.float32)
        expected_dim = 6 + len(self.joint_names)
        if native_state.shape != (expected_dim,):
            raise ValueError(f"Expected native Shadow state [{expected_dim}], got {native_state.shape}")
        policy_wrist = native_state[:6].copy()
        policy_wrist[0:3] -= self.policy_wrist_world_origin
        policy_wrist[3:6] += self.policy_wrist_euler_offset
        observed_shadow_q = native_state[6:].copy()
        if self.observation_mode == "commanded":
            if self._commanded_shadow_q is None:
                raise RuntimeError("Commanded Shadow feedback has no initial joint state")
            policy_shadow_q = self._commanded_shadow_q.copy()
        else:
            policy_shadow_q = observed_shadow_q.copy()
        self.encoded_latent_states.append(observed_shadow_q * self.dataset_to_mujoco_signs)
        self.policy_latent_states.append(policy_shadow_q * self.dataset_to_mujoco_signs)
        policy_observation = dict(observation)
        policy_observation["native_state"] = native_state
        policy_observation["state"] = np.concatenate(
            [policy_wrist, policy_shadow_q * self.dataset_to_mujoco_signs]
        ).astype(np.float32)
        policy_action = np.asarray(self.policy.predict(policy_observation), dtype=np.float32)
        if policy_action.shape != (expected_dim,):
            raise ValueError(f"Physical-joint policy must output [{expected_dim}], got {policy_action.shape}")
        if not np.all(np.isfinite(policy_action)):
            raise ValueError("Physical-joint policy output contains NaN or Inf")
        native_action = policy_action.copy()
        native_action[0:3] += self.policy_wrist_world_origin
        native_action[3:6] -= self.policy_wrist_euler_offset
        native_action[6:] *= self.dataset_to_mujoco_signs
        if self.observation_mode == "commanded":
            self._commanded_shadow_q = native_action[6:].copy()
        return native_action


class GeometricShadowJointPolicyAdapter:
    """Use native Shadow joints as the canonical cross-hand policy space."""

    def __init__(
        self,
        policy: BeingHGesturePolicy,
        *,
        retargeter: GeometryRetargeter,
        target_hand: str,
        dataset_to_mujoco_signs: np.ndarray,
        policy_wrist_euler_offset: np.ndarray,
        policy_wrist_world_origin: np.ndarray,
        action_chunk_mode: str,
        observation_mode: str,
    ) -> None:
        if target_hand == "shadow_hand_right":
            raise ValueError("Use ShadowJointPolicyAdapter for native Shadow evaluation")
        self.policy, self.retargeter, self.target_hand = policy, retargeter, target_hand
        self.target_joint_names, self.shadow_joint_names = (
            retargeter.joint_names(target_hand),
            retargeter.joint_names("shadow_hand_right"),
        )
        self.dataset_to_mujoco_signs = np.asarray(dataset_to_mujoco_signs, dtype=np.float32)
        if self.dataset_to_mujoco_signs.shape != (22,) or not np.all(
            np.isin(self.dataset_to_mujoco_signs, (-1.0, 1.0))
        ):
            raise ValueError("dataset_to_mujoco_signs must contain 22 values in {-1,+1}")
        setattr(self, "policy_wrist_euler_offset", np.asarray(policy_wrist_euler_offset, dtype=np.float32))
        setattr(self, "policy_wrist_world_origin", np.asarray(policy_wrist_world_origin, dtype=np.float32))
        if action_chunk_mode not in {"batch", "sequential"}:
            raise ValueError("action_chunk_mode must be 'batch' or 'sequential'")
        self.action_chunk_mode = action_chunk_mode
        if observation_mode not in {"encoded", "commanded"}:
            raise ValueError("observation_mode must be encoded or commanded")
        self.observation_mode, self.z_dim = observation_mode, 22
        self.encoded_latent_states, self.policy_latent_states, self.retargeted_shadow_observation_joints = [], [], []
        self.model_shadow_action_joints, self.retargeted_target_action_joints = [], []
        self.observation_retarget_results, self.action_retarget_results = [], []
        self._cached_shadow_action_chunk = self._cached_target_action_chunk = self._cached_shadow_observation = None
        self._cached_chunk_index = 0
        self._initial_commanded_shadow_q = self._commanded_shadow_q = None

    @staticmethod
    def _result_summary(results: list[Any]) -> dict[str, Any]:
        if not results:
            return {
                "count": 0,
                "geometry_rmse_mean": None,
                "geometry_rmse_max": None,
                "latency_s_mean": None,
                "latency_s_max": None,
            }
        rmse = np.asarray([result.geometry_rmse for result in results], dtype=np.float64)
        latency = np.asarray([result.elapsed_s for result in results], dtype=np.float64)
        return {
            "count": len(results),
            "frames": int(sum((int(getattr(result, "batch_size", 1)) for result in results))),
            "geometry_rmse_mean": float(rmse.mean()),
            "geometry_rmse_max": float(rmse.max()),
            "latency_s_mean": float(latency.mean()),
            "latency_s_max": float(latency.max()),
        }

    def reset(self) -> None:
        self.policy.reset()
        self.retargeter.reset()
        self._commanded_shadow_q = (
            None if self._initial_commanded_shadow_q is None else self._initial_commanded_shadow_q.copy()
        )
        clear_attributes(
            self,
            (
                "encoded_latent_states",
                "policy_latent_states",
                "retargeted_shadow_observation_joints",
                "model_shadow_action_joints",
                "retargeted_target_action_joints",
                "observation_retarget_results",
                "action_retarget_results",
            ),
        )
        self._cached_shadow_action_chunk = self._cached_target_action_chunk = self._cached_shadow_observation = None
        self._cached_chunk_index = 0


    def set_initial_commanded_z(self, shadow_q: np.ndarray | None) -> None:
        if shadow_q is None:
            setattr(self, "_initial_commanded_shadow_q", None)
            setattr(self, "_commanded_shadow_q", None)
            return
        values = np.asarray(shadow_q, dtype=np.float32)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Initial commanded Shadow joints must be finite [22], got {values.shape}")
        setattr(self, "_initial_commanded_shadow_q", values.copy())
        setattr(self, "_commanded_shadow_q", values.copy())


    def retargeting_metadata(self) -> dict[str, Any]:
        return {
            **self.retargeter.metadata(),
            "canonical_hand": "shadow_hand_right",
            "action_chunk_mode": self.action_chunk_mode,
            "observation_mode": self.observation_mode,
            "observation_direction": f"{self.target_hand}->shadow_hand_right",
            "action_direction": f"shadow_hand_right->{self.target_hand}",
            "observation": self._result_summary(self.observation_retarget_results),
            "action": self._result_summary(self.action_retarget_results),
        }

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        native_state = np.asarray(observation["state"], dtype=np.float32)
        expected = 6 + len(self.target_joint_names)
        if native_state.shape != (expected,):
            raise ValueError(f"Expected {self.target_hand} state [{expected}], got {native_state.shape}")
        policy_observation = self._policy_observation(observation, native_state)
        before = len(self.policy.query_step_indices)
        policy_action = np.asarray(self.policy.predict(policy_observation), dtype=np.float32)
        target = self._retarget_action(policy_action, before, len(self.policy.query_step_indices))
        wrist = policy_action[:6].copy()
        wrist[:3] += self.policy_wrist_world_origin
        wrist[3:6] -= self.policy_wrist_euler_offset
        return np.concatenate([wrist, target]).astype(np.float32)



    def _policy_observation(self, observation: Mapping[str, Any], native_state: np.ndarray) -> dict[str, Any]:
        query_expected = self.policy.will_query_next()
        if self.action_chunk_mode == "sequential" or query_expected or self._cached_shadow_observation is None:
            result = self.retargeter.retarget(
                native_state[6:], self.target_hand, "shadow_hand_right", stream="observation"
            )
            shadow_observation = result.target_q
            self.observation_retarget_results.append(result)
            self._cached_shadow_observation = shadow_observation.copy()
        else:
            shadow_observation = self._cached_shadow_observation.copy()
        self.retargeted_shadow_observation_joints.append(shadow_observation.copy())
        if self.observation_mode == "commanded":
            if self._commanded_shadow_q is None:
                raise RuntimeError("Commanded geometry feedback has no initial Shadow joints")
            policy_shadow = self._commanded_shadow_q.copy()
        else:
            policy_shadow = shadow_observation.copy()
        self.encoded_latent_states.append(shadow_observation * self.dataset_to_mujoco_signs)
        self.policy_latent_states.append(policy_shadow * self.dataset_to_mujoco_signs)
        wrist = native_state[:6].copy()
        wrist[:3] -= self.policy_wrist_world_origin
        wrist[3:6] += self.policy_wrist_euler_offset
        return dict(observation, native_state=native_state,
                    state=np.concatenate([wrist, policy_shadow * self.dataset_to_mujoco_signs]).astype(np.float32))

    def _retarget_action(self, policy_action: np.ndarray, query_before: int, query_after: int) -> np.ndarray:
        if policy_action.shape != (28,) or not np.all(np.isfinite(policy_action)):
            raise ValueError("Physical Shadow-joint policy output must be finite [28]")
        shadow_action = policy_action[6:] * self.dataset_to_mujoco_signs
        if self.observation_mode == "commanded":
            self._commanded_shadow_q = shadow_action.copy()
        target = self._retarget_cached_action(shadow_action, query_before, query_after)
        self.model_shadow_action_joints.append(shadow_action.copy())
        self.retargeted_target_action_joints.append(target.copy())
        return target


    def _retarget_cached_action(self, shadow_action, query_before, query_after):
        if self.action_chunk_mode != "batch":
            result = self.retargeter.retarget(
                shadow_action, "shadow_hand_right", self.target_hand, stream="action"
            )
            self.action_retarget_results.append(result)
            return result.target_q.copy()
        if query_after > query_before:
            canonical = self.policy.current_action_chunk()
            shadow_chunk = canonical[:, 6:] * self.dataset_to_mujoco_signs
            result = self.retargeter.retarget_batch(
                shadow_chunk, "shadow_hand_right", self.target_hand, stream="action_chunk"
            )
            self.action_retarget_results.append(result)
            self._cached_shadow_action_chunk = shadow_chunk.copy()
            self._cached_target_action_chunk = result.target_q.copy()
            self._cached_chunk_index = 0
        if self._cached_shadow_action_chunk is None or self._cached_target_action_chunk is None:
            raise RuntimeError("Batched target action chunk is unavailable")
        if self._cached_chunk_index >= len(self._cached_target_action_chunk):
            raise RuntimeError("Batched target action chunk is exhausted")
        cached = self._cached_shadow_action_chunk[self._cached_chunk_index]
        if not np.allclose(shadow_action, cached, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Selected canonical hand action differs from the cached batched chunk")
        target = self._cached_target_action_chunk[self._cached_chunk_index].copy()
        self._cached_chunk_index += 1
        return target


class BeingHGesturePolicy:
    """Adapt wrist plus z_gesture or Shadow joints to BeingHPolicy's API."""

    def __init__(
        self, *, model_path: Path, data_config_name: str, instruction: str,
        device: str, replan_every: int, seed: int,
                 noise_mode: str, action_selection: str, inference_mode: str, temporal_ensemble_decay: float,
                 temporal_ensemble_max_history: int | None, rate_limiter: TrainingDistributionRateLimiter | None,
                 chunk_smoother: ChunkTemporalSmoother | None, clip_normalized_wrist_action: bool, use_mpg: bool | None,
                 mpg_refinement_iters: int | None, num_inference_timesteps: int | None) -> None:
        params = locals()
        self._validate_init(params)
        self._init_representation(params)
        self._init_policy(params)
        self._init_runtime(params)

    @staticmethod
    def _validate_init(params: Mapping[str, Any]) -> None:
        if params["replan_every"] <= 0:
            raise ValueError("replan_every must be positive")
        if params["noise_mode"] not in {"rollout", "fixed_per_query"}:
            raise ValueError("noise_mode must be 'rollout' or 'fixed_per_query'")
        if params["action_selection"] not in {"chunk", "temporal_ensemble"}:
            raise ValueError("action_selection must be 'chunk' or 'temporal_ensemble'")
        if params["inference_mode"] not in {"sync", "async"}:
            raise ValueError("inference_mode must be 'sync' or 'async'")
        if params["inference_mode"] == "async" and params["action_selection"] != "temporal_ensemble":
            raise ValueError("async inference currently requires temporal_ensemble action selection")
        if params["temporal_ensemble_decay"] < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        history = params["temporal_ensemble_max_history"]
        if history is not None and history <= 0:
            raise ValueError("temporal_ensemble_max_history must be positive")


    @staticmethod
    def _decode_policy_result(result: Mapping[str, Any], hand_action_key: str) -> np.ndarray:
        values = []
        for key in ("action.eef_position", "action.eef_rotation", hand_action_key):
            value = np.asarray(result[key], dtype=np.float32)
            if value.ndim == 3:
                if value.shape[0] != 1:
                    raise ValueError(f"{key}: only batch size one is supported")
                value = value[0]
            values.append(value)
        return np.concatenate(values, axis=-1)

    def reset(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            if not self._pending_future.cancelled():
                self._pending_future.result()
            setattr(self, "_pending_future", None)
            setattr(self, "_pending_query_step", None)
        seed_everything(self.seed)
        self._chunk = self._last_query_launch_step = None
        self._chunk_index = self._step_index = self.async_submitted_queries = self.async_blocking_waits = 0
        self.async_blocking_wait_seconds = 0.0
        clear_attributes(
            self,
            (
                "_ensemble_history",
                "predicted_chunks",
                "smoothed_predicted_chunks",
                "query_step_indices",
                "observed_latent_states",
                "latest_query_actions",
                "temporal_ensemble_actions",
                "temporal_ensemble_candidate_counts",
                "raw_selected_actions",
                "branch_aligned_actions",
                "executed_latent_actions",
                "query_latencies_s",
            ),
        )
        if self.rate_limiter is not None:
            self.rate_limiter.reset()

    def will_query_next(self) -> bool:
        """Return whether the next synchronous chunk-selection step queries Being-H."""
        if self.inference_mode != "sync" or self.action_selection != "chunk":
            return True
        return self._chunk is None or self._chunk_index >= len(self._chunk) or self._chunk_index >= self.replan_every

    def current_action_chunk(self) -> np.ndarray:
        """Return the current post-smoothing canonical action chunk."""
        if self._chunk is None:
            raise RuntimeError("Being-H has not produced an action chunk")
        return self._chunk.copy()

    def finish_episode(self) -> None:
        """Finish any in-flight inference without destroying the reusable policy."""
        if self._pending_future is not None:
            self._collect_async_query(block=True)

    def close(self) -> None:
        self.finish_episode()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32)
        self.observed_latent_states.append(state.copy())
        latest, raw_action, branch_aligned, candidate_count = self._select_action(observation, state)
        self.latest_query_actions.append(latest.copy())
        self.raw_selected_actions.append(raw_action.copy())
        self.temporal_ensemble_actions.append(branch_aligned.copy())
        self.temporal_ensemble_candidate_counts.append(candidate_count)
        self.branch_aligned_actions.append(branch_aligned.copy())
        executed = self.rate_limiter.apply(branch_aligned, state) if self.rate_limiter is not None else branch_aligned
        self.executed_latent_actions.append(executed.copy())
        self._step_index += 1
        return executed

    def _query_model(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.noise_mode == "fixed_per_query":
            seed_everything(self.seed)
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self.policy_action_dim,):
            raise ValueError(
                f"Expected policy state [wrist6,{self.hand_action_representation}"
                f"{self.hand_action_dim}], got {state.shape}"
            )
        observation_images = observation.get("images", {})
        if observation_images is None:
            observation_images = {}
        if not isinstance(observation_images, Mapping):
            raise ValueError("observation['images'] must map camera names to RGB arrays")
        hand_state_key = f"state.{self.hand_action_representation}"
        hand_action_key = f"action.{self.hand_action_representation}"
        policy_observation: dict[str, Any] = {
            "state.eef_position": state[None, 0:3],
            "state.eef_rotation": state[None, 3:6],
            hand_state_key: state[None, 6:],
            "language.instruction": [self.instruction],
        }
        for video_key, camera_name in self.video_camera_names.items():
            image = observation_images.get(camera_name)
            if image is None and len(self.video_camera_names) == 1:
                image = observation.get("image")
            if image is None:
                raise ValueError(
                    f"Being-H checkpoint requires MuJoCo camera {camera_name!r} for policy key {video_key!r}"
                )
            image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{camera_name}: expected HWC RGB image, got {image.shape}")
            policy_observation[video_key] = image[None, ...]
        query_started_at = time.perf_counter()
        result = self.policy.get_action(policy_observation)
        chunk = self._decode_policy_result(result, hand_action_key)
        if chunk.ndim != 2 or chunk.shape[1] != self.policy_action_dim:
            raise ValueError(f"Expected Being-H action chunk [T,{self.policy_action_dim}], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Being-H returned NaN or Inf in its action chunk")
        self.query_latencies_s.append(time.perf_counter() - query_started_at)
        return chunk

    def _register_query_result(self, query_step: int, chunk: np.ndarray) -> None:
        self.predicted_chunks.append(chunk.copy())
        self.query_step_indices.append(query_step)
        query_number = len(self.predicted_chunks)
        if query_number == 1 or query_number % 10 == 0:
            latency_ms = self.query_latencies_s[-1] * 1000.0
            LOGGER.info(f"[policy] query={query_number} env_step={query_step} latency={latency_ms:.1f}ms")

    def _run_sync_query(self, observation: Mapping[str, Any], query_step: int) -> np.ndarray:
        chunk = self._query_model(observation)
        self._register_query_result(query_step, chunk)
        state = np.asarray(observation["state"], dtype=np.float32)
        smoothed = self.chunk_smoother.apply(chunk, state) if self.chunk_smoother is not None else chunk.copy()
        self.smoothed_predicted_chunks.append(smoothed.copy())
        return smoothed

    def _submit_async_query(self, observation: Mapping[str, Any], query_step: int) -> None:
        if self._executor is None:
            raise RuntimeError("Async inference executor is not initialized")
        if self._pending_future is not None:
            raise RuntimeError("Only one Being-H inference may be pending")
        snapshot = {
            "state": np.array(observation["state"], dtype=np.float32, copy=True),
            "image": None
            if observation.get("image") is None
            else np.array(observation["image"], dtype=np.uint8, copy=True),
            "images": {
                camera_name: np.array(image, dtype=np.uint8, copy=True)
                for camera_name, image in observation.get("images", {}).items()
            },
        }
        self._pending_query_step, self._pending_future, self._last_query_launch_step = (
            query_step,
            self._executor.submit(self._query_model, snapshot),
            query_step,
        )
        self.async_submitted_queries += 1

    def _collect_async_query(self, *, block: bool) -> bool:
        future = self._pending_future
        if future is None:
            return False
        if not block and (not future.done()):
            return False
        wait_started = time.perf_counter()
        had_to_wait = block and (not future.done())
        chunk = future.result()
        if had_to_wait:
            self.async_blocking_waits += 1
            self.async_blocking_wait_seconds += time.perf_counter() - wait_started
        query_step = self._pending_query_step
        if query_step is None:
            raise RuntimeError("Pending async query has no query step")
        setattr(self, "_pending_future", None)
        setattr(self, "_pending_query_step", None)
        self._register_query_result(query_step, chunk)
        self.smoothed_predicted_chunks.append(chunk.copy())
        self._chunk = chunk
        self._ensemble_history.append((query_step, chunk))
        return True

    def _has_valid_ensemble_candidate(self) -> bool:
        return any(
            (
                0 <= self._step_index - query_step < min(len(query_chunk), self.temporal_ensemble_max_history)
                for query_step, query_chunk in self._ensemble_history
            )
        )

    def _ensemble_action(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        """Aggregate queried chunks aligned to the current environment step."""
        current_step = self._step_index
        oldest_step = current_step - self.temporal_ensemble_max_history + 1
        setattr(
            self,
            "_ensemble_history",
            [
                (query_step, query_chunk)
                for query_step, query_chunk in self._ensemble_history
                if query_step >= oldest_step and 0 <= current_step - query_step < len(query_chunk)
            ],
        )
        candidates: list[np.ndarray] = []
        ages: list[int] = []
        for query_step, query_chunk in self._ensemble_history:
            age = current_step - query_step
            if age >= self.temporal_ensemble_max_history:
                continue
            candidate = query_chunk[age].copy()
            candidate[3:6] = align_euler_branch(
                candidate[3:6], state[3:6], include_xyz_equivalent=self.include_xyz_equivalent
            )
            candidates.append(candidate)
            ages.append(age)
        if not candidates:
            raise RuntimeError("Temporal ensemble has no prediction for current step")
        weights = np.exp(-self.temporal_ensemble_decay * np.asarray(ages, dtype=np.float64))
        weights /= weights.sum()
        ensemble = np.sum(np.stack(candidates, axis=0).astype(np.float64) * weights[:, None], axis=0).astype(np.float32)
        return (ensemble, len(candidates))

    def _init_representation(self, params: Mapping[str, Any]) -> None:
        data_name = params["data_config_name"]
        self.instruction = params["instruction"]
        self.hand_action_representation = (
            "shadow_joint_position" if data_name in JOINT_ACTION_DATA_CONFIG_NAMES else "z_gesture"
        )
        self.hand_action_dim = (
            DIRECT_JOINT_CONFIG_DIMS[data_name]
            if self.hand_action_representation == "shadow_joint_position" else 24
        )
        self.policy_action_dim, self.replan_every = 6 + self.hand_action_dim, params["replan_every"]
        self.include_xyz_equivalent, self.seed = data_name in ROT6D_DATA_CONFIG_NAMES, int(params["seed"])
        self.noise_mode = params["noise_mode"]
        self.action_selection = params["action_selection"]
        self.inference_mode = params["inference_mode"]
        self.temporal_ensemble_decay = float(params["temporal_ensemble_decay"])
        self.rate_limiter, self.chunk_smoother = params["rate_limiter"], params["chunk_smoother"]
        if self.inference_mode == "async" and self.chunk_smoother is not None and self.chunk_smoother.enabled:
            raise ValueError("Chunk smoothing currently requires synchronous inference")


    def _init_policy(self, params: Mapping[str, Any]) -> None:
        data_name = params["data_config_name"]
        clip_keys = ()
        if params["clip_normalized_wrist_action"]:
            clip_keys = ("action.eef_position", "action.eef_rotation")
            if data_name in JOINT_ACTION_DATA_CONFIG_NAMES:
                clip_keys += ("action.shadow_joint_position",)
        self.policy = BeingHPolicy(str(params["model_path"]), task=PolicyTaskConfig(
            data_config_name=data_name, dataset_name="shadow_grasp_posttrain",
            embodiment_tag="new_embodiment", instruction_template=INSTRUCTION_TEMPLATE),
            device=params["device"], runtime=PolicyRuntimeConfig(
                use_mpg=params["use_mpg"], mpg_refinement_iters=params["mpg_refinement_iters"],
                num_inference_timesteps=params["num_inference_timesteps"], clip_normalized_action_keys=clip_keys,
                enable_rtc=False))
        columns = self.policy.data_config.VIDEO_SOURCE_COLUMNS
        self.video_source_columns = {key: str(columns[key]) for key in self.policy.data_config.VIDEO_KEYS}
        self.video_camera_names = {key: mujoco_camera_name(value) for key, value in self.video_source_columns.items()}
        self.required_cameras = tuple(dict.fromkeys(self.video_camera_names.values()))
        self.policy_wrist_world_origin = np.asarray(
            getattr(self.policy.data_config, "WRIST_WORLD_ORIGIN", (0., 0., 0.)),
            dtype=np.float32,
        )
        if self.policy_wrist_world_origin.shape != (3,) or not np.all(np.isfinite(self.policy_wrist_world_origin)):
            raise ValueError("Data config WRIST_WORLD_ORIGIN must be finite xyz [3]")


    def _init_runtime(self, params: Mapping[str, Any]) -> None:
        model_chunk_length = int(self.policy.action_chunk_length)
        requested_history = params["temporal_ensemble_max_history"] or model_chunk_length
        self.temporal_ensemble_max_history = min(int(requested_history), model_chunk_length)
        if self.action_selection == "temporal_ensemble" and self.replan_every > self.temporal_ensemble_max_history:
            raise ValueError("temporal ensemble query interval cannot exceed its action horizon")
        self._chunk = None
        self._chunk_index = self._step_index = 0
        self._ensemble_history, self.predicted_chunks, self.smoothed_predicted_chunks = [], [], []
        self.query_step_indices, self.observed_latent_states, self.latest_query_actions = [], [], []
        self.temporal_ensemble_actions, self.temporal_ensemble_candidate_counts = [], []
        self.raw_selected_actions = []
        self.branch_aligned_actions = []
        self.executed_latent_actions = []
        self.query_latencies_s = []
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="beingh-inference")
            if self.inference_mode == "async" else None
        )
        self._pending_future = self._pending_query_step = self._last_query_launch_step = None
        self.async_submitted_queries = self.async_blocking_waits = 0
        self.async_blocking_wait_seconds = 0.0


    def _select_temporal_action(self, observation, state):
        if self.inference_mode == "async":
            self._collect_async_query(block=False)
            query_due = (
                self._last_query_launch_step is None
                or self._step_index - self._last_query_launch_step >= self.replan_every
            )
            if query_due:
                if self._pending_future is not None:
                    self._collect_async_query(block=True)
                self._submit_async_query(observation, self._step_index)
            if not self._has_valid_ensemble_candidate():
                if self._pending_future is None:
                    raise RuntimeError("Action horizon expired with no pending async query")
                self._collect_async_query(block=True)
        elif self._chunk is None or self._step_index % self.replan_every == 0:
            self._chunk = self._run_sync_query(observation, self._step_index)
            self._ensemble_history.append((self._step_index, self._chunk))
        if self._chunk is None:
            raise RuntimeError("Being-H has not produced an action chunk")
        latest = self._chunk[0].copy()
        branch, count = self._ensemble_action(state)
        return latest, branch, branch.copy(), count

    def _select_chunk_action(self, observation, state):
        needs_query = (
            self._chunk is None
            or self._chunk_index >= len(self._chunk)
            or self._chunk_index >= self.replan_every
        )
        if needs_query:
            self._chunk, self._chunk_index = self._run_sync_query(observation, self._step_index), 0
        raw = self._chunk[self._chunk_index].copy()
        latest = self._chunk[0].copy()
        self._chunk_index += 1
        branch = raw.copy()
        branch[3:6] = align_euler_branch(
            raw[3:6], state[3:6], include_xyz_equivalent=self.include_xyz_equivalent
        )
        return latest, raw, branch, 1

    def _select_action(self, observation, state):
        if self.action_selection == "temporal_ensemble":
            return self._select_temporal_action(observation, state)
        return self._select_chunk_action(observation, state)
