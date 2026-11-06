# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from native_vae.hand_runtime import NativeHandRuntime
from native_vae.hand_spec import load_native_hand_specs

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_CONFIG = PACKAGE_ROOT / "configs/right_hands.yaml"


@dataclass(frozen=True)


class GeometryRetargeterConfig:
    """Numerical settings for the geometry-only retargeting baseline.

    ``raw`` minimizes only the NativeVAE-compatible 60D geometric error.
    ``stable`` adds normalized joint-space temporal regularization while keeping
    the same geometry, semantic frame, and hard URDF joint limits.
    """

    profile: str = "raw"
    max_iterations: int = 12
    learning_rate: float = 0.8
    tolerance: float = 1e-07
    temporal_weight: float = 0.002
    acceleration_weight: float = 0.0005

    def __post_init__(self) -> None:
        if self.profile not in {"raw", "stable"}:
            raise ValueError("profile must be 'raw' or 'stable'")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.temporal_weight < 0 or self.acceleration_weight < 0:
            raise ValueError("temporal weights must be non-negative")


@dataclass(frozen=True)


class GeometryRetargetResult:
    source_hand: str
    target_hand: str
    target_q: np.ndarray
    source_geometry: np.ndarray
    target_geometry: np.ndarray
    geometry_rmse: float
    geometry_max_error: float
    objective: float
    iterations: int
    converged: bool
    elapsed_s: float

    def metadata(self) -> dict[str, Any]:
        return {
            "source_hand": self.source_hand,
            "target_hand": self.target_hand,
            "geometry_rmse": self.geometry_rmse,
            "geometry_max_error": self.geometry_max_error,
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)


class GeometryRetargetBatchResult:
    source_hand: str
    target_hand: str
    target_q: np.ndarray
    source_geometry: np.ndarray
    target_geometry: np.ndarray
    per_frame_geometry_rmse: np.ndarray
    per_frame_geometry_max_error: np.ndarray
    geometry_rmse: float
    geometry_max_error: float
    objective: float
    iterations: int
    converged: bool
    elapsed_s: float

    @property
    def batch_size(self) -> int:
        return int(self.target_q.shape[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "source_hand": self.source_hand,
            "target_hand": self.target_hand,
            "batch_size": self.batch_size,
            "geometry_rmse": self.geometry_rmse,
            "geometry_max_error": self.geometry_max_error,
            "per_frame_geometry_rmse": self.per_frame_geometry_rmse.tolist(),
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "elapsed_s": self.elapsed_s,
        }


@dataclass


class _StreamState:
    previous_q: torch.Tensor
    previous_previous_q: torch.Tensor | None = None


@dataclass(frozen=True)
class _RetargetRequest:
    q: Any
    source_hand: str
    target_hand: str
    stream: str
    warm_start: Any
    update_state: bool
    batch: bool = False


@dataclass(frozen=True)
class _RetargetContext:
    started_at: float
    source_hand: str
    target_hand: str
    source_q: torch.Tensor
    source_geometry: torch.Tensor
    key: tuple[str, str, str]
    state: _StreamState | None
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None
    initial_q: torch.Tensor | None = None
    update_state: bool = True
    batch_size: int = 1


@dataclass(frozen=True)
class _OptimizationRequest:
    source_geometry: torch.Tensor
    initial_q: torch.Tensor
    target_hand: str
    lower: torch.Tensor
    upper: torch.Tensor
    regularizer: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class _OptimizationOutput:
    target_q: torch.Tensor
    target_geometry: torch.Tensor
    objective: float
    calls: int


class GeometryRetargeter:
    """Retarget native joint angles by optimizing NativeVAE's hand geometry.

    The objective uses the exact normalized 60D ``x_gesture`` representation
    used before NativeVAE encoding: five finger roots followed by three chain
    vectors per finger, all expressed in the fixed semantic palm frame and
    divided by that hand's palm radius. No learned VAE weights are loaded.
    """

    def __init__(
        self,
        *,
        hand_config: str | Path = DEFAULT_HAND_CONFIG,
        device: str | torch.device = "cpu",
        config: GeometryRetargeterConfig | None = None,
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and (not torch.cuda.is_available()):
            requested = torch.device("cpu")
        setattr(self, "device", requested)
        setattr(self, "config", config or GeometryRetargeterConfig())
        setattr(self, "hand_config", Path(hand_config).expanduser().resolve())
        setattr(self, "specs", load_native_hand_specs(self.hand_config))
        setattr(
            self,
            "runtimes",
            {name: NativeHandRuntime.build(spec, device=requested) for name, spec in self.specs.items()},
        )
        self._streams: dict[tuple[str, str, str], _StreamState] = {}

    @property
    def hand_names(self) -> tuple[str, ...]:
        return tuple(self.runtimes)

    @staticmethod
    def _normalize_q(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return 2.0 * (q - lower) / (upper - lower).clamp_min(1e-08) - 1.0

    @staticmethod
    def _bounded_q(unconstrained: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + 0.5 * (torch.tanh(unconstrained) + 1.0) * (upper - lower)

    @staticmethod
    def _unconstrained_q(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        normalized = GeometryRetargeter._normalize_q(q, lower, upper)
        return torch.atanh(normalized.clamp(-0.999, 0.999))

    @staticmethod
    def _identity_single(ctx: _RetargetContext) -> GeometryRetargetResult:
        values = ctx.source_q[0].detach().cpu().numpy().astype(np.float32)
        geometry = ctx.source_geometry[0].cpu().numpy().astype(np.float32)
        return GeometryRetargetResult(ctx.source_hand, ctx.target_hand, values, geometry, geometry.copy(),
                                      0.0, 0.0, 0.0, 0, True, time.perf_counter() - ctx.started_at)

    @staticmethod
    def _identity_batch(ctx: _RetargetContext) -> GeometryRetargetBatchResult:
        values = ctx.source_q.detach().cpu().numpy().astype(np.float32)
        geometry = ctx.source_geometry.cpu().numpy().astype(np.float32)
        zeros = np.zeros(ctx.batch_size, dtype=np.float32)
        return GeometryRetargetBatchResult(ctx.source_hand, ctx.target_hand, values, geometry, geometry.copy(),
                                           zeros, zeros.copy(), 0.0, 0.0, 0.0, 0, True,
                                           time.perf_counter() - ctx.started_at)

    def joint_names(self, hand: str) -> tuple[str, ...]:
        return self._runtime(hand).spec.active_joint_names

    def reset(self, stream: str | None = None) -> None:
        if stream is None:
            self._streams.clear()
            return
        for key in [key for key in self._streams if key[2] == stream]:
            del self._streams[key]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "geometry_optimization",
            "geometry": "NativeVAE-compatible normalized 60D semantic chain vectors",
            "hand_config": str(self.hand_config),
            "device": str(self.device),
            "config": asdict(self.config),
            "hands": {
                name: {"joint_names": list(runtime.spec.active_joint_names), "palm_radius": runtime.palm_radius}
                for name, runtime in self.runtimes.items()
            },
        }

    def geometry(self, q: np.ndarray | torch.Tensor | Sequence[float], hand: str) -> torch.Tensor:
        runtime = self._runtime(hand)
        values = self._q_tensor(q, hand)
        return runtime.kinematic_chain_gesture(values)["x_gesture"] / float(runtime.palm_radius)

    def retarget(self, q, source_hand: str, target_hand: str, *, stream: str = "default",
                 warm_start=None, update_state: bool = True) -> GeometryRetargetResult:
        ctx = self._prepare_context(_RetargetRequest(q, source_hand, target_hand, stream, warm_start, update_state))
        if source_hand == target_hand:
            return self._identity_single(ctx)
        regularizer = self._single_regularizer(ctx.state, ctx.lower, ctx.upper)
        output = self._optimize_geometry(_OptimizationRequest(
            ctx.source_geometry, ctx.initial_q, target_hand, ctx.lower, ctx.upper, regularizer))
        return self._finish_single(ctx, output)

    def retarget_batch(self, q, source_hand: str, target_hand: str, *, stream: str = "action_chunk",
                       warm_start=None, update_state: bool = True) -> GeometryRetargetBatchResult:
        ctx = self._prepare_context(_RetargetRequest(q, source_hand, target_hand, stream, warm_start,
                                                     update_state, batch=True))
        if source_hand == target_hand:
            return self._identity_batch(ctx)
        regularizer = self._batch_regularizer(ctx.state, ctx.lower, ctx.upper)
        output = self._optimize_geometry(_OptimizationRequest(
            ctx.source_geometry, ctx.initial_q, target_hand, ctx.lower, ctx.upper, regularizer))
        return self._finish_batch(ctx, output)

    def retarget_chunk(
        self,
        q: np.ndarray | torch.Tensor,
        source_hand: str,
        target_hand: str,
        *,
        stream: str = "action",
        update_state: bool = True,
    ) -> list[GeometryRetargetResult]:
        values = np.asarray(q, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        expected = len(self.joint_names(source_hand))
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"{source_hand}: expected chunk [T,{expected}], got {values.shape}")
        return [
            self.retarget(frame, source_hand, target_hand, stream=stream, update_state=update_state) for frame in values
        ]

    def project_motion_limits(self, source_limits: np.ndarray, source_hand: str, target_hand: str) -> np.ndarray:
        """Project absolute joint-step limits by semantic slot and range ratio."""
        values = np.asarray(source_limits, dtype=np.float32)
        source_runtime = self._runtime(source_hand)
        target_runtime = self._runtime(target_hand)
        if values.shape != (len(source_runtime.spec.active_joint_names),):
            raise ValueError(
                f"{source_hand}: expected limits [{len(source_runtime.spec.active_joint_names)}], got {values.shape}"
            )
        source_lower, source_upper = self._limits(source_runtime)
        target_lower, target_upper = self._limits(target_runtime)
        source_ranges = (source_upper - source_lower)[0].cpu().numpy()
        target_ranges = (target_upper - target_lower)[0].cpu().numpy()
        fractions_by_semantic: dict[tuple[str, int], float] = {}
        fractions_by_finger: dict[str, list[float]] = {}
        for index, name in enumerate(source_runtime.spec.active_joint_names):
            semantic = source_runtime.spec.joint_semantics[name]
            fraction = float(values[index] / max(float(source_ranges[index]), 1e-08))
            fractions_by_semantic[semantic.finger, semantic.slot] = fraction
            fractions_by_finger.setdefault(semantic.finger, []).append(fraction)
        projected = np.empty(len(target_runtime.spec.active_joint_names), dtype=np.float32)
        for index, name in enumerate(target_runtime.spec.active_joint_names):
            semantic = target_runtime.spec.joint_semantics[name]
            fraction = fractions_by_semantic.get((semantic.finger, semantic.slot))
            if fraction is None:
                fraction = float(np.median(fractions_by_finger[semantic.finger]))
            projected[index] = max(fraction * float(target_ranges[index]), 1e-05)
        return projected

    def _prepare_context(self, request: _RetargetRequest) -> _RetargetContext:
        started_at = time.perf_counter()
        source_q = self._q_tensor(request.q, request.source_hand)
        if request.batch and source_q.shape[0] <= 0:
            raise ValueError("retarget_batch requires at least one pose")
        if not request.batch and source_q.shape[0] != 1:
            raise ValueError("retarget() accepts one pose; use retarget_batch() for [T,D]")
        source_geometry = self.geometry(source_q, request.source_hand).detach()
        key = (request.source_hand, request.target_hand, str(request.stream))
        if request.source_hand == request.target_hand:
            return _RetargetContext(started_at, request.source_hand, request.target_hand,
                                    source_q, source_geometry, key, None,
                                    update_state=request.update_state, batch_size=int(source_q.shape[0]))
        runtime = self._runtime(request.target_hand)
        lower, upper = self._limits(runtime)
        state = self._streams.get(key)
        if request.warm_start is not None:
            initial = self._q_tensor(request.warm_start, request.target_hand)
            if request.batch and initial.shape[0] != source_q.shape[0]:
                raise ValueError("Batch warm_start must have the same number of frames")
        else:
            initial = self._semantic_initial_guess(source_q, request.source_hand, request.target_hand)
            if request.batch and state is not None:
                initial[0] = state.previous_q[0]
            elif state is not None:
                initial = state.previous_q.detach().clone()
        return _RetargetContext(started_at, request.source_hand, request.target_hand, source_q,
                                source_geometry, key, state, lower, upper,
                                initial.clamp(lower, upper), request.update_state, int(source_q.shape[0]))

    def _finish_single(self, ctx: _RetargetContext, output: _OptimizationOutput) -> GeometryRetargetResult:
        difference = output.target_geometry - ctx.source_geometry
        rmse = float(torch.sqrt(torch.mean(difference**2)).cpu())
        max_error = float(torch.max(torch.abs(difference)).cpu())
        if ctx.update_state:
            previous = None if ctx.state is None else ctx.state.previous_q.detach().clone()
            self._streams[ctx.key] = _StreamState(output.target_q.clone(), previous)
        return GeometryRetargetResult(ctx.source_hand, ctx.target_hand,
            output.target_q[0].detach().cpu().numpy().astype(np.float32),
            ctx.source_geometry[0].cpu().numpy().astype(np.float32),
            output.target_geometry[0].detach().cpu().numpy().astype(np.float32),
            rmse, max_error, output.objective, output.calls,
            bool(np.isfinite(output.objective)), time.perf_counter() - ctx.started_at)

    def _finish_batch(self, ctx: _RetargetContext, output: _OptimizationOutput) -> GeometryRetargetBatchResult:
        difference = output.target_geometry - ctx.source_geometry
        per_frame_rmse = torch.sqrt(torch.mean(difference**2, dim=1)).cpu().numpy().astype(np.float32)
        per_frame_max = torch.max(torch.abs(difference), dim=1).values.cpu().numpy().astype(np.float32)
        rmse = float(torch.sqrt(torch.mean(difference**2)).cpu())
        max_error = float(torch.max(torch.abs(difference)).cpu())
        if ctx.update_state:
            previous = output.target_q[-1:].clone()
            prior = output.target_q[-2:-1].clone() if ctx.batch_size >= 2 else None
            if prior is None and ctx.state is not None:
                prior = ctx.state.previous_q.detach().clone()
            self._streams[ctx.key] = _StreamState(previous, prior)
        return GeometryRetargetBatchResult(ctx.source_hand, ctx.target_hand,
            output.target_q.detach().cpu().numpy().astype(np.float32),
            ctx.source_geometry.cpu().numpy().astype(np.float32),
            output.target_geometry.detach().cpu().numpy().astype(np.float32),
            per_frame_rmse, per_frame_max, rmse, max_error, output.objective,
            output.calls, bool(np.isfinite(output.objective)), time.perf_counter() - ctx.started_at)

    def _optimize_geometry(self, request: _OptimizationRequest):
        source_geometry = request.source_geometry
        initial_q = request.initial_q
        target_hand = request.target_hand
        lower, upper, regularizer = request.lower, request.upper, request.regularizer
        unconstrained = torch.nn.Parameter(self._unconstrained_q(initial_q, lower, upper))
        optimizer = torch.optim.LBFGS(
            [unconstrained],
            lr=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            tolerance_grad=self.config.tolerance,
            tolerance_change=self.config.tolerance,
            line_search_fn="strong_wolfe",
        )
        calls = 0
        objective = float("inf")

        def closure():
            nonlocal calls, objective
            optimizer.zero_grad(set_to_none=True)
            target_q = self._bounded_q(unconstrained, lower, upper)
            loss = torch.mean((self.geometry(target_q, target_hand) - source_geometry) ** 2) + regularizer(target_q)
            loss.backward()
            calls += 1
            objective = float(loss.detach().cpu())
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            target_q = self._bounded_q(unconstrained, lower, upper)
            target_geometry = self.geometry(target_q, target_hand)
        return _OptimizationOutput(target_q.detach(), target_geometry.detach(), objective, calls)

    def _single_regularizer(self, state, lower, upper):
        def regularizer(target_q):
            if self.config.profile != "stable" or state is None:
                return target_q.new_zeros(())
            target_norm = self._normalize_q(target_q, lower, upper)
            previous_norm = self._normalize_q(state.previous_q, lower, upper)
            value = self.config.temporal_weight * torch.mean((target_norm - previous_norm) ** 2)
            if state.previous_previous_q is not None:
                previous_previous_norm = self._normalize_q(state.previous_previous_q, lower, upper)
                acceleration = target_norm - 2.0 * previous_norm + previous_previous_norm
                value = value + self.config.acceleration_weight * torch.mean(acceleration ** 2)
            return value
        return regularizer

    def _batch_regularizer(self, state, lower, upper):
        def regularizer(target_q):
            if self.config.profile != "stable":
                return target_q.new_zeros(())
            target_norm = self._normalize_q(target_q, lower, upper)
            if state is None:
                velocity = target_norm[1:] - target_norm[:-1]
                acceleration = target_norm[2:] - 2.0 * target_norm[1:-1] + target_norm[:-2]
            else:
                previous_norm = self._normalize_q(state.previous_q, lower, upper)
                extended = torch.cat([previous_norm, target_norm], dim=0)
                velocity = extended[1:] - extended[:-1]
                if state.previous_previous_q is None:
                    acceleration = extended[2:] - 2.0 * extended[1:-1] + extended[:-2]
                else:
                    previous_previous_norm = self._normalize_q(state.previous_previous_q, lower, upper)
                    extended = torch.cat([previous_previous_norm, previous_norm, target_norm], dim=0)
                    acceleration = extended[2:] - 2.0 * extended[1:-1] + extended[:-2]
            value = target_q.new_zeros(())
            if len(velocity):
                value = value + self.config.temporal_weight * torch.mean(velocity**2)
            if len(acceleration):
                value = value + self.config.acceleration_weight * torch.mean(acceleration**2)
            return value
        return regularizer

    def _runtime(self, hand: str) -> NativeHandRuntime:
        if hand not in self.runtimes:
            raise KeyError(f"Unknown hand {hand!r}; available={self.hand_names}")
        return self.runtimes[hand]

    def _q_tensor(self, q: np.ndarray | torch.Tensor | Sequence[float], hand: str) -> torch.Tensor:
        value = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        expected = len(self.joint_names(hand))
        if value.ndim != 2 or value.shape[1] != expected:
            raise ValueError(f"{hand}: expected q [B,{expected}], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{hand}: q contains NaN or Inf")
        return value

    def _limits(self, runtime: NativeHandRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(runtime.spec.active_joint_names)
        return (
            runtime.q_lower[:count].to(self.device).unsqueeze(0),
            runtime.q_upper[:count].to(self.device).unsqueeze(0),
        )

    def _semantic_initial_guess(self, source_q: torch.Tensor, source_hand: str, target_hand: str) -> torch.Tensor:
        source_runtime = self._runtime(source_hand)
        target_runtime = self._runtime(target_hand)
        source_lower, source_upper = self._limits(source_runtime)
        target_lower, target_upper = self._limits(target_runtime)
        source_norm = self._normalize_q(source_q.clamp(source_lower, source_upper), source_lower, source_upper)
        by_semantic = {
            (
                source_runtime.spec.joint_semantics[name].finger,
                source_runtime.spec.joint_semantics[name].slot,
            ): source_norm[:, index]
            for index, name in enumerate(source_runtime.spec.active_joint_names)
        }
        by_finger: dict[str, list[torch.Tensor]] = {}
        for (finger, _), value in by_semantic.items():
            by_finger.setdefault(finger, []).append(value)
        target_norm_values = []
        for name in target_runtime.spec.active_joint_names:
            semantic = target_runtime.spec.joint_semantics[name]
            key = (semantic.finger, semantic.slot)
            if key in by_semantic:
                value = by_semantic[key]
            else:
                values = by_finger.get(semantic.finger, [])
                value = (
                    torch.stack(values, dim=0).mean(dim=0) if values else source_norm.new_zeros(source_norm.shape[0])
                )
            target_norm_values.append(value)
        target_norm = torch.stack(target_norm_values, dim=1)
        return target_lower + 0.5 * (target_norm + 1.0) * (target_upper - target_lower)
