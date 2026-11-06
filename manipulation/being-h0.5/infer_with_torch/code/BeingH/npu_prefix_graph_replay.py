# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape NPU graph replay for the causal Prefix prefill.

The regular static-prefix path computes the 28 decoder layers eagerly for every
request. This module captures that same ``build_static_prefix_cache`` call with
static tensor addresses and replays it after copying new frame/state values into
those addresses. It is deliberately opt-in: a failed or frozen shape never
silently changes the default eager route.
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from BeingH.npu_capture_replay import (
    _UNSAFE_NPU_GRAPH_QUARANTINE,
    NPUCaptureProcessUnhealthyError,
    TensorSpec,
    _feature_flag_tuple,
    _module_state,
    flatten_prefix_cache,
)
from BeingH.npu_prefix_helpers import (
    prefix_cache_from_flat, prefix_inputs, prepare_prefix_request_from_kwargs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)


class PrefixGraphKey:
    inputs: tuple[TensorSpec, ...]
    feature_flags: tuple[tuple[str, Any], ...]
    module_state: tuple[tuple[int, int], ...]


class _PrefixGraphEntry:

    def __init__(
        self,
        *,
        key: PrefixGraphKey,
        graph: Any,
        static_inputs: tuple[torch.Tensor, ...],
        static_prefix_kv: tuple[torch.Tensor, ...],
        prefix_length: int,
    ) -> None:
        self.key, self.graph, self.static_inputs = key, graph, static_inputs
        self.static_prefix_kv, self.prefix_length = static_prefix_kv, prefix_length

    def reset(self) -> None:
        reset = getattr(self.graph, "reset", None)
        if reset is not None:
            reset()


def _prefix_graph_warmup(runner, stream, static_inputs, prefix_segment_route):
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(runner.warmup_iters):
            runner.forward_for_capture(static_inputs, prefix_segment_route)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()


class NPUStaticPrefixGraphRunner:
    """Bounded, serial NPU graph cache for Prefix prefill.

    The runner only accepts the validated single-sample, causal Prefix route
    (``prefix_segment_route is None``). Unsupported, frozen, or cache-full
    shapes return ``None`` so the caller can use the original eager implementation.
    A failure after NPU graph work starts marks the worker unhealthy, matching
    the existing action-suffix graph safety contract.
    """


    def __init__(self, target_module: torch.nn.Module, *, warmup_iters: int = 2, max_entries: int = 1) -> None:
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be positive")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._target_ref, self._warmup_iters, self._max_entries = weakref.ref(target_module), warmup_iters, max_entries
        self._entries, self._failed_keys, self._request_lock = {}, {}, threading.Lock()
        self._capture_on_miss = True
        self._unhealthy_reason = self._last_fallback_reason = None
        self.capture_count = self.cache_hit_count = self.cache_miss_count = 0
        self.cache_full_fallback_count = self.cache_frozen_fallback_count = self.replay_count = 0

    def __del__(self) -> None:
        if getattr(self, "_unhealthy_reason", None) is not None:
            return
        try:
            self._entries.clear()
        except Exception as error:
            logger.debug("Failed to clear NPU prefix graph entries during finalization: %s", error)




    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def unhealthy(self) -> bool:
        return self._unhealthy_reason is not None

    @property
    def unhealthy_reason(self) -> Optional[str]:
        return self._unhealthy_reason

    @property
    def last_fallback_reason(self) -> Optional[str]:
        return self._last_fallback_reason

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "max_entries": self._max_entries,
            "entry_count": self.entry_count,
            "capture_count": self.capture_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_full_fallback_count": self.cache_full_fallback_count,
            "cache_frozen_fallback_count": self.cache_frozen_fallback_count,
            "capture_on_miss": self._capture_on_miss,
            "replay_count": self.replay_count,
            "failed_key_count": len(self._failed_keys),
            "last_fallback_reason": self._last_fallback_reason,
            "unhealthy": self.unhealthy,
            "unhealthy_reason": self.unhealthy_reason,
        }


    @property
    def warmup_iters(self) -> int:
        return self._warmup_iters

    @staticmethod
    def _quarantine_graph(graph: Any) -> None:
        if graph is not None:
            _UNSAFE_NPU_GRAPH_QUARANTINE.append(graph)

    @staticmethod
    def _copy_inputs(destinations: tuple[torch.Tensor, ...], sources: tuple[torch.Tensor, ...]) -> None:
        with torch.inference_mode(False), torch.no_grad():
            for destination, source in zip(destinations, sources, strict=True):
                destination.copy_(source)

    @staticmethod
    def _as_prefix_cache(entry: _PrefixGraphEntry) -> dict[str, Any]:
        return prefix_cache_from_flat(entry.static_prefix_kv, entry.prefix_length)

    def forward_for_capture(
        self, inputs: tuple[torch.Tensor, ...], prefix_segment_route: Optional[Any]
    ) -> Mapping[str, Any]:
        return self._forward(inputs, prefix_segment_route)



    def raise_if_unhealthy(self) -> None:
        if self._unhealthy_reason is not None:
            raise NPUCaptureProcessUnhealthyError(
                f"NPU Prefix graph process is unhealthy; restart the worker: {self._unhealthy_reason}"
            )

    def freeze(self) -> None:
        self.raise_if_unhealthy()
        with self._request_lock:
            self._capture_on_miss = False

    def unfreeze(self) -> None:
        self.raise_if_unhealthy()
        with self._request_lock:
            self._capture_on_miss = True

    def forward(self, **kwargs: Any) -> Optional[dict[str, Any]]:
        """Replay a cached Prefix graph or return None for eager fallback."""
        self.raise_if_unhealthy()
        packed_sequence_und = kwargs["packed_sequence_und"]
        packed_sequence_gen = kwargs["packed_sequence_gen"]
        packed_position_ids = kwargs["packed_position_ids"]
        packed_und_token_indexes = kwargs["packed_und_token_indexes"]
        packed_gen_token_indexes = kwargs["packed_gen_token_indexes"]
        attention_mask = kwargs["attention_mask"]
        feature_flags = kwargs["feature_flags"]
        prefix_segment_route = kwargs.get("prefix_segment_route")
        request = prepare_prefix_request_from_kwargs(kwargs, self._make_key)
        if request is None:
            self._last_fallback_reason = "requires_npu_inputs"
            return None
        inputs, flags, key = request
        with self._request_lock:
            entry = self._lookup_or_capture(key, inputs, prefix_segment_route)
        if entry is None:
            return None
        try:
            self._copy_inputs(entry.static_inputs, inputs)
            entry.graph.replay()
            self.replay_count += 1
            self._last_fallback_reason = None
            return self._as_prefix_cache(entry)
        except NPUCaptureProcessUnhealthyError:
            raise
        except Exception as error:
            self._quarantine_graph(getattr(entry, "graph", None))
            raise self._mark_unhealthy("replay", error) from error


    def clear(self) -> None:
        self.raise_if_unhealthy()
        with self._request_lock:
            if self._entries:
                torch.npu.synchronize()
            for entry in self._entries.values():
                entry.reset()
            self._entries.clear()
            self._failed_keys.clear()
            self._last_fallback_reason = None

    def _lookup_or_capture(
        self, key: PrefixGraphKey, inputs: tuple[torch.Tensor, ...], prefix_segment_route: Optional[Any]
    ) -> _PrefixGraphEntry:
        entry = self._entries.get(key)
        if entry is None:
            self.cache_miss_count += 1
            if key in self._failed_keys:
                self._last_fallback_reason = f"capture_failed:{self._failed_keys[key]}"
                return None
            if not self._capture_on_miss:
                self.cache_frozen_fallback_count += 1
                self._last_fallback_reason = "graph_cache_frozen_miss"
                return None
            if len(self._entries) >= self._max_entries:
                self.cache_full_fallback_count += 1
                self._last_fallback_reason = "graph_cache_full"
                return None
            try:
                entry = self._capture(key=key, inputs=inputs, prefix_segment_route=prefix_segment_route)
            except NPUCaptureProcessUnhealthyError:
                raise
            except Exception as error:
                reason = f"capture:{type(error).__name__}: {error}"
                self._failed_keys[key] = reason
                self._last_fallback_reason = f"capture_failed:{reason}"
                raise self._mark_unhealthy("capture", error) from error
            self._entries[key] = entry
        else:
            self.cache_hit_count += 1
        return entry


    def _target(self) -> torch.nn.Module:
        target = self._target_ref()
        if target is None:
            raise RuntimeError("Prefix graph target module no longer exists")
        return target

    def _mark_unhealthy(self, phase: str, error: Exception) -> NPUCaptureProcessUnhealthyError:
        reason = f"{phase}:{type(error).__name__}: {error}"
        if self._unhealthy_reason is None:
            self._unhealthy_reason = reason
            logger.critical("NPU Prefix graph %s failed: %s", phase, reason)
        return NPUCaptureProcessUnhealthyError(
            f"NPU Prefix graph process is unhealthy; restart the worker: {self._unhealthy_reason}"
        )

    def _make_key(self, inputs: tuple[torch.Tensor, ...], feature_flags: Mapping[str, Any]) -> PrefixGraphKey:
        return PrefixGraphKey(
            inputs=tuple((TensorSpec.from_tensor(item) for item in inputs)),
            feature_flags=_feature_flag_tuple(feature_flags),
            module_state=_module_state(self._target()),
        )

    def _capture(
        self, *, key: PrefixGraphKey, inputs: tuple[torch.Tensor, ...], prefix_segment_route: Optional[Any]
    ) -> _PrefixGraphEntry:
        if inputs[0].device.type != "npu":
            raise RuntimeError("Prefix NPUGraph capture requires NPU tensors")
        graph = None
        npu_work_started = False
        static_inputs: tuple[torch.Tensor, ...]
        static_prefix_kv: tuple[torch.Tensor, ...]
        prefix_length = int(inputs[2].shape[0])
        try:
            with torch.inference_mode(False), torch.no_grad():
                npu_work_started = True
                static_inputs = tuple((item.clone() for item in inputs))
                warmup_stream = torch.npu.Stream()
                _prefix_graph_warmup(self, warmup_stream, static_inputs, prefix_segment_route)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    static_prefix_kv = self._flat_forward(static_inputs, prefix_segment_route)
                torch.npu.synchronize()
        except Exception:
            if npu_work_started:
                self._quarantine_graph(graph)
            raise
        self.capture_count += 1
        return _PrefixGraphEntry(
            key=key,
            graph=graph,
            static_inputs=static_inputs,
            static_prefix_kv=static_prefix_kv,
            prefix_length=prefix_length,
        )

    def _forward(self, inputs: tuple[torch.Tensor, ...], prefix_segment_route: Optional[Any]) -> Mapping[str, Any]:
        target = self._target()
        return target.build_static_prefix_cache(
            packed_sequence_und=inputs[0],
            packed_sequence_gen=inputs[1],
            packed_position_ids=inputs[2],
            packed_und_token_indexes=inputs[3],
            packed_gen_token_indexes=inputs[4],
            attention_mask=inputs[5],
            prefix_segment_route=prefix_segment_route,
        )

    def _flat_forward(
        self, inputs: tuple[torch.Tensor, ...], prefix_segment_route: Optional[Any]
    ) -> tuple[torch.Tensor, ...]:
        return flatten_prefix_cache(self._forward(inputs, prefix_segment_route))
