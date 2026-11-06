# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape TorchAir/GE compilation for causal Prefix prefill."""

from __future__ import annotations

import logging
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from BeingH.npu_capture_replay import TensorSpec, _feature_flag_tuple, _module_state, flatten_prefix_cache
from BeingH.npu_prefix_helpers import PrefixInputContext, prepare_prefix_request, prefix_cache_from_flat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)


class PrefixCompileKey:
    inputs: tuple[TensorSpec, ...]
    feature_flags: tuple[tuple[str, Any], ...]
    module_state: tuple[tuple[int, int], ...]


class _PrefixCompileModule(torch.nn.Module):
    """Expose a tensor-only GE boundary and flatten the per-layer KV result."""


    def __init__(self, target: torch.nn.Module, prefix_segment_route: Any) -> None:
        super().__init__()
        setattr(self, "target", target)
        setattr(self, "prefix_segment_route", prefix_segment_route)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 6:
            raise ValueError(f"Prefix compile expects 6 tensors, got {len(inputs)}")
        (
            packed_sequence_und,
            packed_sequence_gen,
            packed_position_ids,
            packed_und_token_indexes,
            packed_gen_token_indexes,
            attention_mask,
        ) = inputs
        prefix_cache = self.target.build_static_prefix_cache(
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            attention_mask=attention_mask,
            prefix_segment_route=self.prefix_segment_route,
        )
        return flatten_prefix_cache(prefix_cache)


@dataclass


class _PrefixCompileEntry:
    compiled: Any
    prefix_length: int


def _materialize_projection_module(module: torch.nn.Module) -> None:
    for getter_name in ("_get_npu_fused_qkv_mot_gen", "_get_npu_fused_gate_up_weight"):
        getter = getattr(module, getter_name, None)
        if callable(getter):
            getter()


class NPUStaticPrefixTorchAirRunner:
    """Compile one bounded set of fixed Prefix shapes with TorchAir/GE.

    Unlike NPUGraph, this runner passes each request's tensors directly into a
    GE compiled callable. No request values or KV outputs are cached.
    """


    def __init__(
        self, target_module: torch.nn.Module, *, max_entries: int = 1, cache_dir: Optional[str] = None
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._target_ref, self._max_entries, self._cache_dir = weakref.ref(target_module), max_entries, cache_dir
        self._entries, self._failed_keys, self._request_lock = {}, {}, threading.Lock()
        self._compile_on_miss = True
        self._last_fallback_reason = None
        self.compile_count = self.cache_hit_count = self.cache_miss_count = 0
        self.cache_full_fallback_count = self.cache_frozen_fallback_count = self.run_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    @property
    def unhealthy(self) -> bool:
        return False

    @property
    def last_fallback_reason(self) -> Optional[str]:
        return self._last_fallback_reason

    def stats(self) -> dict[str, Any]:
        return {
            "max_entries": self._max_entries,
            "entry_count": self.entry_count,
            "compile_count": self.compile_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_full_fallback_count": self.cache_full_fallback_count,
            "cache_frozen_fallback_count": self.cache_frozen_fallback_count,
            "compile_on_miss": self._compile_on_miss,
            "run_count": self.run_count,
            "failed_key_count": len(self._failed_keys),
            "last_fallback_reason": self._last_fallback_reason,
            "unhealthy": False,
            "unhealthy_reason": None,
        }

    def freeze(self) -> None:
        with self._request_lock:
            self._compile_on_miss = False

    def unfreeze(self) -> None:
        with self._request_lock:
            self._compile_on_miss = True

    def forward(
        self,
        *,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_flags: Mapping[str, Any],
        prefix_segment_route: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        request = prepare_prefix_request(
            PrefixInputContext(
                packed_sequence_und, packed_sequence_gen, packed_position_ids,
                packed_und_token_indexes, packed_gen_token_indexes, attention_mask,
                feature_flags, prefix_segment_route,
            ), self._make_key
        )
        if request is None:
            self._last_fallback_reason = "requires_npu_inputs"
            return None
        inputs, flags, key = request
        with self._request_lock:
            result = self._lookup_or_compile(key, inputs, prefix_segment_route, packed_position_ids)
        if result is None:
            return None
        entry, flat_kv = result
        self.run_count += 1
        self._last_fallback_reason = None
        return prefix_cache_from_flat(flat_kv, entry.prefix_length)


    def clear(self) -> None:
        with self._request_lock:
            self._entries.clear()
            self._failed_keys.clear()
            self._last_fallback_reason = None

    def _lookup_or_compile(
        self,
        key: PrefixCompileKey,
        inputs: tuple[torch.Tensor, ...],
        prefix_segment_route: Optional[Any],
        packed_position_ids: torch.Tensor,
    ) -> Optional[tuple[_PrefixCompileEntry, Any]]:
        entry = self._entries.get(key)
        if entry is None:
            self.cache_miss_count += 1
            if key in self._failed_keys:
                self._last_fallback_reason = f"compile_failed:{self._failed_keys[key]}"
                return None
            if not self._compile_on_miss:
                self.cache_frozen_fallback_count += 1
                self._last_fallback_reason = "compile_cache_frozen_miss"
                return None
            if len(self._entries) >= self._max_entries:
                self.cache_full_fallback_count += 1
                self._last_fallback_reason = "compile_cache_full"
                return None
            try:
                compiled = self._compile(prefix_segment_route)
                flat_kv = compiled(*inputs)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                self._failed_keys[key] = reason
                self._last_fallback_reason = f"compile_failed:{reason}"
                logger.exception("Prefix TorchAir compilation failed")
                return None
            entry = _PrefixCompileEntry(compiled=compiled, prefix_length=int(packed_position_ids.shape[0]))
            self._entries[key] = entry
            self.compile_count += 1
            return entry, flat_kv
        self.cache_hit_count += 1
        try:
            return entry, entry.compiled(*inputs)
        except Exception as error:
            reason = f"runtime:{type(error).__name__}: {error}"
            self._failed_keys[key] = reason
            self._last_fallback_reason = reason
            raise RuntimeError("Prefix TorchAir runtime failed after successful compile") from error


    def _target(self) -> torch.nn.Module:
        target = self._target_ref()
        if target is None:
            raise RuntimeError("Prefix TorchAir target module no longer exists")
        return target

    def _materialize_fused_projection_views(self) -> None:
        """Resolve shared projection views before Dynamo starts tracing.

        The inference-only shared-storage path lazily verifies QKV and Gate/Up
        adjacency with ``Tensor.untyped_storage().data_ptr()``.  That Python
        storage inspection is valid in eager mode but cannot be represented in
        a TorchAir graph.  Resolve each view once outside the compile boundary;
        subsequent forwards only read the cached tensor view and create no
        duplicate weight allocation.
        """
        with torch.no_grad():
            for module in self._target().modules():
                _materialize_projection_module(module)

    def _make_key(self, inputs: tuple[torch.Tensor, ...], feature_flags: Mapping[str, Any]) -> PrefixCompileKey:
        return PrefixCompileKey(
            inputs=tuple((TensorSpec.from_tensor(item) for item in inputs)),
            feature_flags=_feature_flag_tuple(feature_flags),
            module_state=_module_state(self._target()),
        )

    def _compile(self, prefix_segment_route: Any) -> Any:
        self._materialize_fused_projection_views()
        module = _PrefixCompileModule(self._target(), prefix_segment_route).eval()
        if self._cache_dir:
            from torch_npu.dynamo.torchair.inference import cache_compile

            cache_dir = os.path.join(self._cache_dir, "prefix")
            os.makedirs(cache_dir, exist_ok=True)
            return cache_compile(module.forward, dynamic=False, cache_dir=cache_dir)
        return torch.compile(module, backend="npu", fullgraph=True, dynamic=False)
