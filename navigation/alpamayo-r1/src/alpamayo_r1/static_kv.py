# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""轨迹专家的静态 KV 路径。

`prompt_cache` 由 VLM 每次推理新建，地址每次都不同；图对象若直接引用它，
捕获只在当次有效。这里把 expert 的 KV 放在固定大小的 buffer 里，按桶分配：
prefill 与扩散 token 写进去之后，后续请求的输入形状恒定，一张图可以跨请求复用。
"""

from dataclasses import dataclass

import torch
from transformers.cache_utils import StaticCache

ALIGN = 1024
MIN_BUCKET = 2048


def bucket_size(prefill: int, n_tokens: int, align: int = ALIGN, minimum: int = MIN_BUCKET) -> int:
    """把 prefill + 扩散 token 向上取整到 align 的整数倍。"""
    needed = prefill + n_tokens
    return max(minimum, ((needed + align - 1) // align) * align)


@dataclass
class StaticKVState:
    cache: StaticCache
    mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor


def _cast_layer_update(original, layer):
    """StaticLayer.update 不做 dtype 转换，这里按 cache 的 dtype 写入。"""

    def update(key_states, value_states, cache_kwargs=None, *args, **kwargs):
        return original(
            key_states.to(layer.keys.dtype),
            value_states.to(layer.values.dtype),
            cache_kwargs,
            *args,
            **kwargs,
        )

    return update


class StaticKVStore:
    """按 (batch, 桶大小) 缓存静态 KV，每个组合只分配一次。"""

    def __init__(self):
        self._states: dict[tuple, StaticKVState] = {}

    @staticmethod
    def _allocate(config, batch, n_tokens, bucket, device) -> StaticKVState:
        cache = StaticCache(config=config, max_cache_len=bucket, device=device)
        for layer in cache.layers:
            layer.update = _cast_layer_update(layer.update, layer)
        return StaticKVState(
            cache=cache,
            mask=torch.zeros(
                (batch, 1, n_tokens, bucket), dtype=torch.float32, device=device
            ),
            position_ids=torch.zeros(
                (3, batch, n_tokens), dtype=torch.int64, device=device
            ),
            cache_position=torch.zeros((n_tokens,), dtype=torch.int64, device=device),
        )

    @staticmethod
    def _fill(state, prompt_cache, prefill, n_tokens, position_ids, device) -> None:
        for index, layer in enumerate(state.cache.layers):
            keys, values = prompt_cache[index]
            if not layer.is_initialized:
                layer.lazy_initialization(keys)
            layer.keys[:, :, :prefill].copy_(keys[:, :, :prefill])
            layer.values[:, :, :prefill].copy_(values[:, :, :prefill])
            layer.keys[:, :, prefill:].zero_()
            layer.values[:, :, prefill:].zero_()
            layer.is_initialized = True
        state.mask.zero_()
        state.mask[:, :, :, prefill + n_tokens:].fill_(torch.finfo(torch.float32).min)
        state.position_ids.copy_(position_ids.to(state.position_ids.dtype))
        state.cache_position.copy_(
            torch.arange(prefill, prefill + n_tokens, device=device, dtype=torch.int64)
        )

    def prepare(self, config, batch, n_tokens, prompt_cache, position_ids, device) -> StaticKVState:
        prefill = prompt_cache.get_seq_length()
        key = (batch, bucket_size(prefill, n_tokens))
        state = self._states.get(key)
        if state is None:
            state = self._allocate(config, batch, n_tokens, key[1], device)
            self._states[key] = state
        self._fill(state, prompt_cache, prefill, n_tokens, position_ids, device)
        return state
