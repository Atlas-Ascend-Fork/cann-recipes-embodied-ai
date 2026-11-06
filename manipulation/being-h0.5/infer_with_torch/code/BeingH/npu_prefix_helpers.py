# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch


@dataclass(frozen=True)
class PrefixInputContext:
    packed_sequence_und: torch.Tensor
    packed_sequence_gen: torch.Tensor
    packed_position_ids: torch.Tensor
    packed_und_token_indexes: torch.Tensor
    packed_gen_token_indexes: torch.Tensor
    attention_mask: torch.Tensor
    feature_flags: Mapping[str, Any]
    prefix_segment_route: Optional[Any]


def prefix_inputs(context: PrefixInputContext) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]] | None:
    inputs = (
        context.packed_sequence_und, context.packed_sequence_gen, context.packed_position_ids,
        context.packed_und_token_indexes, context.packed_gen_token_indexes, context.attention_mask,
    )
    if any(item.device.type != "npu" for item in inputs):
        return None
    flags = dict(context.feature_flags)
    flags["prefix_segment_route"] = context.prefix_segment_route
    return inputs, flags


def prepare_prefix_request(context: PrefixInputContext, make_key: Callable[..., Any]):
    prepared = prefix_inputs(context)
    if prepared is None:
        return None
    inputs, flags = prepared
    return inputs, flags, make_key(inputs, flags)


def prefix_cache_from_flat(flat_kv: tuple[torch.Tensor, ...], prefix_length: int) -> dict[str, Any]:
    layers = [
        (flat_kv[index], flat_kv[index + 1])
        for index in range(0, len(flat_kv), 2)
    ]
    return {"layers": layers, "prefix_length": prefix_length}


def prepare_prefix_request_from_kwargs(kwargs: Mapping[str, Any], make_key):
    context = PrefixInputContext(
        kwargs["packed_sequence_und"], kwargs["packed_sequence_gen"],
        kwargs["packed_position_ids"], kwargs["packed_und_token_indexes"],
        kwargs["packed_gen_token_indexes"], kwargs["attention_mask"],
        kwargs["feature_flags"], kwargs.get("prefix_segment_route"),
    )
    return prepare_prefix_request(context, make_key)
