# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Alpamayo-R1 样例的运行时开关。

全部开关都通过环境变量读取。图模式与权重预取默认关闭，可用环境变量或
`sample_trajectories_from_data_with_vlm_rollout` 的同名参数开启，任一为真即生效；
静态 KV 与 vision 元数据缓存默认打开，其余实验性开关默认关闭。
"""

import os

MIB = 1024 * 1024


def graph_enabled() -> bool:
    return os.getenv("AR1_ENABLE_GRAPH", "0") == "1"


def prefetch_enabled() -> bool:
    return os.getenv("AR1_ENABLE_PREFETCH", "0") == "1"


def graph_debug() -> bool:
    return os.getenv("AR1_GRAPH_DEBUG", "0") == "1"


def static_kv_enabled() -> bool:
    return os.getenv("AR1_STATIC_KV", "1") == "1"


def vision_cache_enabled() -> bool:
    return os.getenv("AR1_VISION_CACHE", "1") == "1"


def expert_attention() -> str:
    return os.getenv("AR1_EXPERT_ATTN", "eager")


def legacy_ops() -> bool:
    return os.getenv("AR1_LEGACY_OPS", "0") == "1"


def prefetch_impl() -> str:
    return os.getenv("AR1_PREFETCH_IMPL", "cmo")


def prefetch_max_bytes() -> int:
    return int(float(os.getenv("AR1_PREFETCH_MAX_MB", "128")) * MIB)


def prefetch_max_layers() -> int:
    return int(os.getenv("AR1_PREFETCH_LAYERS", "1"))
