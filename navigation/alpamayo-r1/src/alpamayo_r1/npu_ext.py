# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""样例新增模块的统一入口，供上游文件以一行导入使用。"""

from .flags import graph_enabled as graph_enabled
from .flags import prefetch_enabled as prefetch_enabled
from .flags import static_kv_enabled as static_kv_enabled
from .flags import vision_cache_enabled as vision_cache_enabled
from .graph_runner import wrap_step_fn as wrap_step_fn
from .prefetch import build_prefetcher as build_prefetcher
from .static_kv import StaticKVStore as StaticKVStore
from .vision_cache import VisionMetadataCache as VisionMetadataCache
