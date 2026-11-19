# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""vision 塔网格元数据缓存。

`rot_pos_emb` 与 `fast_pos_embed_interpolate` 只依赖图像网格形状 `grid_thw`，
固定分辨率下每次请求的结果逐位相同，缓存后可以省掉重复计算。
"""

import weakref

_ATTACHED = weakref.WeakKeyDictionary()


def cache_key(name: str, grid_thw):
    """以方法名、网格形状、数据类型与设备拼出缓存键。"""
    return (
        name,
        str(grid_thw.dtype),
        grid_thw.device.type,
        tuple(int(value) for value in grid_thw.reshape(-1).tolist()),
    )


def is_attached(model) -> bool:
    """模型上是否已经挂了元数据缓存。"""
    return _ATTACHED.get(model) is not None


class VisionMetadataCache:
    """按网格形状缓存 vision 塔的两份位置编码，只读消费。"""

    def __init__(self, vision_model, max_entries: int = 8):
        self.vision_model = vision_model
        self.max_entries = max_entries
        self.entries = {}
        self.stats = {"hit": 0, "miss": 0}
        self._original_rot_pos_emb = None
        self._original_interpolate = None

    def attach(self):
        model = self.vision_model
        if is_attached(model):
            return self
        _ATTACHED[model] = self
        self._original_rot_pos_emb = model.rot_pos_emb
        self._original_interpolate = model.fast_pos_embed_interpolate

        def rot_pos_emb(grid_thw):
            return self._lookup("rot", self._original_rot_pos_emb, grid_thw)

        def fast_pos_embed_interpolate(grid_thw):
            return self._lookup("interp", self._original_interpolate, grid_thw)

        model.rot_pos_emb = rot_pos_emb
        model.fast_pos_embed_interpolate = fast_pos_embed_interpolate
        return self

    def detach(self):
        model = self.vision_model
        if _ATTACHED.get(model) is not self:
            return self
        model.rot_pos_emb = self._original_rot_pos_emb
        model.fast_pos_embed_interpolate = self._original_interpolate
        del _ATTACHED[model]
        self.entries.clear()
        return self

    def _lookup(self, name, compute, grid_thw):
        key = cache_key(name, grid_thw)
        if key in self.entries:
            self.stats["hit"] += 1
            return self.entries[key]
        self.stats["miss"] += 1
        value = compute(grid_thw)
        if len(self.entries) < self.max_entries:
            self.entries[key] = value
        return value
