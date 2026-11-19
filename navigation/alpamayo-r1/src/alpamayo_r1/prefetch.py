# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""decoder 层权重预取。

decode 阶段每生成一个 token 都要把各层权重读一遍，属于访存受限的环节。
预取在层计算之前、于独立流上把该层权重搬进片上缓存，搬运与计算重叠。
权重是静态参数，提交点对它没有数据依赖。

两种实现：
- `cmo`：官方 `torch_npu.npu_prefetch` 接口，默认；
- `touch`：侧流上对权重做跨步触读（读流量约为全量的 1/32），没有 CMO 接口时可用。
"""

import logging

import torch

from .flags import prefetch_impl, prefetch_max_bytes, prefetch_max_layers

try:
    import torch_npu
except ImportError:
    torch_npu = None

LOGGER = logging.getLogger(__name__)
FOREACH_NORM = getattr(torch, "_foreach_norm", None)
TARGET_ATTR = "down_proj"


def target_module(layer):
    """返回层内参与预取的权重模块（MLP 的 down_proj），找不到时返回 None。

    Qwen3-VL 的 decoder 层把 MLP 放在 `layer.mlp` 下，`down_proj` 在 MLP 里面，
    因此先看 `layer.mlp.down_proj`，再兼容直接挂在层上的写法。
    """
    mlp = getattr(layer, "mlp", None)
    module = getattr(mlp, TARGET_ATTR, None) if mlp is not None else None
    if module is not None:
        return module
    module = getattr(layer, TARGET_ATTR, None)
    if module is not None:
        return module
    for name, sub in layer.named_modules():
        if name.split(".")[-1] == TARGET_ATTR:
            return sub
    return None


def target_weight(layer):
    """返回层内参与预取的权重，找不到时返回 None。"""
    module = target_module(layer)
    return getattr(module, "weight", None) if module is not None else None


def should_submit(nbytes: int, max_bytes: int) -> bool:
    """单次预取字节数是否在上限之内。"""
    return nbytes <= max_bytes


class CmoWeightPrefetcher:
    """在层 forward 之前对本层 down_proj 提交一次 CMO 预取。"""

    def __init__(self, layers, max_bytes: int | None = None, max_layers: int | None = None):
        self.layers = list(layers)
        self.max_bytes = prefetch_max_bytes() if max_bytes is None else max_bytes
        self.max_layers = prefetch_max_layers() if max_layers is None else max_layers
        self.stream = None
        self.handles = []
        self.stats = {
            "hooks": 0,
            "submitted": 0,
            "skipped_size": 0,
            "skipped_api": 0,
            "errors": 0,
        }

    def attach(self):
        if torch_npu is None or not hasattr(torch_npu, "npu_prefetch"):
            self.stats["skipped_api"] += 1
            return self
        self.stream = torch.npu.Stream()
        for index, layer in enumerate(self.layers[: self.max_layers]):
            self.handles.append(layer.register_forward_pre_hook(self._make_hook(index)))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stream = None
        return self

    def submit(self, index: int) -> None:
        """为第 index 层提交一次预取，越界或超限时计入统计后返回。"""
        weight = target_weight(self.layers[index])
        if weight is None or weight.device.type != "npu":
            return
        nbytes = weight.numel() * weight.element_size()
        if not should_submit(nbytes, self.max_bytes):
            self.stats["skipped_size"] += 1
            return
        try:
            with torch.npu.stream(self.stream):
                torch_npu.npu_prefetch(weight, None, int(nbytes))
            self.stats["submitted"] += 1
        except RuntimeError as error:
            LOGGER.warning("npu_prefetch failed on layer %d: %s", index, error)
            self.stats["errors"] += 1

    def _make_hook(self, index):
        def hook(_module, _inputs):
            self.stats["hooks"] += 1
            self.submit(index)

        return hook


class TouchWeightPrefetcher:
    """在层 i 计算前，于侧流上触读层 i+1 的权重（跨步视图 + 融合归约）。"""

    def __init__(self, layers, stride: int = 32):
        self.layers = list(layers)
        self.stride = stride
        self.stream = None
        self.handles = []
        self.views = [
            [
                param.reshape(-1)[::stride]
                for param in layer.parameters()
                if param.dim() >= 2 and param.is_contiguous()
            ]
            for layer in self.layers
        ]

    def attach(self):
        self.stream = torch.npu.Stream()
        for index in range(len(self.views)):
            next_index = (index + 1) % len(self.views)
            self.handles.append(
                self.layers[index].register_forward_pre_hook(self._make_hook(next_index))
            )
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stream = None
        return self

    def submit(self, index: int) -> None:
        """在侧流上触读第 index 层的权重视图。"""
        views = self.views[index]
        if not views or FOREACH_NORM is None:
            return
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            FOREACH_NORM(views)

    def _make_hook(self, next_index):
        def hook(_module, _inputs):
            self.submit(next_index)

        return hook


def build_prefetcher(layers, impl: str | None = None):
    """按 `AR1_PREFETCH_IMPL` 选择预取实现，默认走官方 CMO 接口。"""
    name = prefetch_impl() if impl is None else impl
    if name == "touch":
        return TouchWeightPrefetcher(layers)
    return CmoWeightPrefetcher(layers)
