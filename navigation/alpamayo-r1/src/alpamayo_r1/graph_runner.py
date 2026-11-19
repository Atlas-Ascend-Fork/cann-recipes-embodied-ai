# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""扩散步的 ACL Graph 捕获与回放。

扩散采样固定执行若干步，每步的输入形状相同，可以把其中一步捕获成图，
后续请求直接回放。捕获失败时退回即时执行，调用方无需感知。
"""

import logging
import time

import torch

from .flags import graph_debug

try:
    import torch_npu  # noqa: F401
except ImportError:  # CPU 环境（单元测试）下只保留即时执行路径
    torch_npu = None


LOGGER = logging.getLogger(__name__)


def _log(message: str) -> None:
    if graph_debug():
        LOGGER.info("[graph] %s", message)


class GraphStepRunner:
    """把形状固定的 step 函数包装成「首次捕获、之后回放」的调用对象。"""

    def __init__(self, step_fn, warmup_iters: int = 2):
        self.step_fn = step_fn
        self.warmup_iters = warmup_iters
        self.graph = None
        self.disabled = torch_npu is None
        self._static_inputs = None
        self._static_output = None

    def __call__(self, **inputs):
        if self.disabled:
            return self.step_fn(**inputs)
        if self.graph is None:
            captured, output = self._capture(inputs)
            if not captured:
                return self.step_fn(**inputs)
            return output
        for name, static in self._static_inputs.items():
            source = inputs[name]
            if static.data_ptr() != source.data_ptr():
                static.copy_(source, non_blocking=True)
        self.graph.replay()
        return self._static_output

    def _capture(self, inputs):
        """返回 (是否捕获成功, 首次输出)；首次输出取预热结果。"""
        try:
            side_stream = torch.npu.Stream()
            side_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side_stream):
                warm_output = None
                for _ in range(self.warmup_iters):
                    warm_output = self.step_fn(**inputs)
            torch.npu.current_stream().wait_stream(side_stream)
            torch.npu.synchronize()

            self._static_inputs = {
                name: value.detach().clone() for name, value in inputs.items()
            }
            graph = torch.npu.NPUGraph()
            started = time.time()
            with torch.npu.graph(graph, capture_error_mode="relaxed"):
                self._static_output = self.step_fn(**self._static_inputs)
            torch.npu.synchronize()
            self.graph = graph
            _log(f"capture ok in {time.time() - started:.2f}s")
            return True, warm_output
        except Exception as exc:  # 捕获期任何异常都退回即时执行
            _log(f"capture failed: {exc!r}, fallback to eager")
            self.disabled = True
            self.graph = None
            self._static_inputs = None
            self._static_output = None
            torch.npu.synchronize()
            return False, None


def wrap_step_fn(step_fn, enable: bool, cache: dict | None = None, key=None):
    """按需包装 step 函数；`cache` 让同一形状的图在多次请求间复用。"""
    if not enable:
        return step_fn
    if cache is None:
        return GraphStepRunner(step_fn)
    runner = cache.get(key)
    if runner is None:
        runner = cache[key] = GraphStepRunner(step_fn)
    return runner
