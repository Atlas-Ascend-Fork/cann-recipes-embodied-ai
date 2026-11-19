# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""四档消融计时：不开图 / 仅图模式 / 仅预取 / 图模式加预取。

示例：
    python bench/ablation.py --model-path model_ckpt --graph --prefetch --iters 3
"""

import argparse
import logging
import numpy as np
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import _common

LOGGER = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model_ckpt")
    parser.add_argument("--clip", default=_common.DEFAULT_CLIP)
    parser.add_argument("--t0-us", type=int, default=_common.DEFAULT_T0_US)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    model, processor = _common.build_model(args.model_path)
    model_inputs, data = _common.prepare_inputs(model, processor, args.clip, args.t0_us)
    gt_xy = _common.ground_truth_xy(data)

    pred_xyz, _, _, warm_seconds = _common.run_inference(
        model, model_inputs, args.graph, args.prefetch
    )
    LOGGER.info(f"warmup {warm_seconds:.2f}s minADE={_common.min_ade(pred_xyz, gt_xy):.4f}")

    times = []
    ade = 0.0
    for _ in range(args.iters):
        pred_xyz, _, _, seconds = _common.run_inference(
            model, model_inputs, args.graph, args.prefetch
        )
        times.append(seconds)
        ade = _common.min_ade(pred_xyz, gt_xy)
        LOGGER.info(f"iter {seconds:.2f}s minADE={ade:.4f}")

    LOGGER.info(
        f"ABLATION graph={args.graph} prefetch={args.prefetch} "
        f"mean={np.mean(times):.3f}s times={[round(value, 2) for value in times]} "
        f"minADE={ade:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
