# Adapted from
# https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# （对照组 bench_vision().upstream() 复刻自 Qwen2_5_VLVisionSdpaAttention.forward 的 mask + SDPA 路径）
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""注意力算子微基准：上游写法 vs npu_fusion_attention。

对应 docs/navigation/internvla-n1/infer_with_torch/README.md「适配点 6」的收益数据，同时给出一个反例：
**语言模型侧换融合算子几乎没有收益**（sdpa 本来就在走融合核），
真正有收益的是视觉塔 —— 因为上游在那里物化了一个块对角 mask。

用法：
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/attn_speed.py
"""
import argparse
import logging
import math
import sys
import time

import torch
import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def bench_text(dev, s, hq=28, hkv=4, d=128):
    """语言模型侧：GQA + causal，无显式 mask —— sdpa 已经能走融合核。"""
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(1, hq, s, d, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, hkv, s, d, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, hkv, s, d, dtype=torch.bfloat16, device=dev)
    rep = hq // hkv
    # sparse_mode=3（压缩下三角）要求 atten_mask 固定为 [2048, 2048]，不随 S 变化，
    # 否则报 AclNN_Parameter_Error(EZ1001)。因此本函数只支持 S <= 2048。
    if s > 2048:
        raise ValueError(f"S={s} 超过 sparse_mode=3 压缩 mask 的上限 2048")
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=dev), diagonal=1)

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            q, repeat_kv(k, rep), repeat_kv(v, rep), attn_mask=None, is_causal=True, scale=scale)

    def fused():
        return torch_npu.npu_fusion_attention(
            q, k, v, hq, "BNSD", atten_mask=mask, scale=scale, sparse_mode=3)[0]

    return timeit(sdpa), timeit(fused)


def bench_vision(dev, n_img, h=16, d=80, per_img=784):
    """视觉塔侧：块对角分段。上游物化 mask，本样例用 TND varlen。"""
    total = per_img * n_img
    cu = [per_img * (i + 1) for i in range(n_img)]
    scale = 1.0 / math.sqrt(d)
    q, k, v = (torch.randn(total, h, d, dtype=torch.bfloat16, device=dev) for _ in range(3))

    def upstream():
        m = torch.zeros([1, total, total], device=dev, dtype=torch.bool)
        lo = 0
        for hi in cu:
            m[..., lo:hi, lo:hi] = True
            lo = hi
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), m, dropout_p=0.0)
        return o.squeeze(0).transpose(0, 1).reshape(total, -1)

    def fused():
        return torch_npu.npu_fusion_attention(
            q, k, v, h, "TND", scale=scale,
            actual_seq_qlen=cu, actual_seq_kvlen=cu)[0].reshape(total, -1)

    return total, timeit(upstream, iters=10), timeit(fused, iters=10)


def setup_logging():
    # 只给本脚本的 logger 挂 stdout handler、不配 root：torch_npu 会把每次读环境变量
    # 都记成 root 下的 INFO 日志，root 开到 INFO 会混进基准输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()
    torch.npu.set_device(int(args.device.split(":")[-1]))
    dev = args.device
    torch.manual_seed(0)

    logger.info("=== 语言模型侧（28 头 / 4 KV / d=128, causal, 无显式 mask） ===")
    logger.info("%6s %10s %10s %8s", "S", "sdpa", "融合", "加速")
    for s in (512, 1024, 2048):
        a, b = bench_text(dev, s)
        logger.info("%6d %9.3fms %9.3fms %7.2fx", s, a, b, a / b)
    logger.info("→ 收益有限且随序列变长衰减：sdpa 在此已走融合核，省下的主要是 repeat_kv 的访存。")

    logger.info("\n=== 视觉塔侧（16 头 / d=80，块对角分段，上游物化 mask） ===")
    logger.info("%6s %7s %11s %10s %8s", "图数", "T", "上游", "融合", "加速")
    for n in (1, 4, 9):
        t, a, b = bench_vision(dev, n)
        logger.info("%6d %7d %10.2fms %9.2fms %7.2fx", n, t, a, b, a / b)
    logger.info("→ 评测稳态为 9 张图 / T=7056，即上表最后一行。")


if __name__ == "__main__":
    setup_logging()
    main()
