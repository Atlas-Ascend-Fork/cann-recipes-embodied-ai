# Adapted from
# https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# （对照组 upstream() 复刻自 Qwen2_5_VLVisionSdpaAttention.forward 的 mask + SDPA 路径）
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""验证视觉塔注意力替换的数值等价性。

对应 docs/navigation/internvla-n1/infer_with_torch/README.md「适配点 6」。本脚本回答三个问题：

  1. `npu_fusion_attention(TND)` 与上游「块对角 mask + SDPA」在各种分段形态下是否等价？
  2. 两条路径谁离 fp32 精确解更近？
  3. 上游 sdpa 路径与官方 flash 路径的 rope 函数是否等价？
     （这一条决定了本样例的实现能否算作「忠实复现论文」，见优化说明文档
     docs/navigation/internvla-n1/infer_with_torch/README.md 的等价性论证。）

用法：
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/attn_equivalence.py
"""
import argparse
import logging
import math
import sys

import torch
import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)

N_HEADS, HEAD_DIM = 16, 80          # Qwen2.5-VL 视觉塔：hidden 1280 / 16 头
PER_IMAGE_TOKENS = 784              # 28x28 patch


def _cu(seglens):
    cu = [0]
    for s in seglens:
        cu.append(cu[-1] + s)
    return cu


def upstream(q, k, v, cu, total):
    """上游 Qwen2_5_VLVisionSdpaAttention 的写法：物化块对角 mask 再交给 SDPA。"""
    mask = torch.zeros([1, total, total], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu)):
        mask[..., cu[i - 1]:cu[i], cu[i - 1]:cu[i]] = True
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        mask, dropout_p=0.0)
    return out.squeeze(0).transpose(0, 1).reshape(total, -1)


def fused(q, k, v, cu, total):
    """本样例的替换：TND varlen 布局，不物化 mask。"""
    out = torch_npu.npu_fusion_attention(
        q, k, v, N_HEADS, "TND",
        scale=1.0 / math.sqrt(HEAD_DIM),
        actual_seq_qlen=cu[1:], actual_seq_kvlen=cu[1:])[0]
    return out.reshape(total, -1)


def exact_fp32(q, k, v, cu, total):
    """块对角注意力的 fp32 精确解，逐段计算以避免物化大矩阵。"""
    out = torch.zeros(total, N_HEADS, HEAD_DIM, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for i in range(1, len(cu)):
        a, b = cu[i - 1], cu[i]
        qq = q[a:b].transpose(0, 1).float()
        kk = k[a:b].transpose(0, 1).float()
        vv = v[a:b].transpose(0, 1).float()
        w = torch.softmax(qq @ kk.transpose(-1, -2) * scale, dim=-1)
        out[a:b] = (w @ vv).transpose(0, 1)
    return out.reshape(total, -1)


# 视觉塔实际用到的两类分段：
#   full-attention 层用 cu_seqlens        —— 每张图一段，784 token
#   其余层用 cu_window_seqlens            —— 每张图 16 个窗口，尺寸 64/32/16
def window_seglens(n_img):
    rows = cols = [4, 4, 4, 2]      # 14 = 4+4+4+2（merged 单位），window=112/14/2=4
    per_img = [r * c * 4 for r in rows for c in cols]
    if sum(per_img) != PER_IMAGE_TOKENS:
        raise ValueError(f"窗口分段之和 {sum(per_img)} != 单图图块数 {PER_IMAGE_TOKENS}")
    return per_img * n_img


CASES = [
    ("整图 9x784（full-att 层）", [PER_IMAGE_TOKENS] * 9),
    ("窗口 144 段（window 层）", window_seglens(9)),
    ("单段 1x7056", [7056]),
    ("不等长 [64,128,256,512]", [64, 128, 256, 512]),
]


def flash_rope(x, cos, sin):
    """官方 flash 路径的 rope：按 flash_attn apply_rotary_emb(interleaved=False) 的语义手写。"""
    c = cos.chunk(2, dim=-1)[0].float().unsqueeze(-2)
    s = sin.chunk(2, dim=-1)[0].float().unsqueeze(-2)
    d = c.shape[-1]
    xf = x.float()
    x1, x2 = xf[..., :d], xf[..., d:2 * d]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).type_as(x)


def check_attention(dev, seed):
    logger.info("=== 1/2. 与上游实现的等价性，以及与 fp32 精确解的误差 ===")
    logger.info("%-30s %10s %10s %11s %11s  更准",
                "分段形态", "相对差", "余弦", "SDPA→fp32", "融合→fp32")
    for name, seglens in CASES:
        cu = _cu(seglens)
        total = cu[-1]
        torch.manual_seed(seed)
        q, k, v = (torch.randn(total, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
                   for _ in range(3))
        a = upstream(q, k, v, cu, total).float()
        b = fused(q, k, v, cu, total).float()
        g = exact_fp32(q, k, v, cu, total)
        rel = ((a - b).abs().max() / a.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        e_sdpa = ((a - g).norm() / g.norm()).item()
        e_fuse = ((b - g).norm() / g.norm()).item()
        who = "融合" if e_fuse < e_sdpa else "SDPA"
        logger.info("%-30s %10.3e %10.6f %11.3e %11.3e  %s",
                    name, rel, cos, e_sdpa, e_fuse, who)


def check_rope(dev, seed):
    logger.info("\n=== 3. rope 等价性：sdpa 路径 vs 官方 flash 路径 ===")
    logger.info("（flash_attn 在昇腾上装不了，按 apply_rotary_emb(interleaved=False) 的语义手写）")
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

    torch.manual_seed(seed)
    t = 512
    q = torch.randn(t, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    k = torch.randn(t, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    half = torch.randn(t, HEAD_DIM // 2, device=dev)
    cos = torch.cat([half.cos()] * 2, dim=-1)
    sin = torch.cat([half.sin()] * 2, dim=-1)
    qa, ka = apply_rotary_pos_emb_vision(q, k, cos, sin)
    for name, x, y in (("q", qa, flash_rope(q, cos, sin)), ("k", ka, flash_rope(k, cos, sin))):
        d = (x.float() - y.float()).abs().max().item()
        logger.info("  %s: max|diff| = %.3e   %s", name, d, "逐位相同" if d == 0 else "不相同")


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.npu.set_device(int(args.device.split(":")[-1]))
    check_attention(args.device, args.seed)
    check_rope(args.device, args.seed)


if __name__ == "__main__":
    setup_logging()
    main()
