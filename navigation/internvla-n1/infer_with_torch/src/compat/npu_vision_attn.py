# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""用 npu_fusion_attention 替换 Qwen2.5-VL 视觉塔的注意力，消除巨型 mask。

## 问题

transformers 的 `Qwen2_5_VLVisionSdpaAttention.forward` 为了表达"每张图只在自己的
图块内做注意力"，显式构造了一个块对角的 bool mask 再交给 SDPA：

    attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = True
    F.scaled_dot_product_attention(q, k, v, attention_mask)

InternVLA-N1 的评测里每次前向带 8 帧历史 + 当前帧，seq_length 达到 7056，
于是每层都要分配并填一个 7056x7056 的 mask（约 50MB），而且**视觉塔 32 层各建一次**。
更要命的是 SDPA 一旦收到显式 mask 就无法走融合核，退化成
matmul -> masked_fill(-inf) -> softmax -> matmul 的数学路径。

算子级 profile（单次 generate，7056 图块）显示这条路径的代价：

    aclnnSWhere       37 次   207 ms   10.7%
    aclnnSoftmax      32 次   196 ms   10.1%
    aclnnIsNegInf     32 次   111 ms    5.7%
    aclnnFlashAttentionScore  140 次  17 ms  0.9%   <- 语言模型的注意力，已经是融合的

调用次数 32 正好是视觉塔层数，而语言模型的 140 次 = 28 层 x 5 次前向，说明
**语言模型的 sdpa 本来就在走融合核，视觉塔没有**。

## 方案

这个块对角 mask 就是 packed sequence（varlen）语义，`npu_fusion_attention` 的 TND
布局配 `actual_seq_qlen` / `actual_seq_kvlen` 原生支持，完全不需要物化 mask。

transformers 里已经有一份按 varlen 写的视觉塔前向：`Qwen2_5_VLVisionFlashAttention2.forward`，
也就是官方 evaluator 硬编码 flash_attention_2 时走的那条。它的 q/k/v 本来就是 (T, N, D)，
正好是 TND，只是要用到 flash_attn 的两个接口，而昇腾上装不了 flash_attn。所以这里不另写
前向，只在模块命名空间里给这两个接口换上昇腾实现，再让 sdpa 类借用这个 forward：

    flash_attn_varlen_func         -> npu_fusion_attention(TND)
    apply_rotary_pos_emb_flashatt  -> sdpa 路径的 apply_rotary_pos_emb_vision

rope 用 sdpa 路径的那个函数，是为了和已跑完的全量评测逐位一致；它与 flash 路径的
rope 本来就逐位相同，见 bench/attn_equivalence.py 第 3 项。语言模型仍走 sdpa，不受影响。

微基准（16 头 / head_dim 80，bf16）：

    T= 784   上游  0.87ms  ->  融合 0.12ms   7.4x
    T=3136   上游  9.19ms  ->  融合 0.31ms  29.8x
    T=7056   上游 50.03ms  ->  融合 0.84ms  59.9x

## 数值差异

融合算子与 SDPA 数学路径的累加顺序不同，bf16 下实测最大相对误差约 3.3e-03，
在 bfloat16 的机器精度（约 7.8e-03）之内。因为差异存在，不能要求评测指标逐位一致，
必须用固定种子跑同一批 episode 做逐条比对，见 docs/navigation/internvla-n1/infer_with_torch/README.md。
"""
import math

import torch_npu
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _m

# 换掉的模块级名字。flash_attn 没装时前一个本来就不存在，disable() 要能把它删回去
_PATCHED_NAMES = ("flash_attn_varlen_func", "apply_rotary_pos_emb_flashatt")
_MISSING = object()
# 打补丁前的原值。空 dict 即「未打补丁」；只增删内容不重新绑定，这样不必用 global
_SAVED = {}


def _npu_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, *_max_seqlens):
    """flash_attn_varlen_func 的昇腾实现。max_seqlen 是 flash_attn 要的，TND 布局用不到。"""
    # actual_seq_* 取累积长度且不含起始的 0，与 cu_seqlens[1:] 一致。
    #
    # tolist 是一次 D2H 拷贝，每层都做。曾经尝试按 cu_seqlens.data_ptr() 缓存来省掉它，
    # 但那是错的：张量释放后地址会被复用，不同内容可能落在同一地址，缓存会静默返回
    # 过期的分段。上游前向刚为 max_seqlen 做过一次 .item() 同步，这里只是再拷一个小张量；
    # 视觉塔 q 与 kv 的分段是同一个张量，只拷一次。
    seq_q = cu_seqlens_q[1:].tolist()
    seq_kv = seq_q if cu_seqlens_k is cu_seqlens_q else cu_seqlens_k[1:].tolist()
    return torch_npu.npu_fusion_attention(
        q, k, v, q.shape[1], "TND",
        scale=1.0 / math.sqrt(q.shape[-1]),
        actual_seq_qlen=seq_q,
        actual_seq_kvlen=seq_kv,
    )[0]


def _vision_rope(q, k, cos, sin):
    """apply_rotary_pos_emb_flashatt 的替身：入参是 (1, T, N, D)，sdpa 路径的 rope 要 (T, N, D)。"""
    q, k = _m.apply_rotary_pos_emb_vision(q.squeeze(0), k.squeeze(0), cos, sin)
    return q.unsqueeze(0), k.unsqueeze(0)


def enable():
    """打上补丁。重复调用是幂等的。"""
    if _SAVED:
        return
    _SAVED["forward"] = _m.Qwen2_5_VLVisionSdpaAttention.forward
    for name in _PATCHED_NAMES:
        _SAVED[name] = getattr(_m, name, _MISSING)
    _m.flash_attn_varlen_func = _npu_varlen_attention
    _m.apply_rotary_pos_emb_flashatt = _vision_rope
    _m.Qwen2_5_VLVisionSdpaAttention.forward = _m.Qwen2_5_VLVisionFlashAttention2.forward


def disable():
    """还原为上游实现，用于 A/B 对比。"""
    if not _SAVED:
        return
    _m.Qwen2_5_VLVisionSdpaAttention.forward = _SAVED["forward"]
    for name in _PATCHED_NAMES:
        if _SAVED[name] is _MISSING:
            delattr(_m, name)
        else:
            setattr(_m, name, _SAVED[name])
    _SAVED.clear()
