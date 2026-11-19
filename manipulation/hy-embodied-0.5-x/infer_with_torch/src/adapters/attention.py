# Copyright (c) 2026 li12345abc123. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-only NPU attention adapter for the two FlashAttention calls used by pinned HY.

Preserves BSND/TND and bottom-right causal semantics. Only segment offsets are
read on Host; Q/K/V and all attention computation remain on NPU.
"""

import inspect
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

CALLS = {'fixed': 0, 'varlen': 0}


@dataclass(frozen=True)
class AttentionOptions:
    """Optional upstream arguments, in their original positional order."""

    dropout_p: float = 0.0
    softmax_scale: object = None
    causal: bool = False
    window_size: tuple = (-1, -1)
    alibi_slopes: object = None
    deterministic: bool = False
    return_attn_probs: bool = False
    softcap: float = 0.0


def _bind_options(positional, keywords):
    names = {field.name for field in fields(AttentionOptions)}
    known = {name: value for name, value in keywords.items() if name in names}
    extra = {name: value for name, value in keywords.items() if name not in names}
    return AttentionOptions(*positional, **known), extra


def _publish_signature(function, required):
    """Expose the same logical signature that the option binder implements."""
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = [inspect.Parameter(name, kind) for name in required]
    for field in fields(AttentionOptions):
        parameters.append(inspect.Parameter(field.name, kind, default=field.default))
    parameters.append(inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD))
    function.__signature__ = inspect.Signature(parameters)


def _check(q, k, v, options, kwargs):
    if any((t.device.type != 'npu' for t in (q, k, v))):
        raise RuntimeError('HY attention requires NPU Q/K/V; CPU fallback is prohibited')
    input_condition = q.device != k.device or q.device != v.device or q.dtype != k.dtype
    if input_condition or q.dtype != v.dtype:
        raise ValueError('Q/K/V device and dtype must match')
    input_condition = (
        options.dropout_p != 0 or tuple(options.window_size) != (-1, -1) or options.alibi_slopes is not None
    )
    input_condition = input_condition or options.softcap != 0 or options.return_attn_probs
    if input_condition or kwargs:
        raise NotImplementedError('This prototype supports only the pinned HY inference call contract')


def flash_attn_func(q, k, v, *args, **kwargs):
    options, extra = _bind_options(args, kwargs)
    _check(q, k, v, options, extra)
    input_condition = q.ndim != 4 or k.ndim != 4 or v.shape != k.shape
    if input_condition or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Expected Q/K/V BSND shapes')
    CALLS['fixed'] += 1
    if q.shape[1] == 0:
        return torch.empty_like(q)
    if k.shape[1] == 0:
        return torch.zeros_like(q)
    hq, hkv = (q.shape[2], k.shape[2])
    if hkv <= 0 or hq % hkv:
        raise ValueError('Query heads must be divisible by KV heads')
    qq = q.transpose(1, 2).contiguous()
    kk = k.transpose(1, 2).contiguous()
    vv = v.transpose(1, 2).contiguous()
    if hq != hkv:
        kk = kk.repeat_interleave(hq // hkv, dim=1)
        vv = vv.repeat_interleave(hq // hkv, dim=1)
    mask = None
    if options.causal:
        nq, nk = (q.shape[1], k.shape[1])
        qi = torch.arange(nq, device=q.device)[:, None] + nk - nq
        kj = torch.arange(nk, device=q.device)[None, :]
        mask = kj <= qi
    output = F.scaled_dot_product_attention(
        qq, kk, vv, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=options.softmax_scale
    )
    if options.causal:
        output = output.masked_fill(~mask.any(dim=-1)[None, None, :, None], 0)
    return output.transpose(1, 2).contiguous()


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs):
    options, extra = _bind_options(args, kwargs)
    _check(q, k, v, options, extra)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError('Expected packed TND shapes')
    qs = cu_seqlens_q.detach().cpu().tolist()
    ks = cu_seqlens_k.detach().cpu().tolist()
    input_condition = len(qs) != len(ks) or len(qs) < 2 or qs[0] != 0
    input_condition = input_condition or ks[0] != 0 or qs[-1] != q.shape[0]
    if input_condition or ks[-1] != k.shape[0]:
        raise ValueError('Invalid packed sequence boundaries')
    if any((b < a for a, b in zip(qs, qs[1:]))) or any((b < a for a, b in zip(ks, ks[1:]))):
        raise ValueError('Packed boundaries must be monotonic')
    if (
        max((b - a for a, b in zip(qs, qs[1:]))) > max_seqlen_q
        or max((b - a for a, b in zip(ks, ks[1:]))) > max_seqlen_k
    ):
        raise ValueError('Packed sequence exceeds declared maximum')
    CALLS['varlen'] += 1
    pieces = []
    for i in range(len(qs) - 1):
        pieces.append(
            flash_attn_func(
                q[qs[i]: qs[i + 1]][None],
                k[ks[i]: ks[i + 1]][None],
                v[ks[i]: ks[i + 1]][None],
                softmax_scale=options.softmax_scale,
                causal=options.causal,
            )[0]
        )
    return torch.cat(pieces, dim=0)


_publish_signature(flash_attn_func, ('q', 'k', 'v'))
_publish_signature(
    flash_attn_varlen_func, ('q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k', 'max_seqlen_q', 'max_seqlen_k')
)
