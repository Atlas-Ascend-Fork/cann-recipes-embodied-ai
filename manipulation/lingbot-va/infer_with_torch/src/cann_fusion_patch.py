# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the NPU RoPE and attention paths to WanAttention."""

import functools
import weakref
from collections import OrderedDict

import torch
import torch_npu


_rope_coefficients = OrderedDict()


def clear_rope_coefficients():
    _rope_coefficients.clear()


def prepare_rope(freqs):
    """Cache converted RoPE coefficients for live tensors."""
    key = id(freqs)
    try:
        version = getattr(freqs, '_version')
    except RuntimeError:
        # Inference tensors do not track mutations. Avoid stale cached values.
        return (freqs.real.float().repeat_interleave(2, dim=-1).contiguous(),
                freqs.imag.float().repeat_interleave(2, dim=-1).contiguous())
    entry = _rope_coefficients.get(key)
    if entry is not None and entry[0]() is freqs and entry[1] == version:
        _rope_coefficients.move_to_end(key)
        return entry[2:]

    def released(ref):
        current = _rope_coefficients.get(key)
        if current is not None and current[0] is ref:
            _rope_coefficients.pop(key, None)

    cos = freqs.real.float().repeat_interleave(2, dim=-1).contiguous()
    sin = freqs.imag.float().repeat_interleave(2, dim=-1).contiguous()
    _rope_coefficients[key] = (weakref.ref(freqs, released), version, cos, sin)
    while len(_rope_coefficients) > 4:
        _rope_coefficients.popitem(last=False)
    return cos, sin


def fused_sdpa(q, k, v):
    # Upstream inputs are BSND, and valid cache tokens have already been selected.
    # There is no causal mask at this call site. Do not introduce one.
    return torch_npu.npu_fusion_attention_v3(
        q.contiguous(), k.contiguous(), v.contiguous(), q.shape[2], "BSND",
        scale=q.shape[-1] ** -0.5, keep_prob=1.0, sparse_mode=0,
    )[0]


def fused_rms_norm(norm, x):
    if norm.weight is None or len(norm.normalized_shape) != 1:
        raise ValueError("Only affine last-dimension RMSNorm is supported")
    epsilon = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    return torch_npu.npu_rms_norm(x.contiguous(), norm.weight.contiguous(), epsilon)[0]


def fused_rope(x, freqs):
    # Use interleaved rotation with FP32 RoPE coefficients.
    if x.shape[-1] % 2 or freqs.shape[-1] * 2 != x.shape[-1]:
        raise ValueError("RoPE frequencies must describe adjacent feature pairs")
    cos, sin = prepare_rope(freqs)
    shape, original_dtype = x.shape, x.dtype
    x = x.float().contiguous()
    # A2/A3 interleave accepts BSND coefficients only as 1S1D.
    # CFG supplies distinct BS1D coefficients. Fold batch into sequence so
    # every token retains its own angles without unsupported batch layout.
    if cos.shape[0] != 1:
        if x.ndim != 4 or cos.shape != (shape[0], shape[1], 1, shape[-1]):
            raise ValueError(f"Unsupported RoPE coefficient layout: {shape}, {cos.shape}")
        x = x.reshape(1, shape[0] * shape[1], shape[2], shape[3])
        cos = cos.reshape(1, shape[0] * shape[1], 1, shape[3])
        sin = sin.reshape_as(cos)
    return torch_npu.npu_rotary_mul(
        x, cos, sin, rotary_mode="interleave",
    ).reshape(shape).to(original_dtype)


def install(model_module, *, rms=True, rope=False, attention=False):
    """Install the selected NPU fusion operators and return an uninstall callback."""
    cls = model_module.WanAttention
    if getattr(cls.forward, "_lingbot_cann_fusion", False):
        raise RuntimeError("Fusion overlay already installed")
    original = cls.forward

    @functools.wraps(original)
    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        if q.device.type != "npu" or self.training or torch.is_grad_enabled():
            return original(self, q, k, v, rotary_emb, update_cache, cache_name)
        cache = (self.attn_caches.get(cache_name)
                 if self.attn_caches is not None else None)
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = fused_rms_norm(self.norm_q, query) if rms else self.norm_q(query)
        key = fused_rms_norm(self.norm_k, key) if rms else self.norm_k(key)
        query = query.unflatten(2, (self.heads, -1))
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            if rope:
                query = fused_rope(query, rotary_emb)
                key = fused_rope(key, rotary_emb)
            else:
                def upstream_rope(x):
                    pairs = x.double().reshape(*x.shape[:3], -1, 2)
                    return torch.view_as_real(
                        torch.view_as_complex(pairs) * rotary_emb,
                    ).flatten(3).to(x.dtype)
                query, key = upstream_rope(query), upstream_rope(key)
        slots = None
        if cache is not None and cache["k"] is not None:
            slots = self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
            key, value = cache["k"][:, valid], cache["v"][:, valid]
        hidden = fused_sdpa(query, key, value) if attention else self.attn_op(query, key, value)
        if update_cache == 0 and cache is not None and cache["k"] is not None:
            self.restore_cache(cache_name, slots)
        hidden = hidden.flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hidden))

    setattr(forward, '_lingbot_cann_fusion', True)
    cls.forward = forward

    def uninstall():
        if cls.forward is not forward:
            raise RuntimeError("Attention forward changed after overlay installation")
        cls.forward = original
        clear_rope_coefficients()
    return uninstall
