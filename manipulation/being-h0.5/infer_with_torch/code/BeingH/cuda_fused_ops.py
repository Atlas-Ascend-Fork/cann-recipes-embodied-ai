# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional CUDA/Triton kernels used by GPU feature-parity inference."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


if triton is not None:

    @triton.jit
    def _round_bfloat16_to_float32(value):
        """Force an observable BF16 rounding point in one fused kernel."""
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 bits;
                .reg .b32 lsb;
                mov.b32 bits, $1;
                shr.u32 lsb, bits, 16;
                and.b32 lsb, lsb, 1;
                add.u32 bits, bits, 0x7fff;
                add.u32 bits, bits, lsb;
                and.b32 bits, bits, 0xffff0000;
                mov.b32 $0, bits;
            }
            asm=""",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fused_rotary_kernel(packed, cos, sin, packed_output, metadata):
        query_numel = tl.load(metadata + 0)
        key_numel = tl.load(metadata + 1)
        query_heads = tl.load(metadata + 2)
        key_heads = tl.load(metadata + 3)
        head_dim = tl.load(metadata + 4)
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        query_mask = offsets < query_numel
        query_offsets = tl.where(query_mask, offsets, 0)
        query_dim = query_offsets % head_dim
        query_row_start = query_offsets - query_dim
        query_rotary_dim = tl.where(
            query_dim < head_dim // 2,
            query_dim + head_dim // 2,
            query_dim - head_dim // 2,
        )
        query_sequence = query_offsets // (query_heads * head_dim)
        query_value = tl.load(packed + query_offsets, mask=query_mask, other=0.0)
        query_rotary = tl.load(packed + query_row_start + query_rotary_dim, mask=query_mask, other=0.0)
        query_rotary = tl.where(query_dim < head_dim // 2, -query_rotary, query_rotary)
        query_cos = tl.load(cos + query_sequence * head_dim + query_dim, mask=query_mask, other=0.0)
        query_sin = tl.load(sin + query_sequence * head_dim + query_dim, mask=query_mask, other=0.0)
        query_left = _round_bfloat16_to_float32(query_value.to(tl.float32) * query_cos.to(tl.float32))
        query_right = _round_bfloat16_to_float32(query_rotary.to(tl.float32) * query_sin.to(tl.float32))
        tl.store(packed_output + query_offsets, (query_left + query_right).to(tl.bfloat16), mask=query_mask)
        key_linear = offsets - query_numel
        key_mask = (key_linear >= 0) & (key_linear < key_numel)
        key_offsets = tl.where(key_mask, key_linear, 0)
        key_dim = key_offsets % head_dim
        key_row_start = key_offsets - key_dim
        key_rotary_dim = tl.where(
            key_dim < head_dim // 2,
            key_dim + head_dim // 2,
            key_dim - head_dim // 2,
        )
        key_sequence = key_offsets // (key_heads * head_dim)
        key_value = tl.load(packed + query_numel + key_offsets, mask=key_mask, other=0.0)
        key_rotary = tl.load(packed + query_numel + key_row_start + key_rotary_dim, mask=key_mask, other=0.0)
        key_rotary = tl.where(key_dim < head_dim // 2, -key_rotary, key_rotary)
        key_cos = tl.load(cos + key_sequence * head_dim + key_dim, mask=key_mask, other=0.0)
        key_sin = tl.load(sin + key_sequence * head_dim + key_dim, mask=key_mask, other=0.0)
        key_left = _round_bfloat16_to_float32(key_value.to(tl.float32) * key_cos.to(tl.float32))
        key_right = _round_bfloat16_to_float32(key_rotary.to(tl.float32) * key_sin.to(tl.float32))
        tl.store(
            packed_output + query_numel + key_offsets,
            (key_left + key_right).to(tl.bfloat16),
            mask=key_mask,
        )

    @triton.jit
    def _fused_swiglu_kernel(gate_up, output, output_numel, intermediate_size: tl.constexpr, block_size: tl.constexpr):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < output_numel
        column = offsets % intermediate_size
        row = offsets // intermediate_size
        gate_offset = row * (2 * intermediate_size) + column
        gate = tl.load(gate_up + gate_offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gate_up + gate_offset + intermediate_size, mask=mask, other=0.0).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
        tl.store(output + offsets, (silu * up.to(tl.bfloat16)).to(tl.bfloat16), mask=mask)


def fused_rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply half-rotation RoPE with one CUDA kernel for Q and K."""
    eligible = (
        triton is not None
        and query.device.type == "cuda"
        and (key.device == query.device)
        and (query.ndim == 3)
        and (key.ndim == 3)
        and (cos.ndim == 2)
        and (sin.ndim == 2)
        and (query.shape[0] == key.shape[0] == cos.shape[0] == sin.shape[0])
        and (query.shape[-1] == key.shape[-1] == cos.shape[-1] == sin.shape[-1])
        and (query.shape[-1] % 2 == 0)
        and (query.dtype == key.dtype == cos.dtype == sin.dtype == torch.bfloat16)
        and query.is_contiguous()
        and key.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
    )
    if not eligible:
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)
        return (
            query * cos_expanded + _rotate_half(query) * sin_expanded,
            key * cos_expanded + _rotate_half(key) * sin_expanded,
        )
    packed = torch.cat((query.reshape(-1), key.reshape(-1)))
    packed_output = torch.empty_like(packed)
    metadata = torch.tensor(
        [query.numel(), key.numel(), query.shape[1], key.shape[1], query.shape[-1]],
        dtype=torch.int32,
        device=query.device,
    )
    grid = (triton.cdiv(packed.numel(), 256),)
    _fused_rotary_kernel[grid](packed, cos, sin, packed_output, metadata)
    split = query.numel()
    return packed_output[:split].view_as(query), packed_output[split:].view_as(key)


def fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SiLU(gate) * up with one CUDA kernel."""
    intermediate_size = gate_up.shape[-1] // 2
    eligible = (
        triton is not None
        and gate_up.device.type == "cuda"
        and (gate_up.shape[-1] % 2 == 0)
        and (gate_up.dtype == torch.bfloat16)
        and gate_up.is_contiguous()
    )
    if not eligible:
        gate, up = gate_up.split(intermediate_size, dim=-1)
        return torch.nn.functional.silu(gate) * up
    output = torch.empty((*gate_up.shape[:-1], intermediate_size), dtype=gate_up.dtype, device=gate_up.device)
    grid = (triton.cdiv(output.numel(), 256),)
    _fused_swiglu_kernel[grid](gate_up, output, output.numel(), intermediate_size=intermediate_size, block_size=256)
    return output
