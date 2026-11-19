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

"""NPU decode regions and modality metadata, owned by one pipeline."""

import contextvars
import functools
import importlib
import threading

import torch

importlib.import_module('torch_npu')  # Register the torch.npu backend.


class ModalityCache:
    """Preserve the frozen text-only path; never cache across decoder calls."""

    def __init__(self):
        self.module = importlib.import_module('transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot')
        self.decoder_type = getattr(self.module, '_HunYuanVLMoTTextModel')
        if getattr(self.decoder_type.forward, '_hy_modality_cache_owner', None) is not None:
            raise RuntimeError('A modality cache is already installed in this process')
        self.original_mask = getattr(self.module, '_mask_apply')
        self.original_forward = self.decoder_type.forward
        self.scope = contextvars.ContextVar('hy_modality_decisions', default=None)
        self.decisions = 0
        self.hits = 0
        self.enabled = False
        self.mask_wrapper = self.mask_apply
        cache = self

        @functools.wraps(self.original_forward)
        def forward(model, *args, **kwargs):
            token = cache.scope.set({})
            try:
                return cache.original_forward(model, *args, **kwargs)
            finally:
                cache.scope.reset(token)

        forward._hy_modality_cache_owner = self
        self.forward_wrapper = forward

    def mask_apply(self, hidden_states, mask, text_funcs, vision_funcs, out_dims=None, padding_mask=None):
        decisions = self.scope.get()
        input_condition = (decisions is not None) and (not torch.is_grad_enabled()) and (hidden_states.shape[1] == 1)
        if input_condition and (padding_mask is None) and (mask is not None):
            key = id(mask)
            if key not in decisions:
                # Retain the actual tensor within the scope, preventing ID reuse.
                decisions[key] = (mask, bool(mask.any()))
                self.decisions += 1
            else:
                self.hits += 1
            if not decisions[key][1]:
                return [function(hidden_states) for function in text_funcs]
        return self.original_mask(hidden_states, mask, text_funcs, vision_funcs, out_dims, padding_mask)

    def enable(self):
        if self.enabled:
            return
        if (
            getattr(self.module, '_mask_apply') is not self.original_mask
            or self.decoder_type.forward is not self.original_forward
        ):
            raise RuntimeError('Another model patch conflicts with the saved baseline')
        setattr(self.module, '_mask_apply', self.mask_wrapper)
        self.decoder_type.forward = self.forward_wrapper
        self.enabled = True

    def disable(self):
        if not self.enabled:
            return
        if (
            getattr(self.module, '_mask_apply') is not self.mask_wrapper
            or self.decoder_type.forward is not self.forward_wrapper
        ):
            raise RuntimeError('Another model patch replaced the owned wrappers')
        setattr(self.module, '_mask_apply', self.original_mask)
        self.decoder_type.forward = self.original_forward
        self.enabled = False


class BlockDecodeGraphs:
    def __init__(self, cache, graph_type, layer_ids):
        self.cache = cache
        self.module = cache.module
        self.original_decoder = cache.decoder_type.forward
        self.decode_scope = contextvars.ContextVar('hy_graph_cached_decode', default=False)
        self.owner_thread = threading.get_ident()
        self.owner_stream = torch.npu.current_stream()
        self.capture_stream = torch.npu.Stream()
        self.entries = {}
        self.replays = 0
        self.enabled = False
        self.graph_type = graph_type
        self.layer_ids = frozenset(layer_ids)
        self.layer_type = self.module.HunYuanVLMoTDecoderLayer
        self.original_layer = self.layer_type.forward
        manager = self

        @functools.wraps(self.original_decoder)
        def decoder(model, *args, **kwargs):
            past = kwargs.get('past_key_values')
            if past is None and len(args) > 3:
                past = args[3]
            token = manager.decode_scope.set(past is not None and past.get_seq_length() > 0)
            try:
                return manager.original_decoder(model, *args, **kwargs)
            finally:
                manager.decode_scope.reset(token)

        self.decoder_wrapper = decoder

        @functools.wraps(self.original_layer)
        def layer_forward(
            layer,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            modality_mask=None,
            **kwargs
        ):
            supported = supports_layer(
                manager,
                cache,
                layer,
                hidden_states,
                past_key_values,
                position_embeddings,
                modality_mask,
                attention_mask,
                kwargs,
            )
            if not supported:
                return manager.original_layer(
                    layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    modality_mask=modality_mask,
                    **kwargs
                )
            attention = layer.self_attn
            cos, sin = position_embeddings

            def pre(hidden, cosine, sine):
                normalized = layer.input_layernorm(hidden)
                shape = (*hidden.shape[:-1], -1, attention.head_dim)
                q = attention.q_proj(normalized).view(shape).transpose(1, 2)
                k = attention.k_proj(normalized).view(shape).transpose(1, 2)
                v = attention.v_proj(normalized).view(shape).transpose(1, 2)
                q, k = getattr(manager.module, '_apply_rotary_pos_emb')(q, k, cosine, sine)
                return attention.query_layernorm(q), attention.key_layernorm(k), v

            q, k, v = manager.region((id(layer), 'pre'), pre, (hidden_states, cos, sin))
            k, v = past_key_values.update(
                k, v, attention.layer_idx, {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            )
            output, _ = getattr(manager.module, '_flash_attention_forward_mot')(
                attention, q, k, v, attention_mask, dropout=0.0, scaling=attention.scaling, **kwargs
            )

            def post(residual, attended):
                projected = attention.o_proj(attended.reshape(*residual.shape[:-1], -1).contiguous())
                after_attention = residual + projected
                return after_attention + layer.mlp(layer.post_attention_layernorm(after_attention))

            return manager.region((id(layer), 'post'), post, (hidden_states, output))[0]

        self.layer_wrapper = layer_forward

    def region(self, key, function, inputs):
        signature = tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device) for x in inputs)
        key = (*key, signature)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 80:
                raise RuntimeError('Unexpected decode-region shape growth')
            static = tuple(torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device) for x in inputs)
            for target, value in zip(static, inputs):
                target.copy_(value)
            self.capture_stream.wait_stream(self.owner_stream)
            with torch.npu.stream(self.capture_stream):
                for _ in range(3):
                    function(*static)
            self.owner_stream.wait_stream(self.capture_stream)
            graph = self.graph_type()
            with torch.npu.graph(graph, stream=self.capture_stream):
                output = function(*static)
                outputs = output if isinstance(output, tuple) else (output,)
            self.owner_stream.wait_stream(self.capture_stream)
            entry = (static, graph, outputs)
            self.entries[key] = entry
        else:
            for target, value in zip(entry[0], inputs):
                target.copy_(value)
        entry[1].replay()
        self.replays += 1
        return entry[2]

    def enable(self):
        if self.enabled:
            return
        if not (self.cache.decoder_type.forward is self.original_decoder):
            raise AssertionError('Owned decoder wrapper changed unexpectedly')
        if not (self.layer_type.forward is self.original_layer):
            raise AssertionError('Owned decoder wrapper changed unexpectedly')
        self.cache.decoder_type.forward = self.decoder_wrapper
        self.layer_type.forward = self.layer_wrapper
        self.enabled = True

    def disable(self):
        if not self.enabled:
            return
        if not (self.layer_type.forward is self.layer_wrapper):
            raise AssertionError('Owned decoder wrapper changed unexpectedly')
        if not (self.cache.decoder_type.forward is self.decoder_wrapper):
            raise AssertionError('Owned decoder wrapper changed unexpectedly')
        self.layer_type.forward = self.original_layer
        self.cache.decoder_type.forward = self.original_decoder
        self.enabled = False


def supports_layer(
    manager, cache, layer, hidden_states, past_key_values, position_embeddings, modality_mask, attention_mask, kwargs
):
    scope = cache.scope.get()
    visual = attention_mask.get('v_seqlens') if isinstance(attention_mask, dict) else None
    supported = (
        id(layer) in manager.layer_ids
        and manager.decode_scope.get()
        and scope is not None
        and not torch.is_grad_enabled()
        and not layer.training
        and hidden_states.device.type == 'npu'
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 3
        and hidden_states.shape[:2] == (1, 1)
        and past_key_values is not None
        and position_embeddings is not None
        and modality_mask is not None
        and modality_mask.shape == (1, 1)
        and isinstance(visual, torch.Tensor)
        and visual.numel() == 0
        and threading.get_ident() == manager.owner_thread
        and torch.npu.current_stream() == manager.owner_stream
        and not kwargs.get('output_attentions', False)
        and not kwargs.get('output_hidden_states', False)
    )
    if supported:
        key = id(modality_mask)
        if key not in scope:
            scope[key] = (modality_mask, bool(modality_mask.any()))
            cache.decisions += 1
        else:
            cache.hits += 1
        supported = not scope[key][1]
    return supported


class SuperGraph(torch.npu.NPUGraph):
    def capture_end(self):
        super().capture_end()
        self.super_kernel_optimize()


class SuperDecode:
    def __init__(self, pipe):
        self.cache = ModalityCache()
        self.graphs = None
        self.cache.enable()
        try:
            layer_type = self.cache.module.HunYuanVLMoTDecoderLayer
            layers = {id(layer) for layer in pipe.model.modules() if isinstance(layer, layer_type)}
            if not layers:
                raise RuntimeError('No pinned HY decoder layers were found')
            self.graphs = BlockDecodeGraphs(self.cache, SuperGraph, layers)
            self.graphs.enable()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.graphs is not None:
            self.graphs.disable()
        self.cache.disable()


def install(pipe):
    """One pipeline per process; callers can explicitly close the owned patches."""
    if getattr(pipe, '_hy_decode_optimizer', None) is not None:
        raise RuntimeError('A decode optimizer is already attached')
    setattr(pipe, '_hy_decode_optimizer', SuperDecode(pipe))
    return pipe
