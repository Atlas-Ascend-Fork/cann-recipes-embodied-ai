# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache immutable inputs used by the inference server."""
import ast
from collections import OrderedDict, namedtuple
import functools
import inspect
from pathlib import Path
import textwrap

import torch

TensorSignature = namedtuple('TensorSignature', 'identity version device dtype shape stride')


def signature(tensor):
    try:
        version = getattr(tensor, '_version')
    except RuntimeError:
        # inference_mode tensors have no mutation counter: do not cache.
        return None
    return TensorSignature(id(tensor), version, tensor.device, tensor.dtype,
                           tuple(tensor.shape), tuple(tensor.stride()))


class TensorMemo:
    def __init__(self, limit=4):
        self.limit = limit
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()

    def call(self, function, tensors, tag=()):
        keys = tuple(signature(t) for t in tensors)
        if torch.is_grad_enabled() or any(k is None for k in keys):
            return function()
        key = (tag, keys)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][1]
        self.misses += 1
        result = function()
        # Keep inputs alive while the cache entry is used.
        self.entries[key] = (tuple(tensors), result)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)
        return result


class State:
    def __init__(self):
        self.grid = OrderedDict()
        self.text = TensorMemo()
        self.batch_grid = TensorMemo()
        self.projection = TensorMemo()
        self.rope = TensorMemo()

    def clear(self):
        self.grid.clear()
        for name in ('text', 'batch_grid', 'projection', 'rope'):
            getattr(self, name).clear()


def state(server):
    if not hasattr(server, '_constant_cache_state'):
        setattr(server, '_constant_cache_state', State())
    return getattr(server, '_constant_cache_state')


def constant_grid(server, *args, **kwargs):
    key = (args, tuple(sorted(kwargs.items())), str(server.device))
    entries = state(server).grid
    if key not in entries:
        original = getattr(server, '_constant_original_mesh')
        entries[key] = original(*args, **kwargs).to(server.device)
        while len(entries) > 4:
            entries.popitem(last=False)
    entries.move_to_end(key)
    return entries[key]


def constant_text(server, cfg=False):
    tensors = [server.prompt_embeds]
    if cfg:
        tensors.append(server.negative_prompt_embeds)

    def build():
        parts = [t.to(server.dtype).clone() for t in tensors]
        return torch.cat(parts, dim=0) if cfg else parts[0]
    return state(server).text.call(build, tensors, (str(server.dtype), cfg))


def batch_grid(server, grid, cfg):
    return state(server).batch_grid.call(
        lambda: grid[None].repeat(2, 1, 1) if cfg else grid[None], [grid], cfg)


class Rewrite(ast.NodeTransformer):
    def __init__(self):
        self.counts = [0, 0, 0, 0]

    def visit(self, node):
        if isinstance(node, ast.Call):
            return self.visit_call(node)
        if isinstance(node, ast.Subscript):
            return self.visit_subscript(node)
        return super().visit(node)

    def visit_call(self, node):
        expression = ast.unparse(node)
        if expression == 'self.prompt_embeds.to(self.dtype).clone()':
            self.counts[0] += 1
            return ast.parse('constant_text(self)').body[0].value
        cfg_expression = ('torch.cat([self.prompt_embeds.to(self.dtype).clone(), '
                          'self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)')
        if expression == cfg_expression:
            self.counts[1] += 1
            return ast.parse('constant_text(self, cfg=True)').body[0].value
        is_mesh_call = (isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'to'
                        and isinstance(node.func.value, ast.Call)
                        and isinstance(node.func.value.func, ast.Name)
                        and node.func.value.func.id == 'get_mesh_id')
        if is_mesh_call and len(node.args) == 1 and ast.unparse(node.args[0]) == 'self.device':
            self.counts[2] += 1
            call = node.func.value
            return ast.Call(func=ast.Name(id='constant_grid', ctx=ast.Load()),
                            args=[ast.Name(id='self', ctx=ast.Load())] + call.args,
                            keywords=call.keywords)
        if expression == "input_dict['grid_id'][None].repeat(2, 1, 1)":
            self.counts[3] += 1
            return ast.parse("batch_grid(self, input_dict['grid_id'], True)").body[0].value
        return self.generic_visit(node)

    def visit_subscript(self, node):
        if ast.unparse(node) == "input_dict['grid_id'][None]":
            self.counts[3] += 1
            return ast.parse("batch_grid(self, input_dict['grid_id'], False)").body[0].value
        return self.generic_visit(node)


def memoize_module(module, memo):
    original = module.forward

    @functools.wraps(original)
    def forward(x, *args, **kwargs):
        extra_arguments = bool(args or kwargs)
        if module.training or extra_arguments or torch.is_grad_enabled():
            return original(x, *args, **kwargs)
        parameters = list(module.parameters())
        if any(p.requires_grad for p in parameters):
            return original(x)
        # Track weight changes/device transfers as well as input mutations.
        dependencies = parameters + list(module.buffers())
        dependencies += _optional_module_buffers(module)
        return memo.call(lambda: original(x), [x] + dependencies)
    module.forward = forward


def _optional_module_buffers(module):
    names = ('f_freqs_base', 'h_freqs_base', 'w_freqs_base')
    return [getattr(module, name) for name in names if hasattr(module, name)]


def _make_guarded(original, candidate):
    @functools.wraps(original)
    def guarded(self, *args, **kwargs):
        if torch.is_grad_enabled() or self.transformer.training:
            return original(self, *args, **kwargs)
        return candidate(self, *args, **kwargs)
    return guarded


def attach(server):
    current = state(server)
    memoize_module(server.transformer.condition_embedder.text_embedder, current.projection)
    memoize_module(server.transformer.rope, current.rope)


def install(server_module):
    cls = server_module.VA_Server
    if getattr(cls, '_constant_patch_installed', False):
        raise RuntimeError('Constant cache patch already installed')
    reference = ast.parse(Path(__file__).with_name('server_reference.py').read_text())
    methods = next(n for n in reference.body if isinstance(n, ast.ClassDef) and n.name == 'VA_Server')
    expected = {n.name: n for n in methods.body if isinstance(n, ast.FunctionDef)}
    patched = {}
    for name, counts in [('_prepare_latent_input', [2, 0, 2, 0]),
                         ('_repeat_input_for_cfg', [0, 1, 0, 2])]:
        original = getattr(cls, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        if ast.dump(tree.body[0]) != ast.dump(expected[name]):
            raise RuntimeError('Unreviewed server method: ' + name)
        rewrite = Rewrite()
        tree = ast.fix_missing_locations(rewrite.visit(tree))
        if rewrite.counts != counts:
            raise RuntimeError(f'Unexpected replacements: {name}: {rewrite.counts}')
        namespace = dict(original.__globals__, constant_grid=constant_grid,
                         constant_text=constant_text, batch_grid=batch_grid)
        exec(compile(tree, '<constant-cache>', 'exec'), namespace)
        candidate = namespace[name]
        patched[name] = _make_guarded(original, candidate)
    original_init = cls.__init__
    original_reset = getattr(cls, '_reset')

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        attach(self)

    @functools.wraps(original_reset)
    def reset(self, *args, **kwargs):
        state(self).clear()
        return original_reset(self, *args, **kwargs)
    for name, method in patched.items():
        setattr(cls, name, method)
    cls.__init__ = init
    setattr(cls, '_reset', reset)
    setattr(cls, '_constant_original_mesh', staticmethod(server_module.get_mesh_id))
    setattr(cls, '_constant_patch_installed', True)
