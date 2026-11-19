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

"""Install instance-scoped erf GELU on the loaded NPU model."""

import functools

import torch


class NpuErfGELU(torch.nn.GELU):
    """Instance-scoped replacement for upstream GELU(approximate='none')."""

    def __init__(self):
        super().__init__(approximate='none')

    @functools.wraps(torch.nn.GELU.forward)
    def forward(self, *args, **kwargs):
        tensor = _input_tensor(args, kwargs)
        if tensor.device.type != 'npu':
            raise ValueError('NpuErfGELU requires an NPU tensor')
        import torch_npu

        return torch_npu.npu_gelu(tensor, approximate='none')


def install_exact_gelu(model):
    """Replace model instances, without changing global APIs or packages."""
    replaced = []

    def visit(parent, prefix=''):
        for name, child in list(parent.named_children()):
            path = prefix + name
            if isinstance(child, torch.nn.GELU) and child.__class__ is torch.nn.GELU and child.approximate == 'none':
                replacement = NpuErfGELU()
                replacement.train(child.training)
                setattr(parent, name, replacement)
                replaced.append(path)
            else:
                visit(child, path + '.')

    visit(model)
    if not replaced and not any(isinstance(m, NpuErfGELU) for m in model.modules()):
        raise RuntimeError('Pinned HY model has no erf GELU modules to adapt')
    setattr(
        model,
        '_hy_erf_gelu_modules',
        tuple(name for name, module in model.named_modules() if isinstance(module, NpuErfGELU)),
    )
    return replaced


def _input_tensor(positional, keywords):
    """Accept the upstream single tensor argument by position or input= keyword."""
    if len(positional) == 1 and not keywords:
        return positional[0]
    if not positional and set(keywords) == {'input'}:
        return keywords['input']
    raise TypeError('GELU.forward requires exactly one input tensor')
