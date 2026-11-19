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

"""Apply and back up the required Transformers source adaptation."""

import pathlib
import sys
import tempfile

from model_loader import check_runtime

_INSTALLED_BACKEND = None


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Pinned HY source changed; refusing to apply an unverified patch')
    return source.replace(old, new, 1)


def patched_source(source, fast_decode):
    source = _replace_once(
        source,
        'from flash_attn import flash_attn_func, flash_attn_varlen_func',
        'from adapters.attention import flash_attn_func, flash_attn_varlen_func',
    )
    if not fast_decode:
        return source
    # All-text one-token inference avoids the unused visual projection branch.
    # Padding, visual tokens and training retain the original implementation.
    source = _replace_once(
        source,
        '    B, S, D = hidden_states.size()\n',
        '    B, S, D = hidden_states.size()\n'
        '    if (not torch.is_grad_enabled() and S == 1 and padding_mask is None\n'
        '            and mask is not None and not bool(mask.any())):\n'
        '        return [fn(hidden_states) for fn in text_funcs]\n',
    )
    # For B=1, packed attention and fixed attention have identical inputs.
    source = _replace_once(
        source,
        '    else:\n        # B==1, no padding, or decode step (S_q != S_k) -- simple reshape path\n',
        '    elif B == 1:\n'
        '        attn_output = flash_attn_func(query, key, value, causal=(S_q == S_k))\n'
        '    else:\n        # Multi-batch unpadded attention.\n',
    )
    # Native cached decode marks no visual segment with [[0,0]]. An empty
    # segment list expresses the same condition without a device reduction.
    source = _replace_once(
        source,
        '        single = torch.tensor([[0, 0]], device=mask.device)\n',
        '        single = torch.empty((0, 2), dtype=torch.long, device=mask.device)\n',
    )
    return source


def install(fast_decode=True):
    """Install a reversible minimal patch in the isolated Transformers environment."""
    global _INSTALLED_BACKEND
    backend = 'graph' if fast_decode else 'native'
    module_name = 'transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot'
    if _INSTALLED_BACKEND is not None and _INSTALLED_BACKEND != backend:
        raise RuntimeError(f'Decode backend is already {_INSTALLED_BACKEND}; select {backend} in a separate process')
    if _INSTALLED_BACKEND is None and module_name in sys.modules:
        raise RuntimeError('Select the decode backend before importing the HY model; start a fresh process')
    runtime = check_runtime()
    import transformers

    target = pathlib.Path(transformers.__file__).parent / 'models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    backup = target.with_suffix('.py.hy-original')
    if not backup.exists():
        original = target.read_text(encoding='utf-8')
        if 'from flash_attn import flash_attn_func, flash_attn_varlen_func' not in original:
            raise RuntimeError('Original pinned HY model source is unavailable')
        try:
            with backup.open('x', encoding='utf-8') as handle:
                handle.write(original)
        except OSError as error:
            raise RuntimeError(
                f'Cannot save the original HY source to {backup}; use a writable dedicated virtual environment'
            ) from error
    body = patched_source(backup.read_text(encoding='utf-8'), fast_decode)
    if target.read_text(encoding='utf-8') != body:
        if module_name in sys.modules:
            raise RuntimeError('Installed HY source changed after import; use a separate writable virtual environment')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=target.parent,
                prefix=target.name + '.hy-',
                suffix='.pending',
                delete=False,
            ) as handle:
                temporary = pathlib.Path(handle.name)
                handle.write(body)
            temporary.replace(target)
        except OSError as error:
            raise RuntimeError(
                f'Cannot apply the HY NPU patch to {target}; use a writable dedicated virtual environment'
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _INSTALLED_BACKEND = backend
    return target, runtime
