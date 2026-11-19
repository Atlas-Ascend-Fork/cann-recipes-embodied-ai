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

"""Check dependencies and load the local processor and BF16 NPU model."""

import ast
import importlib.metadata
import importlib.util
import json
import pathlib
import logging
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

HY_COMMIT = 'cf23678a76dc9c445ab7f7aea7a9a8f60e910a63'
TRANSFORMERS_COMMIT = '9293856c419762ebf98fbe2bd9440f9ce7069f1a'
VERSIONS = {'torch': '2.10.0', 'torch_npu': '2.10.0.post6', 'transformers': '4.57.0', 'hy-embodied': '0.5.0'}


def source_record(name, expected_commit):
    """Record available installer metadata; missing VCS data is not a commit match."""
    distribution = importlib.metadata.distribution(name)
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    vcs = direct.get('vcs_info') or {}
    commit = vcs.get('commit_id')
    if commit is not None and commit != expected_commit:
        raise RuntimeError(f'{name} requires source commit {expected_commit}; installed commit is {commit}')
    location = urlsplit(direct.get('url', ''))
    safe_url = urlunsplit((location.scheme, location.hostname or '', location.path, '', '')) or None
    kind = (
        'vcs'
        if vcs
        else (
            'archive'
            if 'archive_info' in direct
            else 'editable' if direct.get('dir_info', {}).get('editable') else 'local' if direct else 'distribution'
        )
    )
    return {'requiredCommit': expected_commit, 'detectedCommit': commit, 'kind': kind, 'location': safe_url}


def dependency_records():
    """Collect small package metadata for run provenance, without loading libraries."""
    versions = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    for name, commit in [('transformers', TRANSFORMERS_COMMIT), ('hy-embodied', HY_COMMIT)]:
        if versions[name] is not None:
            sources[name] = source_record(name, commit)
    return {'versions': versions, 'sources': sources}


def require_model_interfaces(model_source, pipeline_source):
    """Check the HY entry points required by the adapter, not the whole dependency tree."""
    model = ast.parse(model_source)
    functions = {node.name: node for node in model.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in model.body if isinstance(node, ast.ClassDef)}
    required_classes = {
        'HunYuanVLMoTForConditionalGeneration',
        'HunYuanVLMoTDecoderLayer',
        '_HunYuanVLMoTTextModel',
        '_HYViT2VisionTransformer',
    }
    missing = sorted(required_classes - classes)
    for name, arguments in {
        '_apply_rotary_pos_emb': {'q', 'k', 'cos', 'sin'},
        '_mask_apply': {'hidden_states', 'mask', 'text_funcs', 'vision_funcs'},
        '_flash_attention_forward_mot': {'module', 'query', 'key', 'value', 'attention_mask'},
    }.items():
        node = functions.get(name)
        if node is None or not arguments.issubset({arg.arg for arg in node.args.args + node.args.kwonlyargs}):
            missing.append(name)
    pipeline = next(
        (
            node
            for node in ast.parse(pipeline_source).body
            if isinstance(node, ast.ClassDef) and node.name == 'HyEmbodiedPipeline'
        ),
        None,
    )
    methods = {node.name: node for node in pipeline.body if isinstance(node, ast.FunctionDef)} if pipeline else {}
    missing.extend(
        'HyEmbodiedPipeline.' + name for name in ['__init__', 'build_messages', 'generate'] if name not in methods
    )
    for name, arguments in {
        '__init__': {'model', 'processor', 'device'},
        'build_messages': {'prompt', 'image'},
    }.items():
        node = methods.get(name)
        if node is not None and not arguments.issubset({arg.arg for arg in node.args.args + node.args.kwonlyargs}):
            missing.append('HyEmbodiedPipeline.' + name + ' arguments')
    if missing:
        raise RuntimeError(
            'Required HY interfaces missing: '
            + ', '.join(missing)
            + '; install requirements.txt with the pinned sources'
        )


def package_source(package, relative):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f'Missing {package}; install requirements.txt')
    path = Path(spec.origin).parent / relative
    if not path.is_file():
        raise RuntimeError(f'Required HY source file is missing: {path}; install requirements.txt')
    return path.read_text(encoding='utf-8')


def check_dependencies():
    """Reject incompatible versions/interfaces and known source mismatches before weights load."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('This sample requires Python 3.12')
    records = dependency_records()
    mismatches = [
        f'{name}: expected {version}, installed {records["versions"][name]}'
        for name, version in VERSIONS.items()
        if (records['versions'][name] or '').split('+')[0] != version
    ]
    if mismatches:
        raise RuntimeError('Dependency versions differ: ' + '; '.join(mismatches) + '; install requirements.txt')
    require_model_interfaces(
        package_source('transformers', 'models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'),
        package_source('hy_embodied', 'inference/pipeline.py'),
    )
    for name, record in records['sources'].items():
        if record['detectedCommit'] is None:
            logging.getLogger(__name__).warning(
                '%s: %s installation has no VCS commit metadata; '
                'versions and HY interfaces checked. Required source: %s',
                name,
                record['kind'],
                record['requiredCommit'],
            )
    return records


def check_runtime():
    """Fail before loading weights when the supported NPU dependencies differ."""
    import torch
    import torch_npu

    records = check_dependencies()
    actual = {'torch': torch.__version__.split('+')[0], 'torch_npu': torch_npu.__version__}
    expected = {'torch': '2.10.0', 'torch_npu': '2.10.0.post6'}
    if actual != expected:
        raise RuntimeError(f'Required NPU dependencies: {expected}; installed: {actual}')
    if not torch.npu.is_available():
        raise RuntimeError('An available Ascend NPU is required')
    return records


def load_pipeline(model_path, device='npu:0', decode_backend='graph'):
    import torch

    from adapters.gelu import install_exact_gelu
    from adapters.transformers_patch import install as install_transformers_patch

    if decode_backend not in ['native', 'graph']:
        raise ValueError('decode_backend must be native or graph')
    _, dependencies = install_transformers_patch(decode_backend == 'graph')
    from hy_embodied.inference import HyEmbodiedPipeline

    if not device.startswith('npu'):
        raise ValueError('This sample requires an NPU device')
    torch.npu.set_device(device)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = pathlib.Path(model_path).resolve()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    template = model_path / 'chat_template.jinja'
    if template.exists():
        processor.chat_template = template.read_text(encoding='utf-8')
    model, loading = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation='eager',
        output_loading_info=True,
    )
    if any(loading.get(name) for name in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError('Checkpoint loading failed: ' + json.dumps(loading, default=str))
    if model.generation_config.top_k != 50:
        raise RuntimeError('The model generation configuration must retain top_k=50')
    model.to(device).eval()
    if any(param.device.type != 'npu' for param in model.parameters()):
        raise RuntimeError('Model parameters outside the NPU')
    install_exact_gelu(model)
    pipe = HyEmbodiedPipeline(model, processor, device)
    pipe.hy_decode_backend = decode_backend
    pipe.hy_dependencies = dependencies
    if decode_backend == 'graph':
        from adapters.graph_decode import install as install_super_decode

        install_super_decode(pipe)
    return pipe


def close_pipeline(pipe):
    """Release pipeline-owned wrappers before creating another pipeline."""
    optimizer = getattr(pipe, '_hy_decode_optimizer', None)
    if optimizer is not None:
        optimizer.close()
        delattr(pipe, '_hy_decode_optimizer')
