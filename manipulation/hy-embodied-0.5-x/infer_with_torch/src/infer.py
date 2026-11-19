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

"""Run one image or a serial batch with the Tencent Hunyuan NPU model."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

from inference.generation import GenerationOptions, generate_one
from command_io import emit, parse_config


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON defaults; explicit CLI options take precedence')
    parser.add_argument('--model-path')
    parser.add_argument('--image', nargs='+')
    parser.add_argument('--prompt', nargs='+')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--dtype', choices=['bfloat16', 'BF16'], default='bfloat16')
    parser.add_argument('--decode-backend', choices=['native', 'graph'], default='graph')
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument('--enable-thinking', dest='thinking', action='store_true')
    thinking.add_argument('--disable-thinking', dest='thinking', action='store_false')
    parser.set_defaults(thinking=False)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout-seconds', type=float, default=7200)
    args = parse_config(parser, argv)
    validate_arguments(args)
    return args


def validate_arguments(args):
    generation_options(args)
    for name in ['image', 'prompt']:
        value = getattr(args, name)
        if isinstance(value, str):
            value = [value]
            setattr(args, name, value)
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
            raise ValueError(f'{name} must be nonempty text or a list of nonempty text values')
    if len(args.prompt) not in {1, len(args.image)}:
        raise ValueError('Provide one shared prompt or one prompt per image')
    if args.seed + len(args.image) > 2**32:
        raise ValueError('Per-input seeds must fit the supported 32-bit RNG range')
    if not isinstance(args.model_path, str) or not args.model_path:
        raise ValueError('Provide --model-path or model_path in the configuration')
    if not isinstance(args.device, str) or re.fullmatch(r'npu(?::\d+)?', args.device) is None:
        raise ValueError('device must select an Ascend NPU')
    if args.dtype not in ['bfloat16', 'BF16']:
        raise ValueError('This sample uses BF16 model inference')
    if args.decode_backend not in ['native', 'graph']:
        raise ValueError('decode_backend must be native or graph')


def generation_options(args):
    return GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        thinking=args.thinking,
        timeout_seconds=args.timeout_seconds,
    )


def validate_input_files(args):
    """Check all user inputs on CPU before importing model or NPU dependencies."""
    model = Path(args.model_path)
    if not model.is_dir() or not (model / 'config.json').is_file():
        raise ValueError(f'Model directory must contain config.json: {model}')
    for index, image in enumerate(args.image, 1):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f'Image {index} is not a readable file: {path}')
    from PIL import Image

    for index, image in enumerate(args.image, 1):
        path = Path(image)
        try:
            with Image.open(path) as opened:
                opened.load()
        except (OSError, ValueError) as error:
            raise ValueError(f'Image {index} cannot be decoded: {path}: {error}') from error


def run_inputs(pipe, args):
    """Keep one pipeline and use independent per-input generation state and seeds."""
    options = generation_options(args)
    failed = False
    for index, image in enumerate(args.image):
        text = args.prompt[0] if len(args.prompt) == 1 else args.prompt[index]
        record = generate_one(pipe, image, text, replace(options, seed=args.seed + index))
        record.update(
            index=index,
            device=args.device,
            dtype='BF16',
            decode_backend=args.decode_backend,
            output=record.pop('raw_output'),
        )
        emit(json.dumps(record, ensure_ascii=False))
        failed |= record['error'] is not None
    return 1 if failed else 0


def main(argv=None):
    try:
        args = parse_arguments(argv)
        validate_input_files(args)
    except (ImportError, OSError, ValueError) as error:
        emit(f'Input error: {error}', error=True)
        return 2
    pipe = None
    try:
        from model_loader import close_pipeline, load_pipeline

        pipe = load_pipeline(args.model_path, args.device, args.decode_backend)
        return run_inputs(pipe, args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        emit(f'Inference error: {error}', error=True)
        return 1
    finally:
        if pipe is not None:
            close_pipeline(pipe)


if __name__ == '__main__':
    raise SystemExit(main())
