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

"""Evaluate EmbSpatial-Bench in dataset order with resumable per-question results."""

import argparse
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import platform
import sys
import time

from command_io import emit, parse_config
from evaluation.protocol import PARSER_VERSION, load_dataset, parse_answer, prompt, summarize
from evaluation.results import (
    append_record,
    code_state,
    config_hash,
    prepare_results,
    run_config,
    write_json,
)
from inference.generation import GenerationOptions, generate_one
from inference.monitoring import timing_summary
from model_loader import HY_COMMIT, TRANSFORMERS_COMMIT, dependency_records


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON defaults for the evaluation')
    parser.add_argument('--model-path')
    parser.add_argument('--dataset-path')
    parser.add_argument('--output-path')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--dtype', choices=['bfloat16', 'BF16'], default='bfloat16')
    parser.add_argument('--decode-backend', choices=['native', 'graph'], default='graph')
    parser.add_argument('--model-revision', default='79f8301d8a80dd03ba778c58d6429e414e9ef5d3')
    parser.add_argument('--dataset-revision', default='e2733cacf0f20371518c687176e881d79644e61e')
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument('--enable-thinking', dest='thinking', action='store_true')
    thinking.add_argument('--disable-thinking', dest='thinking', action='store_false')
    parser.set_defaults(thinking=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, help='Use only an initial subset for development')
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout-seconds', type=float, default=7200)
    args = parse_config(parser, argv)
    for name in ['model_path', 'dataset_path', 'output_path']:
        if not isinstance(getattr(args, name), str) or not getattr(args, name):
            raise ValueError(f'Provide {name} in the CLI or evaluation configuration')
    if not isinstance(args.device, str) or not args.device.startswith('npu') or args.dtype not in ['BF16', 'bfloat16']:
        raise ValueError('Evaluation requires an Ascend NPU and BF16')
    if args.decode_backend not in ['native', 'graph']:
        raise ValueError('decode_backend must be native or graph')
    if not isinstance(args.resume, bool):
        raise ValueError('resume must be a boolean')
    if args.limit is not None:
        if not isinstance(args.limit, int) or isinstance(args.limit, bool) or args.limit < 1:
            raise ValueError('limit must be a positive integer')
    generation_options(args)
    return args


def generation_options(args):
    return GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        thinking=args.thinking,
        timeout_seconds=args.timeout_seconds,
        observe=True,
    )


def build_protocol(args, rows, full, dataset_info):
    """Record the model/data sources and all behavior-affecting evaluation settings."""
    settings = {
        'mode': 'generation',
        'implementation': 'hy-embodied-npu-sample-v3',
        'modelPath': str(Path(args.model_path).resolve()),
        'datasetPath': str(Path(args.dataset_path).resolve()),
        'dtype': 'BF16',
        'device': args.device,
        'decodeBackend': args.decode_backend,
        'assets': {'model': 'tencent/HY-Embodied-0.5-X', 'modelRevision': args.model_revision, **dataset_info},
        'upstream': {
            'hy': HY_COMMIT,
            'transformers': TRANSFORMERS_COMMIT,
        },
        'thinking': args.thinking,
        'maxNewTokens': args.max_new_tokens,
        'temperature': float(args.temperature),
        'topP': float(args.top_p),
        'topK': 50,
        'seed': args.seed,
        'seedPolicy': 'base seed + original dataset index',
        'timeoutSeconds': float(args.timeout_seconds),
        'answerParser': PARSER_VERSION,
        'fullDatasetRun': full,
        'selectedIds': [row['question_id'] for row in rows],
        'prompt': 'original question + four options + fixed final-letter instruction',
        'imagePreprocessing': 'Tencent Hunyuan AutoProcessor and checkpoint chat template',
    }
    protocol = {
        'schemaVersion': 3,
        'run_config': settings,
        'provenance': {
            **code_state(Path(__file__).resolve().parents[2]),
            'python': sys.version,
            'platform': platform.platform(),
            'dependencies': dependency_records(),
        },
    }
    protocol['configHash'] = config_hash(protocol)
    return protocol


def evaluate_question(pipe, row, options):
    """Send only the image and prompt to inference; labels remain in the evaluator."""
    from PIL import Image

    started = time.monotonic()
    try:
        with Image.open(io.BytesIO(base64.b64decode(row['image'], validate=True))) as image:
            record = generate_one(pipe, image, prompt(row), options)
    except Exception as error:
        record = dict(
            seed=options.seed,
            raw_output='',
            input_tokens=0,
            output_tokens=0,
            thinking=options.thinking,
            ttftSeconds=None,
            generationSeconds=0.0,
            endToEndSeconds=time.monotonic() - started,
            peakHostRssBytes=None,
            peakNpuAllocatedBytes=None,
            error=f'{type(error).__name__}: {error}',
        )
    record.update(question_id=row['question_id'], relation=row['relation'], ground_truth=row['answer'])
    record['parsed_answer'] = parse_answer(record['raw_output'], options.thinking) if record['error'] is None else None
    record['correct'] = record['error'] is None and record['parsed_answer'] == row['answer']
    return record


def evaluate_rows(pipe, args, rows, previous):
    options = generation_options(args)
    records = list(previous)
    for index in range(len(previous), len(rows)):
        row = rows[index]
        record = evaluate_question(pipe, row, replace(options, seed=args.seed + index))
        append_record(args.output_path, record)
        records.append(record)
        emit(
            json.dumps(
                {
                    'completed': len(records),
                    'total': len(rows),
                    'question_id': row['question_id'],
                    'parsed_answer': record['parsed_answer'],
                    'error': record['error'],
                },
                ensure_ascii=False,
            )
        )
    return records


def run_evaluation(args):
    source, dataset_info = load_dataset(args.dataset_path, args.dataset_revision)
    rows = source[: args.limit] if args.limit is not None else source
    if args.seed + len(rows) > 2**32:
        raise ValueError('Per-question seeds must fit the supported 32-bit RNG range')
    protocol = build_protocol(args, rows, len(rows) == len(source), dataset_info)
    previous = prepare_results(args.output_path, protocol, args.resume)
    summarize(previous, rows, args.thinking)
    if len(previous) == len(rows):
        records = previous
    else:
        from model_loader import close_pipeline, load_pipeline

        pipe = load_pipeline(args.model_path, args.device, args.decode_backend)
        try:
            records = evaluate_rows(pipe, args, rows, previous)
        finally:
            close_pipeline(pipe)
    result = summarize(records, rows, args.thinking)
    result.update(
        complete=len(records) == len(rows),
        fullDatasetRun=run_config(protocol)['fullDatasetRun'],
        performance=timing_summary(records),
    )
    write_json(Path(args.output_path).with_suffix('.summary.json'), result)
    emit(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv=None):
    try:
        return run_evaluation(parse_arguments(argv))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        emit(f'Evaluation error: {error}', error=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
