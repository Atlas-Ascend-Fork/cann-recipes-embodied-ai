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

"""Model-loading, single/batch and fixed-smoke tests, enabled by explicit CLI paths."""

import argparse
import base64
import contextlib
from dataclasses import replace
import gc
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import unittest

from evaluation.protocol import COUNTS, parse_answer, prompt, validate_dataset
from inference.generation import GenerationOptions, generate_one
from evaluation.results import write_json

ARGS = None


def model_arguments(description, profile=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--decode-backend', choices=['native', 'graph'], default='graph')
    if profile:
        parser.add_argument('--profile', action='store_true', help='Also collect ten samples, five captures each')
        parser.add_argument(
            '--compare-backends',
            action='store_true',
            help='Compare bounded native/graph generation in separate processes',
        )
    return parser.parse_known_args()


def read_dataset(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_dataset(rows, full=True)
    return rows


def smoke_rows(rows):
    """Choose ten deterministic rows, including one from every relation."""
    first = {}
    for row in rows:
        first.setdefault(row['relation'], row)
    selected = [first.get(relation) for relation in COUNTS]
    if any(row is None for row in selected):
        raise ValueError('Smoke inputs must cover all six spatial relations')
    ids = {row['question_id'] for row in selected}
    return selected + [row for row in rows if row['question_id'] not in ids][:4]


def generate_row(pipe, row, options, profiler=None):
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(row['image'], validate=True))) as image:
        return generate_one(pipe, image, prompt(row), options, profiler)


def save_case(folder, name, value):
    target = Path(folder) / (name + '.json')
    if target.exists():
        raise FileExistsError('Choose a new test output directory: ' + str(target))
    write_json(target, value)


class ModelNpuTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ARGS is None or importlib.util.find_spec('torch_npu') is None:
            raise unittest.SkipTest('Run this file with explicit model/data/output paths on an Ascend NPU')
        import torch

        importlib.import_module('torch_npu')
        from model_loader import load_pipeline

        if not torch.npu.is_available():
            raise unittest.SkipTest('An available Ascend NPU is required')
        cls.rows = read_dataset(ARGS.dataset_path)
        cls.indices = {row['question_id']: i for i, row in enumerate(cls.rows)}
        cls.pipe = load_pipeline(ARGS.model_path, ARGS.device, ARGS.decode_backend)

    @classmethod
    def tearDownClass(cls):
        import torch
        from model_loader import close_pipeline

        close_pipeline(cls.pipe)
        del cls.pipe
        gc.collect()
        torch.npu.synchronize()


class InferenceNpuTests(ModelNpuTestCase):
    def test_model_loading(self):
        """验证完整模型参数位于 NPU、使用 BF16，且预期的 29 个 GELU 实例已安装。"""
        import torch
        from adapters.gelu import NpuErfGELU

        self.assertTrue(all(p.device.type == 'npu' and p.dtype == torch.bfloat16 for p in self.pipe.model.parameters()))
        self.assertEqual(sum(isinstance(module, NpuErfGELU) for module in self.pipe.model.modules()), 29)

    def test_smoke_thinking_and_direct(self):
        """验证十个固定样本覆盖六类关系，direct/thinking 输出合法且逐步 logits 全部有限。"""
        selected = smoke_rows(self.rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual({row['relation'] for row in selected}, set(COUNTS))
        records = []
        for thinking in [True, False]:
            for row in selected:
                options = GenerationOptions(
                    seed=42 + self.indices[row['question_id']], thinking=thinking, verify_finite=True, observe=True
                )
                record = generate_row(self.pipe, row, options)
                record.update(question_id=row['question_id'], relation=row['relation'])
                records.append(record)
        save_case(ARGS.output_dir, self._testMethodName, records)
        for record in records:
            with self.subTest(question_id=record['question_id'], thinking=record['thinking']):
                self.assertIsNone(record['error'])
                self.assertTrue(record['raw_output'].strip())
                self.assertNotIn('\ufffd', record['raw_output'])
                self.assertFalse(any(ord(c) < 32 and c not in '\n\r\t' for c in record['raw_output']))
                self.assertIn(parse_answer(record['raw_output'], record['thinking']), range(4))
                self.assertEqual(record['finiteLogitsChecks'], record['output_tokens'])

    def test_single_and_serial_batch(self):
        """验证 CLI 串行批量入口与独立单图调用使用相同输入、种子和生成结果。"""
        import infer

        folder = Path(ARGS.output_dir) / self._testMethodName
        folder.mkdir(parents=True, exist_ok=False)
        selected = self.rows[:2]
        paths = []
        for index, row in enumerate(selected):
            path = folder / f'image-{index}.jpg'
            path.write_bytes(base64.b64decode(row['image'], validate=True))
            paths.append(str(path))
        args = infer.parse_arguments(
            [
                '--model-path',
                ARGS.model_path,
                '--image',
                *paths,
                '--prompt',
                *[prompt(row) for row in selected],
                '--temperature',
                '0',
                '--max-new-tokens',
                '64',
                '--device',
                ARGS.device,
                '--decode-backend',
                ARGS.decode_backend,
            ]
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = infer.run_inputs(self.pipe, args)
        batch = [json.loads(line) for line in stream.getvalue().splitlines()]
        single = [
            generate_row(self.pipe, row, GenerationOptions(max_new_tokens=64, temperature=0, seed=42 + i))
            for i, row in enumerate(selected)
        ]
        write_json(folder / 'comparison.json', dict(batch=batch, single=single))
        self.assertEqual(status, 0)
        self.assertEqual([r['output'] for r in batch], [r['raw_output'] for r in single])
        self.assertEqual([r['seed'] for r in batch], [42, 43])

    def test_observation_preserves_generated_output(self):
        """验证同一输入在 64 token 上限内启用观测前后的文本、token 数和结束状态一致。"""
        options = GenerationOptions(max_new_tokens=64, temperature=0, thinking=True, seed=42)
        plain = generate_row(self.pipe, self.rows[0], options)
        observed = generate_row(self.pipe, self.rows[0], replace(options, observe=True))
        save_case(ARGS.output_dir, self._testMethodName, dict(plain=plain, observed=observed))
        for field in ['raw_output', 'output_tokens', 'error']:
            self.assertEqual(plain[field], observed[field])
        self.assertNotIn('ttftSeconds', plain)
        self.assertGreater(observed['generationSeconds'], 0)
        self.assertGreater(observed['peakNpuAllocatedBytes'], 0)

    def test_missing_image_and_truncation(self):
        """验证缺图形成清楚的错误记录，单 token 上限生效且截断的 thinking 不会计分。"""
        missing = generate_one(
            self.pipe, str(Path(ARGS.output_dir) / 'missing-image.png'), 'Describe this image.', GenerationOptions()
        )
        row = self.rows[0]
        capped = generate_row(self.pipe, row, GenerationOptions(max_new_tokens=1, thinking=True, seed=42))
        save_case(ARGS.output_dir, self._testMethodName, dict(missing=missing, capped=capped))
        self.assertIn('FileNotFoundError', missing['error'])
        self.assertEqual(capped['output_tokens'], 1)
        self.assertIsNone(parse_answer(capped['raw_output'], True))


class BackendNpuTests(unittest.TestCase):
    def test_decoder_backend_equivalence(self):
        """验证独立进程中的 native/graph 在同一输入和 64 token 上限下的文本、token 数及结束状态一致。"""
        if ARGS is None or not ARGS.compare_backends:
            self.skipTest('Add --compare-backends to select this two-process check')
        source = '''import json,sys
from inference.generation import GenerationOptions
from model_loader import close_pipeline,load_pipeline
from check_model import read_dataset,generate_row
pipe=load_pipeline(sys.argv[1],sys.argv[3],sys.argv[4])
try:
 record=generate_row(pipe,read_dataset(sys.argv[2])[0],GenerationOptions(max_new_tokens=64,temperature=0,thinking=True,seed=42))
 owner=getattr(pipe,'_hy_decode_optimizer',None)
 record['graphReplays']=owner.graphs.replays if owner is not None else 0
 record['backend']=sys.argv[4]
 print(json.dumps(record,ensure_ascii=False))
finally:close_pipeline(pipe)
'''
        records = []
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment['PYTHONPATH'] = (
            str(Path(__file__).resolve().parent) + os.pathsep + environment.get('PYTHONPATH', '')
        )
        for backend in ['native', 'graph']:
            result = subprocess.run(
                [sys.executable, '-c', source, ARGS.model_path, ARGS.dataset_path, ARGS.device, backend],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=600,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
            self.assertTrue(lines, result.stdout)
            records.append(json.loads(lines[-1]))
        save_case(ARGS.output_dir, self._testMethodName, records)
        for field in ['raw_output', 'output_tokens', 'error']:
            self.assertEqual(records[0][field], records[1][field], field)
        self.assertIn(records[0]['error'], [None, 'GenerationTokenLimit'])
        self.assertEqual(records[0]['graphReplays'], 0)
        self.assertGreater(records[1]['graphReplays'], 0)


class StabilityNpuTests(ModelNpuTestCase):
    def test_twenty_samples_two_rounds(self):
        """验证二十个固定输入重复两轮后输出一致，清理后的活跃 NPU 内存保持稳定。"""
        rounds = []
        for _ in range(2):
            records = []
            for row in self.rows[:20]:
                options = GenerationOptions(
                    thinking=True, seed=42 + self.indices[row['question_id']], verify_finite=True, observe=True
                )
                records.append(generate_row(self.pipe, row, options))
            rounds.append(records)
        save_case(ARGS.output_dir, self._testMethodName, rounds)
        self.assertEqual([r['raw_output'] for r in rounds[0]], [r['raw_output'] for r in rounds[1]])
        self.assertTrue(all(r['error'] is None for records in rounds for r in records))
        first, second = rounds[0][-1], rounds[1][-1]
        self.assertEqual(first['npuAllocatedAfterCleanupBytes'], second['npuAllocatedAfterCleanupBytes'])

    def test_profile_ten_samples_five_times(self):
        """按显式 --profile 选项采集十个固定样本各五次的 prefill 与前三输出步内存和算子记录。"""
        if not ARGS.profile:
            self.skipTest('Add --profile to collect the taskbook profiling records')
        import torch_npu

        records = []
        for row in smoke_rows(self.rows):
            for repeat in range(5):
                folder = Path(ARGS.output_dir) / 'profile' / f"{row['question_id']}-{repeat}"
                folder.mkdir(parents=True, exist_ok=False)
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder), async_mode=False),
                    record_shapes=True,
                    profile_memory=True,
                    with_modules=True,
                )
                options = GenerationOptions(
                    thinking=True, seed=42 + self.indices[row['question_id']], verify_finite=True, observe=True
                )
                record = generate_row(self.pipe, row, options, profiler)
                record.update(question_id=row['question_id'], repeat=repeat, profiled=True)
                records.append(record)
        save_case(ARGS.output_dir, self._testMethodName, records)
        self.assertEqual(len(records), 50)
        self.assertTrue(all(record['error'] is None for record in records))


if __name__ == '__main__':
    ARGS, remaining = model_arguments(__doc__, profile=True)
    unittest.main(argv=[sys.argv[0], *remaining], defaultTest='InferenceNpuTests')
