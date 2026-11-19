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

"""CPU tests for the supported command, resume and result-processing contracts."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluation import evaluator as evaluation
import infer
from inference import generation as inference
import model_loader as dependencies
from inference.generation import GenerationOptions, finish_output
from inference.monitoring import TokenTiming
from evaluation.results import append_record, config_hash, prepare_results, read_records, write_json
from command_io import emit
from inference.monitoring import timing_summary
from evaluation.results import export, write_question_csv
from evaluation.protocol import COUNTS, PARSER_VERSION, load_dataset


class CommandTests(unittest.TestCase):
    def test_core_inference_imports_without_evaluation_or_model_stack(self):
        """验证核心推理包独立导入，既不依赖评测包，也不提前加载模型运行库。"""
        code = (
            'import sys; import infer; from inference import generation; '
            'assert not any(k.startswith("evaluation") for k in sys.modules); '
            'assert not any(k in sys.modules for k in ["torch", "torch_npu", "hy_embodied"])'
        )
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_and_cli_precedence(self):
        """验证配置文件可提供批量输入，显式 CLI 参数覆盖对应配置值。"""
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'request.json'
            write_json(
                config, dict(model_path='model', image=['a.jpg', 'b.jpg'], prompt='question', thinking=True, seed=42)
            )
            args = infer.parse_arguments(['--config', str(config), '--seed', '43', '--disable-thinking'])
        self.assertEqual(
            (args.image, args.prompt, args.seed, args.thinking), (['a.jpg', 'b.jpg'], ['question'], 43, False)
        )

    def test_invalid_configuration_rejected(self):
        """验证未知配置字段、输入数量不匹配及非法生成参数在加载模型前被拒绝。"""
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'request.json'
            for changes in [
                {'extra': True},
                {'image': []},
                {'prompt': ['x', 'y']},
                {'dtype': 'float32'},
                {'temperature': float('inf')},
                {'thinking': 'yes'},
            ]:
                request = dict(model_path='model', image='a.jpg', prompt='question')
                request.update(changes)
                config.write_text(json.dumps(request), encoding='utf-8')
                with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                    infer.parse_arguments(['--config', str(config)])

    def test_generation_parameter_boundaries(self):
        """验证温度零和 top-p 上界可用，并拒绝 NaN、非正 token 上限及非法超时。"""
        self.assertEqual(GenerationOptions(temperature=0, top_p=1).temperature, 0)
        for changes in [
            dict(temperature=float('nan')),
            dict(top_p=0),
            dict(max_new_tokens=0),
            dict(timeout_seconds=0),
            dict(seed=True),
            dict(max_new_tokens=True),
            dict(temperature=True),
            dict(top_p=True),
            dict(timeout_seconds=True),
            dict(thinking=1),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GenerationOptions(**changes)

    def test_serial_batch_uses_per_input_seed(self):
        """验证串行批量入口逐条调用同一推理函数，并使用基础种子加输入索引。"""
        args = infer.parse_arguments(['--model-path', 'model', '--image', 'a', 'b', '--prompt', 'x', 'y'])
        seen = []

        def generate(pipe, image, text, options):
            seen.append((image, text, options.seed))
            return dict(raw_output='answer', error=None)

        stream = io.StringIO()
        with patch.object(infer, 'generate_one', side_effect=generate), contextlib.redirect_stdout(stream):
            self.assertEqual(infer.run_inputs(object(), args), 0)
        self.assertEqual(seen, [('a', 'x', 42), ('b', 'y', 43)])
        self.assertEqual([json.loads(s)['output'] for s in stream.getvalue().splitlines()], ['answer', 'answer'])

    def test_shared_processor_contract(self):
        """验证共用输入路径调用腾讯混元消息构造接口，保留原图像素、模板参数和目标设备。"""
        from PIL import Image

        moved = object()
        batch = SimpleNamespace(to=Mock(return_value=moved))
        processor = SimpleNamespace(apply_chat_template=Mock(return_value=batch))
        pipe = SimpleNamespace(processor=processor, model=SimpleNamespace(device='npu:1'))
        with Image.new('RGB', (2, 2), (1, 2, 3)) as image:

            def messages(text, *, image):
                self.assertEqual(image.tobytes(), bytes([1, 2, 3] * 4))
                return [
                    {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}
                ]

            pipe.build_messages = Mock(side_effect=messages)
            self.assertIs(inference.prepare_inputs(pipe, image, 'original question', True), moved)
            self.assertEqual(image.tobytes(), bytes([1, 2, 3] * 4))
        self.assertEqual(pipe.build_messages.call_args.args[0], 'original question')
        self.assertEqual(
            processor.apply_chat_template.call_args.kwargs,
            dict(
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=True
            ),
        )
        batch.to.assert_called_once_with('npu:1')

    def test_generation_settings_reach_model_unchanged(self):
        """验证生成器接收到固定的采样参数、原输入及 Cache 设置，输出解码保留既定选项。"""
        import torch

        inputs = {'input_ids': torch.tensor([[3, 4]])}
        pipe = SimpleNamespace(processor=SimpleNamespace(batch_decode=Mock(return_value=['D'])))
        deadline = SimpleNamespace(expired=False)
        module = SimpleNamespace(LogitsProcessorList=list, StoppingCriteriaList=list)
        for temperature in [0, 0.05]:
            pipe.model = SimpleNamespace(
                generate=Mock(return_value=torch.tensor([[3, 4, 9, 2]])),
                generation_config=SimpleNamespace(eos_token_id=2),
            )
            record = dict(input_tokens=2, error=None)
            with patch.dict(sys.modules, {'transformers': module}), patch.object(
                torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True
            ), patch.object(inference, 'generation_hooks', return_value=(deadline, SimpleNamespace(checks=0))):
                inference.generate_completion(
                    pipe, inputs, GenerationOptions(max_new_tokens=2, temperature=temperature), record, None
                )
            options = pipe.model.generate.call_args.kwargs
            self.assertIs(options['input_ids'], inputs['input_ids'])
            self.assertEqual(
                (
                    options['max_new_tokens'],
                    options['temperature'],
                    options['top_p'],
                    options['do_sample'],
                    options['use_cache'],
                ),
                (2, temperature, 1.0, temperature > 0, True),
            )
            self.assertNotIn('top_k', options)
            self.assertNotIn('streamer', options)
            self.assertEqual(
                pipe.processor.batch_decode.call_args.kwargs,
                dict(skip_special_tokens=True, clean_up_tokenization_spaces=False),
            )
            self.assertEqual((record['raw_output'], record['output_tokens'], record['error']), ('D', 2, None))
        self.assertFalse(
            infer.generation_options(
                infer.parse_arguments(['--model-path', 'model', '--image', 'image', '--prompt', 'question'])
            ).observe
        )
        self.assertTrue(
            evaluation.generation_options(
                evaluation.parse_arguments(
                    ['--model-path', 'model', '--dataset-path', 'data', '--output-path', 'output']
                )
            ).observe
        )

    def test_help_does_not_load_model_packages(self):
        """验证查看命令帮助和读取结果工具时无需导入 PyTorch、torch_npu 或模型。"""
        code = (
            "import sys; import infer; from evaluation import evaluator, results; "
            "assert not any(k in sys.modules for k in ['torch', 'torch_npu', 'hy_embodied'])"
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_commit_not_just_version(self):
        """验证已知的不同源码提交被拒绝，缺失 Git 元数据保持未知且不会伪称提交匹配。"""
        distribution = SimpleNamespace(read_text=lambda name: '{}')
        with patch.object(dependencies.importlib.metadata, 'distribution', return_value=distribution):
            origin = dependencies.source_record('transformers', dependencies.TRANSFORMERS_COMMIT)
            self.assertIsNone(origin['detectedCommit'])
            self.assertEqual(origin['requiredCommit'], dependencies.TRANSFORMERS_COMMIT)
            distribution.read_text = Mock(return_value=json.dumps({'vcs_info': {'commit_id': 'another-source'}}))
            with self.assertRaisesRegex(RuntimeError, 'requires source commit'):
                dependencies.source_record('transformers', dependencies.TRANSFORMERS_COMMIT)
            distribution.read_text = Mock(
                return_value=json.dumps({'vcs_info': {'commit_id': dependencies.TRANSFORMERS_COMMIT}})
            )
            self.assertEqual(
                dependencies.source_record('transformers', dependencies.TRANSFORMERS_COMMIT)['detectedCommit'],
                dependencies.TRANSFORMERS_COMMIT,
            )

    def test_archive_editable_sources_and_required_interfaces(self):
        """验证归档及可编辑安装通过版本与必要接口检查，缺少模型接口或版本错误仍会拒绝。"""
        model = '''class HunYuanVLMoTForConditionalGeneration: pass
class HunYuanVLMoTDecoderLayer: pass
class _HunYuanVLMoTTextModel: pass
class _HYViT2VisionTransformer: pass
def _apply_rotary_pos_emb(q,k,cos,sin): pass
def _mask_apply(hidden_states,mask,text_funcs,vision_funcs): pass
def _flash_attention_forward_mot(module,query,key,value,attention_mask): pass
'''
        pipeline = '''class HyEmbodiedPipeline:
 def __init__(self,model,processor,device): pass
 def build_messages(prompt,*,image=None): pass
 def generate(self): pass
'''
        for origin, kind in [
            ({'archive_info': {}, 'url': 'file:///cached.tar.gz'}, 'archive'),
            ({'dir_info': {'editable': True}, 'url': 'file:///source'}, 'editable'),
        ]:
            distribution = SimpleNamespace(read_text=lambda name: json.dumps(origin))
            with self.subTest(kind=kind), patch.object(dependencies.sys, 'version_info', (3, 12)), patch.object(
                dependencies.importlib.metadata, 'version', side_effect=lambda name: dependencies.VERSIONS[name]
            ), patch.object(dependencies.importlib.metadata, 'distribution', return_value=distribution), patch.object(
                dependencies,
                'package_source',
                side_effect=lambda name, relative: model if name == 'transformers' else pipeline,
            ):
                with self.assertLogs('model_loader', level='WARNING'):
                    report = dependencies.check_dependencies()
                self.assertEqual(report['sources']['transformers']['kind'], kind)
                self.assertIsNone(report['sources']['transformers']['detectedCommit'])
                with patch.object(dependencies, 'package_source', return_value=''), self.assertRaisesRegex(
                    RuntimeError, 'Required HY interfaces missing'
                ):
                    dependencies.check_dependencies()
                with patch.object(
                    dependencies.importlib.metadata, 'version', return_value='0.0'
                ), self.assertRaisesRegex(RuntimeError, 'Dependency versions differ'):
                    dependencies.check_dependencies()

    def test_source_urls_do_not_expose_credentials(self):
        """验证来源记录移除 URL 中的认证字段、查询串和片段。"""
        direct = {'url': 'https://user:secret@example.org/source?token=private#fragment'}
        with patch.object(
            dependencies.importlib.metadata,
            'distribution',
            return_value=SimpleNamespace(read_text=lambda name: json.dumps(direct)),
        ):
            record = dependencies.source_record('transformers', dependencies.TRANSFORMERS_COMMIT)
        self.assertEqual(record['location'], 'https://example.org/source')

    def test_bad_images_are_rejected_before_model_import(self):
        """验证缺图、图片路径为目录和损坏图片在导入 NPU 运行时之前报出具体路径。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text('{}', encoding='utf-8')
            corrupt = root / 'corrupt.jpg'
            corrupt.write_bytes(b'not an image')
            for image in [root / 'missing.jpg', root, corrupt]:
                stream = io.StringIO()
                with patch.dict(sys.modules, {'model_loader': None}), contextlib.redirect_stderr(stream):
                    result = infer.main(['--model-path', str(root), '--image', str(image), '--prompt', 'question'])
                self.assertEqual(result, 2)
                self.assertIn(str(image), stream.getvalue())

    def test_batch_validation_and_one_pipeline(self):
        """验证所有批量图片先完成检查，合法请求只加载一个模型并按序输出。"""
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text('{}', encoding='utf-8')
            image = root / 'valid.png'
            Image.new('RGB', (2, 2)).save(image)
            loader, closer = Mock(return_value=object()), Mock()
            runtime = SimpleNamespace(load_pipeline=loader, close_pipeline=closer)
            command = [
                '--model-path',
                str(root),
                '--image',
                str(image),
                str(root / 'missing.png'),
                '--prompt',
                'question',
            ]
            with patch.dict(sys.modules, {'model_loader': runtime}), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(infer.main(command), 2)
            loader.assert_not_called()
            command[4] = str(image)
            stream = io.StringIO()
            with patch.dict(sys.modules, {'model_loader': runtime}), patch.object(
                infer,
                'generate_one',
                side_effect=lambda pipe, path, text, options: dict(raw_output=text, error=None, seed=options.seed),
            ), contextlib.redirect_stdout(stream):
                self.assertEqual(infer.main(command), 0)
            loader.assert_called_once_with(str(root), 'npu:0', 'graph')
            closer.assert_called_once()
            self.assertEqual([json.loads(line)['seed'] for line in stream.getvalue().splitlines()], [42, 43])

    def test_help_and_scoring_without_site_packages(self):
        """验证三个帮助入口及 CPU 复算在禁用第三方包目录后仍能执行。"""
        source = Path(infer.__file__).resolve().parents[1]
        for filename in ['src/infer.py', 'tools/evaluate_embspatial.py', 'tools/score_results.py']:
            result = subprocess.run(
                [sys.executable, '-S', str(source / filename), '--help'], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result_file = root / 'results.jsonl'
            result_file.write_text(
                json.dumps(
                    dict(
                        question_id='fixture',
                        relation='left',
                        ground_truth=0,
                        raw_output='A',
                        parsed_answer=0,
                        error=None,
                        thinking=False,
                    )
                )
                + '\n',
                encoding='utf-8',
            )
            result = subprocess.run(
                [
                    sys.executable,
                    '-S',
                    str(source / 'tools/score_results.py'),
                    '--results-path',
                    str(result_file),
                    '--output-dir',
                    str(root / 'summary'),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'summary/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['labelSource'], 'result')
            self.assertFalse(report['labelsCheckedAgainstReference'])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'results.jsonl'
        self.protocol = dict(seed=42, selectedIds=['one', 'two'], temperature=0.05)
        self.protocol['configHash'] = config_hash(self.protocol)

    def test_same_protocol_resumes_once(self):
        """验证续跑保留首次完整结果，并从下一个未完成题号继续。"""
        self.assertEqual(prepare_results(self.path, self.protocol, False), [])
        append_record(self.path, dict(question_id='one', seed=42))
        before = self.path.read_bytes()
        self.assertEqual(len(prepare_results(self.path, self.protocol, True)), 1)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(ValueError):
            prepare_results(self.path, self.protocol, False)

    def test_dataset_revision_and_file_identity(self):
        """验证评测入口拒绝其他数据 revision 及与指定 human-verified split 不一致的文件。"""
        self.path.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'revision'):
            load_dataset(self.path, 'another-revision')
        with self.assertRaisesRegex(ValueError, 'file size'):
            load_dataset(self.path)

    def test_different_configuration_is_rejected(self):
        """验证修改生成参数或损坏配置哈希后，续跑拒绝追加且原结果保持原样。"""
        prepare_results(self.path, self.protocol, False)
        append_record(self.path, dict(question_id='one', seed=42))
        before = self.path.read_bytes()
        changed = {**self.protocol, 'temperature': 0.1}
        changed['configHash'] = config_hash(changed)
        with self.assertRaises(ValueError):
            prepare_results(self.path, changed, True)
        write_json(self.path.with_suffix('.protocol.json'), {**self.protocol, 'configHash': 'invalid'})
        with self.assertRaises(ValueError):
            prepare_results(self.path, self.protocol, True)
        self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_order_and_seed_drift_rejected(self):
        """验证重复题号、非原始顺序及种子漂移均不能混入续跑。"""
        prepare_results(self.path, self.protocol, False)
        bad = [
            [dict(question_id='two', seed=43)],
            [dict(question_id='one', seed=99)],
            [dict(question_id='one', seed=42), dict(question_id='one', seed=42)],
        ]
        for records in bad:
            self.path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            with self.subTest(records=records), self.assertRaises(ValueError):
                prepare_results(self.path, self.protocol, True)

    def test_jsonl_read_is_read_only(self):
        """验证复算读取末尾缺换行的有效 JSONL 时保留原字节，追加时正确分隔新结果。"""
        self.path.write_text('{"question_id":"one","seed":42}', encoding='utf-8')
        before = self.path.read_bytes()
        self.assertEqual(read_records(self.path)[0]['question_id'], 'one')
        self.assertEqual(self.path.read_bytes(), before)
        append_record(self.path, dict(question_id='two', seed=43))
        self.assertEqual(len(read_records(self.path)), 2)
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_records(self.path)

    def test_provenance_changes_do_not_change_resume_settings(self):
        """验证 Git 与环境追溯变化不阻断相同配置续跑，生成参数和后端变化会列出字段差异。"""
        settings = dict(seed=42, selectedIds=['one'], temperature=0.05, decodeBackend='graph')
        protocol = dict(schemaVersion=3, run_config=settings, provenance={'codeCommit': 'first'})
        protocol['configHash'] = config_hash(protocol)
        prepare_results(self.path, protocol, False)
        append_record(self.path, dict(question_id='one', seed=42))
        before = self.path.read_bytes()
        changed = {**protocol, 'provenance': {'codeCommit': 'second', 'sourceModified': True}}
        self.assertEqual(len(prepare_results(self.path, changed, True)), 1)
        for key, value in [('temperature', 0.1), ('decodeBackend', 'native')]:
            invalid = {**changed, 'run_config': {**settings, key: value}}
            invalid['configHash'] = config_hash(invalid)
            with self.assertRaisesRegex(ValueError, key):
                prepare_results(self.path, invalid, True)
        self.assertEqual(self.path.read_bytes(), before)

    def test_completed_resume_does_not_load_model(self):
        """验证已完成的同一配置续跑只重建汇总，既不导入模型也不追加结果。"""
        args = evaluation.parse_arguments(
            ['--model-path', 'model', '--dataset-path', 'data', '--output-path', str(self.path), '--resume']
        )
        row = dict(question_id='one', relation='left', answer=0)
        with patch.object(
            evaluation, 'load_dataset', return_value=([row], {'datasetRevision': 'revision'})
        ), patch.object(evaluation, 'dependency_records', return_value={}), patch.object(
            evaluation, 'code_state', return_value={}
        ):
            protocol = evaluation.build_protocol(args, [row], True, {'datasetRevision': 'revision'})
            prepare_results(self.path, protocol, False)
            append_record(
                self.path, dict(question_id='one', seed=42, raw_output='A', parsed_answer=0, ground_truth=0, error=None)
            )
            before = self.path.read_bytes()
            with patch.dict(sys.modules, {'model_loader': None}), contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(
                    ['--model-path', 'model', '--dataset-path', 'data', '--output-path', str(self.path), '--resume']
                )
            self.assertEqual(self.path.read_bytes(), before)


class ResultTests(unittest.TestCase):
    def test_nested_protocol_and_independent_labels(self):
        """验证修改协议中的 Git 提交记录后计分结果保持一致，独立标签核对标记正确，标签不一致时报错。"""
        questions = []
        for relation, count in COUNTS.items():
            for index in range(count):
                questions.append(
                    dict(
                        question_id=f'{relation}-{index}',
                        relation=relation,
                        ground_truth=0,
                        originalIndex=len(questions),
                    )
                )
        first = questions[0]
        record = dict(
            question_id=first['question_id'],
            relation=first['relation'],
            ground_truth=0,
            parsed_answer=0,
            raw_output='A',
            error=None,
            seed=42,
            thinking=False,
        )
        protocol = dict(
            schemaVersion=3,
            run_config=dict(
                mode='generation',
                seed=42,
                thinking=False,
                answerParser=PARSER_VERSION,
                selectedIds=[first['question_id']],
                assets={'datasetRevision': 'revision'},
            ),
            provenance={'codeCommit': 'first'},
        )
        protocol['configHash'] = config_hash(protocol)
        protocol['provenance']['codeCommit'] = 'second'
        reference = dict(datasetRevision='revision', questions=questions)
        report, _ = export([record], protocol, reference)
        self.assertEqual(report['correct'], 1)
        self.assertEqual(report['labelSource'], 'reference')
        self.assertTrue(report['labelsCheckedAgainstReference'])
        self.assertFalse(report['complete'])
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            export([{**record, 'ground_truth': 1}], protocol, reference)

    def test_unknown_memory_is_not_reported_as_zero(self):
        """验证缺少内存观测时报告未知值，保留真实记录的峰值。"""
        self.assertIsNone(timing_summary([{}])['peakNpuAllocatedBytes'])
        self.assertEqual(timing_summary([{'peakHostRssBytes': 42}, {}])['peakHostRssBytes'], 42)

    def test_direct_mode_and_mixed_results(self):
        """验证独立复算识别记录中的 direct 模式，并拒绝混合不同模式的结果。"""
        row = dict(
            question_id='one',
            relation='left',
            ground_truth=3,
            parsed_answer=3,
            raw_output='D',
            thinking=False,
            error=None,
        )
        summary, _ = export([row])
        self.assertEqual(summary['correct'], 1)
        self.assertFalse(summary['thinking'])
        with self.assertRaises(ValueError):
            export([row, {**row, 'question_id': 'two', 'thinking': True}])

    def test_profile_window_stops_after_three_output_tokens(self):
        """验证 profiling 包含输入预填充及前三输出步，结束和异常清理重复调用时只停止一次。"""
        profiler = SimpleNamespace(stop=Mock())
        timer = TokenTiming(0, profiler)
        token = SimpleNamespace(numel=lambda: 1)
        timer.put(token)
        timer.put(token)
        timer.put(token)
        profiler.stop.assert_not_called()
        timer.put(token)
        timer.close()
        timer.end()
        profiler.stop.assert_called_once()

    def test_timeout_and_non_eos_cap_are_errors(self):
        """验证超时和未到 EOS 的长度上限计为异常，正常到 EOS 的结果保留。"""
        for expired, token, expected in [
            (True, 2, 'GenerationTimeout'),
            (False, 1, 'GenerationTokenLimit'),
            (False, 2, None),
        ]:
            record = dict(output_tokens=1, error=None)
            finish_output(record, token, [2], expired, 1)
            self.assertEqual(record['error'], expected)

    def test_oom_record_is_not_scored_correct(self):
        """验证生成器返回 OOM 时保留原始错误记录，答案与标签相同也计错。"""
        row = dict(
            question_id='fixture',
            relation='left',
            answer=0,
            question='question',
            answer_options=['a', 'b', 'c', 'd'],
            image='invalid',
        )
        generated = dict(raw_output='A', input_tokens=1, output_tokens=1, error='OutOfMemoryError: fixture')
        image = SimpleNamespace(__enter__=lambda self: self, __exit__=lambda *args: None)
        with patch('PIL.Image.open') as opened, patch.object(evaluation, 'generate_one', return_value=generated):
            opened.return_value.__enter__.return_value = image
            row['image'] = 'YQ=='
            record = evaluation.evaluate_question(object(), row, GenerationOptions())
        self.assertFalse(record['correct'])
        self.assertIsNone(record['parsed_answer'])
        self.assertEqual(record['error'], 'OutOfMemoryError: fixture')

    def test_independent_export_uses_raw_answer_and_full_denominator(self):
        """验证独立导出复算原始输出、保留异常分母，并拒绝伪造的解析答案。"""
        row = dict(
            question_id='fixture',
            relation='left',
            ground_truth=0,
            parsed_answer=0,
            raw_output='reason</think><answer>A</answer>',
            seed=42,
            error=None,
        )
        summary, table = export([row])
        self.assertEqual((summary['correct'], summary['total'], summary['complete']), (1, 3640, False))
        failed, _ = export([{**row, 'error': 'GenerationTimeout'}])
        self.assertEqual(failed['correct'], 0)
        with self.assertRaises(ValueError):
            export([{**row, 'parsed_answer': 1}])
        with tempfile.TemporaryDirectory() as folder:
            write_question_csv(Path(folder), table)
            self.assertIn('question_id', (Path(folder) / 'per-question.csv').read_text(encoding='utf-8-sig'))

    def test_json_logging_preserves_stdout(self):
        """验证输出仍为一行合法 JSON，且日志函数不会关闭调用方的输出流。"""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            emit(json.dumps({'output': '测试'}))
            stream.write('usable\n')
        self.assertEqual(json.loads(stream.getvalue().splitlines()[0]), {'output': '测试'})
        self.assertFalse(stream.closed)


if __name__ == '__main__':
    unittest.main()
