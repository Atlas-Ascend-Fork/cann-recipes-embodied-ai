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

import unittest
from evaluation.protocol import prompt, parse_answer, summarize, validate_dataset


class ProtocolTests(unittest.TestCase):
    def row(self, question_id='a', answer=2):
        return {
            'question_id': question_id,
            'question': 'Where is the box?',
            'answer_options': ['left', 'right', 'above', 'under'],
            'answer': answer,
            'relation': 'above',
            'image': 'encoded-image',
            'objects': [{'bbox': [111, 222, 333, 444]}],
        }

    def test_prompt_has_no_label_or_bbox(self):
        """验证模型上下文只包含原问题和选项，隔离标签与对象坐标。"""
        text = prompt(self.row())
        self.assertNotIn('111', text)
        self.assertNotIn('answer": 2', text)
        self.assertIn('C. above', text)

    def test_thinking_does_not_scan_reasoning(self):
        """验证 thinking 解析忽略推理中的选项字母，只读取最终答案。"""
        self.assertEqual(parse_answer('<think>A might be wrong; compare B.</think><answer>C</answer>', True), 2)
        self.assertIsNone(parse_answer('<think>C</think>', True))
        self.assertIsNone(parse_answer('<answer>A</answer><answer>C</answer>', True))

    def test_ambiguous_answer_is_failure(self):
        """验证包含多个候选答案的输出作为解析失败处理。"""
        self.assertIsNone(parse_answer('A or C', False))
        self.assertIsNone(parse_answer('The answer might be C.', False))
        self.assertEqual(parse_answer('C.', False), 2)

    def test_direct_answer_wrapper(self):
        """验证 direct 模式支持规定的答案包装格式。"""
        self.assertEqual(parse_answer('<answer>C</answer>', False), 2)

    def test_final_line_after_closed_thinking(self):
        """验证 thinking 结束后的最终答案行可以按协议解析。"""
        self.assertEqual(parse_answer('A? No. B? No.\n</think>\nFinal answer: C', True), 2)
        self.assertIsNone(parse_answer('<think>The answer is C', True))

    def test_failed_missing_count_against_total(self):
        """验证异常及缺失结果保留在完整题目分母中。"""
        rows = [self.row('a'), self.row('b'), self.row('c')]
        result = summarize(
            [{'question_id': 'a', 'parsed_answer': 2}, {'question_id': 'b', 'parsed_answer': None, 'error': 'timeout'}],
            rows,
        )
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['total'], 3)

    def test_repeated_ids_rejected(self):
        """验证重复题号在计分前被拒绝。"""
        with self.assertRaises(ValueError):
            validate_dataset([self.row(), self.row()], False)

    def test_sft_rejected(self):
        """验证不符合 human-verified evaluation 格式的数据被拒绝。"""
        row = self.row()
        row.pop('answer_options')
        with self.assertRaises(ValueError):
            validate_dataset([row], False)

    def test_all_options(self):
        """验证 A/B/C/D 四种选项都映射到正确的候选索引。"""
        for index, letter in enumerate('ABCD'):
            self.assertEqual(parse_answer(letter, False), index)
            self.assertEqual(parse_answer('</think><answer>' + letter + '</answer>', True), index)

    def test_reasoning_answer_is_not_scored(self):
        """验证仅在推理中出现答案的输出不会得到正确分。"""
        self.assertIsNone(parse_answer('<think><answer>C</answer></think>', True))
        self.assertIsNone(parse_answer('<think><answer>C</answer>', True))
        self.assertEqual(parse_answer('<think><answer>A</answer></think><answer>C</answer>', True), 2)

    def test_truncated_wrapper(self):
        """验证未闭合的 thinking 或 answer 标签被判为解析失败。"""
        self.assertIsNone(parse_answer('</think><answer>C', True))

    def test_error_never_counts_correct(self):
        """验证异常结果即使包含正确字母仍然计错。"""
        result = summarize([{'question_id': 'a', 'parsed_answer': 2, 'error': 'OOM'}], [self.row()])
        self.assertEqual(result['correct'], 0)

    def test_independent_parser_does_not_trust_record(self):
        """验证复算从原始文本重新解析，并拒绝与保存答案不一致的结果。"""
        with self.assertRaises(ValueError):
            summarize([{'question_id': 'a', 'raw_output': 'A', 'parsed_answer': 2}], [self.row()], False)

    def test_changed_ground_truth_rejected(self):
        """验证结果中的标签与数据集标签不一致时拒绝计分。"""
        with self.assertRaises(ValueError):
            summarize([{'question_id': 'a', 'parsed_answer': 2, 'ground_truth': 0}], [self.row()])

    def test_duplicate_results_rejected(self):
        """验证独立复算也拒绝重复的结果题号。"""
        record = {'question_id': 'a', 'parsed_answer': 2}
        with self.assertRaises(ValueError):
            summarize([record, record], [self.row()])


if __name__ == '__main__':
    unittest.main()
