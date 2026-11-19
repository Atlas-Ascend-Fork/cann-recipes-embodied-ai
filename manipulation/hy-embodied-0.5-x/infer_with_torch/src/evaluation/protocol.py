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

"""Dataset validation, answer-only prompts and independent sample-level scoring."""

import collections
import hashlib
import json
from pathlib import Path
import re

COUNTS = {'close': 612, 'far': 594, 'above': 596, 'under': 602, 'left': 616, 'right': 620}
PARSER_VERSION = 'closed-thinking-final-answer-v2'


def load_dataset(path, revision=None):
    """Read the recorded human-verified split and check its revision and file identity."""
    lock = Path(__file__).resolve().parents[2] / 'configs/asset-lock.json'
    expected = json.loads(lock.read_text(encoding='utf-8'))['dataset']
    if revision is not None and revision != expected['revision']:
        raise ValueError('Dataset revision differs from configs/asset-lock.json')
    raw = Path(path).read_bytes()
    if len(raw) != expected['bytes']:
        raise ValueError('Dataset file size differs from the recorded human-verified split')
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected['sha256']:
        raise ValueError('Dataset content differs from the recorded human-verified split')
    rows = json.loads(raw)
    validate_dataset(rows, full=True)
    return rows, {'dataset': expected['repository'], 'datasetRevision': expected['revision'], 'datasetSha256': digest}


def validate_dataset(rows, full=True):
    ids = set()
    counts = collections.Counter()
    for row in rows:
        if not isinstance(row.get('question_id'), str) or row['question_id'] in ids:
            raise ValueError('Missing or duplicate question ID')
        ids.add(row['question_id'])
        counts[row['relation']] += 1
        if row['relation'] not in COUNTS or not isinstance(row.get('question'), str):
            raise ValueError('Not the benchmark schema')
        options = row.get('answer_options')
        if not isinstance(options, list) or len(options) != 4 or any(not isinstance(s, str) for s in options):
            raise ValueError('Expected four text choices')
        if type(row.get('answer')) is not int or not 0 <= row['answer'] < 4:
            raise ValueError('Invalid ground truth')
        if not isinstance(row.get('image'), str) or not row['image']:
            raise ValueError('Missing image')
    if full and dict(counts) != COUNTS:
        raise ValueError(f'Benchmark relation counts mismatch: {dict(counts)}')
    return dict(counts)


def prompt(row):
    choices = '\n'.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(row['answer_options']))
    return (
        row['question']
        + '\n'
        + choices
        + '\nSelect the best answer. Put only one letter (A, B, C or D) in the final answer.'
    )


def parse_answer(text, thinking):
    if not isinstance(text, str):
        return None
    if thinking:
        if '</think>' not in text:
            return None
        text = text.rsplit('</think>', 1)[1].strip()
    elif '<think>' in text or '</think>' in text:
        return None
    fields = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if fields:
        if len(fields) != 1:
            return None
        final = fields[0]
    elif '<answer>' in text or '</answer>' in text:
        return None
    else:
        lines = text.strip().splitlines()
        if not lines:
            return None
        final = lines[-1] if thinking else text.strip()
    final = re.sub(r'^\s*(?:Answer|Final answer)\s*:\s*', '', final, flags=re.IGNORECASE)
    match = re.fullmatch(r'\s*([A-D])\s*[.)]?\s*', final, flags=re.IGNORECASE)
    return ord(match[1].upper()) - 65 if match else None


def summarize(records, expected_rows, thinking=None):
    expected = {row['question_id']: row for row in expected_rows}
    if len(expected) != len(expected_rows) or not expected:
        raise ValueError('Expected question set must be nonempty and unique')
    seen = set()
    correct = 0
    by = collections.defaultdict(lambda: {'correct': 0, 'total': 0})
    errors = []
    for record in records:
        qid = record['question_id']
        if qid in seen or qid not in expected:
            raise ValueError('Duplicate or foreign result ID')
        seen.add(qid)
        source = expected[qid]
        parsed = record.get('parsed_answer')
        if thinking is not None and record.get('error') is None:
            recomputed = parse_answer(record.get('raw_output'), thinking)
            if parsed != recomputed:
                raise ValueError('Stored answer differs from independently parsed output: ' + qid)
            parsed = recomputed
        if record.get('ground_truth', source['answer']) != source['answer']:
            raise ValueError('Stored ground truth differs from dataset: ' + qid)
        ok = record.get('error') is None and type(parsed) is int and parsed == source['answer']
        correct += int(ok)
        by[source['relation']]['correct'] += int(ok)
        if record.get('error') is not None or parsed is None:
            errors.append({'question_id': qid, 'error': record.get('error') or 'UnparseableAnswer'})
    for source in expected_rows:
        by[source['relation']]['total'] += 1
    for value in by.values():
        value['accuracy'] = 100 * value['correct'] / value['total']
    return {
        'correct': correct,
        'total': len(expected_rows),
        'completed': len(seen),
        'accuracy': 100 * correct / len(expected_rows),
        'relations': dict(by),
        'errors': errors,
        'missingIds': [qid for qid in expected if qid not in seen],
    }
