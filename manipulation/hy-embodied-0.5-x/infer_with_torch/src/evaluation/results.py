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

"""Store and resume evaluation records, reparse answers, and export scores."""

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from command_io import emit
from evaluation.protocol import COUNTS, PARSER_VERSION, load_dataset, parse_answer
from inference.monitoring import timing_summary


def config_hash(config):
    """Hash only the canonical run configuration, as required for safe resume."""
    body = run_config(config)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    ).hexdigest()


def run_config(protocol):
    """Read current run settings or a historical flat protocol for CPU recomputation."""
    if 'run_config' in protocol:
        if not isinstance(protocol['run_config'], dict):
            raise ValueError('run_config must be a JSON object')
        return protocol['run_config']
    return {key: value for key, value in protocol.items() if key != 'configHash'}


def config_differences(saved, current, prefix=''):
    """Describe changed fields without dumping long question-ID lists."""
    differences = []
    for key in sorted(set(saved) | set(current)):
        name = prefix + key
        if key not in saved or key not in current:
            differences.append(f'{name}: field missing from saved or current configuration')
            continue
        left, right = saved[key], current[key]
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(config_differences(left, right, name + '.'))
        elif isinstance(left, list) and isinstance(right, list):
            differences.extend(list_differences(left, right, name))
        elif type(left) is not type(right) or left != right:
            differences.append(f'{name}: saved={str(left)[:100]!r}, current={str(right)[:100]!r}')
    return differences


def list_differences(saved, current, name):
    """Report a length change and the first differing list item."""
    differences = []
    if len(saved) != len(current):
        differences.append(f'{name}: saved length={len(saved)}, current length={len(current)}')
    for index, (left, right) in enumerate(zip(saved, current)):
        if type(left) is not type(right) or left != right:
            differences.append(f'{name}[{index}]: saved={str(left)[:100]!r}, current={str(right)[:100]!r}')
            break
    return differences


def code_state(folder):
    """Record an available Git revision without making Git or a clean checkout a launch gate."""
    git = shutil.which('git')
    if git is None:
        return {'codeCommit': None, 'sourceModified': None}
    try:
        head = subprocess.check_output(
            [git, '-C', str(folder), 'rev-parse', 'HEAD'],
            text=True,
            encoding='utf-8',
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = subprocess.check_output(
            [git, '-C', str(folder), 'status', '--porcelain', '--', 'src', 'configs', 'requirements.txt'],
            text=True,
            encoding='utf-8',
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        return {'codeCommit': head, 'sourceModified': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'codeCommit': None, 'sourceModified': None}


def write_json(path, value):
    """Replace a small metadata file after its complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_records(path):
    """Read JSONL without modifying the source, including a final valid unterminated line."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid result JSON at line {index}') from error
            if not isinstance(record, dict):
                raise ValueError(f'Expected a result object at line {index}')
            records.append(record)
    return records


def append_record(path, record):
    """Persist one complete result before moving to the next sample."""
    path = Path(path)
    separator = ''
    if path.exists() and path.stat().st_size:
        with path.open('rb') as previous:
            previous.seek(-1, os.SEEK_END)
            separator = '' if previous.read(1) == b'\n' else '\n'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(separator + json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def prepare_results(path, protocol, resume):
    """Require an identical saved protocol and unique completed IDs before appending."""
    path = Path(path)
    metadata = path.with_suffix('.protocol.json')
    if protocol.get('configHash') != config_hash(protocol):
        raise ValueError('Invalid current configuration hash')
    if path.exists() or metadata.exists():
        if not resume or not metadata.exists():
            raise ValueError('Existing results require --resume and their original protocol')
        stored = json.loads(metadata.read_text(encoding='utf-8'))
        if stored.get('configHash') != config_hash(stored):
            raise ValueError(f'Saved configuration hash is invalid: {metadata}')
        if ('run_config' in stored) != ('run_config' in protocol):
            raise ValueError('Protocol format differs; resume with the original runner or choose a new output path')
        if stored['configHash'] != protocol['configHash']:
            differences = config_differences(run_config(stored), run_config(protocol))
            raise ValueError('Resume configuration differs:\n' + '\n'.join(differences[:8]))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(metadata, protocol)
    records = read_records(path)
    settings = run_config(protocol)
    expected = settings['selectedIds']
    ids = [record['question_id'] for record in records]
    if len(ids) != len(set(ids)) or ids != expected[: len(ids)]:
        raise ValueError('Resume results must be a unique prefix in the original question order')
    if any(record.get('seed') != settings['seed'] + index for index, record in enumerate(records)):
        raise ValueError('Stored seeds differ from the original question indices')
    return records


def export(records, protocol=None, reference=None):
    thinking, expected = validate_export_reference(records, protocol, reference)
    protocol = run_config(protocol) if protocol is not None else None
    expected_by_id = {r['question_id']: r for r in expected} if expected is not None else {}
    seen = set()
    by = {
        r: {'correct': 0, 'total': n, 'completed': 0, 'answerMismatch': 0, 'unparseable': 0, 'executionErrors': 0}
        for r, n in COUNTS.items()
    }
    table, errors = [], []
    collect_export_rows(records, thinking, expected, expected_by_id, protocol, seen, by, table, errors)
    if protocol is not None and [r['question_id'] for r in records] != protocol['selectedIds'][: len(records)]:
        raise ValueError('Result order or coverage differs from the saved protocol')
    complete = len(seen) == 3640 and all(v['completed'] == v['total'] for v in by.values())
    if any(v['completed'] > v['total'] for v in by.values()):
        raise ValueError('Too many records in a relation')
    correct = sum(v['correct'] for v in by.values())
    for item in by.values():
        item['accuracy'] = 100 * item['correct'] / item['total']
    summary = {
        'total': 3640,
        'completed': len(seen),
        'correct': correct,
        'accuracy': 100 * correct / 3640,
        'complete': complete,
        'relations': by,
        'errors': errors,
        'answerParser': PARSER_VERSION,
        'thinking': thinking,
        'labelVerification': 'independent compact reference' if expected is not None else 'embedded result labels only',
        'labelSource': 'reference' if expected is not None else 'result',
        'labelsCheckedAgainstReference': expected is not None,
        'protocolVerified': protocol is not None,
        'protocolValidationFields': [
            'answerParser',
            'mode',
            'thinking',
            'selectedIds',
            'datasetRevision with reference',
            'seed with reference',
        ],
        'performance': timing_summary(records),
    }
    return summary, table


def validate_export_reference(records, protocol, reference):
    if not records:
        raise ValueError('No result records')
    if protocol is not None:
        if protocol.get('configHash') != config_hash(protocol):
            raise ValueError('Invalid saved evaluation configuration hash')
        protocol = run_config(protocol)
    thinking = validate_result_mode(records, protocol)
    expected = validate_reference_questions(protocol, reference)
    return thinking, expected


def validate_result_mode(records, protocol):
    """Require one boolean answer mode matching the saved parser and evaluation mode."""
    fallback = protocol['thinking'] if protocol is not None else True
    declared = [record.get('thinking', fallback) for record in records]
    if any(not isinstance(mode, bool) for mode in declared) or len(set(declared)) != 1:
        raise ValueError('Results must use one declared thinking/direct mode')
    thinking = declared[0] if protocol is None else protocol['thinking']
    if not isinstance(thinking, bool) or any(record.get('thinking', thinking) != thinking for record in records):
        raise ValueError('Result modes differ from the saved evaluation protocol')
    if protocol is not None and protocol['answerParser'] != PARSER_VERSION:
        raise ValueError('Unsupported answer parser')
    if protocol is not None and protocol.get('mode') != 'generation':
        raise ValueError('This exporter handles generation outputs, not likelihood scores')
    return thinking


def validate_reference_questions(protocol, reference):
    """Check independent labels, original indices and the complete relation counts."""
    expected = None
    if reference is not None:
        if protocol is not None and reference['datasetRevision'] != protocol['assets']['datasetRevision']:
            raise ValueError('Reference revision differs from the saved protocol')
        expected = reference['questions']
        if len(expected) != 3640 or len({r['question_id'] for r in expected}) != 3640:
            raise ValueError('Compact reference must contain 3640 unique official question IDs')
        if dict(collections.Counter(r['relation'] for r in expected)) != COUNTS:
            raise ValueError('Reference relation counts do not match the official split')
        if [r['originalIndex'] for r in expected] != list(range(3640)) or any(
            type(r['ground_truth']) is not int or not 0 <= r['ground_truth'] < 4 for r in expected
        ):
            raise ValueError('Invalid reference labels or original indices')
    return expected


def collect_export_rows(records, thinking, expected, expected_by_id, protocol, seen, by, table, errors):
    for row in records:
        qid, relation, truth = row['question_id'], row['relation'], row['ground_truth']
        input_condition = (not isinstance(qid, str)) or (not qid) or (qid in seen)
        if input_condition or (relation not in COUNTS):
            raise ValueError('Missing, duplicate or invalid question ID/relation')
        if type(truth) is not int or not 0 <= truth < 4:
            raise ValueError('Invalid ground truth')
        if expected is not None:
            target = expected_by_id.get(qid)
            if target is None or target['relation'] != relation or target['ground_truth'] != truth:
                raise ValueError('Result labels differ from the independent reference: ' + qid)
            if protocol is not None and row.get('seed') != protocol['seed'] + target['originalIndex']:
                raise ValueError('Result seed differs from the original question index: ' + qid)
        seen.add(qid)
        parsed = parse_answer(row.get('raw_output'), thinking)
        error = row.get('error')
        cached = row.get('parsed_answer')
        invalid_cached_type = cached is not None and type(cached) is not int
        if error is None and (cached != parsed or invalid_cached_type):
            raise ValueError('Stored answer differs from raw output: ' + qid)
        ok = error is None and type(parsed) is int and parsed == truth
        item = by[relation]
        item['completed'] += 1
        item['correct'] += int(ok)
        if error is not None:
            item['executionErrors'] += 1
        elif parsed is None:
            item['unparseable'] += 1
        elif not ok:
            item['answerMismatch'] += 1
        category = error or ('UnparseableAnswer' if parsed is None else None)
        if category:
            errors.append({'question_id': qid, 'relation': relation, 'error': category})
        table.append({**{k: v for k, v in row.items() if k != 'raw_output'}, 'parsed_answer': parsed, 'correct': ok})


def write_question_csv(target, table):
    fields = [
        'question_id',
        'relation',
        'ground_truth',
        'parsed_answer',
        'correct',
        'error',
        'seed',
        'input_tokens',
        'output_tokens',
        'ttftSeconds',
        'generationSeconds',
        'endToEndSeconds',
        'peakHostRssBytes',
        'peakNpuAllocatedBytes',
    ]
    with (target / 'per-question.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--protocol-path')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset-path', help='Full human-verified evaluation JSON, used only for its labels')
    source.add_argument('--reference-path', help='Compact dataset revision / ID / relation / label index')
    args = parser.parse_args(argv)
    try:
        return score_files(args)
    except (OSError, ValueError, KeyError) as error:
        emit(f'Result error: {error}', error=True)
        return 2


def score_files(args):
    records = read_records(args.results_path)
    protocol = json.loads(Path(args.protocol_path).read_text(encoding='utf-8')) if args.protocol_path else None
    reference = json.loads(Path(args.reference_path).read_text(encoding='utf-8')) if args.reference_path else None
    if args.dataset_path:
        revision = run_config(protocol)['assets']['datasetRevision'] if protocol else None
        rows, dataset_info = load_dataset(args.dataset_path, revision)
        reference = {
            'datasetRevision': dataset_info['datasetRevision'],
            'questions': [
                dict(question_id=r['question_id'], relation=r['relation'], ground_truth=r['answer'], originalIndex=i)
                for i, r in enumerate(rows)
            ],
        }
    summary, table = export(records, protocol, reference)
    folder = Path(args.output_dir).resolve()
    original = Path(args.results_path).resolve()
    if original in [folder / 'summary.json', folder / 'per-question.csv']:
        raise ValueError('The export destination would overwrite the source results')
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'summary.json', summary)
    write_question_csv(folder, table)
    emit(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
