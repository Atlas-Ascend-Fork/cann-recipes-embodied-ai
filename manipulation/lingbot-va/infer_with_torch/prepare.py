# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the source tree and checkpoints used by the server."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import logging
import sys


LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
GIT = '/usr/bin/git'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_source(root):
    lock = json.loads((HERE / 'configs/source_lock.json').read_text())
    head = subprocess.check_output([GIT, '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != lock['upstream_commit']:
        raise RuntimeError('Use the exact official upstream commit: ' + lock['upstream_commit'])
    patch = HERE / 'patches/upstream_to_npu.patch'
    if sha(patch) != lock['patch_sha256']:
        raise RuntimeError('Source patch hash mismatch')
    dirty = subprocess.check_output([GIT, '-C', str(root), 'diff', 'HEAD', '--name-only'], text=True).splitlines()
    if set(dirty) - set(lock['patched_files']):
        raise RuntimeError('Unreviewed upstream modifications: ' + repr(dirty))
    already_applied = all((root / name).is_file() and sha(root / name) == value
                          for name, value in lock['patched_files'].items())
    if not already_applied:
        if dirty:
            raise RuntimeError('Partial or changed patch detected; use a clean pinned checkout')
        subprocess.run([GIT, '-C', str(root), 'apply', '--unidiff-zero', '--check', str(patch)], check=True)
        subprocess.run([GIT, '-C', str(root), 'apply', '--unidiff-zero', str(patch)], check=True)
    for name, expected in lock['patched_files'].items():
        if sha(root / name) != expected:
            raise RuntimeError('Patched source differs from evaluated source: ' + name)
    tree = ast.parse((root / 'wan_va/wan_va_server.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'VA_Server')
    methods = []
    for name, expected in lock['reference_method_ast_sha256'].items():
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
        if hashlib.sha256(ast.dump(method).encode()).hexdigest() != expected:
            raise RuntimeError('Unreviewed cache target method: ' + name)
        methods.append(method)
    # Generate the selected upstream methods for validation.
    reference = ast.Module(body=[ast.ClassDef(name='VA_Server', bases=[], keywords=[],
                                             body=methods, decorator_list=[])], type_ignores=[])
    target = HERE / 'src/server_reference.py'
    temporary = target.with_suffix('.tmp')
    temporary.write_text('# Generated from pinned Robbyant/LingBot-VA for AST validation only.\n'
                         + ast.unparse(ast.fix_missing_locations(reference)) + '\n')
    temporary.replace(target)
    result = json.dumps(dict(upstream_commit=head,
                             patched_files_verified=len(lock['patched_files']),
                             generated_reference=str(target)))
    LOGGER.info(result)
    return result


def _prepare_shared_links(root, shared_links):
    for relative, target in shared_links.items():
        link = root / relative
        expected = (link.parent / target).resolve()
        if not expected.is_dir():
            raise FileNotFoundError(expected)
        if link.is_symlink():
            if link.resolve() != expected:
                raise RuntimeError('Incorrect shared component link: ' + str(link))
        elif link.exists():
            raise RuntimeError('Use the locked shared-component link: ' + str(link))
        else:
            link.symlink_to(target, target_is_directory=True)


def prepare_checkpoints(root, download=False):
    manifest = json.loads((HERE / 'configs/checkpoints.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    verified = []
    for suite in manifest['suite_mapping']:
        folder = root / suite['model_subdirectory']
        if download:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=suite['model_repository'], revision=suite['weights_revision'],
                              allow_patterns=[p for p in suite['files'] if p != 'configuration.json'],
                              local_dir=folder)
        # Reject unrecognized configuration changes before writing files.
        config = folder / 'transformer/config.json'
        locked = HERE / 'configs' / suite['transformer_config_file']
        expected = suite['files']['transformer/config.json']['sha256']
        if sha(locked) != expected:
            raise RuntimeError('Bundled config hash mismatch: ' + suite['suite'])
        published = json.loads(locked.read_text())
        for key, values in suite['config_overrides'].items():
            published[key] = values['published']
        if config.exists():
            current = json.loads(config.read_text())
            if current not in (json.loads(locked.read_text()), published):
                raise RuntimeError('Unreviewed local checkpoint config: ' + str(config))
        else:
            raise FileNotFoundError(config)
        config.write_bytes(locked.read_bytes())
        metadata = folder / 'configuration.json'
        metadata_bytes = b'{"framework":"Pytorch","task":"robotics"}'
        if metadata.exists() and metadata.read_bytes() != metadata_bytes:
            raise RuntimeError('Unexpected additional repository metadata: ' + str(metadata))
        metadata.write_bytes(metadata_bytes)
        for relative, info in suite['files'].items():
            path = folder / relative
            if (not path.is_file() or path.stat().st_size != info['size']
                    or sha(path) != info['sha256']):
                raise RuntimeError('Checkpoint file hash/size mismatch: ' + str(path))
        verified.append(dict(suite=suite['suite'], revision=suite['weights_revision'],
                             file_count=len(suite['files']), config_overrides=suite['config_overrides']))
    _prepare_shared_links(root, manifest['shared_links'])
    result = json.dumps(dict(verified=verified,
                             shared_links_verified=len(manifest['shared_links'])))
    LOGGER.info(result)
    return result


def main():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.handlers = [handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    source = operations.add_parser('source')
    source.add_argument('upstream_root', type=Path)
    checkpoints = operations.add_parser('checkpoints')
    checkpoints.add_argument('--models-root', type=Path, required=True)
    checkpoints.add_argument('--download', action='store_true')
    args = parser.parse_args()
    # Each preparation operation verifies files while also updating configs or
    # AST guards. Serialize these writes across clients using this sample.
    with (HERE / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.operation == 'source':
            result = prepare_source(args.upstream_root.resolve())
        else:
            result = prepare_checkpoints(args.models_root.resolve(), args.download)


if __name__ == '__main__':
    main()
