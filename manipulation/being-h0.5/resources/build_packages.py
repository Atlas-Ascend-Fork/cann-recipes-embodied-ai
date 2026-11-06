#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build reproducible resource archives from an external staging directory."""
import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import tarfile

LOGGER = logging.getLogger(__name__)


def digest(path):
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def add_asset(tar, path, root):
    relative = path.relative_to(root).as_posix()
    with path.open('rb') as stream:
        if stream.read(50).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'LFS pointer, not an asset: {path}')
        stream.seek(0)
        info = tar.gettarinfo(str(path), arcname=relative)
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        info.mode = 0o644
        tar.addfile(info, stream)
    return relative, {'size': path.stat().st_size, 'sha256': digest(path)}


def write_archive(archive, root, files):
    inventory = {}
    with archive.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for path in files:
                relative, metadata = add_asset(tar, path, root)
                inventory[relative] = metadata
    return inventory


def build(source, output, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest['packages'].items():
        root = source / name
        files = sorted(p for p in root.rglob('*') if p.is_file())
        if not files or any(p.is_symlink() for p in root.rglob('*')):
            raise ValueError(f'Empty package or symlink in {root}')
        archive = output / spec['filename']
        inventory = write_archive(archive, root, files)
        spec.update(size=archive.stat().st_size, sha256=digest(archive), files=inventory)
        # A rebuilt archive must be explicitly published before it has a URL.
        spec['url'] = ''
        LOGGER.info('%s: %s files, SHA256 %s', archive, len(files), spec['sha256'])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='External directory containing runtime/ and videos/')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), args.manifest)
