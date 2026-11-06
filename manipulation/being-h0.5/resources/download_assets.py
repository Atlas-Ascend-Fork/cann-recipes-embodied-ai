#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Verify and install an offline resource archive or download it from OBS (Python 3.9+)."""
import argparse
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import urllib.request

LOGGER = logging.getLogger(__name__)


def digest(path):
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    if '\\' in name or path.as_posix() != name:
        raise ValueError(f'Unsafe archive path: {name}')
    return path


def stage_member(tar, member, expected, target):
    if member.size != expected['size']:
        raise ValueError(f'Unexpected size: {member.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tar.extractfile(member) as src, target.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    if digest(target) != expected['sha256']:
        raise ValueError(f'File SHA256 mismatch: {member.name}')


def stage_archive(archive, files, staged):
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            relative = safe_name(name)
            if not member.isfile() or name in seen or name not in files:
                raise ValueError(f'Unexpected, duplicate, or non-regular entry: {name}')
            stage_member(tar, member, files[name], staged / relative)
            seen.add(name)
    if seen != set(files):
        raise ValueError('Archive is missing manifest entries')
    return seen


def validate_destination(target, destination, expected):
    if any(p.is_symlink() for p in [target, *target.parents] if p != destination.parent):
        raise ValueError(f'Symlink in destination: {target}')
    if target.exists() and (not target.is_file() or digest(target) != expected['sha256']):
        raise ValueError(f'Refusing to overwrite different file: {target}')
    for parent in target.parents:
        if parent == destination:
            break
        if parent.exists() and not parent.is_dir():
            raise ValueError(f'Non-directory parent: {parent}')


def install(spec, archive, destination):
    if archive.stat().st_size != spec['size'] or digest(archive) != spec['sha256']:
        raise ValueError('Archive size/SHA256 mismatch; destination was not changed')
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with tempfile.TemporaryDirectory(prefix='beingh-assets-') as tmp:
        staged = Path(tmp)
        seen = stage_archive(archive, spec['files'], staged)
        # Validate every destination before copying; never traverse an existing symlink.
        for name in seen:
            validate_destination(destination / name, destination, spec['files'][name])
        for name in sorted(seen):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(staged / name, target)
    LOGGER.info('Installed and verified %s files in %s', len(seen), destination)


def download(url, archive, expected_size):
    with urllib.request.urlopen(url, timeout=60) as src, archive.open('wb') as dst:
        if not src.geturl().startswith('https://'):
            raise ValueError('Download redirected outside HTTPS')
        remaining = expected_size
        while chunk := src.read(min(1024 * 1024, remaining + 1)):
            remaining -= len(chunk)
            if remaining < 0:
                raise ValueError('Download exceeds expected archive size')
            dst.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', choices=['runtime', 'videos'], default='runtime')
    parser.add_argument('--archive', type=Path, help='Offline archive; no network access when specified')
    parser.add_argument('--url', help='Community-provided HTTPS URL; manifest checksum is still required')
    parser.add_argument('--dest', required=True, type=Path,
                        help='Assembled Being-H05 root, or external video directory')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())['packages'][args.package]
    if args.archive:
        install(spec, args.archive, args.dest)
        return
    url = args.url or spec['url']
    if not url:
        parser.error('OBS URL is pending. Use --archive /path/to/' + spec['filename'])
    if not url.startswith('https://'):
        parser.error('Use an HTTPS resource URL')
    with tempfile.TemporaryDirectory(prefix='beingh-download-') as tmp:
        archive = Path(tmp) / spec['filename']
        download(url, archive, spec['size'])
        install(spec, archive, args.dest)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as error:
        LOGGER.error('Asset installation failed: %s', error)
        sys.exit(1)
