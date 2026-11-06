#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Export a locally built ARM64 image for OBS distribution via docker load."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

LOGGER = logging.getLogger(__name__)


def export(image, output):
    repository = Path(__file__).resolve().parents[3]
    output = output.resolve()
    if output == repository or repository in output.parents:
        raise ValueError('Keep image archives outside the repository')
    executable = shutil.which('docker')
    if executable is None:
        raise FileNotFoundError('Docker CLI was not found in PATH')
    docker = str(Path(executable).resolve())
    details = json.loads(subprocess.check_output([docker, 'image', 'inspect', image], text=True))[0]
    if (details.get('Os'), details.get('Architecture')) != ('linux', 'arm64'):
        raise ValueError('Expected a Linux ARM64 image; refusing to publish a different platform')
    output.mkdir(parents=True, exist_ok=True)
    names = ['beingh05-image.tar', 'beingh05-image.json', 'SHA256SUMS']
    if any((output / name).exists() for name in names):
        raise ValueError('Use a new output directory; existing release files will not be overwritten')
    with tempfile.TemporaryDirectory(prefix='.image-export-', dir=output) as tmp:
        stage = Path(tmp)
        archive = stage / names[0]
        # Export the inspected ID so a concurrent retag cannot change the image.
        # Own the output file even when the Docker CLI is a sudo wrapper.
        with archive.open('wb') as stream:
            subprocess.run([docker, 'save', details['Id']], stdout=stream, check=True)
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        metadata = {
            'requested_tag': image, 'image_id': details['Id'],
            'os': 'linux', 'architecture': 'arm64',
            'archive': names[0], 'size': archive.stat().st_size,
            'sha256': digest.hexdigest(), 'url': '',
            'load_note': 'After docker load, tag image_id with requested_tag if needed.',
        }
        (stage / names[1]).write_text(json.dumps(metadata, indent=2) + '\n')
        (stage / names[2]).write_text(f'{digest.hexdigest()}  {names[0]}\n')
        for name in names:
            (stage / name).rename(output / name)
    LOGGER.info('Exported %s (SHA256 %s)', output / names[0], digest.hexdigest())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='beingh05-npu:assets-v4')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.image, args.output)
