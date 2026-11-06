#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Adapt the pinned upstream requirements to the ARM64/FSDP recipe."""
from pathlib import Path
import re
import sys

# The recipe already has an OpenCV fallback for decord, whose pinned release has
# no ARM64 wheel. Training uses PyTorch FSDP; no assembled source imports deepspeed.
OMIT = {'decord', 'deepspeed'}
# This upstream dependency is pinned to the published PyPI
# wheel with a content hash rather than an unpinned alternate package.
PYTORCH3D = (
    'pipablepytorch3d @ https://repo.huaweicloud.com/repository/pypi/packages/'
    '8a/4a/66eb75c2c10577952a3f544ec2831289c90c7b994a48f553f1d7486b61d4/'
    'pipablepytorch3d-0.7.6-py3-none-any.whl'
    '#sha256=dc652d0bd6eaf71f5c79a250368231eab6b4f803caf1ac2dc71ec6268f3fb314'
)


def prepare(sources, destination):
    lines = []
    for source in sources:
        for line in source.read_text().splitlines():
            package = re.split(r'[<>=!~\[;\s]', line.strip(), maxsplit=1)[0].lower()
            if package not in OMIT:
                lines.append(line)
    lines = [PYTORCH3D if line.strip() == 'pipablepytorch3d' else line for line in lines]
    # NumPy 1.26.4 needs OpenCV 4.11; upstream 4.12 requires NumPy 2 on Python 3.11.
    lines = ['opencv-python==4.11.0.86' if line.strip() == 'opencv-python==4.12.0.88'
             else line for line in lines]
    lines.extend(['torchvision==0.22.1', 'av>=12.0', 'sentencepiece>=0.2'])
    destination.write_text('\n'.join(dict.fromkeys(lines)) + '\n')


if __name__ == '__main__':
    prepare([Path(path) for path in sys.argv[1:-1]], Path(sys.argv[-1]))
