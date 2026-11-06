#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check installed imports, real simulation assets and optional NPU arithmetic."""
import argparse
import importlib
import logging
from pathlib import Path
import sys
import unittest

LOGGER = logging.getLogger(__name__)


def check_simulation(root):
    mujoco = importlib.import_module('mujoco')
    scene = root / 'vae/sim/assets/demos/tissuebox_idx83_shadow/scene.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for _ in range(5):
        mujoco.mj_step(model, data)
    LOGGER.info('MUJOCO_OK %s %s', model.nq, model.nbody)


def check_npu():
    torch = importlib.import_module('torch')
    importlib.import_module('torch_npu')
    if not torch.npu.is_available():
        raise RuntimeError('NPU is not available in the container')
    x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    actual = (x.to('npu:0') @ x.to('npu:0').T).cpu()
    torch.testing.assert_close(actual, x @ x.T)
    torch.npu.synchronize()
    LOGGER.info('NPU_MATMUL_OK %s', torch.npu.get_device_name(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npu', action='store_true', help='Also run a small matrix multiplication on npu:0')
    args = parser.parse_args()
    root = Path('/opt/beingh05')
    sys.path.insert(0, str(root))
    for module in ['torch', 'torchvision', 'numpy', 'pyarrow', 'pyarrow.parquet', 'datasets',
                   'mujoco', 'pytorch3d.transforms',
                   'native_vae', 'BeingH.utils.video_utils', 'BeingH.model.beingvla']:
        importlib.import_module(module)
        LOGGER.info('IMPORT_OK %s', module)
    check_simulation(root)
    suite = unittest.defaultTestLoader.discover(str(root / 'vae/tests'), pattern='test_training.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    if args.npu:
        check_npu()
    LOGGER.info('SMOKE_TEST_PASSED')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    sys.exit(main())
