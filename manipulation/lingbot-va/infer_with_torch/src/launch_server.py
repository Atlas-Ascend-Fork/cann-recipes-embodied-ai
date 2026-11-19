# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start the LingBot-VA NPU server."""
import ast
import hashlib
import importlib
import inspect

import numpy as np
import torch
import torch_npu  # noqa: F401
import wan_va_server as server

from cann_fusion_patch import install as install_fusions
from constant_cache_patch import install as install_constants
from npu_inference_patch import install as retain_weights


def main():
    if server.VA_CONFIGS['libero'].enable_offload:
        raise RuntimeError('The validated configuration requires all core components on NPU')
    retain_weights(server)
    install_constants(server)
    module = importlib.import_module(server.load_transformer.__module__.rsplit('.', 1)[0] + '.model')
    attention_ast = ast.parse(inspect.getsource(module.WanAttention)).body[0]
    fingerprint = hashlib.sha256(ast.dump(attention_ast).encode()).hexdigest()
    if fingerprint != '03b253e87d97a1e760f612f74f30f745afc75ead2819b51589f87b75f245ace9':
        raise RuntimeError('Unreviewed upstream attention implementation')
    install_fusions(module, rms=False, rope=True, attention=True)
    original = server.VA_Server.infer

    def infer(self, obs):
        if obs.get('reset') and '_e2e_seed' in obs:
            seed = int(obs['_e2e_seed'])
            torch.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            np.random.seed(seed)
        return original(self, obs)

    server.VA_Server.infer = infer
    server.init_logger()
    server.main()


if __name__ == '__main__':
    main()
