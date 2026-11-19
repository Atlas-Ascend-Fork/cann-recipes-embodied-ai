# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep frozen inference weights resident with FSDP."""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def retain_inference_weights(model, param_dtype=torch.bfloat16,
                             reduce_dtype=torch.float32):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Retaining weights is restricted to frozen inference")
    policy = MixedPrecisionPolicy(param_dtype=param_dtype,
                                  reduce_dtype=reduce_dtype,
                                  cast_forward_inputs=False)
    # Retain block weights across diffusion steps. Block-level wrapping also
    # avoids redundant attention/FFN FSDP hooks without a model-sized gather.
    options = {"mp_policy": policy, "reshard_after_forward": False}
    for block in model.blocks:
        fully_shard(block, **options)
    fully_shard(model, **options)
    return model


def install(server_module):
    server_module.shard_model = retain_inference_weights
