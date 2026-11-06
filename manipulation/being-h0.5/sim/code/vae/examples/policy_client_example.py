# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from native_vae import NativeVAE
from sim import CallablePolicy, GesturePolicyAdapter, GraspEnv, GraspEnvConfig, PolicyEvaluationClient

LOGGER = logging.getLogger(__name__)

BEINGH_ROOT = Path(__file__).resolve().parents[4]
for code_root in (BEINGH_ROOT / "sim" / "code" / "vae", BEINGH_ROOT / "vae" / "code" / "vae"):
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))


def main() -> None:
    hand = "sharpa_hand_right"
    vae = NativeVAE.from_pretrained(device="cuda")

    def latent_policy(observation):
        return np.asarray(observation["state"], dtype=np.float32)

    policy = GesturePolicyAdapter(CallablePolicy(latent_policy), vae=vae, target_hand=hand, encode_observation=True)
    with GraspEnv(GraspEnvConfig(hand=hand, max_steps=30)) as env:
        result = PolicyEvaluationClient(env, policy).run(max_steps=30)
    LOGGER.info(f"success={result.success}, max_lift={result.max_lift_m:.3f}m")


if __name__ == "__main__":
    main()
