#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging

import torch
from _common import PROJECT_ROOT
from native_vae import NativeVAE

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the minimal Native-VAE package and print its contract.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    vae = NativeVAE.from_pretrained(device=args.device)
    LOGGER.info(f"project_root: {PROJECT_ROOT}")
    LOGGER.info(f"device: {vae.device}")
    LOGGER.info("z_gesture_dim: 24")
    for hand in vae.hand_names:
        runtime = vae.runtimes[hand]
        q = torch.zeros(args.batch_size, len(runtime.spec.active_joint_names), dtype=torch.float32, device=vae.device)
        z = vae.encode(q, hand)
        decoded = vae.decode(z, hand)
        tips = vae.fingerpads(decoded, hand)
        LOGGER.info(
            f"{hand}: q={tuple(q.shape)}, z={tuple(z.shape)}, "
            f"decoded={tuple(decoded.shape)}, fingerpads={tuple(tips.shape)}"
        )


if __name__ == "__main__":
    main()
