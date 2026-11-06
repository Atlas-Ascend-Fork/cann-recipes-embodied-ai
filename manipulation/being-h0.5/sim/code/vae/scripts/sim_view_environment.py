#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sim import GraspEnv, GraspEnvConfig, list_object_ids
from sim.viewer import run_hold_gui

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HANDS = ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the standalone grasp environment in MuJoCo GUI.")
    parser.add_argument("--hand", choices=HANDS, default="shadow_hand_right")
    parser.add_argument("--scene_xml", type=Path, default=None)
    parser.add_argument("--base_scene", type=Path, default=None)
    parser.add_argument("--camera", choices=["free", "ego", "ego_opposite", "wrist"], default="free")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drive_mode", choices=["ctrl", "qpos"], default="ctrl")
    parser.add_argument("--home_wrist", type=float, nargs=6, default=None)
    object_group = parser.add_mutually_exclusive_group()
    object_group.add_argument("--object_id", type=str, default=None)
    object_group.add_argument("--object_mesh", type=Path, nargs="+", default=None)
    parser.add_argument("--list_objects", action="store_true")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--object_position", type=float, nargs=3, default=[0, 0, 0.035])
    parser.add_argument("--object_quaternion", type=float, nargs=4, default=[1, 0, 0, 0])
    args = parser.parse_args()
    if args.list_objects:
        LOGGER.info("\n".join(list_object_ids()))
        return
    config_kwargs = dict(
        hand=args.hand,
        scene_xml=args.scene_xml,
        fps=args.fps,
        drive_mode=args.drive_mode,
        render_images=False,
        home_wrist=None if args.home_wrist is None else tuple(args.home_wrist),
        object_id=args.object_id,
        object_meshes=tuple(args.object_mesh or ()),
        object_scale=args.object_scale,
        object_position=tuple(args.object_position),
        object_quaternion=tuple(args.object_quaternion),
    )
    if args.base_scene is not None:
        config_kwargs["base_scene"] = args.base_scene
    with GraspEnv(GraspEnvConfig(**config_kwargs)) as env:
        LOGGER.info(f"[gui] hand={args.hand} action_dim={env.action_dim} scene={env.scene_path}")
        run_hold_gui(env, args.camera)


if __name__ == "__main__":
    main()
