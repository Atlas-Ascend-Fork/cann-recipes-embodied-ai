#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from typing import Any
from pathlib import Path

import mujoco.viewer
import numpy as np
from sim import GraspEnv, GraspEnvConfig

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_actions(path: Path, key: str, episode: int) -> np.ndarray:
    if path.suffix == ".npy":
        actions = np.load(path)
    else:
        payload = np.load(path, allow_pickle=True)
        if key not in payload.files:
            raise KeyError(f"{key!r} not found; available={payload.files}")
        actions = payload[key]
        if actions.ndim == 3:
            actions = actions[episode]
            if "length" in payload.files:
                actions = actions[: int(payload["length"][episode])]
    if actions.ndim != 2:
        raise ValueError(f"Expected [T,D] actions, got {actions.shape}")
    return np.asarray(actions, dtype=np.float32)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Saving MP4 requires the system 'ffmpeg' executable.")
    if not frames:
        raise ValueError("No RGB frames were rendered.")
    video = np.stack(frames).astype(np.uint8, copy=False)
    height, width = video.shape[1:3]
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, input=video.tobytes(), check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay native-q grasp actions in MuJoCo.")
    parser.add_argument("--hand", required=True, choices=["shadow_hand_right", "gaia_hand_right", "sharpa_hand_right"])
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scene_xml", type=Path, default=None)
    parser.add_argument("--base_scene", type=Path, default=None)
    parser.add_argument("--camera", default="ego_opposite")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drive_mode", choices=["ctrl", "qpos"], default="ctrl")
    parser.add_argument("--physics_substep_multiplier", type=int, default=1)
    parser.add_argument("--hold_frames", type=int, default=60)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


def _build_env_config(args, action_count: int) -> GraspEnvConfig:
    values = dict(
        hand=args.hand, scene_xml=args.scene_xml, camera=args.camera,
        width=args.width, height=args.height, fps=args.fps,
        drive_mode=args.drive_mode,
        physics_substep_multiplier=args.physics_substep_multiplier,
        max_steps=action_count, render_images=args.output_video is not None,
    )
    if args.base_scene is not None:
        values["base_scene"] = args.base_scene
    return GraspEnvConfig(**values)


def _step_environment(env: GraspEnv, actions: np.ndarray, args, frames: list[np.ndarray]) -> tuple[dict[str, Any], Any]:
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.gui else None
    try:
        for action in actions:
            started = time.perf_counter()
            observation, _, _, _, info = env.step(action)
            if observation["image"] is not None:
                frames.append(observation["image"])
            if viewer is not None:
                viewer.sync()
            if args.realtime:
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - started)))
    finally:
        if viewer is not None:
            viewer.close()
    return info, viewer


def _run_environment(args, actions: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames: list[np.ndarray] = []
    with GraspEnv(_build_env_config(args, len(actions))) as env:
        env.reset(initial_action=actions[0])
        info, _ = _step_environment(env, actions, args, frames)
    return frames, info


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    actions = load_actions(args.actions, args.action_key, args.episode)
    if len(actions) == 0:
        raise ValueError("Action trajectory is empty.")
    if args.hold_frames > 0:
        actions = np.concatenate(
            [actions, np.repeat(actions[-1:], args.hold_frames, axis=0)], axis=0
        )
    frames, info = _run_environment(args, actions)
    if args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(args.output_video, frames, args.fps)
    return {"frames": len(actions), "success": int(info["success"]), "max_lift_m": info["max_lift_m"]}


def main() -> None:
    args = _parse_args()
    result = _replay(args)
    LOGGER.info(
        f"[done] hand={args.hand} frames={result['frames']} "
        f"success={result['success']} max_lift={result['max_lift_m']:.4f}m"
    )


if __name__ == "__main__":
    main()
