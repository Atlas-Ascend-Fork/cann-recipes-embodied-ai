# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import mujoco


def run_hold_gui(env, camera: str = "free") -> None:
    observation, _ = env.reset()
    hold_action = observation["state"].copy()
    viewer = env.launch_gui()
    if camera != "free":
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise KeyError(f"Camera {camera!r} is missing from the scene.")
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
    while env.gui_is_running:
        started = time.perf_counter()
        observation, _, _, _, _ = env.step(hold_action)
        env.sync_gui()
        time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))
