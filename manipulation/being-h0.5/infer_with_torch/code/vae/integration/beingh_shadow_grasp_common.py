# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared imports and constants for Being-H closed-loop evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import av
import numpy as np
import torch
from BeingH.inference.beingh_policy import BeingHPolicy, PolicyRuntimeConfig, PolicyTaskConfig
from native_vae import NativeVAE
from vae.optimize import GeometryRetargeter, GeometryRetargeterConfig
from scipy.spatial.transform import Rotation
from vae.sim import (
    POLICY_WRIST_EULER_OFFSETS,
    GesturePolicyAdapter,
    GraspEnv,
    GraspEnvConfig,
    PolicyEvaluationClient,
    load_dataset_object_episode,
)

from .checkpoint_data_config import resolve_checkpoint_data_config

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_ROOT = REPO_ROOT / "vae"
for import_root in (REPO_ROOT, VAE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SUPPORTED_HANDS = ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right")
SUCCESS_PROFILES = {
    "strict": {"lift_m": 0.2, "frames": 10},
    "loose": {"lift_m": 0.1, "frames": 1},
    "dex-demo": {"lift_m": 0.08, "frames": 1},
}
RAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture"
Q99_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture_q99"
MINMAX_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture_minmax"
WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_minmax_gesture_raw"
WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_euler_minmax_gesture_raw"
WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_rot6d_minmax_gesture_raw"
TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_2cam_wrist_euler_minmax_gesture_raw"
TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_2cam_wrist_rot6d_minmax_gesture_raw"
TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME = "shadow_grasp_2cam_wrist_rot6d_minmax_joints"
SHARPA_JOINT_DATA_CONFIG_NAME = "sharpa_grasp_2cam_wrist_rot6d_minmax_joints"
GAIA_JOINT_DATA_CONFIG_NAME = "gaia_grasp_2cam_wrist_rot6d_minmax_joints"
SUPPORTED_DATA_CONFIG_NAMES = (
    RAW_DATA_CONFIG_NAME,
    Q99_DATA_CONFIG_NAME,
    MINMAX_DATA_CONFIG_NAME,
    WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
ROT6D_DATA_CONFIG_NAMES = (
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
TWO_CAMERA_DATA_CONFIG_NAMES = (
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
JOINT_ACTION_DATA_CONFIG_NAMES = (
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
DIRECT_JOINT_CONFIG_HANDS = {
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: "shadow_hand_right",
    SHARPA_JOINT_DATA_CONFIG_NAME: "sharpa_hand_right",
    GAIA_JOINT_DATA_CONFIG_NAME: "gaia_hand_right",
}
DIRECT_JOINT_CONFIG_DIMS = {
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: 22,
    SHARPA_JOINT_DATA_CONFIG_NAME: 22,
    GAIA_JOINT_DATA_CONFIG_NAME: 15,
}
