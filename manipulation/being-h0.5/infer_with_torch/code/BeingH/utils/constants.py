# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Constants and architecture registry used by production inference."""

from enum import Enum

from BeingH.model import InternVisionConfig, InternVisionModel, Qwen3Config, Qwen3ForCausalLM
from BeingH.model.layers import InternVLConnector

LLM_MODEL_ARCH = {"Qwen3ForCausalLM": (Qwen3Config, Qwen3ForCausalLM)}
VIT_MODEL_ARCH = {"InternVisionModel": (InternVisionConfig, InternVisionModel)}
CONNECTOR_ARCH = {"internvl_connector": InternVLConnector}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EmbodimentTag(Enum):
    LIBERO_FRANKA = "libero_franka_gripper"
    LIBERO = "libero"
    ROBOCASA = "robocasa"
    NEW_EMBODIMENT = "new_embodiment"


EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.LIBERO_FRANKA.value: 0,
    EmbodimentTag.LIBERO.value: 0,
    EmbodimentTag.ROBOCASA.value: 31,
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
}
TARGET_STATE_ROTATION_TYPE = "axis_angle"
TARGET_ACTION_ROTATION_TYPE = "axis_angle"
TARGET_STATE_ROTATION_DIM = 3
TARGET_ACTION_ROTATION_DIM = 3
AGIBOT_ABS_OR_RELA = "relative"
