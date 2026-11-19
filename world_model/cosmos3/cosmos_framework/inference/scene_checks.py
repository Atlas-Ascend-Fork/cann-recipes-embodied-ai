# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Early validation for the Cosmos3 NPU inference scenes."""


def check_scene_inputs(samples):
    """Validate sound output dependencies.

    Called after upstream URL resolution, before model allocation. Keep upstream
    argument parsing and model-specific dimension validation authoritative.
    """
    for sample in samples:
        mode = getattr(sample.model_mode, "value", sample.model_mode)
        if sample.enable_sound:
            if mode == "text2image":
                raise ValueError("enable_sound is unsupported for single-frame image generation")
            try:
                import av
                av.codec.Codec("aac", "w")
            except (ImportError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    "enable_sound requires PyAV with an AAC encoder; "
                    "install the documented encoding dependencies"
                ) from error
