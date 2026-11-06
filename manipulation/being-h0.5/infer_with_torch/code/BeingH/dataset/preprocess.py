# Copyright (c) 2024 OpenGVLab
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing for the production Being-H0.5 inference path."""

import torchvision.transforms as transforms
from BeingH.utils.constants import IMAGENET_MEAN, IMAGENET_STD


def build_vit_transform_base(
    is_train: bool, force_image_size: int, transform_type: str = "default", normalize_type: str = "imagenet", **_unused
):
    """Build the fixed ImageNet transform used by the released checkpoint."""
    if is_train:
        raise ValueError("the production inference runtime does not build train transforms")
    if transform_type != "default":
        raise ValueError("the production inference runtime requires the default transform")
    if normalize_type != "imagenet":
        raise ValueError("the Being-H0.5 checkpoint requires ImageNet normalization")
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            transforms.Resize(force_image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(force_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return (None, transform)
