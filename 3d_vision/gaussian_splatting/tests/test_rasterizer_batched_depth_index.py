# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_rasterizer():
    mock_names = (
        "acl",
        "rasterization",
        "rasterization.utils",
        "rasterization.config",
        "meta_gauss_render",
    )
    original_modules = {name: sys.modules.get(name) for name in mock_names}
    try:
        sys.modules["acl"] = types.ModuleType("acl")

        rasterization = types.ModuleType("rasterization")
        rasterization.__path__ = []
        sys.modules["rasterization"] = rasterization

        utils = types.ModuleType("rasterization.utils")

        def validate_inputs(*args, **kwargs):
            return None

        utils.validate_inputs = validate_inputs
        sys.modules["rasterization.utils"] = utils

        config = types.ModuleType("rasterization.config")
        config.Config = type("Config", (), {})
        sys.modules["rasterization.config"] = config

        meta_gauss_render = types.ModuleType("meta_gauss_render")
        for name in (
            "spherical_harmonics",
            "projection_three_dims_gaussian_fused",
            "flash_gaussian_build_mask",
            "gaussian_sort",
            "calc_render",
            "get_render_schedule",
        ):
            setattr(meta_gauss_render, name, lambda *args, **kwargs: None)
        sys.modules["meta_gauss_render"] = meta_gauss_render

        source = (
            Path(__file__).resolve().parents[1]
            / "rasterization"
            / "rasterizer.py"
        )
        spec = importlib.util.spec_from_file_location("rasterizer_under_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Rasterizer
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


Rasterizer = _load_rasterizer()


class BatchedDepthIndexTest(unittest.TestCase):
    def test_single_camera_positive_control(self):
        depths = self._render_input(camera_count=1, camera_index=0)
        self.assertEqual(tuple(depths.shape), (1, 3))
        self.assertTrue(torch.equal(depths, torch.tensor([[0.0, 1.0, 2.0]])))

    def test_second_camera_uses_its_depth_row(self):
        depths = self._render_input(camera_count=2, camera_index=1)
        self.assertEqual(tuple(depths.shape), (1, 3))
        self.assertTrue(torch.equal(depths, torch.tensor([[3.0, 4.0, 5.0]])))

    def _render_input(self, camera_count, camera_index):
        gaussian_count = 3
        rasterizer = Rasterizer(object())
        rasterizer.tile_size = 1
        rasterizer.padded_height = 1
        rasterizer.padded_width = 1
        rasterizer.pix_coord = torch.zeros((1, 1, 2), dtype=torch.float32)

        means2d = torch.zeros((1, camera_count, 2, gaussian_count))
        colors = torch.zeros((1, camera_count, 3, gaussian_count))
        opacities = torch.zeros((1, camera_count, gaussian_count))
        conics = torch.zeros((1, camera_count, 3, gaussian_count))
        depths = torch.arange(camera_count * gaussian_count, dtype=torch.float32)
        depths = depths.reshape(1, camera_count, gaussian_count)

        return rasterizer.get_render_input(
            (means2d, colors, opacities, conics), depths, camera_index
        )[0][-1]


if __name__ == "__main__":
    unittest.main()
