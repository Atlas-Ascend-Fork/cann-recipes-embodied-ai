# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
from native_vae.api import _build_model
from native_vae.dataset import NativeTensorBundleDataset, generate_tensor_bundle
from native_vae.losses import NativeHandBank, compute_native_vae_loss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def _run_random_training_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.pt"
        generate_tensor_bundle(hand_config=ROOT / "configs/right_hands.yaml", output=path,
                               samples_per_hand=4, seed=7, fk_batch_size=4, device="cpu")
        dataset = NativeTensorBundleDataset(path)
        batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
        config = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
        model = _build_model(config, torch.device("cpu"))
        bank = NativeHandBank.build(ROOT / "configs/right_hands.yaml", "cpu", dataset.hand_names)
        options = {key: batch[key] for key in ("hand_name", "joint_queries", "q_lower", "q_upper", "q_mask")}
        output = model(batch["x_gesture_norm"], batch["morphology_vec"], **options)
        losses = compute_native_vae_loss(
            model=model, output=output, batch=batch, hand_bank=bank,
            lambda_q=1.0, lambda_tip=5.0, lambda_cross_abs=5.0,
            lambda_cross_pair_vector=1.0, lambda_cross_pair_distance=0.3,
            beta_action=0.0001, beta_morphology=1e-05,
            finger_weights_cross_abs=(1.5, 1.2, 1.2, 0.8, 0.8),
        )
        losses["loss"].backward()
        return (
            losses["loss"], losses["loss_cross_tip_abs"],
            losses["loss_cross_pair_vector"], losses["loss_cross_pair_distance"],
            any(parameter.grad is not None for parameter in model.parameters()),
        )


class NativeVAETrainingTests(unittest.TestCase):
    def test_random_data_and_cross_hand_backward(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            generate_tensor_bundle(hand_config=ROOT / "configs/right_hands.yaml", output=path,
                                   samples_per_hand=4, seed=7, fk_batch_size=4, device="cpu")
            dataset = NativeTensorBundleDataset(path)
            self.assertEqual(len(dataset), 12)
            self.assertEqual(set(dataset.hand_names), {"shadow_hand_right", "gaia_hand_right", "sharpa_hand_right"})
            for hand, indices in dataset.hand_indices.items():
                static = dataset.static_by_hand[hand]
                q = torch.stack([dataset[index]["q"] for index in indices])
                mask = static["q_mask"].bool()
                self.assertTrue((q[:, mask] >= static["q_lower"][mask] - 1e-06).all())
                self.assertTrue((q[:, mask] <= static["q_upper"][mask] + 1e-06).all())
        loss, cross_abs, pair_vector, pair_distance, has_grad = _run_random_training_case()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(cross_abs), 0.0)
        self.assertGreater(float(pair_vector), 0.0)
        self.assertGreater(float(pair_distance), 0.0)
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
