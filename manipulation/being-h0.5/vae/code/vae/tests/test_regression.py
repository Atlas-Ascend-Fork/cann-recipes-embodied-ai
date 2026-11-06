# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from native_vae import NativeVAE

REFERENCE = Path(__file__).resolve().parent / "reference_outputs.json"


class NativeVAERegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vae = NativeVAE.from_pretrained(device="cpu")
        with REFERENCE.open() as stream:
            reference = json.load(stream)
        cls.reference = {
            name: np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])
            for name, item in reference.items()
        }

    def test_against_original_project(self) -> None:
        source = "shadow_hand_right"
        q = self.reference["shadow_q"]
        z = self.vae.encode(q, source)
        np.testing.assert_allclose(z.detach().cpu().numpy(), self.reference["shadow_z"], rtol=1e-06, atol=1e-06)
        for target in self.vae.hand_names:
            result = self.vae.retarget(q, source, target)
            np.testing.assert_allclose(
                result.target_q.detach().cpu().numpy(), self.reference[f"{target}_q"], rtol=1e-06, atol=1e-06
            )
            np.testing.assert_allclose(
                result.target_fingerpads.detach().cpu().numpy(),
                self.reference[f"{target}_tips"],
                rtol=1e-06,
                atol=1e-06,
            )


if __name__ == "__main__":
    unittest.main()
