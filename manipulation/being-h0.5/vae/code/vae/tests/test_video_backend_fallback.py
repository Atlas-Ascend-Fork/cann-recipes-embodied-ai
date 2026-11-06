# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from BeingH.utils import video_utils
from BeingH.utils.opencv_video_reader import OpenCVVideoReader


class OpenCVVideoReaderTests(unittest.TestCase):
    def test_decord_compatible_rgb_batch_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for red in (20, 80, 140):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                frame[:, :, 2] = red
                writer.write(frame)
            writer.release()

            reader = OpenCVVideoReader(path)
            frames = reader.get_batch([0, 2]).asnumpy()
            self.assertEqual(len(reader), 3)
            self.assertEqual(frames.shape, (2, 24, 32, 3))
            self.assertGreater(float(frames[0, :, :, 0].mean()), 10.0)
            self.assertLess(float(frames[0, :, :, 2].mean()), 5.0)
            np.testing.assert_allclose(reader.get_frame_timestamp([0, 2])[:, 0], [0.0, 0.2], atol=0.02)

    def test_torchvision_backend_falls_back_without_pyav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for red in (20, 80, 140):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                frame[:, :, 2] = red
                writer.write(frame)
            writer.release()

            original_av = video_utils.av
            try:
                video_utils.av = None
                frames = video_utils.get_frames_by_timestamps(str(path), [0.0, 0.2], video_backend="torchvision_av")
            finally:
                video_utils.av = original_av
            self.assertEqual(frames.shape, (2, 24, 32, 3))
            self.assertGreater(float(frames[0, :, :, 0].mean()), 10.0)
            self.assertLess(float(frames[0, :, :, 2].mean()), 5.0)


if __name__ == "__main__":
    unittest.main()
