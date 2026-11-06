# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Small Decord-compatible reader used when no AArch64 Decord wheel exists."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ArrayBatch:
    """Match the ``decord.NDArray.asnumpy`` contract used by Being-H."""

    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def asnumpy(self) -> np.ndarray:
        return self._value


class OpenCVVideoReader:
    """Implement the subset of Decord's VideoReader required by Being-H."""

    def __init__(self, source: str | Path | io.BytesIO, **kwargs) -> None:
        self._temporary_path: str | None = None
        self._width = int(kwargs.get("width", 0) or 0)
        self._height = int(kwargs.get("height", 0) or 0)
        if hasattr(source, "read"):
            payload = source.getvalue() if hasattr(source, "getvalue") else source.read()
            handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                handle.write(payload)
            finally:
                handle.close()
            self._temporary_path = handle.name
            self._source = handle.name
        else:
            self._source = str(source)
        capture = self._open()
        try:
            self._length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        finally:
            capture.release()
        if self._length <= 0:
            self.close()
            raise ValueError(f"Unable to determine video length: {self._source}")

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self._length

    def get_avg_fps(self) -> float:
        return self._fps

    def get_frame_timestamp(self, indices: Iterable[int]) -> np.ndarray:
        values = np.asarray(list(indices), dtype=np.float64)
        starts = values / self._fps
        return np.stack((starts, starts + 1.0 / self._fps), axis=1)

    def get_batch(self, indices: Iterable[int]) -> ArrayBatch:
        capture = self._open()
        frames: list[np.ndarray] = []
        try:
            for raw_index in indices:
                index = int(raw_index)
                if index < 0 or index >= self._length:
                    raise IndexError(f"Frame index {index} outside [0, {self._length})")
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Unable to read frame {index}: {self._source}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._width and self._height:
                    frame = cv2.resize(frame, (self._width, self._height))
                frames.append(frame)
        finally:
            capture.release()
        return ArrayBatch(np.stack(frames, axis=0))

    def close(self) -> None:
        if self._temporary_path:
            try:
                os.unlink(self._temporary_path)
            except FileNotFoundError:
                pass
            self._temporary_path = None

    def _open(self):
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open video: {self._source}")
        return capture


class DecordCompat:
    VideoReader = OpenCVVideoReader


decord_compat = DecordCompat()
