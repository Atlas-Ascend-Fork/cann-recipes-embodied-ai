# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the full production dataset class against real Pillow image files."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


class RetryLimitExceeded(BaseException):
    """Stop a regressed unbounded loop without being caught as an I/O error."""


@pytest.fixture(scope="module")
def sky_module():
    source = Path(__file__).resolve().parents[2] / "3d_vision/SLARM/tools/extract_sky.py"
    spec = importlib.util.spec_from_file_location("actual_extract_sky", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == source
    return module


def bounded_loader(data, attempts):
    reader = data.loader

    def counted_reader(path):
        attempts.append(path)
        if len(attempts) > len(data.samples):
            raise RetryLimitExceeded("Production loader retried beyond one dataset cycle")
        return reader(path)

    return counted_reader


def dataset(sky_module, tmp_path, paths, **kwargs):
    file_list = tmp_path / "images.txt"
    file_list.write_text("".join(str(path) + "\n" for path in paths))
    return sky_module.ListDataset(str(file_list), **kwargs)


@pytest.mark.parametrize("count", [1, 3])
def test_all_missing_images_fail_after_one_bounded_cycle(sky_module, tmp_path, count):
    data = dataset(sky_module, tmp_path, [tmp_path / f"missing-{index}.png" for index in range(count)])
    attempts = []

    data.loader = bounded_loader(data, attempts)
    with pytest.raises(RuntimeError, match="No readable image found"):
        data[0]
    assert len(attempts) == count


def test_all_corrupt_images_fail_after_one_cycle(sky_module, tmp_path):
    paths = [tmp_path / "bad-a.png", tmp_path / "bad-b.png"]
    for path in paths:
        path.write_bytes(b"not an image")
    data = dataset(sky_module, tmp_path, paths)
    data.loader = bounded_loader(data, [])
    with pytest.raises(RuntimeError, match="No readable image found"):
        data[0]


def test_missing_image_falls_forward_to_actual_valid_image(sky_module, tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(valid)
    data = dataset(sky_module, tmp_path, [tmp_path / "missing.png", valid], return_path=True)
    image, path = data[0]
    assert path == str(valid) and image.size == (2, 3)
    assert image.getpixel((0, 0)) == (20, 40, 60)


def test_transform_and_non_path_return_are_preserved(sky_module, tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(valid)
    data = dataset(sky_module, tmp_path, [valid], transform=lambda image: image.size)
    assert data[0] == (2, 3)


def test_empty_dataset_keeps_index_error(sky_module, tmp_path):
    data = dataset(sky_module, tmp_path, [])
    with pytest.raises(IndexError):
        data[0]


def test_out_of_range_index_keeps_index_error(sky_module, tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (1, 1)).save(valid)
    data = dataset(sky_module, tmp_path, [valid])
    with pytest.raises(IndexError):
        data[1]


def test_negative_index_keeps_standard_sequence_behavior(sky_module, tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 3)).save(valid)
    data = dataset(sky_module, tmp_path, [valid], return_path=True)
    assert data[-1][1] == str(valid)


def test_static_sample_loader_accepts_the_one_documented_path(sky_module, tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("first.png\nsecond.png\n")
    cls = sky_module.ListDataset
    loader = getattr(cls, "_load_samples")
    assert loader(str(path)) == ["first.png", "second.png"]


@pytest.mark.parametrize("count", [1, 3])
def test_reader_method_bounds_retries_independently_of_constructor(sky_module, tmp_path, count):
    # A method fixture isolates retry behavior from the separately tested
    # constructor binding. This is the actual class, not a dataset substitute.
    cls = sky_module.ListDataset
    data = cls.__new__(cls)
    data.samples = [str(tmp_path / f"missing-{index}.png") for index in range(count)]
    data.transform = None
    data.return_path = False
    data.loader = sky_module.default_loader
    attempts = []

    data.loader = bounded_loader(data, attempts)
    with pytest.raises(RuntimeError, match="No readable image found"):
        data[0]
    assert len(attempts) == count


def test_reader_method_rejects_corrupt_files_without_constructor(sky_module, tmp_path):
    cls = sky_module.ListDataset
    data = cls.__new__(cls)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    data.samples = [str(bad)]
    data.loader = sky_module.default_loader
    data.transform = None
    data.return_path = False
    data.loader = bounded_loader(data, [])
    with pytest.raises(RuntimeError, match="No readable image found"):
        data[0]


def test_reader_method_valid_image_control_without_constructor(sky_module, tmp_path):
    cls = sky_module.ListDataset
    data = cls.__new__(cls)
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 3)).save(valid)
    data.samples = [str(valid)]
    data.loader = sky_module.default_loader
    data.transform = None
    data.return_path = True
    assert data[0][1] == str(valid)
