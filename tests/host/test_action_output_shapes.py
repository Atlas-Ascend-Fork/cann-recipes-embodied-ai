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
"""Exercise complete comparison functions with real CPU PyTorch tensors."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture(params=["pi05", "xvla"])
def comparison(request, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    kind = request.param
    source = repository / "manipulation" / kind / "infer_with_torch" / f"verify_{kind}_accuracy_ascend.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location(f"actual_{kind}_accuracy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == source
    return kind, module.compare_actions


def compare(comparison, reference, actual):
    kind, function = comparison
    return function(reference, actual, 3 if kind == "pi05" else 0.99)


@pytest.mark.parametrize("shape", [(3, 4), (1, 1, 3, 4), (1, 4, 3), (1, 3, 5)])
def test_different_output_shapes_return_false(comparison, shape):
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)
    if math.prod(shape) == reference.numel():
        actual = reference.reshape(shape)
    else:
        actual = torch.ones(shape, dtype=torch.float32)
    assert compare(comparison, reference, actual) is False


def test_equal_outputs_keep_success(comparison):
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)
    assert compare(comparison, reference, reference.clone()) is True


def test_opposite_outputs_keep_failure(comparison):
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)
    assert compare(comparison, reference, -reference) is False


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_existing_nonfinite_rejection_is_preserved(comparison, value):
    reference = torch.ones((1, 3, 4), dtype=torch.float32)
    actual = reference.clone()
    actual[0, 0, 0] = value
    assert compare(comparison, reference, actual) is False
