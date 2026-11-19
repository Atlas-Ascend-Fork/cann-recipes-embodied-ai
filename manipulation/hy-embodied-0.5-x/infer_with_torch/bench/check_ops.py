# Copyright (c) 2026 li12345abc123. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

"""Explicit NPU operator checks; no model checkpoint is required."""

import importlib.util
import unittest

NPU = importlib.util.find_spec('torch_npu') is not None
if NPU:
    import torch

    importlib.import_module('torch_npu')
    from adapters.attention import flash_attn_func, flash_attn_varlen_func
    from adapters.gelu import NpuErfGELU, install_exact_gelu

    NPU = torch.npu.is_available()


@unittest.skipUnless(NPU, 'Requires an available Ascend NPU and torch_npu')
class AttentionNpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)

    def test_causal_gqa_decode_and_bidirectional(self):
        """验证右下对齐因果 mask、GQA、单 token KV 解码及视觉双向注意力。"""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype), torch.inference_mode():
                q = torch.zeros(1, 2, 4, 16, device='npu', dtype=dtype)
                k = torch.zeros(1, 5, 2, 16, device='npu', dtype=dtype)
                v = torch.arange(5, device='npu', dtype=dtype)[None, :, None, None].expand(1, 5, 2, 16).contiguous()
                expected = torch.tensor([1.5, 2.0])[None, :, None, None].expand(1, 2, 4, 16)
                torch.testing.assert_close(
                    flash_attn_func(q, k, v, causal=True).cpu().float(), expected, atol=0.02, rtol=0.02
                )
                for queries, causal in [(q[:, :1], True), (q, False)]:
                    actual = flash_attn_func(queries, k, v, causal=causal).cpu().float()
                    torch.testing.assert_close(actual, torch.full(actual.shape, 2.0), atol=0.02, rtol=0.02)

    def test_packed_segments_are_isolated(self):
        """验证拼接的变长序列仅访问各自 KV 段，段间数值保持隔离。"""
        with torch.inference_mode():
            q = torch.zeros(5, 4, 16, device='npu', dtype=torch.bfloat16)
            k = torch.zeros(5, 2, 16, device='npu', dtype=torch.bfloat16)
            v = torch.cat([torch.full((2, 2, 16), 3.0), torch.full((3, 2, 16), 7.0)]).npu().bfloat16()
            cu = torch.tensor([0, 2, 5], device='npu', dtype=torch.int32)
            expected = torch.cat([torch.full((2, 4, 16), 3.0), torch.full((3, 4, 16), 7.0)])
            actual = flash_attn_varlen_func(q, k, v, cu, cu, 3, 3, causal=True)
            torch.testing.assert_close(actual.cpu().float(), expected, atol=0.02, rtol=0.02)

    def test_attention_argument_and_empty_input_contract(self):
        """验证空 query 保留形状与设备，位置/关键字参数绑定一致且非法参数被拒绝。"""
        q = torch.empty(1, 0, 2, 16, device='npu', dtype=torch.bfloat16)
        k = torch.zeros(1, 2, 2, 16, device='npu', dtype=torch.bfloat16)
        a = flash_attn_func(q, k, k, 0.0, 0.5, False)
        b = flash_attn_func(q, k, k, softmax_scale=0.5, causal=False)
        self.assertEqual((a.shape, a.device, a.dtype), (q.shape, q.device, q.dtype))
        self.assertTrue(torch.equal(a, b))
        with self.assertRaises(NotImplementedError):
            flash_attn_func(q, k, k, unsupported=True)
        with self.assertRaises((ValueError, NotImplementedError)):
            flash_attn_func(q, k, k, dropout_p=0.5)


@unittest.skipUnless(NPU, 'Requires an available Ascend NPU and torch_npu')
class GeluNpuTests(unittest.TestCase):
    def test_erf_reference_and_dtype(self):
        """验证 FP32/FP16/BF16 和非连续输入的 erf 参考误差、形状与 dtype。"""
        original = torch.linspace(-8, 8, 8192, dtype=torch.float64).reshape(128, 64)
        for dtype, rtol, atol in [
            (torch.float32, 2e-5, 2e-6),
            (torch.float16, 1e-3, 2e-4),
            (torch.bfloat16, 8e-3, 2e-5),
        ]:
            with self.subTest(dtype=dtype):
                rounded = original.to(dtype)
                reference = rounded.double() * 0.5 * (1 + torch.erf(rounded.double() / (2**0.5)))
                value = rounded.npu().transpose(0, 1)
                actual = NpuErfGELU()(value)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, value.shape)
                self.assertTrue(bool(torch.isfinite(actual).all().item()))
                torch.testing.assert_close(actual.cpu().double(), reference.transpose(0, 1), rtol=rtol, atol=atol)

    def test_none_semantics_are_not_tanh(self):
        """验证 none 模式执行 erf 公式而非 tanh 近似。"""
        value = torch.tensor([-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0], device='npu')
        actual = NpuErfGELU()(value)
        approximate = torch.nn.functional.gelu(value, approximate='tanh')
        self.assertFalse(bool(torch.equal(actual, approximate)))
        reference = value.cpu().double()
        reference = reference * 0.5 * (1 + torch.erf(reference / (2**0.5)))
        torch.testing.assert_close(actual.cpu().double(), reference, rtol=2e-5, atol=2e-6)

    def test_instance_scope_and_idempotency(self):
        """验证 GELU 替换只作用于模型实例，重复安装保持幂等且参数值原样。"""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.GELU(approximate='tanh')).eval()
        original_state = {key: value.clone() for key, value in model.state_dict().items()}
        tanh_module, global_forward = model[2], torch.nn.GELU.forward
        replaced = install_exact_gelu(model)
        self.assertEqual(replaced, ['1'])
        self.assertIsInstance(model[1], NpuErfGELU)
        self.assertFalse(model[1].training)
        self.assertIs(model[2], tanh_module)
        self.assertIs(torch.nn.GELU.forward, global_forward)
        self.assertEqual(install_exact_gelu(model), [])
        self.assertEqual(original_state.keys(), model.state_dict().keys())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[key]))

    def test_rejects_cpu_forward_and_unrecognized_model(self):
        """验证 GELU 不在 CPU 执行，并拒绝不含预期激活模块的模型。"""
        with self.assertRaises(ValueError):
            NpuErfGELU()(torch.ones(2))
        with self.assertRaises(RuntimeError):
            install_exact_gelu(torch.nn.Sequential(torch.nn.ReLU()))


if __name__ == '__main__':
    unittest.main()
