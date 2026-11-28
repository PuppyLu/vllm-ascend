# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_weights.py"
spec = importlib.util.spec_from_file_location("mega_kernel_weights_cpu_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WeightTests(unittest.TestCase):
    def test_native_expert_shares_storage_shape_stride_and_values(self):
        for dtype in (torch.int8, torch.bfloat16):
            for shape in ((2, 4, 4), (2, 3, 4)):
                with self.subTest(dtype=dtype, shape=shape):
                    native = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
                    layer = SimpleNamespace(weight=native)
                    view = module._native_expert_weight(layer, "weight", shape, dtype)
                    self.assertEqual(view.data_ptr(), native.data_ptr())
                    self.assertEqual(view.untyped_storage().data_ptr(), native.untyped_storage().data_ptr())
                    self.assertEqual(tuple(view.shape), shape)
                    self.assertEqual(tuple(view.stride()), tuple(native.stride()))
                    self.assertTrue(torch.equal(view, native))
                    physical = module._validate_native_expert(view)
                    self.assertEqual(physical.data_ptr(), native.data_ptr())
                    self.assertTrue(physical.is_contiguous())
                    # The source used by prefill still has the original layout.
                    self.assertTrue(native.is_contiguous())
                    self.assertEqual(tuple(native.shape), shape)

    def test_native_expert_rejects_noncontiguous_and_wrong_contract(self):
        native = torch.arange(24, dtype=torch.int8).reshape(2, 3, 4)
        with self.assertRaisesRegex(module.WeightContractError, "contiguous"):
            module._native_expert_weight(
                SimpleNamespace(weight=native.transpose(1, 2)), "weight", (2, 4, 3), torch.int8
            )
        with self.assertRaisesRegex(module.WeightContractError, "expected"):
            module._native_expert_weight(SimpleNamespace(weight=native), "weight", native.shape, torch.bfloat16)
        with self.assertRaisesRegex(module.WeightContractError, "contiguous"):
            module._validate_native_expert(native.transpose(1, 2))
        with self.assertRaisesRegex(module.WeightContractError, "contiguous"):
            module._validate_native_expert(native[:, :, ::2])

    def test_nz_pair_is_borrowed_with_native_shape_and_no_copy(self):
        # Mock only the format query: physical NZ storage is validated by the
        # Torch C++ entry and needs real NPU tests, not these CPU tensors.
        for dtype in (torch.int8, torch.bfloat16):
            up = torch.arange(32, dtype=dtype).reshape(2, 4, 4)
            down = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
            expert = SimpleNamespace(w13_weight=up, w2_weight=down)
            copy = unittest.mock.Mock(side_effect=AssertionError("must not copy expert weight"))
            with patch.object(module, "_expert_format", return_value=29):
                mapped_up, mapped_down = module._expert_weight_pair(expert, up.shape, down.shape, dtype)
                self.assertEqual(mapped_up.data_ptr(), up.data_ptr())
                self.assertEqual(mapped_down.data_ptr(), down.data_ptr())
                self.assertEqual(tuple(mapped_down.shape), tuple(down.shape))
                self.assertEqual(tuple(mapped_down.stride()), tuple(down.stride()))
                self.assertIs(module._validate_native_expert(mapped_down), mapped_down)
                mapped = module._reuse_or_align_weights(
                    {"batch_expert_weight": mapped_up, "batch_expert_down_weight": mapped_down},
                    "cpu",
                    lambda value, device, **kwargs: None,
                    copy,
                )
            self.assertEqual(mapped["batch_expert_weight"].data_ptr(), up.data_ptr())
            self.assertEqual(mapped["batch_expert_down_weight"].data_ptr(), down.data_ptr())
            copy.assert_not_called()

    def test_expert_pair_rejects_mixed_formats_and_unsupported_nz_dtype(self):
        up = torch.zeros((2, 4, 4), dtype=torch.int8)
        down = torch.zeros((2, 3, 4), dtype=torch.int8)
        expert = SimpleNamespace(w13_weight=up, w2_weight=down)
        with (
            patch.object(
                module, "_expert_format", side_effect=lambda value: 29 if value.data_ptr() == up.data_ptr() else 2
            ),
            self.assertRaisesRegex(module.WeightContractError, "both be ND or both be"),
        ):
            module._expert_weight_pair(expert, up.shape, down.shape, torch.int8)
        invalid = torch.zeros((2, 3, 4), dtype=torch.float32)
        with (
            patch.object(module, "_expert_format", return_value=29),
            self.assertRaisesRegex(module.WeightContractError, "unsupported expert storage format"),
        ):
            module._native_expert_weight(SimpleNamespace(weight=invalid), "weight", invalid.shape, torch.float32)
        with (
            patch.object(module, "_expert_format", return_value=30),
            self.assertRaisesRegex(module.WeightContractError, "unsupported expert storage format"),
        ):
            module._native_expert_weight(SimpleNamespace(weight=down), "weight", down.shape, torch.int8)

    def test_bootstrap_reuses_expert_storage_or_rejects_it_without_copy(self):
        source = torch.arange(24, dtype=torch.int8).reshape(2, 3, 4)
        view = module._native_expert_weight(SimpleNamespace(weight=source), "weight", source.shape, source.dtype)
        copy = unittest.mock.Mock(side_effect=AssertionError("must not copy expert weight"))
        checked = []

        def check(value, device, *, allow_expert_view=False):
            self.assertEqual(device, "cpu")
            self.assertTrue(allow_expert_view)
            self.assertTrue(value.is_contiguous())
            self.assertEqual(tuple(value.shape), (2, 3, 4))
            checked.append(value.data_ptr())

        mapped = module._reuse_or_align_weights({"batch_expert_weight": view}, "cpu", check, copy)
        self.assertIs(mapped["batch_expert_weight"], view)
        self.assertEqual(checked, [source.data_ptr()])
        copy.assert_not_called()

        def reject(value, device, *, allow_expert_view=False):
            raise ValueError("insufficient alignment")

        with self.assertRaisesRegex(ValueError, "insufficient alignment"):
            module._reuse_or_align_weights({"batch_expert_weight": view}, "cpu", reject, copy)
        copy.assert_not_called()

        with self.assertRaisesRegex(ValueError, "overlaps"):
            module._reuse_or_align_weights(
                {"batch_expert_weight": view, "batch_expert_down_weight": view}, "cpu", check, copy
            )
        copy.assert_not_called()

    def test_dynamic_projection_matches_postload_layout_and_scale(self):
        scheme_type = type(
            "AscendW8A8DynamicLinearMethod",
            (),
            {
                "__module__": "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
                "act_quant_type": torch.int8,
            },
        )
        layer = SimpleNamespace(
            weight=torch.arange(12, dtype=torch.int8).reshape(3, 4),
            bias=None,
            quant_method=SimpleNamespace(quant_method=scheme_type()),
            weight_scale=torch.tensor([0.5, 0.25, 1.0, 2.0], dtype=torch.bfloat16),
            weight_offset=torch.ones(4),
        )
        weights, scale = module._projection_pair(layer, 4, 3, allow_nz=False, projection=True)
        self.assertEqual(tuple(weights.shape), (4, 3))
        self.assertEqual(int(weights[1, 2]), 9)
        self.assertEqual(scale.dtype, torch.bfloat16)
        # The pinned vLLM W8A8_DYNAMIC forward does not use weight_offset.
        # This checks parity with that code, not asymmetric quantization support.
        self.assertTrue(module._w8a8_code_is_verified())
        layer.weight_scale = torch.tensor([0.1, 0.25, 1.0, 2.0], dtype=torch.float32)
        with self.assertRaises(module.WeightContractError):
            module._projection_pair(layer, 4, 3, allow_nz=False, projection=True)

    def test_bf16_and_unknown_int8(self):
        layer = SimpleNamespace(weight=torch.ones(4, 3, dtype=torch.bfloat16), bias=None)
        weight, scale = module._projection_pair(layer, 4, 3, allow_nz=False)
        self.assertEqual(tuple(weight.shape), (4, 3))
        self.assertIsNone(scale)
        layer.weight = torch.ones(3, 4, dtype=torch.int8)
        with self.assertRaises(module.WeightContractError):
            module._projection_pair(layer, 4, 3, allow_nz=False)


if __name__ == "__main__":
    unittest.main()
