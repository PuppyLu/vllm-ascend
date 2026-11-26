# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_weights.py"
spec = importlib.util.spec_from_file_location("mega_kernel_weights_cpu_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WeightTests(unittest.TestCase):
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
