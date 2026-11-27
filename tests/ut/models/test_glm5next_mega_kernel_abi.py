# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU contract tests; actual device allocation still requires A3 coverage."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_abi.py"
spec = importlib.util.spec_from_file_location("mega_kernel_abi_test", path)
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)


class MegaKernelAbiTest(unittest.TestCase):
    def test_ordered_contract_has_expected_boundaries(self):
        # Freeze the reviewed operator argument order, not just its length.
        expected_inputs = [
            "x",
            "hc_fn",
            "hc_scale",
            "hc_base",
            "post_h_res",
            "post_h_out",
            "post_h_post",
            "norm_weight",
            "projection_weight",
            "conv_weight",
            "conv_state",
            "raw_gate",
            "recurrent_state",
            "cu_seqlens",
            "state_indices",
            "a_log",
            "dt_bias",
            "g_b_weight",
            "o_norm_weight",
            "o_weight",
            "f_b_weight",
            "ffn_c0_diagnostic",
            "ffn_hc_fn",
            "ffn_hc_scale",
            "ffn_hc_base",
            "ffn_norm_weight",
            "router_weight",
            "router_bias",
            "expert_weight",
            "expert_channel_scale",
            "expert_down_weight",
            "expert_down_channel_scale",
            "top8_rows",
            "top8_expert_ids",
            "top8_permutation",
            "batch_expert_weight",
            "batch_expert_channel_scale",
            "batch_expert_down_weight",
            "batch_expert_down_channel_scale",
            "o_quant_weight",
            "o_quant_channel_scale",
            "in_projection_scale",
            "g_projection_scale",
            "o_projection_scale",
            "f_projection_scale",
            "external_route_ids",
            "external_route_weights",
        ]
        expected_attrs = [
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "norm_eps",
            "layer_rms_eps",
            "stage_limit",
            "conv_slot",
            "lower_bound",
            "output_norm_eps",
            "ffn_hc_eps",
            "ffn_norm_eps",
            "ffn_layer_rms_eps",
            "mhc_post_mult_value",
            "router_mode",
            "expert_id",
            "route_ordinal",
            "down_expert_id",
            "ep_world_size",
        ]
        self.assertEqual(abi.MC2_INPUT_NAMES, tuple(expected_inputs))
        self.assertEqual(abi.ATTR_NAMES, tuple(expected_attrs))

    def test_allocations_have_alignment_and_tail(self):
        for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int8):
            with self.subTest(dtype=dtype):
                tensor = abi.aligned_empty((3, 7), dtype, "cpu")
                storage = tensor.untyped_storage()
                end = tensor.data_ptr() - storage.data_ptr() + tensor.numel() * tensor.element_size()
                self.assertEqual(tensor.data_ptr() % 512, 0)
                self.assertGreaterEqual(storage.nbytes() - end, 512)
                self.assertTrue(tensor.is_contiguous())

    def test_copy_does_not_alias_strided_source(self):
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        copy = abi.aligned_copy(source)
        torch.testing.assert_close(copy, source)
        source.add_(1)
        self.assertFalse(torch.equal(copy, source))
        self.assertEqual(copy.data_ptr() % 512, 0)

    def test_empty_and_nonpositive_shapes_are_rejected(self):
        for shape in ((0,), (-1,), (True,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                abi.aligned_empty(shape, torch.float32, "cpu")

    def test_scalar_allocation_keeps_tail_and_can_be_copied(self):
        tensor = abi.aligned_copy(torch.tensor(3.0))
        self.assertEqual(tensor.shape, torch.Size([]))
        self.assertEqual(tensor.item(), 3.0)
        end = tensor.data_ptr() - tensor.untyped_storage().data_ptr() + tensor.element_size()
        self.assertGreaterEqual(tensor.untyped_storage().nbytes() - end, 512)

    def test_expert_partition_dtype_and_native_layout(self):
        # Meta tensors cover true model shapes without allocating expert banks.
        width, n = 64, 36
        for dtype in (torch.bfloat16, torch.int8):
            with self.subTest(dtype=dtype):
                make = lambda shape, typ: torch.empty(shape, dtype=typ, device="meta")
                values = {
                    "expert_weight": make((2 * width, 4096), dtype),
                    "expert_down_weight": make((4096, width), dtype),
                    "batch_expert_weight": make((n, 4096, 4096), dtype),
                    "batch_expert_down_weight": make((n, 2048, 4096), dtype),
                }
                if dtype == torch.int8:
                    values.update(
                        expert_channel_scale=make((2 * width,), torch.float32),
                        expert_down_channel_scale=make((4096,), torch.bfloat16),
                        batch_expert_channel_scale=make((n, 4096), torch.float32),
                        batch_expert_down_channel_scale=make((n, 4096), torch.bfloat16),
                    )
                abi.validate_expert_bank(values, 3, 8, "sharded", shared_width=width)
                values["batch_expert_down_weight"] = make((n, 4096, 2048), dtype)
                abi.validate_expert_bank(values, 3, 8, "sharded", shared_width=width)
                values["batch_expert_down_weight"] = make((n - 1, 4096, 2048), dtype)
                with self.assertRaisesRegex(ValueError, "expert tensor"):
                    abi.validate_expert_bank(values, 3, 8, "sharded", shared_width=width)

    def test_expert_format_and_scale_contracts(self):
        n, width = 36, 64
        make = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="meta")
        values = {
            "expert_weight": make((2 * width, 4096), torch.int8),
            "expert_down_weight": make((4096, width), torch.int8),
            "batch_expert_weight": make((n, 4096, 4096), torch.int8),
            "batch_expert_down_weight": make((n, 2048, 4096), torch.int8),
            "expert_channel_scale": make((2 * width,), torch.float32),
            "expert_down_channel_scale": make((4096,), torch.bfloat16),
            "batch_expert_channel_scale": make((n, 4096), torch.float32),
            "batch_expert_down_channel_scale": make((n, 4096), torch.bfloat16),
        }
        with patch.object(abi, "_is_nz", side_effect=[True, False]), self.assertRaisesRegex(ValueError, "paired"):
            abi.validate_expert_bank(values, 0, 8, "sharded", shared_width=width)
        values["batch_expert_down_channel_scale"] = make((n, 4096), torch.float32)
        with self.assertRaisesRegex(ValueError, "same BF16 or FP32"):
            abi.validate_expert_bank(values, 0, 8, "sharded", shared_width=width)


if __name__ == "__main__":
    unittest.main()
