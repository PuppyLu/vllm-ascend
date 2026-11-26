# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_adapter.py"
spec = importlib.util.spec_from_file_location("glm5next_mega_kernel_adapter_cpu_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
UnsupportedDecode = module.UnsupportedDecode
prepare_decode_inputs = module.prepare_decode_inputs
compute_model_routes = module.compute_model_routes


class DecodeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.conv_slots = torch.tensor([1, 3], dtype=torch.int32)
        self.recurrent_slots = torch.tensor([2, 0], dtype=torch.int32)
        self.meta = NS(
            num_prefills=0,
            num_spec_decodes=0,
            num_decodes=2,
            num_actual_tokens=2,
            num_decode_tokens=2,
            non_spec_query_start_loc=torch.tensor([0, 1, 2]),
            non_spec_state_indices_tensor=self.recurrent_slots,
            non_spec_decode_metadata=NS(causal_conv1d=NS(cache_indices=self.conv_slots)),
            pcp_context=None,
        )
        attention = NS(
            prefix="layer.kda",
            pcp_size=1,
            projection_tp=None,
            use_recoverssm=False,
            local_num_heads=1,
            kv_cache=(torch.zeros(4, 3, 384, dtype=torch.bfloat16), torch.zeros(4, 1, 128, 128)),
            _conv_state_dim_first=False,
        )
        self.layer = NS(
            layer_kind="kda",
            _mlp_is_moe=True,
            mhc=True,
            is_mtp_layer=False,
            is_sequence_parallel=False,
            layer_idx=4,
            num_hidden_layers=45,
            self_attn=attention,
            mlp=NS(enable_eplb=False),
        )
        self.context = NS(attn_metadata={"layer.kda": self.meta})
        self.hidden = torch.zeros(2, 8)
        self.residual = torch.zeros(2, 4, 8)
        self.post = torch.zeros(2, 4, 1)
        self.comb = torch.zeros(2, 4, 4)

    def prepare(self, *, evidence=True, mode="eager"):
        return prepare_decode_inputs(
            self.layer,
            self.context,
            self.hidden,
            self.residual,
            self.post,
            self.comb,
            execution_mode=mode,
            scheduler_pure_decode=evidence,
        )

    def test_real_layer_replay_protocol_in_isolated_cpu_process(self):
        worker = Path(__file__).with_name("mega_kernel_layer_bench_fake_worker.py")
        result = subprocess.run([sys.executable, str(worker)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_slot_columns_and_single_token_template(self):
        result = self.prepare()
        torch.testing.assert_close(result.state_indices, torch.tensor([[1, 2], [3, 0]], dtype=torch.int32))
        torch.testing.assert_close(result.cu_seqlens, torch.tensor([0, 1], dtype=torch.int32))
        self.assertIs(result.conv_state, self.layer.self_attn.kv_cache[0])
        self.assertIs(result.recurrent_state, self.layer.self_attn.kv_cache[1])

    def test_reject_unproven_prefill_mixed_and_graph(self):
        with self.assertRaises(UnsupportedDecode):
            self.prepare(evidence=False)
        with self.assertRaises(UnsupportedDecode):
            self.prepare(mode="graph")
        self.meta.num_prefills = 1
        with self.assertRaises(UnsupportedDecode):
            self.prepare()

    def test_first_layer_accepts_absent_prior_mhc_state(self):
        self.layer.layer_idx = 0
        result = prepare_decode_inputs(
            self.layer, self.context, self.hidden, None, None, None, execution_mode="eager", scheduler_pure_decode=True
        )
        self.assertEqual(result.rows, 2)
        with self.assertRaises(UnsupportedDecode):
            prepare_decode_inputs(
                self.layer,
                self.context,
                self.hidden,
                self.residual,
                None,
                None,
                execution_mode="eager",
                scheduler_pure_decode=True,
            )

    def test_transposed_convolution_state_retains_writeback_view(self):
        original = torch.arange(4 * 384 * 3).reshape(4, 384, 3).to(torch.bfloat16)
        self.layer.self_attn.kv_cache = (original, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()
        self.assertTrue(decoded.conv_state.is_contiguous())
        self.assertIsNotNone(decoded.conv_writeback)
        decoded.conv_state.add_(5)
        decoded.conv_writeback.copy_(decoded.conv_state)
        torch.testing.assert_close(original.transpose(-1, -2), decoded.conv_state)

    def test_temporary_cache_copies_only_the_selected_state_slots(self):
        conv = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        rec = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        slots = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        tmp_conv, tmp_rec = module._copy_live_kda_slots((conv, rec), slots)
        torch.testing.assert_close(tmp_conv[[1, 3]], conv[[1, 3]])
        torch.testing.assert_close(tmp_rec[[2, 0]], rec[[2, 0]])
        tmp_conv[1].zero_()
        self.assertNotEqual(conv[1].count_nonzero().item(), 0)

    def test_original_router_receives_ffn_input_and_model_states_stay_unchanged(self):
        class Attention:
            def __init__(self):
                self.kv_cache = (torch.zeros(2, 3), torch.zeros(2, 3))

            def __call__(self, *, hidden_states, positions):
                self.kv_cache[0].add_(2)
                self.kv_cache[1].add_(3)
                return hidden_states + 1

        attn = Attention()

        def hc(hidden, residual, post, comb, *args, **kwargs):
            return residual, post, comb, hidden + 1

        def select(hidden, logits, enable_force_load_balance):
            self.assertFalse(enable_force_load_balance)
            torch.testing.assert_close(logits, hidden + 10)
            return torch.full((2, 8), 0.25), torch.arange(8).repeat(2, 1)

        runner = NS(
            is_internal_router=True,
            _compute_router_logits=lambda hidden, ignored: hidden + 10,
            routed_experts=NS(_select_experts=select),
        )
        layer = NS(
            layer_idx=4,
            self_attn=attn,
            hc_post_pre=hc,
            hc_attn_fn=None,
            hc_attn_scale=None,
            hc_attn_base=None,
            hc_ffn_fn=None,
            hc_ffn_scale=None,
            hc_ffn_base=None,
            input_layernorm=NS(weight=torch.ones(1), variance_epsilon=1e-6),
            post_attention_layernorm=NS(weight=torch.ones(1), variance_epsilon=1e-6),
            mlp=NS(experts=runner),
        )
        fake = ModuleType("vllm.distributed.kv_transfer")
        fake.has_kv_transfer_group = lambda: False
        orig_cache = attn.kv_cache
        with patch.dict(sys.modules, {"vllm.distributed.kv_transfer": fake}):
            ids, weights = compute_model_routes(layer, None, self.hidden, self.residual, self.post, self.comb)
        self.assertIs(attn.kv_cache, orig_cache)
        self.assertEqual(attn.kv_cache[0].count_nonzero().item(), 0)
        self.assertEqual(attn.kv_cache[1].count_nonzero().item(), 0)
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(weights.dtype, torch.float32)

    def test_internal_router_contract_rejects_incompatible_model(self):
        bias = torch.zeros(288)
        router = type("AscendFusedTopKRouter", (), {})()
        router.top_k = 8
        router.scoring_func = "sigmoid"
        router.renormalize = True
        router.routed_scaling_factor = 2.5
        router.num_expert_group = 1
        router.topk_group = 1
        router.e_score_correction_bias = bias
        routed = NS(router=router, log2phy=None, mix_placement=False)
        layer = NS(mlp=NS(experts=NS(routed_experts=routed), gate=NS(e_score_correction_bias=bias)))
        module._validate_internal_router(layer, router)
        for obj, field, value in (
            (router, "top_k", 4),
            (router, "num_expert_group", 2),
            (router, "routed_scaling_factor", 1.0),
            (router, "eplb_state", object()),
            (router, "custom_routing_function", lambda: None),
            (router, "e_score_correction_bias", torch.zeros(288)),
            (routed, "log2phy", torch.arange(288)),
            (routed, "mix_placement", True),
        ):
            previous = getattr(obj, field, None)
            setattr(obj, field, value)
            with self.assertRaises(UnsupportedDecode, msg=field):
                module._validate_internal_router(layer, router)
            setattr(obj, field, previous)
        grouped = type("AscendGroupedTopKRouter", (), {})()
        grouped.__dict__.update(router.__dict__)
        grouped.use_grouped_topk = False
        with self.assertRaises(UnsupportedDecode):
            module._validate_internal_router(layer, grouped)
        grouped.use_grouped_topk = True
        module._validate_internal_router(layer, grouped)

    def test_internal_routing_skips_reference_prefix_and_passes_no_external_routes(self):
        names = (
            "x",
            "post_h_out",
            "post_h_post",
            "post_h_res",
            "conv_state",
            "recurrent_state",
            "cu_seqlens",
            "state_indices",
            "external_route_ids",
            "external_route_weights",
            "static",
        )
        projection = ModuleType("mc2_integration.projection_contract")
        projection.MC2_INPUT_NAMES = names
        bridge = ModuleType("mc2_integration.torch_acl_bridge")
        bridge._check_tensor = lambda *a: None
        bridge.aligned_copy = lambda value: value.clone()
        bridge.aligned_empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
        collectives = ModuleType("cross_rank_prototype.staged_collectives")
        collectives._agree = lambda error, *a: None
        seen = {}

        def forward(inputs, attrs, identity):
            seen.update(inputs)
            return (self.hidden, self.residual, self.post, self.comb)

        session = NS(forward=forward, config=NS(control_device="cpu", control_group=None))
        backend = module.MegaKernelModelBackend(
            session, {"static": 1}, {}, lambda *a: "identity", routing_mode="internal"
        )
        decode = self.prepare()
        fake_npu = NS(current_stream=lambda device: NS(synchronize=lambda: None))
        modules = {
            "mc2_integration.projection_contract": projection,
            "mc2_integration.torch_acl_bridge": bridge,
            "cross_rank_prototype.staged_collectives": collectives,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(module, "compute_model_routes", side_effect=AssertionError("repeated prefix")),
            patch.object(module, "_copy_live_kda_slots", side_effect=AssertionError("temporary cache")),
        ):
            result = backend(self.layer, None, self.hidden, self.residual, self.post, self.comb, decode)
        self.assertEqual(len(result), 4)
        self.assertIsNone(seen["external_route_ids"])
        self.assertIsNone(seen["external_route_weights"])

    def test_reject_nonuniform_tokens_and_duplicate_slots(self):
        self.meta.non_spec_query_start_loc = torch.tensor([0, 0, 2])
        with self.assertRaises(UnsupportedDecode):
            self.prepare()
        self.meta.non_spec_query_start_loc = torch.tensor([0, 1, 2])
        self.conv_slots[1] = self.conv_slots[0]
        with self.assertRaises(UnsupportedDecode):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
