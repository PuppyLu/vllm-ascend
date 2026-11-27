# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
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

    def test_benchmark_consumer_has_no_host_read(self):
        calls = []
        fake_npu = NS(synchronize=lambda: calls.append("sync"))
        orig_cpu = torch.Tensor.cpu

        def observe(tensor, *args, **kwargs):
            calls.append("d2h")
            return orig_cpu(tensor, *args, **kwargs)

        with patch.object(torch, "npu", fake_npu, create=True), patch.object(torch.Tensor, "cpu", observe):
            module.benchmark_consumer_ready((self.hidden, None, self.post))
        self.assertEqual(calls, ["sync"])

    def test_stale_mode_options_are_rejected(self):
        module.validate_model_path_config({})
        for config in ({"routing_mode": "internal"}, {"routing_mode": "external"}, {"execution_mode": "fast"}):
            with self.assertRaisesRegex(ValueError, "remove obsolete"):
                module.validate_model_path_config(config)

    def test_real_layer_replay_protocol_in_isolated_cpu_process(self):
        worker = Path(__file__).with_name("mega_kernel_layer_bench_fake_worker.py")
        result = subprocess.run([sys.executable, str(worker)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_group_preflight_in_two_gloo_processes(self):
        worker = Path(__file__).with_name("mega_kernel_coordination_gloo_worker.py")
        result = subprocess.run(
            [sys.executable, str(worker)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_slot_columns_and_single_token_template(self):
        result = self.prepare()
        torch.testing.assert_close(result.state_indices, torch.tensor([[1, 2], [3, 0]], dtype=torch.int32))
        torch.testing.assert_close(result.cu_seqlens, torch.tensor([0, 1], dtype=torch.int32))
        self.assertIs(result.conv_state, self.layer.self_attn.kv_cache[0])
        self.assertIs(result.recurrent_state, self.layer.self_attn.kv_cache[1])

    def test_integer_query_and_slot_values_share_one_host_read(self):
        original_cpu = torch.Tensor.cpu
        transfers = []

        def read_to_cpu(tensor, *args, **kwargs):
            transfers.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", read_to_cpu):
            result = self.prepare()
        self.assertEqual(transfers, [(7,)])
        torch.testing.assert_close(result.state_indices, torch.tensor([[1, 2], [3, 0]], dtype=torch.int32))

    def test_noninteger_query_keeps_original_exact_check(self):
        self.meta.non_spec_query_start_loc = torch.tensor([0.0, 1.5, 2.0])
        with self.assertRaises(UnsupportedDecode):
            self.prepare()

    def test_aligned_mhc_views_are_disjoint_and_survive_parent_release(self):
        def aligned_empty(shape, dtype, device):
            size = 1
            for dim in shape:
                size *= dim
            element = torch.empty((), dtype=dtype).element_size()
            storage = torch.empty(size + 1024 // element, dtype=dtype, device=device)
            offset = (-storage.data_ptr()) % 512
            return storage[offset // element : offset // element + size].view(shape)

        for rows in (1, 3, 7, 32, 65):
            post, comb = module._aligned_mhc_buffers(rows, torch.float32, "cpu", aligned_empty)
            gc.collect()
            self.assertEqual(comb.data_ptr() - post.data_ptr(), rows * 512)
            for buffer in (post, comb):
                self.assertTrue(buffer.is_contiguous())
                self.assertFalse(buffer.requires_grad)
                self.assertEqual(buffer.data_ptr() % 512, 0)
                available = buffer.untyped_storage().nbytes() - buffer.storage_offset() * buffer.element_size()
                required = ((buffer.numel() * buffer.element_size() + 31) // 32) * 32
                self.assertGreaterEqual(available, required)
                self.assertEqual(buffer.count_nonzero().item(), 0)
            post.fill_(1)
            self.assertEqual(comb.count_nonzero().item(), 0)
            comb.fill_(2)
            self.assertTrue(torch.all(post == 1))

    def test_mhc_buffer_cache_reuses_only_matching_geometry_and_clears_padding(self):
        allocations = []

        def aligned_empty(shape, dtype, device):
            allocations.append(shape)
            element = torch.empty((), dtype=dtype).element_size()
            count = shape[0] * shape[1] * shape[2]
            storage = torch.empty(count + 1024 // element, dtype=dtype, device=device)
            offset = (-storage.data_ptr()) % 512 // element
            return storage[offset : offset + count].view(shape)

        cache = module._MhcBufferCache()
        post, comb = cache.get(2, torch.float32, torch.device("cpu"), aligned_empty)
        post.fill_(1)
        comb.fill_(2)
        new_post, new_comb = cache.get(2, torch.float32, torch.device("cpu"), aligned_empty)
        self.assertEqual((post.data_ptr(), comb.data_ptr()), (new_post.data_ptr(), new_comb.data_ptr()))
        self.assertEqual(int(new_post.count_nonzero()), 0)
        self.assertEqual(int(new_comb.count_nonzero()), 0)
        cache.get(3, torch.float32, torch.device("cpu"), aligned_empty)
        self.assertEqual(allocations, [(2, 2, 128), (2, 3, 128)])

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

    def test_transposed_convolution_state_uses_live_view_for_active_slots(self):
        original = torch.arange(4 * 384 * 3).reshape(4, 384, 3).to(torch.bfloat16)
        self.layer.self_attn.kv_cache = (original, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()
        self.assertFalse(decoded.conv_state.is_contiguous())
        self.assertIsNone(decoded.conv_writeback)
        self.assertEqual(decoded.conv_state.data_ptr(), original.data_ptr())
        prior = original.clone()
        slots = decoded.state_indices[:, 0].to(torch.int64)
        updated = decoded.conv_state.index_select(0, slots) + 5
        decoded.conv_state.index_copy_(0, slots, updated)
        torch.testing.assert_close(original[[1, 3]], updated.transpose(-1, -2))
        torch.testing.assert_close(original[[0, 2]], prior[[0, 2]])

    def test_strided_recurrent_state_keeps_interleaved_other_layer_and_inactive_slots(self):
        # Physical storage has adjacent layer planes per slot, not a dense
        # pool for one layer. Writes through layer zero must not affect layer one.
        pool = torch.arange(4 * 2 * 128 * 128, dtype=torch.float32).reshape(4, 2, 1, 128, 128)
        recurrent = pool[:, 0]
        before = pool.clone()
        self.layer.self_attn.kv_cache = (self.layer.self_attn.kv_cache[0], recurrent)
        decoded = self.prepare()
        self.assertIs(decoded.recurrent_state, recurrent)
        self.assertFalse(recurrent.is_contiguous())
        slots = decoded.state_indices[:, 1].to(torch.int64)
        staged = recurrent.index_select(0, slots).contiguous()
        staged.add_(7)
        # Before publication the model pool is untouched.
        torch.testing.assert_close(pool, before)
        recurrent.index_copy_(0, slots, staged)
        torch.testing.assert_close(pool[:, 1], before[:, 1])
        torch.testing.assert_close(pool[[1, 3], 0], before[[1, 3], 0])
        torch.testing.assert_close(pool[[0, 2], 0], before[[0, 2], 0] + 7)

    def test_overlapping_recurrent_view_falls_back_before_any_write(self):
        source = torch.zeros(1, 1, 128, 128)
        recurrent = source.expand(4, 1, 128, 128)
        self.layer.self_attn.kv_cache = (self.layer.self_attn.kv_cache[0], recurrent)
        with self.assertRaisesRegex(UnsupportedDecode, "may overlap"):
            self.prepare()
        self.assertEqual(source.count_nonzero().item(), 0)

    def test_recurrent_stride_proof_accepts_transpose_and_rejects_alias(self):
        dense = torch.zeros(4, 2, 128, 128)
        self.assertTrue(module._non_overlapping_positive_strides(dense.transpose(-1, -2)))
        overlapped = torch.as_strided(dense, (4, 2, 128, 128), (0, 128 * 128, 128, 1))
        self.assertFalse(module._non_overlapping_positive_strides(overlapped))

    def test_recurrent_stride_proof_matches_session_conservative_padding_rule(self):
        # The address ranges do not overlap, but the session uses the more
        # conservative 130*128 lower bound for the next stride.
        storage = torch.empty(4 * 16640, dtype=torch.float32)
        recurrent = torch.as_strided(storage, (4, 1, 128, 128), (16638, 16638, 130, 1))
        self.layer.self_attn.kv_cache = (self.layer.self_attn.kv_cache[0], recurrent)
        with self.assertRaisesRegex(UnsupportedDecode, "may overlap"):
            self.prepare()

    def test_mamba_page_components_keep_conv_and_padding_untouched(self):
        # Reproduce create_cache_component_views: each page stores a BF16
        # convolution component followed by an FP32 recurrent component.
        conv_bytes = 3 * 384 * 2
        rec_bytes = 128 * 128 * 4
        page_bytes = conv_bytes + rec_bytes + 64
        raw = torch.full((4 * page_bytes,), 0, dtype=torch.uint8)
        conv = torch.as_strided(raw.view(torch.bfloat16), (4, 384, 3), (page_bytes // 2, 3, 1))
        recurrent = torch.as_strided(
            raw.view(torch.float32),
            (4, 1, 128, 128),
            (page_bytes // 4, 128 * 128, 128, 1),
            storage_offset=conv_bytes // 4,
        )
        self.layer.self_attn.kv_cache = (conv, recurrent)
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()
        self.assertIs(decoded.recurrent_state, recurrent)
        self.assertIsNotNone(decoded.conv_writeback)
        self.assertFalse(recurrent.is_contiguous())
        slots = decoded.state_indices[:, 1].to(torch.int64)
        staged = recurrent.index_select(0, slots).contiguous()
        staged.add_(7)
        self.assertEqual(raw.count_nonzero().item(), 0)
        recurrent.index_copy_(0, slots, staged)
        self.assertEqual(conv.count_nonzero().item(), 0)
        self.assertEqual(recurrent[[1, 3]].count_nonzero().item(), 0)
        self.assertTrue(torch.all(recurrent[[0, 2]] == 7))
        for page in range(4):
            pad_start = page * page_bytes + conv_bytes + rec_bytes
            self.assertEqual(raw[pad_start : pad_start + 64].count_nonzero().item(), 0)

    def test_noncontiguous_physical_cache_keeps_safe_fallback(self):
        pool = torch.arange(8 * 384 * 3).reshape(8, 384, 3).to(torch.bfloat16)
        original = pool[::2]
        self.layer.self_attn.kv_cache = (original, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()
        self.assertTrue(decoded.conv_state.is_contiguous())
        self.assertIsNotNone(decoded.conv_writeback)
        before = original.clone()
        slots = decoded.state_indices[:, 0].to(torch.int64)
        decoded.conv_state.index_copy_(0, slots, decoded.conv_state.index_select(0, slots) + 5)
        torch.testing.assert_close(original, before)
        decoded.conv_writeback.copy_(decoded.conv_state)
        torch.testing.assert_close(original[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(original[[1, 3]], decoded.conv_state[[1, 3]].transpose(-1, -2))

    def test_backend_commits_only_live_slots_and_keeps_group_agreement(self):
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
        bridge.aligned_empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
        collectives = ModuleType("vllm_ascend.models.glm5next.mega_kernel_collectives")
        votes = []

        def agree(error, device, group, phase):
            votes.append(phase)
            if error:
                raise error

        collectives.agree_or_abort = agree
        modules = {
            "mc2_integration.projection_contract": projection,
            "mc2_integration.torch_acl_bridge": bridge,
            "vllm_ascend.models.glm5next.mega_kernel_collectives": collectives,
        }
        syncs = []
        fake_npu = NS(current_stream=lambda device: NS(synchronize=lambda: syncs.append("sync")))

        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                physical = torch.zeros(8 if fallback else 4, 384, 3, dtype=torch.bfloat16)
                conv = physical[::2] if fallback else physical
                recurrent = torch.zeros(4, 1, 128, 128)
                self.layer.self_attn.kv_cache = (conv, recurrent)
                self.layer.self_attn._conv_state_dim_first = True
                decoded = self.prepare()
                self.assertEqual(decoded.conv_writeback is not None, fallback)
                votes.clear()
                syncs.clear()

                def forward(inputs, attrs, identity, *, validated_step, fallback=fallback, conv=conv):
                    self.assertTrue(validated_step)
                    for column, name, increment in ((0, "conv_state", 3), (1, "recurrent_state", 7)):
                        state = inputs[name]
                        slots = inputs["state_indices"][:, column].to(torch.int64)
                        state.index_copy_(0, slots, state.index_select(0, slots) + increment)
                    if fallback:
                        self.assertEqual(conv.count_nonzero().item(), 0)
                    return (self.hidden, self.residual, self.post, self.comb)

                session = NS(forward=forward, config=NS(control_device="cpu", control_group=None))
                backend = module.MegaKernelModelBackend(session, {"static": 1}, {}, lambda *a: "identity")
                with patch.dict(sys.modules, modules), patch.object(torch, "npu", fake_npu, create=True):
                    backend(self.layer, None, self.hidden, self.residual, self.post, self.comb, decoded)
                self.assertTrue(torch.all(conv[[1, 3]] == 3))
                self.assertEqual(conv[[0, 2]].count_nonzero().item(), 0)
                self.assertTrue(torch.all(recurrent[[0, 2]] == 7))
                self.assertEqual(recurrent[[1, 3]].count_nonzero().item(), 0)
                self.assertEqual(votes, ["model route preparation", "model convolution state writeback"])
                self.assertEqual(syncs, ["sync"] if fallback else [])

    def test_backend_failure_does_not_publish_staged_cache(self):
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
        bridge.aligned_empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
        collectives = ModuleType("vllm_ascend.models.glm5next.mega_kernel_collectives")
        collectives.agree_or_abort = lambda error, *args: None
        modules = {
            "mc2_integration.projection_contract": projection,
            "mc2_integration.torch_acl_bridge": bridge,
            "vllm_ascend.models.glm5next.mega_kernel_collectives": collectives,
        }
        physical = torch.zeros(8, 384, 3, dtype=torch.bfloat16)[::2]
        self.layer.self_attn.kv_cache = (physical, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()

        def fail(inputs, attrs, identity, *, validated_step):
            inputs["conv_state"].fill_(9)
            raise RuntimeError("distributed launch failed")

        session = NS(forward=fail, config=NS(control_device="cpu", control_group=None))
        backend = module.MegaKernelModelBackend(session, {"static": 1}, {}, lambda *a: "identity")
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "distributed launch failed"):
            backend(self.layer, None, self.hidden, self.residual, self.post, self.comb, decoded)
        self.assertEqual(physical.count_nonzero().item(), 0)

    def test_temporary_cache_copies_only_the_selected_state_slots(self):
        conv = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        rec = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        slots = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        tmp_conv, tmp_rec = module._copy_live_kda_slots((conv, rec), slots)
        torch.testing.assert_close(tmp_conv[[1, 3]], conv[[1, 3]])
        torch.testing.assert_close(tmp_rec[[2, 0]], rec[[2, 0]])
        tmp_conv[1].zero_()
        self.assertNotEqual(conv[1].count_nonzero().item(), 0)

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

    def test_internal_routing_passes_validated_inputs_without_replaying_prefix(self):
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
        collectives = ModuleType("vllm_ascend.models.glm5next.mega_kernel_collectives")
        votes = []
        collectives.agree_or_abort = lambda error, *a: votes.append(a[-1])
        seen = {}
        validated = []

        def forward(inputs, attrs, identity, *, validated_step):
            seen.update(inputs)
            validated.append(validated_step)
            return (self.hidden, self.residual, self.post, self.comb)

        session = NS(forward=forward, config=NS(control_device="cpu", control_group=None))
        backend = module.MegaKernelModelBackend(session, {"static": 1}, {}, lambda *a: "identity")
        decode = self.prepare()
        fake_npu = NS(current_stream=lambda device: NS(synchronize=lambda: None))
        modules = {
            "mc2_integration.projection_contract": projection,
            "mc2_integration.torch_acl_bridge": bridge,
            "vllm_ascend.models.glm5next.mega_kernel_collectives": collectives,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(module, "_copy_live_kda_slots", side_effect=AssertionError("repeated prefix")),
        ):
            result = backend(self.layer, None, self.hidden, self.residual, self.post, self.comb, decode)
        self.assertEqual(len(result), 4)
        self.assertEqual(validated, [True])
        self.assertIsNone(seen["external_route_ids"])
        self.assertIsNone(seen["external_route_weights"])
        self.assertIs(seen["x"], self.residual)
        self.assertIs(seen["post_h_out"], self.hidden)
        self.assertIs(seen["state_indices"], decode.state_indices)
        self.assertIs(seen["cu_seqlens"], decode.cu_seqlens)
        self.assertEqual(votes, ["model route preparation", "model convolution state writeback"])
        post_pad, comb_pad = seen["post_h_post"], seen["post_h_res"]
        self.assertEqual(comb_pad.data_ptr() - post_pad.data_ptr(), 2 * 128 * 4)
        torch.testing.assert_close(post_pad[:, :4], self.post.reshape(2, 4))
        torch.testing.assert_close(comb_pad[:, :16], self.comb.reshape(2, 16))
        self.assertEqual(post_pad[:, 4:].count_nonzero().item(), 0)
        self.assertEqual(comb_pad[:, 16:].count_nonzero().item(), 0)

    def test_reject_nonuniform_tokens_and_duplicate_slots(self):
        self.meta.non_spec_query_start_loc = torch.tensor([0, 0, 2])
        with self.assertRaises(UnsupportedDecode):
            self.prepare()
        self.meta.non_spec_query_start_loc = torch.tensor([0, 1, 2])
        self.conv_slots[1] = self.conv_slots[0]
        with self.assertRaises(UnsupportedDecode):
            self.prepare()

    def test_slot_validation_preserves_int64_until_range_check(self):
        self.meta.non_spec_state_indices_tensor = torch.tensor([2, 2**32], dtype=torch.int64)
        with self.assertRaises(UnsupportedDecode):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
