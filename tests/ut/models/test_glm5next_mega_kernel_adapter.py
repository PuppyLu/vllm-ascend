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

def _load_adapter_module():
    glm5next_dir = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"
    if importlib.util.find_spec("vllm") is not None:
        return importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_adapter")
    modules = {}
    for pkg in ("vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.glm5next"):
        mod = ModuleType(pkg)
        mod.__path__ = [str(glm5next_dir)]
        modules[pkg] = mod
    for mod_name in ("mega_kernel_abi", "mega_kernel_collectives", "mega_kernel_adapter"):
        fq = f"vllm_ascend.models.glm5next.{mod_name}"
        spec = importlib.util.spec_from_file_location(fq, glm5next_dir / f"{mod_name}.py")
        loaded = importlib.util.module_from_spec(spec)
        modules[fq] = loaded
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(loaded)
    return modules["vllm_ascend.models.glm5next.mega_kernel_adapter"]


module = _load_adapter_module()
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

    def test_single_token_template_is_reused_for_the_same_layer(self):
        first = self.prepare()
        second = self.prepare()
        self.assertIs(first.cu_seqlens, second.cu_seqlens)
        torch.testing.assert_close(second.cu_seqlens, torch.tensor([0, 1], dtype=torch.int32))

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
        dynamic_names = {
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
        }
        static_weights = {name: 1 for name in module.MC2_INPUT_NAMES if name not in dynamic_names}
        votes = []

        def agree(error, device, group, phase):
            votes.append(phase)
            if error:
                raise error

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
                backend = module.MegaKernelModelBackend(session, static_weights, {}, lambda *a: "identity")
                with patch.object(module, "agree_or_abort", agree), patch.object(torch, "npu", fake_npu, create=True):
                    backend(self.layer, None, self.hidden, self.residual, self.post, self.comb, decoded)
                self.assertTrue(torch.all(conv[[1, 3]] == 3))
                self.assertEqual(conv[[0, 2]].count_nonzero().item(), 0)
                self.assertTrue(torch.all(recurrent[[0, 2]] == 7))
                self.assertEqual(recurrent[[1, 3]].count_nonzero().item(), 0)
                self.assertEqual(votes, ["model route preparation", "model convolution state writeback"])
                self.assertEqual(syncs, ["sync"] if fallback else [])

    def test_backend_failure_does_not_publish_staged_cache(self):
        dynamic_names = {
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
        }
        static_weights = {name: 1 for name in module.MC2_INPUT_NAMES if name not in dynamic_names}
        physical = torch.zeros(8, 384, 3, dtype=torch.bfloat16)[::2]
        self.layer.self_attn.kv_cache = (physical, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True
        decoded = self.prepare()

        def fail(inputs, attrs, identity, *, validated_step):
            inputs["conv_state"].fill_(9)
            raise RuntimeError("distributed launch failed")

        session = NS(forward=fail, config=NS(control_device="cpu", control_group=None))
        backend = module.MegaKernelModelBackend(session, static_weights, {}, lambda *a: "identity")
        with (
            patch.object(module, "agree_or_abort", lambda error, *a: None),
            self.assertRaisesRegex(RuntimeError, "distributed launch failed"),
        ):
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
        dynamic_names = {
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
        }
        static_weights = {name: 1 for name in module.MC2_INPUT_NAMES if name not in dynamic_names}
        votes = []
        seen = {}
        validated = []

        def forward(inputs, attrs, identity, *, validated_step):
            seen.update(inputs)
            validated.append(validated_step)
            return (self.hidden, self.residual, self.post, self.comb)

        session = NS(forward=forward, config=NS(control_device="cpu", control_group=None))
        backend = module.MegaKernelModelBackend(session, static_weights, {}, lambda *a: "identity")
        decode = self.prepare()
        fake_npu = NS(current_stream=lambda device: NS(synchronize=lambda: None))
        with (
            patch.object(module, "agree_or_abort", lambda error, *a: votes.append(a[-1])),
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

    def test_compact_strided_conv_skips_full_pool_copy_on_fast_path(self):
        pool = torch.arange(8 * 384 * 3, dtype=torch.bfloat16).reshape(8, 384, 3)
        strided = pool[::2]
        self.layer.self_attn.kv_cache = (strided, torch.zeros(4, 1, 128, 128))
        self.layer.self_attn._conv_state_dim_first = True

        legacy = prepare_decode_inputs(
            self.layer,
            self.context,
            self.hidden,
            self.residual,
            self.post,
            self.comb,
            execution_mode="eager",
            scheduler_pure_decode=True,
            compact_strided_conv=False,
        )
        self.assertIsNotNone(legacy.conv_writeback)

        fast = prepare_decode_inputs(
            self.layer,
            self.context,
            self.hidden,
            self.residual,
            self.post,
            self.comb,
            execution_mode="eager",
            scheduler_pure_decode=True,
            compact_strided_conv=True,
        )
        self.assertIsNone(fast.conv_writeback)
        self.assertEqual(fast.conv_state.data_ptr(), strided.data_ptr())
        self.assertIsNotNone(fast.conv_slots_i64)
        self.assertIsNotNone(fast.recurrent_slots_i64)

    def test_step_metadata_cache_collapses_34_layers_to_single_d2h_read(self):
        step_cache = {}
        transfers = []
        orig_cpu = torch.Tensor.cpu

        def track_cpu(tensor, *args, **kwargs):
            transfers.append(tuple(tensor.shape))
            return orig_cpu(tensor, *args, **kwargs)

        layers = []
        attn_map = {}
        for idx in range(34):
            prefix = f"layer.{idx}.kda"
            attn_map[prefix] = NS(
                num_prefills=0,
                num_spec_decodes=0,
                num_decodes=2,
                num_actual_tokens=2,
                num_decode_tokens=2,
                non_spec_query_start_loc=self.meta.non_spec_query_start_loc,
                non_spec_state_indices_tensor=self.recurrent_slots,
                non_spec_decode_metadata=NS(causal_conv1d=NS(cache_indices=self.conv_slots)),
                pcp_context=None,
            )
            layers.append(
                NS(
                    layer_kind="kda",
                    _mlp_is_moe=True,
                    mhc=True,
                    is_mtp_layer=False,
                    is_sequence_parallel=False,
                    layer_idx=idx,
                    num_hidden_layers=45,
                    self_attn=NS(
                        prefix=prefix,
                        pcp_size=1,
                        projection_tp=None,
                        use_recoverssm=False,
                        local_num_heads=1,
                        kv_cache=(
                            torch.zeros(4, 3, 384, dtype=torch.bfloat16),
                            torch.zeros(4, 1, 128, 128),
                        ),
                        _conv_state_dim_first=False,
                    ),
                    mlp=NS(enable_eplb=False),
                )
            )
        ctx = NS(attn_metadata=attn_map)
        with patch.object(torch.Tensor, "cpu", track_cpu):
            decoded = [
                prepare_decode_inputs(
                    layer,
                    ctx,
                    self.hidden,
                    None if layer.layer_idx == 0 else self.residual,
                    None if layer.layer_idx == 0 else self.post,
                    None if layer.layer_idx == 0 else self.comb,
                    execution_mode="eager",
                    scheduler_pure_decode=True,
                    step_cache=step_cache,
                    compact_strided_conv=True,
                )
                for layer in layers
            ]
        self.assertEqual(len(transfers), 1)
        for item in decoded[1:]:
            self.assertIs(item.state_indices, decoded[0].state_indices)
            self.assertIs(item.conv_slots_i64, decoded[0].conv_slots_i64)
            self.assertIs(item.recurrent_slots_i64, decoded[0].recurrent_slots_i64)
            self.assertIs(item.cu_seqlens, decoded[0].cu_seqlens)

    def test_no_runtime_dependency_on_ops_test_directories(self):
        glm5next_dir = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"
        for py_file in sorted(glm5next_dir.glob("mega_kernel_*.py")):
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("mc2_integration", text, msg=py_file.name)
            self.assertNotIn("cross_rank_prototype", text, msg=py_file.name)


class StepDecodeCoordinatorTests(unittest.TestCase):
    def _make_layers_and_ctx(self, count=3, *, corrupt_last_layer=False):
        conv_slots = torch.tensor([1, 3], dtype=torch.int32)
        rec_slots = torch.tensor([2, 0], dtype=torch.int32)
        query = torch.tensor([0, 1, 2], dtype=torch.int32)
        bias = torch.zeros(288)
        router = type("AscendFusedTopKRouter", (), {})()
        router.top_k = 8
        router.scoring_func = "sigmoid"
        router.renormalize = True
        router.routed_scaling_factor = 2.5
        router.num_expert_group = 1
        router.topk_group = 1
        router.e_score_correction_bias = bias
        routed = NS(
            router=router,
            log2phy=None,
            mix_placement=False,
            _select_experts=lambda *a: None,
            moe_config=NS(swiglu_limit=10.0),
        )
        attn_map = {}
        layers = []
        for idx in range(count):
            prefix = f"layer.{idx}.kda"
            attn_map[prefix] = NS(
                num_prefills=0,
                num_spec_decodes=0,
                num_decodes=2,
                num_actual_tokens=2,
                num_decode_tokens=2,
                non_spec_query_start_loc=query,
                non_spec_state_indices_tensor=rec_slots,
                non_spec_decode_metadata=NS(causal_conv1d=NS(cache_indices=conv_slots)),
                pcp_context=None,
            )
            num_heads = 2 if (corrupt_last_layer and idx == count - 1) else 1
            layer = NS(
                layer_kind="kda",
                _mlp_is_moe=True,
                mhc=True,
                is_mtp_layer=False,
                is_sequence_parallel=False,
                layer_idx=idx,
                num_hidden_layers=45,
                self_attn=NS(
                    prefix=prefix,
                    pcp_size=1,
                    projection_tp=None,
                    use_recoverssm=False,
                    local_num_heads=num_heads,
                    kv_cache=(torch.zeros(4, 3, 384, dtype=torch.bfloat16), torch.zeros(4, 1, 128, 128)),
                    _conv_state_dim_first=False,
                ),
                mlp=NS(
                    enable_eplb=False,
                    routed_scaling_factor=2.5,
                    experts=NS(routed_experts=routed, is_internal_router=True, _compute_router_logits=lambda *a: None),
                    shared_experts=NS(swiglu_limit=10.0),
                    gate=NS(e_score_correction_bias=bias),
                ),
            )
            layers.append(layer)
        return layers, NS(attn_metadata=attn_map)

    def _external_stubs(self, ctx):
        fwd = ModuleType("vllm.forward_context")
        fwd.get_forward_context = lambda: ctx
        kv = ModuleType("vllm.distributed.kv_transfer")
        kv.has_kv_transfer_group = lambda: False
        asc = ModuleType("vllm_ascend.ascend_config")
        asc.get_ascend_config = lambda: NS(enable_force_eplb=False)
        kda = ModuleType("vllm_ascend.models.glm5next.kda")
        kda.maybe_save_kv_layer_to_connector = lambda *a: None
        kda.record_attention_compute_start = lambda: None
        kda.wait_for_kv_layer_from_connector = lambda *a: None
        return {
            "vllm": ModuleType("vllm"),
            "vllm.forward_context": fwd,
            "vllm.distributed": ModuleType("vllm.distributed"),
            "vllm.distributed.kv_transfer": kv,
            "vllm_ascend": ModuleType("vllm_ascend"),
            "vllm_ascend.ascend_config": asc,
            "vllm_ascend.models": ModuleType("vllm_ascend.models"),
            "vllm_ascend.models.glm5next": ModuleType("vllm_ascend.models.glm5next"),
            "vllm_ascend.models.glm5next.kda": kda,
        }

    def test_step_coordinator_single_preflight_fallback_and_midstep_abort(self):
        layers, ctx = self._make_layers_and_ctx(count=3, corrupt_last_layer=False)
        preflight_calls = []
        finalize_calls = []
        abort_calls = []
        ensure_calls = []
        forward_calls = []

        class FakeGate:
            def __init__(self):
                self.cache = {}
                self.verdict = None
                self.cleanups = []

            def current_step_cache(self):
                return self.cache

            def get_step_preflight(self):
                return self.verdict

            def set_step_preflight(self, v):
                self.verdict = v

            def register_step_cleanup(self, cb):
                if cb not in self.cleanups:
                    self.cleanups.append(cb)

        class FakePool:
            def __init__(self):
                self._pending = []

            def get_preflight_scratch(self):
                return torch.zeros(1, dtype=torch.int32)

            def has_pending(self):
                return bool(self._pending)

            def finalize_step(self, dev, grp):
                finalize_calls.append(len(self._pending))
                self._pending.clear()

            def abort_step(self, mark_failed=True):
                abort_calls.append((mark_failed, len(self._pending)))
                self._pending.clear()

        gate = FakeGate()
        pool = FakePool()
        hidden = torch.ones(2, 4096, dtype=torch.bfloat16)
        res = torch.ones(2, 4, 4096, dtype=torch.bfloat16)
        post = torch.ones(2, 4, 1)
        comb = torch.ones(2, 4, 4)

        entries = []
        for layer in layers:
            session = NS(
                forward=lambda *a, **k: None,
                ensure_geometry=lambda rows, stream, idx=layer.layer_idx: ensure_calls.append((idx, rows)),
                forward_layer=lambda **kw: (pool._pending.append(1) or forward_calls.append(kw["is_first_layer"]) or (hidden, res, post, comb)),
            )
            backend = module.MegaKernelModelBackend(session, {"static": 1}, {"stage_limit": 28}, lambda *a: "id")
            backend.session.bridge = NS(stage_limit=28)
            entries.append((layer, backend))

        coord = module.StepDecodeCoordinator(
            gate=gate,
            pool=pool,
            control_device="cpu",
            control_group=None,
            layer_entries=entries,
            defer_state_commit=True,
        )

        def fake_preflight(err, pure, dev, grp, *, scratch=None):
            preflight_calls.append(err)
            if err is not None and not isinstance(err, UnsupportedDecode):
                raise module.MegaKernelCollectiveError("abort") from err
            return err is None and pure

        orig_calls = []
        for layer, backend in entries:
            layer.forward = lambda p, h, r=None, po=None, c=None, idx=layer.layer_idx: orig_calls.append(idx) or (h, r, po, c)
            module.DecodeLayerAdapter(
                backend,
                lambda l, e: True,
                lambda: "eager",
                lambda l: True,
                coordinator=coord,
            ).install(layer)

        stubs = self._external_stubs(ctx)
        fake_npu = NS(current_stream=lambda d: NS(npu_stream=1))
        with (
            patch.dict(sys.modules, stubs),
            patch.object(module, "coordinate_layer_preflight", fake_preflight),
            patch.object(torch, "npu", fake_npu, create=True),
        ):
            h, r, p, c = hidden, None, None, None
            for layer in layers:
                h, r, p, c = layer.forward(None, h, r, p, c)
            self.assertEqual(len(preflight_calls), 1)
            self.assertEqual(len(ensure_calls), 3)
            self.assertEqual(forward_calls, [True, False, False])
            self.assertEqual(finalize_calls, [3])

            # Early exit while TX pending triggers abort + MegaKernelCollectiveError.
            pool._pending.append(1)
            with self.assertRaises(module.MegaKernelCollectiveError):
                coord.on_step_exit(failed=False)
            self.assertEqual(abort_calls, [(True, 1)])

            # Mid-step failure after group accept raises MegaKernelCollectiveError.
            gate.verdict = None
            gate.cache.clear()
            layers[0].forward(None, hidden, None, None, None)
            layers[1].self_attn.use_recoverssm = True
            with self.assertRaises(module.MegaKernelCollectiveError):
                layers[1].forward(None, hidden, res, post, comb)
            pool._pending.clear()

        # Unsupported last layer detected during Layer-0 preflight routes ALL 3 layers to original path.
        layers2, ctx2 = self._make_layers_and_ctx(count=3, corrupt_last_layer=True)
        preflight_calls.clear()
        orig_calls.clear()
        entries2 = []
        for layer in layers2:
            session = NS(forward=lambda *a, **k: None, ensure_geometry=lambda *a: None, forward_layer=lambda **k: None)
            backend = module.MegaKernelModelBackend(session, {"static": 1}, {"stage_limit": 28}, lambda *a: "id")
            backend.session.bridge = NS(stage_limit=28)
            entries2.append((layer, backend))
        coord2 = module.StepDecodeCoordinator(
            gate=FakeGate(),
            pool=FakePool(),
            control_device="cpu",
            control_group=None,
            layer_entries=entries2,
            defer_state_commit=True,
        )
        for layer, backend in entries2:
            layer.forward = lambda p, h, r=None, po=None, c=None, idx=layer.layer_idx: orig_calls.append(idx) or (h, r, po, c)
            module.DecodeLayerAdapter(
                backend, lambda l, e: False, lambda: "eager", lambda l: True, coordinator=coord2
            ).install(layer)
        with (
            patch.dict(sys.modules, self._external_stubs(ctx2)),
            patch.object(module, "coordinate_layer_preflight", fake_preflight),
            patch.object(torch, "npu", fake_npu, create=True),
        ):
            for layer in layers2:
                layer.forward(None, hidden, res, post, comb)
        self.assertEqual(len(preflight_calls), 1)
        self.assertIsInstance(preflight_calls[0], UnsupportedDecode)
        self.assertEqual(orig_calls, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
