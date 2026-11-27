# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for Torch operator argument order and bound-buffer lifecycle."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def _load_module():
    glm5next_dir = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"
    if importlib.util.find_spec("vllm") is not None:
        return importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_torch_session")
    modules = {}
    for pkg in ("vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.glm5next"):
        mod = ModuleType(pkg)
        mod.__path__ = [str(glm5next_dir)]
        modules[pkg] = mod
    for mod_name in (
        "mega_kernel_abi",
        "mega_kernel_collectives",
        "mega_kernel_weights",
        "mega_kernel_torch_session",
    ):
        fq = f"vllm_ascend.models.glm5next.{mod_name}"
        spec = importlib.util.spec_from_file_location(fq, glm5next_dir / f"{mod_name}.py")
        loaded = importlib.util.module_from_spec(spec)
        modules[fq] = loaded
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(loaded)
    return modules["vllm_ascend.models.glm5next.mega_kernel_torch_session"]


class BoundBuffersTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.inputs = {name: torch.tensor([i]) for i, name in enumerate(self.module.MC2_INPUT_NAMES)}
        self.outputs = [torch.zeros(1) for _ in range(50)]
        self.attrs = {
            name: i if name in self.module._INT_ATTRIBUTES else i + 0.25
            for i, name in enumerate(self.module.ATTR_NAMES)
        }
        self.groups = ("tp", "ep", "tp_algo", "ep_algo", 1, 3)
        self.sync_count = 0
        self.calls = []
        self.stream = SimpleNamespace(npu_stream=42, synchronize=self.sync)
        self.npu = SimpleNamespace(current_stream=lambda device: self.stream)

    def sync(self):
        self.sync_count += 1

    def op(self, *args):
        self.calls.append(args)
        return args[1]

    def test_operator_arguments_and_buffer_reuse(self):
        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        with patch.object(torch, "npu", self.npu, create=True):
            result = bound.run()
            bound.run()
        self.assertEqual(self.sync_count, 2)
        self.assertIs(result[0], self.outputs[0])
        self.assertEqual(self.calls[0][0], list(self.inputs.values()))
        self.assertEqual(self.calls[0][2], [int(self.attrs[n]) for n in self.module._INT_ATTRIBUTES])
        self.assertEqual(self.calls[0][3], [float(self.attrs[n]) for n in self.module._FLOAT_ATTRIBUTES])
        self.assertEqual(self.calls[0][4:], self.groups)

    def test_replaced_input_is_rejected_before_launch(self):
        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        self.inputs["x"] = self.inputs["x"].clone()
        with patch.object(torch, "npu", self.npu, create=True), self.assertRaisesRegex(RuntimeError, "changed"):
            bound.run()
        self.assertFalse(self.calls)

    def test_invalid_attributes_fail_before_launch(self):
        self.attrs["hc_mult"] = 2**63
        with self.assertRaisesRegex(ValueError, "int64"):
            self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        self.assertFalse(self.calls)

    def test_integer_attributes_reject_float_and_boolean_before_launch(self):
        for value in (1.2, 1.0, True, np.bool_(True), torch.tensor(True)):
            with self.subTest(value=value):
                self.attrs["hc_mult"] = value
                with self.assertRaisesRegex(TypeError, "integer attribute hc_mult"):
                    self.module._TorchBoundBuffers(
                        self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op
                    )
        self.assertFalse(self.calls)

    def test_integer_attributes_accept_index_protocol_and_int64_bounds(self):
        class IntLike:
            def __index__(self):
                return 7

        for value, expected in ((IntLike(), 7), (np.int64(9), 9), (-(2**63), -(2**63)), (2**63 - 1, 2**63 - 1)):
            with self.subTest(value=value):
                self.attrs["hc_mult"] = value
                bound = self.module._TorchBoundBuffers(
                    self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op
                )
                self.assertEqual(bound.int_attrs[0], expected)
        self.assertFalse(self.calls)

    def test_failure_disables_reuse_and_keeps_resources(self):
        def failed(*args):
            raise RuntimeError("launch failed")

        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, failed)
        with patch.object(torch, "npu", self.npu, create=True), self.assertRaisesRegex(RuntimeError, "launch failed"):
            bound.run()
        bound.close()
        self.assertIs(self.module._FAILED_LAUNCHES[-1], bound)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            bound.validate_bindings()


class _FakePreparedHandle:
    def __init__(self, inputs, outputs, workspace_obj, run_cb):
        self.inputs = inputs
        self.outputs = outputs
        self._workspace = workspace_obj
        self._run_cb = run_cb
        self.closed = False
        self.failed = False
        self.run_calls = 0
        self.close_calls = 0

    def validate_bindings(self):
        if self.closed or self.failed:
            raise RuntimeError("closed or failed")

    def run(self):
        self.run_calls += 1
        self._run_cb(self.inputs, self.outputs)
        return self.outputs

    def close(self):
        self.closed = True
        self.close_calls += 1

    def workspace(self):
        return self._workspace

    def workspace_size(self):
        return 0 if self._workspace is None else int(self._workspace.numel())


class SessionAndPoolTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.header_sha = "a" * 64
        self.model_digest = "b" * 64
        self.abi = self.module.ReviewedAbi(
            self.header_sha,
            self.module.NATIVE_SHARED_PROTOCOL,
            max_rows_per_source=64,
            max_total_rows=512,
            variable_rows=True,
            zero_rows=True,
            output_count=50,
        )
        self.attrs = {
            name: (28 if name == "stage_limit" else i) if name in self.module._INT_ATTRIBUTES else i + 0.5
            for i, name in enumerate(self.module.ATTR_NAMES)
        }
        self.static_weights = {
            name: torch.zeros(1)
            for name in self.module.MC2_INPUT_NAMES
            if name
            not in (
                "x",
                "post_h_res",
                "post_h_out",
                "post_h_post",
                "conv_state",
                "recurrent_state",
                "cu_seqlens",
                "state_indices",
                "external_route_ids",
                "external_route_weights",
            )
        }

    def _make_config(self, **overrides):
        kwargs = dict(
            opapi_path="/abs/opapi.so",
            generated_header_path="/abs/header.h",
            reviewed_abi=self.abi,
            heads_per_shard=1,
            model_num_heads=1,
            local_head_range=(0, 1),
            control_group=SimpleNamespace(),
            tp_group=SimpleNamespace(),
            ep_group=SimpleNamespace(),
            control_device=torch.device("cpu"),
            tp_algorithm="ring",
            ep_algorithm="alltoall",
            expert_shard_digests=("c" * 64,),
            generated_header_sha256=self.header_sha,
            tp_group_name="tp_comm",
            ep_group_name="ep_comm",
            ep_rank=0,
            propagation_rounds=1,
        )
        kwargs.update(overrides)
        return self.module.Mc2SessionConfig(**kwargs)

    def _make_session(self, pool, layer_idx, ordinal, prepare_fn, *, config=None, op=None):
        cfg = config or self._make_config()
        return self.module.MegaKernelTorchSession(
            cfg,
            lambda layer_idx=layer_idx: f"{self.model_digest}:{layer_idx}",
            op=op or (lambda *a: a[1]),
            prepare_fn=prepare_fn,
            layer_idx=layer_idx,
            layer_ordinal=ordinal,
            static_weights=self.static_weights,
            attrs=self.attrs,
            buffer_pool=pool,
        )

    def test_shared_pool_ping_pong_workspace_reuse_and_atomic_strided_commit(self):
        pool = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        syncs = []
        stream = SimpleNamespace(npu_stream=101, synchronize=lambda: syncs.append("sync"))
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        shared_ws = torch.empty(1024, dtype=torch.uint8)
        prepared_handles = []

        def fake_prepare(inputs, outputs, int_attrs, float_attrs, *rest):
            passed_ws = rest[-1]
            # Layer 0 receives None, receives shared_ws; Layer 1/2 receive shared_ws back.
            if not prepared_handles:
                self.assertIsNone(passed_ws)
                ws_out = shared_ws
            else:
                self.assertIs(passed_ws, shared_ws)
                ws_out = shared_ws
            idx_conv = self.module.MC2_INPUT_NAMES.index("conv_state")
            idx_rec = self.module.MC2_INPUT_NAMES.index("recurrent_state")

            def run_cb(in_list, out_list):
                for sidx in self.module.STATUS_OUTPUT_INDICES:
                    out_list[sidx].zero_()
                in_list[idx_conv].add_(3)
                in_list[idx_rec].add_(7)
                out_list[37].fill_(1.5)
                out_list[15].fill_(2.0)
                out_list[17][:, :4].fill_(0.25)
                out_list[18][:, :16].fill_(0.5)
                out_list[22][:, :8].fill_(5)
                out_list[21][:, :8].fill_(0.125)

            handle = _FakePreparedHandle(inputs, outputs, ws_out, run_cb)
            prepared_handles.append(handle)
            return handle

        sessions = [self._make_session(pool, i, i, fake_prepare) for i in range(3)]
        rows = 2
        slots = torch.tensor([3, 1], dtype=torch.int64)
        # Strided physical caches: dim-first conv [4, 384, 3] and interleaved 2-plane recurrent [4, 2, 1, 128, 128].
        caches = []
        for _ in range(3):
            raw_conv = torch.zeros(4, 384, 3, dtype=torch.bfloat16)
            conv_view = raw_conv.transpose(-1, -2)
            rec_pool = torch.zeros(4, 2, 1, 128, 128, dtype=torch.float32)
            rec_view = rec_pool[:, 0]
            caches.append((raw_conv, conv_view, rec_pool, rec_view))

        reduce_calls = []

        def fake_all_reduce(tensor, op=None, group=None):
            reduce_calls.append(int(tensor.numel()))

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.distributed, "all_reduce", side_effect=fake_all_reduce),
        ):
            hidden = torch.ones(rows, 4096, dtype=torch.bfloat16)
            res = None
            post = None
            comb = None
            for idx, sess in enumerate(sessions):
                _, conv_view, _, rec_view = caches[idx]
                out, res, post, comb = sess.forward_layer(
                    hidden_states=hidden,
                    residual=res,
                    post=post,
                    comb=comb,
                    conv_state=conv_view,
                    recurrent_state=rec_view,
                    conv_slots_i64=slots,
                    rec_slots_i64=slots,
                    is_first_layer=(idx == 0),
                    capture_routes=False,
                    defer_commit=True,
                )
                hidden = out
            # Before finalize_step, none of the physical caches have been mutated!
            for raw_conv, _, rec_pool, _ in caches:
                self.assertEqual(raw_conv.count_nonzero().item(), 0)
                self.assertEqual(rec_pool.count_nonzero().item(), 0)
            pool.finalize_step("cpu", SimpleNamespace())

        # Ping-pong checks: Layer 0 and Layer 2 share bank 0; Layer 1 uses bank 1; all share x and workspace.
        b0 = sessions[0]._bindings[(rows, 101)]
        b1 = sessions[1]._bindings[(rows, 101)]
        b2 = sessions[2]._bindings[(rows, 101)]
        self.assertIs(b0.bucket, b1.bucket)
        self.assertIs(b0.bucket.shared_workspace, shared_ws)
        self.assertIsNot(b0.bank, b1.bank)
        self.assertIs(b0.bank, b2.bank)
        self.assertEqual(syncs, [])
        self.assertEqual(reduce_calls, [1])
        self.assertEqual([h.run_calls for h in prepared_handles], [1, 1, 1])
        self.assertIsNone(sessions[0].last_route_ids)

        # Active slots [3, 1] committed across all 3 layers; inactive slots [0, 2] and interleaved plane 1 untouched.
        for raw_conv, _, rec_pool, _ in caches:
            self.assertTrue(torch.all(raw_conv[[3, 1]] == 3))
            self.assertEqual(raw_conv[[0, 2]].count_nonzero().item(), 0)
            self.assertTrue(torch.all(rec_pool[[3, 1], 0] == 7))
            self.assertEqual(rec_pool[[0, 2], 0].count_nonzero().item(), 0)
            self.assertEqual(rec_pool[:, 1].count_nonzero().item(), 0)

    def test_nonzero_status_and_early_abort_roll_back_all_layers(self):
        pool = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        stream = SimpleNamespace(npu_stream=202, synchronize=lambda: None)
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        status_to_fail = [False]

        def make_prepare(should_fail):
            def fake_prepare(inputs, outputs, *rest):
                def run_cb(in_list, out_list):
                    for sidx in self.module.STATUS_OUTPUT_INDICES:
                        out_list[sidx].zero_()
                    idx_conv = self.module.MC2_INPUT_NAMES.index("conv_state")
                    in_list[idx_conv].fill_(9)
                    if should_fail and status_to_fail[0]:
                        out_list[29].fill_(1)

                return _FakePreparedHandle(inputs, outputs, None, run_cb)

            return fake_prepare

        s0 = self._make_session(pool, 0, 0, make_prepare(False))
        s1 = self._make_session(pool, 1, 1, make_prepare(True))
        c0 = torch.zeros(4, 3, 384, dtype=torch.bfloat16)
        r0 = torch.zeros(4, 1, 128, 128, dtype=torch.float32)
        c1 = torch.zeros(4, 3, 384, dtype=torch.bfloat16)
        r1 = torch.zeros(4, 1, 128, 128, dtype=torch.float32)
        slots = torch.tensor([0, 2], dtype=torch.int64)
        status_to_fail[0] = True
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.distributed, "all_reduce", side_effect=lambda *a, **k: None),
        ):
            _, res, post, comb = s0.forward_layer(
                hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                residual=None,
                post=None,
                comb=None,
                conv_state=c0,
                recurrent_state=r0,
                conv_slots_i64=slots,
                rec_slots_i64=slots,
                is_first_layer=True,
                defer_commit=True,
            )
            s1.forward_layer(
                hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                residual=res,
                post=post,
                comb=comb,
                conv_state=c1,
                recurrent_state=r1,
                conv_slots_i64=slots,
                rec_slots_i64=slots,
                is_first_layer=False,
                defer_commit=True,
            )
            with self.assertRaises(self.module.MegaKernelCollectiveError):
                pool.finalize_step("cpu", SimpleNamespace())
        self.assertEqual(c0.count_nonzero().item(), 0)
        self.assertEqual(c1.count_nonzero().item(), 0)
        self.assertTrue(s0._failed and s1._failed)

        # Even if failure happens before record_pending (empty _pending), abort_step marks all registered sessions failed.
        pool2 = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        sa = self._make_session(pool2, 0, 0, None)
        sb = self._make_session(pool2, 1, 1, None)
        pool2.abort_step(mark_failed=True)
        self.assertTrue(sa._failed and sb._failed)

    def test_bounded_lru_geometry_eviction_and_flags(self):
        pool = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        stream = SimpleNamespace(npu_stream=303, synchronize=lambda: None)
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        handles = []

        def fake_prepare(inputs, outputs, *rest):
            def run_cb(in_list, out_list):
                for sidx in self.module.STATUS_OUTPUT_INDICES:
                    out_list[sidx].zero_()

            h = _FakePreparedHandle(inputs, outputs, None, run_cb)
            handles.append(h)
            return h

        cfg = self._make_config(sync_per_layer=False, consumer_readback=True)
        sess = self._make_session(pool, 0, 0, fake_prepare, config=cfg)
        cpu_reads = []
        orig_cpu = torch.Tensor.cpu

        def track_cpu(tensor, *args, **kwargs):
            cpu_reads.append(tensor.numel())
            return orig_cpu(tensor, *args, **kwargs)

        with patch.object(torch, "npu", fake_npu, create=True):
            # Sequence 64 -> 32 -> 64 -> 16: with capacity 2, (32) must be evicted and its handle closed.
            for rows in (6, 3, 6, 2):
                sess.ensure_geometry(rows, stream)
            self.assertEqual(list(pool._buckets.keys()), [(6, 303), (2, 303)])
            self.assertEqual(handles[0].close_calls, 0)
            self.assertEqual(handles[1].close_calls, 1)
            self.assertEqual(handles[2].close_calls, 0)

            # consumer_readback=True triggers D2H on outputs 37 and 48; capture_routes=True populates route tensors.
            c = torch.zeros(8, 3, 384, dtype=torch.bfloat16)
            r = torch.zeros(8, 1, 128, 128, dtype=torch.float32)
            slots = torch.tensor([0, 1], dtype=torch.int64)
            with (
                patch.object(torch.Tensor, "cpu", track_cpu),
                patch.object(torch.distributed, "all_reduce", side_effect=lambda *a, **k: None),
            ):
                sess.forward_layer(
                    hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                    residual=None,
                    post=None,
                    comb=None,
                    conv_state=c,
                    recurrent_state=r,
                    conv_slots_i64=slots,
                    rec_slots_i64=slots,
                    is_first_layer=True,
                    capture_routes=True,
                    defer_commit=False,
                )
            self.assertEqual(len(cpu_reads), 2)
            self.assertIsNotNone(sess.last_route_ids)
            self.assertIsNotNone(sess.last_route_weights)

    def test_load_torch_mc2_extension_checks_supports_repeatable(self):
        fake_prepare = object()
        fake_op = object()
        fake_ops = SimpleNamespace(cann_ops_transformer=SimpleNamespace(mega_kernel_mc2_out=fake_op))
        for supported, expected in ((False, None), (True, fake_prepare)):
            fake_mod = SimpleNamespace(
                _builder=SimpleNamespace(load=lambda: None),
                prepare_mega_kernel_mc2=fake_prepare,
                supports_repeatable_executor=lambda s=supported: s,
            )
            with (
                patch.object(importlib, "import_module", return_value=fake_mod),
                patch.object(torch, "ops", fake_ops),
            ):
                op, prep = self.module.load_torch_mc2_extension(load_module=True)
            self.assertIs(op, fake_op)
            self.assertIs(prep, expected)

    def test_legacy_forward_acquires_lock_and_checks_epoch(self):
        pool = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        stream = SimpleNamespace(npu_stream=404, synchronize=lambda: None)
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        epochs = [f"{self.model_digest}:0", f"{self.model_digest}:0", "changed" + "0" * 57]

        cfg = self._make_config()
        sess = self.module.MegaKernelTorchSession(
            cfg,
            lambda: epochs.pop(0),
            op=lambda *a: (
                [o.zero_() if idx in self.module.STATUS_OUTPUT_INDICES else o.fill_(1.0) for idx, o in enumerate(a[1])]
                and a[1]
            ),
            prepare_fn=None,
            layer_idx=0,
            layer_ordinal=0,
            static_weights=self.static_weights,
            attrs=self.attrs,
            buffer_pool=pool,
        )
        rows = 2
        d_inputs = {name: torch.zeros(1) for name in self.module.MC2_INPUT_NAMES}
        d_inputs.update(self.static_weights)
        d_inputs.update(
            x=torch.zeros(rows, 4, 4096, dtype=torch.bfloat16),
            post_h_out=torch.zeros(rows, 4096, dtype=torch.bfloat16),
            post_h_post=torch.zeros(rows, 128, dtype=torch.float32),
            post_h_res=torch.zeros(rows, 128, dtype=torch.float32),
            conv_state=torch.zeros(4, 3, 384, dtype=torch.bfloat16),
            recurrent_state=torch.zeros(4, 1, 128, 128, dtype=torch.float32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            state_indices=torch.tensor([[0, 0], [1, 1]], dtype=torch.int32),
            external_route_ids=None,
            external_route_weights=None,
        )
        identity = self.module.StepIdentity(0, 1, self.model_digest, "d" * 64)
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.distributed, "all_reduce", side_effect=lambda *a, **k: None),
        ):
            sess.forward(d_inputs, self.attrs, identity, validated_step=True, capture_routes=False)
            self.assertTrue(sess._lock.acquire(blocking=False))
            try:
                with self.assertRaisesRegex(RuntimeError, "concurrent MC2 session"):
                    sess.forward(d_inputs, self.attrs, identity, validated_step=True)
            finally:
                sess._lock.release()
            with self.assertRaisesRegex(RuntimeError, "epoch changed"):
                sess.forward(d_inputs, self.attrs, identity, validated_step=True)

    def test_create_torch_session_defaults_capture_routes_false_and_respects_toggle(self):
        pool = self.module.SharedStepBufferPool("cpu", max_cached_geometries=2)
        stream = SimpleNamespace(npu_stream=505, synchronize=lambda: None)
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        sess = self.module.create_torch_session(
            self._make_config(),
            lambda: f"{self.model_digest}:0",
            layer_idx=0,
            layer_ordinal=0,
            static_weights=self.static_weights,
            attrs=self.attrs,
            buffer_pool=pool,
            op=lambda *a: (
                [o.zero_() if idx in self.module.STATUS_OUTPUT_INDICES else o.fill_(1.0) for idx, o in enumerate(a[1])]
                and a[1]
            ),
            prepare_fn=lambda inp, out, *r: _FakePreparedHandle(
                inp,
                out,
                None,
                lambda i, o: [
                    o[idx].zero_() if idx in self.module.STATUS_OUTPUT_INDICES else o[idx].fill_(2.0) for idx in range(50)
                ],
            ),
        )
        self.assertIs(sess.capture_routes, False)
        c = torch.zeros(4, 3, 384, dtype=torch.bfloat16)
        r = torch.zeros(4, 1, 128, 128, dtype=torch.float32)
        slots = torch.tensor([0, 1], dtype=torch.int64)
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.distributed, "all_reduce", side_effect=lambda *a, **k: None),
        ):
            sess.forward_layer(
                hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                residual=None,
                post=None,
                comb=None,
                conv_state=c,
                recurrent_state=r,
                conv_slots_i64=slots,
                rec_slots_i64=slots,
                is_first_layer=True,
                defer_commit=False,
            )
            self.assertIsNone(sess.last_route_ids)
            self.assertIsNone(sess.last_route_weights)

            sess.capture_routes = True
            sess.forward_layer(
                hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                residual=None,
                post=None,
                comb=None,
                conv_state=c,
                recurrent_state=r,
                conv_slots_i64=slots,
                rec_slots_i64=slots,
                is_first_layer=True,
                defer_commit=False,
            )
            self.assertIsNotNone(sess.last_route_ids)
            self.assertIsNotNone(sess.last_route_weights)

            sess.capture_routes = False
            sess.forward_layer(
                hidden_states=torch.ones(2, 4096, dtype=torch.bfloat16),
                residual=None,
                post=None,
                comb=None,
                conv_state=c,
                recurrent_state=r,
                conv_slots_i64=slots,
                rec_slots_i64=slots,
                is_first_layer=True,
                defer_commit=False,
            )
            self.assertIsNone(sess.last_route_ids)


if __name__ == "__main__":
    unittest.main()
