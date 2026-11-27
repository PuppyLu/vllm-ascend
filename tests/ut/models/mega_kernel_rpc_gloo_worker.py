# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise ordered worker RPCs and the real Decode gate with CPU Gloo.

This simulates the shared FIFO dispatch, not vLLM's shared-memory transport,
the model or the MC2 session. Each worker receives each RPC and only rank 0
returns the result, like a single-output-rank executor.
"""

import datetime
import importlib.util
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, size, rendezvous, inboxes, response):
    scheduler = load("rpc_scheduler_cpu_test", "mega_kernel_scheduler.py")
    coord = load("rpc_collectives_cpu_test", "mega_kernel_collectives.py")
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        gate = None

        class Runner:
            pcp_manager = None
            vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True),
                compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
            )

            def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
                return self.batch

            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                self.prepare_inputs(scheduler_output, self.req, self.desc)
                if not gate.is_pure_decode():
                    return "original"
                # Exercise collective ordering and cache geometry consensus per
                # layer. This is a protocol simulation, not MC2 admission.
                for layer in range(4):
                    changed = self.previous_rows[layer] != self.batch.num_reqs
                    status = torch.tensor([int(changed)], dtype=torch.int32)
                    dist.all_reduce(status, op=dist.ReduceOp.MAX)
                    if status.item():
                        self.previous_rows[layer] = self.batch.num_reqs
                        self.rebuilds[layer] += 1
                    dist.all_reduce(torch.ones(1, dtype=torch.int32))
                return "fused"

        gate = scheduler.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        gate.set_step_agreement(
            lambda accepted, digest, error: coord.agree_decode_step(
                accepted, digest, "cpu", dist.group.WORLD, local_error=error
            )
        )
        gate.install()
        runner = Runner()
        runner.previous_rows = [None] * 4
        runner.rebuilds = [0] * 4
        runner.desc = SimpleNamespace(cg_mode=SimpleNamespace(name="NONE"))
        while True:
            command = inboxes[rank].get(timeout=20)
            if command is None:
                break
            phase, request, rows = command
            ids = [f"{request}-{i}" for i in range(rows)]
            runner.req = SimpleNamespace(
                req_ids=ids,
                num_scheduled_tokens=np.ones(rows, dtype=np.int64),
                is_prefilling_np=np.full(rows, phase == "prefill", dtype=np.bool_),
                prefill_runs_as_decode_np=None,
                prefill_len_np=np.full(rows, 4),
                num_computed_prefill_tokens_np=np.full(rows, 4),
            )
            runner.batch = SimpleNamespace(
                req_ids=ids,
                is_dummy=False,
                num_reqs=rows,
                num_tokens=rows,
                num_tokens_after_padding=rows,
                num_draft_tokens=0,
                num_computed_tokens_np=np.full(rows, 5),
            )
            output = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1), scheduled_spec_decode_tokens={})
            result = runner.execute_model(output, dummy_run=phase == "dummy")
            if rank == 0:
                response.put((result, tuple(runner.rebuilds)))
        gate.uninstall()
    finally:
        dist.destroy_process_group()


def _load_full_stack():
    modules = {}
    for pkg in ("vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.glm5next"):
        mod = ModuleType(pkg)
        mod.__path__ = [str(HERE)]
        modules[pkg] = mod
    loaded = {}
    for name in (
        "mega_kernel_abi",
        "mega_kernel_collectives",
        "mega_kernel_weights",
        "mega_kernel_scheduler",
        "mega_kernel_torch_session",
        "mega_kernel_adapter",
    ):
        fq = f"vllm_ascend.models.glm5next.{name}"
        spec = importlib.util.spec_from_file_location(fq, HERE / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        modules[fq] = m
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(m)
        loaded[name] = m
    return loaded, modules


def _make_external_stubs(base_modules, ctx, *, fail_final=False):
    fwd = ModuleType("vllm.forward_context")
    fwd.get_forward_context = lambda: ctx
    kv = ModuleType("vllm.distributed.kv_transfer")
    kv.has_kv_transfer_group = lambda: False
    asc = ModuleType("vllm_ascend.ascend_config")
    asc.get_ascend_config = lambda: SimpleNamespace(enable_force_eplb=False)
    kda = ModuleType("vllm_ascend.models.glm5next.kda")
    kda.maybe_save_kv_layer_to_connector = lambda *a: None
    kda.record_attention_compute_start = lambda: None
    kda.wait_for_kv_layer_from_connector = lambda *a: None
    mhc = ModuleType("vllm_ascend.models.glm5next.ops.mhc_ops")

    def hc_contract(value, n):
        del n
        if fail_final():
            raise RuntimeError("injected final hc_contract failure")
        return value

    mhc.hc_contract = hc_contract
    stubs = dict(base_modules)
    stubs.update(
        {
            "vllm": ModuleType("vllm"),
            "vllm.forward_context": fwd,
            "vllm.distributed": ModuleType("vllm.distributed"),
            "vllm.distributed.kv_transfer": kv,
            "vllm_ascend.ascend_config": asc,
            "vllm_ascend.models.glm5next.kda": kda,
            "vllm_ascend.models.glm5next.ops": ModuleType("vllm_ascend.models.glm5next.ops"),
            "vllm_ascend.models.glm5next.ops.mhc_ops": mhc,
        }
    )
    return stubs


def full_lifecycle_worker(rank, size, rendezvous, inboxes, response):
    """Exercise full V2SchedulerEvidence + StepDecodeCoordinator + DecodeLayerAdapter + MegaKernelTorchSession."""
    mods, base_modules = _load_full_stack()
    abi = mods["mega_kernel_abi"]
    collectives = mods["mega_kernel_collectives"]
    scheduler = mods["mega_kernel_scheduler"]
    session_mod = mods["mega_kernel_torch_session"]
    adapter = mods["mega_kernel_adapter"]

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=size,
        timeout=datetime.timedelta(seconds=40),
    )
    try:
        members = list(range(size))
        control = dist.new_group(ranks=members, backend="gloo")
        tp = dist.new_group(ranks=members, backend="gloo")

        header_sha = "a" * 64
        model_digest = "b" * 64
        rev_abi = abi.ReviewedAbi(header_sha, abi.NATIVE_SHARED_PROTOCOL, 16, 128, True, True, 50)
        static_weights = {
            n: torch.zeros(1)
            for n in abi.MC2_INPUT_NAMES
            if n
            not in (
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
            )
        }
        attrs = {
            n: (28 if n == "stage_limit" else i) if n in abi.INT_ATTR_NAMES else i + 0.5
            for i, n in enumerate(abi.ATTR_NAMES)
        }
        sync_log = []
        stream = SimpleNamespace(npu_stream=700 + rank, synchronize=lambda: sync_log.append("sync"))
        fake_npu = SimpleNamespace(current_stream=lambda dev: stream)
        idx_conv = abi.MC2_INPUT_NAMES.index("conv_state")
        idx_rec = abi.MC2_INPUT_NAMES.index("recurrent_state")

        bias = torch.zeros(288)
        router = type("AscendFusedTopKRouter", (), {})()
        router.top_k = 8
        router.scoring_func = "sigmoid"
        router.renormalize = True
        router.routed_scaling_factor = 2.5
        router.num_expert_group = 1
        router.topk_group = 1
        router.e_score_correction_bias = bias
        routed = SimpleNamespace(
            router=router,
            log2phy=None,
            mix_placement=False,
            _select_experts=lambda *a: None,
            moe_config=SimpleNamespace(swiglu_limit=10.0),
        )
        ctx = SimpleNamespace(attn_metadata={})
        failure_flags = {"prepare": False, "preflight": None, "final": False}

        def build_stack(fail_commit_on_rank1: bool = False):
            # Test LRU eviction under concurrency ramp with capacity 2.
            pool = session_mod.SharedStepBufferPool("cpu", max_cached_geometries=2, control_device="cpu")
            cfg = session_mod.Mc2SessionConfig(
                "/abs/opapi.so",
                "/abs/h.h",
                rev_abi,
                1,
                1,
                (0, 1),
                control,
                tp,
                tp,
                torch.device("cpu"),
                "ring",
                "alltoall",
                ("c" * 64,),
                tp_group_name="tp",
                ep_group_name="ep",
                ep_rank=rank,
                sync_per_layer=True,
                defer_state_commit=True,
                max_cached_geometries=2,
                generated_header_sha256=header_sha,
                compute_device=torch.device("cpu"),
            )
            num_layers = 4
            caches = []
            layers = []
            entries = []
            orig_calls = []

            for i in range(num_layers):
                raw_conv = torch.zeros(16, 384, 3, dtype=torch.bfloat16)
                rec_pool = torch.zeros(16, 2, 1, 128, 128, dtype=torch.float32)
                conv_v = raw_conv.transpose(-1, -2)
                rec_v = rec_pool[:, 0]
                caches.append((raw_conv, conv_v, rec_pool, rec_v))

                def make_op(layer_idx):
                    def op(in_list, outputs, int_attrs, float_attrs, *groups):
                        for sidx in abi.STATUS_OUTPUT_INDICES:
                            outputs[sidx].zero_()
                        in_list[idx_conv].add_(1 + layer_idx)
                        in_list[idx_rec].add_(2 + layer_idx)
                        outputs[37].fill_(float(1 + layer_idx))
                        outputs[15].fill_(float(1 + layer_idx))
                        outputs[17][:, :4].fill_(0.25)
                        outputs[18][:, :16].fill_(0.5)
                        return outputs

                    return op

                sess = session_mod.MegaKernelTorchSession(
                    cfg,
                    lambda idx=i: f"{model_digest}:{idx}",
                    op=make_op(i),
                    prepare_fn=None,  # Simulate ops-transformer 1e591b193d (no prepare_mega_kernel_mc2)
                    layer_idx=i,
                    layer_ordinal=i,
                    static_weights=static_weights,
                    attrs=attrs,
                    buffer_pool=pool,
                )
                assert sess.capture_routes is False

                def make_orig(layer_idx):
                    def forward(positions, hidden_states, residual=None, post=None, comb=None):
                        del positions
                        orig_calls.append(layer_idx)
                        rows = hidden_states.shape[0]
                        if layer_idx == num_layers - 1:
                            return hidden_states, None, None, None
                        return (
                            hidden_states,
                            residual if residual is not None else torch.ones(rows, 4, 4096, dtype=torch.bfloat16),
                            post if post is not None else torch.ones(rows, 4, 1),
                            comb if comb is not None else torch.ones(rows, 4, 4),
                        )

                    return forward

                layer = SimpleNamespace(
                    layer_kind="kda",
                    _mlp_is_moe=True,
                    mhc=True,
                    is_mtp_layer=False,
                    is_sequence_parallel=False,
                    layer_idx=i,
                    num_hidden_layers=num_layers,
                    n=4,
                    self_attn=SimpleNamespace(
                        prefix=f"layer.{i}.kda",
                        pcp_size=1,
                        projection_tp=None,
                        use_recoverssm=False,
                        local_num_heads=1,
                        kv_cache=(raw_conv, rec_v),
                        _conv_state_dim_first=True,
                    ),
                    mlp=SimpleNamespace(
                        enable_eplb=False,
                        routed_scaling_factor=2.5,
                        experts=SimpleNamespace(
                            routed_experts=routed,
                            is_internal_router=True,
                            _compute_router_logits=lambda *a: None,
                        ),
                        shared_experts=SimpleNamespace(swiglu_limit=10.0),
                        gate=SimpleNamespace(e_score_correction_bias=bias),
                    ),
                    hc_post=lambda out, res, p, c: out + res[:, 0, :],
                    forward=make_orig(i),
                )
                backend = adapter.MegaKernelModelBackend(
                    sess, static_weights, attrs, lambda actual_layer, d: "step-id"
                )
                backend.session.bridge = SimpleNamespace(stage_limit=28)
                layers.append(layer)
                entries.append((layer, backend))

            if fail_commit_on_rank1 and rank == 1:
                orig_commit = session_mod._commit_compact_slots

                def broken_commit(*a, **kw):
                    raise RuntimeError("injected rank-1 commit failure")

                session_mod._commit_compact_slots = broken_commit
            else:
                orig_commit = None

            class Runner:
                pcp_manager = None
                vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(enforce_eager=True),
                    compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
                )

                def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
                    if failure_flags["prepare"]:
                        raise RuntimeError("injected rank-1 prepare failure")
                    return self.batch

                def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                    if self.zero_tokens:
                        return "no_forward"
                    self.prepare_inputs(scheduler_output, self.req, self.desc)
                    rows = self.batch.num_reqs
                    h = torch.ones(rows, 4096, dtype=torch.bfloat16)
                    res, post, comb = None, None, None
                    prev_hits = [lyr._megakernel_stats["hits"] for lyr in layers]
                    for lyr in layers:
                        h, res, post, comb = lyr.forward(None, h, res, post, comb)
                    return "fused" if all(lyr._megakernel_stats["hits"] == prev + 1 for lyr, prev in zip(layers, prev_hits)) else "original"

            gate = scheduler.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
            step_scratch = pool.get_step_contract_scratch()
            gate.set_step_agreement(
                lambda accepted, digest, rows, local_error=None: collectives.agree_decode_step_contract(
                    accepted,
                    rows,
                    digest,
                    16,
                    128,
                    size,
                    "cpu",
                    control,
                    scratch=step_scratch,
                    local_error=local_error,
                )[0]
            )
            gate.install()
            coord = adapter.StepDecodeCoordinator(
                gate=gate,
                pool=pool,
                control_device="cpu",
                control_group=control,
                layer_entries=entries,
                defer_state_commit=True,
            )
            preflight_scratch = pool.get_preflight_scratch()
            for lyr, backend in entries:
                adapter.DecodeLayerAdapter(
                    backend,
                    lambda actual_layer, err: adapter.coordinate_layer_preflight(
                        err, gate.is_pure_decode(), "cpu", control, scratch=preflight_scratch
                    ),
                    gate.execution_mode,
                    lambda actual_layer: gate.is_pure_decode(),
                    coordinator=coord,
                ).install(lyr)

            return pool, layers, caches, orig_calls, gate, Runner(), orig_commit

        pool, layers, caches, orig_calls, gate, runner, _ = build_stack()
        orig_ensure_geometry = layers[2]._megakernel_session.ensure_geometry
        runner.desc = SimpleNamespace(cg_mode=SimpleNamespace(name="NONE"))
        runner.zero_tokens = False
        stubs = _make_external_stubs(base_modules, ctx, fail_final=lambda: failure_flags["final"])

        output_queue = queue.Queue() if rank == 0 else None

        def output_loop():
            while True:
                item = output_queue.get()
                if item is None:
                    return
                time.sleep(0.01)
                response.put(item)

        output_thread = threading.Thread(target=output_loop, daemon=True) if rank == 0 else None
        if output_thread is not None:
            output_thread.start()

        try:
            with patch.dict(sys.modules, stubs), patch.object(torch, "npu", fake_npu, create=True):
                expected_conv = [0] * 16
                expected_rec = [0] * 16
                while True:
                    cmd = inboxes[rank].get(timeout=40)
                    if cmd is None:
                        break
                    kind, phase, request, slots = cmd
                    if kind == "sample":
                        v = torch.ones(1, dtype=torch.int32)
                        dist.all_reduce(v, group=tp)
                        assert int(v.item()) == size
                        if rank == 0:
                            output_queue.put((kind, request, "sampled"))
                        continue

                    if kind == "inject":
                        target, mode = phase, request
                        if target == "prepare":
                            failure_flags["prepare"] = rank == 1 and mode == "fail"
                        elif target == "preflight_err":
                            layers[2].mlp.enable_eplb = rank == 1 and mode == "eplb"
                        elif target == "preflight_exc":
                            if rank == 1:
                                if mode == "exc":
                                    def _fail_geom(*args, **kwargs):
                                        raise RuntimeError("injected preflight failure")
                                    layers[2]._megakernel_session.ensure_geometry = _fail_geom
                                else:
                                    layers[2]._megakernel_session.ensure_geometry = orig_ensure_geometry
                        elif target == "final":
                            failure_flags["final"] = rank == 1 and mode == "fail"
                        elif target == "commit_fail_group":
                            # Run self-contained rank-1 commit failure on a fresh pool+stack.
                            old_gate = gate
                            gate.uninstall()
                            f_pool, f_layers, f_caches, _, f_gate, f_runner, orig_commit = build_stack(
                                fail_commit_on_rank1=True
                            )
                            f_runner.desc = SimpleNamespace(cg_mode=SimpleNamespace(name="NONE"))
                            f_runner.zero_tokens = False
                            ids = ["commit-0"]
                            s_t = torch.tensor([0], dtype=torch.int32)
                            for i in range(4):
                                ctx.attn_metadata[f"layer.{i}.kda"] = SimpleNamespace(
                                    num_prefills=0,
                                    num_spec_decodes=0,
                                    num_decodes=1,
                                    num_actual_tokens=1,
                                    num_decode_tokens=1,
                                    non_spec_query_start_loc=torch.tensor([0, 1]),
                                    non_spec_state_indices_tensor=s_t,
                                    non_spec_decode_metadata=SimpleNamespace(
                                        causal_conv1d=SimpleNamespace(cache_indices=s_t)
                                    ),
                                    pcp_context=None,
                                )
                            f_runner.req = SimpleNamespace(
                                req_ids=ids,
                                num_scheduled_tokens=np.ones(1, dtype=np.int64),
                                is_prefilling_np=np.zeros(1, dtype=bool),
                                prefill_runs_as_decode_np=None,
                                prefill_len_np=np.ones(1, dtype=np.int64),
                                num_computed_prefill_tokens_np=np.ones(1, dtype=np.int64),
                            )
                            f_runner.batch = SimpleNamespace(
                                req_ids=ids,
                                is_dummy=False,
                                num_reqs=1,
                                num_tokens=1,
                                num_tokens_after_padding=1,
                                num_draft_tokens=0,
                                num_computed_tokens_np=np.array([2]),
                            )
                            sched = SimpleNamespace(
                                num_scheduled_tokens=dict.fromkeys(ids, 1),
                                scheduled_spec_decode_tokens={},
                            )
                            raised = False
                            try:
                                f_runner.execute_model(sched)
                            except collectives.MegaKernelCollectiveError:
                                raised = True
                            finally:
                                if orig_commit is not None:
                                    session_mod._commit_compact_slots = orig_commit
                                f_gate.uninstall()
                                gate = old_gate
                                gate.install()
                            assert raised, (rank, "commit failure must raise MegaKernelCollectiveError on all ranks")
                            if rank == 0:
                                output_queue.put(("inject", target, "ok"))
                            continue
                        if rank == 0:
                            output_queue.put(("inject", target, "ok"))
                        continue

                    if phase == "zero":
                        runner.zero_tokens = True
                        out_tag = runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
                        runner.zero_tokens = False
                        if rank == 0:
                            output_queue.put((kind, request, out_tag))
                        continue

                    rows = len(slots)
                    ids = [f"{request}-{i}" for i in range(rows)]
                    slot_t = torch.tensor(slots, dtype=torch.int32)
                    q_loc = torch.arange(rows + 1, dtype=torch.int32)
                    for i in range(4):
                        ctx.attn_metadata[f"layer.{i}.kda"] = SimpleNamespace(
                            num_prefills=0,
                            num_spec_decodes=0,
                            num_decodes=rows,
                            num_actual_tokens=rows,
                            num_decode_tokens=rows,
                            non_spec_query_start_loc=q_loc,
                            non_spec_state_indices_tensor=slot_t,
                            non_spec_decode_metadata=SimpleNamespace(
                                causal_conv1d=SimpleNamespace(cache_indices=slot_t)
                            ),
                            pcp_context=None,
                        )
                    runner.req = SimpleNamespace(
                        req_ids=ids,
                        num_scheduled_tokens=np.ones(rows, dtype=np.int64),
                        is_prefilling_np=np.full(rows, phase == "prefill", dtype=bool),
                        prefill_runs_as_decode_np=None,
                        prefill_len_np=np.full(rows, 4),
                        num_computed_prefill_tokens_np=np.full(rows, 3 if phase == "prefill" else 4),
                    )
                    runner.batch = SimpleNamespace(
                        req_ids=ids,
                        is_dummy=False,
                        num_reqs=rows,
                        num_tokens=rows,
                        num_tokens_after_padding=rows,
                        num_draft_tokens=0,
                        num_computed_tokens_np=np.full(rows, 5),
                    )
                    sched = SimpleNamespace(
                        num_scheduled_tokens=dict.fromkeys(ids, 1),
                        scheduled_spec_decode_tokens={},
                    )
                    try:
                        out_tag = runner.execute_model(sched, dummy_run=(phase == "dummy"))
                    except collectives.MegaKernelCollectiveError:
                        out_tag = "aborted"
                    if out_tag == "fused" and all(l._megakernel_stats["hits"] > 0 for l in layers):
                        if phase == "decode":
                            # Only check committed caches when step succeeded on decode.
                            for sl in slots:
                                expected_conv[sl] += 1
                                expected_rec[sl] += 1
                    for layer_idx, (raw_conv, _, rec_pool, _) in enumerate(caches):
                        for sl in range(16):
                            c_exp = expected_conv[sl] * (1 + layer_idx)
                            r_exp = expected_rec[sl] * (2 + layer_idx)
                            assert torch.all(raw_conv[sl] == c_exp), (rank, request, layer_idx, sl, c_exp)
                            assert torch.all(rec_pool[sl, 0] == r_exp), (rank, request, layer_idx, sl, r_exp)
                            assert rec_pool[sl, 1].count_nonzero().item() == 0
                    if rank == size - 1 and phase == "decode":
                        time.sleep(0.015)
                    if rank == 0:
                        output_queue.put((kind, request, (out_tag, [l._megakernel_stats["group_fallback"] for l in layers])))
        finally:
            if output_thread is not None:
                output_queue.put(None)
                output_thread.join(timeout=40)
                if output_thread.is_alive():
                    raise AssertionError("output thread did not drain")
            gate.uninstall()
            for lyr in layers:
                lyr._megakernel_session._closed = True
    finally:
        dist.destroy_process_group()


def _run_full_lifecycle(context, directory):
    for size in (2, 8):
        inboxes = [context.Queue(maxsize=8) for _ in range(size)]
        response = context.Queue(maxsize=8)
        workers = mp.spawn(
            full_lifecycle_worker,
            args=(size, str(Path(directory) / f"full-rdzv-{size}"), inboxes, response),
            nprocs=size,
            join=False,
        )
        # Scenario A: Request 1 -> 0-token finish -> Request 2 -> batch ramp (LRU eviction) -> 0-token -> Req 3.
        # Interleaved as 2-in-flight batches modeling async scheduling.
        seq_cmds = (
            ("prefill", "req1", [0]),
            ("zero", "idle1", []),
            ("decode", "req1", [0]),
            ("decode", "req1", [0]),
            ("zero", "finish1", []),
            ("prefill", "req2", [1]),
            ("decode", "req2", [1]),
            ("decode", "req2", [1]),
            ("decode", "ramp2", [1, 2]),
            ("decode", "ramp4", [1, 2, 3, 4]),
            ("decode", "ramp8", [0, 1, 2, 3, 4, 5, 6, 7]),
            ("decode", "ramp4b", [2, 3, 4, 5]),
            ("decode", "ramp2b", [3, 4]),
            ("decode", "ramp1b", [4]),
            ("zero", "finish2", []),
            ("prefill", "req3", [9]),
            ("decode", "req3", [9]),
        )
        try:
            for offset in range(0, len(seq_cmds), 2):
                window = seq_cmds[offset : offset + 2]
                dispatched = []
                for phase, req, slots in window:
                    for inbox in inboxes:
                        inbox.put(("execute", phase, req, slots))
                    if phase != "zero":
                        for inbox in inboxes:
                            inbox.put(("sample", phase, req, slots))
                    dispatched.append((phase, req))
                for phase, req in dispatched:
                    if phase == "zero":
                        res = response.get(timeout=40)
                        assert res == ("execute", req, "no_forward"), res
                    else:
                        ex = response.get(timeout=40)
                        expected_tag = "original" if phase == "prefill" else "fused"
                        assert ex == ("execute", req, (expected_tag, [0, 0, 0, 0])), (size, phase, req, ex)
                        sp = response.get(timeout=40)
                        assert sp == ("sample", req, "sampled"), (size, req, sp)

            # Scenario B: Asymmetric rank-1 failures (fallback, prepare abort, preflight abort, final-layer abort, commit abort).
            scenario_b = (
                (("inject", "preflight_err", "eplb", []), "ok"),
                (("execute", "decode", "fallback-step", [0]), ("original", [1, 1, 1, 1])),
                (("inject", "preflight_err", "clear", []), "ok"),
                (("execute", "decode", "recover-step", [0]), ("fused", [1, 1, 1, 1])),
                (("inject", "prepare", "fail", []), "ok"),
                (("execute", "decode", "prep-abort", [0]), ("aborted", [1, 1, 1, 1])),
                (("inject", "prepare", "clear", []), "ok"),
                (("inject", "preflight_exc", "exc", []), "ok"),
                (("execute", "decode", "preflight-abort", [0]), ("aborted", [1, 1, 1, 1])),
                (("inject", "preflight_exc", "clear", []), "ok"),
                (("inject", "final", "fail", []), "ok"),
                (("execute", "decode", "final-abort", [0]), ("aborted", [1, 1, 1, 1])),
                (("inject", "final", "clear", []), "ok"),
                (("inject", "commit_fail_group", "run", []), "ok"),
            )
            for cmd, expected_res in scenario_b:
                for inbox in inboxes:
                    inbox.put(cmd)
                kind, phase, req, _ = cmd
                out = response.get(timeout=40)
                if kind == "inject":
                    assert out == ("inject", phase, expected_res), (size, cmd, out)
                else:
                    assert out == ("execute", req, expected_res), (size, cmd, out)

            for inbox in inboxes:
                inbox.put(None)
            deadline = time.monotonic() + 30
            while not workers.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise AssertionError(f"{size} full-lifecycle workers failed to exit")
        finally:
            for process in workers.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


def main():
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        for size in (2, 8):
            inboxes = [context.Queue() for _ in range(size)]
            response = context.Queue()
            workers = mp.spawn(
                worker,
                args=(size, str(Path(directory) / f"rdzv-{size}"), inboxes, response),
                nprocs=size,
                join=False,
            )
            commands = (
                (("prefill", "request-a", 2), "original", 0),
                (("decode", "request-a", 2), "fused", 1),
                (("decode", "request-a", 2), "fused", 1),
                (("prefill", "request-b", 3), "original", 1),
                (("decode", "request-b", 3), "fused", 2),
                (("dummy", "profile", 3), "original", 2),
                (("decode", "request-b", 3), "fused", 2),
                (("prefill", "request-c", 2), "original", 2),
                (("decode", "request-c", 2), "fused", 3),
            )
            try:
                for command, expected, count in commands:
                    for inbox in inboxes:
                        inbox.put(command)
                    try:
                        result, rebuilds = response.get(timeout=20)
                    except queue.Empty as exc:
                        raise AssertionError(f"{size} rank RPC did not complete: {command}") from exc
                    assert result == expected and rebuilds == (count,) * 4, (size, command, result, rebuilds)
                for inbox in inboxes:
                    inbox.put(None)
                deadline = time.monotonic() + 30
                while not workers.join(timeout=1):
                    if time.monotonic() >= deadline:
                        raise AssertionError(f"{size} rank workers failed to exit")
            finally:
                # Do not leave worker processes alive if the harness fails.
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
        _run_full_lifecycle(context, directory)


if __name__ == "__main__":
    main()
