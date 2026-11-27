# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise layer preflight consensus with real multi-process Gloo collectives."""

import datetime
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GLM5NEXT_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"


def _load_stack():
    if importlib.util.find_spec("vllm") is not None:
        return (
            importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_abi"),
            importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_collectives"),
            importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_weights"),
            importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_torch_session"),
            importlib.import_module("vllm_ascend.models.glm5next.mega_kernel_adapter"),
        )
    modules = {}
    for pkg in ("vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.glm5next"):
        mod = ModuleType(pkg)
        mod.__path__ = [str(GLM5NEXT_DIR)]
        modules[pkg] = mod
    loaded = []
    for name in (
        "mega_kernel_abi",
        "mega_kernel_collectives",
        "mega_kernel_weights",
        "mega_kernel_torch_session",
        "mega_kernel_adapter",
    ):
        fq = f"vllm_ascend.models.glm5next.{name}"
        spec = importlib.util.spec_from_file_location(fq, GLM5NEXT_DIR / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        modules[fq] = m
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(m)
        loaded.append(m)
    return tuple(loaded)


abi, coord, weights_mod, session_mod, adapter = _load_stack()


def run(rank, rendezvous, world_size):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        cases = (
            (None, True, True),
            (adapter.UnsupportedDecode("unsupported") if rank == 1 else None, True, False),
            (None, rank == 0, False),
            (RuntimeError("local error") if rank == 1 else None, True, "abort"),
            (RuntimeError("local error") if rank == 1 else adapter.UnsupportedDecode("unsupported"), True, "abort"),
        )
        for error, pure, expected in cases:
            try:
                result = adapter.coordinate_layer_preflight(error, pure, "cpu", dist.group.WORLD)
            except coord.MegaKernelCollectiveError:
                assert expected == "abort"
            else:
                assert result is expected
        step_cases = (
            (True, "0123456789abcdef" * 4, True),
            (True, ("0123456789abcdef" if rank == 0 else "fedcba9876543210") * 4, False),
            (True, "0" * 63 + ("0" if rank == 0 else "1"), False),
            (rank == 0, "f" * 64, False),
            (True, "e" * 64 if rank == 0 else None, False),
            (True, "a" * 64 if rank == 0 else "g" * 64, False),
            (True, "0" * 64, True),
            (True, "f" * 64, True),
        )
        for accepted, digest, expected in step_cases:
            result = coord.agree_decode_step(accepted, digest, "cpu", dist.group.WORLD)
            results = [None] * world_size
            dist.all_gather_object(results, result)
            assert results == [expected] * world_size, (rank, results, expected)

        # Missing route-capture API on one rank must abort all participants before launch.
        cap_session = NS(capture_routes=False) if rank != 1 else NS()
        prior = None
        changed = False
        cap_err = None
        try:
            prior = coord.enable_diagnostic_route_capture(cap_session)
            changed = True
        except Exception as exc:
            cap_err = exc
        launched = False
        try:
            coord.agree_or_abort(cap_err, "cpu", dist.group.WORLD, "benchmark preparation")
            launched = True
        except coord.MegaKernelCollectiveError:
            pass
        finally:
            if changed:
                cap_session.capture_routes = prior
        cap_results = [None] * world_size
        dist.all_gather_object(cap_results, launched)
        assert cap_results == [False] * world_size, (rank, cap_results)
        if rank != 1:
            assert cap_session.capture_routes is False

        # Contract collective with rows + SHA-256 digest across both 2-rank and 8-rank groups.
        contract_cases = (
            (True, 32, "0123456789abcdef" * 4, None, (True, 32)),
            (True, 32 if rank == 0 else 33, "0123456789abcdef" * 4, None, (False, 0)),
            (True, 300, "0123456789abcdef" * 4, None, (False, 0)),
            (True, 32, "0123456789abcdef" * 4, RuntimeError("boom") if rank == 1 else None, "abort"),
        )
        for accepted, rows, digest, local_err, expected in contract_cases:
            try:
                res = coord.agree_decode_step_contract(
                    accepted, rows, digest, 256, 512, world_size, "cpu", dist.group.WORLD, local_error=local_err
                )
            except coord.MegaKernelCollectiveError:
                assert expected == "abort"
            else:
                assert res == expected, (rank, res, expected)

        if world_size != 2:
            return

        # Multi-layer SharedStepBufferPool atomic commit vs rank-1 Layer-2 status abort on strided caches
        header_sha = "a" * 64
        model_digest = "b" * 64
        rev_abi = abi.ReviewedAbi(header_sha, abi.NATIVE_SHARED_PROTOCOL, 64, 512, True, True, 50)
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
        stream = NS(npu_stream=77, synchronize=lambda: None)
        fake_npu = NS(current_stream=lambda dev: stream)
        idx_conv = abi.MC2_INPUT_NAMES.index("conv_state")
        idx_rec = abi.MC2_INPUT_NAMES.index("recurrent_state")

        def run_3layer_tx(fail_rank1_layer2: bool, *, midstep_abort: bool = False):
            pool = session_mod.SharedStepBufferPool("cpu", max_cached_geometries=2)
            cfg = session_mod.Mc2SessionConfig(
                "/abs/opapi.so",
                "/abs/h.h",
                rev_abi,
                1,
                1,
                (0, 1),
                dist.group.WORLD,
                dist.group.WORLD,
                dist.group.WORLD,
                torch.device("cpu"),
                "ring",
                "alltoall",
                ("c" * 64,),
                tp_group_name="tp",
                ep_group_name="ep",
                ep_rank=rank,
                generated_header_sha256=header_sha,
            )

            class FakeHandle:
                def __init__(self, layer_idx, inputs, outputs):
                    self.layer_idx, self.inputs, self.outputs = layer_idx, inputs, outputs
                    self.closed = self.failed = False

                def validate_bindings(self):
                    pass

                def workspace(self):
                    return None

                def run(self):
                    for sidx in abi.STATUS_OUTPUT_INDICES:
                        self.outputs[sidx].zero_()
                    self.inputs[idx_conv].add_(3 + self.layer_idx)
                    self.inputs[idx_rec].add_(7 + self.layer_idx)
                    self.outputs[37].fill_(1.0)
                    self.outputs[15].fill_(1.0)
                    if fail_rank1_layer2 and rank == 1 and self.layer_idx == 2:
                        self.outputs[29].fill_(1)
                    return self.outputs

                def close(self):
                    self.closed = True

            sessions = [
                session_mod.MegaKernelTorchSession(
                    cfg,
                    lambda idx=i: f"{model_digest}:{idx}",
                    op=lambda *a: a[1],
                    prepare_fn=lambda inp, out, *r, idx=i: FakeHandle(idx, inp, out),
                    layer_idx=i,
                    layer_ordinal=i,
                    static_weights=static_weights,
                    attrs=attrs,
                    buffer_pool=pool,
                )
                for i in range(3)
            ]
            slots = torch.tensor([1, 3], dtype=torch.int64)
            caches = []
            for _ in range(3):
                raw_conv = torch.zeros(4, 384, 3, dtype=torch.bfloat16)
                rec_pool = torch.zeros(4, 2, 1, 128, 128, dtype=torch.float32)
                caches.append((raw_conv, raw_conv.transpose(-1, -2), rec_pool, rec_pool[:, 0]))
            local_err = None
            try:
                with patch.object(torch, "npu", fake_npu, create=True):
                    h, res, p, c = torch.ones(2, 4096, dtype=torch.bfloat16), None, None, None
                    for i, s in enumerate(sessions):
                        if midstep_abort and rank == 1 and i == 2:
                            raise RuntimeError("mid-step host error")
                        _, conv_v, _, rec_v = caches[i]
                        h, res, p, c = s.forward_layer(
                            hidden_states=h,
                            residual=res,
                            post=p,
                            comb=c,
                            conv_state=conv_v,
                            recurrent_state=rec_v,
                            conv_slots_i64=slots,
                            rec_slots_i64=slots,
                            is_first_layer=(i == 0),
                            defer_commit=True,
                        )
            except Exception as exc:
                local_err = exc
                pool.abort_step(mark_failed=True)
            try:
                for raw_conv, _, rec_pool, _ in caches:
                    assert raw_conv.count_nonzero().item() == 0
                    assert rec_pool.count_nonzero().item() == 0
                pool.finalize_step("cpu", dist.group.WORLD, local_error=local_err)
                return caches, sessions, None
            except Exception as exc:
                pool.abort_step(mark_failed=True)
                return caches, sessions, exc

        # Success case commits all 3 layers on both ranks
        caches_ok, _, err_ok = run_3layer_tx(False)
        assert err_ok is None, err_ok
        for i, (raw_conv, _, rec_pool, _) in enumerate(caches_ok):
            assert torch.all(raw_conv[[1, 3]] == 3 + i)
            assert raw_conv[[0, 2]].count_nonzero().item() == 0
            assert torch.all(rec_pool[[1, 3], 0] == 7 + i)
            assert rec_pool[[0, 2], 0].count_nonzero().item() == 0
            assert rec_pool[:, 1].count_nonzero().item() == 0

        # Failure on rank 1 layer 2 aborts BOTH ranks and leaves ALL 3 layers untouched!
        caches_fail, sessions_fail, err_fail = run_3layer_tx(True)
        assert isinstance(err_fail, coord.MegaKernelCollectiveError), (rank, err_fail)
        for raw_conv, _, rec_pool, _ in caches_fail:
            assert raw_conv.count_nonzero().item() == 0
            assert rec_pool.count_nonzero().item() == 0
        assert all(s._failed for s in sessions_fail)

        # Mid-step exception before layer 2 on rank 1 aborts BOTH ranks cleanly without mutating any layer cache!
        caches_mid, sessions_mid, err_mid = run_3layer_tx(False, midstep_abort=True)
        assert isinstance(err_mid, coord.MegaKernelCollectiveError), (rank, err_mid)
        for raw_conv, _, rec_pool, _ in caches_mid:
            assert raw_conv.count_nonzero().item() == 0
            assert rec_pool.count_nonzero().item() == 0
        if rank == 1:
            assert all(s._failed for s in sessions_mid)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        for world_size in (2, 8):
            rendezvous = str(Path(directory) / f"rdzv-{world_size}")
            mp.spawn(run, args=(rendezvous, world_size), nprocs=world_size, join=True)
