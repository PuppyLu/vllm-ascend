# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise layer preflight consensus with real multi-process Gloo collectives."""

import datetime
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_adapter.py"
spec = importlib.util.spec_from_file_location("coordination_adapter_cpu_test", path)
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
coord_path = path.with_name("mega_kernel_collectives.py")
coord_spec = importlib.util.spec_from_file_location("coordination_collectives_cpu_test", coord_path)
coord = importlib.util.module_from_spec(coord_spec)
coord_spec.loader.exec_module(coord)
sys.modules["vllm_ascend.models.glm5next.mega_kernel_collectives"] = coord
sched_spec = importlib.util.spec_from_file_location(
    "coordination_scheduler_cpu_test", path.with_name("mega_kernel_scheduler.py")
)
scheduler = importlib.util.module_from_spec(sched_spec)
sys.modules[sched_spec.name] = scheduler
sched_spec.loader.exec_module(scheduler)


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
        # Real rank collectives around successive model requests: an eligible
        # decode, a group fallback when one rank observes prefill, and another
        # request must all leave the gate reusable. An actual prepare failure
        # on one rank must abort peers instead of stranding them in the vote.
        ids = ["req-a", "req-b"]
        req = SimpleNamespace(
            req_ids=ids,
            num_scheduled_tokens=np.array([1, 1]),
            is_prefilling_np=np.zeros(2, dtype=bool),
            prefill_runs_as_decode_np=None,
            prefill_len_np=np.array([5, 7]),
            num_computed_prefill_tokens_np=np.array([5, 7]),
        )
        batch = SimpleNamespace(
            req_ids=ids,
            is_dummy=False,
            num_reqs=2,
            num_tokens=2,
            num_tokens_after_padding=2,
            num_draft_tokens=0,
            num_computed_tokens_np=np.array([9, 12]),
        )
        desc = SimpleNamespace(cg_mode=SimpleNamespace(name="NONE"))
        output = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1), scheduled_spec_decode_tokens={})

        class Runner:
            pcp_manager = None
            vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True),
                compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
            )

            def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
                if self.inject_error:
                    raise RuntimeError("injected prepare error")
                return batch

            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                self.prepare_inputs(scheduler_output, req, desc)
                return gate.is_pure_decode()

        gate = scheduler.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        gate.set_step_agreement(
            lambda accepted, digest, error: coord.agree_decode_step(
                accepted, digest, "cpu", dist.group.WORLD, local_error=error
            )
        )
        gate.install()
        runner = Runner()
        runner.inject_error = False
        try:
            assert runner.execute_model(output)
            assert not gate.is_pure_decode()
            req.is_prefilling_np[0] = rank == 1
            assert not runner.execute_model(output)
            # Finish the first batch and admit a different request set and
            # geometry, as happens when a subsequent user request arrives.
            next_ids = ["req-c", "req-d", "req-e"]
            req.req_ids = next_ids
            req.num_scheduled_tokens = np.ones(3, dtype=np.int64)
            req.is_prefilling_np = np.zeros(3, dtype=bool)
            req.prefill_len_np = np.array([2, 4, 6])
            req.num_computed_prefill_tokens_np = np.array([2, 4, 6])
            batch.req_ids = next_ids
            batch.num_reqs = batch.num_tokens = batch.num_tokens_after_padding = 3
            batch.num_computed_tokens_np = np.array([3, 5, 7])
            output.num_scheduled_tokens = dict.fromkeys(next_ids, 1)
            assert runner.execute_model(output)
            runner.inject_error = rank == 1
            try:
                runner.execute_model(output)
            except coord.MegaKernelCollectiveError:
                pass
            else:
                raise AssertionError("all ranks must abort on one rank's prepare failure")
            runner.inject_error = False

            def capture_guard():
                if rank == 1:
                    raise RuntimeError("injected capture guard failure")
                return True

            gate.capture_guard = capture_guard
            try:
                runner.execute_model(output)
            except coord.MegaKernelCollectiveError:
                pass
            else:
                raise AssertionError("all ranks must abort on one rank's capture guard failure")
        finally:
            gate.uninstall()
        # A missing route-capture API on one rank must abort all participants
        # at preparation, before any peer can launch the fused collective.
        session = SimpleNamespace(capture_routes=False) if rank != 1 else SimpleNamespace()
        prior = None
        changed = False
        error = None
        try:
            prior = coord.enable_diagnostic_route_capture(session)
            changed = True
        except Exception as exc:
            error = exc
        launched = False
        try:
            coord.agree_or_abort(error, "cpu", dist.group.WORLD, "benchmark preparation")
            launched = True
        except coord.MegaKernelCollectiveError:
            pass
        finally:
            if changed:
                session.capture_routes = prior
        results = [None] * world_size
        dist.all_gather_object(results, launched)
        assert results == [False] * world_size, (rank, results)
        if rank != 1:
            assert session.capture_routes is False
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        for world_size in (2, 8):
            rendezvous = str(Path(directory) / f"rdzv-{world_size}")
            mp.spawn(run, args=(rendezvous, world_size), nprocs=world_size, join=True)
