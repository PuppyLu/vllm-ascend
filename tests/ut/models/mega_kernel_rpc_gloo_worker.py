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
import time
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    main()
