# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU simulation of queued model RPCs, real Decode gate and MC2 sessions.

Each worker reads the same execute/sample FIFO, and only rank 0 responds via
a delayed output thread. Two execute/sample pairs can be queued ahead. The
operator repo supplies its fake Prepared harness. Actual HCCL, MC2 device
execution, NPU D2H and vLLM shared-memory transport are not exercised.
"""

import argparse
import datetime
import importlib.util
import queue
import runpy
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, world, rendezvous, ops, revision, layer_count, inbox, response):
    torch.set_num_threads(1)
    saved = {
        name: getattr(dist, name)
        for name in ("is_initialized", "get_process_group_ranks", "all_reduce", "get_world_size", "all_gather_object")
    }
    fake = runpy.run_path(str(Path(ops) / "experimental/attention/mega_kernel/tests/repeatable_bridge_fake_worker.py"))
    for name, method in saved.items():
        setattr(dist, name, method)
    session_module = fake["m"]
    if revision:
        source = subprocess.run(
            [
                "git",
                "-C",
                ops,
                "show",
                f"{revision}:experimental/attention/mega_kernel/tests/support/vllm_mc2_repeatable.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        exec(compile(source, f"{revision}:vllm_mc2_repeatable.py", "exec"), session_module.__dict__)
    root = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next"
    scheduler = _load("full_rpc_scheduler_cpu_test", root / "mega_kernel_scheduler.py")
    collectives = _load("full_rpc_collectives_cpu_test", root / "mega_kernel_collectives.py")
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world, timeout=datetime.timedelta(seconds=40)
    )
    try:
        control, tp, ep = [dist.new_group(list(range(world)), backend="gloo") for _ in range(3)]

        def agree(error, device, group, phase):
            failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
            if failed.item():
                raise fake["CollectiveProtocolError"](phase) from error

        session_module._agree = agree
        config = SimpleNamespace(
            control_group=control,
            tp_group=tp,
            ep_group=ep,
            control_device="cpu",
            reviewed_abi=SimpleNamespace(max_rows_per_source=16, max_total_rows=128),
        )
        sessions, states, builds = [], [], [0] * layer_count
        for layer in range(layer_count):
            session = session_module.RepeatableMegaKernelSession(
                SimpleNamespace(config=config, bridge=SimpleNamespace(heads_per_shard=1)), lambda: "static-epoch"
            )

            def prepare(inputs, *unused, layer=layer):
                builds[layer] += 1
                return fake["Prepared"](inputs)

            session._new_prepared = prepare
            sessions.append(session)
            states.append((torch.full((4, 3, 384), 2, dtype=torch.bfloat16), torch.full((4, 1, 128, 128), 3.0)))

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
                for session, (conv, recurrent) in zip(sessions, states):
                    inputs = fake["inputs"](self.batch.num_reqs)
                    inputs["conv_state"], inputs["recurrent_state"] = conv, recurrent
                    inputs["state_indices"] = torch.tensor([[slot, slot] for slot in self.slots], dtype=torch.int32)
                    session.forward(inputs, {"attr": 1}, fake["Identity"](), validated_step=True)
                return "fused"

        gate = scheduler.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        gate.set_step_agreement(
            lambda accepted, digest, error: collectives.agree_decode_step(
                accepted, digest, "cpu", control, local_error=error
            )
        )
        gate.install()
        runner = Runner()
        runner.desc = SimpleNamespace(cg_mode=SimpleNamespace(name="NONE"))
        expected_states = [0] * 4
        pending = None
        # The production async output thread may wait for D2H while the worker
        # main thread continues reading RPCs. The sleep only models scheduling.
        output_queue = queue.Queue() if rank == 0 else None

        def output_loop():
            while True:
                value = output_queue.get()
                if value is None:
                    return
                time.sleep(0.02)
                response.put(value)

        output_thread = threading.Thread(target=output_loop, daemon=True) if rank == 0 else None
        if output_thread is not None:
            output_thread.start()
        try:
            while True:
                command = inbox.get(timeout=40)
                if command is None:
                    if pending is not None:
                        raise AssertionError("shutdown with unsampled execution")
                    break
                kind, phase, request, slots, expected_builds = command
                if kind == "sample":
                    if pending != request:
                        raise AssertionError((rank, "sampling wrong model step", pending, request))
                    # Simulate a sample-time TP collective on a domain distinct
                    # from the Decode control group, with all workers joining.
                    sample_value = torch.ones(1, dtype=torch.int32)
                    dist.all_reduce(sample_value, group=tp)
                    assert sample_value.item() == world
                    pending = None
                    if rank == 0:
                        output_queue.put((kind, request, "sampled"))
                    continue
                if kind != "execute" or pending is not None:
                    raise AssertionError((rank, "execute before sampling", command, pending))
                rows = len(slots)
                ids = [f"{request}-{i}" for i in range(rows)]
                runner.slots = slots
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
                if result == "fused":
                    for slot in slots:
                        expected_states[slot] += 1
                else:
                    # Model a prefill writing a fresh state into a reused slot.
                    # The original model prefill itself is not being executed.
                    for slot in slots:
                        expected_states[slot] = 0
                        for conv, recurrent in states:
                            conv[slot].fill_(2)
                            recurrent[slot].fill_(3)
                assert builds == [expected_builds] * layer_count, (rank, phase, builds)
                for conv, recurrent in states:
                    for slot, count in enumerate(expected_states):
                        assert torch.all(conv[slot] == 2 + count), (rank, phase, slot)
                        assert torch.all(recurrent[slot] == 3 + count), (rank, phase, slot)
                # A worker can be later than the response rank in real vLLM.
                if rank == world - 1 and phase == "decode":
                    time.sleep(0.05)
                pending = request
                if rank == 0:
                    output_queue.put((kind, request, result))
        finally:
            if output_thread is not None:
                output_queue.put(None)
                output_thread.join(timeout=40)
                if output_thread.is_alive():
                    raise AssertionError("output thread did not drain")
            gate.uninstall()
            for session in sessions:
                session.close()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-repo", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--layers", type=int, default=31)
    args = parser.parse_args()
    if args.layers < 1:
        raise ValueError("layers must be positive")
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        for size in (2, 8):
            inboxes = [context.Queue(maxsize=4) for _ in range(size)]
            response = context.Queue(maxsize=2)
            # Distinct per-rank FIFO queues model a broadcast delivered to all workers.
            workers = [
                context.Process(
                    target=worker,
                    args=(
                        rank,
                        size,
                        str(Path(directory) / f"rdzv-{size}"),
                        str(Path(args.ops_repo).resolve()),
                        args.revision,
                        args.layers,
                        inboxes[rank],
                        response,
                    ),
                )
                for rank in range(size)
            ]
            for process in workers:
                process.start()
            commands = (
                ("prefill", "first", [0], 0),
                ("decode", "first", [0], 1),
                ("decode", "first", [0], 1),
                ("prefill", "second", [1], 1),
                ("decode", "second", [1], 1),
                ("decode", "batch", [1, 3], 2),
                ("decode", "last", [2], 3),
            )
            try:
                for offset in range(0, len(commands), 2):
                    # Model two in-flight batches: send execute(N), sample(N),
                    # execute(N+1), sample(N+1) before reading either result.
                    batch = commands[offset : offset + 2]
                    for phase, request, slots, expected_builds in batch:
                        for kind in ("execute", "sample"):
                            command = (kind, phase, request, slots, expected_builds)
                            for inbox in inboxes:
                                inbox.put(command)
                    for phase, request, slots, expected_builds in batch:
                        for kind, expected in (
                            ("execute", "original" if phase == "prefill" else "fused"),
                            ("sample", "sampled"),
                        ):
                            try:
                                result = response.get(timeout=40)
                            except queue.Empty as exc:
                                raise AssertionError(
                                    f"{size} rank queued RPC did not complete: {phase}, {request}"
                                ) from exc
                            assert result == (kind, request, expected), (phase, slots, expected_builds, result)
                for inbox in inboxes:
                    inbox.put(None)
                for process in workers:
                    process.join(timeout=40)
                    if process.exitcode != 0:
                        raise AssertionError(f"worker exit code: {process.exitcode}")
            finally:
                for process in workers:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            print(
                f"FULL_RPC_SESSION_PASS world={size} layers={args.layers} revision={args.revision or 'worktree'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
