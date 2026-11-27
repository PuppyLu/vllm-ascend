# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional cross-repo CPU test of real session sequencing with Gloo.

Requires --ops-repo; the operator repository's fake harness provides CPU
bindings. The executor and cold admission remain fake; no HCCL or NPU runs.
"""

import argparse
import datetime
import runpy
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, size, rendezvous, ops, revision):
    torch.set_num_threads(1)
    methods = {
        name: getattr(dist, name)
        for name in ("is_initialized", "get_process_group_ranks", "all_reduce", "get_world_size", "all_gather_object")
    }
    fake = runpy.run_path(str(Path(ops) / "experimental/attention/mega_kernel/tests/repeatable_bridge_fake_worker.py"))
    for name, method in methods.items():
        setattr(dist, name, method)
    session_module = fake["m"]
    if revision:
        session_path = "experimental/attention/mega_kernel/tests/support/vllm_mc2_repeatable.py"
        source = subprocess.run(
            [
                "git",
                "-C",
                ops,
                "show",
                f"{revision}:{session_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        exec(compile(source, f"{revision}:vllm_mc2_repeatable.py", "exec"), session_module.__dict__)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        groups = [dist.new_group(list(range(size)), backend="gloo") for _ in range(3)]

        def agree(error, device, group, phase):
            failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
            if failed.item():
                raise fake["CollectiveProtocolError"](phase) from error

        session_module._agree = agree
        config = SimpleNamespace(
            control_group=groups[0],
            tp_group=groups[1],
            ep_group=groups[2],
            control_device="cpu",
            reviewed_abi=SimpleNamespace(max_rows_per_source=16, max_total_rows=128),
        )
        session = session_module.RepeatableMegaKernelSession(
            SimpleNamespace(config=config, bridge=SimpleNamespace(heads_per_shard=1)), lambda: "static-epoch"
        )
        prepared = []

        def prepare(inputs, *unused):
            result = fake["Prepared"](inputs)
            prepared.append(result)
            return result

        session._new_prepared = prepare
        conv = torch.full((4, 3, 384), 2, dtype=torch.bfloat16)
        recurrent = torch.full((4, 1, 128, 128), 3.0)
        expected = [0] * 4
        for rows, slots, expected_builds in ((1, [0], 1), (1, [1], 1), (2, [1, 3], 2), (1, [2], 3)):
            inputs = fake["inputs"](rows)
            inputs["conv_state"], inputs["recurrent_state"] = conv, recurrent
            inputs["state_indices"] = torch.tensor([[slot, slot] for slot in slots], dtype=torch.int32)
            session.forward(inputs, {"attr": 1}, fake["Identity"](), validated_step=True)
            for slot in slots:
                expected[slot] += 1
            assert len(prepared) == expected_builds, (rank, rows, len(prepared))
            for slot, count in enumerate(expected):
                assert torch.all(conv[slot] == 2 + count), (rank, rows, slot)
                assert torch.all(recurrent[slot] == 3 + count), (rank, rows, slot)
        session.close()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-repo", required=True)
    parser.add_argument("--revision", default="", help="optional git revision of actual session implementation")
    args = parser.parse_args()
    ops = str(Path(args.ops_repo).resolve())
    with tempfile.TemporaryDirectory() as directory:
        for size in (2, 8):
            mp.spawn(worker, args=(size, str(Path(directory) / f"rdzv-{size}"), ops, args.revision), nprocs=size)
            print(f"SESSION_GLOO_PASS world={size} revision={args.revision or 'worktree'}", flush=True)


if __name__ == "__main__":
    main()
