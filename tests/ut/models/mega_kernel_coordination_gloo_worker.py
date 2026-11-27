# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise layer preflight consensus with real multi-process Gloo collectives."""

import datetime
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
