# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise layer preflight consensus with real multi-process Gloo collectives."""

import datetime
import importlib.util
import sys
import tempfile
from pathlib import Path

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


def run(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
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
            results = [None, None]
            dist.all_gather_object(results, result)
            assert results == [expected, expected], (rank, results, expected)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(run, args=(str(Path(directory) / "rdzv"),), nprocs=2, join=True)
