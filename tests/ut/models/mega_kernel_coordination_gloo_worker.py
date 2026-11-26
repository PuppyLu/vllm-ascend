# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise layer preflight consensus with real multi-process Gloo collectives."""

import datetime
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp


class CollectiveProtocolError(RuntimeError):
    pass


staged = types.ModuleType("cross_rank_prototype.staged_collectives")
staged.CollectiveProtocolError = CollectiveProtocolError
sys.modules["cross_rank_prototype"] = types.ModuleType("cross_rank_prototype")
sys.modules["cross_rank_prototype.staged_collectives"] = staged
path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_adapter.py"
spec = importlib.util.spec_from_file_location("coordination_adapter_cpu_test", path)
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


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
            except CollectiveProtocolError:
                assert expected == "abort"
            else:
                assert result is expected
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(run, args=(str(Path(directory) / "rdzv"),), nprocs=2, join=True)
