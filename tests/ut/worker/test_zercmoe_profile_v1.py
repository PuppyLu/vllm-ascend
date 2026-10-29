# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker import model_runner_v1 as runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture
def runner():
    result = object.__new__(NPUModelRunner)
    result.vllm_config = SimpleNamespace(kv_transfer_config=None)
    result.max_num_tokens = 32
    result.get_model = Mock(return_value=object())
    result._dummy_run = Mock()
    result.eplb_warmup = Mock()
    result.sparse_kv_offload_enabled = False
    return result


@contextmanager
def profile_environment(events, *, active=True):
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.ascend_forward_context.use_cann_zercmoe", return_value=active) as gate,
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.initialize_zercmoe_buffer",
            side_effect=lambda: events.append("heap"),
        ) as initialize,
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.initialize_zercmoe_padding_ids",
            side_effect=lambda: events.append("padding_ids"),
        ) as padding_ids,
        patch.object(runner_module, "get_mc2_tokens_capacity", return_value=32),
        patch.object(runner_module, "select_moe_comm_method", return_value=MoECommType.ZERC_MOE) as select,
        patch.object(runner_module, "disable_compilation", side_effect=lambda _: nullcontext()) as disable,
        patch.object(GPUModelRunner, "profile_run", side_effect=lambda: events.append("super")) as parent_profile,
    ):
        yield SimpleNamespace(
            gate=gate,
            initialize=initialize,
            padding_ids=padding_ids,
            select=select,
            disable=disable,
            parent=parent_profile,
        )


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer", "kv_both"])
def test_zercmoe_profile_initializes_heap_before_parent_even_for_short_batches(runner, active, kv_role):
    # A disabled backend never bootstraps SHMEM. Any eligible A3 worker
    # reserves its heap before KV sizing, including short P or D batches.
    runner.vllm_config.kv_transfer_config = SimpleNamespace(kv_role=kv_role)
    events = []
    with profile_environment(events, active=active) as probes:
        runner.profile_run()
    assert events == (["heap"] if active else []) + ["super"]
    assert probes.initialize.call_count == int(active)
    probes.gate.assert_called_once_with(runner.vllm_config)
    probes.parent.assert_called_once_with()
    probes.select.assert_not_called()
    probes.disable.assert_not_called()
    runner._dummy_run.assert_not_called()
    runner.eplb_warmup.assert_called_once_with()


def test_zercmoe_profile_keeps_capacity_dummy_run_before_parent(runner):
    runner.max_num_tokens = 128
    events = []
    runner._dummy_run.side_effect = lambda *args, **kwargs: events.append("dummy")
    with profile_environment(events) as probes:
        runner.profile_run()
    assert events == ["heap", "dummy", "super"]
    probes.initialize.assert_called_once_with()
    probes.select.assert_called_once_with(32, runner.vllm_config)
    probes.disable.assert_called_once_with(runner.get_model.return_value)
    runner._dummy_run.assert_called_once_with(32, with_prefill=True, is_profile=True)
    probes.parent.assert_called_once_with()


def test_zercmoe_profile_warms_maximum_eager_shape_before_decode_graph(runner, monkeypatch):
    runner.vllm_config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    events = []
    runner._dummy_run.side_effect = lambda *args, **kwargs: events.append("dummy")
    monkeypatch.setattr(runner_module.torch.npu, "synchronize", lambda: events.append("sync"), raising=False)
    with profile_environment(events) as probes:
        runner.profile_run()
    assert events == ["heap", "padding_ids", "dummy", "sync", "super"]
    runner._dummy_run.assert_called_once_with(
        runner.max_num_tokens, with_prefill=True, cudagraph_runtime_mode=CUDAGraphMode.NONE, is_profile=True
    )
    probes.disable.assert_called_once_with(runner.get_model.return_value)


def test_zercmoe_profile_heap_failure_stops_before_parent_or_dummy(runner):
    events = []
    with profile_environment(events) as probes:
        probes.initialize.side_effect = RuntimeError("heap allocation failed")
        with pytest.raises(RuntimeError, match="heap allocation failed"):
            runner.profile_run()
    probes.initialize.assert_called_once_with()
    probes.parent.assert_not_called()
    probes.select.assert_not_called()
    runner._dummy_run.assert_not_called()
    assert events == []
