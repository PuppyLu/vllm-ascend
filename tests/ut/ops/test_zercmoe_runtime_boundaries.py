# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ops.fused_moe import moe_comm_method, token_dispatcher
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize(
    "mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.PIECEWISE],
)
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_zercmoe_a3_preserves_every_graph_mode_without_probing_backend(monkeypatch, mode, enforce_eager):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
    )
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    active = Mock(side_effect=AssertionError("A3 graph preservation must not inspect or load a backend"))
    monkeypatch.setattr(afc, "is_zercmoe_active_by_config", active)

    assert afc.get_zercmoe_cudagraph_mode(config) is mode
    assert config.compilation_config.cudagraph_mode is mode
    active.assert_not_called()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_zercmoe_a2_retains_full_graph_demotion_only_when_required(monkeypatch, mode, enforce_eager, active):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
    )
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    probe = Mock(return_value=active)
    monkeypatch.setattr(afc, "is_zercmoe_active_by_config", probe)
    expected = CUDAGraphMode.FULL_DECODE_ONLY if mode is CUDAGraphMode.FULL and not enforce_eager and active else mode

    assert afc.get_zercmoe_cudagraph_mode(config) is expected
    assert probe.call_count == int(mode is CUDAGraphMode.FULL and not enforce_eager)


@pytest.mark.parametrize("capacity,is_fused,expected", [(20000, True, 5000), (None, True, 32), (None, False, 16)])
def test_zercmoe_real_dispatcher_constructor_respects_optional_capacity(monkeypatch, capacity, is_fused, expected):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=4))
    group = SimpleNamespace(rank_in_group=0, world_size=16, device_group=Mock())
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "cpu-test-group"
    monkeypatch.setattr(token_dispatcher, "get_mc2_group", lambda: group)
    monkeypatch.setattr(token_dispatcher, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(token_dispatcher, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(token_dispatcher, "should_skip_allreduce_across_dp_group", lambda _: False)
    monkeypatch.setattr(
        token_dispatcher, "get_ascend_config", lambda: SimpleNamespace(get_mc2_comm_alg=lambda: "fullmesh")
    )
    monkeypatch.setattr(token_dispatcher, "get_mc2_tokens_capacity", lambda: 128)
    monkeypatch.setattr(token_dispatcher, "get_dispatch_v2_tokens_capacity", lambda: 64)
    with patch.object(torch.distributed, "get_rank", return_value=0):
        result = token_dispatcher.TokenDispatcherWithMC2(is_fused_mc2=is_fused, token_capacity=capacity)

    assert result.max_num_tokens_per_rank == expected
    assert result.ep_world_size == 16 and result.ep_rank_id == 0
    assert result.global_bs == 0


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_zercmoe_real_dispatcher_factory_passes_independent_capacity_only_on_a3(monkeypatch, device):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=100001),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )
    monkeypatch.setattr(moe_comm_method, "get_ascend_device_type", lambda: device)
    factory = Mock(return_value=object())
    monkeypatch.setattr(moe_comm_method, "TokenDispatcherWithMC2", factory)
    with patch("vllm.config.get_current_vllm_config", return_value=config):
        runner = object.__new__(moe_comm_method.ZercMoECommImpl)
        result = runner._get_token_dispatcher()

    assert result is factory.return_value
    if device == AscendDeviceType.A3:
        factory.assert_called_once_with(is_fused_mc2=True, token_capacity=100004)
    else:
        factory.assert_called_once_with(is_fused_mc2=True)


def test_zercmoe_does_not_change_fused_mc2_dispatcher_capacity(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setattr(moe_comm_method, "TokenDispatcherWithMC2", factory)
    runner = object.__new__(moe_comm_method.FusedMC2CommImpl)

    assert runner._get_token_dispatcher() is factory.return_value
    factory.assert_called_once_with(is_fused_mc2=True)
