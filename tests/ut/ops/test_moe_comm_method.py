import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    FusedMC2CommImpl,
    MC2CommImpl,
    MoECommMethod,
    ZercMoECommImpl,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEFusedExpertsInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithMC2
from vllm_ascend.ops.fused_moe.zercmoe_adapter import prepare_w8a8_input
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.utils import AscendDeviceType


def _make_zercmoe_comm():
    comm = object.__new__(ZercMoECommImpl)
    comm.moe_config = SimpleNamespace(
        num_experts=256, experts_per_token=8, hidden_dim=16, intermediate_size_per_partition=32
    )
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.ep_world_size = 32
    comm.token_dispatcher.ep_rank_id = 3
    comm.token_dispatcher.max_num_tokens_per_rank = 128
    comm.prepare_finalize = SimpleNamespace(all_ranks_empty=False, input_dtype=torch.bfloat16)
    comm._zercmoe = None
    comm._a3_zercmoe = None
    comm._zerc_w8a8_operator = None
    comm._logged_zercmoe_launch = False
    comm._logged_zercmoe_fallbacks = set()
    comm._w8a8_padding_ids = None
    comm._w8a8_padding_key = None
    comm._w8a8_padding_inputs = None
    comm._w8a8_padding_inputs_key = None
    comm._zerc_buffer_weight_dtype = None
    comm._zerc_buffer_input_dtype = None
    comm.zerc_symm_buffer = None
    return comm


def _make_zercmoe_input():
    return MoEFusedExpertsInput(
        hidden_states=torch.zeros(2, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(2, 8, dtype=torch.bfloat16),
        topk_ids=torch.zeros(2, 8, dtype=torch.int64),
        weights=MoEWeights(w1=torch.empty(32, dtype=torch.bfloat16), w2=torch.empty(32, dtype=torch.bfloat16)),
        routing=MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(),
    )


@pytest.mark.parametrize("rank,world_size", [(None, None), ("3", "32"), ("3", None), (None, "32")])
@patch.dict(os.environ)
def test_zercmoe_bootstrap_uses_ep_rank_environment(monkeypatch, rank, world_size):
    comm = _make_zercmoe_comm()
    for name, value in (("RANK", rank), ("WORLD_SIZE", world_size)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    config = SimpleNamespace(zercmoe_ipport="tcp://ep-bootstrap:28765", zercmoe_shmem_mb=1024)
    wheel = MagicMock()

    def load_wheel():
        assert os.environ["RANK"] == "3"
        assert os.environ["WORLD_SIZE"] == "32"
        return wheel

    def create_buffer(**kwargs):
        assert os.environ["RANK"] == "3" and os.environ["WORLD_SIZE"] == "32"
        return wheel.get_symm_buffer.return_value

    wheel.get_symm_buffer.side_effect = create_buffer

    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=config),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.comm_utils.load_zercmoe_ops", side_effect=load_wheel),
        patch("torch.npu.current_device", return_value=1),
    ):
        result = comm._init_zerc_symm_buffer()

    assert result is wheel.get_symm_buffer.return_value
    wheel.get_symm_buffer.assert_called_once_with(
        moe_expert_num=256,
        ep_world_size=32,
        num_max_tokens_per_rank=128,
        num_topk=8,
        hidden=16,
        intermediate_hidden=32,
        ipport="tcp://ep-bootstrap:28765",
        device_id=1,
        mem_size_mb=1024,
    )


@pytest.mark.parametrize("heap_mb", [None, 0, 1024, 2048, 8192, 17999, 18000, 24000])
@patch.dict(os.environ)
def test_zercmoe_a3_bootstrap_uses_documented_wheel_and_actual_ep_identity(monkeypatch, heap_mb):
    comm = _make_zercmoe_comm()
    group = SimpleNamespace(world_size=32, rank_in_group=3, device_group=object())
    backend = SimpleNamespace(get_symm_buffer=MagicMock())
    config = SimpleNamespace(zercmoe_max_recv_token_num=None, zercmoe_ipport="tcp://ep:28765", zercmoe_shmem_mb=heap_mb)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_PORT", "19999")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7,8,9,10,11")
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=config),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.comm_utils.load_a3_zercmoe_ops", return_value=backend),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.comm_utils.load_zercmoe_ops") as a2_load,
        patch("torch.npu.current_device", return_value=1),
        patch("torch.npu.device_count", return_value=8, create=True),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.info") as log_info,
    ):
        result = comm._init_zerc_symm_buffer(w_dtype=torch.int8)
    log_info.assert_any_call(
        "A3 ZercMoE device visibility: EP rank=%s/%s ASCEND_RT_VISIBLE_DEVICES=%s "
        "visible_device_count=%s current_logical_device=%s",
        3,
        32,
        "4,5,6,7,8,9,10,11",
        8,
        1,
    )
    a2_load.assert_not_called()
    assert os.environ["RANK"] == "3" and os.environ["WORLD_SIZE"] == "32"
    assert os.environ["LOCAL_RANK"] == "1" and os.environ["MASTER_PORT"] == "19999"
    assert result is backend.get_symm_buffer.return_value
    backend.get_symm_buffer.assert_called_once_with(
        moe_expert_num=256,
        ep_world_size=32,
        num_max_tokens_per_rank=128,
        num_topk=8,
        hidden=16,
        intermediate_hidden=32,
        ipport="tcp://ep:28765",
        device_id=1,
        w_dtype=torch.int8,
        mem_size_mb=heap_mb or 0,
    )


@pytest.mark.parametrize("heap_mb,max_recv", [(1, None), (1023, None), (0, 8192)])
def test_zercmoe_a3_rejects_old_capacity_or_small_heap_before_import(heap_mb, max_recv):
    comm = _make_zercmoe_comm()
    config = SimpleNamespace(zercmoe_shmem_mb=heap_mb, zercmoe_max_recv_token_num=max_recv)
    group = SimpleNamespace(world_size=32, rank_in_group=3)
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=config),
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        pytest.raises(ValueError, match="ZercMoE|zercmoe"),
    ):
        comm._init_zerc_symm_buffer(w_dtype=torch.int8)
    load.assert_not_called()


def test_zercmoe_profile_heap_is_created_once_and_reused_by_forward():
    comm = _make_zercmoe_comm()
    buffer = object()
    with patch.object(comm, "_init_zerc_symm_buffer", return_value=buffer) as bootstrap:
        comm.initialize_w8a8_buffer()
        comm.initialize_w8a8_buffer()
    bootstrap.assert_called_once_with(w_dtype=torch.int8)
    assert comm.zerc_symm_buffer is buffer
    assert comm._zerc_buffer_weight_dtype == torch.int8
    assert comm._zerc_buffer_input_dtype == torch.bfloat16


@pytest.mark.parametrize(
    "rank,world_size",
    [("invalid", None), ("2", None), (None, "4"), ("3", "invalid"), ("3", "0")],
)
def test_zercmoe_bootstrap_rejects_conflicting_environment_before_import(monkeypatch, rank, world_size):
    comm = _make_zercmoe_comm()
    for name, value in (("RANK", rank), ("WORLD_SIZE", world_size)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config"),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.comm_utils.load_zercmoe_ops") as load_wheel,
        pytest.raises(ValueError, match="ZercMoE"),
    ):
        comm._init_zerc_symm_buffer()

    load_wheel.assert_not_called()
    # Validate both variables before publishing either one.
    assert os.environ.get("RANK") == rank
    assert os.environ.get("WORLD_SIZE") == world_size


@pytest.mark.parametrize("rank,world_size", [(-1, 32), (32, 32), (0, 0)])
def test_zercmoe_bootstrap_rejects_invalid_group_before_import(rank, world_size):
    comm = _make_zercmoe_comm()
    comm.token_dispatcher.ep_rank_id = rank
    comm.token_dispatcher.ep_world_size = world_size
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config"),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.comm_utils.load_zercmoe_ops") as load_wheel,
        pytest.raises(ValueError, match="Invalid ZercMoE EP group"),
    ):
        comm._init_zerc_symm_buffer()
    load_wheel.assert_not_called()


@pytest.mark.parametrize(
    "invalid_input", ["quantized", "fp16", "weight_list", "weight_rank", "weight_dtype", "capacity"]
)
def test_zercmoe_rejects_invalid_payload_before_bootstrap(invalid_input):
    comm = _make_zercmoe_comm()
    payload = _make_zercmoe_input()
    if invalid_input == "quantized":
        payload = replace(payload, quant=MoEQuantParams(quant_type=QuantType.W8A8MXFP))
    elif invalid_input == "fp16":
        payload = replace(payload, hidden_states=payload.hidden_states.to(torch.float16))
    elif invalid_input == "weight_list":
        payload = replace(payload, weights=replace(payload.weights, w1=[payload.weights.w1]))
    elif invalid_input == "weight_rank":
        payload = replace(payload, weights=replace(payload.weights, w2=torch.empty(4, 8, dtype=torch.bfloat16)))
    elif invalid_input == "weight_dtype":
        payload = replace(payload, weights=replace(payload.weights, w2=torch.empty(32, dtype=torch.int8)))
    else:
        payload = replace(payload, hidden_states=torch.zeros(129, 16, dtype=torch.bfloat16))

    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch.object(comm, "_ensure_zercmoe_loaded") as load_wheel,
        patch.object(comm, "_init_zerc_symm_buffer") as bootstrap,
        pytest.raises(ValueError, match="ZERC_MOE|ZercMoE"),
    ):
        comm.fused_experts(payload)
    load_wheel.assert_not_called()
    bootstrap.assert_not_called()


def test_zercmoe_a3_rejects_unquantized_payload_before_loading_the_a2_backend():
    comm = _make_zercmoe_comm()
    payload = _make_zercmoe_input()
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        patch.object(comm, "_init_zerc_symm_buffer") as bootstrap,
        pytest.raises(ValueError, match="A3 ZercMoE requires W8A8"),
    ):
        comm.fused_experts(payload)
    load.assert_not_called()
    bootstrap.assert_not_called()


def test_zercmoe_valid_bf16_call_keeps_abi_and_reuses_buffer():
    comm = _make_zercmoe_comm()
    payload = _make_zercmoe_input()
    buffer = SimpleNamespace(moe_expert_num=256, ep_world_size=32, ccl_buffer_size=4096)
    out = torch.zeros_like(payload.hidden_states)
    counts = torch.zeros(8, dtype=torch.int32)
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=buffer) as bootstrap,
        patch("torch.npu.current_stream") as current_stream,
        patch("torch.ops.npu.zerc_moe", return_value=(out, counts), create=True) as op,
    ):
        first = comm.fused_experts(payload)
        second = comm.fused_experts(payload)

    bootstrap.assert_called_once()
    assert op.call_count == 2
    args = op.call_args.args
    assert args[0] is payload.hidden_states
    assert args[1].dtype == torch.int32
    assert args[2].dtype == torch.float32
    assert args[3][0] is payload.weights.w1
    assert args[4][0] is payload.weights.w2
    assert args[5:] == (256, 32, 4096, 0, 0)
    assert first.routed_out is out and second.routed_out is out
    assert first.expert_tokens is counts
    assert first.before_dispatch_evt is current_stream.return_value.record_event.return_value


def _make_zercmoe_w8a8_case():
    comm = _make_zercmoe_comm()
    comm.moe_config.hidden_dim = 32
    comm.moe_config.intermediate_size_per_partition = 64
    payload = replace(
        _make_zercmoe_input(),
        hidden_states=torch.zeros(2, 32, dtype=torch.bfloat16),
        topk_ids=torch.arange(8, dtype=torch.int64).expand(2, -1).contiguous(),
        weights=MoEWeights(
            w1=torch.zeros(8 * 32 * 128, dtype=torch.int8),
            w2=torch.zeros(8 * 64 * 32, dtype=torch.int8),
            w1_scale=torch.ones(8, 128, dtype=torch.int64),
            w2_scale=torch.ones(8, 32, dtype=torch.int64),
        ),
        routing=replace(_make_zercmoe_input().routing, pertoken_scale=None),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
    )
    return comm, payload


@pytest.mark.parametrize(
    "invalid",
    ["scale_missing", "scale_dtype", "scale_shape", "packed_size", "x_rank", "pertoken_scale", "activation", "clamp"],
)
def test_zercmoe_w8a8_invalid_payload_fails_before_import_and_bootstrap(invalid):
    comm, payload = _make_zercmoe_w8a8_case()
    if invalid == "scale_missing":
        payload = replace(payload, weights=replace(payload.weights, w1_scale=None))
    elif invalid == "scale_dtype":
        payload = replace(payload, weights=replace(payload.weights, w2_scale=payload.weights.w2_scale.float()))
    elif invalid == "scale_shape":
        payload = replace(payload, weights=replace(payload.weights, w1_scale=torch.ones(8, 127, dtype=torch.int64)))
    elif invalid == "packed_size":
        payload = replace(payload, weights=replace(payload.weights, w1=payload.weights.w1[:-1]))
    elif invalid == "x_rank":
        payload = replace(payload, hidden_states=payload.hidden_states.unsqueeze(0))
    elif invalid == "pertoken_scale":
        payload = replace(payload, routing=replace(payload.routing, pertoken_scale=torch.ones(2)))
    elif invalid == "activation":
        payload = replace(payload, activation="gelu")
    else:
        payload = replace(payload, swiglu_limit=7.0)
    with (
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        patch.object(comm, "_init_zerc_symm_buffer") as bootstrap,
        pytest.raises(ValueError, match="ZERC_MOE"),
    ):
        comm.fused_experts(payload)
    load.assert_not_called()
    bootstrap.assert_not_called()


def test_zercmoe_w8a8_old_launcher_fails_before_bootstrap():
    comm, payload = _make_zercmoe_w8a8_case()
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer") as bootstrap,
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator",
            side_effect=RuntimeError("missing scale ABI"),
        ),
        pytest.raises(RuntimeError, match="missing scale ABI"),
    ):
        comm.fused_experts(payload)
    bootstrap.assert_not_called()


@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("record_event", [False, True])
def test_zercmoe_w8a8_call_preserves_bf16_and_reuses_packed_weights(empty_rank, record_event):
    comm, payload = _make_zercmoe_w8a8_case()
    payload = replace(
        payload,
        routing=replace(payload.routing, mc2_mask=torch.full((2,), not empty_rank, dtype=torch.bool)),
    )
    buffer = SimpleNamespace(num_experts=256, ep_world_size=32, ccl_buffer_size=4096)
    operator = MagicMock(return_value=(torch.zeros(2, 32, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.int32)))
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=buffer) as bootstrap,
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator) as resolve,
        patch("torch_npu.npu_dynamic_quant", create=True) as quant,
        patch("torch.npu.current_stream") as current_stream,
    ):
        for _ in range(2):
            result = comm.fused_experts(payload, record_dispatch_event=record_event)
    bootstrap.assert_called_once_with(w_dtype=torch.int8)
    resolve.assert_called_once()
    quant.assert_not_called()
    assert operator.call_count == 2
    sent, sent_buffer = operator.call_args.args
    assert sent_buffer is buffer
    assert sent.hidden_states.dtype == torch.bfloat16
    assert sent.weights is payload.weights
    assert sent.weights.w1.ndim == sent.weights.w2.ndim == 1
    assert sent.topk_weights.dtype == torch.float32
    assert sent.topk_ids.dtype == torch.int32
    assert sent.routing.pertoken_scale is None
    if empty_rank:
        assert torch.count_nonzero(sent.topk_weights) == 0
        assert torch.count_nonzero(sent.hidden_states) == 0
        # EP rank 3 owns physical experts 24..31; padding stays local.
        assert sent.topk_ids[0].tolist() == list(range(24, 32))
    else:
        torch.testing.assert_close(sent.hidden_states, payload.hidden_states)
    assert result.routed_out is operator.return_value[0]
    assert current_stream.return_value.record_event.call_count == (2 if record_event else 0)
    if not record_event:
        assert result.before_dispatch_evt is None


@pytest.mark.parametrize("ep_rank", [3, 19])
def test_zercmoe_padding_vector_is_cached_for_the_actual_ep_rank(ep_rank):
    comm, payload = _make_zercmoe_w8a8_case()
    comm.token_dispatcher.ep_rank_id = ep_rank
    payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.zeros(2, dtype=torch.bool)))
    first = comm._get_w8a8_padding_ids(payload)
    with patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.make_w8a8_padding_ids", side_effect=AssertionError("reallocated")
    ):
        assert comm._get_w8a8_padding_ids(payload) is first
        # Physical batch size may change; the topK device vector is reusable.
        assert comm._get_w8a8_padding_ids(replace(payload, hidden_states=payload.hidden_states[:1])) is first
        comm.prepare_finalize.all_tokens_valid = True
        assert comm._get_w8a8_padding_ids(payload) is None
    assert first.tolist() == list(range(ep_rank * 8, ep_rank * 8 + 8))
    comm.prepare_finalize.all_tokens_valid = False
    shorter = replace(payload, topk_ids=payload.topk_ids[:, :4])
    assert comm._get_w8a8_padding_ids(shorter).tolist() == first[:4].tolist()
    assert comm._get_w8a8_padding_ids(shorter) is not first


def test_zercmoe_static_mapping_does_not_remap_physical_padding_at_launch():
    comm, payload = _make_zercmoe_w8a8_case()
    mapping = torch.roll(torch.arange(256, dtype=torch.int32), 1)
    ids = payload.topk_ids.clone()
    ids[1] = 10000
    payload = replace(
        payload, topk_ids=ids, routing=replace(payload.routing, mc2_mask=torch.tensor([True, False]), log2phy=mapping)
    )
    operator = MagicMock(return_value=(torch.zeros_like(payload.hidden_states), torch.zeros(8, dtype=torch.int32)))
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=object()),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator),
    ):
        comm.fused_experts(payload, record_dispatch_event=False)
    sent = operator.call_args.args[0]
    torch.testing.assert_close(sent.topk_ids[0], mapping[ids[0]])
    assert sent.topk_ids[1].tolist() == list(range(24, 32))
    assert sent.routing.log2phy is None


def test_zercmoe_full_padding_cache_keeps_current_layer_weights_and_collective_calls():
    comm, payload = _make_zercmoe_w8a8_case()
    payload = replace(
        payload,
        hidden_states=torch.full_like(payload.hidden_states, float("nan")),
        topk_ids=torch.full_like(payload.topk_ids, -10000),
        routing=replace(
            payload.routing,
            mc2_mask=torch.zeros(2, dtype=torch.bool),
            log2phy=torch.roll(torch.arange(256, dtype=torch.int32), 1),
        ),
    )
    comm.prepare_finalize.all_tokens_padding = True
    other_layer = replace(payload, weights=replace(payload.weights, w1=payload.weights.w1.clone()))
    operator = MagicMock(return_value=(torch.zeros_like(payload.hidden_states), torch.zeros(8, dtype=torch.int32)))
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=object()) as bootstrap,
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator),
        patch("torch.where", side_effect=AssertionError("full-padding where")),
        patch("torch.Tensor.item", side_effect=AssertionError("device synchronization")),
    ):
        first = comm.fused_experts(payload, record_dispatch_event=False)
        cached = comm._w8a8_padding_inputs
        with patch("torch.zeros", side_effect=AssertionError("cache reallocation")):
            second = comm.fused_experts(other_layer, record_dispatch_event=False)
    bootstrap.assert_called_once()
    assert operator.call_count == 2  # An empty source rank still owns remote experts.
    sent1, sent2 = [call.args[0] for call in operator.call_args_list]
    assert sent1.weights is payload.weights and sent2.weights is other_layer.weights
    assert sent1.hidden_states is sent2.hidden_states is cached[0]
    assert sent1.topk_ids is sent2.topk_ids is cached[1]
    assert sent1.topk_weights is sent2.topk_weights is cached[2]
    assert sent1.routing.log2phy is sent2.routing.log2phy is None
    assert sent1.topk_ids[0].tolist() == list(range(24, 32))
    assert all(t.is_contiguous() for t in cached)
    assert torch.count_nonzero(cached[0]) == torch.count_nonzero(cached[2]) == 0
    assert first.routed_out is second.routed_out is operator.return_value[0]
    assert cached[0].data_ptr() != first.routed_out.data_ptr()


@pytest.mark.parametrize("change", ["rows", "hidden", "topk", "rank", "world", "experts"])
def test_zercmoe_full_padding_cache_rebuilds_when_its_geometry_changes(change):
    comm, payload = _make_zercmoe_w8a8_case()
    comm.prepare_finalize.all_tokens_padding = True
    payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.zeros(2, dtype=torch.bool)))
    first = comm._get_w8a8_padding_inputs(payload)
    assert comm._get_w8a8_padding_inputs(payload) is first
    if change == "rows":
        payload = replace(payload, hidden_states=payload.hidden_states[:1], topk_ids=payload.topk_ids[:1])
    elif change == "hidden":
        payload = replace(payload, hidden_states=payload.hidden_states[:, :16])
    elif change == "topk":
        payload = replace(payload, topk_ids=payload.topk_ids[:, :4])
    elif change == "rank":
        comm.token_dispatcher.ep_rank_id = 19
    elif change == "world":
        comm.token_dispatcher.ep_world_size = 16
    else:
        comm.moe_config.num_experts = 512
    new_ids = comm._get_w8a8_padding_ids(payload)
    second = comm._get_w8a8_padding_inputs(payload)
    assert second is not first
    assert comm._get_w8a8_padding_inputs(payload) is second
    assert second[0].shape == payload.hidden_states.shape
    assert second[1].shape == second[2].shape == payload.topk_ids.shape
    torch.testing.assert_close(second[1][0], new_ids)


def test_zercmoe_full_padding_cache_is_not_used_for_real_mixed_or_unknown_tokens():
    comm, payload = _make_zercmoe_w8a8_case()
    payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.zeros(2, dtype=torch.bool)))
    comm.prepare_finalize.all_tokens_padding = True
    cached = comm._get_w8a8_padding_inputs(payload)
    for valid in (False, True, False):
        comm.prepare_finalize.all_tokens_padding = False
        comm.prepare_finalize.all_tokens_valid = valid
        assert comm._get_w8a8_padding_inputs(payload) is None
        assert comm._w8a8_padding_inputs is None
    comm.prepare_finalize.all_tokens_padding = True
    assert comm._get_w8a8_padding_inputs(payload) is not cached


def test_zercmoe_full_padding_cache_rebuilds_on_device_or_stream_changes_without_sync():
    comm, payload = _make_zercmoe_w8a8_case()
    comm.prepare_finalize.all_tokens_padding = True
    payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.zeros(2, dtype=torch.bool)))
    cpu_entry = comm._get_w8a8_padding_inputs(payload)
    # Device/stream identity is metadata here; all allocations stay on CPU.
    fake_npu_input = SimpleNamespace(ndim=2, shape=(2, 32), dtype=torch.bfloat16, device=torch.device("npu:0"))
    payload = replace(payload, hidden_states=fake_npu_input)
    stream = SimpleNamespace(npu_stream=17)
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return zeros(*args, **kwargs)

    with (
        patch("torch.npu.current_stream", return_value=stream),
        patch("torch.npu.synchronize", create=True, side_effect=AssertionError("cache must not synchronize")),
        patch("torch.zeros", side_effect=cpu_zeros),
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.make_w8a8_padding_ids",
            return_value=torch.arange(24, 32, dtype=torch.int32),
        ) as make_ids,
    ):
        first = comm._get_w8a8_padding_inputs(payload)
        assert first is not cpu_entry
        assert comm._get_w8a8_padding_inputs(payload) is first
        stream.npu_stream = 23
        second = comm._get_w8a8_padding_inputs(payload)
        assert second is not first
        assert comm._get_w8a8_padding_inputs(payload) is second
    assert make_ids.call_count == 2  # Each stream initializes its own physical IDs.


def test_zercmoe_full_padding_mixed_valid_batch_transitions_preserve_real_tokens():
    comm, payload = _make_zercmoe_w8a8_case()
    payload.hidden_states.fill_(0.25)
    physical_ids = torch.arange(24, 32, dtype=torch.int32)
    operator = MagicMock(return_value=(torch.zeros_like(payload.hidden_states), torch.zeros(8, dtype=torch.int32)))
    cached = None
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=object()),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator),
    ):
        for mask, all_valid, all_padding in (
            ([False, False], False, True),
            ([True, False], False, False),
            ([True, True], True, False),
            ([False, False], False, True),
        ):
            current = replace(payload, routing=replace(payload.routing, mc2_mask=torch.tensor(mask)))
            reference = prepare_w8a8_input(current, padding_ids=physical_ids)
            comm.prepare_finalize.all_tokens_valid = all_valid
            comm.prepare_finalize.all_tokens_padding = all_padding
            with patch("torch.where", wraps=torch.where) as where:
                comm.fused_experts(current, record_dispatch_event=False)
            sent = operator.call_args.args[0]
            assert where.call_count == (3 if not all_valid and not all_padding else 0)
            for actual, expected in (
                (sent.hidden_states, reference.hidden_states),
                (sent.topk_ids, reference.topk_ids),
                (sent.topk_weights, reference.topk_weights),
            ):
                torch.testing.assert_close(actual, expected)
            if all_padding:
                if cached is None:
                    cached = comm._w8a8_padding_inputs
                assert sent.hidden_states is cached[0]
            else:
                assert cached is None or sent.hidden_states is not cached[0]
                assert comm._w8a8_padding_inputs is None
                cached = None
    assert operator.call_count == 4
    torch.testing.assert_close(payload.hidden_states, torch.full_like(payload.hidden_states, 0.25))


def test_zercmoe_a3_prepared_payload_is_forwarded_without_a_second_wrapper():
    comm, payload = _make_zercmoe_w8a8_case()
    prepared = replace(payload, topk_ids=payload.topk_ids.int(), topk_weights=payload.topk_weights.float())
    operator = MagicMock(return_value=(torch.zeros_like(payload.hidden_states), torch.zeros(8, dtype=torch.int32)))
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.prepare_w8a8_input", return_value=prepared),
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=object()),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator),
    ):
        comm.fused_experts(payload, record_dispatch_event=False)
    assert operator.call_args.args[0] is prepared


@pytest.mark.parametrize(
    "is_request,profile,draft", [(False, False, False), (None, False, False), (True, True, False), (True, False, True)]
)
def test_zercmoe_request_log_excludes_unidentified_and_dummy_forwards(is_request, profile, draft):
    comm = _make_zercmoe_comm()
    comm.token_dispatcher.ep_rank_id = 0
    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
            SimpleNamespace(request_forward=is_request, in_profile_run=profile, is_draft_model=draft),
        ),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger") as log,
    ):
        comm._log_request_launch(1)
    log.info.assert_not_called()
    assert not comm._logged_zercmoe_launch


def test_zercmoe_request_logs_are_bounded_by_reason_not_shape():
    comm = _make_zercmoe_comm()
    comm.token_dispatcher.ep_rank_id = 0
    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
            SimpleNamespace(request_forward=True, in_profile_run=False, is_draft_model=False),
        ),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger") as log,
    ):
        for rows in (1, 2, 4, 128):
            comm._log_request_launch(rows)
            comm.log_request_fallback("MC2", graph=True, num_tokens=rows, capacity=256)
            comm.log_request_fallback("MC2", graph=False, num_tokens=rows + 256, capacity=256)
    assert log.info.call_count == 3


def test_zercmoe_request_launch_is_logged_only_after_successful_submission():
    comm, payload = _make_zercmoe_w8a8_case()
    comm.token_dispatcher.ep_rank_id = 0
    operator = MagicMock(side_effect=RuntimeError("launcher failure"))
    with (
        patch.object(comm, "_ensure_zercmoe_loaded"),
        patch.object(comm, "_init_zerc_symm_buffer", return_value=object()),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_w8a8_operator", return_value=operator),
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
            SimpleNamespace(request_forward=True, in_profile_run=False, is_draft_model=False),
        ),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger") as log,
    ):
        with pytest.raises(RuntimeError, match="launcher failure"):
            comm.fused_experts(payload, record_dispatch_event=False)
        log.info.assert_not_called()
        operator.side_effect = None
        operator.return_value = (torch.zeros_like(payload.hidden_states), torch.zeros(8, dtype=torch.int32))
        for _ in range(2):
            comm.fused_experts(payload, record_dispatch_event=False)
        log.info.assert_called_once()


@pytest.mark.parametrize("all_empty", [False, True])
def test_zercmoe_empty_input_skips_only_for_synchronized_empty_ep_group(all_empty):
    comm, payload = _make_zercmoe_w8a8_case()
    comm.prepare_finalize.all_ranks_empty = all_empty
    payload = replace(payload, hidden_states=payload.hidden_states[:0])
    with (
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        patch.object(comm, "_init_zerc_symm_buffer") as bootstrap,
    ):
        if all_empty:
            result = comm.fused_experts(payload)
            assert result.routed_out.shape == (0, 32)
            assert result.routed_out.dtype == torch.bfloat16
        else:
            with pytest.raises(ValueError, match="all-EP-empty"):
                comm.fused_experts(payload)
    load.assert_not_called()
    bootstrap.assert_not_called()


def test_zercmoe_w8a8_does_not_reuse_bf16_heap():
    comm, payload = _make_zercmoe_w8a8_case()
    comm.zerc_symm_buffer = object()
    comm._zerc_buffer_weight_dtype = torch.bfloat16
    with (
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        pytest.raises(ValueError, match="different weight dtypes"),
    ):
        comm.fused_experts(payload)
    load.assert_not_called()


def test_zercmoe_w8a8_does_not_reuse_heap_across_input_dtypes():
    comm, payload = _make_zercmoe_w8a8_case()
    comm.zerc_symm_buffer = object()
    comm._zerc_buffer_weight_dtype = torch.int8
    comm._zerc_buffer_input_dtype = torch.float16
    with (
        patch.object(comm, "_ensure_zercmoe_loaded") as load,
        pytest.raises(ValueError, match="activation dtypes"),
    ):
        comm.fused_experts(payload)
    load.assert_not_called()


class TestMoECommMethod(TestBase):
    def setUp(self):
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant = False
        self.mock_ascend_config.enable_fused_mc2 = False
        self.mock_ascend_config.mega_moe_max_tokens = 65536
        self.mock_ascend_config.scheduler_config.recompute_scheduler_enable = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_module = patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_forward_context = patch(
            "vllm_ascend.ascend_forward_context.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self._patch_get_ascend_config_module.start()
        self._patch_get_ascend_config_forward_context.start()
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.pcp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.global_redundant_expert_num = 0

    def _make_fused_mc2_comm_for_buffer_init(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.moe_config = self.moe_config
        comm_impl.moe_config.hidden_dim = 16
        comm_impl.moe_config.intermediate_size_per_partition = 32
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 0
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 128
        comm_impl.token_dispatcher.ep_world_size = 8
        comm_impl.token_dispatcher.ep_rank_id = 0
        comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value="symm_buffer")
        comm_impl.mega_moe_symm_buffer = None
        return comm_impl

    def tearDown(self):
        self._patch_get_ascend_config.stop()
        self._patch_get_ascend_config_module.stop()
        self._patch_get_ascend_config_forward_context.stop()

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning_once")
    def test_mega_moe_symm_buffer_uses_mega_moe_max_tokens(self, mock_warning_once, mock_get_mc2_group):
        self.mock_ascend_config.mega_moe_max_tokens = 32768
        mock_mc2_group = MagicMock()
        mock_mc2_group.device_group = "mc2_group"
        mock_get_mc2_group.return_value = mock_mc2_group
        comm_impl = self._make_fused_mc2_comm_for_buffer_init()

        comm_impl._init_mega_moe_symm_buffer(is_decode_only_node=False)

        comm_impl.get_symm_buffer_for_mega_moe.assert_called_once()
        call_args = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call_args.args[:4], ("mc2_group", 8, 128, 2))
        self.assertEqual(call_args.kwargs["max_recv_token_num"], 32768)
        mock_warning_once.assert_called_once()
        self.assertIn("mega_moe_max_tokens", mock_warning_once.call_args.args[0])

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning_once")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group")
    def test_mega_moe_symm_buffer_uses_safe_capacity_for_d_node(self, mock_get_mc2_group, mock_warning_once):
        self.mock_ascend_config.mega_moe_max_tokens = 32768
        mock_mc2_group = MagicMock()
        mock_mc2_group.device_group = "mc2_group"
        mock_get_mc2_group.return_value = mock_mc2_group
        comm_impl = self._make_fused_mc2_comm_for_buffer_init()

        comm_impl._init_mega_moe_symm_buffer(is_decode_only_node=True)

        call_args = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call_args.kwargs["max_recv_token_num"], 1024)
        mock_warning_once.assert_not_called()

    def test_fused_mc2_situ_falls_back_to_decomposed_mc2_pipeline(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        fused_input = MoEFusedExpertsInput(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            weights=MoEWeights(
                w1=[torch.randn(1, 4, 4)],
                w2=[torch.randn(1, 2, 4)],
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=False,
            ),
            quant=MoEQuantParams(),
            activation=SituActivationConfig(beta=4.0, linear_beta=25.0),
        )
        expected = object()

        with patch.object(MoECommMethod, "fused_experts", return_value=expected) as mock_decomposed:
            result = comm_impl.fused_experts(fused_input)

        self.assertIs(result, expected)
        mock_decomposed.assert_called_once_with(fused_input)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_all_gather_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2")
    def test_mc2_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "mc2"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=torch.tensor([1, 0, 1, 0]),
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = MC2CommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AlltoAllCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        _ = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp")
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_method(
        self, mock_unified_apply_mlp, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        dispatch_topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(6, 8),
            group_list=torch.tensor([2, 2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=dispatch_topk_weights,
                expanded_row_idx=torch.arange(8, dtype=torch.int32),
                restore_shape=torch.Size([4, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(4, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        # Mock unified_apply_mlp returns (tensor, event) tuple
        mock_unified_apply_mlp.return_value = (torch.randn(6, 8), MagicMock())

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test fused_experts method
        hidden_states = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        topk_weights = dispatch_topk_weights
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])

        # Make sure tensors are contiguous and have correct strides
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        result = comm_impl.fused_experts(
            fused_experts_input=MoEFusedExpertsInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                weights=MoEWeights(
                    w1=[w1],
                    w2=[w2],
                ),
                routing=MoERoutingParams(
                    expert_map=None,
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                activation="silu",
                need_trans=False,
                dynamic_eplb=False,
                quant=MoEQuantParams(),
            )
        )

        # Verify result shape
        self.assertEqual(result.routed_out.shape, (4, 8))

        # Verify token_dispatch was called
        mock_td_instance.token_dispatch.assert_called_once()

        # Verify unified_apply_mlp was called
        mock_unified_apply_mlp.assert_called_once()
        mlp_compute_input = mock_unified_apply_mlp.call_args.kwargs["mlp_compute_input"]
        self.assertFalse(mlp_compute_input.fusion)
        self.assertFalse(mlp_compute_input.quant.is_mxfp)

        # Verify token_combine was called
        mock_td_instance.token_combine.assert_called_once_with(
            hidden_states=mock_unified_apply_mlp.return_value[0],
            combine_metadata=mock_td_instance.token_dispatch.return_value.combine_metadata,
        )


def test_zercmoe_close_synchronizes_then_destroys_once():
    comm = _make_zercmoe_comm()
    calls = []
    comm.zerc_symm_buffer = SimpleNamespace(destroy=lambda: calls.append("destroy"))
    with patch("torch.npu.synchronize", side_effect=lambda: calls.append("synchronize"), create=True):
        comm.close()
        comm.close()
    assert calls == ["synchronize", "destroy"]
    assert comm.zerc_symm_buffer is None


def test_zercmoe_close_does_nothing_without_a_heap():
    comm = _make_zercmoe_comm()
    with patch("torch.npu.synchronize", create=True) as synchronize:
        comm.close()
    synchronize.assert_not_called()
