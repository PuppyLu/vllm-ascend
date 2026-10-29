from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_dispatch_v2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_moe_quant_mismatch", None)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            enable_zerc_moe=0,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )


def _make_vllm_config(
    *,
    enable_expert_parallel: bool = True,
    world_size: int = 8,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    num_experts: int = 128,
    quant_type: str | None = None,
    top_k_experts: int = 1,
    num_experts_per_tok: int | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int = 0,
    max_num_batched_tokens: int = 4096,
    hidden_size: int = 2048,
    kv_connector: str | None = None,
    kv_role: str | None = None,
    recompute_scheduler_enable: bool = False,
):
    hf_text_config_attrs: dict[str, object] = {
        "top_k_experts": top_k_experts,
        "moe_intermediate_size": 2048,
    }
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok
    hf_text_config_attrs["hidden_size"] = hidden_size

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(**hf_text_config_attrs),
        get_num_experts=lambda: num_experts,
        dtype=torch.bfloat16,
        is_quantized=quant_type is not None,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        world_size_across_dp=world_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        max_cudagraph_capture_size=max_cudagraph_capture_size,
    )
    kv_transfer_config = (
        SimpleNamespace(kv_connector=kv_connector, kv_role=kv_role)
        if kv_connector is not None or kv_role is not None
        else None
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            recompute_scheduler_enable=recompute_scheduler_enable,
        ),
        kv_transfer_config=kv_transfer_config,
    )


def _patch_select_moe_comm_method_deps(
    monkeypatch,
    *,
    device_type,
    capacity: int = 128,
    ep_world_size: int = 8,
    enable_fused_mc2: int = 0,
    enable_prefill_mc2: int = 0,
    is_moe: bool = True,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: False)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=enable_fused_mc2, enable_prefill_mc2=enable_prefill_mc2, enable_zerc_moe=0
        ),
    )


def test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank():
    vllm_config = _make_vllm_config(tensor_parallel_size=6)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=200, uniform_decode_query_len=3)

    assert afc.get_mc2_tokens_capacity() == 600


def test_set_mc2_tokens_capacity_with_cudagraph_uses_capture_size_and_aligns():
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 264


def test_set_mc2_tokens_capacity_prefill_mc2_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_prefill_mc2=True, enable_fused_mc2=0, enable_zerc_moe=0),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=8, max_num_batched_tokens=513)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 520


def test_set_mc2_tokens_capacity_decode_only_uses_cudagraph_capture_size(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            enable_zerc_moe=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 264


def test_set_mc2_tokens_capacity_decode_only_without_cudagraph_uses_decode_shape(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            enable_zerc_moe=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=4,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=2)

    assert afc.get_mc2_tokens_capacity() == 32


def test_set_mc2_tokens_capacity_disable_recompute_decode_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=1,
            enable_zerc_moe=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=False),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
        recompute_scheduler_enable=False,
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 8192


def test_select_moe_comm_method_returns_none_for_non_moe(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        is_moe=False,
    )

    assert afc.select_moe_comm_method(16, _make_vllm_config()) is None


@pytest.mark.parametrize(
    ("enable_expert_parallel", "ep_world_size"),
    [
        (False, 8),
        (True, 1),
    ],
)
def test_select_moe_comm_method_uses_allgather_without_effective_expert_parallel(
    monkeypatch,
    enable_expert_parallel,
    ep_world_size,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        ep_world_size=ep_world_size,
    )
    vllm_config = _make_vllm_config(enable_expert_parallel=enable_expert_parallel)

    assert afc.select_moe_comm_method(16, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLGATHER),
    ],
)
def test_select_moe_comm_method_a2_uses_mc2_within_capacity(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=16,
    )
    vllm_config = _make_vllm_config(world_size=16, num_experts=128)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


def test_select_moe_comm_method_a2_uses_allgather_for_more_than_512_experts(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=64,
    )
    vllm_config = _make_vllm_config(world_size=64, num_experts=896)

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
        (128, 128, MoECommType.MC2),
        (4097, 8, MoECommType.FUSED_MC2),
        (4097, 128, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_enable_fused_mc2_mode_1(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_without_fused_mc2(
    monkeypatch,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        enable_prefill_mc2=1,
    )
    vllm_config = _make_vllm_config()

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a16(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a16")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w8a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w8a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_mc2_invalid_hidden_size(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=0,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8", hidden_size=512)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "world_size", "top_k_experts", "expected"),
    [
        (128, 4, 2, MoECommType.MC2),
        (129, 2, 4, MoECommType.ALLGATHER),
        (129, 8, 4, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a5(monkeypatch, num_tokens, world_size, top_k_experts, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
    )
    vllm_config = _make_vllm_config(world_size=world_size, top_k_experts=top_k_experts)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("is_draft_model", "moe_quant_mismatch", "num_tokens", "expected"),
    [
        (True, True, 128, MoECommType.MC2),
        (True, True, 129, MoECommType.ALLTOALL),
        (False, True, 128, MoECommType.FUSED_MC2),
        (True, False, 128, MoECommType.FUSED_MC2),
    ],
)
def test_select_a3_draft_quant_mismatch(
    monkeypatch,
    is_draft_model,
    moe_quant_mismatch,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=256,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "get_dispatch_v2_tokens_capacity", lambda: 128)
    monkeypatch.setattr(afc, "_moe_quant_mismatch", moe_quant_mismatch)

    vllm_config = _make_vllm_config()

    assert afc.select_moe_comm_method(num_tokens, vllm_config, is_draft_model=is_draft_model) == expected


@pytest.mark.parametrize(
    ("num_tokens", "is_decode", "is_draft_model", "expected"),
    [
        (4096, False, False, MoECommType.ZERC_MOE),
        (65, False, False, MoECommType.ZERC_MOE),
        (64, False, False, MoECommType.ALLGATHER),
        (1, False, False, MoECommType.ALLGATHER),
        (4096, True, False, MoECommType.ALLGATHER),
        (4096, False, True, MoECommType.ALLGATHER),
        (4097, False, False, MoECommType.ALLGATHER),
    ],
)
def test_select_a2_zercmoe_priority(monkeypatch, num_tokens, is_decode, is_draft_model, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=4096,
        ep_world_size=4,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)

    vllm_config = _make_vllm_config(num_experts=256)

    assert (
        afc.select_moe_comm_method(num_tokens, vllm_config, is_draft_model=is_draft_model, is_decode=is_decode)
        == expected
    )


def test_select_a2_zercmoe_gate_off_keeps_legacy(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=4096,
        ep_world_size=4,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: False)

    vllm_config = _make_vllm_config(num_experts=256)

    # With the gate off, the legacy A2 rule applies: experts_per_device
    # (256 // 4 = 64) > 24 -> ALLGATHER.
    assert afc.select_moe_comm_method(4096, vllm_config) == MoECommType.ALLGATHER


def test_set_mc2_tokens_capacity_a2_zercmoe_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=1,
            enable_zerc_moe=1,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=4, max_num_batched_tokens=16384)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    # max_num_batched_tokens/tp = 4096, capped by the mega-moe per-rank limit.
    assert afc.get_mc2_tokens_capacity() == 16384


@pytest.mark.parametrize(
    "num_tokens,is_decode,is_draft_model,is_graph_mode,expected",
    [
        (None, False, False, False, MoECommType.ZERC_MOE),
        (0, False, False, False, MoECommType.ZERC_MOE),
        (1, False, False, False, MoECommType.ZERC_MOE),
        (63, False, False, False, MoECommType.ZERC_MOE),
        (64, False, False, False, MoECommType.ZERC_MOE),
        (65, False, False, False, MoECommType.ZERC_MOE),
        (4096, False, False, False, MoECommType.ZERC_MOE),
        (4097, False, False, False, MoECommType.FUSED_MC2),
        (1, True, False, False, MoECommType.ZERC_MOE),
        (128, True, False, False, MoECommType.ZERC_MOE),
        (64, False, True, False, MoECommType.FUSED_MC2),
        (64, False, False, True, MoECommType.FUSED_MC2),
    ],
)
def test_select_a3_zercmoe_routing_requires_adapter_gate(
    monkeypatch, num_tokens, is_decode, is_draft_model, is_graph_mode, expected
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=32,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    config = _make_vllm_config(
        world_size=32,
        tensor_parallel_size=16,
        data_parallel_size=2,
        kv_role="kv_producer",
        max_num_batched_tokens=4096,
    )
    assert (
        afc.select_moe_comm_method(
            num_tokens,
            config,
            is_decode=is_decode,
            is_draft_model=is_draft_model,
            is_graph_mode=is_graph_mode,
        )
        == expected
    )


@pytest.mark.parametrize("device_type", [afc.AscendDeviceType.A2, afc.AscendDeviceType.A3])
@pytest.mark.parametrize("recompute", [False, True])
def test_zercmoe_pd_consumer_eager_support_is_device_specific(monkeypatch, device_type, recompute):
    _patch_select_moe_comm_method_deps(
        monkeypatch, device_type=device_type, capacity=4096, ep_world_size=32, enable_fused_mc2=1
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1, scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute)
        ),
    )
    config = _make_vllm_config(
        world_size=32,
        tensor_parallel_size=4,
        data_parallel_size=8,
        kv_connector="SfaRemoteD2HConnector",
        kv_role="kv_consumer",
    )
    assert afc._is_decode_only_node(config) is recompute
    expected = MoECommType.MC2 if device_type == afc.AscendDeviceType.A2 else MoECommType.ZERC_MOE
    assert afc.select_moe_comm_method(1, config, is_decode=True) == expected


@pytest.mark.parametrize(
    ("kv_connector", "kv_role", "expected"),
    [
        (None, None, MoECommType.ZERC_MOE),
        ("MultiConnector", "kv_producer", MoECommType.ZERC_MOE),
        ("MultiConnector", "kv_both", MoECommType.ZERC_MOE),
        ("DecodeBenchConnector", "kv_both", MoECommType.ALLGATHER),
    ],
)
def test_zercmoe_pd_role_selection(monkeypatch, kv_connector, kv_role, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch, device_type=afc.AscendDeviceType.A2, capacity=4096, ep_world_size=4, enable_fused_mc2=1
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    vllm_config = _make_vllm_config(num_experts=256, kv_connector=kv_connector, kv_role=kv_role)

    assert afc.select_moe_comm_method(128, vllm_config) == expected


@pytest.mark.parametrize(
    ("device_type", "quant_type", "dp_size", "expected"),
    [
        (afc.AscendDeviceType.A2, None, 1, True),
        (afc.AscendDeviceType.A2, "w8a8", 1, False),
        (afc.AscendDeviceType.A2, None, 2, False),
        (afc.AscendDeviceType.A3, None, 1, False),
        (afc.AscendDeviceType.A3, "w8a8", 2, True),
        (afc.AscendDeviceType.A3, "w8a8", 8, True),
    ],
)
def test_zercmoe_config_gate_preserves_a2_and_enables_a3_w8a8(monkeypatch, device_type, quant_type, dp_size, expected):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=4, data_parallel_size=dp_size, quant_type=quant_type)

    assert afc.is_zercmoe_active_by_config(vllm_config) is expected


@pytest.mark.parametrize(
    "role,enabled,dynamic_eplb,expected",
    [
        ("kv_producer", 1, False, True),
        ("kv_consumer", 1, False, True),
        ("kv_producer", 0, False, False),
        ("kv_producer", 1, True, False),
    ],
)
@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_zercmoe_w8a8_preparation_requires_bf16_but_not_fused_mc2(
    monkeypatch, role, enabled, dynamic_eplb, expected, model_dtype
):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_zerc_moe=enabled, enable_fused_mc2=0, eplb_config=SimpleNamespace(dynamic_eplb=dynamic_eplb)
        ),
    )
    config = _make_vllm_config(tensor_parallel_size=16, data_parallel_size=2, quant_type="w8a8", kv_role=role)
    config.model_config.dtype = model_dtype
    monkeypatch.setattr(afc, "enable_sp", lambda _: True)
    assert afc.should_prepare_zercmoe_w8a8_weights(config) is (expected and model_dtype == torch.bfloat16)
    assert afc.is_zercmoe_active_by_config(config) is (expected and model_dtype == torch.bfloat16)


@pytest.mark.parametrize("pp,pcp,expected", [(1, 1, True), (2, 1, False), (1, 2, False)])
def test_zercmoe_a3_runtime_requires_single_pp_and_pcp(monkeypatch, pp, pcp, expected):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    config = _make_vllm_config(pipeline_parallel_size=pp, quant_type="w8a8")
    config.parallel_config.prefill_context_parallel_size = pcp

    assert afc.should_prepare_zercmoe_w8a8_weights(config) is True
    assert afc.is_zercmoe_active_by_config(config) is expected


@pytest.mark.parametrize("device_type", [afc.AscendDeviceType.A2])
def test_zercmoe_a2_d_role_does_not_probe_either_backend(monkeypatch, device_type):
    from vllm_ascend.ops.fused_moe import comm_utils

    def unexpected_import():
        pytest.fail("D worker must not probe either ZercMoE backend")

    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    monkeypatch.setattr(comm_utils, "zercmoe_lib_available", unexpected_import)
    monkeypatch.setattr(comm_utils, "a3_zercmoe_lib_available", unexpected_import)
    monkeypatch.setattr(afc, "get_ep_group", unexpected_import)
    config = _make_vllm_config(
        tensor_parallel_size=4,
        quant_type="w8a8" if device_type == afc.AscendDeviceType.A3 else None,
        kv_role="kv_consumer",
    )
    assert afc.should_prepare_zercmoe_w8a8_weights(config) is False
    assert afc.use_cann_zercmoe(config) is False


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("backend_available", [False, True])
def test_zercmoe_a3_dp_tp_sp_uses_internal_wheel(monkeypatch, sp, backend_available):
    from vllm_ascend.ops.fused_moe import comm_utils

    ascend_config = SimpleNamespace(
        enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)
    )
    cann_backend = Mock(return_value=backend_available)
    external_backend = Mock(side_effect=AssertionError("A3 must use the version-checked 1.5 loader"))
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "enable_sp", lambda _: sp)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=32))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(comm_utils, "a3_zercmoe_lib_available", cann_backend)
    monkeypatch.setattr(comm_utils, "zercmoe_lib_available", external_backend)
    config = _make_vllm_config(
        world_size=32, tensor_parallel_size=16, data_parallel_size=2, quant_type="w8a8", kv_role="kv_producer"
    )
    assert afc.is_zercmoe_active_by_config(config) is True
    assert afc.use_cann_zercmoe(config) is backend_available
    cann_backend.assert_called_once_with()
    external_backend.assert_not_called()

    ascend_config.eplb_config.dynamic_eplb = True
    assert afc.use_cann_zercmoe(config) is False
    cann_backend.assert_called_once_with()


@pytest.mark.parametrize(
    "world_size,num_experts,expected",
    [(1, 128, False), (2, 128, True), (32, 128, True), (128, 256, True), (129, 258, True), (32, 130, False)],
)
def test_zercmoe_a3_requires_supported_ep_group(monkeypatch, world_size, num_experts, expected):
    from vllm_ascend.ops.fused_moe import comm_utils

    cann_backend = Mock(return_value=True)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=world_size))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    monkeypatch.setattr(comm_utils, "a3_zercmoe_lib_available", cann_backend)
    monkeypatch.setattr(
        comm_utils,
        "zercmoe_lib_available",
        Mock(side_effect=AssertionError("A3 must use the version-checked 1.5 loader")),
    )
    config = _make_vllm_config(quant_type="w8a8", num_experts=num_experts)

    assert afc.use_cann_zercmoe(config) is expected
    assert cann_backend.call_count == int(expected)


@pytest.mark.parametrize("world_size,sp,expected", [(64, False, True), (65, False, False), (64, True, False)])
def test_zercmoe_a2_retains_external_backend_and_ep_sp_limits(monkeypatch, world_size, sp, expected):
    from vllm_ascend.ops.fused_moe import comm_utils

    external_backend = Mock(return_value=True)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "enable_sp", lambda _: sp)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=world_size))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    monkeypatch.setattr(comm_utils, "zercmoe_lib_available", external_backend)
    monkeypatch.setattr(
        comm_utils, "a3_zercmoe_lib_available", Mock(side_effect=AssertionError("A2 must retain the external wheel"))
    )
    config = _make_vllm_config(tensor_parallel_size=64)

    assert afc.use_cann_zercmoe(config) is expected
    assert external_backend.call_count == int(expected)


@pytest.mark.parametrize(
    "model_dtype,hidden_size,intermediate_size,expected",
    [
        (torch.bfloat16, 6144, 4096, True),
        (torch.float16, 6144, 4096, False),
        (torch.float32, 6144, 4096, False),
        (torch.bfloat16, 9856, 4096, True),
        (torch.bfloat16, 6144, 4128, True),
        (torch.bfloat16, 6160, 4096, True),
        (torch.bfloat16, 6161, 4096, False),
        (torch.bfloat16, 6144, 4104, False),
    ],
)
def test_zercmoe_a3_preparation_uses_documented_bf16_shape_gate(
    monkeypatch, model_dtype, hidden_size, intermediate_size, expected
):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, enable_fused_mc2=1, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    config = _make_vllm_config(quant_type="w8a8", hidden_size=hidden_size)
    config.model_config.dtype = model_dtype
    config.model_config.hf_text_config.moe_intermediate_size = intermediate_size

    assert afc.should_prepare_zercmoe_w8a8_weights(config) is expected
    assert afc.is_zercmoe_active_by_config(config) is expected


def test_zercmoe_mask_capacity_covers_tp_rounding(monkeypatch):
    monkeypatch.setattr(afc, "_reserved_mc2_mask", None)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    config = _make_vllm_config(tensor_parallel_size=16, max_num_batched_tokens=1000)
    afc.set_mc2_mask(config, "cpu")
    assert afc.get_mc2_mask().shape == (1008,)
    assert afc.get_mc2_mask().dtype == torch.bool


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("local_tokens", [0, 65])
@pytest.mark.parametrize("request_forward", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_zercmoe_selection_and_padding_share_synchronized_dp_metadata(
    monkeypatch, sp, local_tokens, request_forward, graph
):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ops.fused_moe import moe_comm_method

    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([0, 65])))

    @contextmanager
    def base_context(**kwargs):
        yield

    monkeypatch.setattr(afc, "set_forward_context", base_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "has_layer_idx", lambda _: False)
    monkeypatch.setattr(afc, "enable_sp", lambda _: sp)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", torch.empty(80, dtype=torch.bool))
    comm = Mock()
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda _: comm)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_zerc_moe=1))
    select = Mock(return_value=MoECommType.FUSED_MC2 if graph else MoECommType.ZERC_MOE)
    monkeypatch.setattr(afc, "select_moe_comm_method", select)
    config = _make_vllm_config(tensor_parallel_size=16, data_parallel_size=2)
    with afc.set_ascend_forward_context(
        None,
        config,
        num_tokens=local_tokens,
        num_tokens_across_dp=torch.tensor([0, 0]),
        num_actual_tokens=local_tokens,
        aclgraph_runtime_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE,
        request_forward=request_forward,
    ):
        assert context.max_tokens_across_dp == 65
        assert context.padded_num_tokens == 80
        assert context.mc2_num_actual_tokens == local_tokens
        assert context.request_forward is request_forward
        torch.testing.assert_close(context.mc2_mask, torch.arange(80) < local_tokens)
    select.assert_called_once_with(65, config, is_draft_model=False, is_graph_mode=graph, is_decode=False)
    assert comm.log_request_fallback.call_count == int(request_forward and graph)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    "heap_mb,max_recv,invalid",
    [
        (None, None, False),
        (0, None, False),
        (1, None, True),
        (1023, None, True),
        (1024, None, False),
        (2048, None, False),
        (17999, None, False),
        (18000, None, False),
        (0, 8192, True),
    ],
)
def test_zercmoe_internal_heap_settings_validate_p_and_d(monkeypatch, role, heap_mb, max_recv, invalid):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_zerc_moe=1,
            enable_fused_mc2=1,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
            zercmoe_shmem_mb=heap_mb,
            zercmoe_max_recv_token_num=max_recv,
        ),
    )
    config = _make_vllm_config(quant_type="w8a8", kv_role=role)
    if invalid:
        with pytest.raises(ValueError, match="zercmoe|ZercMoE"):
            afc.should_prepare_zercmoe_w8a8_weights(config)
    else:
        assert afc.should_prepare_zercmoe_w8a8_weights(config) is True


@pytest.mark.parametrize(
    "kv_connector,kv_role",
    [
        (None, None),
        ("MultiConnector", "kv_producer"),
        ("MultiConnector", "kv_consumer"),
        ("MultiConnector", "kv_both"),
        ("DecodeBenchConnector", "kv_both"),
    ],
)
@pytest.mark.parametrize("num_tokens,is_decode", [(1, False), (1, True), (64, False), (128, True)])
@pytest.mark.parametrize("enable_fused_mc2", [0, 1])
def test_zercmoe_a3_all_eager_roles_and_sizes_are_independent_of_fused_mc2(
    monkeypatch, kv_connector, kv_role, num_tokens, is_decode, enable_fused_mc2
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=32,
        ep_world_size=16,
        enable_fused_mc2=enable_fused_mc2,
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    config = _make_vllm_config(
        world_size=16,
        tensor_parallel_size=4,
        data_parallel_size=4,
        kv_connector=kv_connector,
        kv_role=kv_role,
        max_num_batched_tokens=512,
    )
    assert afc.select_moe_comm_method(num_tokens, config, is_decode=is_decode) == MoECommType.ZERC_MOE


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("enable_fused_mc2", [0, 1])
def test_zercmoe_a3_prepares_and_probes_backend_for_every_role(monkeypatch, role, enable_fused_mc2):
    from vllm_ascend.ops.fused_moe import comm_utils

    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=16))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_zerc_moe=1, enable_fused_mc2=enable_fused_mc2, eplb_config=SimpleNamespace(dynamic_eplb=False)
        ),
    )
    backend = Mock(return_value=True)
    monkeypatch.setattr(comm_utils, "a3_zercmoe_lib_available", backend)
    monkeypatch.setattr(comm_utils, "zercmoe_lib_available", Mock(side_effect=AssertionError("wrong backend")))
    config = _make_vllm_config(quant_type="w8a8", tensor_parallel_size=4, kv_role=role)
    assert afc.should_prepare_zercmoe_w8a8_weights(config) is True
    assert afc.is_zercmoe_active_by_config(config) is True
    assert afc.use_cann_zercmoe(config) is True
    backend.assert_called_once_with()


@pytest.mark.parametrize("maximum,tp,expected", [(1, 16, 16), (17, 4, 20), (4097, 16, 4112), (100000, 2, 100000)])
def test_zercmoe_tokens_capacity_is_positive_scheduler_budget_rounded_to_tp(maximum, tp, expected):
    config = _make_vllm_config(max_num_batched_tokens=maximum, tensor_parallel_size=tp)
    assert afc.get_zercmoe_tokens_capacity(config) == expected


@pytest.mark.parametrize("maximum,tp", [(0, 4), (-1, 4), (16, 0), (16, -1)])
def test_zercmoe_tokens_capacity_rejects_nonpositive_dimensions(maximum, tp):
    config = _make_vllm_config(max_num_batched_tokens=maximum, tensor_parallel_size=tp)
    with pytest.raises(ValueError):
        afc.get_zercmoe_tokens_capacity(config)


def test_zercmoe_a3_capacity_does_not_inherit_mc2_or_4096_per_rank_limit(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch, device_type=afc.AscendDeviceType.A3, capacity=128, ep_world_size=16, enable_fused_mc2=0
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    config = _make_vllm_config(
        max_num_batched_tokens=100000, tensor_parallel_size=2, world_size=16, kv_role="kv_consumer"
    )
    assert afc.get_zercmoe_tokens_capacity(config) == 100000
    assert afc.select_moe_comm_method(90000, config, is_decode=True) == MoECommType.ZERC_MOE


@pytest.mark.parametrize(
    "fused,mc2_capacity,dispatch_capacity,ep,expected",
    [
        (1, 128, None, 16, MoECommType.FUSED_MC2),
        (0, 128, None, 16, MoECommType.MC2),
        (0, 32, None, 16, MoECommType.ALLTOALL),
        (0, 32, 128, 16, MoECommType.MC2),
        (1, 128, None, 64, MoECommType.MC2),
    ],
)
def test_zercmoe_a3_over_capacity_retains_original_fallback(
    monkeypatch, fused, mc2_capacity, dispatch_capacity, ep, expected
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=mc2_capacity,
        ep_world_size=ep,
        enable_fused_mc2=fused,
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "get_dispatch_v2_tokens_capacity", lambda: dispatch_capacity)
    config = _make_vllm_config(max_num_batched_tokens=32, tensor_parallel_size=4, world_size=ep)
    assert afc.select_moe_comm_method(64, config) == expected


@pytest.mark.parametrize("fused", [0, 1])
@pytest.mark.parametrize("prefill_mc2", [0, 1])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("capture_size", [0, 257])
def test_zercmoe_a3_toggle_does_not_change_existing_mc2_budget(monkeypatch, fused, prefill_mc2, role, capture_size):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=prefill_mc2,
            enable_fused_mc2=fused,
            enable_zerc_moe=1,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    config = _make_vllm_config(
        tensor_parallel_size=8,
        max_num_batched_tokens=100000,
        kv_role=role,
        cudagraph_capture_sizes=[capture_size] if capture_size else [],
        max_cudagraph_capture_size=capture_size,
    )
    budgets = []
    for enabled in (False, True):
        monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
        monkeypatch.setattr(afc, "_dispatch_v2_tokens_capacity", None)
        monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _, enabled=enabled: enabled)
        afc.set_mc2_tokens_capacity(config, max_num_reqs=16, uniform_decode_query_len=1)
        budgets.append((afc.get_mc2_tokens_capacity(), afc.get_dispatch_v2_tokens_capacity()))
    assert budgets[0] == budgets[1]
    assert afc.get_zercmoe_tokens_capacity(config) == 100000


def test_zercmoe_a3_rejects_shared_expert_mix_placement_before_backend_load(monkeypatch):
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_zerc_moe=1, mix_placement=True, eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    config = _make_vllm_config(quant_type="w8a8")
    with pytest.raises(ValueError, match="mix_placement"):
        afc.should_prepare_zercmoe_w8a8_weights(config)


def _zercmoe_single_weight_setup(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch, device_type=afc.AscendDeviceType.A3, capacity=128, ep_world_size=8, enable_fused_mc2=1
    )
    ascend_config = SimpleNamespace(
        enable_zerc_moe=1,
        enable_fused_mc2=1,
        enable_prefill_mc2=False,
        zercmoe_single_weight=True,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        xlite_graph_config=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    config = _make_vllm_config(quant_type="w8a8", tensor_parallel_size=8, max_num_batched_tokens=128)
    config.model_config.enforce_eager = True
    return config, ascend_config


def test_zercmoe_single_weight_accepts_supported_eager_configuration(monkeypatch):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    afc.validate_zercmoe_single_weight_config(config)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", None])
def test_zercmoe_a3_accepts_modelslim_without_hf_quantization_marker(monkeypatch, role):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    config.model_config.is_quantized = False
    config.model_config.quantization = "ascend"
    config.kv_transfer_config = SimpleNamespace(kv_role=role) if role else None
    assert afc.should_prepare_zercmoe_w8a8_weights(config)
    assert afc.is_zercmoe_active_by_config(config)
    afc.validate_zercmoe_single_weight_config(config)


@pytest.mark.parametrize(
    "quantized,quantization,expected",
    [
        (False, None, False),
        (False, "other", False),
        (False, "ascend", True),
        (True, None, True),
        (True, "other", True),
    ],
)
def test_zercmoe_a3_quantization_candidate_keeps_legacy_and_rejects_unselected(
    monkeypatch, quantized, quantization, expected
):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    config.model_config.is_quantized = quantized
    config.model_config.quantization = quantization
    assert afc.should_prepare_zercmoe_w8a8_weights(config) is expected


def test_zercmoe_single_weight_accepts_full_decode_only_graph(monkeypatch):
    from vllm.config import CUDAGraphMode

    config, ascend_config = _zercmoe_single_weight_setup(monkeypatch)
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    afc.validate_zercmoe_single_weight_config(config)
    config.parallel_config.use_ubatching = True
    with pytest.raises(ValueError, match="microbatching"):
        afc.validate_zercmoe_single_weight_config(config)
    config.parallel_config.use_ubatching = False
    config.parallel_config.enable_dbo = True
    with pytest.raises(ValueError, match="microbatching"):
        afc.validate_zercmoe_single_weight_config(config)
    config.parallel_config.enable_dbo = False
    ascend_config.multistream_overlap_shared_expert = True
    afc.validate_zercmoe_single_weight_config(config)


def test_zercmoe_decode_graph_forward_context_selects_zerc(monkeypatch):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ops.fused_moe import moe_comm_method

    config, ascend_config = _zercmoe_single_weight_setup(monkeypatch)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    context = SimpleNamespace(dp_metadata=None)
    comm = Mock()

    @contextmanager
    def base_context(**kwargs):
        yield

    monkeypatch.setattr(afc, "set_forward_context", base_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(afc, "has_layer_idx", lambda _: False)
    monkeypatch.setattr(afc, "enable_sp", lambda _: False)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", None)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda _: comm)
    with afc.set_ascend_forward_context(
        None,
        config,
        num_tokens=8,
        aclgraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=SimpleNamespace(uniform=True),
        request_forward=True,
    ):
        assert context.zercmoe_graph_execution
        assert context.moe_comm_type == MoECommType.ZERC_MOE
        comm.log_request_graph_selection.assert_called_once_with(8)


def test_zercmoe_single_weight_accepts_glm_mtp_with_matching_tp(monkeypatch):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    config.model_config.hf_config = SimpleNamespace(model_type="glm_moe_dsa")
    config.speculative_config = SimpleNamespace(method="mtp", draft_tensor_parallel_size=8)
    afc.validate_zercmoe_single_weight_config(config)
    assert afc.select_moe_comm_method(1, config, is_draft_model=True) != MoECommType.ZERC_MOE


@pytest.mark.parametrize("draft,missing", [(False, False), (True, False), (True, True)])
def test_zercmoe_single_weight_mtp_forward_uses_role_specific_comm(monkeypatch, draft, missing):
    from vllm_ascend.ops.fused_moe import moe_comm_method

    config, ascend_config = _zercmoe_single_weight_setup(monkeypatch)
    context = SimpleNamespace(dp_metadata=None)
    target_comm, draft_comm = object(), object()

    @contextmanager
    def base_context(**kwargs):
        yield

    monkeypatch.setattr(afc, "set_forward_context", base_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(afc, "is_drafter_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "has_layer_idx", lambda _: False)
    monkeypatch.setattr(afc, "enable_sp", lambda _: False)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", None)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(
        moe_comm_method,
        "get_moe_comm_method",
        lambda _, *, is_draft_model=False: (None if missing else draft_comm) if is_draft_model else target_comm,
    )
    if missing:
        with (
            pytest.raises(RuntimeError, match="draft MoE communication was not initialized"),
            afc.set_ascend_forward_context(None, config, num_tokens=1, is_draft_model=True),
        ):
            pytest.fail("draft forward must not run without its communication object")
        return
    with afc.set_ascend_forward_context(None, config, num_tokens=1, is_draft_model=draft):
        assert context.moe_comm_method is (draft_comm if draft else target_comm)
        assert (context.moe_comm_type == MoECommType.ZERC_MOE) is not draft


@pytest.mark.parametrize(
    "method,model_type,draft_tp", [("eagle", "glm_moe_dsa", 8), ("mtp", "other", 8), ("mtp", "glm_moe_dsa", 1)]
)
def test_zercmoe_single_weight_rejects_unsupported_mtp(monkeypatch, method, model_type, draft_tp):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    config.model_config.hf_config = SimpleNamespace(model_type=model_type)
    config.speculative_config = SimpleNamespace(method=method, draft_tensor_parallel_size=draft_tp)
    with pytest.raises(ValueError, match="A3 ZercMoE single-weight"):
        afc.validate_zercmoe_single_weight_config(config)


@pytest.mark.parametrize(
    "invalid",
    [
        "a2",
        "unquantized",
        "float16",
        "no_ep",
        "pp",
        "pcp",
        "dcp",
        "graph",
        "speculative",
        "sleep",
        "transfer",
        "xlite",
        "v2",
        "uva",
        "prefetch",
        "lora",
        "dynamic_eplb",
        "backend",
    ],
)
def test_zercmoe_single_weight_rejects_unsupported_configuration(monkeypatch, invalid):
    config, ascend_config = _zercmoe_single_weight_setup(monkeypatch)
    if invalid == "a2":
        monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    elif invalid == "unquantized":
        config.model_config.is_quantized = False
    elif invalid == "float16":
        config.model_config.dtype = torch.float16
    elif invalid == "no_ep":
        config.parallel_config.enable_expert_parallel = False
    elif invalid == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif invalid == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif invalid == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif invalid == "graph":
        config.model_config.enforce_eager = False
    elif invalid == "speculative":
        config.speculative_config = object()
    elif invalid == "sleep":
        config.model_config.enable_sleep_mode = True
    elif invalid == "transfer":
        config.weight_transfer_config = object()
    elif invalid == "xlite":
        ascend_config.xlite_graph_config.enabled = True
    elif invalid == "v2":
        config.use_v2_model_runner = True
    elif invalid == "uva":
        config.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=1))
    elif invalid == "prefetch":
        config.offload_config = SimpleNamespace(prefetch=SimpleNamespace(offload_group_size=4))
    elif invalid == "lora":
        config.lora_config = object()
    elif invalid == "dynamic_eplb":
        ascend_config.eplb_config.dynamic_eplb = True
    else:
        monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: False)
    with pytest.raises(ValueError, match="A3 ZercMoE single-weight"):
        afc.validate_zercmoe_single_weight_config(config)


def test_zercmoe_single_weight_disabled_does_not_change_default_configuration(monkeypatch):
    config, ascend_config = _zercmoe_single_weight_setup(monkeypatch)
    ascend_config.zercmoe_single_weight = False
    config.model_config.enforce_eager = False
    config.speculative_config = object()
    afc.validate_zercmoe_single_weight_config(config)


@pytest.mark.parametrize("num_tokens", [0, 1, 128])
def test_zercmoe_single_weight_selects_supported_batches(monkeypatch, num_tokens):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    assert afc.select_moe_comm_method(num_tokens, config) == MoECommType.ZERC_MOE


@pytest.mark.parametrize("invalid", ["graph", "capacity", "backend", "no_ep", "lora", "non_moe"])
def test_zercmoe_single_weight_rejects_fallback_before_execution(monkeypatch, invalid):
    config, _ = _zercmoe_single_weight_setup(monkeypatch)
    kwargs = {}
    num_tokens = 1
    if invalid == "graph":
        kwargs["is_graph_mode"] = True
    elif invalid == "capacity":
        num_tokens = 129
    elif invalid == "backend":
        monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: False)
    elif invalid == "no_ep":
        config.parallel_config.enable_expert_parallel = False
    elif invalid == "lora":
        config.lora_config = object()
    else:
        monkeypatch.setattr(afc, "is_moe_model", lambda _: False)
    with pytest.raises(RuntimeError, match="A3 ZercMoE single-weight"):
        afc.select_moe_comm_method(num_tokens, config, **kwargs)
