# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU interface chain tests; native math and collectives are explicit fakes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import experts_selector, fused_moe, moe_comm_method, prepare_finalize
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.ops.fused_moe.zercmoe_adapter import prepare_w8a8_layer
from vllm_ascend.quantization.methods import w8a8_dynamic
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _packed_layer(tag=1, experts=8):
    # TP16 + DP2 => EP32, 256 global experts => 8 local experts.
    return SimpleNamespace(
        w13_weight_zr=torch.full((experts * 32 * 32,), tag, dtype=torch.int8),
        w2_weight_zr=torch.full((experts * 16 * 32,), tag, dtype=torch.int8),
        w13_weight_scale_zr_packed=torch.full((experts, 32), tag, dtype=torch.int64),
        w2_weight_scale_zr_packed=torch.full((experts, 32), tag, dtype=torch.int64),
        swiglu_limit=0.0,
        zercmoe_needs_dispatch_event=False,
    )


@pytest.fixture
def early_chain(monkeypatch):
    """Construct real production classes, replacing only external boundaries."""
    state = SimpleNamespace(tp=16, ep=32, rank=3, ep_rank=19, expected_full=None, launches=[])
    config = SimpleNamespace(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=32,
        intermediate_size_per_partition=16,
        tp_group=SimpleNamespace(device_group=object()),
    )
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: state.tp)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: state.rank)
    pf = prepare_finalize.PrepareAndFinalizeWithZercMoE(config)
    comm = object.__new__(moe_comm_method.ZercMoECommImpl)
    comm.moe_config = config
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.ep_world_size = 32
    comm.token_dispatcher.ep_rank_id = state.ep_rank
    comm.token_dispatcher.max_num_tokens_per_rank = 128
    comm.prepare_finalize = pf
    comm._zercmoe = None
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

    ctx = SimpleNamespace(
        moe_comm_type=MoECommType.ZERC_MOE,
        moe_comm_method=comm,
        flash_comm_v1_enabled=False,
        in_profile_run=False,
        is_draft_model=False,
        request_forward=True,
        padded_num_tokens=32,
        mc2_num_actual_tokens=0,
        mc2_mask=torch.zeros(32, dtype=torch.bool),
    )
    for module in (fused_moe, w8a8_dynamic, moe_comm_method, prepare_finalize):
        monkeypatch.setattr(module, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(fused_moe, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(moe_comm_method, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(
        moe_comm_method, "get_mc2_group", lambda: SimpleNamespace(world_size=state.ep, rank_in_group=state.ep_rank)
    )
    monkeypatch.setattr(
        moe_comm_method,
        "get_ascend_config",
        lambda: SimpleNamespace(zercmoe_max_recv_token_num=None, zercmoe_shmem_mb=0, zercmoe_ipport=None),
    )
    monkeypatch.setenv("RANK", "19")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)

    scheme = object.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
    scheme._prepare_zercmoe_w8a8 = True
    scheme.dynamic_eplb = False
    scheme.in_dtype = torch.bfloat16
    runner = SimpleNamespace(
        enable_npugraph_ex_static_kernel=False,
        routed_experts=_packed_layer(),
        quant_type=QuantType.W8A8,
        _quant_method=scheme,
        top_k=8,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        scoring_func="sigmoid",
        e_score_correction_bias=None,
        custom_routing_function=None,
        tid2eid=None,
        enable_shared_expert_dp=False,
        renormalize=True,
        moe_config=config,
        _expert_map=None,
        _original_routed_scaling_factor=2.5,
        activation="silu",
        apply_router_weight_on_input=False,
        log2phy=None,
        global_redundant_expert_num=0,
        dynamic_eplb=False,
    )
    # The test calls the original method with ordinary state; no runner stub.
    state.forward = lambda x, logits: fused_moe.AscendMoERunner.no_shared_forward_impl(runner, x, logits)

    # Spies execute original functions and let native-entry snapshots distinguish
    # prepare work from the legitimate TP split inside finalize.
    state.pad = Mock(wraps=torch.nn.functional.pad)
    state.split = Mock(wraps=torch.tensor_split)
    state.select = Mock(wraps=experts_selector.select_experts)
    monkeypatch.setattr(torch.nn.functional, "pad", state.pad)
    monkeypatch.setattr(torch, "tensor_split", state.split)
    monkeypatch.setattr(w8a8_dynamic, "select_experts", state.select)

    def gating(logits, *, k, norm_type, renorm, routed_scaling_factor, **kwargs):
        scores = logits.sigmoid() if norm_type == 1 else logits.softmax(-1)
        probs, ids = scores.topk(k, dim=-1)
        if renorm:
            probs = probs / probs.sum(-1, keepdim=True)
        return probs * routed_scaling_factor, ids.to(torch.int32), None

    state.gating = Mock(side_effect=gating)
    monkeypatch.setattr(experts_selector.DeviceOperator, "moe_gating_top_k", state.gating)
    heap = object()

    def native(x, ids, probs, w1, w2, sym_buffer, *, return_expert_tokens, l1_weights_sf, l2_weights_sf):
        assert sym_buffer is heap
        assert return_expert_tokens == 0
        assert x.dtype == torch.bfloat16 and ids.dtype == torch.int32 and probs.dtype == torch.float32
        assert x.shape[0] == ctx.padded_num_tokens // state.tp
        assert ids.shape == probs.shape == (x.shape[0], 8)
        assert w1 is runner.routed_experts.w13_weight_zr
        assert w2 is runner.routed_experts.w2_weight_zr
        assert l1_weights_sf is runner.routed_experts.w13_weight_scale_zr_packed
        assert l2_weights_sf is runner.routed_experts.w2_weight_scale_zr_packed
        state.launches.append(
            SimpleNamespace(
                x=x,
                ids=ids,
                probs=probs,
                w1=w1,
                w2=w2,
                scale1=l1_weights_sf,
                scale2=l2_weights_sf,
                pad_calls=state.pad.call_count,
                split_calls=state.split.call_count,
                select_calls=state.select.call_count,
            )
        )
        # Deterministic interface oracle, not a claim about MoE numerical math.
        out = x + probs.sum(-1, keepdim=True).to(torch.bfloat16)
        return out, torch.empty(0, dtype=torch.int32)

    state.native = Mock(side_effect=native)
    state.heap = Mock(return_value=heap)
    comm._a3_zercmoe = SimpleNamespace(
        __version__="1.5.0",
        pack_weights=lambda first, second: (first, second),
        pack_weights_sf=lambda scale: scale,
        get_symm_buffer=state.heap,
        mega_moe=state.native,
    )

    def gather(outputs, local, group):
        assert group is config.tp_group.device_group
        rows = ctx.padded_num_tokens // state.tp
        assert local.shape == (rows, 32)
        torch.testing.assert_close(local, state.expected_full[state.rank * rows : (state.rank + 1) * rows])
        for index, output in enumerate(outputs):
            output.copy_(state.expected_full[index * rows : (index + 1) * rows])

    state.gather = Mock(side_effect=gather)
    monkeypatch.setattr(prepare_finalize.dist, "all_gather", state.gather)
    state.ctx, state.comm, state.pf, state.runner = ctx, comm, pf, runner
    return state


def _batch(chain, *, actual, sp, target=32, poison_padding=False):
    ctx = chain.ctx
    ctx.padded_num_tokens = target
    ctx.mc2_num_actual_tokens = actual
    ctx.mc2_mask = torch.arange(target) < actual
    ctx.flash_comm_v1_enabled = sp
    full = torch.zeros(target, 32, dtype=torch.bfloat16)
    full[:actual] = torch.arange(actual * 32, dtype=torch.float32).reshape(actual, 32).to(torch.bfloat16) / 32
    chain.expected_full = full.clone()
    chain.expected_full[:actual] += 2.5
    if poison_padding:
        full[actual:] = float("nan")
    rows = target // chain.tp
    x = full[chain.rank * rows : (chain.rank + 1) * rows].clone() if sp else full[:actual].clone()
    logits = torch.arange(256, dtype=torch.float32).repeat(x.shape[0], 1) / 256
    return x, logits


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("actual", [0, 7, 32])
def test_zercmoe_single_weight_postload_to_full_forward_chain(early_chain, monkeypatch, sp, actual):
    chain = early_chain
    experts = 8
    layer = SimpleNamespace(
        w13_weight=torch.ones(experts, 32, 32, dtype=torch.int8),
        w2_weight=torch.ones(experts, 32, 16, dtype=torch.int8),
        w13_weight_scale=torch.ones(experts, 32, 1, dtype=torch.float32),
        w2_weight_scale=torch.ones(experts, 32, 1, dtype=torch.float32),
        w13_weight_offset=torch.zeros(experts, 32, 1),
        w2_weight_offset=torch.zeros(experts, 32, 1),
        swiglu_limit=0.0,
        zercmoe_needs_dispatch_event=False,
    )
    scheme = chain.runner._quant_method
    scheme._zercmoe_single_weight = True

    def prepare(layer):
        prepare_w8a8_layer(
            layer,
            pack_weights=lambda w1, w2: (w1.flatten().clone(), w2.flatten().clone()),
            pack_weights_sf=lambda sf: sf.contiguous().view(torch.int32).to(torch.int64),
        )

    monkeypatch.setattr(w8a8_dynamic, "prepare_w8a8_layer", prepare)
    scheme.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
    chain.runner.routed_experts = layer
    x, logits = _batch(chain, actual=actual, sp=sp)
    output = chain.forward(x, logits)
    assert chain.native.call_count == 1
    assert chain.launches[0].w1 is layer.w13_weight_zr
    assert chain.launches[0].w2 is layer.w2_weight_zr
    expected = chain.expected_full[6:8] if sp else chain.expected_full[:actual]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("actual", [0, 7, 32])
def test_zercmoe_graph_padding_uses_current_mask_even_for_full_or_empty_rank(early_chain, sp, actual):
    chain = early_chain
    chain.ctx.zercmoe_graph_execution = True
    x, logits = _batch(chain, actual=actual, sp=sp, poison_padding=True)
    chain.comm.initialize_w8a8_padding_ids(device=x.device)
    output = chain.forward(x, logits)
    assert chain.select.call_count == 1
    assert chain.native.call_count == 1
    assert not chain.pf.all_tokens_padding and not chain.pf.all_tokens_valid
    mask = chain.ctx.mc2_mask[chain.rank * 2 : (chain.rank + 1) * 2]
    launched = chain.launches[-1]
    assert torch.all(launched.probs[~mask] == 0)
    assert torch.all(launched.x[~mask] == 0)
    expected = chain.expected_full[6:8] if sp else chain.expected_full[:actual]
    torch.testing.assert_close(output, expected)


def test_zercmoe_graph_requires_padding_ids_before_capture(early_chain):
    chain = early_chain
    chain.ctx.zercmoe_graph_execution = True
    x, logits = _batch(chain, actual=7, sp=False)
    with pytest.raises(RuntimeError, match="padding IDs were not initialized"):
        chain.forward(x, logits)
    chain.native.assert_not_called()


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("actual", [0, 5])
def test_zercmoe_early_padding_chain_skips_prepare_and_route_but_still_launches(early_chain, sp, actual):
    chain = early_chain
    x, logits = _batch(chain, actual=actual, sp=sp, poison_padding=True)
    before_x, before_logits = x.clone(), logits.clone()
    output = chain.forward(x, logits)

    assert chain.pf.all_tokens_padding and not chain.pf.all_ranks_empty
    assert len(chain.launches) == 1
    sent = chain.launches[0]
    assert sent.pad_calls == sent.split_calls == sent.select_calls == 0
    assert torch.count_nonzero(sent.x) == torch.count_nonzero(sent.probs) == 0
    assert sent.ids.tolist() == [list(range(152, 160))] * 2
    assert chain.select.call_count == chain.gating.call_count == 0
    assert chain.gather.call_count == (0 if sp else 1)
    assert chain.split.call_count == (0 if sp else 1)  # finalize only
    torch.testing.assert_close(x, before_x, equal_nan=True)
    torch.testing.assert_close(logits, before_logits)
    expected = chain.expected_full[6:8] if sp else chain.expected_full[:actual]
    torch.testing.assert_close(output, expected)
    chain.heap.assert_called_once()


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("actual", [7, 32])
def test_zercmoe_early_padding_chain_keeps_original_mixed_and_valid_routing(early_chain, sp, actual):
    chain = early_chain
    x, logits = _batch(chain, actual=actual, sp=sp)
    output = chain.forward(x, logits)
    sent = chain.launches[0]
    assert not chain.pf.all_tokens_padding
    assert chain.pf.padding_inputs is None
    assert sent.select_calls == chain.gating.call_count == 1
    assert sent.pad_calls == (2 if not sp and actual < 32 else 0)
    assert sent.split_calls == (1 if sp else 3)
    local_valid = min(2, max(0, actual - 6))
    torch.testing.assert_close(sent.probs[:local_valid].sum(-1), torch.full((local_valid,), 2.5))
    if local_valid < 2:
        assert torch.count_nonzero(sent.x[local_valid:]) == 0
        assert torch.count_nonzero(sent.probs[local_valid:]) == 0
        assert sent.ids[local_valid:].tolist() == [list(range(152, 160))]
    expected = chain.expected_full[6:8] if sp else chain.expected_full[:actual]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("sp", [False, True])
def test_zercmoe_early_padding_chain_reuses_only_inputs_across_layers_and_releases_for_mixed(early_chain, sp):
    chain = early_chain
    x, logits = _batch(chain, actual=0, sp=sp)
    chain.forward(x, logits)
    first = chain.launches[-1]
    chain.runner.routed_experts = _packed_layer(tag=2)
    chain.forward(x, logits)
    second = chain.launches[-1]
    assert second.x is first.x and second.ids is first.ids and second.probs is first.probs
    assert second.w1 is not first.w1 and second.scale1 is not first.scale1
    assert chain.native.call_count == 2
    chain.heap.assert_called_once()

    x, logits = _batch(chain, actual=7, sp=sp)
    chain.forward(x, logits)
    assert chain.pf.padding_inputs is None
    assert chain.comm._w8a8_padding_inputs is None
    assert chain.select.call_count == 1
    x, logits = _batch(chain, actual=0, sp=sp)
    chain.forward(x, logits)
    assert chain.launches[-1].x is not first.x
    assert chain.native.call_count == 4 and chain.select.call_count == 1
    chain.heap.assert_called_once()


@pytest.mark.parametrize("sp", [False, True])
def test_zercmoe_early_padding_chain_global_empty_never_routes_or_launches(early_chain, sp):
    chain = early_chain
    x, logits = _batch(chain, actual=0, sp=sp, target=0)
    result = chain.forward(x, logits)
    assert result.shape == (0, 32) and result.dtype == torch.bfloat16
    assert chain.pf.all_ranks_empty and not chain.pf.all_tokens_padding
    chain.select.assert_not_called()
    chain.gating.assert_not_called()
    chain.native.assert_not_called()
    chain.heap.assert_not_called()
    chain.gather.assert_not_called()


def test_zercmoe_early_padding_chain_custom_router_still_runs(early_chain):
    chain = early_chain
    custom = Mock(
        side_effect=lambda hidden_states, gating_output, topk, renormalize: (
            torch.full((hidden_states.shape[0], topk), 1 / topk),
            torch.arange(topk).expand(hidden_states.shape[0], -1),
        )
    )
    chain.runner.custom_routing_function = custom
    x, logits = _batch(chain, actual=0, sp=False)
    chain.forward(x, logits)
    assert chain.pf.padding_inputs is None
    assert chain.select.call_count == custom.call_count == 1
    assert chain.launches[0].pad_calls == 2 and chain.launches[0].split_calls == 3
    chain.gating.assert_not_called()
    chain.native.assert_called_once()


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("sp", [False, True])
def test_zercmoe_early_padding_tp8_decode_keeps_every_rank_in_collective(early_chain, monkeypatch, rank, sp):
    chain = early_chain
    chain.tp = chain.ep = chain.pf.tp_size = chain.comm.token_dispatcher.ep_world_size = 8
    chain.rank = chain.ep_rank = chain.pf.tp_rank = chain.comm.token_dispatcher.ep_rank_id = rank
    chain.runner.routed_experts = _packed_layer(experts=32)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "8")
    # A nonidentity mapping must not be reapplied to physical padding IDs.
    chain.runner.log2phy = torch.roll(torch.arange(256, dtype=torch.int32), 1)
    x, logits = _batch(chain, actual=1, sp=sp, target=8, poison_padding=True)
    before_x, before_logits = x.clone(), logits.clone()
    output = chain.forward(x, logits)
    sent = chain.launches[0]
    assert chain.native.call_count == 1
    assert chain.gather.call_count == (not sp)
    assert chain.select.call_count == int(rank == 0)
    if rank:
        assert sent.pad_calls == sent.split_calls == sent.select_calls == 0
        assert sent.ids.tolist() == [list(range(rank * 32, rank * 32 + 8))]
        assert torch.count_nonzero(sent.probs) == torch.count_nonzero(sent.x) == 0
    else:
        torch.testing.assert_close(sent.x, before_x[:1])
        torch.testing.assert_close(sent.probs.sum(-1), torch.tensor([2.5]))
    torch.testing.assert_close(x, before_x, equal_nan=True)
    torch.testing.assert_close(logits, before_logits)
    expected = chain.expected_full[rank : rank + 1] if sp else chain.expected_full[:1]
    torch.testing.assert_close(output, expected)


def test_zercmoe_early_padding_special_route_and_unknown_metadata_clear_prepared_state(early_chain):
    chain = early_chain
    x, logits = _batch(chain, actual=0, sp=False)
    chain.forward(x, logits)
    cached = chain.comm._w8a8_padding_inputs
    custom = Mock(
        side_effect=lambda hidden_states, gating_output, topk, renormalize: (
            torch.ones(hidden_states.shape[0], topk),
            torch.arange(topk).expand(hidden_states.shape[0], -1),
        )
    )
    chain.runner.custom_routing_function = custom
    chain.forward(x, logits)
    assert chain.pf.padding_inputs is None
    custom.assert_called_once()
    assert chain.comm._w8a8_padding_inputs is cached
    chain.runner.custom_routing_function = None
    chain.ctx.mc2_num_actual_tokens = None
    chain.forward(x, logits)
    assert chain.pf.padding_inputs is None
    assert chain.comm._w8a8_padding_inputs is None
    chain.ctx.mc2_num_actual_tokens = 0
    chain.forward(x, logits)
    assert chain.pf.padding_inputs is not None
    assert chain.comm._w8a8_padding_inputs is not cached
    assert chain.native.call_count == 4
    assert chain.select.call_count == 2
