import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.zercmoe_adapter import (
    W8A8Launcher,
    can_skip_w8a8_padding_routing,
    get_w8a8_operator,
    make_w8a8_padding_ids,
    needs_w8a8_dispatch_event,
    prepare_w8a8_input,
    prepare_w8a8_layer,
    validate_w8a8_payload,
)


@pytest.mark.parametrize(
    "case",
    [
        "standard",
        "bf16_logits",
        "grouped",
        "custom",
        "hash",
        "profile",
        "zero_expert",
        "mix",
        "input_dtype",
        "logits_dtype",
        "logits_rows",
        "logits_rank",
        "logits_device",
        "topk_zero",
        "topk_large",
        "topk_float",
        "scoring",
        "missing_group",
        "zero_group",
        "group_count",
        "group_divisibility",
        "group_capacity",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "single_expert_group",
        "renorm",
        "expert_limit",
        "group_alignment",
        "softplus",
        "softplus_groups",
        "softplus_bias_dtype",
    ],
)
def test_zercmoe_early_padding_guard_preserves_nonstandard_and_invalid_routing(case):
    layer = SimpleNamespace()
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    logits = torch.zeros(1, 256)
    kwargs = dict(
        top_k=8,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(256),
        custom_routing_function=None,
        tid2eid=None,
        enable_force_load_balance=False,
    )
    if case == "bf16_logits":
        logits = logits.bfloat16()
    elif case == "grouped":
        kwargs.update(num_expert_group=8, topk_group=2)
    elif case == "custom":
        kwargs["custom_routing_function"] = lambda *a: None
    elif case == "hash":
        kwargs["tid2eid"] = torch.zeros(1, 8, dtype=torch.int32)
    elif case == "profile":
        kwargs["enable_force_load_balance"] = True
    elif case == "zero_expert":
        layer.zero_expert_num = 1
    elif case == "mix":
        layer.mix_placement = True
    elif case == "input_dtype":
        x = x.float()
    elif case == "logits_dtype":
        logits = logits.int()
    elif case == "logits_rows":
        logits = logits.expand(2, 256)
    elif case == "logits_rank":
        logits = logits.flatten()
    elif case == "logits_device":
        logits = torch.empty(1, 256, device="meta")
    elif case == "topk_zero":
        kwargs["top_k"] = 0
    elif case == "topk_large":
        kwargs["top_k"] = 257
    elif case == "topk_float":
        kwargs["top_k"] = 8.5
    elif case == "scoring":
        kwargs["scoring_func"] = "unknown"
    elif case == "missing_group":
        kwargs["num_expert_group"] = None
    elif case == "zero_group":
        kwargs["num_expert_group"] = 0
    elif case == "group_count":
        kwargs["topk_group"] = 2
    elif case == "group_divisibility":
        kwargs["num_expert_group"] = 3
    elif case == "group_capacity":
        kwargs["num_expert_group"] = 64
    elif case == "bias_shape":
        kwargs["e_score_correction_bias"] = torch.zeros(1, 256)
    elif case == "bias_dtype":
        kwargs["e_score_correction_bias"] = torch.zeros(256, dtype=torch.int32)
    elif case == "bias_device":
        kwargs["e_score_correction_bias"] = torch.empty(256, device="meta")
    elif case == "single_expert_group":
        kwargs.update(top_k=1, num_expert_group=256)
    elif case == "renorm":
        kwargs["renormalize"] = 2
    elif case == "expert_limit":
        logits = torch.zeros(1, 4096)
        kwargs["e_score_correction_bias"] = None
    elif case == "group_alignment":
        kwargs.update(top_k=1, num_expert_group=128)
    elif case.startswith("softplus"):
        kwargs["scoring_func"] = "sqrtsoftplus"
        if case == "softplus_groups":
            kwargs["num_expert_group"] = 8
        elif case == "softplus_bias_dtype":
            kwargs["e_score_correction_bias"] = torch.zeros(256, dtype=torch.bfloat16)
    with patch("torch.Tensor.item", side_effect=AssertionError("eligibility must not read device values")):
        assert can_skip_w8a8_padding_routing(layer, x, logits, **kwargs) == (
            case in ("standard", "bf16_logits", "grouped", "softplus")
        )


def _make_layer():
    # Different K/I and channel values catch accidental transposes/reordering.
    return SimpleNamespace(
        w13_weight=torch.arange(2 * 64 * 64, dtype=torch.int8).reshape(2, 64, 64),
        w2_weight=torch.arange(2 * 32 * 64, dtype=torch.int8).reshape(2, 32, 64),
        w13_weight_scale=(torch.arange(128, dtype=torch.float32) / 131 + 0.125).reshape(2, 64, 1),
        w2_weight_scale=(torch.arange(128, dtype=torch.float32) / 139 + 0.25).reshape(2, 64, 1),
        w13_weight_offset=torch.zeros(2, 64, 1),
        w2_weight_offset=torch.zeros(2, 64, 1),
    )


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("shared_kind", ["none", "quantized", "unquantized", "mixed"])
def test_zercmoe_dispatch_event_keeps_only_shared_stream_consumers(overlap, shared_kind):
    shared = None
    if shared_kind != "none":
        shared = SimpleNamespace(gate_up_proj=SimpleNamespace(), down_proj=SimpleNamespace())
        if shared_kind in ("quantized", "mixed"):
            shared.gate_up_proj.weight_scale = object()
        if shared_kind == "quantized":
            shared.down_proj.weight_scale = object()
    assert needs_w8a8_dispatch_event(shared, overlap) == (overlap and shared_kind in ("unquantized", "mixed"))


def test_zercmoe_all_valid_input_avoids_copies_without_reading_the_device_mask():
    original = _payload()
    payload = replace(
        original,
        topk_ids=original.topk_ids.to(torch.int32),
        topk_weights=original.topk_weights.to(torch.float32),
        routing=replace(original.routing, mc2_mask=torch.ones(3, dtype=torch.bool)),
    )
    with (
        patch("torch.where", side_effect=AssertionError("unnecessary padding copy")),
        patch("torch.Tensor.item", side_effect=AssertionError("device synchronization")),
    ):
        result = prepare_w8a8_input(payload, all_tokens_valid=True)
    assert result.hidden_states is payload.hidden_states
    assert result.topk_ids is payload.topk_ids
    assert result.topk_weights is payload.topk_weights


@pytest.mark.parametrize("mapped", [False, True])
def test_zercmoe_full_padding_input_matches_masked_path_without_device_work(mapped):
    original = _payload()
    payload = replace(
        original,
        hidden_states=torch.full_like(original.hidden_states, float("nan")),
        topk_ids=torch.full_like(original.topk_ids, 10000),
        topk_weights=torch.full_like(original.topk_weights, float("inf")),
        routing=replace(
            original.routing,
            mc2_mask=torch.zeros(3, dtype=torch.bool),
            log2phy=torch.tensor([2, 3, 0, 1], dtype=torch.int32) if mapped else None,
        ),
    )
    physical_ids = torch.tensor([2, 3], dtype=torch.int32)
    reference = prepare_w8a8_input(payload, padding_ids=physical_ids)
    cached = (
        torch.zeros_like(payload.hidden_states),
        physical_ids.repeat(3, 1),
        torch.zeros(3, 2, dtype=torch.float32),
    )
    with (
        patch("torch.where", side_effect=AssertionError("full-padding where")),
        patch("torch.Tensor.item", side_effect=AssertionError("device synchronization")),
    ):
        result = prepare_w8a8_input(payload, all_tokens_padding=True, padding_inputs=cached)
    for fast, masked in zip((result.hidden_states, result.topk_ids, result.topk_weights), cached):
        assert fast is masked
    torch.testing.assert_close(result.hidden_states, reference.hidden_states)
    torch.testing.assert_close(result.topk_ids, reference.topk_ids)
    torch.testing.assert_close(result.topk_weights, reference.topk_weights)
    assert result.routing.log2phy is None
    assert result.weights is payload.weights
    assert result.hidden_states.data_ptr() != payload.hidden_states.data_ptr()
    assert torch.isnan(payload.hidden_states).all()


@pytest.mark.parametrize(
    "invalid",
    ["no_mask", "both_flags", "no_cache", "ids_device", "probs_device", "mask_dtype", "external_scale", "input_dtype"],
)
def test_zercmoe_full_padding_cache_does_not_bypass_input_checks(invalid):
    payload = _payload()
    payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.zeros(3, dtype=torch.bool)))
    cached = (torch.zeros_like(payload.hidden_states), torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 2))
    if invalid == "no_mask":
        payload = replace(payload, routing=replace(payload.routing, mc2_mask=None))
    elif invalid == "ids_device":
        payload = replace(payload, topk_ids=payload.topk_ids.to("meta"))
    elif invalid == "probs_device":
        payload = replace(payload, topk_weights=payload.topk_weights.to("meta"))
    elif invalid == "mask_dtype":
        payload = replace(payload, routing=replace(payload.routing, mc2_mask=payload.routing.mc2_mask.int()))
    elif invalid == "external_scale":
        payload = replace(payload, routing=replace(payload.routing, pertoken_scale=torch.ones(3)))
    elif invalid == "input_dtype":
        payload = replace(payload, hidden_states=payload.hidden_states.float())
    with pytest.raises(ValueError, match="ZERC_MOE"):
        prepare_w8a8_input(
            payload,
            all_tokens_valid=invalid == "both_flags",
            all_tokens_padding=True,
            padding_inputs=None if invalid == "no_cache" else cached,
        )


@pytest.mark.parametrize(
    "world,experts,rank,top_k",
    [(8, 256, rank, 8) for rank in range(8)] + [(4, 8, 3, 5), (8, 32, 7, 8), (2, 4, 1, 4)],
)
def test_zercmoe_padding_prefers_local_physical_experts_and_spills_without_duplicates(world, experts, rank, top_k):
    ids = make_w8a8_padding_ids(num_experts=experts, ep_world_size=world, ep_rank=rank, top_k=top_k, device="cpu")
    assert ids.dtype == torch.int32 and ids.shape == (top_k,)
    assert len(set(ids.tolist())) == top_k
    assert all(0 <= i < experts for i in ids.tolist())
    destinations = ids // (experts // world)
    assert int((destinations == rank).sum()) == min(top_k, experts // world)
    if (world, experts, rank, top_k) == (4, 8, 3, 5):
        assert ids.tolist() == [6, 7, 0, 1, 2]


@pytest.mark.parametrize(
    "experts,world,rank,top_k",
    [(0, 8, 0, 8), (256, 0, 0, 8), (255, 8, 0, 8), (256, 8, 8, 8), (256, 8, -1, 8), (256, 8, 0, 0), (256, 8, 0, 257)],
)
def test_zercmoe_padding_rejects_invalid_geometry_before_device_allocation(experts, world, rank, top_k):
    with (
        patch("torch.arange", side_effect=AssertionError("invalid geometry")),
        pytest.raises(ValueError, match="ZERC_MOE"),
    ):
        make_w8a8_padding_ids(num_experts=experts, ep_world_size=world, ep_rank=rank, top_k=top_k, device="cpu")


def test_zercmoe_ep8_single_token_padding_no_longer_targets_rank_zero():
    received = torch.zeros(8, dtype=torch.int64)
    old_received = torch.zeros_like(received)
    remote = 0
    for rank in range(8):
        original = _payload()
        # One real token selecting one expert per rank; the other seven ranks
        # have one physical input row but no valid local token.
        real_ids = torch.arange(8, dtype=torch.int64).reshape(1, 8) * 32
        payload = replace(
            original,
            hidden_states=original.hidden_states[:1],
            topk_ids=real_ids,
            topk_weights=torch.full((1, 8), 0.125),
            routing=replace(original.routing, mc2_mask=torch.tensor([rank == 0])),
        )
        prepared = prepare_w8a8_input(
            payload,
            all_tokens_valid=rank == 0,
            padding_ids=make_w8a8_padding_ids(num_experts=256, ep_world_size=8, ep_rank=rank, top_k=8, device="cpu"),
        )
        if rank == 0:
            torch.testing.assert_close(prepared.topk_ids, real_ids.int())
        else:
            destinations = prepared.topk_ids.flatten() // 32
            received += torch.bincount(destinations, minlength=8)
            old_received += torch.bincount(torch.arange(8) // 32, minlength=8)
            remote += int((destinations != rank).sum())
            assert torch.count_nonzero(prepared.hidden_states) == 0
            assert torch.count_nonzero(prepared.topk_weights) == 0
    assert old_received.tolist() == [56, 0, 0, 0, 0, 0, 0, 0]
    assert received.tolist() == [0, 8, 8, 8, 8, 8, 8, 8]
    assert remote == 0


@pytest.mark.parametrize("mask_kind", ["mixed", "all_valid", "none"])
def test_zercmoe_maps_real_routes_once_before_inserting_physical_padding(mask_kind):
    original = _payload()
    mapping = torch.tensor([3, 0, 1, 2], dtype=torch.int32)
    mask = torch.tensor([True, False, True]) if mask_kind == "mixed" else torch.ones(3, dtype=torch.bool)
    ids = original.topk_ids.clone()
    ids[1] = torch.tensor([-1, 100000]) if mask_kind == "mixed" else torch.tensor([0, 1])
    probs = original.topk_weights.clone()
    if mask_kind != "mixed":
        probs[1] = 0.5
    payload = replace(
        original,
        topk_ids=ids,
        topk_weights=probs,
        routing=replace(original.routing, log2phy=mapping, mc2_mask=None if mask_kind == "none" else mask),
    )
    prepared = prepare_w8a8_input(
        payload,
        all_tokens_valid=mask_kind == "all_valid",
        padding_ids=torch.tensor([2, 3], dtype=torch.int32) if mask_kind == "mixed" else None,
    )
    assert prepared.routing.log2phy is None  # the comm layer must not remap again
    assert payload.routing.log2phy is mapping
    torch.testing.assert_close(prepared.topk_ids[mask], mapping[ids[mask]])
    torch.testing.assert_close(prepared.hidden_states[mask], payload.hidden_states[mask])
    torch.testing.assert_close(prepared.topk_weights[mask], payload.topk_weights[mask])
    torch.testing.assert_close(payload.topk_ids, ids)
    if mask_kind == "mixed":
        assert prepared.topk_ids[1].tolist() == [2, 3]  # physical IDs, not mapping[[2,3]]
        assert torch.count_nonzero(prepared.topk_weights[1]) == 0
        assert torch.count_nonzero(prepared.hidden_states[1]) == 0


def test_zercmoe_padding_distribution_preserves_valid_swiglu_reference_outputs():
    original = _payload()
    mapping = torch.tensor([3, 0, 1, 2], dtype=torch.int32)
    payload = replace(original, routing=replace(original.routing, log2phy=mapping))
    prepared = prepare_w8a8_input(payload, padding_ids=torch.tensor([2, 3], dtype=torch.int32))
    w1 = torch.arange(4 * 64 * 32, dtype=torch.float32).sin().reshape(4, 64, 32) * 0.01
    w2 = torch.arange(4 * 16 * 64, dtype=torch.float32).cos().reshape(4, 16, 64) * 0.01

    def reference(x, ids, probs):
        result = torch.zeros_like(x, dtype=torch.float32)
        for row in range(x.shape[0]):
            for expert, prob in zip(ids[row].tolist(), probs[row]):
                gate, up = (x[row].float() @ w1[expert]).chunk(2)
                result[row] += prob * ((torch.nn.functional.silu(gate) * up) @ w2[expert])
        return result

    valid = payload.routing.mc2_mask
    expected = reference(payload.hidden_states[valid], mapping[payload.topk_ids[valid]], payload.topk_weights[valid])
    actual = reference(prepared.hidden_states, prepared.topk_ids, prepared.topk_weights)
    torch.testing.assert_close(actual[valid], expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~valid]) == 0


def _pack_nd(w1, w2):
    # This CPU stub only tracks the interface boundary; it does not model NZ.
    return w1.flatten().clone(), w2.flatten().clone()


def _pack_sf(scale):
    # CPU fixture for scale-slot transport tests, not the real wheel encoder.
    return scale.contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xFFFFE000)


def _mega_moe(
    x,
    topk_ids,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    return_expert_tokens=0,
    l1_weights_sf=None,
    l2_weights_sf=None,
):
    raise AssertionError("interface probing must not execute an operator")


def _get_symm_buffer(
    *,
    moe_expert_num,
    ep_world_size,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    ipport,
    device_id,
    w_dtype=torch.int8,
    mem_size_mb=0,
):
    raise AssertionError("interface probing must not bootstrap communication")


def _backend(operator):
    return SimpleNamespace(
        __version__="1.5.0",
        pack_weights=_pack_nd,
        pack_weights_sf=_pack_sf,
        get_symm_buffer=_get_symm_buffer,
        mega_moe=operator,
    )


def test_prepare_packs_once_and_preserves_raw_fp32_channels():
    layer = _make_layer()
    originals = dict(vars(layer))
    expected_weights = {name: getattr(layer, f"{name}_weight").clone() for name in ("w13", "w2")}
    packer = Mock(side_effect=_pack_nd)

    encoder = Mock(side_effect=_pack_sf)
    prepare_w8a8_layer(layer, pack_weights=packer, pack_weights_sf=encoder)

    packer.assert_called_once()
    assert encoder.call_count == 2
    assert layer.w13_zr_shape == (2, 64, 64)
    assert layer.w2_zr_shape == (2, 32, 64)
    packed_inputs = packer.call_args.args
    assert len(packed_inputs) == 2
    for index, name in enumerate(("w13", "w2")):
        original_weight = originals[f"{name}_weight"]
        torch.testing.assert_close(packed_inputs[index], expected_weights[name])
        assert packed_inputs[index].ndim == 3 and packed_inputs[index].is_contiguous()
        cached_weight = getattr(layer, f"{name}_weight_zr")
        assert getattr(layer, f"{name}_weight") is original_weight
        assert cached_weight.dtype == torch.int8 and cached_weight.ndim == 1 and cached_weight.is_contiguous()
        torch.testing.assert_close(cached_weight, expected_weights[name].flatten())
        assert cached_weight.data_ptr() != original_weight.data_ptr()
        scale = getattr(layer, f"{name}_weight_scale_zr")
        assert getattr(layer, f"{name}_weight_scale") is originals[f"{name}_weight_scale"]
        assert scale.shape == (2, 64)
        assert scale.dtype == torch.float32 and scale.is_contiguous()
        torch.testing.assert_close(scale, originals[f"{name}_weight_scale"].squeeze(-1))
        assert scale.data_ptr() != originals[f"{name}_weight_scale"].data_ptr()
        # Subsequent fallback NZ/dtype updates must not mutate packed weights/scales.
        expected_scale = scale.clone()
        original_weight.zero_()
        originals[f"{name}_weight_scale"].zero_()
        torch.testing.assert_close(cached_weight, expected_weights[name].flatten())
        torch.testing.assert_close(scale, expected_scale)
        encoded = getattr(layer, f"{name}_weight_scale_zr_packed")
        assert encoded.dtype == torch.int64
        torch.testing.assert_close(encoded, _pack_sf(expected_scale))


def test_prepare_default_loader_uses_external_packer_once(monkeypatch):
    from vllm_ascend.ops.fused_moe import comm_utils

    packer = Mock(side_effect=_pack_nd)
    encoder = Mock(side_effect=_pack_sf)
    loader = Mock(return_value=SimpleNamespace(pack_weights=packer, pack_weights_sf=encoder))
    monkeypatch.setattr(comm_utils, "load_a3_zercmoe_ops", loader)
    layer = _make_layer()
    prepare_w8a8_layer(layer)
    loader.assert_called_once_with()
    packer.assert_called_once()
    assert encoder.call_count == 2
    assert layer.w13_weight_zr.ndim == layer.w2_weight_zr.ndim == 1


@pytest.mark.parametrize("layout", ["matrix", "flat", "tiled"])
def test_zercmoe_preencoded_scales_match_automatic_encoding_and_are_reused(layout):
    # This fake wheel isolates encoding/transport equivalence, not NPU W8A8.
    layer = _make_layer()

    def encode(scale):
        slots = _pack_sf(scale)
        if layout == "flat":
            return slots.flatten()
        if layout == "tiled":
            return slots.reshape(2, 2, -1)
        return slots

    encoder = Mock(side_effect=encode)
    prepare_w8a8_layer(layer, pack_weights=_pack_nd, pack_weights_sf=encoder)
    assert encoder.call_count == 2
    payload = _payload()
    payload = replace(
        payload,
        topk_ids=torch.tensor([[0, 1], [-1, -1], [1, 0]]),
        weights=MoEWeights(
            w1=layer.w13_weight_zr,
            w2=layer.w2_weight_zr,
            w1_scale=layer.w13_weight_scale_zr_packed,
            w2_scale=layer.w2_weight_scale_zr_packed,
        ),
    )
    payload = prepare_w8a8_input(payload, padding_ids=torch.arange(2, dtype=torch.int32))
    validate_w8a8_payload(payload, num_experts=4, ep_world_size=2, hidden=64, intermediate=32)
    runtime_encoder = Mock(side_effect=encode)

    def operator(x, ids, probs, w1, w2, sym, *, return_expert_tokens, l1_weights_sf, l2_weights_sf):
        assert return_expert_tokens == 0
        # Model the wheel's FP32 conversion branch and an independent dense
        # SwiGLU reference. INT64 inputs must bypass conversion entirely.
        scales = []
        for sf in (l1_weights_sf, l2_weights_sf):
            slots = runtime_encoder(sf) if sf.dtype == torch.float32 else sf
            scales.append(slots.to(torch.int32).view(torch.float32).reshape(2, 64))
        weights1, weights2 = w1.reshape(2, 64, 64).float(), w2.reshape(2, 32, 64).float()
        output = torch.zeros_like(x, dtype=torch.float32)
        for row in range(x.shape[0]):
            for expert, prob in zip(ids[row].tolist(), probs[row]):
                gate, up = ((x[row].float() @ weights1[expert]) * scales[0][expert]).chunk(2)
                output[row] += prob * ((torch.nn.functional.silu(gate) * up) @ weights2[expert]) * scales[1][expert]
        return output.to(torch.bfloat16), torch.empty(0, dtype=torch.int32)

    actual_operator = Mock(side_effect=operator)
    raw = replace(
        payload, weights=replace(payload.weights, w1_scale=layer.w13_weight_scale_zr, w2_scale=layer.w2_weight_scale_zr)
    )
    expected, _ = W8A8Launcher(actual_operator)(raw, _buffer())
    assert runtime_encoder.call_count == 2
    runtime_encoder.reset_mock()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-forward host copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("per-forward CPU encoding")),
    ):
        for _ in range(3):
            actual, _ = W8A8Launcher(actual_operator)(payload, _buffer())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    runtime_encoder.assert_not_called()
    assert encoder.call_count == 2
    sent = actual_operator.call_args.kwargs
    assert sent["l1_weights_sf"] is layer.w13_weight_scale_zr_packed
    assert sent["l2_weights_sf"] is layer.w2_weight_scale_zr_packed
    torch.testing.assert_close(sent["l1_weights_sf"].flatten(), encode(layer.w13_weight_scale_zr).flatten())
    # A normal post-load/reload pass creates new per-layer encoded values.
    previous = layer.w13_weight_scale_zr_packed
    prepare_w8a8_layer(layer, pack_weights=_pack_nd, pack_weights_sf=encoder)
    assert encoder.call_count == 4
    assert layer.w13_weight_scale_zr_packed is not previous


@pytest.mark.parametrize("invalid", ["dtype", "length", "noncontiguous", "device", "exception"])
@pytest.mark.parametrize("fail_at", [1, 2])
def test_zercmoe_invalid_scale_encoding_does_not_publish_partial_layer(invalid, fail_at):
    layer = _make_layer()
    calls = 0

    def encoder(scale):
        nonlocal calls
        calls += 1
        result = _pack_sf(scale)
        if calls != fail_at:
            return result
        if invalid == "exception":
            raise RuntimeError("scale encoder failure")
        if invalid == "dtype":
            return result.float()
        if invalid == "length":
            return result.flatten()[:-1]
        if invalid == "noncontiguous":
            return result.t()
        return torch.empty(result.shape, dtype=torch.int64, device="meta")

    with pytest.raises(RuntimeError, match="scale|INT64"):
        prepare_w8a8_layer(layer, pack_weights=_pack_nd, pack_weights_sf=encoder)
    for name in (
        "w13_weight_zr",
        "w2_weight_zr",
        "w13_weight_scale_zr",
        "w2_weight_scale_zr",
        "w13_weight_scale_zr_packed",
        "w2_weight_scale_zr_packed",
    ):
        assert not hasattr(layer, name)


def test_zercmoe_encoder_cannot_mutate_the_preserved_raw_scales():
    layer = _make_layer()
    expected = layer.w13_weight_scale.squeeze(-1).clone()

    def encoder(scale):
        result = _pack_sf(scale)
        scale.zero_()
        return result

    prepare_w8a8_layer(layer, pack_weights=_pack_nd, pack_weights_sf=encoder)
    torch.testing.assert_close(layer.w13_weight_scale_zr, expected, rtol=0, atol=0)
    torch.testing.assert_close(layer.w13_weight_scale_zr_packed, _pack_sf(expected), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "alignment", "scale_shape", "scale_dtype", "offset"])
def test_prepare_rejects_invalid_input_before_packing_or_publishing_cache(invalid):
    layer = _make_layer()
    if invalid == "dtype":
        layer.w13_weight = layer.w13_weight.to(torch.bfloat16)
    elif invalid == "shape":
        layer.w2_weight = torch.zeros(2, 64, 32, dtype=torch.int8)
    elif invalid == "alignment":
        layer.w13_weight = torch.zeros(2, 47, 64, dtype=torch.int8)
        layer.w2_weight = torch.zeros(2, 32, 47, dtype=torch.int8)
    elif invalid == "scale_shape":
        layer.w2_weight_scale = torch.ones(64, 2)
    elif invalid == "scale_dtype":
        layer.w2_weight_scale = layer.w2_weight_scale.to(torch.int64)
    else:
        layer.w2_weight_offset[0, 0, 0] = 1
    packer = Mock(side_effect=_pack_nd)
    with pytest.raises(ValueError, match="ZERC_MOE"):
        prepare_w8a8_layer(layer, pack_weights=packer, pack_weights_sf=_pack_sf)
    packer.assert_not_called()
    assert not hasattr(layer, "w13_weight_zr")
    assert not hasattr(layer, "w13_weight_scale_zr")


@pytest.mark.parametrize("invalid", ["dtype", "nd_weight", "length", "exception"])
def test_invalid_packer_output_does_not_publish_half_prepared_cache(invalid):
    layer = _make_layer()
    packed = _pack_nd(layer.w13_weight, layer.w2_weight)
    if invalid == "dtype":
        packed = (packed[0].float(), packed[1])
    elif invalid == "nd_weight":
        packed = (layer.w13_weight.clone(), packed[1])
    elif invalid == "length":
        packed = (packed[0][:-1], packed[1])
    packer = Mock(return_value=packed)
    if invalid == "exception":
        packer.side_effect = RuntimeError("packing failed")
    with pytest.raises((ValueError, RuntimeError)):
        prepare_w8a8_layer(layer, pack_weights=packer, pack_weights_sf=_pack_sf)
    packer.assert_called_once()
    assert not any(
        hasattr(layer, name) for name in ("w13_weight_zr", "w2_weight_zr", "w13_weight_scale_zr", "w2_weight_scale_zr")
    )


def test_operator_binds_external_mega_moe_without_invoking_it():
    operator = Mock(side_effect=_mega_moe)
    operator.__signature__ = inspect.signature(_mega_moe)
    backend = _backend(operator)
    launcher = get_w8a8_operator(backend)
    assert launcher.operator is operator
    operator.assert_not_called()


def test_operator_default_loader_never_uses_the_cann_torch_operator(monkeypatch):
    from vllm_ascend.ops.fused_moe import comm_utils

    operator = Mock(side_effect=_mega_moe)
    operator.__signature__ = inspect.signature(_mega_moe)
    backend = _backend(operator)
    loader = Mock(return_value=backend)
    monkeypatch.setattr(comm_utils, "load_a3_zercmoe_ops", loader)
    launcher = get_w8a8_operator()
    loader.assert_called_once_with()
    assert launcher.operator is operator
    operator.assert_not_called()


def _payload(dtype=torch.bfloat16):
    return MoEFusedExpertsInput(
        hidden_states=torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64).to(dtype),
        topk_ids=torch.tensor([[2, 3], [-1, -1], [1, 2]], dtype=torch.int64),
        topk_weights=torch.tensor([[0.123456, 0.876544], [float("nan"), 1], [0.75, 0.25]]),
        weights=MoEWeights(
            w1=torch.ones(2 * 64 * 64, dtype=torch.int8),
            w2=torch.ones(2 * 32 * 64, dtype=torch.int8),
            w1_scale=_pack_sf(torch.full((2, 64), 0.3)),
            w2_scale=_pack_sf(torch.full((2, 64), 0.7)),
        ),
        routing=MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=torch.tensor([True, False, True]),
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(),
    )


def _buffer():
    # The external wheel documents a handle, not CANN/context metadata fields.
    return object()


def test_bf16_input_preserves_valid_values_and_clears_padding_without_quantizing():
    payload = _payload()
    original = payload.hidden_states.clone()
    result = prepare_w8a8_input(payload, padding_ids=torch.arange(2, dtype=torch.int32))
    torch.testing.assert_close(payload.hidden_states, original)
    torch.testing.assert_close(result.hidden_states[[0, 2]], original[[0, 2]])
    torch.testing.assert_close(result.topk_weights[[0, 2]], payload.topk_weights[[0, 2]])
    assert torch.count_nonzero(result.topk_weights[1]) == 0
    assert torch.count_nonzero(result.hidden_states[1]) == 0
    assert result.topk_ids[1].tolist() == [0, 1]
    assert result.hidden_states.dtype == torch.bfloat16 and result.hidden_states.is_contiguous()
    assert result.routing.pertoken_scale is None
    assert result.topk_ids.dtype == torch.int32
    assert result.topk_weights.dtype == torch.float32
    assert result.weights is payload.weights


def test_zercmoe_single_token_reaches_the_same_public_launcher():
    original = _payload()
    payload = replace(
        original,
        hidden_states=original.hidden_states[:1],
        topk_ids=original.topk_ids[:1],
        topk_weights=original.topk_weights[:1],
        routing=replace(original.routing, mc2_mask=original.routing.mc2_mask[:1]),
    )
    prepared = prepare_w8a8_input(payload, padding_ids=torch.arange(2, dtype=torch.int32))
    validate_w8a8_payload(prepared, num_experts=4, ep_world_size=2, hidden=64, intermediate=32)
    expected = torch.zeros_like(prepared.hidden_states)
    counts = torch.empty(0, dtype=torch.int32)
    operator = Mock(return_value=(expected, counts))
    result, _ = W8A8Launcher(operator)(prepared, _buffer())
    assert result is expected
    operator.assert_called_once()
    assert operator.call_args.kwargs["return_expert_tokens"] == 0
    assert operator.call_args.args[0].shape == (1, 64)
    torch.testing.assert_close(operator.call_args.args[0], original.hidden_states[:1])


@pytest.mark.parametrize("invalid", ["int8", "fp16", "fp32", "external_scale", "mask_shape", "mask_dtype"])
def test_preparation_rejects_non_bf16_or_external_quantization_metadata(invalid):
    payload = _payload()
    if invalid in ("int8", "fp16", "fp32"):
        dtype = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32}[invalid]
        payload = replace(payload, hidden_states=payload.hidden_states.to(dtype))
    elif invalid == "external_scale":
        payload = replace(payload, routing=replace(payload.routing, pertoken_scale=torch.ones(3)))
    elif invalid == "mask_shape":
        payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.ones(2, dtype=torch.bool)))
    else:
        payload = replace(payload, routing=replace(payload.routing, mc2_mask=torch.ones(3, dtype=torch.int32)))
    with pytest.raises(ValueError, match="ZERC_MOE"):
        prepare_w8a8_input(payload)


@pytest.mark.parametrize(
    "invalid",
    ["input", "external_scale", "nd_weight", "packed_length", "scale_dtype", "scale_shape", "ep_size", "activation"],
)
def test_payload_validation_rejects_incompatible_packed_contract(invalid):
    payload = prepare_w8a8_input(_payload(), padding_ids=torch.arange(2, dtype=torch.int32))
    ep = 2
    if invalid == "input":
        payload = replace(payload, hidden_states=payload.hidden_states.to(torch.int8))
    elif invalid == "external_scale":
        payload = replace(payload, routing=replace(payload.routing, pertoken_scale=torch.ones(3)))
    elif invalid == "nd_weight":
        payload = replace(payload, weights=replace(payload.weights, w1=payload.weights.w1.reshape(2, 64, 64)))
    elif invalid == "packed_length":
        payload = replace(payload, weights=replace(payload.weights, w2=payload.weights.w2[:-1]))
    elif invalid == "scale_dtype":
        payload = replace(payload, weights=replace(payload.weights, w2_scale=payload.weights.w2_scale.float()))
    elif invalid == "scale_shape":
        payload = replace(payload, weights=replace(payload.weights, w1_scale=torch.ones(2, 32, dtype=torch.int64)))
    elif invalid == "ep_size":
        ep = 1
    else:
        payload = replace(payload, swiglu_limit=1.0)
    with pytest.raises(ValueError, match="ZERC_MOE"):
        validate_w8a8_payload(payload, num_experts=4, ep_world_size=ep, hidden=64, intermediate=32)


def test_launcher_passes_bf16_packed_weights_and_encoded_int64_scales_without_quantizing():
    payload = prepare_w8a8_input(_payload(), padding_ids=torch.arange(2, dtype=torch.int32))
    validate_w8a8_payload(payload, num_experts=4, ep_world_size=2, hidden=64, intermediate=32)
    out = torch.zeros_like(payload.hidden_states)
    counts = torch.empty(0, dtype=torch.int32)
    operator = Mock(return_value=(out, counts))
    buffer = _buffer()
    launcher = W8A8Launcher(operator)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("forward must not synchronize counts")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("forward must not read scalar counts")),
    ):
        for _ in range(2):
            result, expert_tokens = launcher(payload, buffer)
            assert result is out and expert_tokens is counts
    assert operator.call_count == 2
    args = operator.call_args.args
    assert len(args) == 6
    assert args[0] is payload.hidden_states and args[0].dtype == torch.bfloat16
    assert args[1] is payload.topk_ids and args[2] is payload.topk_weights
    assert args[3] is payload.weights.w1 and args[4] is payload.weights.w2
    assert args[5] is buffer
    kwargs = operator.call_args.kwargs
    assert set(kwargs) == {"return_expert_tokens", "l1_weights_sf", "l2_weights_sf"}
    assert kwargs["return_expert_tokens"] == 0
    assert kwargs["l1_weights_sf"] is payload.weights.w1_scale
    assert kwargs["l2_weights_sf"] is payload.weights.w2_scale
    assert kwargs["l1_weights_sf"].dtype == kwargs["l2_weights_sf"].dtype == torch.int64


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "shape",
        "missing_output",
        "counts_dtype",
        "counts_shape",
        "counts_2d",
        "missing_counts",
        "device",
        "exception",
    ],
)
def test_launcher_never_swallows_wrong_outputs_or_retries(invalid):
    payload = prepare_w8a8_input(_payload(), padding_ids=torch.arange(2, dtype=torch.int32))
    out, counts = torch.zeros_like(payload.hidden_states), torch.empty(0, dtype=torch.int32)
    if invalid == "dtype":
        out = out.to(torch.float16)
    elif invalid == "shape":
        out = out[:2]
    elif invalid == "missing_output":
        out = None
    elif invalid == "counts_dtype":
        counts = counts.to(torch.int64)
    elif invalid == "counts_shape":
        counts = torch.zeros(2, dtype=torch.int32)
    elif invalid == "counts_2d":
        counts = torch.empty(0, 2, dtype=torch.int32)
    elif invalid == "missing_counts":
        counts = None
    elif invalid == "device":
        out = torch.empty(out.shape, dtype=out.dtype, device="meta")
    operator = Mock(return_value=(out, counts))
    if invalid == "exception":
        operator.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError):
        W8A8Launcher(operator)(payload, _buffer())
    operator.assert_called_once()
    assert operator.call_args.kwargs["return_expert_tokens"] == 0


def test_launcher_accepts_empty_cpu_counts_without_requiring_the_output_device():
    payload = prepare_w8a8_input(_payload(), padding_ids=torch.arange(2, dtype=torch.int32))
    # Meta is a CPU-only stand-in for a distinct output device. The mock
    # operator performs no compute, allocation on NPU, or communication.
    payload = replace(
        payload,
        hidden_states=torch.empty(payload.hidden_states.shape, dtype=torch.bfloat16, device="meta"),
    )
    out = torch.empty_like(payload.hidden_states)
    counts = torch.empty(0, dtype=torch.int32)
    operator = Mock(return_value=(out, counts))
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("counts must stay on their returned device")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("no per-layer count synchronization")),
    ):
        result, expert_tokens = W8A8Launcher(operator)(payload, _buffer())
    assert result is out and expert_tokens is counts
    operator.assert_called_once()
