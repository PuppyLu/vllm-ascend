import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.zercmoe_adapter import (
    prepare_w8a8_layer,
    validate_w8a8_model,
)
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.methods.w8a8_pdmix import AscendW8A8PDMixFusedMoeMethod
from vllm_ascend.quantization.methods.w8a8fp8_dynamic import (
    AscendW8A8FP8DynamicFusedMoEMethod,
)
from vllm_ascend.zercmoe_config import zercmoe_draft_model_loading

_MODULE = "vllm_ascend.quantization.methods.w8a8_dynamic"


def _pack_sf(scale):
    # CPU protocol fixture; production uses the wheel's encoder.
    return scale.contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xFFFFE000)


def _make_method(prepare=True):
    method = object.__new__(AscendW8A8DynamicFusedMoEMethod)
    method._prepare_zercmoe_w8a8 = prepare
    method.dynamic_eplb = False
    method.in_dtype = torch.bfloat16
    return method


def _make_layer():
    layer = torch.nn.Module()
    layer.w13_weight = torch.arange(2 * 64 * 96, dtype=torch.int8).reshape(2, 64, 96)
    layer.w2_weight = torch.arange(2 * 96 * 32, dtype=torch.int8).reshape(2, 96, 32)
    layer.w13_weight_scale = (torch.arange(128, dtype=torch.float32) / 131).reshape(2, 64, 1)
    layer.w2_weight_scale = (torch.arange(192, dtype=torch.float32) / 139).reshape(2, 96, 1)
    layer.w13_weight_offset = torch.zeros(2, 64, 1)
    layer.w2_weight_offset = torch.zeros(2, 96, 1)
    layer.swiglu_limit = 0.0
    return layer


def _cache_nd_weights(layer):
    layer.w13_weight = layer.w13_weight.transpose(1, 2).contiguous()
    layer.w2_weight = layer.w2_weight.transpose(1, 2).contiguous()
    prepare_w8a8_layer(
        layer, pack_weights=lambda w1, w2: (w1.flatten().clone(), w2.flatten().clone()), pack_weights_sf=_pack_sf
    )


@pytest.mark.parametrize("prepare", [True, False])
def test_zercmoe_scale_storage_dtype_selected_before_checkpoint_loading(prepare):
    params = _make_method(prepare).get_dynamic_quant_param(2, 32, 96, torch.bfloat16)
    expected = torch.float32 if prepare else torch.bfloat16
    assert params["w13_weight_scale"].dtype == params["w2_weight_scale"].dtype == expected
    assert params["w13_weight_offset"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "method_class,eligible",
    [
        (AscendW8A8DynamicFusedMoEMethod, True),
        (AscendW8A8PDMixFusedMoeMethod, False),
        (AscendW8A8FP8DynamicFusedMoEMethod, False),
    ],
)
def test_zercmoe_only_prepares_the_confirmed_dynamic_int8_scheme(method_class, eligible):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=None),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    with (
        patch(f"{_MODULE}.get_current_vllm_config", return_value=config),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False)),
        ),
        patch(f"{_MODULE}.get_mc2_group", side_effect=AttributeError),
        patch(f"{_MODULE}.should_prepare_zercmoe_w8a8_weights", return_value=True) as preparation_gate,
    ):
        method = method_class()
    assert method._prepare_zercmoe_w8a8 is eligible
    assert preparation_gate.call_count == int(eligible)


def test_single_weight_mtp_w8a8_draft_keeps_legacy_format():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=None),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    with (
        patch(f"{_MODULE}.get_current_vllm_config", return_value=config),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False), zercmoe_single_weight=True),
        ),
        patch(f"{_MODULE}.get_mc2_group", side_effect=AttributeError),
        patch(f"{_MODULE}.should_prepare_zercmoe_w8a8_weights", return_value=True) as preparation_gate,
        zercmoe_draft_model_loading(),
    ):
        method = AscendW8A8DynamicFusedMoEMethod()
    assert not method._prepare_zercmoe_w8a8
    assert not method._zercmoe_single_weight
    preparation_gate.assert_not_called()


def test_default_a3_zerc_prepares_single_weight_without_explicit_option():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=None),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        zercmoe_single_weight=True,
    )
    layer = _make_layer()

    def pack_layer(experts):
        prepare_w8a8_layer(
            experts,
            pack_weights=lambda w1, w2: (w1.flatten().clone(), w2.flatten().clone()),
            pack_weights_sf=_pack_sf,
        )

    with (
        patch(f"{_MODULE}.get_current_vllm_config", return_value=config),
        patch(f"{_MODULE}.get_ascend_config", return_value=ascend_config),
        patch(f"{_MODULE}.get_mc2_group", side_effect=AttributeError),
        patch(f"{_MODULE}.should_prepare_zercmoe_w8a8_weights", return_value=True),
        patch(f"{_MODULE}.prepare_w8a8_layer", side_effect=pack_layer),
        patch(f"{_MODULE}.torch_npu.npu_format_cast") as cast_nz,
    ):
        method = AscendW8A8DynamicFusedMoEMethod()
        assert method._zercmoe_single_weight
        method.process_weights_after_loading(layer)

    cast_nz.assert_not_called()
    assert layer._zercmoe_weights_only
    assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
    assert layer.w13_weight_zr.dtype == layer.w2_weight_zr.dtype == torch.int8


@pytest.mark.parametrize("prepare", [True, False])
def test_zercmoe_weight_loading_packs_nd_or_keeps_legacy_nz(prepare):
    layer, method = _make_layer(), _make_method(prepare)
    if not prepare:
        layer.w13_weight_scale = layer.w13_weight_scale.to(torch.bfloat16)
        layer.w2_weight_scale = layer.w2_weight_scale.to(torch.bfloat16)
    original_w1, original_w2 = layer.w13_weight.clone(), layer.w2_weight.clone()
    original_s1, original_s2 = (
        layer.w13_weight_scale.clone(),
        layer.w2_weight_scale.clone(),
    )
    expected_w1 = original_w1.transpose(1, 2).contiguous()
    expected_w2 = original_w2.transpose(1, 2).contiguous()
    events = []

    def encode_scale(scale):
        events.append("encode")
        return _pack_sf(scale)

    def retain_nd_weights(nd_layer):
        events.append("pack")
        torch.testing.assert_close(nd_layer.w13_weight, expected_w1)
        torch.testing.assert_close(nd_layer.w2_weight, expected_w2)
        assert nd_layer.w13_weight.is_contiguous() and nd_layer.w2_weight.is_contiguous()
        prepare_w8a8_layer(
            nd_layer,
            pack_weights=lambda w1, w2: (w1.flatten().clone(), w2.flatten().clone()),
            pack_weights_sf=encode_scale,
        )
        assert nd_layer.w13_weight_zr.data_ptr() != nd_layer.w13_weight.data_ptr()
        assert nd_layer.w2_weight_zr.data_ptr() != nd_layer.w2_weight.data_ptr()

    def cast_nz(weight, _format):
        events.append("nz")
        return weight.clone()

    with (
        patch("vllm_ascend.ops.fused_moe.comm_utils.zercmoe_lib_available", return_value=False) as available,
        patch(
            "vllm_ascend.ops.fused_moe.comm_utils.load_zercmoe_ops",
            side_effect=AssertionError("must not load an operator package"),
        ) as load,
        patch(f"{_MODULE}.prepare_w8a8_layer", side_effect=retain_nd_weights) as retain,
        patch(f"{_MODULE}.torch_npu.npu_format_cast", side_effect=cast_nz),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        patch(f"{_MODULE}.use_cann_megamoe", return_value=False),
        patch(f"{_MODULE}.get_current_vllm_config"),
    ):
        method.process_weights_after_loading(layer)

    available.assert_not_called()
    load.assert_not_called()
    assert retain.call_count == int(prepare)
    if prepare:
        assert events == ["pack", "encode", "encode"]
        assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
        assert layer._zercmoe_weights_only
        torch.testing.assert_close(layer.w13_weight_zr, expected_w1.flatten())
        torch.testing.assert_close(layer.w2_weight_zr, expected_w2.flatten())
        assert layer.w13_weight_zr.shape == (2 * 96 * 64,)
        assert layer.w13_zr_shape == (2, 96, 64)
        assert layer.w2_weight_zr.shape == (2 * 32 * 96,)
        assert layer.w2_zr_shape == (2, 32, 96)
        assert layer.w13_weight_zr.dtype == layer.w2_weight_zr.dtype == torch.int8
        torch.testing.assert_close(layer.w13_weight_scale_zr, original_s1.squeeze(-1))
        torch.testing.assert_close(layer.w2_weight_scale_zr, original_s2.squeeze(-1))
        assert layer.w13_weight_scale_zr.dtype == layer.w2_weight_scale_zr.dtype == torch.float32
        torch.testing.assert_close(layer.w13_weight_scale_zr_packed, _pack_sf(original_s1.squeeze(-1)))
        torch.testing.assert_close(layer.w2_weight_scale_zr_packed, _pack_sf(original_s2.squeeze(-1)))
    else:
        assert events == ["nz", "nz"]
        torch.testing.assert_close(layer.w13_weight, expected_w1)
        torch.testing.assert_close(layer.w2_weight, expected_w2)
        torch.testing.assert_close(layer.w13_weight_scale_fp32, original_s1.squeeze(-1).to(torch.bfloat16).float())
        torch.testing.assert_close(layer.w2_weight_scale, original_s2.squeeze(-1).to(torch.bfloat16))
        assert layer.w2_weight_scale.dtype == torch.bfloat16
        assert layer.fused_w1_scale.dtype == layer.fused_w2_scale.dtype == torch.int64
        assert not hasattr(layer, "w13_weight_zr")
        assert not hasattr(layer, "w13_weight_scale_zr")


@pytest.mark.parametrize("comm_type", [MoECommType.ZERC_MOE, MoECommType.FUSED_MC2])
def test_zercmoe_apply_uses_cached_weights_only_for_zerc(comm_type):
    layer, method = _make_layer(), _make_method()
    layer.zercmoe_needs_dispatch_event = False
    layer.w13_weight_scale_fp32 = layer.w13_weight_scale.squeeze(-1)
    layer.fused_w1_scale = torch.zeros(128, dtype=torch.int64)
    layer.fused_w2_scale = torch.zeros(192, dtype=torch.int64)
    layer.w13_weight_zr = layer.w13_weight.transpose(1, 2).contiguous().flatten().clone()
    layer.w2_weight_zr = layer.w2_weight.transpose(1, 2).contiguous().flatten().clone()
    layer.w13_weight_scale_zr = layer.w13_weight_scale.squeeze(-1).clone()
    layer.w2_weight_scale_zr = layer.w2_weight_scale.squeeze(-1).clone()
    layer.w13_weight_scale_zr_packed = _pack_sf(layer.w13_weight_scale_zr).flatten()
    layer.w2_weight_scale_zr_packed = _pack_sf(layer.w2_weight_scale_zr)
    comm = Mock()
    comm.prepare_finalize = SimpleNamespace(padding_inputs=None)
    context = SimpleNamespace(moe_comm_type=comm_type, moe_comm_method=comm, use_mega_moe=False)
    x = torch.zeros(3, 96, dtype=torch.bfloat16)
    with (
        patch(f"{_MODULE}._EXTRA_CTX", context),
        patch(
            f"{_MODULE}.select_experts",
            return_value=(
                torch.full((3, 1), 0.1234567),
                torch.zeros(3, 1, dtype=torch.int64),
            ),
        ),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        patch(f"{_MODULE}.prepare_w8a8_layer") as retain,
    ):
        for _ in range(2):
            method.apply(layer, x, torch.zeros(3, 2), top_k=1, renormalize=True, num_experts=2)
    retain.assert_not_called()
    payload = comm.fused_experts.call_args.kwargs["fused_experts_input"]
    assert payload.hidden_states is x and payload.hidden_states.dtype == torch.bfloat16
    assert payload.routing.pertoken_scale is None
    if comm_type == MoECommType.ZERC_MOE:
        assert comm.fused_experts.call_args.kwargs["record_dispatch_event"] is False
        assert payload.topk_weights.dtype == torch.float32
        torch.testing.assert_close(payload.topk_weights, torch.full((3, 1), 0.1234567), rtol=0, atol=0)
        assert payload.weights.w1 is layer.w13_weight_zr
        assert payload.weights.w2 is layer.w2_weight_zr
        assert payload.weights.w1.shape == (2 * 96 * 64,)
        assert payload.weights.w2.shape == (2 * 32 * 96,)
        assert payload.weights.w1_scale is layer.w13_weight_scale_zr_packed
        assert payload.weights.w2_scale is layer.w2_weight_scale_zr_packed
        assert payload.weights.w1_scale.dtype == payload.weights.w2_scale.dtype == torch.int64
        assert payload.weights.w1_scale_bias is payload.weights.w2_scale_bias is None
    else:
        assert "record_dispatch_event" not in comm.fused_experts.call_args.kwargs
        assert payload.topk_weights.dtype == torch.bfloat16
        assert payload.weights.w1[0] is layer.w13_weight
        assert payload.weights.w2[0] is layer.w2_weight
        assert payload.weights.w1_scale[0] is layer.fused_w1_scale
        assert payload.weights.w2_scale[0] is layer.fused_w2_scale


@pytest.mark.parametrize(
    "scheme_class",
    [
        AscendW8A8DynamicFusedMoEMethod,
        AscendW8A8PDMixFusedMoeMethod,
        AscendW8A8FP8DynamicFusedMoEMethod,
    ],
)
def test_zercmoe_model_preflight_checks_every_routed_layer(scheme_class):
    good, other = _make_layer(), _make_layer()
    good.quant_method = SimpleNamespace(quant_method=_make_method())
    other.quant_method = SimpleNamespace(quant_method=object.__new__(scheme_class))
    other.quant_method.quant_method._prepare_zercmoe_w8a8 = True
    _cache_nd_weights(good)
    _cache_nd_weights(other)
    model = Mock()
    model.named_modules.return_value = [
        ("layer0", SimpleNamespace(routed_experts=good)),
        ("layer1", SimpleNamespace(routed_experts=other)),
    ]
    if scheme_class is AscendW8A8DynamicFusedMoEMethod:
        validate_w8a8_model(model)
        del other.w2_weight_zr
        with pytest.raises(ValueError, match="preparation is incomplete.*layer1"):
            validate_w8a8_model(model)
    else:
        with pytest.raises(ValueError, match="incompatible layer layer1"):
            validate_w8a8_model(model)


def test_zercmoe_single_weight_releases_each_layer_after_successful_pack():
    method = _make_method()
    method._zercmoe_single_weight = True
    layers = [_make_layer(), _make_layer()]
    expected = []
    for layer in layers:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight, requires_grad=False)
        expected.append(
            (
                layer.w13_weight.detach().transpose(1, 2).contiguous().flatten().clone(),
                layer.w2_weight.detach().transpose(1, 2).contiguous().flatten().clone(),
            )
        )
    old_parameters = []

    def pack_layer(layer):
        # The previous layer's registered parameters are gone before the next
        # layer is packed. No mock retains the tensor arguments.
        if old_parameters:
            assert all(reference() is None for reference in old_parameters[-1])
        old_parameters.append((weakref.ref(layer.w13_weight), weakref.ref(layer.w2_weight)))
        prepare_w8a8_layer(
            layer,
            pack_weights=lambda w1, w2: (w1.flatten().clone(), w2.flatten().clone()),
            pack_weights_sf=_pack_sf,
        )

    with (
        patch(f"{_MODULE}.prepare_w8a8_layer", new=pack_layer),
        patch(f"{_MODULE}.torch_npu.npu_format_cast", side_effect=AssertionError("legacy NZ conversion")),
        patch(f"{_MODULE}.scale_from_float_to_int64", side_effect=AssertionError("legacy scale encoding")),
    ):
        for layer in layers:
            method.process_weights_after_loading(layer)

    for layer, (w1, w2) in zip(layers, expected):
        assert layer._zercmoe_weights_only is True
        assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
        assert "w13_weight" not in dict(layer.named_parameters())
        assert "w2_weight" not in dict(layer.named_parameters())
        torch.testing.assert_close(layer.w13_weight_zr, w1)
        torch.testing.assert_close(layer.w2_weight_zr, w2)
        assert layer.w13_weight_scale_zr_packed.dtype == torch.int64
        assert layer.w2_weight_scale_zr_packed.dtype == torch.int64
        with pytest.raises(RuntimeError, match="cannot be reloaded"):
            method.process_weights_after_loading(layer)
    assert all(reference() is None for reference in old_parameters[-1])


@pytest.mark.parametrize("failure", ["pack", "second_scale"])
def test_zercmoe_single_weight_preserves_legacy_tensors_when_packing_fails(failure):
    method, layer = _make_method(), _make_layer()
    method._zercmoe_single_weight = True
    encodes = 0

    def pack(w1, w2):
        if failure == "pack":
            raise RuntimeError("pack failed")
        return w1.flatten().clone(), w2.flatten().clone()

    def encode(scale):
        nonlocal encodes
        encodes += 1
        if failure == "second_scale" and encodes == 2:
            raise RuntimeError("scale failed")
        return _pack_sf(scale)

    def prepare(layer):
        prepare_w8a8_layer(layer, pack_weights=pack, pack_weights_sf=encode)

    with patch(f"{_MODULE}.prepare_w8a8_layer", new=prepare), pytest.raises(RuntimeError, match="failed"):
        method.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_zr")
    assert not getattr(layer, "_zercmoe_weights_only", False)


def test_zercmoe_single_weight_preserves_packed_alias_when_old_attribute_is_released():
    method, layer = _make_method(), _make_layer()
    method._zercmoe_single_weight = True
    expected = layer.w13_weight.transpose(1, 2).contiguous().flatten().clone()

    def prepare(layer):
        prepare_w8a8_layer(layer, pack_weights=lambda w1, w2: (w1.flatten(), w2.flatten()), pack_weights_sf=_pack_sf)

    with patch(f"{_MODULE}.prepare_w8a8_layer", new=prepare):
        method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight")
    torch.testing.assert_close(layer.w13_weight_zr, expected)


def test_zercmoe_single_weight_preserves_legacy_weights_if_pack_completion_fails(monkeypatch):
    method, layer = _make_method(), _make_layer()
    method._zercmoe_single_weight = True

    def prepare(layer):
        layer.w13_weight_zr = SimpleNamespace(device=SimpleNamespace(type="npu"))

    monkeypatch.setattr(f"{_MODULE}.prepare_w8a8_layer", prepare)
    monkeypatch.setattr(
        torch.npu, "synchronize", Mock(side_effect=RuntimeError("pack completion failed")), raising=False
    )
    with pytest.raises(RuntimeError, match="pack completion failed"):
        method.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")
    assert not getattr(layer, "_zercmoe_weights_only", False)


def test_zercmoe_single_weight_rejects_direct_fallback_before_routing():
    method, layer = _make_method(), _make_layer()
    layer._zercmoe_weights_only = True
    with (
        patch(f"{_MODULE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2)),
        patch(f"{_MODULE}.select_experts", side_effect=AssertionError("must not route")),
        pytest.raises(RuntimeError, match="fallback MoE backend"),
    ):
        method.apply(layer, torch.zeros(1, 96), torch.zeros(1, 2), top_k=1, renormalize=True)


def test_zercmoe_single_weight_model_preflight_requires_released_weights():
    layer = _make_layer()
    layer.quant_method = SimpleNamespace(quant_method=_make_method())
    _cache_nd_weights(layer)
    model = Mock()
    model.named_modules.return_value = [("layer0", SimpleNamespace(routed_experts=layer))]
    with pytest.raises(ValueError, match="single-weight preparation"):
        validate_w8a8_model(model, require_single_weight=True)
    del layer.w13_weight
    del layer.w2_weight
    layer._zercmoe_weights_only = True
    validate_w8a8_model(model, require_single_weight=True)
