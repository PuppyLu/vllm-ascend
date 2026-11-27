# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM5Next -> MC2 static weight mapping prototype, vLLM-Ascend ebaaeeec5.

Only loaded weights are mapped here; per-step inputs, live states, routes,
communication groups, MC2 invocation, graph lifetime and refit invalidation
are the integrator's responsibility. This code has NOT been tested on A3.
Call only after vLLM's process_weights_after_loading; expert matrices share
the original contiguous ND [E,K,N] or INT8 NZ [E,K,N] storage directly.
Neither form copies an expert bank.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import torch


class WeightContractError(ValueError):
    pass


def _tensor(obj, name):
    value = getattr(obj, name, None)
    if not isinstance(value, torch.Tensor):
        raise WeightContractError(f"missing tensor {type(obj).__name__}.{name}")
    if value.device.type not in ("npu", "cpu"):
        raise WeightContractError(f"unsupported tensor device for {name}: {value.device}")
    return value.detach()


def _nd(value, name, *, allow_nz):
    """Cast a confirmed NPU internal format only when explicitly allowed."""
    if value.device.type == "npu":
        import torch_npu

        fmt = int(torch_npu.get_npu_format(value))
        if fmt not in (0, 2):
            if not allow_nz:
                raise WeightContractError(f"{name}: internal NPU format {fmt}; ND conversion not enabled")
            value = torch_npu.npu_format_cast(value, 2)
            if int(torch_npu.get_npu_format(value)) not in (0, 2):
                raise WeightContractError(f"{name}: conversion to ND failed")
    return value


def _weight(obj, name, shape, dtype, *, allow_nz):
    value = _nd(_tensor(obj, name), name, allow_nz=allow_nz)
    if value.dtype != dtype or tuple(value.shape) != tuple(shape):
        raise WeightContractError(f"{name}: expected {dtype} {tuple(shape)}, got {value.dtype} {tuple(value.shape)}")
    return value.contiguous()


def _expert_format(value):
    if value.device.type != "npu":
        return 2
    import torch_npu

    return int(torch_npu.get_npu_format(value))


def _native_expert_weight(obj, name, native_shape, dtype):
    """Borrow contiguous ND or INT8 NZ native [E,K,N] storage directly.

    Prefill keeps using the original tensor. Never cast or materialize an
    expert bank: either operation duplicates its entire storage.
    """
    value = _tensor(obj, name)
    fmt = _expert_format(value)
    if value.dtype != dtype or tuple(value.shape) != tuple(native_shape):
        raise WeightContractError(
            f"{name}: expected {dtype} native {tuple(native_shape)}, got {value.dtype} {tuple(value.shape)}"
        )
    if fmt not in (0, 2, 29) or (fmt == 29 and dtype != torch.int8):
        raise WeightContractError(f"{name}: unsupported expert storage format {fmt}")
    if not value.is_contiguous():
        raise WeightContractError(f"{name}: native [E,K,N] expert storage must be contiguous")
    _validate_native_expert(value)
    return value


def _validate_native_expert(value):
    """Validate a contiguous [E,K,N] ND or INT8 NZ expert bank."""
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise WeightContractError("expert bank must be a three-dimensional tensor")
    fmt = _expert_format(value)
    if fmt == 29:
        if value.dtype != torch.int8 or not value.is_contiguous():
            raise WeightContractError("FRACTAL_NZ expert bank must be contiguous INT8")
        return value
    if fmt not in (0, 2):
        raise WeightContractError(f"unsupported expert storage format {fmt}")
    if min(value.shape) <= 0 or not value.is_contiguous():
        raise WeightContractError("expert bank must have contiguous native [E,K,N] storage")
    return value


def _expert_weight_pair(expert, up_shape, down_shape, dtype):
    up = _native_expert_weight(expert, "w13_weight", up_shape, dtype)
    down = _native_expert_weight(expert, "w2_weight", down_shape, dtype)
    if (_expert_format(up) == 29) != (_expert_format(down) == 29):
        raise WeightContractError("routed expert banks must both be ND or both be INT8 FRACTAL_NZ")
    return up, down


def _reuse_or_align_weights(weights, device, check_tensor, aligned_copy):
    """Bind compatible weights without ever copying a large expert bank."""
    mapped = {}
    spans = []
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor):
            mapped[name] = value
            continue
        if name in ("batch_expert_weight", "batch_expert_down_weight"):
            # Prefill owns this storage. Fail on invalid alignment, padding,
            # layout or aliasing instead of allocating another expert bank.
            _validate_native_expert(value)
            check_tensor(value, device, allow_expert_view=True)
            start, end = value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()
            if any(start < previous_end and previous_start < end for previous_start, previous_end in spans):
                raise ValueError("expert weight overlaps another ABI input")
            mapped[name] = value
            spans.append((start, end))
            continue
        try:
            check_tensor(value, device)
            start, end = value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()
            if any(start < previous_end and previous_start < end for previous_start, previous_end in spans):
                raise ValueError("static input overlaps another ABI input")
            mapped[name] = value
        except ValueError:
            mapped[name] = aligned_copy(value)
        start = mapped[name].data_ptr()
        spans.append((start, start + mapped[name].numel() * mapped[name].element_size()))
    return mapped


def _scheme(module):
    method = getattr(module, "quant_method", None)
    return getattr(method, "quant_method", None)


@lru_cache(maxsize=1)
def _w8a8_code_is_verified():
    # The inspected ebaaeeec5 W8A8_DYNAMIC forward uses weight and scale,
    # but never reads weight_offset (which may be uninitialized). Refuse other
    # code versions until their actual forward semantics are independently checked.
    path = Path(__file__).resolve().parents[2] / "quantization/methods/w8a8/w8a8_dynamic.py"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False
    return digest == "f25ce677d8a1149949e5cd2fc82a4d1a9aac8ad6f02fd987663aea6c1f924343"


def _dynamic_scheme(module, expected):
    scheme = _scheme(module)
    cls = type(scheme)
    return (
        cls.__name__ == expected
        and cls.__module__ == "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic"
        and getattr(scheme, "act_quant_type", None) == torch.int8
        and _w8a8_code_is_verified()
    )


def _scale(module, name, shape, *, allow_nz, projection=False):
    value = _nd(_tensor(module, name), name, allow_nz=allow_nz)
    if tuple(value.shape) != tuple(shape) or value.dtype not in (torch.bfloat16, torch.float32):
        raise WeightContractError(f"{name}: unsupported scale shape or dtype")
    if not bool(torch.isfinite(value).all().item()) or bool(torch.any(value < 0).item()):
        raise WeightContractError(f"{name}: non-finite or negative scale")
    if projection:
        # MC2 INT8 projection scale ABI is BF16. Refuse lossy conversion.
        converted = value.to(torch.bfloat16)
        if not bool(torch.equal(converted.float(), value.float())):
            raise WeightContractError(f"{name}: FP32 scale cannot be represented exactly in BF16")
        return converted.contiguous()
    # FP32 is accepted for both down scales; up scales are required FP32.
    return value.float().contiguous()


def _projection_pair(module, out_size, in_size, *, allow_nz, projection=False):
    value = _tensor(module, "weight")
    if getattr(module, "bias", None) is not None:
        raise WeightContractError("MC2 projection does not accept a bias")
    if value.dtype == torch.bfloat16:
        if _dynamic_scheme(module, "AscendW8A8DynamicLinearMethod"):
            raise WeightContractError("dynamic-quantized linear has unexpected BF16 weight")
        return _weight(module, "weight", (out_size, in_size), torch.bfloat16, allow_nz=allow_nz), None
    if value.dtype != torch.int8 or not _dynamic_scheme(module, "AscendW8A8DynamicLinearMethod"):
        raise WeightContractError("only BF16 or verified Ascend W8A8_DYNAMIC linear weights are supported")
    # W8A8_DYNAMIC process_weights_after_loading transposes [N,K] -> [K,N].
    value = _weight(module, "weight", (in_size, out_size), torch.int8, allow_nz=allow_nz)
    scale = _scale(module, "weight_scale", (out_size,), allow_nz=allow_nz, projection=projection)
    return value.transpose(0, 1).contiguous(), scale


def _projection(module, out_size, in_size, *, allow_nz):
    value, scale = _projection_pair(module, out_size, in_size, allow_nz=allow_nz)
    if scale is not None:
        raise WeightContractError("this MC2 field requires an unquantized BF16 weight")
    return value


def _known_unquantized_expert(layer):
    method = getattr(layer, "quant_method", None)
    return (
        type(method).__name__ == "AscendUnquantizedFusedMoEMethod"
        and type(method).__module__ == "vllm_ascend.ops.fused_moe.routed_experts"
    )


def map_static_weights(layer, *, ep_rank, ep_world, allow_nz=False, expert_postprocessed=False):
    """Map one real *middle* KDA/MoE/mHC layer's immutable BF16 weights.

    The caller must explicitly attest vLLM post-load processing is complete.
    Accept verified BF16 or Ascend W8A8_DYNAMIC, EP-contiguous experts, no
    EPLB/PCP/fine-grained projection/sequence parallel. W8A8 uses the
    pinned official implementation's weight-and-scale semantics; it does not
    claim support for other asymmetric offset semantics. This is not a
    validation of numerical parity.
    Set allow_nz to use torch_npu's official format conversion (not validated
    here). Checkpoint updates invalidate every returned pointer; recreate only
    at a safe session boundary and re-run group admission after any refit.
    """
    if type(ep_world) is not int or type(ep_rank) is not int or not 1 <= ep_world <= 256 or not 0 <= ep_rank < ep_world:
        raise WeightContractError("invalid EP rank/world")
    if not expert_postprocessed:
        raise WeightContractError("explicitly verify vLLM expert post-load transposition before mapping")
    if (
        getattr(layer, "layer_kind", None) != "kda"
        or not getattr(layer, "_mlp_is_moe", False)
        or not getattr(layer, "mhc", False)
        or getattr(layer, "is_mtp_layer", False)
    ):
        raise WeightContractError("expected non-MTP KDA + MoE + mHC")
    if layer.hidden_size != 4096 or layer.mhc_num_residual_streams != 4 or layer.is_sequence_parallel:
        raise WeightContractError("unsupported hidden/mHC/sequence-parallel configuration")
    attn, moe = layer.self_attn, layer.mlp
    if getattr(attn, "use_pcp_head_sharding", False) or getattr(attn, "projection_tp", None) is not None:
        raise WeightContractError("PCP or fine-grained input-projection TP not mapped")
    if attn.head_dim != 128 or attn.conv_size != 4:
        raise WeightContractError("expected head_dim=128 and conv kernel width=4")
    heads = int(attn.local_num_heads)
    if not 1 <= heads <= 64:
        raise WeightContractError("invalid local KDA heads")
    if getattr(moe, "enable_eplb", False) or getattr(moe, "n_redundant_experts", 0) != 0:
        raise WeightContractError("EPLB/redundant experts not mapped")
    if moe.n_routed_experts != 288 or moe.ep_size != ep_world or moe.ep_rank != ep_rank:
        raise WeightContractError("model expert count or EP group mismatch")
    lo, hi = ep_rank * 288 // ep_world, (ep_rank + 1) * 288 // ep_world
    if (moe.physical_expert_start, moe.physical_expert_end) != (lo, hi):
        raise WeightContractError("vLLM physical expert ownership differs from MC2 contiguous partition")
    if moe.shared_experts is None:
        raise WeightContractError("shared expert absent")
    shared = moe.shared_experts
    gate_up = _tensor(shared.gate_up_proj, "weight")
    if gate_up.ndim != 2:
        raise WeightContractError("shared gate/up shape invalid")
    gate_up_n = gate_up.shape[1] if gate_up.dtype == torch.int8 else gate_up.shape[0]
    if gate_up_n % 2:
        raise WeightContractError("shared gate/up shape invalid")
    width = int(gate_up_n // 2)
    if not 1 <= width <= 2048 or width % 128:
        raise WeightContractError("MC2 compact shared width must be 128-aligned, in 1..2048")
    expert = getattr(moe.experts, "routed_experts", None)
    if expert is None:
        raise WeightContractError("missing vLLM routed_experts")
    actual_map = getattr(expert, "ascend_expert_map", None)
    if actual_map is None:
        if ep_world != 1:
            raise WeightContractError("multi-rank EP requires the actual global-to-local expert map")
    else:
        if not isinstance(actual_map, torch.Tensor) or actual_map.ndim != 1 or actual_map.numel() != 288:
            raise WeightContractError("invalid vLLM expert placement map")
        observed = actual_map.detach().cpu().tolist()
        expected = [i - lo if lo <= i < hi else -1 for i in range(288)]
        if observed != expected:
            raise WeightContractError("vLLM expert placement is not the MC2 contiguous expert partition")
    expert_weight = _tensor(expert, "w13_weight")
    if expert_weight.dtype == torch.bfloat16:
        if not _known_unquantized_expert(expert):
            raise WeightContractError("unsupported BF16 expert implementation")
        expert_dtype = torch.bfloat16
        routed_up_scale = routed_down_scale = None
    elif expert_weight.dtype == torch.int8 and _dynamic_scheme(expert, "AscendW8A8DynamicFusedMoEMethod"):
        expert_dtype = torch.int8
        routed_up_scale = _scale(expert, "w13_weight_scale", (hi - lo, 4096), allow_nz=allow_nz)
        routed_down_scale = _scale(expert, "w2_weight_scale", (hi - lo, 4096), allow_nz=allow_nz)
    else:
        raise WeightContractError("routed expert type is not verified BF16 or W8A8_DYNAMIC")
    # vLLM post-load: checkpoint [E,N,K] -> native [E,K,N], ND or INT8 NZ.
    # Both layouts are passed in their native shape with contiguous strides.
    # Never convert whole expert banks. Keep original weights for prefill.
    w13, w2 = _expert_weight_pair(expert, (hi - lo, 4096, 4096), (hi - lo, 2048, 4096), expert_dtype)
    shared_up, shared_up_scale = _projection_pair(shared.gate_up_proj, width * 2, 4096, allow_nz=allow_nz)
    shared_down, shared_down_scale = _projection_pair(shared.down_proj, 4096, width, allow_nz=allow_nz)
    if shared_up.dtype != expert_dtype or shared_down.dtype != expert_dtype:
        raise WeightContractError("shared and routed expert dtypes must agree")
    p_weight, p_scale = _projection_pair(
        attn.in_proj_qkvbfg_a, 385 * heads + 256, 4096, allow_nz=allow_nz, projection=True
    )
    g_weight, g_scale = _projection_pair(attn.g_b_proj, heads * 128, 128, allow_nz=allow_nz, projection=True)
    o_weight, o_scale = _projection_pair(attn.o_proj, 4096, heads * 128, allow_nz=allow_nz, projection=True)
    f_weight, f_scale = _projection_pair(attn.f_b_proj, heads * 128, 128, allow_nz=allow_nz, projection=True)
    conv = _tensor(attn, "_merged_conv_weight")
    if conv.dtype != torch.bfloat16 or tuple(conv.shape) != (4, heads * 384):
        raise WeightContractError("post-load merged QKV convolution must be BF16 [4,384*local_heads]")
    a_log = _tensor(attn, "A_log")
    dt_bias = _tensor(attn, "dt_bias")
    if (
        a_log.dtype != torch.float32
        or a_log.numel() != heads
        or dt_bias.dtype != torch.float32
        or dt_bias.numel() != heads * 128
    ):
        raise WeightContractError("A_log/dt_bias do not match local KDA heads")
    values = {
        "hc_fn": _weight(layer, "hc_attn_fn", (24, 16384), torch.float32, allow_nz=allow_nz),
        "hc_scale": _weight(layer, "hc_attn_scale", (3,), torch.float32, allow_nz=allow_nz),
        "hc_base": _weight(layer, "hc_attn_base", (24,), torch.float32, allow_nz=allow_nz),
        "norm_weight": _weight(layer.input_layernorm, "weight", (4096,), torch.bfloat16, allow_nz=allow_nz),
        "projection_weight": p_weight,
        "conv_weight": _nd(conv, "conv_weight", allow_nz=allow_nz).contiguous(),
        "a_log": _nd(a_log, "a_log", allow_nz=allow_nz).reshape(heads).contiguous(),
        "dt_bias": _nd(dt_bias, "dt_bias", allow_nz=allow_nz).reshape(heads, 128).contiguous(),
        "g_b_weight": g_weight,
        "o_norm_weight": _weight(attn.o_norm, "weight", (128,), torch.bfloat16, allow_nz=allow_nz),
        "o_weight": o_weight,
        "f_b_weight": f_weight,
        "ffn_hc_fn": _weight(layer, "hc_ffn_fn", (24, 16384), torch.float32, allow_nz=allow_nz),
        "ffn_hc_scale": _weight(layer, "hc_ffn_scale", (3,), torch.float32, allow_nz=allow_nz),
        "ffn_hc_base": _weight(layer, "hc_ffn_base", (24,), torch.float32, allow_nz=allow_nz),
        "ffn_norm_weight": _weight(
            layer.post_attention_layernorm, "weight", (4096,), torch.bfloat16, allow_nz=allow_nz
        ),
        "router_weight": _projection(moe.gate, 288, 4096, allow_nz=allow_nz),
        "router_bias": _weight(moe.gate, "e_score_correction_bias", (288,), torch.float32, allow_nz=allow_nz),
        "expert_weight": shared_up,
        "expert_down_weight": shared_down,
        "batch_expert_weight": w13,
        "batch_expert_down_weight": w2,
        "expert_channel_scale": shared_up_scale,
        "expert_down_channel_scale": shared_down_scale,
        "batch_expert_channel_scale": routed_up_scale,
        "batch_expert_down_channel_scale": routed_down_scale,
        "in_projection_scale": p_scale,
        "g_projection_scale": g_scale,
        "o_projection_scale": o_scale,
        "f_projection_scale": f_scale,
        "o_quant_weight": None,
        "o_quant_channel_scale": None,
    }
    devices = {v.device for v in values.values() if isinstance(v, torch.Tensor)}
    if len(devices) != 1:
        raise WeightContractError("mapped tensors span devices")
    return values
