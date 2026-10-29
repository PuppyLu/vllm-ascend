# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A3 ZercMoE: BF16 inputs, packed INT8 weights and encoded scales.

The wheel owns activation quantization and Fixpipe encoding. Encode static
weight scales during loading and preserve the original FP32 checkpoint values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from vllm_ascend.zercmoe_config import ZERCMOE_ALIGNMENT


def needs_w8a8_dispatch_event(shared_experts, multistream_overlap: bool) -> bool:
    """Only an unquantized shared MLP on another stream consumes this event."""
    if shared_experts is None or not multistream_overlap:
        return False
    return not (
        hasattr(shared_experts.gate_up_proj, "weight_scale") and hasattr(shared_experts.down_proj, "weight_scale")
    )


def can_skip_w8a8_padding_routing(
    layer,
    hidden_states,
    router_logits,
    *,
    top_k,
    renormalize,
    use_grouped_topk,
    topk_group,
    num_expert_group,
    scoring_func,
    e_score_correction_bias,
    custom_routing_function,
    tid2eid,
    enable_force_load_balance,
):
    """Only bypass ordinary, well-formed routing on a proven empty shard."""
    if (
        custom_routing_function is not None
        or tid2eid is not None
        or enable_force_load_balance
        or getattr(layer, "zero_expert_num", 0)
        or getattr(layer, "mix_placement", False)
        or hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or router_logits.ndim != 2
        or router_logits.shape[0] != hidden_states.shape[0]
        or router_logits.device != hidden_states.device
        or router_logits.dtype not in (torch.bfloat16, torch.float32)
        or scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus")
        or not isinstance(renormalize, (bool, int))
        or renormalize not in (0, 1)
        or router_logits.shape[1] > 2048
        or not isinstance(top_k, int)
        or not 0 < top_k <= router_logits.shape[1]
    ):
        return False
    # Invalid or nonstandard routing must still reach the original selector.
    if use_grouped_topk and (topk_group is None or num_expert_group is None):
        return False
    groups = 1 if num_expert_group is None else num_expert_group
    selected_groups = 1 if topk_group is None else topk_group
    if (
        not isinstance(groups, int)
        or not isinstance(selected_groups, int)
        or not 0 < selected_groups <= groups
        or router_logits.shape[1] % groups
        or router_logits.shape[1] // groups < 2
        or top_k > selected_groups * (router_logits.shape[1] // groups)
    ):
        return False
    # Stay within the selector kernels' supported grouping geometry. Outside
    # it, keep their ordinary validation/fallback rather than hiding an error.
    per_group = router_logits.shape[1] // groups
    if selected_groups != groups and groups * ((per_group + 31) // 32 * 32) > 2048:
        return False
    if scoring_func == "sqrtsoftplus" and (
        groups != 1 or (e_score_correction_bias is not None and e_score_correction_bias.dtype != router_logits.dtype)
    ):
        return False
    return e_score_correction_bias is None or (
        e_score_correction_bias.shape == (router_logits.shape[1],)
        and e_score_correction_bias.device == router_logits.device
        and e_score_correction_bias.is_floating_point()
    )


def make_w8a8_padding_ids(*, num_experts: int, ep_world_size: int, ep_rank: int, top_k: int, device):
    """Prefer this EP rank's physical experts; spill only when topK requires it."""
    if (
        ep_world_size <= 0
        or num_experts <= 0
        or num_experts % ep_world_size
        or not 0 <= ep_rank < ep_world_size
        or not 0 < top_k <= num_experts
    ):
        raise ValueError("ZERC_MOE padding requires a valid EP identity, equal expert partitions and topK <= experts.")
    first_expert = ep_rank * (num_experts // ep_world_size)
    return (torch.arange(top_k, dtype=torch.int32, device=device) + first_expert) % num_experts


def prepare_w8a8_input(
    payload,
    *,
    all_tokens_valid: bool = False,
    all_tokens_padding: bool = False,
    padding_ids: torch.Tensor | None = None,
    padding_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Resolve real routes before inserting physical padding routes; keep BF16 inputs."""
    x = payload.hidden_states
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("ZERC_MOE requires unquantized BF16 activations [M,K].")
    if payload.routing.pertoken_scale is not None:
        raise ValueError("ZERC_MOE quantizes activations internally; external activation scales are unsupported.")
    ids, probs = payload.topk_ids, payload.topk_weights
    if (
        ids.ndim != 2
        or ids.shape[0] != x.shape[0]
        or probs.shape != ids.shape
        or ids.shape[1] == 0
        or ids.device != x.device
        or probs.device != x.device
    ):
        raise ValueError("ZERC_MOE routing must have matching [M,topK] shapes.")
    mask = payload.routing.mc2_mask
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != (x.shape[0],) or mask.device != x.device:
            raise ValueError("ZERC_MOE padding mask must match the local token shard.")
    if all_tokens_padding:
        if mask is None or all_tokens_valid or padding_inputs is None:
            raise ValueError("ZERC_MOE full-padding inputs require a padding mask and exclusive CPU metadata.")
        # These private tensors are initialized once per shape/stream. The
        # wheel reads inputs; reuse avoids three where launches on dummy ranks.
        # Physical padding IDs must never pass through the logical mapping.
        x, ids, probs = padding_inputs
        routing = payload.routing
        if routing.log2phy is not None:
            routing = replace(routing, log2phy=None)
        return replace(payload, hidden_states=x, topk_ids=ids, topk_weights=probs, routing=routing)
    # These flags come from the CPU metadata used to construct the mask.
    # Never read mask.all().item(): that would synchronize every MoE layer.
    has_padding = mask is not None and not all_tokens_valid
    routing = payload.routing
    if routing.log2phy is not None:
        # Padding can contain invalid logical IDs. Mask before the lookup,
        # then overwrite with physical IDs below; never remap local padding.
        if has_padding:
            ids = torch.where(mask[:, None], ids, 0)
        ids = routing.log2phy[ids]
        routing = replace(routing, log2phy=None)
    if has_padding:
        if (
            padding_ids is None
            or padding_ids.shape != (ids.shape[1],)
            or padding_ids.dtype != torch.int32
            or padding_ids.device != x.device
        ):
            raise ValueError("ZERC_MOE requires EP-local physical padding IDs matching topK and the input device.")
        ids = torch.where(mask[:, None], ids, padding_ids)
        probs = torch.where(mask[:, None], probs, 0)
        x = torch.where(mask[:, None], x, 0)
    return replace(
        payload,
        hidden_states=x.contiguous(),
        topk_ids=ids.to(torch.int32).contiguous(),
        topk_weights=probs.to(torch.float32).contiguous(),
        routing=routing,
    )


def prepare_w8a8_layer(layer: torch.nn.Module, pack_weights=None, pack_weights_sf=None) -> None:
    """Pack weights and encode scales once before the legacy NZ conversion."""
    w1, w2 = layer.w13_weight.data, layer.w2_weight.data
    if not all(w.dtype == torch.int8 and w.ndim == 3 and w.is_contiguous() for w in (w1, w2)):
        raise ValueError("ZERC_MOE W8A8 requires contiguous three-dimensional ND INT8 weights.")
    experts, hidden, gate_up = w1.shape
    intermediate = gate_up // 2
    if (
        experts <= 0
        or hidden <= 0
        or intermediate <= 0
        or gate_up % 2
        or hidden % ZERCMOE_ALIGNMENT
        or intermediate % ZERCMOE_ALIGNMENT
        or w2.shape != (experts, intermediate, hidden)
        or w2.device != w1.device
    ):
        raise ValueError("ZERC_MOE requires W1=[E,K,2I], W2=[E,I,K], with K and I divisible by 16.")

    scales = []
    for name, channels in (("w13", gate_up), ("w2", hidden)):
        scale = getattr(layer, f"{name}_weight_scale").data
        if (
            scale.shape not in ((experts, channels), (experts, channels, 1))
            or not scale.is_floating_point()
            or scale.device != w1.device
        ):
            raise ValueError(f"ZERC_MOE {name} scale must have shape [E,{channels}] or [E,{channels},1].")
        offset = getattr(layer, f"{name}_weight_offset").data
        # This synchronization is once at weight loading, never per forward.
        # This symmetric-weight ABI has no zero-point argument; never discard nonzero offsets.
        if torch.count_nonzero(offset).item() != 0:
            raise ValueError("ZERC_MOE W8A8 currently requires symmetric weights with zero offsets.")
        scales.append(scale.detach().reshape(experts, channels).to(dtype=torch.float32, copy=True).contiguous())

    if pack_weights is None or pack_weights_sf is None:
        # Worker-only: P, D and colocated workers use the same public packer.
        from vllm_ascend.ops.fused_moe.comm_utils import load_a3_zercmoe_ops

        backend = load_a3_zercmoe_ops()
        pack_weights = backend.pack_weights if pack_weights is None else pack_weights
        pack_weights_sf = backend.pack_weights_sf if pack_weights_sf is None else pack_weights_sf
    packed = pack_weights(w1, w2)
    if not isinstance(packed, (tuple, list)) or len(packed) != 2:
        raise RuntimeError("ZERC_MOE pack_weights must return two packed INT8 tensors.")
    for source, result in zip((w1, w2), packed):
        if (
            not isinstance(result, torch.Tensor)
            or result.dtype != torch.int8
            or result.ndim != 1
            or result.numel() < source.numel()
            or result.device != source.device
            or not result.is_contiguous()
        ):
            raise RuntimeError("ZERC_MOE pack_weights must return contiguous 1D INT8 tensors on the weight device.")
    encoded_scales = []
    for scale in scales:
        # The public wheel otherwise does D2H -> CPU encoding -> H2D on every
        # FP32 call. Move that work here, using its exact encoder and layout.
        # Give the encoder a private copy so in-place work cannot alter raw FP32.
        encoded = pack_weights_sf(scale.clone())
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.dtype != torch.int64
            or encoded.numel() != scale.numel()
            or not encoded.is_contiguous()
        ):
            raise RuntimeError("ZERC_MOE pack_weights_sf must return contiguous INT64 slots matching the scale count.")
        if encoded.device != w1.device:
            if encoded.device.type != "cpu":
                raise RuntimeError("ZERC_MOE encoded scales must be on the weight device or returned on CPU.")
            encoded = encoded.to(device=w1.device)
        encoded_scales.append(encoded)
    # Publish only after both weights, raw scales and encoded scales validated.
    layer.w13_weight_zr, layer.w2_weight_zr = packed
    layer.w13_zr_shape, layer.w2_zr_shape = tuple(w1.shape), tuple(w2.shape)
    layer.w13_weight_scale_zr, layer.w2_weight_scale_zr = scales
    layer.w13_weight_scale_zr_packed, layer.w2_weight_scale_zr_packed = encoded_scales


def validate_w8a8_model(model: torch.nn.Module, *, require_single_weight: bool = False) -> None:
    """Check all routed layers after loading, before SHMEM bootstrap."""
    # Worker-only, after model loading; avoid a circular scheme/adapter import.
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

    found = False
    for name, module in model.named_modules():
        layer = getattr(module, "routed_experts", None)
        if layer is None:
            continue
        found = True
        method = getattr(layer, "quant_method", None)
        scheme = getattr(method, "quant_method", method)
        if type(scheme) is not AscendW8A8DynamicFusedMoEMethod or not scheme._prepare_zercmoe_w8a8:
            raise ValueError(
                f"ZERC_MOE A3 requires W8A8_DYNAMIC routed experts in every layer; incompatible layer {name}."
            )
        required = [
            "w13_weight_scale_zr",
            "w2_weight_scale_zr",
            "w13_weight_scale_zr_packed",
            "w2_weight_scale_zr_packed",
            "w13_weight_zr",
            "w2_weight_zr",
            "w13_zr_shape",
            "w2_zr_shape",
        ]
        if any(not hasattr(layer, attr) for attr in required):
            raise ValueError(f"ZERC_MOE A3 weight preparation is incomplete for layer {name}.")
        if require_single_weight and (
            not getattr(layer, "_zercmoe_weights_only", False)
            or hasattr(layer, "w13_weight")
            or hasattr(layer, "w2_weight")
        ):
            raise ValueError(f"ZERC_MOE single-weight preparation is incomplete for layer {name}.")
    if not found:
        raise ValueError("ZERC_MOE A3 did not find any routed expert layers to validate.")


def validate_single_weight_mtp(target: torch.nn.Module, draft: torch.nn.Module) -> None:
    """Reject draft experts that share or lost target-only expert weights."""
    target_layers = [getattr(module, "routed_experts", None) for module in target.modules()]
    target_layers = [layer for layer in target_layers if layer is not None]
    target_ids = {id(layer) for layer in target_layers}

    def storage_key(tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        return tensor.device, tensor.untyped_storage().data_ptr()

    target_storages = {
        key
        for layer in target_layers
        for name in ("w13_weight_zr", "w2_weight_zr")
        if (key := storage_key(getattr(layer, name, None))) is not None
    }
    found = False
    for module in draft.modules():
        layer = getattr(module, "routed_experts", None)
        if layer is None:
            continue
        found = True
        if id(layer) in target_ids or getattr(layer, "_zercmoe_weights_only", False):
            raise ValueError("A3 ZercMoE single-weight requires MTP experts independent of target experts.")
        if hasattr(layer, "w13_weight_zr") or hasattr(layer, "w2_weight_zr"):
            raise ValueError("A3 ZercMoE single-weight requires MTP experts to retain only their original format.")
        for name in ("w13_weight", "w2_weight"):
            weights = getattr(layer, name, None)
            if weights is None:
                weights = getattr(layer, name + "_list", None)
            if weights is None:
                raise ValueError(f"A3 ZercMoE single-weight MTP expert {name} is missing.")
            tensors = weights if isinstance(weights, (list, tuple, torch.nn.ParameterList)) else (weights,)
            if len(tensors) == 0:
                raise ValueError(f"A3 ZercMoE single-weight MTP expert {name} is empty.")
            for tensor in tensors:
                key = storage_key(tensor)
                if key is None:
                    raise ValueError(f"A3 ZercMoE single-weight MTP expert {name} is not a non-empty tensor.")
                if key in target_storages:
                    raise ValueError("A3 ZercMoE single-weight does not support shared target/MTP expert storage.")
    if not found:
        raise ValueError("A3 ZercMoE single-weight did not find MTP routed experts to validate.")


def validate_w8a8_payload(payload, *, num_experts: int, ep_world_size: int, hidden: int, intermediate: int) -> None:
    """Validate BF16 activations and packed weights before SHMEM bootstrap."""
    x = payload.hidden_states
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != hidden or not x.is_contiguous():
        raise ValueError("ZERC_MOE W8A8 requires contiguous BF16 activations [M,K].")
    if payload.routing.pertoken_scale is not None:
        raise ValueError("ZERC_MOE W8A8 does not accept externally quantized activation scales.")
    if (
        ep_world_size <= 1
        or num_experts <= 0
        or num_experts % ep_world_size
        or hidden <= 0
        or intermediate <= 0
        or hidden % ZERCMOE_ALIGNMENT
        or intermediate % ZERCMOE_ALIGNMENT
    ):
        raise ValueError("ZERC_MOE W8A8 requires equal expert partitions and K/I divisible by 16.")
    experts = num_experts // ep_world_size
    for weight, scale, channels, shape in (
        (payload.weights.w1, payload.weights.w1_scale, 2 * intermediate, (experts, hidden, 2 * intermediate)),
        (payload.weights.w2, payload.weights.w2_scale, hidden, (experts, intermediate, hidden)),
    ):
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.int8
            or weight.ndim != 1
            or weight.numel() < shape[0] * shape[1] * shape[2]
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            raise ValueError("ZERC_MOE W8A8 requires contiguous packed 1D INT8 weights from pack_weights().")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.int64
            or scale.numel() != experts * channels
            or scale.device != x.device
            or not scale.is_contiguous()
        ):
            raise ValueError("ZERC_MOE W8A8 requires pre-encoded contiguous INT64 scales with E*2I and E*K slots.")
    if (
        payload.topk_ids.ndim != 2
        or payload.topk_ids.shape[0] != x.shape[0]
        or payload.topk_weights.shape != payload.topk_ids.shape
        or payload.topk_ids.device != x.device
        or payload.topk_weights.device != x.device
        or not 0 < payload.topk_ids.shape[1] <= num_experts
    ):
        raise ValueError("ZERC_MOE routing tensors must have matching [M,topK] shapes and devices.")
    activation = getattr(payload.activation, "value", payload.activation)
    if (
        activation not in ("silu", "swiglu")
        or payload.swiglu_limit != 0.0
        or payload.swiglu_alpha != 1.0
        or payload.swiglu_beta != 0.0
        or payload.routing.apply_router_weight_on_input
        or payload.dynamic_eplb
        or payload.lora_context is not None
        or any(
            getattr(payload.weights, name) is not None
            for name in ("w1_bias", "w2_bias", "w1_scale_bias", "w2_scale_bias", "w1_offset", "w2_offset")
        )
    ):
        raise ValueError("ZERC_MOE W8A8 supports symmetric weights and standard SwiGLU without EPLB/LoRA.")


def validate_w8a8_backend(ops_module):
    """Check that the installed package exposes the APIs used by the framework."""
    for name in ("pack_weights", "pack_weights_sf", "get_symm_buffer", "mega_moe"):
        if not callable(getattr(ops_module, name, None)):
            raise RuntimeError(f"ZERC_MOE A3 requires callable zercmoe.{name}.")
    return ops_module


def get_w8a8_operator(ops_module=None):
    """Bind the wheel's public API; do not load the hand-written CANN ABI."""
    if ops_module is None:
        from vllm_ascend.ops.fused_moe.comm_utils import load_a3_zercmoe_ops

        ops_module = load_a3_zercmoe_ops()
    validate_w8a8_backend(ops_module)
    return W8A8Launcher(ops_module.mega_moe)


@dataclass(frozen=True)
class W8A8Launcher:
    operator: Callable

    def __call__(self, payload, sym_buffer):
        """Use cached scales and skip the wheel's synchronous count collection."""
        out, expert_tokens = self.operator(
            payload.hidden_states,
            payload.topk_ids,
            payload.topk_weights,
            payload.weights.w1,
            payload.weights.w2,
            sym_buffer,
            # Dynamic EPLB is excluded by the A3 gate and payload validator,
            # so no consumer needs counts. Requesting them synchronizes the
            # wheel's stream and copies the count table through the host.
            return_expert_tokens=0,
            l1_weights_sf=payload.weights.w1_scale,
            l2_weights_sf=payload.weights.w2_scale,
        )
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != payload.hidden_states.shape
            or out.dtype != payload.hidden_states.dtype
            or out.device != payload.hidden_states.device
            or not isinstance(expert_tokens, torch.Tensor)
            or expert_tokens.dtype != torch.int32
            or expert_tokens.shape != (0,)
        ):
            raise RuntimeError(
                "ZERC_MOE must return matching BF16 output and an empty INT32 count tensor (shape (0,))."
            )
        # The 1.5 asynchronous API returns an empty count placeholder; preserve
        # it without a host copy, scalar read or device requirement.
        # Wheel exceptions propagate; a failed call is never retried/fallen back.
        return out, expert_tokens
