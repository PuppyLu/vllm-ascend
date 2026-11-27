# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Names and storage helpers for the MegaKernelMc2 Torch ABI.

The order must match the generated ACLNN header and the Torch extension.
Aligned allocations include the tail required by the extension's DMA checks.
"""

import math

import torch

MC2_INPUT_NAMES = (
    "x",
    "hc_fn",
    "hc_scale",
    "hc_base",
    "post_h_res",
    "post_h_out",
    "post_h_post",
    "norm_weight",
    "projection_weight",
    "conv_weight",
    "conv_state",
    "raw_gate",
    "recurrent_state",
    "cu_seqlens",
    "state_indices",
    "a_log",
    "dt_bias",
    "g_b_weight",
    "o_norm_weight",
    "o_weight",
    "f_b_weight",
    "ffn_c0_diagnostic",
    "ffn_hc_fn",
    "ffn_hc_scale",
    "ffn_hc_base",
    "ffn_norm_weight",
    "router_weight",
    "router_bias",
    "expert_weight",
    "expert_channel_scale",
    "expert_down_weight",
    "expert_down_channel_scale",
    "top8_rows",
    "top8_expert_ids",
    "top8_permutation",
    "batch_expert_weight",
    "batch_expert_channel_scale",
    "batch_expert_down_weight",
    "batch_expert_down_channel_scale",
    "o_quant_weight",
    "o_quant_channel_scale",
    "in_projection_scale",
    "g_projection_scale",
    "o_projection_scale",
    "f_projection_scale",
    "external_route_ids",
    "external_route_weights",
)

ATTR_NAMES = (
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "norm_eps",
    "layer_rms_eps",
    "stage_limit",
    "conv_slot",
    "lower_bound",
    "output_norm_eps",
    "ffn_hc_eps",
    "ffn_norm_eps",
    "ffn_layer_rms_eps",
    "mhc_post_mult_value",
    "router_mode",
    "expert_id",
    "route_ordinal",
    "down_expert_id",
    "ep_world_size",
)

_ALIGNMENT_BYTES = 512
_TAIL_BYTES = 512
_MODEL_HIDDEN = 4096
_ROUTED_INTERMEDIATE = 2048
_ROUTED_EXPERTS = 288


def aligned_empty(shape, dtype, device):
    """Allocate positive contiguous storage aligned to the Torch MC2 ABI."""
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise ValueError("MC2 buffer dimensions must be positive")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    if _ALIGNMENT_BYTES % element_bytes:
        raise ValueError("dtype element size is incompatible with MC2 alignment")
    count = math.prod(shape)
    reserve = (_ALIGNMENT_BYTES + _TAIL_BYTES) // element_bytes
    storage = torch.empty(count + reserve, dtype=dtype, device=device)
    delta = (-storage.data_ptr()) % _ALIGNMENT_BYTES
    if delta % element_bytes:
        raise RuntimeError("allocator returned an unsupported pointer alignment")
    first = delta // element_bytes
    return storage.narrow(0, first, count).view(shape)


def aligned_copy(tensor):
    """Copy a tensor to an independently owned, padded MC2 buffer."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("MC2 buffer must be a tensor")
    result = aligned_empty(tuple(tensor.shape), tensor.dtype, tensor.device)
    result.copy_(tensor)
    return result


def _is_nz(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu":
        return False
    import torch_npu

    return int(torch_npu.get_npu_format(tensor)) == 29


def validate_expert_bank(inputs, rank, world, layout, *, shared_width=2048):
    """Validate the existing BF16/W8A8 expert partition and layout contract."""
    if (
        isinstance(rank, bool)
        or isinstance(world, bool)
        or not isinstance(rank, int)
        or not isinstance(world, int)
        or not 1 <= world <= 256
        or not 0 <= rank < world
        or layout not in ("replicated", "sharded")
    ):
        raise ValueError("invalid expert partition or layout")
    count = (
        (rank + 1) * _ROUTED_EXPERTS // world - rank * _ROUTED_EXPERTS // world
        if layout == "sharded"
        else _ROUTED_EXPERTS
    )
    weight_names = ("expert_weight", "expert_down_weight", "batch_expert_weight", "batch_expert_down_weight")
    scale_names = (
        "expert_channel_scale",
        "expert_down_channel_scale",
        "batch_expert_channel_scale",
        "batch_expert_down_channel_scale",
    )
    first = inputs.get(weight_names[0])
    if not isinstance(first, torch.Tensor) or first.dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("expert weights must be BF16 or INT8")
    for name in weight_names:
        tensor = inputs.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != first.dtype:
            raise ValueError(f"expert weight dtype mismatch: {name}")
    if first.dtype == torch.bfloat16:
        if any(inputs.get(name) is not None for name in scale_names):
            raise ValueError("BF16 experts must not supply channel scales")
    else:
        for name in ("expert_channel_scale", "batch_expert_channel_scale"):
            tensor = inputs.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                raise ValueError(f"INT8 experts require FP32 scale: {name}")
        shared_down = inputs.get("expert_down_channel_scale")
        routed_down = inputs.get("batch_expert_down_channel_scale")
        if (
            not isinstance(shared_down, torch.Tensor)
            or not isinstance(routed_down, torch.Tensor)
            or shared_down.dtype not in (torch.float32, torch.bfloat16)
            or routed_down.dtype != shared_down.dtype
        ):
            raise ValueError("INT8 shared and routed down scales must have the same BF16 or FP32 dtype")

    up = inputs["batch_expert_weight"]
    down = inputs["batch_expert_down_weight"]
    nz_up, nz_down = _is_nz(up), _is_nz(down)
    if nz_up != nz_down or (nz_up and first.dtype != torch.int8):
        raise ValueError("routed expert weights require paired INT8 FRACTAL_NZ or paired ND")
    native = tuple(down.shape) == (count, _ROUTED_INTERMEDIATE, _MODEL_HIDDEN)
    down_shape = (
        (count, _ROUTED_INTERMEDIATE, _MODEL_HIDDEN) if native else (count, _MODEL_HIDDEN, _ROUTED_INTERMEDIATE)
    )
    shapes = {
        "expert_weight": (2 * shared_width, _MODEL_HIDDEN),
        "expert_down_weight": (_MODEL_HIDDEN, shared_width),
        "batch_expert_weight": (count, _MODEL_HIDDEN, _MODEL_HIDDEN),
        "batch_expert_down_weight": down_shape,
    }
    if first.dtype == torch.int8:
        shapes.update(
            expert_channel_scale=(2 * shared_width,),
            expert_down_channel_scale=(_MODEL_HIDDEN,),
            batch_expert_channel_scale=(count, _MODEL_HIDDEN),
            batch_expert_down_channel_scale=(count, _MODEL_HIDDEN),
        )
    for name, shape in shapes.items():
        tensor = inputs[name]
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"invalid {layout} expert tensor: {name}")
