# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Names and storage helpers for the MegaKernelMc2 Torch ABI.

The order must match the generated ACLNN header and the Torch extension.
Aligned allocations include the tail required by the extension's DMA checks.
"""

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from typing import Any

import regex as re
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
    native = tuple(down.shape) == (count, _ROUTED_INTERMEDIATE, _MODEL_HIDDEN)
    if nz_up != nz_down or (nz_up and not native):
        raise ValueError("routed expert weights require paired native [E,K,N] FRACTAL_NZ or paired ND")
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


PROTOCOL = "stage24-v9-tp1-ep-no-padding-bf16-w8a8-v1"
PADDING_PROTOCOL = "stage25-tp1-ep-padding-bf16-w8a8-v2"
SHARDED_SHARED_PROTOCOL = "stage26-overlap-tp-ep-padded-shared-v1"
GRID_PROTOCOL = "stage27-tp-ep-grid-padded-shared-v1"
NATIVE_SHARED_PROTOCOL = "stage28-tp-ep-grid-native-shared-v1"
GENERAL_PROTOCOL = "stage29-irregular-tp-ep-native-shared-v1"

PROTOCOL_STAGE_MAP = {
    PROTOCOL: 24,
    PADDING_PROTOCOL: 25,
    SHARDED_SHARED_PROTOCOL: 26,
    GRID_PROTOCOL: 27,
    NATIVE_SHARED_PROTOCOL: 28,
    GENERAL_PROTOCOL: 29,
}

_HEX256_RE = re.compile(r"^[0-9a-f]{64}$")

PROJECTION_PAIRS = (
    ("projection_weight", "in_projection_scale"),
    ("g_b_weight", "g_projection_scale"),
    ("o_weight", "o_projection_scale"),
    ("f_b_weight", "f_projection_scale"),
)

INT_ATTR_NAMES = (
    "hc_mult",
    "hc_sinkhorn_iters",
    "stage_limit",
    "conv_slot",
    "router_mode",
    "expert_id",
    "route_ordinal",
    "down_expert_id",
    "ep_world_size",
)

FLOAT_ATTR_NAMES = tuple(name for name in ATTR_NAMES if name not in INT_ATTR_NAMES)

STATUS_OUTPUT_INDICES = (8, 25, 29, 41, 49)
NON_STATUS_OUTPUT_INDICES = tuple(idx for idx in range(50) if idx not in STATUS_OUTPUT_INDICES)

MODEL_FIELDS = (
    "hc_fn",
    "hc_scale",
    "hc_base",
    "norm_weight",
    "projection_weight",
    "conv_weight",
    "a_log",
    "dt_bias",
    "g_b_weight",
    "o_norm_weight",
    "o_weight",
    "f_b_weight",
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
    "in_projection_scale",
    "g_projection_scale",
    "o_projection_scale",
    "f_projection_scale",
    "o_quant_weight",
    "o_quant_channel_scale",
)

HEAD_FIELDS = frozenset(
    (
        "projection_weight",
        "conv_weight",
        "a_log",
        "dt_bias",
        "g_b_weight",
        "o_weight",
        "f_b_weight",
        "in_projection_scale",
        "g_projection_scale",
        "o_projection_scale",
        "f_projection_scale",
    )
)

SHARED_FIELDS = frozenset(
    (
        "expert_weight",
        "expert_channel_scale",
        "expert_down_weight",
        "expert_down_channel_scale",
    )
)

_EXPERT_WEIGHTS = (
    "expert_weight",
    "expert_down_weight",
    "batch_expert_weight",
    "batch_expert_down_weight",
)

_EXPERT_SCALES = (
    "expert_channel_scale",
    "expert_down_channel_scale",
    "batch_expert_channel_scale",
    "batch_expert_down_channel_scale",
)

_SUPPORTED_DTYPES = frozenset((torch.bfloat16, torch.float32, torch.int32, torch.int8))

_BASE_OUTPUT_DTYPES_0_14 = (
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.int32,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
)

_EXTRA_OUTPUT_SPECS_15_49 = (
    (16384, torch.bfloat16),
    (4096, torch.bfloat16),
    (128, torch.float32),
    (128, torch.float32),
    (4096, torch.bfloat16),
    (384, torch.float32),
    (128, torch.float32),
    (128, torch.int32),
    (128, torch.int32),
    (128, torch.int32),
    (128, torch.int32),
    (4096, torch.int8),
    (128, torch.float32),
    (4096, torch.int32),
    (128, torch.int32),
    (4096, torch.float32),
    (2048, torch.float32),
    (2048, torch.int8),
    (128, torch.float32),
    (4096, torch.int32),
    (4096, torch.bfloat16),
    (4096, torch.float32),
    (4096, torch.bfloat16),
    (256, torch.bfloat16),
    (4096, torch.float32),
    (4096, torch.bfloat16),
    (128, torch.int32),
    (32768, torch.int32),
    (32768, torch.float32),
    (16384, torch.float32),
    (16384, torch.int8),
    (1024, torch.float32),
    (32768, torch.int32),
    (32768, torch.bfloat16),
    (128, torch.int32),
)

is_fractal_nz = _is_nz


@dataclass(frozen=True)
class StepIdentity:
    layer: int
    step: int
    model_digest: str
    local_tokens_digest: str

    def validate(self) -> None:
        for val in (self.layer, self.step):
            if isinstance(val, bool) or not isinstance(val, int) or not 0 <= val < 2**31:
                raise ValueError("layer and step must be nonnegative int32")
        for digest in (self.model_digest, self.local_tokens_digest):
            if not isinstance(digest, str) or not _HEX256_RE.fullmatch(digest):
                raise ValueError("identity digests must be lowercase SHA256")


@dataclass(frozen=True)
class ReviewedAbi:
    generated_header_sha256: str
    protocol: str
    max_rows_per_source: int
    max_total_rows: int
    variable_rows: bool
    zero_rows: bool
    output_count: int

    def validate(self) -> None:
        if not isinstance(self.generated_header_sha256, str) or not _HEX256_RE.fullmatch(self.generated_header_sha256):
            raise ValueError("invalid reviewed generated header SHA256")
        if (
            self.protocol not in PROTOCOL_STAGE_MAP
            or type(self.variable_rows) is not bool
            or type(self.zero_rows) is not bool
            or (self.protocol in (PROTOCOL, SHARDED_SHARED_PROTOCOL) and (self.variable_rows or self.zero_rows))
            or self.output_count != 50
        ):
            raise ValueError("unsupported independent EP protocol/ABI or zero-row policy")
        for limit in (self.max_rows_per_source, self.max_total_rows):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("invalid verified per-source / aggregate row capacity")


def integer_attribute(name: str, value: Any) -> int:
    """Convert strict Python/NumPy/Torch integer without bool or float coercion."""
    dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or dtype is torch.bool or dtype is bool or getattr(dtype, "kind", None) == "b":
        raise TypeError(f"MC2 integer attribute {name} cannot be bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"MC2 integer attribute {name} must be an integer") from exc
    if result < -(2**63) or result >= 2**63:
        raise ValueError(f"MC2 integer attribute {name} is outside the int64 range")
    return result


def _pitch_256(width: int) -> int:
    return ((width + 255) // 256) * 256


def stage28_output_specs(heads: int, rows: int) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """Physical 50-output tensor shapes and dtypes for batched MC2 execution."""
    if isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= 64:
        raise ValueError("local head count must be within 1..64")
    if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 2**31 - 1:
        raise ValueError("rows must fit positive int32")
    head_pitch = _pitch_256(128 * heads)
    widths_0_14 = (
        4096,
        128,
        128,
        16384,
        4096,
        _pitch_256(385 * heads + 256),
        _pitch_256(384 * heads),
        head_pitch,
        0,
        256,
        head_pitch,
        head_pitch,
        4096,
        256,
        head_pitch,
    )
    specs: list[tuple[tuple[int, ...], torch.dtype]] = []
    for idx, (width, dtype) in enumerate(zip(widths_0_14, _BASE_OUTPUT_DTYPES_0_14)):
        shape = (rows, 2, 128) if idx == 8 else (rows, width)
        specs.append((shape, dtype))
    for width, dtype in _EXTRA_OUTPUT_SPECS_15_49:
        specs.append(((rows, width), dtype))
    return tuple(specs)


def allocate_output_buffers(
    device: Any,
    heads: int,
    rows: int,
    *,
    init_non_status: bool = True,
    status_buffers: dict[int, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Allocate 512-byte aligned MC2 outputs; optionally reuse preallocated status tensors."""
    specs = stage28_output_specs(heads, rows)
    outputs: list[torch.Tensor] = []
    for idx, (shape, dtype) in enumerate(specs):
        if status_buffers is not None and idx in status_buffers:
            val = status_buffers[idx]
            val.fill_(-(2**31))
            outputs.append(val)
            continue
        val = aligned_empty(shape, dtype, device)
        if idx in STATUS_OUTPUT_INDICES:
            val.fill_(-(2**31))
        elif init_non_status:
            if dtype in (torch.bfloat16, torch.float32):
                val.fill_(float("nan"))
            elif dtype == torch.int8:
                val.fill_(-101)
            else:
                val.fill_(-(2**31))
        outputs.append(val)
    return outputs


def reset_status_outputs(outputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
    if len(outputs) != 50:
        raise ValueError("invalid prepared output set")
    sentinel = -(2**31)
    for idx in STATUS_OUTPUT_INDICES:
        outputs[idx].fill_(sentinel)


def non_overlapping_positive_strides(tensor: Any) -> bool:
    """Conservatively verify that a possibly strided tensor has disjoint positive-stride addresses."""
    dimensions = sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    span = 1
    for stride, size in dimensions:
        if stride < span:
            return False
        span = stride * size
    return True


def state_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return conservative [begin, end) DMA-padded byte span for a positive-stride view."""
    if not non_overlapping_positive_strides(tensor):
        raise ValueError("state view overlaps itself or has unsupported strides")
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()) if size > 0)
    bytes_padded = ((span * tensor.element_size() + 31) // 32) * 32
    ptr = int(tensor.data_ptr())
    return ptr, ptr + bytes_padded


def check_abi_tensor(value: Any, device: Any, *, allow_expert_view: bool = False) -> None:
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("tensor device or dtype not supported")
    if value.device.type == "npu":
        import torch_npu

        fmt = int(torch_npu.get_npu_format(value))
        if fmt not in (0, 2) and not (
            fmt == 29
            and allow_expert_view
            and value.dtype in (torch.int8, torch.bfloat16)
            and value.ndim == 3
            and value.shape[1] in (2048, 4096)
            and value.shape[2] == 4096
            and value.is_contiguous()
        ):
            raise ValueError("unsupported tensor format for MegaKernelMc2")
    if value.requires_grad or not value.is_contiguous() or value.data_ptr() % 512:
        raise ValueError("tensor must be contiguous, detached and 512-byte aligned")
    available = value.untyped_storage().nbytes() - value.storage_offset() * value.element_size()
    required = ((value.numel() * value.element_size() + 31) // 32) * 32
    if value.numel() == 0 or available < required:
        raise ValueError("tensor must have padded accessible storage")


def expert_partition(rank: int, world: int) -> tuple[int, int]:
    if (
        isinstance(rank, bool)
        or isinstance(world, bool)
        or not isinstance(rank, int)
        or not isinstance(world, int)
        or not 1 <= world <= 256
        or not 0 <= rank < world
    ):
        raise ValueError("invalid expert partition rank/world")
    return rank * 288 // world, (rank + 1) * 288 // world


def expert_mode(inputs: dict[str, Any]) -> str:
    first = inputs.get("expert_weight")
    if not isinstance(first, torch.Tensor) or first.dtype not in (torch.int8, torch.bfloat16):
        raise ValueError("expert weights must be INT8 or BF16")
    for name in _EXPERT_WEIGHTS:
        val = inputs.get(name)
        if not isinstance(val, torch.Tensor) or val.dtype != first.dtype:
            raise ValueError(f"expert weight dtype mismatch: {name}")
    if first.dtype == torch.bfloat16:
        if any(inputs.get(name) is not None for name in _EXPERT_SCALES):
            raise ValueError("BF16 experts must not supply channel scales")
        return "bf16"
    for name in ("expert_channel_scale", "batch_expert_channel_scale"):
        val = inputs.get(name)
        if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
            raise ValueError(f"INT8 experts require FP32 scale: {name}")
    shared = inputs.get("expert_down_channel_scale")
    routed = inputs.get("batch_expert_down_channel_scale")
    if (
        not isinstance(shared, torch.Tensor)
        or not isinstance(routed, torch.Tensor)
        or shared.dtype not in (torch.float32, torch.bfloat16)
        or routed.dtype != shared.dtype
    ):
        raise ValueError("INT8 shared and routed down scales must have the same BF16 or FP32 dtype")
    return "int8"


def projection_modes(inputs: dict[str, Any]) -> tuple[str, ...]:
    modes: list[str] = []
    for weight_name, scale_name in PROJECTION_PAIRS:
        weight, scale = inputs.get(weight_name), inputs.get(scale_name)
        if not isinstance(weight, torch.Tensor) or weight.dtype not in (torch.bfloat16, torch.int8):
            raise ValueError(f"{weight_name} must be BF16 or INT8")
        if weight.dtype == torch.bfloat16:
            if scale is not None:
                raise ValueError(f"{scale_name} must be absent for BF16 weight")
            modes.append("bf16")
        else:
            if not isinstance(scale, torch.Tensor) or scale.dtype != torch.bfloat16:
                raise ValueError(f"{scale_name} must be BF16 for INT8 weight")
            modes.append("w8a8_dynamic")
    return tuple(modes)


def validate_projection_shapes(inputs: dict[str, Any], heads: int) -> tuple[str, ...]:
    if isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= 64:
        raise ValueError("heads must be in 1..64")
    modes = projection_modes(inputs)
    shapes = (
        (385 * heads + 256, 4096),
        (heads * 128, 128),
        (4096, heads * 128),
        (heads * 128, 128),
    )
    for (weight_name, scale_name), shape, mode in zip(PROJECTION_PAIRS, shapes, modes):
        weight = inputs[weight_name]
        if tuple(weight.shape) != shape or not weight.is_contiguous():
            raise ValueError(f"{weight_name} must be contiguous ND {shape}")
        if mode == "w8a8_dynamic":
            scale = inputs[scale_name]
            if tuple(scale.shape) != (shape[0],) or not scale.is_contiguous():
                raise ValueError(f"{scale_name} must be contiguous [{shape[0]}]")
    return modes


def _expert_hash_view(inputs: dict[str, Any], name: str) -> torch.Tensor:
    val = inputs[name]
    down = inputs.get("batch_expert_down_weight")
    if (
        name in ("batch_expert_weight", "batch_expert_down_weight")
        and isinstance(down, torch.Tensor)
        and tuple(down.shape[1:]) == (2048, 4096)
        and not is_fractal_nz(val)
    ):
        return val.transpose(1, 2)
    return val


def tensor_sha256(named_tensors) -> bytes:
    """Hash named tensors in bounded CPU chunks without materializing NZ or transposed banks."""
    digest = hashlib.sha256()

    def iter_chunks(val: torch.Tensor, elem_limit: int):
        if is_fractal_nz(val):
            if val.ndim == 3:
                yield from val.unbind(0)
            else:
                yield val
            return
        if val.numel() <= elem_limit:
            yield val
        elif val.is_contiguous():
            flat = val.view(-1)
            for off in range(0, flat.numel(), elem_limit):
                yield flat[off : off + elem_limit]
        else:
            row_elems = val.numel() // val.shape[0]
            rows_per_chunk = max(1, elem_limit // row_elems)
            for off in range(0, val.shape[0], rows_per_chunk):
                if rows_per_chunk == 1 and row_elems > elem_limit:
                    yield from iter_chunks(val[off], elem_limit)
                else:
                    yield from iter_chunks(val[off : off + rows_per_chunk], elem_limit)

    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"missing tensor {name}")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"), allow_nan=False)
        digest.update(header.encode("utf-8"))
        chunk_elems = max(1, (8 * 1024 * 1024) // tensor.element_size())
        for piece in iter_chunks(tensor.detach(), chunk_elems):
            flat_cpu = piece.to(device="cpu").contiguous().reshape(-1)
            digest.update(flat_cpu.view(torch.uint8).numpy().tobytes())
    return digest.digest()


def expert_bank_digest(inputs: dict[str, Any]) -> str:
    val = inputs.get("batch_expert_weight")
    if not isinstance(val, torch.Tensor) or val.dtype not in (torch.int8, torch.bfloat16):
        raise ValueError("invalid expert bank dtype")
    fields = (
        ("batch_expert_weight", "batch_expert_down_weight")
        if val.dtype == torch.bfloat16
        else (
            "batch_expert_weight",
            "batch_expert_channel_scale",
            "batch_expert_down_weight",
            "batch_expert_down_channel_scale",
        )
    )
    return tensor_sha256((name, _expert_hash_view(inputs, name)) for name in fields).hex()
