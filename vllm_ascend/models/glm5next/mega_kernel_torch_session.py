# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-side MC2 buffer lifecycle using the operator wheel's Torch entry.

The session reuses tensor storage and admission, not the ACLNN executor: the
Torch extension creates an executor on every call. It is an eager-only path.
The caller must abort the distributed job after a failed launch or sync.
"""

import hashlib
import importlib
import operator
from pathlib import Path
from types import SimpleNamespace

import torch
from mc2_integration.projection_contract import validate_external_routes, validate_projection_shapes
from mc2_integration.stage24_admission import (
    GENERAL_PROTOCOL,
    GRID_PROTOCOL,
    NATIVE_SHARED_PROTOCOL,
    PADDING_PROTOCOL,
    PROTOCOL,
    SHARDED_SHARED_PROTOCOL,
)
from mc2_integration.torch_acl_bridge import _check_tensor, initialized_outputs
from mc2_integration.vllm_mc2_repeatable import RepeatableMegaKernelSession

from vllm_ascend.models.glm5next.mega_kernel_abi import ATTR_NAMES, MC2_INPUT_NAMES, validate_expert_bank
from vllm_ascend.models.glm5next.mega_kernel_weights import _validate_native_expert

_INT_ATTRIBUTES = (
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
_FLOAT_ATTRIBUTES = tuple(name for name in ATTR_NAMES if name not in _INT_ATTRIBUTES)
# Failed asynchronous launches may still own device work. Keep their tensors,
# stream and process groups alive until job termination; never reuse them.
_FAILED_LAUNCHES = []


def _signature(value):
    if value is None:
        return None
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _integer_attribute(name, value):
    """Accept Python's integer protocol without silently truncating floats."""
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


class _TorchBoundBuffers:
    """Stable inputs/outputs and communication arguments, not an executor."""

    def __init__(self, inputs, outputs, attrs, groups, stream, op):
        self.inputs, self.outputs, self.attrs = inputs, outputs, attrs
        self.groups, self.stream, self.op = groups, stream, op
        # Conversion happens during coordinated cold admission, before any
        # rank can enter the operator's device collective.
        self.int_attrs = [_integer_attribute(name, attrs[name]) for name in _INT_ATTRIBUTES]
        self.float_attrs = [float(attrs[name]) for name in _FLOAT_ATTRIBUTES]
        self.keepalive = ()
        self.closed = False
        self.failed = False
        self._input_names = tuple(inputs)
        self._bound = (*inputs.values(), *outputs)
        self._signatures = tuple(_signature(value) for value in self._bound)

    def validate_bindings(self):
        current = (*self.inputs.values(), *self.outputs)
        if self.closed or self.failed or tuple(self.inputs) != self._input_names or len(current) != len(self._bound):
            raise RuntimeError("MC2 Torch buffers are closed, failed or changed")
        if any(
            value is not bound or _signature(value) != sig
            for value, bound, sig in zip(current, self._bound, self._signatures)
        ):
            raise RuntimeError("MC2 Torch buffers changed address or metadata")
        if int(torch.npu.current_stream(self.inputs["x"].device).npu_stream) != int(self.stream.npu_stream):
            raise RuntimeError("MC2 Torch stream changed")

    def run(self):
        self.validate_bindings()
        try:
            result = self.op(
                [self.inputs[name] for name in MC2_INPUT_NAMES],
                self.outputs,
                self.int_attrs,
                self.float_attrs,
                *self.groups,
            )
            self.stream.synchronize()
            if len(result) != len(self.outputs):
                raise RuntimeError("MC2 Torch operator returned an invalid output set")
            return tuple(result)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.failed:
            _FAILED_LAUNCHES.append(self)


class _TorchOperatorAdmission:
    """Perform the existing cold admission before binding the Torch operator."""

    def __init__(self, config, op):
        self.heads_per_shard = config.heads_per_shard
        self.native_shared = config.native_shared
        if config.general_topology and not config.native_shared:
            raise ValueError("general topology requires native shared mode")
        if config.native_shared and not (config.shared_tp_sharded and config.padding_supported):
            raise ValueError("native shared mode requires sharded shared and padding support")
        self.independent_ep_protocol, self.stage_limit = (
            (GENERAL_PROTOCOL, 29)
            if config.general_topology
            else (NATIVE_SHARED_PROTOCOL, 28)
            if config.native_shared
            else (GRID_PROTOCOL, 27)
            if config.shared_tp_sharded and config.padding_supported
            else (SHARDED_SHARED_PROTOCOL, 26)
            if config.shared_tp_sharded
            else (PADDING_PROTOCOL, 25)
            if config.padding_supported
            else (PROTOCOL, 24)
        )
        header = Path(config.generated_header_path)
        if not header.is_absolute() or not header.is_file():
            raise ValueError("supply absolute generated ACLNN header path")
        self.generated_header_sha256 = hashlib.sha256(header.read_bytes()).hexdigest()
        config.reviewed_abi.validate()
        if (
            self.generated_header_sha256 != config.reviewed_abi.generated_header_sha256
            or self.independent_ep_protocol != config.reviewed_abi.protocol
        ):
            raise ValueError("generated header or protocol disagrees with reviewed ABI")
        self.op = op

    def prepare_independent(
        self,
        inputs,
        attrs,
        *,
        tp_group,
        ep_group,
        tp_algorithm,
        ep_algorithm,
        ep_rank,
        propagation_rounds,
        source_rows,
        source_valid_rows,
        layer,
        step,
        protocol,
        repeatable,
    ):
        del layer, step, repeatable
        if set(inputs) != set(MC2_INPUT_NAMES) or set(attrs) != set(ATTR_NAMES):
            raise ValueError("input or attribute names do not match the MC2 ABI")
        x = inputs["x"]
        if (
            protocol != self.independent_ep_protocol
            or attrs["stage_limit"] != self.stage_limit
            or inputs["ffn_c0_diagnostic"] is not None
            or not source_rows
            or x.shape[0] <= 0
            or any(row != x.shape[0] for row in source_rows)
            or len(source_valid_rows) != len(source_rows)
            or any(count < 0 or count > x.shape[0] for count in source_valid_rows)
            or isinstance(ep_rank, bool)
            or not isinstance(ep_rank, int)
            or not 0 <= ep_rank < len(source_rows)
            or isinstance(propagation_rounds, bool)
            or not isinstance(propagation_rounds, int)
            or not 0 < propagation_rounds < 2**63
        ):
            raise ValueError("invalid independent EP contract")
        if (
            any(not isinstance(v, str) or not v or "\0" in v for v in (tp_group, ep_group, tp_algorithm, ep_algorithm))
            or tp_group == ep_group
        ):
            raise ValueError("invalid MC2 communication names")
        actual = sum((c, r) != (-1, -1) for c, r in inputs["state_indices"].cpu().tolist())
        if actual != source_valid_rows[ep_rank]:
            raise ValueError("source validity disagrees with local metadata")
        validate_projection_shapes(inputs, self.heads_per_shard)
        validate_external_routes(inputs, int(x.shape[0]), inputs["state_indices"][:, 0] >= 0)
        bank = inputs.get("batch_expert_weight")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 3:
            raise ValueError("missing expert bank")
        bank_args = {"shared_width": inputs["expert_down_weight"].shape[1]} if self.native_shared else {}
        validate_expert_bank(
            inputs, ep_rank, attrs["ep_world_size"], "replicated" if bank.shape[0] == 288 else "sharded", **bank_args
        )
        if x.device.type != "npu" or x.device.index != torch.npu.current_device():
            raise ValueError("input must belong to the current NPU")
        outputs = initialized_outputs(x.device, self.heads_per_shard, rows=int(x.shape[0]), stage23=True)
        spans = []
        for name, value in (*inputs.items(), *((None, output) for output in outputs)):
            if value is not None:
                is_expert = name in ("batch_expert_weight", "batch_expert_down_weight")
                if is_expert:
                    _validate_native_expert(value)
                    _check_tensor(value, x.device, allow_expert_view=True)
                else:
                    _check_tensor(value, x.device)
                start = int(value.data_ptr())
                spans.append((start, start + ((value.numel() * value.element_size() + 31) // 32) * 32))
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("MC2 input/output buffers overlap")
        groups = (tp_group, ep_group, tp_algorithm, ep_algorithm, ep_rank, propagation_rounds)
        return _TorchBoundBuffers(inputs, outputs, dict(attrs), groups, torch.npu.current_stream(x.device), self.op)


def create_torch_session(config, model_epoch):
    """Create the existing admission/state lifecycle with a Torch operator launch."""
    # Importing the wheel registers the operator; fail bootstrap on all ranks
    # rather than falling back to a different collective path on one rank.
    # The selected-operator wheel is packaged under the custom vendor name.
    # Compile/load before the bootstrap's all-rank agreement so a missing
    # extension cannot make just one rank fail at the first collective launch.
    module = importlib.import_module("cann_ops_transformer_custom.ops.attention.mega_kernel_mc2.mega_kernel_mc2")
    module._builder.load()
    op = torch.ops.cann_ops_transformer.mega_kernel_mc2_out
    admission = _TorchOperatorAdmission(config, op)
    session = RepeatableMegaKernelSession(SimpleNamespace(config=config, bridge=admission), model_epoch)
    if not hasattr(session, "capture_routes"):
        raise RuntimeError("MC2 session lacks capture_routes; update the matching ops-transformer runtime")
    # Route snapshots are needed by the diagnostic layer comparison only.
    # Avoid making two additional tensor copies on every production layer.
    session.capture_routes = False
    return session
