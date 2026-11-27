# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-side MC2 buffer pool, repeatable executor lifecycle, and step transactions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import threading
from typing import Any

import torch

from vllm_ascend.models.glm5next.mega_kernel_abi import (
    ATTR_NAMES,
    FLOAT_ATTR_NAMES,
    INT_ATTR_NAMES,
    MC2_INPUT_NAMES,
    NATIVE_SHARED_PROTOCOL,
    NON_STATUS_OUTPUT_INDICES,
    PROTOCOL_STAGE_MAP,
    STATUS_OUTPUT_INDICES,
    ReviewedAbi,
    StepIdentity,
    aligned_copy,
    aligned_empty,
    check_abi_tensor,
    integer_attribute,
    non_overlapping_positive_strides,
    reset_status_outputs,
    stage28_output_specs,
    state_byte_span,
)
from vllm_ascend.models.glm5next.mega_kernel_collectives import (
    MegaKernelCollectiveError,
    agree_or_abort,
    agree_preflight,
)
from vllm_ascend.models.glm5next.mega_kernel_weights import _validate_native_expert

_INT_ATTRIBUTES = INT_ATTR_NAMES
_FLOAT_ATTRIBUTES = FLOAT_ATTR_NAMES

# Failed asynchronous launches may still own device work. Keep their tensors,
# stream, prepared executors, and process groups alive until job termination.
_FAILED_LAUNCHES: list[Any] = []


def _signature(value: Any):
    if value is None:
        return None
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _integer_attribute(name: str, value: Any) -> int:
    return integer_attribute(name, value)


def _physical_dim_first_conv(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 3 and tensor.shape[1] == 3 and tensor.stride(1) == 1 and tensor.stride(2) > 1


def _gather_compact_slots(dst: torch.Tensor, src: torch.Tensor, slots_i64: torch.Tensor, *, is_conv: bool) -> None:
    if not slots_i64.numel():
        return
    if is_conv and _physical_dim_first_conv(src):
        selected = src.transpose(-1, -2).index_select(0, slots_i64).transpose(-1, -2)
        dst.copy_(selected)
        return
    torch.index_select(src, 0, slots_i64, out=dst)


def _commit_compact_slots(dst: torch.Tensor, src: torch.Tensor, slots_i64: torch.Tensor, *, is_conv: bool) -> None:
    if not slots_i64.numel():
        return
    if is_conv and _physical_dim_first_conv(dst):
        dst.transpose(-1, -2).index_copy_(0, slots_i64, src.transpose(-1, -2).contiguous())
        return
    dst.index_copy_(0, slots_i64, src)


@dataclass(frozen=True)
class Mc2SessionConfig:
    opapi_path: str
    generated_header_path: str
    reviewed_abi: ReviewedAbi
    heads_per_shard: int
    model_num_heads: int
    local_head_range: tuple[int, int]
    control_group: Any
    tp_group: Any
    ep_group: Any
    control_device: Any
    tp_algorithm: str
    ep_algorithm: str
    expert_shard_digests: tuple[str, ...]
    head_shard_digests: tuple[str, ...] | None = None
    shared_shard_digests: tuple[str, ...] | None = None
    expert_id_digests: tuple[str, ...] | None = None
    model_shared_intermediate: int | None = None
    shared_shard_ranges: tuple[tuple[int, int], ...] | None = None
    padding_supported: bool = True
    shared_tp_sharded: bool = True
    native_shared: bool = True
    general_topology: bool = False
    allow_overlapping_tp: bool = True
    allow_grid: bool = True
    framework_admission: bool = True
    consumer_readback: bool = False
    repeatable_executor: bool = True
    sync_per_layer: bool = False
    defer_state_commit: bool = True
    max_cached_geometries: int = 8
    tp_group_name: str | None = None
    ep_group_name: str | None = None
    ep_rank: int | None = None
    propagation_rounds: int = 1
    generated_header_sha256: str | None = None
    compute_device: Any = None


class _TorchBoundBuffers:
    """Fallback per-launch Torch operator runner when C++ repeatable preparation is unavailable."""

    def __init__(self, inputs, outputs, attrs, groups, stream, op, *, sync_on_run: bool = True):
        self.inputs, self.outputs, self.attrs = inputs, outputs, attrs
        self.groups, self.stream, self.op = groups, stream, op
        self.sync_on_run = bool(sync_on_run)
        self.int_attrs = [_integer_attribute(name, attrs[name]) for name in _INT_ATTRIBUTES]
        self.float_attrs = [float(attrs[name]) for name in _FLOAT_ATTRIBUTES]
        self.keepalive = ()
        self.closed = False
        self.failed = False
        self.executor_repeatable = False
        self.workspace_tensor = None
        self._input_names = tuple(inputs)
        self._input_list = [inputs[name] for name in MC2_INPUT_NAMES]
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

    def run(self, sync: bool | None = None):
        self.validate_bindings()
        should_sync = self.sync_on_run if sync is None else bool(sync)
        try:
            result = self.op(
                self._input_list,
                self.outputs,
                self.int_attrs,
                self.float_attrs,
                *self.groups,
            )
            if should_sync:
                self.stream.synchronize()
            if len(result) != len(self.outputs):
                raise RuntimeError("MC2 Torch operator returned an invalid output set")
            return tuple(self.outputs)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.failed:
            _FAILED_LAUNCHES.append(self)


class _PreparedCppRunner:
    """Wrapper around C++ PreparedMegaKernelMc2 repeatable ACLNN executor."""

    def __init__(self, handle: Any, inputs: dict[str, Any], outputs: list[torch.Tensor], stream: Any):
        self._handle = handle
        self.inputs = inputs
        self.outputs = outputs
        self.stream = stream
        self.closed = False
        self.failed = False
        self.executor_repeatable = True
        self.workspace_tensor = handle.workspace()

    def validate_bindings(self):
        if self.closed or self.failed:
            raise RuntimeError("MC2 prepared executor is closed or failed")
        self._handle.validate_bindings()

    def run(self, sync: bool = False):
        if self.closed or self.failed:
            raise RuntimeError("MC2 prepared executor is closed or failed")
        try:
            self._handle.run()
            if sync:
                self.stream.synchronize()
            return tuple(self.outputs)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._handle.close()
        except BaseException:
            self.failed = True
            _FAILED_LAUNCHES.append(self)
            raise
        if self.failed or getattr(self._handle, "failed", False):
            _FAILED_LAUNCHES.append(self)


class _OutputBank:
    """Reusable non-status outputs and compact incoming mHC views for one ping-pong slot."""

    def __init__(self, heads: int, rows: int, device: Any):
        self.heads = heads
        self.rows = rows
        self.device = device
        specs = stage28_output_specs(heads, rows)
        self.non_status: dict[int, torch.Tensor] = {
            idx: aligned_empty(specs[idx][0], specs[idx][1], device) for idx in NON_STATUS_OUTPUT_INDICES
        }
        self.compact_post = torch.empty((rows, 4, 1), dtype=torch.float32, device=device)
        self.compact_comb = torch.empty((rows, 4, 4), dtype=torch.float32, device=device)

    def build_outputs_list(self, status_buffers: dict[int, torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for idx in range(50):
            if idx in status_buffers:
                outputs.append(status_buffers[idx])
            else:
                outputs.append(self.non_status[idx])
        return outputs


class _GeometryBucket:
    """Shared dynamic inputs, Layer-0 constants, ping-pong output banks, and workspace per (rows, stream)."""

    def __init__(self, rows: int, stream_ptr: int, device: Any):
        self.rows = rows
        self.stream_ptr = stream_ptr
        self.device = device
        self.shared_inputs: dict[str, torch.Tensor] = {
            "x": aligned_empty((rows, 4, 4096), torch.bfloat16, device),
            "post_h_out": aligned_empty((rows, 4096), torch.bfloat16, device),
            "post_h_post": aligned_empty((rows, 128), torch.float32, device),
            "post_h_res": aligned_empty((rows, 128), torch.float32, device),
            "cu_seqlens": aligned_copy(torch.tensor([0, 1], dtype=torch.int32, device=device)),
            "state_indices": aligned_copy(
                torch.arange(rows, dtype=torch.int32, device=device).unsqueeze(1).expand(rows, 2).contiguous()
            ),
        }
        self.shared_inputs["post_h_post"].zero_()
        self.shared_inputs["post_h_res"].zero_()
        self.eye_flat = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 16).expand(rows, 16).contiguous()
        self.output_banks: dict[tuple[int, int], _OutputBank] = {}
        self.shared_workspace: torch.Tensor | None = None
        self.extra_workspaces: list[torch.Tensor] = []

    def get_bank(self, heads: int, bank_index: int) -> _OutputBank:
        key = (heads, bank_index & 1)
        bank = self.output_banks.get(key)
        if bank is None:
            bank = _OutputBank(heads, self.rows, self.device)
            self.output_banks[key] = bank
        return bank

    def update_workspace(self, workspace: torch.Tensor | None) -> None:
        if workspace is None:
            return
        if self.shared_workspace is None:
            self.shared_workspace = workspace
            return
        if workspace is self.shared_workspace:
            return
        if workspace.numel() > self.shared_workspace.numel():
            self.extra_workspaces.append(self.shared_workspace)
            self.shared_workspace = workspace
        else:
            self.extra_workspaces.append(workspace)


@dataclass
class _PendingCommit:
    session: Any
    prov_conv: torch.Tensor
    prov_rec: torch.Tensor
    dst_conv: torch.Tensor
    dst_rec: torch.Tensor
    conv_slots: torch.Tensor
    rec_slots: torch.Tensor
    conv_writeback: torch.Tensor | None
    status_buffers: tuple[torch.Tensor, ...]


class SharedStepBufferPool:
    """Worker-wide bounded LRU pool sharing step buffers, ping-pong banks, workspace, and atomic TX state."""

    def __init__(self, device: Any, max_cached_geometries: int = 8, *, control_device: Any = None):
        if type(max_cached_geometries) is not int or max_cached_geometries < 1:
            raise ValueError("max_cached_geometries must be a positive integer")
        self.device = torch.device(device)
        self.control_device = torch.device(self.device if control_device is None else control_device)
        self.max_cached_geometries = max_cached_geometries
        self._buckets: OrderedDict[tuple[int, int], _GeometryBucket] = OrderedDict()
        self._sessions: list[Any] = []
        self._pending: list[_PendingCommit] = []
        self._tx_failed = False
        self._agree_scratch: torch.Tensor | None = None
        self._preflight_scratch: torch.Tensor | None = None
        self._step_contract_scratch: torch.Tensor | None = None

    def register_session(self, session: Any) -> None:
        if session not in self._sessions:
            self._sessions.append(session)

    def get_agree_scratch(self) -> torch.Tensor:
        if self._agree_scratch is None:
            self._agree_scratch = torch.zeros(1, dtype=torch.int32, device=self.control_device)
        return self._agree_scratch

    def get_preflight_scratch(self) -> torch.Tensor:
        if self._preflight_scratch is None:
            self._preflight_scratch = torch.zeros(1, dtype=torch.int32, device=self.control_device)
        return self._preflight_scratch

    def get_step_contract_scratch(self) -> torch.Tensor:
        if self._step_contract_scratch is None:
            self._step_contract_scratch = torch.zeros(35, dtype=torch.int32, device=self.control_device)
        return self._step_contract_scratch

    def touch_bucket(self, key: tuple[int, int]) -> None:
        if key in self._buckets:
            self._buckets.move_to_end(key)

    def get_or_create_bucket(self, rows: int, stream_ptr: int) -> _GeometryBucket:
        key = (rows, stream_ptr)
        bucket = self._buckets.get(key)
        if bucket is not None:
            self._buckets.move_to_end(key)
            return bucket
        if self._pending:
            raise RuntimeError("cannot evict or allocate a new geometry bucket while a step transaction is pending")
        if len(self._buckets) >= self.max_cached_geometries and self.device.type == "npu" and hasattr(torch, "npu"):
            stream = torch.npu.current_stream(self.device)
            sync = getattr(stream, "synchronize", None)
            if callable(sync):
                sync()
        while len(self._buckets) >= self.max_cached_geometries:
            evicted_key, _ = self._buckets.popitem(last=False)
            for session in self._sessions:
                session._evict_geometry(evicted_key)
        bucket = _GeometryBucket(rows, stream_ptr, self.device)
        self._buckets[key] = bucket
        return bucket

    def has_pending(self) -> bool:
        return bool(self._pending)

    def record_pending(self, entry: _PendingCommit) -> None:
        if self._tx_failed:
            raise RuntimeError("MC2 step transaction already failed; abort the distributed job")
        self._pending.append(entry)

    def _mark_sessions_failed(self) -> None:
        self._tx_failed = True
        for session in self._sessions:
            session._failed = True
        for entry in self._pending:
            entry.session._failed = True
        self._pending.clear()

    def abort_step(self, *, mark_failed: bool = True) -> None:
        if mark_failed:
            self._mark_sessions_failed()
            return
        self._pending.clear()

    def finalize_step(self, control_device: Any, control_group: Any, *, local_error: Exception | None = None) -> None:
        """Validate all pending layers' status outputs in one reduction and commit active slots atomically."""
        if not self._pending and local_error is None:
            return
        pending = self._pending
        ctrl_device = torch.device(control_device)
        scratch = self.get_agree_scratch() if ctrl_device == self.control_device else None
        try:
            check_error: Exception | None = None
            local_bad = 1
            if local_error is None and pending:
                try:
                    checks = []
                    for entry in pending:
                        for status_tensor in entry.status_buffers:
                            checks.append(torch.any(status_tensor != 0))
                    local_bad = int(bool(torch.any(torch.stack(checks)).item()))
                except Exception as exc:
                    check_error = exc
                    local_bad = 1
            cause = local_error if local_error is not None else check_error
            flag = scratch if scratch is not None else torch.empty(1, dtype=torch.int32, device=ctrl_device)
            flag[0] = local_bad
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=control_group)
            if bool(flag.item()):
                raise MegaKernelCollectiveError(
                    "MC2 step status or execution check failed on at least one rank; abort job without state commit"
                ) from cause
            commit_error = None
            try:
                for entry in pending:
                    _commit_compact_slots(entry.dst_conv, entry.prov_conv, entry.conv_slots, is_conv=True)
                    _commit_compact_slots(entry.dst_rec, entry.prov_rec, entry.rec_slots, is_conv=False)
                    if entry.conv_writeback is not None:
                        entry.conv_writeback.copy_(entry.dst_conv)
            except Exception as exc:
                commit_error = exc
            # All ranks must unconditionally participate in commit agreement so a
            # local commit failure on one rank never deadlocks peers.
            agree_or_abort(
                commit_error,
                ctrl_device,
                control_group,
                "MC2 deferred state commit",
                scratch=scratch,
            )
            self._pending = []
        except BaseException:
            self._mark_sessions_failed()
            raise

    def close_all(self) -> None:
        self._pending.clear()
        self._buckets.clear()


@dataclass
class _LayerGeometryBinding:
    key: tuple[int, int]
    bucket: _GeometryBucket
    bank: _OutputBank
    prov_conv: torch.Tensor
    prov_rec: torch.Tensor
    status_buffers: dict[int, torch.Tensor]
    outputs: list[torch.Tensor]
    runner: _PreparedCppRunner | _TorchBoundBuffers

    def close(self) -> None:
        self.runner.close()


class _LegacyBridgeInfo:
    """Expose reviewed protocol metadata expected by existing adapter checks."""

    def __init__(self, protocol: str, stage_limit: int, header_sha256: str, heads_per_shard: int):
        self.independent_ep_protocol = protocol
        self.stage_limit = stage_limit
        self.generated_header_sha256 = header_sha256
        self.heads_per_shard = heads_per_shard


class MegaKernelTorchSession:
    """Layer-bound MC2 execution session using shared step buffers and repeatable C++ executors."""

    def __init__(
        self,
        config: Mc2SessionConfig,
        model_epoch: Callable[[], str],
        *,
        op: Any,
        prepare_fn: Any = None,
        layer_idx: int = 0,
        layer_ordinal: int = 0,
        static_weights: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        buffer_pool: SharedStepBufferPool | None = None,
        bridge: Any = None,
    ):
        if not callable(model_epoch):
            raise TypeError("reliable model_epoch callable is required")
        config.reviewed_abi.validate()
        protocol, stage_limit = self._resolve_protocol(config)
        header_sha = config.generated_header_sha256
        if header_sha is None:
            header = Path(config.generated_header_path)
            if not header.is_absolute() or not header.is_file():
                raise ValueError("supply absolute generated ACLNN header path")
            header_sha = hashlib.sha256(header.read_bytes()).hexdigest()
        if header_sha != config.reviewed_abi.generated_header_sha256 or protocol != config.reviewed_abi.protocol:
            raise ValueError("generated header or protocol disagrees with reviewed ABI")
        self.config = config
        self.model_epoch = model_epoch
        self._op = op
        self._prepare_fn = prepare_fn
        self.layer_idx = layer_idx
        self.layer_ordinal = layer_ordinal
        self.heads_per_shard = config.heads_per_shard
        self.stage_limit = stage_limit
        self.protocol = protocol
        compute_dev = config.compute_device if config.compute_device is not None else config.control_device
        self.device = torch.device(compute_dev)
        self.control_device = torch.device(config.control_device)
        self.pool = buffer_pool or SharedStepBufferPool(
            self.device, config.max_cached_geometries, control_device=self.control_device
        )
        self.pool.register_session(self)
        self.bridge = bridge or _LegacyBridgeInfo(protocol, stage_limit, header_sha, config.heads_per_shard)
        self._epoch: str | None = None
        self._static_weights = dict(static_weights) if static_weights is not None else None
        self._attrs = dict(attrs) if attrs is not None else None
        self._int_attrs = (
            [_integer_attribute(name, self._attrs[name]) for name in _INT_ATTRIBUTES] if self._attrs is not None else None
        )
        self._float_attrs = (
            [float(self._attrs[name]) for name in _FLOAT_ATTRIBUTES] if self._attrs is not None else None
        )
        self._groups = (
            config.tp_group_name,
            config.ep_group_name,
            config.tp_algorithm,
            config.ep_algorithm,
            config.ep_rank,
            config.propagation_rounds,
        )
        self._bindings: OrderedDict[tuple[int, int], _LayerGeometryBinding] = OrderedDict()
        self._closed = False
        self._failed = False
        self._lock = threading.Lock()
        self.capture_routes: bool = False
        self.last_route_ids: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None

    @staticmethod
    def _resolve_protocol(config: Mc2SessionConfig) -> tuple[str, int]:
        if config.general_topology and not config.native_shared:
            raise ValueError("general topology requires native shared mode")
        if config.native_shared and not (config.shared_tp_sharded and config.padding_supported):
            raise ValueError("native shared mode requires sharded shared and padding support")
        for candidate, flag in (
            ("stage29-irregular-tp-ep-native-shared-v1", config.general_topology),
            (NATIVE_SHARED_PROTOCOL, config.native_shared),
            ("stage27-tp-ep-grid-padded-shared-v1", config.shared_tp_sharded and config.padding_supported),
            ("stage26-overlap-tp-ep-padded-shared-v1", config.shared_tp_sharded),
            ("stage25-tp1-ep-padding-bf16-w8a8-v2", config.padding_supported),
        ):
            if flag:
                return candidate, PROTOCOL_STAGE_MAP[candidate]
        return "stage24-v9-tp1-ep-no-padding-bf16-w8a8-v1", 24

    def _evict_geometry(self, key: tuple[int, int]) -> None:
        binding = self._bindings.pop(key, None)
        if binding is not None:
            binding.close()

    def _check_epoch_and_groups(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("MC2 session is closed or failed")
        epoch = self.model_epoch()
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("model/communicator epoch must be a nonempty string")
        if self._epoch is None:
            self._epoch = epoch
        elif epoch != self._epoch:
            raise RuntimeError("model/communicator epoch changed; discard session")

    def _resolve_comm_groups(self) -> tuple[str, str, str, str, int, int]:
        if all(x is not None for x in self._groups):
            return self._groups  # type: ignore[return-value]
        tp_name = self.config.tp_group._get_backend(self.device).get_hccl_comm_name(
            torch.distributed.get_rank(self.config.tp_group)
        )
        ep_name = self.config.ep_group._get_backend(self.device).get_hccl_comm_name(
            torch.distributed.get_rank(self.config.ep_group)
        )
        ep_rank = torch.distributed.get_rank(self.config.ep_group)
        self._groups = (
            tp_name,
            ep_name,
            self.config.tp_algorithm,
            self.config.ep_algorithm,
            ep_rank,
            self.config.propagation_rounds,
        )
        return self._groups

    def _build_binding(self, rows: int, stream: Any) -> _LayerGeometryBinding:
        if self._static_weights is None or self._attrs is None or self._int_attrs is None or self._float_attrs is None:
            raise RuntimeError("static weights and attrs must be bound before preparing MC2 geometry")
        if rows <= 0 or rows > self.config.reviewed_abi.max_rows_per_source:
            raise ValueError("rows exceeds reviewed ABI max_rows_per_source")
        stream_ptr = int(stream.npu_stream)
        if not stream_ptr:
            raise ValueError("current NPU stream is unavailable")
        bucket = self.pool.get_or_create_bucket(rows, stream_ptr)
        bank = bucket.get_bank(self.heads_per_shard, self.layer_ordinal & 1)
        prov_conv = aligned_empty((rows, 3, self.heads_per_shard * 384), torch.bfloat16, self.device)
        prov_rec = aligned_empty((rows, self.heads_per_shard, 128, 128), torch.float32, self.device)
        specs = stage28_output_specs(self.heads_per_shard, rows)
        status_buffers = {idx: aligned_empty(specs[idx][0], specs[idx][1], self.device) for idx in STATUS_OUTPUT_INDICES}
        outputs = bank.build_outputs_list(status_buffers)
        inputs: dict[str, Any] = {name: None for name in MC2_INPUT_NAMES}
        inputs.update(self._static_weights)
        inputs.update(
            x=bucket.shared_inputs["x"],
            post_h_out=bucket.shared_inputs["post_h_out"],
            post_h_post=bucket.shared_inputs["post_h_post"],
            post_h_res=bucket.shared_inputs["post_h_res"],
            cu_seqlens=bucket.shared_inputs["cu_seqlens"],
            state_indices=bucket.shared_inputs["state_indices"],
            conv_state=prov_conv,
            recurrent_state=prov_rec,
        )
        groups = self._resolve_comm_groups()
        input_list = [inputs[name] for name in MC2_INPUT_NAMES]
        if self.config.repeatable_executor and self._prepare_fn is not None:
            handle = self._prepare_fn(
                input_list,
                outputs,
                self._int_attrs,
                self._float_attrs,
                *groups,
                bucket.shared_workspace,
            )
            bucket.update_workspace(handle.workspace())
            runner: _PreparedCppRunner | _TorchBoundBuffers = _PreparedCppRunner(handle, inputs, outputs, stream)
        else:
            runner = _TorchBoundBuffers(
                inputs,
                outputs,
                self._attrs,
                groups,
                stream,
                self._op,
                sync_on_run=self.config.sync_per_layer,
            )
        binding = _LayerGeometryBinding(
            key=(rows, stream_ptr),
            bucket=bucket,
            bank=bank,
            prov_conv=prov_conv,
            prov_rec=prov_rec,
            status_buffers=status_buffers,
            outputs=outputs,
            runner=runner,
        )
        self._bindings[(rows, stream_ptr)] = binding
        return binding

    def ensure_geometry(self, rows: int, stream: Any | None = None) -> _LayerGeometryBinding:
        self._check_epoch_and_groups()
        if stream is None:
            stream = torch.npu.current_stream(self.device)
        key = (rows, int(stream.npu_stream))
        binding = self._bindings.get(key)
        if binding is not None:
            self._bindings.move_to_end(key)
            self.pool.touch_bucket(key)
            return binding
        return self._build_binding(rows, stream)

    def _validate_and_bind_fallback(self, inputs: Mapping[str, Any], attrs: Mapping[str, Any]) -> None:
        if self._static_weights is None:
            self._static_weights = {
                name: inputs[name]
                for name in MC2_INPUT_NAMES
                if name
                not in (
                    "x",
                    "post_h_res",
                    "post_h_out",
                    "post_h_post",
                    "conv_state",
                    "recurrent_state",
                    "cu_seqlens",
                    "state_indices",
                    "external_route_ids",
                    "external_route_weights",
                )
            }
        if self._attrs is None:
            self._attrs = dict(attrs)
            self._int_attrs = [_integer_attribute(name, self._attrs[name]) for name in _INT_ATTRIBUTES]
            self._float_attrs = [float(self._attrs[name]) for name in _FLOAT_ATTRIBUTES]

    def forward_layer(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_slots_i64: torch.Tensor,
        rec_slots_i64: torch.Tensor,
        is_first_layer: bool,
        conv_writeback: torch.Tensor | None = None,
        capture_routes: bool | None = None,
        defer_commit: bool | None = None,
        clone_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent MC2 session use is unsupported; abort the distributed job")
        try:
            self.last_route_ids = None
            self.last_route_weights = None
            do_capture = self.capture_routes if capture_routes is None else bool(capture_routes)
            defer = self.config.defer_state_commit if defer_commit is None else bool(defer_commit)
            rows = int(hidden_states.shape[0])
            stream = torch.npu.current_stream(self.device)
            binding = self.ensure_geometry(rows, stream)
            shared = binding.bucket.shared_inputs
            if is_first_layer:
                shared["x"].copy_(hidden_states.unsqueeze(1).expand(-1, 4, -1))
                shared["post_h_out"].zero_()
                shared["post_h_post"][:, :4].zero_()
                shared["post_h_res"][:, :16].copy_(binding.bucket.eye_flat)
            else:
                if residual is None or post is None or comb is None:
                    raise ValueError("non-first layer requires residual, post, and comb")
                if shared["x"].data_ptr() != residual.data_ptr():
                    shared["x"].copy_(residual)
                if shared["post_h_out"].data_ptr() != hidden_states.data_ptr():
                    shared["post_h_out"].copy_(hidden_states)
                shared["post_h_post"][:, :4].copy_(post.reshape(rows, 4))
                shared["post_h_res"][:, :16].copy_(comb.reshape(rows, 16))
            _gather_compact_slots(binding.prov_conv, conv_state, conv_slots_i64, is_conv=True)
            _gather_compact_slots(binding.prov_rec, recurrent_state, rec_slots_i64, is_conv=False)
            reset_status_outputs(binding.outputs)
            out = binding.runner.run(sync=self.config.sync_per_layer)
            if self.config.consumer_readback:
                _ = out[37].detach().cpu()
                _ = out[48].detach().cpu()
            bank = binding.bank
            output = out[37]
            ffn_residual = out[15].view(rows, 4, 4096)
            bank.compact_post.reshape(rows, 4).copy_(out[17][:, :4])
            bank.compact_comb.reshape(rows, 16).copy_(out[18][:, :16])
            if do_capture:
                self.last_route_ids = out[22][:, :8].clone()
                self.last_route_weights = out[21][:, :8].clone()
            entry = _PendingCommit(
                session=self,
                prov_conv=binding.prov_conv,
                prov_rec=binding.prov_rec,
                dst_conv=conv_state,
                dst_rec=recurrent_state,
                conv_slots=conv_slots_i64,
                rec_slots=rec_slots_i64,
                conv_writeback=conv_writeback,
                status_buffers=tuple(binding.status_buffers.values()),
            )
            if defer:
                self.pool.record_pending(entry)
            else:
                # Diagnostic / benchmark mode: isolate single layer status check & commit.
                single_tx_pending = self.pool._pending
                self.pool._pending = [entry]
                try:
                    self.pool.finalize_step(self.control_device, self.config.control_group)
                finally:
                    self.pool._pending = single_tx_pending
            if clone_outputs:
                return (
                    aligned_copy(output),
                    aligned_copy(ffn_residual),
                    bank.compact_post.clone(),
                    bank.compact_comb.clone(),
                )
            return output, ffn_residual, bank.compact_post, bank.compact_comb
        except BaseException:
            self._failed = True
            self.pool.abort_step(mark_failed=True)
            raise
        finally:
            self._lock.release()

    def forward(
        self,
        inputs: Mapping[str, Any],
        attrs: Mapping[str, Any],
        identity: StepIdentity | Any,
        *,
        validated_step: bool = False,
        capture_routes: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible single-call entrypoint accepting a full 47-tensor dict."""
        if validated_step is not True:
            raise ValueError("MC2 requires per-step scheduler and input attestation")
        if hasattr(identity, "validate"):
            identity.validate()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent MC2 session use is unsupported; abort the distributed job")
        self.last_route_ids = None
        self.last_route_weights = None
        do_capture = self.capture_routes if capture_routes is None else bool(capture_routes)
        try:
            self._check_epoch_and_groups()
            self._validate_and_bind_fallback(inputs, attrs)
            x = inputs["x"]
            rows = int(x.shape[0])
            slots = inputs["state_indices"].to(dtype=torch.int64)
            conv_slots = slots[:, 0]
            rec_slots = slots[:, 1]
            conv = inputs["conv_state"]
            rec = inputs["recurrent_state"]
            if not non_overlapping_positive_strides(rec):
                raise ValueError("recurrent state view may overlap")
            stream = torch.npu.current_stream(self.device)
            binding = self.ensure_geometry(rows, stream)
            shared = binding.bucket.shared_inputs
            shared["x"].copy_(x)
            shared["post_h_out"].copy_(inputs["post_h_out"])
            shared["post_h_post"].copy_(inputs["post_h_post"])
            shared["post_h_res"].copy_(inputs["post_h_res"])
            _gather_compact_slots(binding.prov_conv, conv, conv_slots, is_conv=True)
            _gather_compact_slots(binding.prov_rec, rec, rec_slots, is_conv=False)
            reset_status_outputs(binding.outputs)
            out = binding.runner.run(sync=True)
            entry = _PendingCommit(
                session=self,
                prov_conv=binding.prov_conv,
                prov_rec=binding.prov_rec,
                dst_conv=conv,
                dst_rec=rec,
                conv_slots=conv_slots,
                rec_slots=rec_slots,
                conv_writeback=None,
                status_buffers=tuple(binding.status_buffers.values()),
            )
            prev_pending = self.pool._pending
            self.pool._pending = [entry]
            try:
                self.pool.finalize_step(self.control_device, self.config.control_group)
            finally:
                self.pool._pending = prev_pending
            output = aligned_copy(out[37])
            ffn_residual = aligned_copy(out[15].view(rows, 4, 4096))
            post_view = out[17][:, :4].reshape(rows, 4, 1).clone(memory_format=torch.contiguous_format)
            comb_view = out[18][:, :16].reshape(rows, 4, 4).clone(memory_format=torch.contiguous_format)
            if do_capture:
                self.last_route_ids = aligned_copy(out[22][:, :8])
                self.last_route_weights = aligned_copy(out[21][:, :8])
            return output, ffn_residual, post_view, comb_view
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an MC2 session while a launch is active")
        try:
            self.last_route_ids = None
            self.last_route_weights = None
            if self._closed:
                return
            self._closed = True
            close_error = None
            for binding in self._bindings.values():
                try:
                    binding.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
            self._bindings.clear()
            scratch = self.pool.get_agree_scratch() if self.control_device == self.pool.control_device else None
            agree_or_abort(
                close_error,
                self.control_device,
                self.config.control_group,
                "MC2 session teardown",
                scratch=scratch,
            )
            if close_error is not None:
                raise close_error
        finally:
            self._lock.release()


def load_torch_mc2_extension(*, load_module: bool = True):
    """Load the custom ACLNN-backed Torch NPU extension once per worker."""
    module = importlib.import_module("cann_ops_transformer_custom.ops.attention.mega_kernel_mc2.mega_kernel_mc2")
    if load_module:
        module._builder.load()
    op = torch.ops.cann_ops_transformer.mega_kernel_mc2_out
    prepare_fn = getattr(module, "prepare_mega_kernel_mc2", None)
    if load_module and prepare_fn is not None:
        supports_fn = getattr(module, "supports_repeatable_executor", None)
        if callable(supports_fn) and not supports_fn():
            prepare_fn = None
    return op, prepare_fn


def create_torch_session(
    config: Mc2SessionConfig,
    model_epoch: Callable[[], str],
    *,
    layer_idx: int = 0,
    layer_ordinal: int = 0,
    static_weights: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    buffer_pool: SharedStepBufferPool | None = None,
    op: Any = None,
    prepare_fn: Any = None,
) -> MegaKernelTorchSession:
    """Create a production Torch session, loading the extension if not pre-supplied."""
    if op is None:
        op, prepare_fn = load_torch_mc2_extension()
    return MegaKernelTorchSession(
        config,
        model_epoch,
        op=op,
        prepare_fn=prepare_fn,
        layer_idx=layer_idx,
        layer_ordinal=layer_ordinal,
        static_weights=static_weights,
        attrs=attrs,
        buffer_pool=buffer_pool,
    )
