# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental vLLM-Ascend GLM5Next eager Decode dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import threading
import types
from typing import Any

from vllm_ascend.models.glm5next.mega_kernel_abi import (
    MC2_INPUT_NAMES,
    aligned_empty,
    non_overlapping_positive_strides as _non_overlapping_positive_strides,
)
from vllm_ascend.models.glm5next.mega_kernel_collectives import (
    MegaKernelCollectiveError,
    agree_or_abort,
    agree_preflight,
)

logger = logging.getLogger(__name__)


class UnsupportedDecode(ValueError):
    """Before a collective launch, this layer must use the original path."""


def validate_model_path_config(config: dict[str, Any]) -> None:
    """Reject stale mode keys instead of silently changing their semantics."""
    if "routing_mode" in config or "execution_mode" in config:
        raise ValueError("remove obsolete routing_mode and execution_mode; the model has one MC2 path")


def benchmark_consumer_ready(outputs: Any) -> None:
    """Use identical timing boundaries on the two replay paths."""
    import torch

    del outputs
    torch.npu.synchronize()


def coordinate_layer_preflight(
    error: Exception | None, pure_decode: bool, device: Any, group: Any, *, scratch: Any = None
) -> bool:
    """Agree on accept, fallback, or abort in one group reduction."""
    return agree_preflight(error, pure_decode, device, group, scratch=scratch, unsupported_exc=UnsupportedDecode)


@dataclass(frozen=True)
class DecodeInputs:
    metadata: Any
    conv_state: Any
    conv_writeback: Any
    recurrent_state: Any
    state_indices: Any
    cu_seqlens: Any
    rows: int
    conv_slots_i64: Any = None
    recurrent_slots_i64: Any = None


def _int_attr(obj: Any, name: str) -> int:
    value = getattr(obj, name, None)
    if type(value) is not int or value < 0:
        raise UnsupportedDecode(f"{name} is not a nonnegative Python integer")
    return value


def _one_dim_slots(tensor: Any, count: int, name: str, torch: Any) -> Any:
    if not isinstance(tensor, torch.Tensor):
        raise UnsupportedDecode(f"{name} is not a tensor")
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 1 or tensor.numel() != count or tensor.dtype not in (torch.int32, torch.int64):
        raise UnsupportedDecode(f"{name} must be a one-dimensional integer tensor for each request")
    return tensor


def _tensor_version(tensor: Any) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


def prepare_decode_inputs(
    layer: Any,
    forward_context: Any,
    hidden_states: Any,
    residual: Any,
    post: Any,
    comb: Any,
    *,
    execution_mode: str,
    scheduler_pure_decode: bool,
    step_cache: dict[Any, Any] | None = None,
    cu_seqlens_buf: Any = None,
    compact_strided_conv: bool = False,
) -> DecodeInputs:
    """Build and validate MC2 Decode inputs, sharing host metadata reads within a step."""
    import torch

    if scheduler_pure_decode is not True:
        raise UnsupportedDecode("scheduler did not prove a pure, ordinary decode batch")
    if execution_mode != "eager":
        raise UnsupportedDecode("only an explicitly verified eager execution is currently supported")
    if (
        getattr(layer, "layer_kind", None) != "kda"
        or getattr(layer, "_mlp_is_moe", False) is not True
        or getattr(layer, "mhc", False) is not True
        or getattr(layer, "is_mtp_layer", True) is not False
        or getattr(layer, "is_sequence_parallel", True) is not False
    ):
        raise UnsupportedDecode("layer is not a supported KDA+MoE+mHC layer")
    idx = _int_attr(layer, "layer_idx")
    total = _int_attr(layer, "num_hidden_layers")
    if total <= 0 or idx >= total or hidden_states is None:
        raise UnsupportedDecode("invalid layer index or missing hidden states")
    first = idx == 0 and post is None
    if first:
        if residual is not None or comb is not None:
            raise UnsupportedDecode("first layer received inconsistent incoming mHC state")
    elif any(t is None for t in (residual, post, comb)):
        raise UnsupportedDecode("incoming deferred mHC state is incomplete")
    attn = layer.self_attn
    if getattr(attn, "pcp_size", None) != 1 or getattr(attn, "projection_tp", None) is not None:
        raise UnsupportedDecode("PCP and separate projection TP are not implemented")
    if getattr(attn, "use_recoverssm", False):
        raise UnsupportedDecode("RecoverSSM is not implemented")
    if getattr(layer.mlp, "enable_eplb", False):
        raise UnsupportedDecode("dynamic EPLB ownership is not implemented")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 1:
        raise UnsupportedDecode("hidden state tensor missing")
    rows = hidden_states.shape[0]
    if rows <= 0:
        raise UnsupportedDecode("empty batch")
    metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        raise UnsupportedDecode("missing actual attention metadata (profile or warmup)")
    metadata = metadata.get(attn.prefix)
    if metadata is None or getattr(metadata, "pcp_context", None) is not None:
        raise UnsupportedDecode("missing KDA metadata or PCP metadata is active")
    if _int_attr(metadata, "num_prefills") != 0 or _int_attr(metadata, "num_spec_decodes") != 0:
        raise UnsupportedDecode("prefill, mixed or speculative batch")
    count = _int_attr(metadata, "num_decodes")
    actual = _int_attr(metadata, "num_actual_tokens")
    if count != rows or actual != rows or _int_attr(metadata, "num_decode_tokens") != rows:
        raise UnsupportedDecode("only one real token per request without padding is supported")
    decode = getattr(metadata, "non_spec_decode_metadata", None)
    conv_meta = getattr(decode, "causal_conv1d", None)
    query = getattr(metadata, "non_spec_query_start_loc", None)
    if not isinstance(query, torch.Tensor) or query.ndim != 1 or query.numel() != rows + 1:
        raise UnsupportedDecode("decode query boundaries are missing or padded")
    conv_slots = _one_dim_slots(getattr(conv_meta, "cache_indices", None), rows, "conv slots", torch)
    recurrent_slots = _one_dim_slots(
        getattr(metadata, "non_spec_state_indices_tensor", None), rows, "recurrent slots", torch
    )
    if conv_slots.device != hidden_states.device or recurrent_slots.device != hidden_states.device:
        raise UnsupportedDecode("state slots are not on the model device")
    cache = getattr(attn, "kv_cache", None)
    if not isinstance(cache, (tuple, list)) or len(cache) != 2:
        raise UnsupportedDecode("KDA KV state pair is unavailable")
    conv_state, recurrent_state = cache
    if not isinstance(conv_state, torch.Tensor) or not isinstance(recurrent_state, torch.Tensor):
        raise UnsupportedDecode("KDA state must be tensors")
    if getattr(attn, "_conv_state_dim_first", False):
        conv_state = conv_state.transpose(-1, -2)
    heads = _int_attr(attn, "local_num_heads")
    if (
        heads <= 0
        or conv_state.dtype != torch.bfloat16
        or recurrent_state.dtype != torch.float32
        or conv_state.ndim != 3
        or tuple(conv_state.shape[1:]) != (3, heads * 384)
        or recurrent_state.ndim != 4
        or tuple(recurrent_state.shape[1:]) != (heads, 128, 128)
        or conv_state.device != hidden_states.device
        or recurrent_state.device != hidden_states.device
    ):
        raise UnsupportedDecode("KDA state dtype, shape or device differs from the MC2 ABI")
    conv_writeback = None
    transposed_contiguous = getattr(attn, "_conv_state_dim_first", False) and cache[0].is_contiguous()
    if not conv_state.is_contiguous() and not transposed_contiguous:
        if not (compact_strided_conv and _non_overlapping_positive_strides(conv_state)):
            conv_writeback = conv_state
            conv_state = conv_state.contiguous()
    if not _non_overlapping_positive_strides(recurrent_state):
        raise UnsupportedDecode("recurrent state layout may overlap")

    conv_cap = int(conv_state.shape[0])
    rec_cap = int(recurrent_state.shape[0])
    cache_key = (
        query.data_ptr(),
        _tensor_version(query),
        query.dtype,
        query.device,
        conv_slots.data_ptr(),
        _tensor_version(conv_slots),
        conv_slots.dtype,
        conv_slots.stride(0),
        recurrent_slots.data_ptr(),
        _tensor_version(recurrent_slots),
        recurrent_slots.dtype,
        recurrent_slots.stride(0),
        rows,
        conv_cap,
        rec_cap,
        hidden_states.device,
    )
    if step_cache is not None and cache_key in step_cache:
        cached = step_cache[cache_key]
        return DecodeInputs(
            metadata,
            conv_state,
            conv_writeback,
            recurrent_state,
            cached["state_indices"],
            cached["cu_seqlens"],
            rows,
            cached["conv_slots_i64"],
            cached["recurrent_slots_i64"],
        )

    packed_query = query.device == hidden_states.device and query.dtype in (torch.int32, torch.int64)
    if not packed_query:
        boundaries = query.detach().cpu().tolist()
        if boundaries != list(range(rows + 1)):
            raise UnsupportedDecode("not exactly one token for each request")

    conv_slots_i64 = conv_slots.to(torch.int64)
    same_slots = (
        conv_slots.data_ptr() == recurrent_slots.data_ptr()
        and conv_slots.dtype == recurrent_slots.dtype
        and tuple(conv_slots.stride()) == tuple(recurrent_slots.stride())
    )
    rec_slots_i64 = conv_slots_i64 if same_slots else recurrent_slots.to(torch.int64)
    slot_pairs = torch.stack((conv_slots_i64, rec_slots_i64), dim=1)
    if packed_query:
        if same_slots:
            packed = torch.cat((query.to(torch.int64), conv_slots_i64))
            metadata_values = packed.detach().cpu().tolist()
            if metadata_values[: rows + 1] != list(range(rows + 1)):
                raise UnsupportedDecode("not exactly one token for each request")
            vals = metadata_values[rows + 1 :]
            if len(set(vals)) != rows or any(v < 0 or v >= min(conv_cap, rec_cap) for v in vals):
                raise UnsupportedDecode("conv/recurrent slots contain padding, invalid or duplicate state")
        else:
            packed = torch.cat((query.to(torch.int64), slot_pairs.reshape(-1)))
            metadata_values = packed.detach().cpu().tolist()
            if metadata_values[: rows + 1] != list(range(rows + 1)):
                raise UnsupportedDecode("not exactly one token for each request")
            flat_pairs = metadata_values[rows + 1 :]
            conv_vals = flat_pairs[::2]
            rec_vals = flat_pairs[1::2]
            for name, capacity, vals in (
                ("conv", conv_cap, conv_vals),
                ("recurrent", rec_cap, rec_vals),
            ):
                if any(v < 0 or v >= capacity for v in vals) or len(set(vals)) != rows:
                    raise UnsupportedDecode(f"{name} slots contain padding, invalid or duplicate state")
    else:
        host_pairs = slot_pairs.detach().cpu().tolist()
        for column, name, capacity in ((0, "conv", conv_cap), (1, "recurrent", rec_cap)):
            vals = [pair[column] for pair in host_pairs]
            if any(v < 0 or v >= capacity for v in vals) or len(set(vals)) != rows:
                raise UnsupportedDecode(f"{name} slots contain padding, invalid or duplicate state")

    indices = slot_pairs.to(dtype=torch.int32).contiguous()
    if cu_seqlens_buf is not None:
        cu_seqlens = cu_seqlens_buf
    elif step_cache is not None:
        cu_key = ("cu_seqlens", hidden_states.device)
        cu_seqlens = step_cache.get(cu_key)
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)
            step_cache[cu_key] = cu_seqlens
    else:
        device = hidden_states.device
        stream = int(torch.npu.current_stream(device).npu_stream) if device.type == "npu" else None
        seq_cache_key = (device, stream)
        template = getattr(layer, "_megakernel_seq_template", None)
        if template is None or template[0] != seq_cache_key:
            template = (seq_cache_key, torch.tensor([0, 1], dtype=torch.int32, device=device))
            layer._megakernel_seq_template = template
        cu_seqlens = template[1]

    if step_cache is not None:
        step_cache.setdefault("_refs", []).append((cache[0], cache[1], query, conv_slots, recurrent_slots))
        step_cache[cache_key] = {
            "state_indices": indices,
            "cu_seqlens": cu_seqlens,
            "conv_slots_i64": conv_slots_i64,
            "recurrent_slots_i64": rec_slots_i64,
        }
    return DecodeInputs(
        metadata,
        conv_state,
        conv_writeback,
        recurrent_state,
        indices,
        cu_seqlens,
        rows,
        conv_slots_i64,
        rec_slots_i64,
    )


class StepDecodeCoordinator:
    """Coordinate step-level preflight (1 collective) and deferred state commit across target layers."""

    def __init__(
        self,
        *,
        gate: Any,
        pool: Any,
        control_device: Any,
        control_group: Any,
        layer_entries: list[tuple[Any, Any]],
        defer_state_commit: bool = True,
    ):
        self.gate = gate
        self.pool = pool
        self.control_device = control_device
        self.control_group = control_group
        self.layer_entries = list(layer_entries)
        self.first_layer_idx = self.layer_entries[0][0].layer_idx
        self.last_layer_idx = self.layer_entries[-1][0].layer_idx
        self.defer_state_commit = bool(defer_state_commit)
        self._first_inputs_cache: dict[int, DecodeInputs] = {}

    def on_step_exit(self, failed: bool) -> None:
        self._first_inputs_cache.clear()
        if self.pool is not None and self.pool.has_pending():
            self.pool.abort_step(mark_failed=True)
            if not failed:
                raise MegaKernelCollectiveError(
                    "MC2 step exited with uncommitted deferred state; abort distributed job"
                )

    def preflight_all_layers(
        self,
        *,
        first_layer: Any,
        forward_context: Any,
        hidden_states: Any,
        residual: Any,
        post: Any,
        comb: Any,
        execution_mode: str,
        scheduler_pure_decode: bool,
        step_cache: dict[Any, Any] | None,
    ) -> tuple[bool, DecodeInputs | None, Exception | None]:
        """Validate all target layers and prepare geometry across ranks before entering HCCL."""
        self._first_inputs_cache.clear()
        first_inputs = None
        local_error: Exception | None = None
        try:
            import torch

            rows = int(hidden_states.shape[0]) if hasattr(hidden_states, "shape") else 0
            stream = torch.npu.current_stream(hidden_states.device) if rows > 0 and hasattr(torch, "npu") else None
            for layer, backend in self.layer_entries:
                if layer is first_layer:
                    decoded = prepare_decode_inputs(
                        layer,
                        forward_context,
                        hidden_states,
                        residual,
                        post,
                        comb,
                        execution_mode=execution_mode,
                        scheduler_pure_decode=scheduler_pure_decode,
                        step_cache=step_cache,
                        compact_strided_conv=hasattr(backend.session, "forward_layer"),
                    )
                    backend.validate(layer, decoded, hidden_states, residual, post, comb)
                    first_inputs = decoded
                else:
                    # Validate non-first layer metadata & KV cache layout using dummy placeholders
                    # matching first layer's incoming mHC shapes.
                    decoded = prepare_decode_inputs(
                        layer,
                        forward_context,
                        hidden_states,
                        hidden_states,
                        hidden_states,
                        hidden_states,
                        execution_mode=execution_mode,
                        scheduler_pure_decode=scheduler_pure_decode,
                        step_cache=step_cache,
                        compact_strided_conv=hasattr(backend.session, "forward_layer"),
                    )
                    backend.validate_static_once(layer)
                if hasattr(backend.session, "ensure_geometry"):
                    backend.session.ensure_geometry(decoded.rows, stream)
        except Exception as exc:
            local_error = exc
        scratch = self.pool.get_preflight_scratch() if self.pool is not None else None
        accepted = coordinate_layer_preflight(
            local_error,
            scheduler_pure_decode,
            self.control_device,
            self.control_group,
            scratch=scratch,
        )
        if not accepted:
            self._first_inputs_cache.clear()
            return False, None, local_error
        if local_error is not None or first_inputs is None:
            raise MegaKernelCollectiveError("coordinator accepted an unsupported step; abort job")
        return True, first_inputs, None

    def finalize_step(self, *, local_error: Exception | None = None, coordinate_error: bool = False) -> None:
        if self.defer_state_commit and self.pool is not None:
            if local_error is None:
                self.pool.finalize_step(self.control_device, self.control_group)
            else:
                self.pool.finalize_step(self.control_device, self.control_group, local_error=local_error)
        elif coordinate_error or local_error is not None:
            agree_or_abort(local_error, self.control_device, self.control_group, "model final layer contraction")
            if local_error is not None:
                raise local_error


class DecodeLayerAdapter:
    """Wrap one existing layer, coordinating the decision before any MC2 launch."""

    def __init__(
        self,
        backend: Callable[..., Any],
        coordinate: Callable[..., bool],
        execution_mode: Callable[[], str],
        decode_evidence: Callable[[Any], bool],
        benchmark: Callable[..., bool] | None = None,
        coordinator: StepDecodeCoordinator | None = None,
    ):
        if not all(callable(f) for f in (backend, coordinate, execution_mode, decode_evidence)):
            raise TypeError("backend, coordinator and execution mode are required")
        self.backend, self.coordinate, self.execution_mode = backend, coordinate, execution_mode
        self.decode_evidence = decode_evidence
        if benchmark is not None and not callable(benchmark):
            raise TypeError("benchmark must be callable")
        self.benchmark = benchmark
        self.coordinator = coordinator
        self._benchmark_active = False
        self._benchmark_done = False

    def install(self, layer: Any) -> None:
        if hasattr(layer, "_megakernel_original_forward"):
            raise RuntimeError("already installed")
        original = layer.forward
        stats = {"hits": 0, "nondecode": 0, "group_fallback": 0}
        layer._megakernel_stats = stats
        adapter = self
        execution_lock = threading.Lock()

        def forward_impl(
            this: Any, positions: Any, hidden_states: Any, residual: Any = None, post: Any = None, comb: Any = None
        ):
            from vllm.forward_context import get_forward_context

            if not adapter.decode_evidence(this):
                stats["nondecode"] += 1
                return original(positions, hidden_states, residual, post, comb)

            coord = adapter.coordinator
            forward_context = get_forward_context()
            use_step_fast_path = coord is not None and not adapter._benchmark_active
            step_cache = coord.gate.current_step_cache() if coord is not None and hasattr(coord.gate, "current_step_cache") else None
            use_compact_conv = hasattr(getattr(adapter.backend, "session", None), "forward_layer")

            if use_step_fast_path:
                assert coord is not None
                if hasattr(coord.gate, "register_step_cleanup"):
                    coord.gate.register_step_cleanup(coord.on_step_exit)
                verdict = coord.gate.get_step_preflight()
                if verdict is None:
                    accepted, inputs, local_error = coord.preflight_all_layers(
                        first_layer=this,
                        forward_context=forward_context,
                        hidden_states=hidden_states,
                        residual=residual,
                        post=post,
                        comb=comb,
                        execution_mode=adapter.execution_mode(),
                        scheduler_pure_decode=adapter.decode_evidence(this),
                        step_cache=step_cache,
                    )
                    coord.gate.set_step_preflight("accepted" if accepted else "fallback")
                    if not accepted:
                        stats["group_fallback"] += 1
                        if stats["group_fallback"] == 1:
                            logger.warning(
                                "MegaKernel step used group fallback at layer %s; local reason: %s",
                                this.layer_idx,
                                local_error,
                            )
                        return original(positions, hidden_states, residual, post, comb)
                elif verdict == "fallback":
                    stats["group_fallback"] += 1
                    return original(positions, hidden_states, residual, post, comb)
                elif verdict == "accepted":
                    try:
                        inputs = prepare_decode_inputs(
                            this,
                            forward_context,
                            hidden_states,
                            residual,
                            post,
                            comb,
                            execution_mode=adapter.execution_mode(),
                            scheduler_pure_decode=adapter.decode_evidence(this),
                            step_cache=step_cache,
                            compact_strided_conv=use_compact_conv,
                        )
                        adapter.backend.validate(this, inputs, hidden_states, residual, post, comb)
                    except Exception as exc:
                        raise MegaKernelCollectiveError(
                            f"MegaKernel mid-step failure on layer {this.layer_idx} after group accept; abort job"
                        ) from exc
                else:
                    raise RuntimeError(f"unknown step preflight verdict: {verdict}")
            else:
                local_error = None
                inputs = None
                try:
                    inputs = prepare_decode_inputs(
                        this,
                        forward_context,
                        hidden_states,
                        residual,
                        post,
                        comb,
                        execution_mode=adapter.execution_mode(),
                        scheduler_pure_decode=adapter.decode_evidence(this),
                        compact_strided_conv=use_compact_conv,
                    )
                except Exception as exc:
                    local_error = exc
                if local_error is None:
                    try:
                        adapter.backend.validate(this, inputs, hidden_states, residual, post, comb)
                    except Exception as exc:
                        local_error = exc
                accepted = adapter.coordinate(this, local_error)
                if type(accepted) is not bool:
                    raise RuntimeError("coordinator returned no definite decision")
                if not accepted:
                    stats["group_fallback"] += 1
                    if stats["group_fallback"] == 1:
                        logger.warning(
                            "MegaKernel layer %s used group fallback; local reason: %s", this.layer_idx, local_error
                        )
                    return original(positions, hidden_states, residual, post, comb)
                if local_error is not None or inputs is None:
                    raise RuntimeError("coordinator accepted an unsupported local call")

            if adapter.benchmark is not None and not adapter._benchmark_active and not adapter._benchmark_done:
                adapter._benchmark_active = True
                saved_pending = None
                if coord is not None and coord.pool is not None:
                    saved_pending = list(coord.pool._pending)
                    coord.pool._pending = []
                try:
                    completed = adapter.benchmark(this, positions, hidden_states, residual, post, comb, inputs)
                    if type(completed) is not bool:
                        raise RuntimeError("single-layer benchmark must return a completion flag")
                    adapter._benchmark_done = completed
                finally:
                    if saved_pending is not None:
                        coord.pool._pending = saved_pending
                    adapter._benchmark_active = False

            from vllm_ascend.models.glm5next.kda import (
                maybe_save_kv_layer_to_connector,
                record_attention_compute_start,
                wait_for_kv_layer_from_connector,
            )

            wait_for_kv_layer_from_connector(this.self_attn.prefix)
            record_attention_compute_start()
            deferred_final = (
                use_step_fast_path
                and coord is not None
                and this.layer_idx == coord.last_layer_idx
                and this.layer_idx == this.num_hidden_layers - 1
                and isinstance(adapter.backend, MegaKernelModelBackend)
            )
            if adapter._benchmark_active:
                out = adapter.backend(
                    this,
                    positions,
                    hidden_states,
                    residual,
                    post,
                    comb,
                    inputs,
                    defer_commit=False,
                    capture_routes=None,
                    clone_outputs=True,
                )
            elif coord is not None:
                backend_kwargs: dict[str, Any] = {
                    "defer_commit": coord.defer_state_commit,
                    "capture_routes": None,
                    "clone_outputs": False,
                }
                if deferred_final:
                    backend_kwargs["defer_final_contract"] = True
                out = adapter.backend(
                    this,
                    positions,
                    hidden_states,
                    residual,
                    post,
                    comb,
                    inputs,
                    **backend_kwargs,
                )
            else:
                out = adapter.backend(this, positions, hidden_states, residual, post, comb, inputs)
            if not isinstance(out, tuple) or len(out) != 4:
                raise RuntimeError("MC2 backend did not return the layer's four values")
            if deferred_final:
                assert coord is not None
                final_error: Exception | None = None
                try:
                    out = _finish_final_layer(this, out, hidden_states.device)
                except Exception as exc:
                    final_error = exc
                coord.finalize_step(local_error=final_error, coordinate_error=True)
                if final_error is not None:
                    raise final_error
                maybe_save_kv_layer_to_connector(this.self_attn.prefix, [])
            else:
                if use_step_fast_path and coord is not None and this.layer_idx == coord.last_layer_idx:
                    coord.finalize_step()
                maybe_save_kv_layer_to_connector(this.self_attn.prefix, [])
            stats["hits"] += 1
            if stats["hits"] == 1:
                logger.info("MegaKernel layer %s first successful Decode call", this.layer_idx)
            return out

        def forward(
            this: Any, positions: Any, hidden_states: Any, residual: Any = None, post: Any = None, comb: Any = None
        ):
            if not execution_lock.acquire(blocking=False):
                raise RuntimeError("concurrent MegaKernel layer calls are unsupported; abort the distributed job")
            try:
                return forward_impl(this, positions, hidden_states, residual, post, comb)
            finally:
                execution_lock.release()

        layer._megakernel_original_forward = original
        layer._megakernel_session = getattr(adapter.backend, "session", None)
        layer._megakernel_benchmark_forward = types.MethodType(forward_impl, layer)
        layer.forward = types.MethodType(forward, layer)


def _copy_live_kda_slots(cache: Any, state_indices: Any) -> tuple[Any, Any]:
    """Build temporary KDA pools containing only this step's live state slots."""
    import torch

    if len(cache) != 2 or state_indices.ndim != 2 or state_indices.shape[1] != 2:
        raise ValueError("invalid KDA cache or state-index mapping")
    result = []
    for column, source in enumerate(cache):
        slots = state_indices[:, column].to(dtype=torch.int64)
        target = torch.empty_like(source)
        target.index_copy_(0, slots, source.index_select(0, slots))
        result.append(target)
    return result[0], result[1]


def _aligned_mhc_buffers(rows: int, dtype: Any, device: Any, aligned_empty: Callable[..., Any]) -> tuple[Any, Any]:
    """Allocate disjoint, independently aligned post and combination buffers."""
    mixing = aligned_empty((2, rows, 128), dtype, device)
    mixing.zero_()
    return mixing[0], mixing[1]


class _MhcBufferCache:
    """Reuse the last layer-local geometry; callers serialize layer execution."""

    def __init__(self):
        self.key = None
        self.buffers = None

    def get(self, rows: int, dtype: Any, device: Any, aligned_empty: Callable[..., Any]) -> tuple[Any, Any]:
        key = (rows, dtype, device)
        if key != self.key:
            self.buffers = _aligned_mhc_buffers(rows, dtype, device, aligned_empty)
            self.key = key
        else:
            for value in self.buffers:
                value.zero_()
        return self.buffers


def _validate_internal_router(layer: Any, router: Any) -> None:
    """Reject model routing features that the MC2 internal router cannot implement."""
    routed = layer.mlp.experts.routed_experts
    if router is None or type(router).__name__ not in ("AscendFusedTopKRouter", "AscendGroupedTopKRouter"):
        raise UnsupportedDecode("MC2 internal routing requires a known Ascend top-k router")
    if type(router).__name__ == "AscendGroupedTopKRouter" and getattr(router, "use_grouped_topk", None) is not True:
        raise UnsupportedDecode("ungrouped native router uses different correction-bias weight semantics")
    if (
        getattr(router, "top_k", None) != 8
        or getattr(router, "scoring_func", None) != "sigmoid"
        or getattr(router, "renormalize", None) is not True
        or getattr(router, "routed_scaling_factor", None) != 2.5
        or getattr(router, "num_expert_group", None) not in (None, 1)
        or getattr(router, "topk_group", None) not in (None, 1)
    ):
        raise UnsupportedDecode("model router configuration differs from MC2 sigmoid top-8 contract")
    if any(
        getattr(router, field, None) is not None
        for field in ("eplb_state", "custom_routing_function", "tid2eid", "bias_vl")
    ):
        raise UnsupportedDecode("MC2 internal routing does not implement custom routing or EPLB")
    if getattr(routed, "log2phy", None) is not None or getattr(routed, "mix_placement", False):
        raise UnsupportedDecode("MC2 internal routing does not implement expert remapping or mixed placement")
    if getattr(router, "num_fused_shared_experts", 0):
        raise UnsupportedDecode("MC2 internal routing does not append shared expert IDs")
    if getattr(router, "e_score_correction_bias", None) is not getattr(layer.mlp.gate, "e_score_correction_bias", None):
        raise UnsupportedDecode("MC2 internal routing requires the model gate correction bias")
    if getattr(layer.mlp.gate, "e_score_correction_bias", None) is None:
        raise UnsupportedDecode("MC2 internal routing requires a correction bias")


def _finish_final_layer(layer: Any, result: tuple[Any, Any, Any, Any], device: Any) -> tuple[Any, Any, Any, Any]:
    """Apply final-layer mHC post/contract and ensure device stream completion before returning logits input."""
    import torch
    from vllm_ascend.models.glm5next.ops.mhc_ops import hc_contract

    output, ffn_residual, ffn_post, ffn_comb = result
    output = hc_contract(layer.hc_post(output, ffn_residual, ffn_post, ffn_comb), layer.n)
    if hasattr(torch, "npu"):
        stream = torch.npu.current_stream(device)
        sync = getattr(stream, "synchronize", None)
        if callable(sync):
            sync()
    return output, None, None, None


class MegaKernelModelBackend:
    """Assemble real model tensors and invoke a group-bound MC2 session."""

    def __init__(
        self,
        session: Any,
        static_weights: dict[str, Any],
        attrs: dict[str, Any],
        identity_factory: Callable[..., Any],
    ):
        if not callable(getattr(session, "forward", None)) or not callable(identity_factory):
            raise TypeError("MC2 session and model identity factory are required")
        self.session, self.static_weights, self.attrs = session, dict(static_weights), dict(attrs)
        self.identity_factory = identity_factory
        self._mhc_buffer_cache = _MhcBufferCache()
        self._static_validated = False

    def validate_static_once(self, layer: Any) -> None:
        if self._static_validated:
            return
        self.validate_static(layer)
        self._static_validated = True

    def validate_static(self, layer: Any) -> None:
        if not self.static_weights:
            raise UnsupportedDecode("MC2 static weights are unavailable")
        if self.attrs.get("stage_limit") != self.session.bridge.stage_limit:
            raise UnsupportedDecode("MC2 stage limit differs from the configured bridge")
        runner = layer.mlp.experts
        routed = getattr(runner, "routed_experts", None)
        if not callable(getattr(routed, "_select_experts", None)):
            raise UnsupportedDecode("vLLM runner does not expose the verified router path")
        if getattr(runner, "is_internal_router", None) is True and not callable(
            getattr(runner, "_compute_router_logits", None)
        ):
            raise UnsupportedDecode("vLLM runner does not expose its internal gate path")
        if getattr(runner, "is_internal_router", None) is not True and not callable(layer.mlp.gate):
            raise UnsupportedDecode("vLLM model does not expose its gate")
        router = getattr(routed, "router", None)
        if getattr(router, "routed_scaling_factor", None) != getattr(layer.mlp, "routed_scaling_factor", None):
            raise UnsupportedDecode("vLLM router and model routed scales differ")
        from vllm_ascend.ascend_config import get_ascend_config

        if getattr(get_ascend_config(), "enable_force_eplb", False):
            raise UnsupportedDecode("MC2 internal routing does not support forced expert load balancing")
        _validate_internal_router(layer, router)
        routed_cfg = getattr(routed, "moe_config", None)
        if (
            getattr(layer.mlp.shared_experts, "swiglu_limit", None) != 10.0
            or getattr(routed_cfg, "swiglu_limit", None) != 10.0
        ):
            raise UnsupportedDecode("model expert activation clamp differs from MC2 limit 10")

    def validate(
        self, layer: Any, decode: DecodeInputs, hidden_states: Any, residual: Any, post: Any, comb: Any
    ) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        rows = decode.rows
        if rows < 1:
            raise UnsupportedDecode("MC2 active rows are unavailable")
        self.validate_static_once(layer)
        if tuple(hidden_states.shape) != (rows, 4096):
            raise UnsupportedDecode("hidden state does not match the MC2 ABI")
        first = layer.layer_idx == 0 and post is None
        if not first:
            if tuple(residual.shape) != (rows, 4, 4096):
                raise UnsupportedDecode("mHC residual does not match the MC2 ABI")
            if tuple(post.shape) != (rows, 4, 1) or tuple(comb.shape) != (rows, 4, 4):
                raise UnsupportedDecode("incoming mHC mixing tensors do not match the model contract")
        if has_kv_transfer_group():
            raise UnsupportedDecode("MC2 adapter does not support a KV transfer connector")

    def __call__(
        self,
        layer: Any,
        positions: Any,
        hidden_states: Any,
        residual: Any,
        post: Any,
        comb: Any,
        decode: DecodeInputs,
        *,
        defer_commit: bool | None = None,
        capture_routes: bool | None = None,
        clone_outputs: bool = False,
        defer_final_contract: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        del positions
        import torch

        if hasattr(self.session, "forward_layer"):
            conv_slots_i64 = (
                decode.conv_slots_i64
                if decode.conv_slots_i64 is not None
                else decode.state_indices[:, 0].to(dtype=torch.int64)
            )
            rec_slots_i64 = (
                decode.recurrent_slots_i64
                if decode.recurrent_slots_i64 is not None
                else decode.state_indices[:, 1].to(dtype=torch.int64)
            )
            result = self.session.forward_layer(
                hidden_states=hidden_states,
                residual=residual,
                post=post,
                comb=comb,
                conv_state=decode.conv_state,
                recurrent_state=decode.recurrent_state,
                conv_slots_i64=conv_slots_i64,
                rec_slots_i64=rec_slots_i64,
                is_first_layer=(layer.layer_idx == 0 and post is None),
                conv_writeback=decode.conv_writeback,
                capture_routes=capture_routes,
                defer_commit=defer_commit,
                clone_outputs=clone_outputs,
            )
            if layer.layer_idx == layer.num_hidden_layers - 1 and not defer_final_contract:
                final_error: Exception | None = None
                try:
                    result = _finish_final_layer(layer, result, hidden_states.device)
                except Exception as exc:
                    final_error = exc
                agree_or_abort(
                    final_error,
                    self.session.config.control_device,
                    self.session.config.control_group,
                    "model final layer contraction",
                )
                if final_error is not None:
                    raise final_error
                return result
            return result

        rows = decode.rows
        error = None
        try:
            padded_post, padded_comb = self._mhc_buffer_cache.get(
                rows, torch.float32, hidden_states.device, aligned_empty
            )
            if layer.layer_idx == 0 and post is None:
                model_residual = hidden_states.unsqueeze(1).expand(-1, 4, -1).contiguous()
                model_post_out = torch.zeros_like(hidden_states)
                identity_eye = torch.eye(4, dtype=torch.float32, device=hidden_states.device)
                padded_comb[:, :16].copy_(identity_eye.reshape(1, 16).expand(rows, -1))
            else:
                model_residual = residual
                model_post_out = hidden_states
                padded_post[:, :4].copy_(post.reshape(rows, 4))
                padded_comb[:, :16].copy_(comb.reshape(rows, 16))
            inputs = {name: None for name in MC2_INPUT_NAMES}
            inputs.update(self.static_weights)
            inputs.update(
                x=model_residual,
                post_h_out=model_post_out,
                post_h_post=padded_post,
                post_h_res=padded_comb,
                conv_state=decode.conv_state,
                recurrent_state=decode.recurrent_state,
                cu_seqlens=decode.cu_seqlens,
                state_indices=decode.state_indices,
            )
            if set(inputs) != set(MC2_INPUT_NAMES):
                raise RuntimeError("mapped inputs differ from MC2 ABI")
            identity = self.identity_factory(layer, decode)
        except Exception as exc:
            error = exc
        agree_or_abort(
            error, self.session.config.control_device, self.session.config.control_group, "model route preparation"
        )
        if error is not None:
            raise error
        result = self.session.forward(inputs, self.attrs, identity, validated_step=True)
        error = None
        try:
            if decode.conv_writeback is not None:
                decode.conv_writeback.copy_(decode.conv_state)
                torch.npu.current_stream(hidden_states.device).synchronize()
        except Exception as exc:
            error = exc
        agree_or_abort(
            error,
            self.session.config.control_device,
            self.session.config.control_group,
            "model convolution state writeback",
        )
        if error is not None:
            raise error
        if layer.layer_idx == layer.num_hidden_layers - 1:
            final_error = None
            try:
                result = _finish_final_layer(layer, result, hidden_states.device)
            except Exception as exc:
                final_error = exc
            agree_or_abort(
                final_error,
                self.session.config.control_device,
                self.session.config.control_group,
                "model final layer contraction",
            )
            if final_error is not None:
                raise final_error
            return result
        return result
