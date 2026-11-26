# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental vLLM-Ascend GLM5Next eager Decode dispatch.

This file supplies the model-layer dispatch and metadata/state adaptation only.
``backend`` must be a real, prevalidated MC2 binding, not a fixture. This
prototype refuses ACL graph, PCP, SP, speculation and unknown metadata.
First/last mHC handling is implemented but requires device validation. It requires an all-rank admission coordinator;
coordinator and backend failures terminate the worker and must abort the job.
No integration or NPU correctness claim is implied.
"""

from __future__ import annotations

import logging
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class UnsupportedDecode(ValueError):
    """Before a collective launch, this layer must use the original path."""


@dataclass(frozen=True)
class DecodeInputs:
    metadata: Any
    conv_state: Any
    conv_writeback: Any
    recurrent_state: Any
    state_indices: Any
    cu_seqlens: Any
    rows: int


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
) -> DecodeInputs:
    """Build MC2 state indices from actual KDA metadata in synchronous eager mode.

    Scalar device-to-host checks are deliberate correctness-first admission;
    their cost belongs in performance reporting. This implementation supports
    only unpadded rows and does not infer decode from a graph shape. To support
    graph replay or eliminate these synchronizations, the scheduler must supply
    a separately validated stable contract and device-side slot handling.
    """
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
    # Exact sequence boundaries preclude e.g. 0+2 token mixes with equal sums.
    boundaries = query.detach().cpu().tolist()
    if boundaries != list(range(rows + 1)):
        raise UnsupportedDecode("not exactly one token for each request")
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
    # MC2 requires contiguous ND. When the model stores dimension-first
    # convolution state, retain its view and publish the provisional copy back
    # after the collective succeeds; never silently update a detached copy.
    conv_writeback = None
    if not conv_state.is_contiguous():
        conv_writeback = conv_state
        conv_state = conv_state.contiguous()
    if not recurrent_state.is_contiguous():
        raise UnsupportedDecode("recurrent state layout is not MC2 contiguous")
    for name, slots, capacity in (
        ("conv", conv_slots, conv_state.shape[0]),
        ("recurrent", recurrent_slots, recurrent_state.shape[0]),
    ):
        vals = slots.detach().cpu().tolist()
        if any(v < 0 or v >= capacity for v in vals) or len(set(vals)) != rows:
            raise UnsupportedDecode(f"{name} slots contain padding, invalid or duplicate state")
    indices = torch.stack((conv_slots, recurrent_slots), dim=1).to(dtype=torch.int32).contiguous()
    # This MC2 ABI takes a single-token template, not the cumulative query offsets.
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)
    return DecodeInputs(metadata, conv_state, conv_writeback, recurrent_state, indices, cu_seqlens, rows)


class DecodeLayerAdapter:
    """Wrap one existing layer, coordinating the decision before any MC2 launch.

    ``coordinate(layer, local_error) -> bool`` must run on all members and
    return True only if every rank has an identical, supported call contract.
    It must return False to all ranks for any local error. It must raise/job
    abort on missing members rather than allow a unilateral fallback.

    ``backend(layer, positions, hidden_states, residual, post, comb, decode)``
    must execute the real full-layer MC2 ABI with matching weights and groups,
    and must return the four output values after required consumer visibility.
    MC2 updates decode.conv_state / recurrent_state exactly once. Failures after
    entering backend MUST propagate; do not fall back/retry on one rank.
    ``execution_mode()`` must identify actual eager, not just an intended mode.
    ``decode_evidence(layer)`` must use the current scheduler metadata to prove
    that every active request is in generation phase and has one token; GDN
    metadata alone can reclassify single-token prompt tails as decode.
    """

    def __init__(
        self,
        backend: Callable[..., Any],
        coordinate: Callable[..., bool],
        execution_mode: Callable[[], str],
        decode_evidence: Callable[[Any], bool],
        benchmark: Callable[..., bool] | None = None,
    ):
        if not all(callable(f) for f in (backend, coordinate, execution_mode, decode_evidence)):
            raise TypeError("backend, coordinator and execution mode are required")
        self.backend, self.coordinate, self.execution_mode = backend, coordinate, execution_mode
        self.decode_evidence = decode_evidence
        if benchmark is not None and not callable(benchmark):
            raise TypeError("benchmark must be callable")
        self.benchmark = benchmark
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

            # The runner already agreed this step across the control group.
            # Prefill/dummy paths do not issue per-layer HCCL collectives.
            if not adapter.decode_evidence(this):
                stats["nondecode"] += 1
                return original(positions, hidden_states, residual, post, comb)
            local_error = None
            inputs = None
            try:
                inputs = prepare_decode_inputs(
                    this,
                    get_forward_context(),
                    hidden_states,
                    residual,
                    post,
                    comb,
                    execution_mode=adapter.execution_mode(),
                    scheduler_pure_decode=adapter.decode_evidence(this),
                )
            except Exception as exc:
                local_error = exc
            # The backend must validate all fixed weights and group mapping
            # BEFORE this agreement. It must not start any collective here.
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
                try:
                    completed = adapter.benchmark(this, positions, hidden_states, residual, post, comb, inputs)
                    if type(completed) is not bool:
                        raise RuntimeError("single-layer benchmark must return a completion flag")
                    adapter._benchmark_done = completed
                finally:
                    adapter._benchmark_active = False
            # Match upstream kda.py KV connector ordering and attention fence.
            from vllm_ascend.models.glm5next.kda import (
                maybe_save_kv_layer_to_connector,
                record_attention_compute_start,
                wait_for_kv_layer_from_connector,
            )

            wait_for_kv_layer_from_connector(this.self_attn.prefix)
            record_attention_compute_start()
            out = adapter.backend(this, positions, hidden_states, residual, post, comb, inputs)
            if not isinstance(out, tuple) or len(out) != 4:
                raise RuntimeError("MC2 backend did not return the layer's four values")
            maybe_save_kv_layer_to_connector(this.self_attn.prefix, [])
            stats["hits"] += 1
            if stats["hits"] == 1:
                logger.info("MegaKernel layer %s first successful Decode call", this.layer_idx)
            return out

        def forward(
            this: Any, positions: Any, hidden_states: Any, residual: Any = None, post: Any = None, comb: Any = None
        ):
            if not execution_lock.acquire(blocking=False):
                # A local wait can desynchronize collective order across ranks.
                # Abort the worker instead of queueing a concurrent model call.
                raise RuntimeError("concurrent MegaKernel layer calls are unsupported; abort the distributed job")
            try:
                return forward_impl(this, positions, hidden_states, residual, post, comb)
            finally:
                execution_lock.release()

        layer._megakernel_original_forward = original
        # The diagnostic replay runs synchronously inside the already-held
        # layer lock; only that private path may call the implementation.
        layer._megakernel_benchmark_forward = types.MethodType(forward_impl, layer)
        layer.forward = types.MethodType(forward, layer)


def compute_model_routes(
    layer: Any, positions: Any, hidden_states: Any, residual: Any, post: Any, comb: Any, *, temporary_cache: Any = None
) -> tuple[Any, Any]:
    """Compute the real vLLM router on a temporary copy of the input KDA states.

    This eager-only path repeats attention and is intentionally slow. It avoids
    routing from the wrong hidden state; its cost belongs in fused-path timing.
    KV transfer connectors are not supported for this provisional execution.
    """
    import torch
    from vllm.distributed.kv_transfer import has_kv_transfer_group

    if has_kv_transfer_group():
        raise UnsupportedDecode("external-route prefix does not support a KV transfer connector")
    attn = layer.self_attn
    old_cache = attn.kv_cache
    if not isinstance(old_cache, (tuple, list)) or len(old_cache) != 2:
        raise UnsupportedDecode("missing KDA state")
    temporary = temporary_cache if temporary_cache is not None else (old_cache[0].clone(), old_cache[1].clone())
    attn.kv_cache = temporary
    try:
        # Matches the unmodified Glm5NextDecoderLayer.forward prefix. The KDA
        # custom op resolves this same attention object via static context.
        if layer.layer_idx == 0 and post is None:
            from vllm_ascend.models.glm5next.ops.mhc_ops import hc_expand

            x_res = hc_expand(hidden_states, layer.n)
            x_post, x_comb, x = layer.hc_pre(
                x_res,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=layer.input_layernorm.weight.data,
                norm_eps=layer.input_layernorm.variance_epsilon,
            )
        else:
            x_res, x_post, x_comb, x = layer.hc_post_pre(
                hidden_states,
                residual,
                post,
                comb,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=layer.input_layernorm.weight.data,
                norm_eps=layer.input_layernorm.variance_epsilon,
            )
        x = attn(hidden_states=x, positions=positions)
        _, _, _, ffn_input = layer.hc_post_pre(
            x,
            x_res,
            x_post,
            x_comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            norm_weight=layer.post_attention_layernorm.weight.data,
            norm_eps=layer.post_attention_layernorm.variance_epsilon,
        )
        # Use the real Ascend runner gate and router to preserve its precision,
        # top-k order, score correction, normalization and scaling.
        runner = layer.mlp.experts
        routed = runner.routed_experts
        if not callable(getattr(runner, "_compute_router_logits", None)):
            raise UnsupportedDecode("vLLM runner does not expose the verified gate path")
        if not callable(getattr(routed, "_select_experts", None)):
            raise UnsupportedDecode("vLLM runner does not expose its router")
        if runner.is_internal_router:
            logits = runner._compute_router_logits(ffn_input, ffn_input)
        else:
            logits, _ = layer.mlp.gate(ffn_input)
        weights, ids = routed._select_experts(ffn_input, logits, enable_force_load_balance=False)
        if ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("router returned incompatible ids/weights")
        return ids.to(dtype=torch.int32).contiguous(), weights.to(dtype=torch.float32).contiguous()
    finally:
        attn.kv_cache = old_cache


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


class MegaKernelModelBackend:
    """Assemble real model tensors and invoke a group-bound MC2 session.

    Static weights and attrs must come from the checked checkpoint mapper and
    reviewed generated ABI. The session is supplied by the operator repository.
    The identity factory derives the current layer/step and trusted checkpoint
    identity; no locally invented cross-rank step counter is used here.
    """

    def __init__(
        self, session: Any, static_weights: dict[str, Any], attrs: dict[str, Any], identity_factory: Callable[..., Any]
    ):
        if not callable(getattr(session, "forward", None)) or not callable(identity_factory):
            raise TypeError("MC2 session and model identity factory are required")
        self.session, self.static_weights, self.attrs = session, dict(static_weights), dict(attrs)
        self.identity_factory = identity_factory

    def validate(
        self, layer: Any, decode: DecodeInputs, hidden_states: Any, residual: Any, post: Any, comb: Any
    ) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        rows = decode.rows
        if rows < 1 or not self.static_weights:
            raise UnsupportedDecode("MC2 static weights or active rows are unavailable")
        if self.attrs.get("stage_limit") != self.session.bridge.stage_limit:
            raise UnsupportedDecode("MC2 stage limit differs from the configured bridge")
        if tuple(hidden_states.shape) != (rows, 4096):
            raise UnsupportedDecode("hidden state does not match the MC2 ABI")
        first = layer.layer_idx == 0 and post is None
        if not first:
            if tuple(residual.shape) != (rows, 4, 4096):
                raise UnsupportedDecode("mHC residual does not match the MC2 ABI")
            if tuple(post.shape) != (rows, 4, 1) or tuple(comb.shape) != (rows, 4, 4):
                raise UnsupportedDecode("incoming mHC mixing tensors do not match the model contract")
        if has_kv_transfer_group():
            raise UnsupportedDecode("external-route prefix does not support a KV transfer connector")
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
            raise UnsupportedDecode("vLLM routed scale is not fully included in external routing weights")
        routed_cfg = getattr(routed, "moe_config", None)
        if (
            getattr(layer.mlp.shared_experts, "swiglu_limit", None) != 10.0
            or getattr(routed_cfg, "swiglu_limit", None) != 10.0
        ):
            raise UnsupportedDecode("model expert activation clamp differs from MC2 limit 10")

    def __call__(
        self, layer: Any, positions: Any, hidden_states: Any, residual: Any, post: Any, comb: Any, decode: DecodeInputs
    ) -> tuple[Any, Any, Any, Any]:
        import torch
        from cross_rank_prototype.staged_collectives import _agree
        from mc2_integration.projection_contract import MC2_INPUT_NAMES
        from mc2_integration.torch_acl_bridge import _check_tensor, aligned_copy, aligned_empty

        rows = decode.rows
        error = None
        try:
            # Allocate everything that can fail locally before the reference
            # prefix enters the real TP collective.
            temporary_cache = _copy_live_kda_slots(layer.self_attn.kv_cache, decode.state_indices)
            padded_post = aligned_empty((rows, 128), torch.float32, hidden_states.device)
            padded_comb = aligned_empty((rows, 128), torch.float32, hidden_states.device)
            padded_post.zero_()
            padded_comb.zero_()
            if layer.layer_idx == 0 and post is None:
                # HcPost(identity, zero) reproduces the first layer's hc_expand.
                model_residual = hidden_states.unsqueeze(1).expand(-1, 4, -1).contiguous()
                model_post_out = torch.zeros_like(hidden_states)
                identity = torch.eye(4, dtype=torch.float32, device=hidden_states.device)
                padded_comb[:, :16].copy_(identity.reshape(1, 16).expand(rows, -1))
            else:
                model_residual = residual
                model_post_out = hidden_states
                padded_post[:, :4].copy_(post.reshape(rows, 4))
                padded_comb[:, :16].copy_(comb.reshape(rows, 16))
            # The live vLLM allocations are not guaranteed to satisfy MC2's
            # 512-byte alignment and padded-tail ABI. Keep explicit copies and
            # write them back after a successful group execution when needed.
            staged_states = []
            for state in (decode.conv_state, decode.recurrent_state):
                try:
                    _check_tensor(state, hidden_states.device)
                    staged = state
                except ValueError:
                    staged = aligned_copy(state)
                    _check_tensor(staged, hidden_states.device)
                staged_states.append(staged)
            inputs = {name: None for name in MC2_INPUT_NAMES}
            inputs.update(self.static_weights)
            inputs.update(
                x=aligned_copy(model_residual),
                post_h_out=aligned_copy(model_post_out),
                post_h_post=padded_post,
                post_h_res=padded_comb,
                conv_state=staged_states[0],
                recurrent_state=staged_states[1],
                cu_seqlens=aligned_copy(decode.cu_seqlens),
                state_indices=aligned_copy(decode.state_indices),
            )
            if set(inputs) != set(MC2_INPUT_NAMES):
                raise RuntimeError("mapped inputs differ from MC2 ABI")
            identity = self.identity_factory(layer, decode)
        except Exception as exc:
            error = exc
        _agree(error, self.session.config.control_device, self.session.config.control_group, "model route preparation")
        if error is not None:
            raise error
        error = None
        try:
            ids, weights = compute_model_routes(
                layer, positions, hidden_states, residual, post, comb, temporary_cache=temporary_cache
            )
            inputs.update(external_route_ids=aligned_copy(ids), external_route_weights=aligned_copy(weights))
        except Exception as exc:
            # A failure inside an already-entered TP collective requires the
            # distributed job watchdog, not local rollback or retry.
            error = exc
        _agree(error, self.session.config.control_device, self.session.config.control_group, "model external routing")
        if error is not None:
            raise error
        result = self.session.forward(inputs, self.attrs, identity)
        # Every rank enters the post-commit agreement, even when only some
        # layouts require a separate convolution-state copy.
        error = None
        try:
            if staged_states[0] is not decode.conv_state:
                decode.conv_state.copy_(staged_states[0])
            if staged_states[1] is not decode.recurrent_state:
                decode.recurrent_state.copy_(staged_states[1])
            if decode.conv_writeback is not None:
                decode.conv_writeback.copy_(decode.conv_state)
            torch.npu.current_stream(hidden_states.device).synchronize()
        except Exception as exc:
            error = exc
        _agree(
            error,
            self.session.config.control_device,
            self.session.config.control_group,
            "model convolution state writeback",
        )
        if error is not None:
            raise error
        if layer.layer_idx == layer.num_hidden_layers - 1:
            from vllm_ascend.models.glm5next.ops.mhc_ops import hc_contract

            output, ffn_residual, ffn_post, ffn_comb = result
            output = hc_contract(layer.hc_post(output, ffn_residual, ffn_post, ffn_comb), layer.n)
            return output, None, None, None
        return result
