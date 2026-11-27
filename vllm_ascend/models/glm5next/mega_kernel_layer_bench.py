# SPDX-License-Identifier: Apache-2.0
"""Live GLM5Next single-layer paired replay; experimental and NOT A3-verified.

Call `measure` collectively from every TP/EP rank inside the SAME real eager,
ordinary Decode forward context, before the layer's normal invocation. The
layer must already have DecodeLayerAdapter installed and the bootstrap must
have completed. Afterwards the caller executes its selected layer path once
on the untouched real cache. This module does not load or start a model.

Every run uses independent clones of both live KDA caches and never commits
any replay result. It does a separate instrumented route-capture correctness
pair, then alternates the uninstrumented original/installed wrapper paths for
warmups and samples. Outputs and active state slots are compared after each
pair. No route monkeypatch is enabled during timed runs. Both host and NPU
intervals include the full installed wrapper on the fused side; event intervals
can include host-induced stream gaps and are not pure kernel times.

This helper assumes the caller has already agreed the same logical step and
layer across ranks. A stream/event observation cannot establish MC2's output
visibility on its own: the real production consumer readiness policy must be
provided by caller and applied equally. Enclosing job watchdog is mandatory:
a rank exception *inside* an already-entered HCCL collective can hang peers.
"""

from __future__ import annotations

import math
import statistics
import time
import types
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from vllm.forward_context import get_forward_context

from vllm_ascend.models.glm5next.mega_kernel_adapter import _copy_live_kda_slots, prepare_decode_inputs
from vllm_ascend.models.glm5next.mega_kernel_collectives import agree_or_abort as _agree


def _clone(value):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("output and state must be torch tensors or None")
    return value.detach().to("cpu", copy=True).clone()


def _assert_pair(original, fused, *, rtol=0.03, atol=0.005):
    if len(original) != 4 or len(fused) != 4:
        raise ValueError("original and fused must both return the actual four layer values")
    for name, ref, actual in zip(("hidden", "residual", "post", "comb"), original, fused):
        if ref is None or actual is None:
            if ref is not actual:
                raise AssertionError(f"{name}: None mismatch")
            continue
        if ref.numel() == 0 or actual.numel() == 0:
            raise AssertionError(f"{name}: empty tensor")
        if torch.is_floating_point(ref) and (
            not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(actual).all())
        ):
            raise AssertionError(f"{name}: non-finite values")
        try:
            torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc


def _assert_state(ref, actual, *, rtol=0.03, atol=0.005):
    for name, r, a in zip(("conv_state", "recurrent_state"), ref, actual):
        if r.numel() == 0 or a.numel() == 0:
            raise AssertionError(f"{name}: empty active-state snapshot")
        if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{name}: non-finite values")
        try:
            torch.testing.assert_close(a, r, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc


def _original_route_indices(rows, captured_rows, rank, world):
    """Map the observed original router rows to source rows, or reject.

    MC2/All2All prepare splits contiguous token rows across TP before calling
    the router. Only the verified, unpadded layout is handled here; never infer
    a mapping from matching tensor sizes alone.
    """
    if captured_rows == rows:
        return torch.arange(rows, dtype=torch.int64)
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    comm = getattr(_EXTRA_CTX, "moe_comm_method", None)
    prepare = getattr(comm, "prepare_finalize", None)
    cls = type(prepare)
    if (
        cls.__module__ != "vllm_ascend.ops.fused_moe.prepare_finalize"
        or cls.__name__ not in ("PrepareAndFinalizeWithMC2", "PrepareAndFinalizeWithAll2All")
        or getattr(prepare, "tp_size", None) != world
        or getattr(prepare, "tp_rank", None) != rank
        or getattr(prepare, "replace_allreduce", None) is not False
    ):
        raise ValueError("cannot establish the original router token partition")
    if rows < world:
        raise ValueError("padded original router partitions are not supported by this comparison")
    if cls.__name__ == "PrepareAndFinalizeWithMC2":
        if getattr(_EXTRA_CTX, "padded_num_tokens", None) != rows:
            raise ValueError("padded MC2 router input cannot be mapped without its complete mask")
        mask = getattr(_EXTRA_CTX, "mc2_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.numel() != rows:
            raise ValueError("MC2 router validity mask is missing")
        if not bool(torch.all(mask).item()):
            raise ValueError("MC2 router contains invalid rows")
    indices = torch.tensor_split(torch.arange(rows, dtype=torch.int64), world)[rank]
    if indices.numel() != captured_rows:
        raise ValueError("original router partition size does not match TP token splitting")
    return indices


def _assert_routes(ref, actual, rows, original_indices):
    if len(ref) != 1 or len(actual) != 1:
        raise ValueError("original and fused must each perform exactly one observable router selection")
    rw, ri = ref[0]
    aw, ai = actual[0]
    if ai.shape != (rows, 8) or aw.shape != ai.shape or ri.ndim != 2 or ri.shape[1] != 8 or rw.shape != ri.shape:
        raise ValueError("cannot prove the original/fused router rows match this local batch")
    if original_indices.ndim != 1 or original_indices.numel() != ri.shape[0]:
        raise ValueError("original router row mapping is invalid")
    if ri.dtype not in (torch.int32, torch.int64) or ai.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert IDs must be integer tensors")
    torch.testing.assert_close(ai[original_indices].to(torch.int64), ri.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(aw[original_indices].float(), rw.float(), rtol=0.03, atol=0.005)


def _summary(records, world, samples):
    if len(records) != world or {r["rank"] for r in records} != set(range(world)):
        raise ValueError("incomplete or duplicate rank records")
    result = {}
    for path in ("original", "fused"):
        result[path] = {}
        for kind in ("event_ms", "consumer_ready_ms"):
            if any(len(r[path][kind]) != samples for r in records):
                raise ValueError("missing rank measurements")
            maximum = [max(r[path][kind][i] for r in records) for i in range(samples)]
            if not all(math.isfinite(v) and v > 0 for v in maximum):
                raise ValueError("invalid duration")
            result[path][kind] = {
                "p50": statistics.median(maximum),
                "p95": sorted(maximum)[math.ceil(samples * 0.95) - 1],
            }
    for kind in ("event_ms", "consumer_ready_ms"):
        result["speedup_" + kind] = result["original"][kind]["p50"] / result["fused"][kind]["p50"]
    return result


def measure(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    post: torch.Tensor | None,
    comb: torch.Tensor | None,
    *,
    control_group: Any,
    control_device: Any,
    consumer_ready: Callable[[str, Any], None],
    events_join_required_work: bool,
    logical_step_agreed: bool,
    expected_rows: int = 64,
    warmup: int = 3,
    samples: int = 50,
) -> dict[str, Any]:
    """Run a single captured layer without mutating the real live KDA states.

    `consumer_ready(mode, four_outputs)` must apply the *actual same* read-side
    policy for both paths and return only when the next layer may consume all
    four outputs. It must not modify model state. Caller must confirm event
    stream dependencies for BOTH paths before `events_join_required_work=True`.
    `logical_step_agreed=True` attests that every group member is measuring the
    same layer/request step. The measured end marker is enqueued after the
    consumer callback, then the entire NPU is synchronized.

    This function never writes report files or advances the real model. Execute
    the selected path exactly once after returning, with the restored live cache.
    It does not account for the layer's invocation by the scheduler itself.
    """
    local_error = None
    snapshot = None
    original_cache = None
    route_owner = None
    original_stats = None
    world = dist.get_world_size(group=control_group)
    rank = dist.get_rank(group=control_group)
    try:
        if events_join_required_work is not True or logical_step_agreed is not True:
            raise ValueError("cross-stream completion and logical-step agreement must be proven")
        if not callable(consumer_ready) or type(expected_rows) is not int or expected_rows <= 0:
            raise ValueError("a real consumer policy and positive row count are required")
        if type(warmup) is not int or warmup < 0 or type(samples) is not int or samples < 1:
            raise ValueError("invalid warmup/sample count")
        if int(hidden_states.shape[0]) != expected_rows:
            raise ValueError("wrong valid decode rows")
        if torch.compiler.is_compiling() or torch.npu.is_current_stream_capturing():
            raise ValueError("eager execution only; no graph capture or replay")
        if not callable(getattr(layer, "_megakernel_original_forward", None)) or not isinstance(
            getattr(layer, "_megakernel_stats", None), dict
        ):
            raise ValueError("the real layer has no installed MegaKernel model adapter")
        original_stats = dict(layer._megakernel_stats)
        original_cache = layer.self_attn.kv_cache
        route_owner = layer.mlp.experts.routed_experts
        if not callable(getattr(route_owner, "_select_experts", None)):
            raise ValueError("actual vLLM expert-selection hook unavailable")
        if getattr(route_owner, "enable_npugraph_ex_static_kernel", False):
            raise ValueError("static graph MoE layer-index side effects are unsupported")
        router = getattr(route_owner, "router", None)
        if router is not None and getattr(router, "eplb_state", None) is not None:
            raise ValueError("EPLB load accounting would be changed by replay")
        from vllm_ascend.ascend_config import get_ascend_config

        if getattr(get_ascend_config(), "enable_force_eplb", False):
            raise ValueError("forced expert load balancing makes replay unsafe")
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        if has_kv_transfer_group():
            raise ValueError("KV transfer connector state is not supported in replay")
        if not isinstance(original_cache, (tuple, list)) or len(original_cache) != 2:
            raise ValueError("model KDA cache not available")
        # Verify current metadata and obtain the exact physical active slots.
        decoded = prepare_decode_inputs(
            layer,
            get_forward_context(),
            hidden_states,
            residual,
            post,
            comb,
            execution_mode="eager",
            scheduler_pure_decode=True,
        )
        active_indices = decoded.state_indices.detach().cpu().to(torch.int64)
        snapshot = _copy_live_kda_slots(original_cache, decoded.state_indices)
        torch.npu.synchronize()
    except Exception as exc:
        local_error = exc
    _agree(local_error, control_device, control_group, "live layer benchmark preflight")
    if local_error is not None:
        raise local_error
    stats = {
        "rank": rank,
        "original": {"event_ms": [], "consumer_ready_ms": []},
        "fused": {"event_ms": [], "consumer_ready_ms": []},
    }

    def run_once(mode, *, capture, timed):
        error, work, saved_route = None, None, None
        routes = []
        try:
            work = _copy_live_kda_slots(snapshot, decoded.state_indices)
            layer.self_attn.kv_cache = work
            if capture:
                had_instance_attr = "_select_experts" in route_owner.__dict__
                prior_instance_attr = route_owner.__dict__.get("_select_experts")
                saved_route = route_owner._select_experts

                def observed(this, *a, **kw):
                    weights, ids = saved_route(*a, **kw)
                    routes.append((weights.detach().clone(), ids.detach().clone()))
                    return weights, ids

                route_owner._select_experts = types.MethodType(observed, route_owner)
            torch.npu.synchronize()
        except Exception as exc:
            error = exc
        try:
            _agree(error, control_device, control_group, f"live layer {mode} state preparation")
            if error is not None:
                raise error
            dist.barrier(group=control_group)
            error = None
            try:
                stream = torch.npu.current_stream(hidden_states.device)
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                previous_hits = layer._megakernel_stats["hits"]
                begin = time.perf_counter_ns()
                start.record(stream)
                fn = layer._megakernel_original_forward if mode == "original" else layer._megakernel_benchmark_forward
                out = fn(positions, hidden_states, residual, post, comb)
                device_values = None
                if capture:
                    if not isinstance(out, tuple) or len(out) != 4:
                        raise ValueError("model layer must return four values")
                    # Observe the output through a later device operation
                    # before any diagnostic host read. Not a substitute for
                    # checking the actual next layer and multi-step inference.
                    device_values = tuple(x.detach().clone() if isinstance(x, torch.Tensor) else x for x in out)
                consumer_ready(mode, out)
                end.record(stream)
                torch.npu.synchronize()
                host_ms = (time.perf_counter_ns() - begin) / 1e6
                event_ms = float(start.elapsed_time(end))
                hits = layer._megakernel_stats["hits"] - previous_hits
                if hits != (1 if mode == "fused" else 0):
                    raise RuntimeError(f"{mode} model call did not take the expected path: hits={hits}")
                if not isinstance(out, tuple) or len(out) != 4:
                    raise ValueError("model layer must return four values")
                values = tuple(_clone(x) for x in (device_values if device_values is not None else out))
                conv = (
                    work[0].transpose(-1, -2) if getattr(layer.self_attn, "_conv_state_dim_first", False) else work[0]
                )
                state = (
                    _clone(conv.index_select(0, active_indices[:, 0].to(conv.device))),
                    _clone(work[1].index_select(0, active_indices[:, 1].to(work[1].device))),
                )
                if capture and mode == "fused":
                    session = getattr(layer, "_megakernel_session", None)
                    weights = getattr(session, "last_route_weights", None)
                    ids = getattr(session, "last_route_ids", None)
                    if not isinstance(weights, torch.Tensor) or not isinstance(ids, torch.Tensor):
                        raise ValueError("MC2 did not expose its actual routing outputs")
                    routes = [(weights, ids)]
                routed = [(_clone(w), _clone(i)) for w, i in routes]
                original_indices = None
                if capture and mode == "original" and len(routed) == 1:
                    original_indices = _original_route_indices(expected_rows, routed[0][1].shape[0], rank, world)
            except Exception as exc:
                error = exc
            _agree(error, control_device, control_group, f"live layer {mode} execution")
            if error is not None:
                raise error
            if timed:
                stats[mode]["event_ms"].append(event_ms)
                stats[mode]["consumer_ready_ms"].append(host_ms)
            return values, state, routed, original_indices
        finally:
            if capture and saved_route is not None:
                if had_instance_attr:
                    route_owner._select_experts = prior_instance_attr
                else:
                    delattr(route_owner, "_select_experts")
            layer.self_attn.kv_cache = original_cache

    def pair(index, *, capture, timed):
        order = ("original", "fused") if index % 2 == 0 else ("fused", "original")
        results = {mode: run_once(mode, capture=capture, timed=timed) for mode in order}
        error = None
        try:
            _assert_pair(results["original"][0], results["fused"][0])
            _assert_state(results["original"][1], results["fused"][1])
            if capture:
                _assert_routes(results["original"][2], results["fused"][2], expected_rows, results["original"][3])
        except Exception as exc:
            error = exc
        _agree(error, control_device, control_group, "live layer paired output/state/route comparison")
        if error is not None:
            raise error

    try:
        pair(0, capture=True, timed=False)
        for i in range(warmup + samples):
            pair(i, capture=False, timed=i >= warmup)
        records = [None] * world
        dist.all_gather_object(records, stats, group=control_group)
        summary = _summary(records, world, samples)
        return {
            "status": "PAIRED_LIVE_LAYER_PASS",
            "layer": layer.layer_idx,
            "rows": expected_rows,
            "rank": rank,
            "world_size": world,
            "warmup": warmup,
            "samples": samples,
            "device_consumer_snapshot": True,
            "summary": summary,
            "rank_records": records,
            "measurement_note": (
                "NPU event stream interval may contain host gaps; max of rank-local times is not cross-rank "
                "global wall; consumer visibility is caller-attested; route IDs checked on separate untimed "
                "real-path pair; inputs and initial live state fixed; real cache not advanced by replay"
            ),
        }
    finally:
        # Benchmark replays must not masquerade as real service layer hits.
        layer._megakernel_stats.clear()
        layer._megakernel_stats.update(original_stats)
        layer.self_attn.kv_cache = original_cache
