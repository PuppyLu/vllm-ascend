# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental GLM5Next MegaKernel bootstrap for vLLM-Ascend ebaaeeec5.

The V2 runner installs this opt-in after weights load on every rank. A supplied
independent manifest or an explicitly accepted loaded-weight observation is
required. The latter checks self-consistency, not checkpoint provenance. The
mapper and bridge are correctness-first prototypes, not an A3 validation.

Only eager, PP=1, TP and EP equal the entire world in identical rank order are
accepted here. Those restrictions are for this bootstrap, not the operator.
Do not infer support of graph replay, separate topology or end-to-end speedup.
A per-process teardown or rank failure after submission requires job abort.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from mc2_integration.stage24_admission import NATIVE_SHARED_PROTOCOL, IndependentIdentity, ReviewedIndependentAbi
from mc2_integration.torch_acl_bridge import _check_tensor
from mc2_integration.vllm_mc2_bridge import Mc2SessionConfig

from vllm_ascend.models.glm5next.mega_kernel_abi import ATTR_NAMES, aligned_copy
from vllm_ascend.models.glm5next.mega_kernel_adapter import (
    DecodeLayerAdapter,
    MegaKernelModelBackend,
    benchmark_consumer_ready,
    coordinate_layer_preflight,
    validate_model_path_config,
)
from vllm_ascend.models.glm5next.mega_kernel_collectives import agree_decode_step
from vllm_ascend.models.glm5next.mega_kernel_collectives import agree_or_abort as _agree
from vllm_ascend.models.glm5next.mega_kernel_scheduler import V2SchedulerEvidence
from vllm_ascend.models.glm5next.mega_kernel_torch_session import create_torch_session
from vllm_ascend.models.glm5next.mega_kernel_weights import _reuse_or_align_weights, map_static_weights

_gate = None


def install_runner_hooks(runner_cls):
    """Install phase tracking once per worker before model execution."""
    global _gate
    if _gate is not None:
        raise RuntimeError("MC2 runner hook already installed")
    gate = V2SchedulerEvidence(runner_cls)
    gate.install()
    _gate = gate
    return gate


def uninstall_runner_hooks():
    """Remove the phase hook after all trace wrappers and model steps stop."""
    global _gate
    if _gate is None:
        return
    _gate.uninstall()
    _gate = None


def _all_true(local, device, group):
    flag = torch.tensor([int(bool(local))], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _layer_list(model):
    from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer

    found = [
        m
        for _, m in model.named_modules()
        if isinstance(m, Glm5NextDecoderLayer)
        and getattr(m, "layer_kind", None) == "kda"
        and getattr(m, "_mlp_is_moe", False)
        and getattr(m, "mhc", False)
        and not getattr(m, "is_mtp_layer", True)
    ]
    by_id = {m.layer_idx: m for m in found}
    if not by_id or len(by_id) != len(found):
        raise ValueError("missing or duplicate KDA+MoE target layers")
    return [by_id[i] for i in sorted(by_id)]


def _check_attrs(layer, attrs, stage, world):
    if not isinstance(attrs, dict) or set(attrs) != set(ATTR_NAMES):
        raise ValueError("manifest must contain all 18 actual operator attributes")
    expected = {
        "hc_mult": layer.n,
        "hc_sinkhorn_iters": layer.mhc_sinkhorn_iterations,
        "hc_eps": layer.hc_eps,
        "norm_eps": layer.rms_norm_eps,
        "layer_rms_eps": layer.input_layernorm.variance_epsilon,
        "stage_limit": stage,
        "conv_slot": 0,
        "lower_bound": layer.self_attn.kda_lower_bound,
        "ffn_hc_eps": layer.hc_eps,
        "ffn_norm_eps": layer.rms_norm_eps,
        "ffn_layer_rms_eps": layer.post_attention_layernorm.variance_epsilon,
        "mhc_post_mult_value": layer.mhc_post_mult_value,
        "router_mode": 1,
        "expert_id": -1,
        "route_ordinal": 0,
        "down_expert_id": -1,
        "ep_world_size": world,
    }
    for key, value in expected.items():
        if attrs[key] != value:
            raise ValueError(f"model and reviewed manifest disagree on {key}")
    # output_norm_eps must independently match the actual FLA layer; accepted
    # property names depend on deployed vLLM version, so reject absent values.
    eps = getattr(layer.self_attn.o_norm, "eps", None)
    if eps is None or attrs["output_norm_eps"] != eps:
        raise ValueError("output norm epsilon missing or disagrees with model")


def bootstrap(runner: Any, config_path: str):
    """Install all supported target layers; no hidden per-layer factory needed.

    JSON config: `manifest_path`, `opapi_path`, `generated_header_path`,
    `reviewed_generated_header: true`, `expert_postprocessed: true`,
    `tp_algorithm`, `ep_algorithm`, and optional `allow_nz: false`. The model
    uses the internal router and a single SPMD-trusted execution path without
    the output D2H workaround; actual model accuracy requires A3 validation.
    Manifest schema `megakernel.model_manifest.v1`: `model_digest`,
    `generated_header_sha256`, `max_rows_per_source`, `max_total_rows`,
    `layers` keyed by layer_idx. Every layer contains `attrs`, ordered
    `expert_shard_digests`, `head_shard_digests`, `shared_shard_digests`, and
    `model_shared_intermediate`. A manifest may be independently supplied, or
    generated from loaded tensors with an explicit self-consistency opt-in.
    The latter is not independent checkpoint evidence. Config and any supplied
    manifest must be identical across ranks. No weight refit is allowed.
    """
    from vllm.distributed.parallel_state import get_ep_group, get_pp_group, get_tp_group

    if _gate is None or not dist.is_initialized():
        raise RuntimeError("install runner hooks and initialize distributed first")
    world, rank = dist.get_world_size(), dist.get_rank()
    members = list(range(world))
    device = torch.device("npu", torch.npu.current_device())
    local_error = None
    try:
        cfg_raw = Path(config_path).read_bytes()
        cfg = json.loads(cfg_raw)
        validate_model_path_config(cfg)
        if cfg.get("reviewed_generated_header") is not True or cfg.get("expert_postprocessed") is not True:
            raise ValueError("explicit ABI review and expert post-load attestation required")
        if cfg.get("weights_and_groups_static_for_process") is not True:
            raise ValueError("repeatable MC2 requires explicitly disabling weight refit and group replacement")
        if getattr(runner.vllm_config.model_config, "enforce_eager", None) is not True:
            raise ValueError("experimental Decode layer kernel currently requires --enforce-eager")
        if (
            get_pp_group().world_size != 1
            or dist.get_process_group_ranks(get_tp_group().device_group) != members
            or dist.get_process_group_ranks(get_ep_group().device_group) != members
        ):
            raise ValueError("requires PP1 and full-world ordered TP=EP in this bootstrap")
        header_hash = hashlib.sha256(Path(cfg["generated_header_path"]).read_bytes()).hexdigest()
        manifest_path = cfg.get("manifest_path")
        if manifest_path is not None:
            manifest_raw = Path(manifest_path).read_bytes()
            manifest = json.loads(manifest_raw)
        else:
            if cfg.get("accept_loaded_weight_self_consistency") is not True:
                raise ValueError("supply a reviewed manifest or explicitly accept a loaded-weight observation")
            manifest_raw = b""
            manifest = None
        layers = _layer_list(runner.model)
        ids = [layer.layer_idx for layer in layers]
        bench_cfg = cfg.get("single_layer_benchmark")
        if bench_cfg is not None:
            if not isinstance(bench_cfg, dict):
                raise ValueError("single_layer_benchmark must be an object")
            if (
                type(bench_cfg.get("layer")) is not int
                or bench_cfg["layer"] not in ids
                or type(bench_cfg.get("rows")) is not int
                or bench_cfg["rows"] <= 0
                or type(bench_cfg.get("warmup")) is not int
                or bench_cfg["warmup"] < 0
                or type(bench_cfg.get("samples")) is not int
                or bench_cfg["samples"] < 1
                or not isinstance(bench_cfg.get("output"), str)
                or not Path(bench_cfg["output"]).is_absolute()
                or bench_cfg.get("event_stream_join_attested") is not True
            ):
                raise ValueError("invalid single-layer benchmark config or missing event-stream attestation")
    except Exception as exc:
        local_error = exc
    # This is the existing world group, before any new HCCL group is created.
    _agree(local_error, device, dist.group.WORLD, "MC2 bootstrap local setup")
    if local_error is not None:
        raise local_error
    control = dist.new_group(ranks=members, backend="hccl")
    tp = dist.new_group(ranks=members, backend="hccl")
    ep = dist.new_group(ranks=members, backend="hccl")
    hashes = [None] * world
    digest = hashlib.sha256(cfg_raw + b"\0" + manifest_raw).hexdigest()
    dist.all_gather_object(hashes, digest, group=control)
    if len(set(hashes)) != 1:
        raise RuntimeError("bootstrap config/manifest differs across ranks")
    all_ids = [None] * world
    dist.all_gather_object(all_ids, ids, group=control)
    if any(x != ids for x in all_ids):
        raise RuntimeError("target layer lists disagree across ranks")
    if manifest is None:
        from vllm_ascend.models.glm5next.mega_kernel_manifest import build_loaded_manifest

        manifest = build_loaded_manifest(
            layers,
            control_group=control,
            tp_group=tp,
            ep_group=ep,
            generated_header_path=cfg["generated_header_path"],
            max_rows_per_source=cfg["max_rows_per_source"],
            max_total_rows=cfg["max_total_rows"],
            allow_nz=cfg.get("allow_nz") is True,
            expert_postprocessed=True,
        )
    local_error = None
    try:
        if (
            manifest.get("schema") != "megakernel.model_manifest.v1"
            or manifest.get("protocol") != NATIVE_SHARED_PROTOCOL
        ):
            raise ValueError("unsupported checkpoint manifest schema or protocol")
        if header_hash != manifest["generated_header_sha256"]:
            raise ValueError("generated header differs from reviewed manifest")
        if set(map(str, ids)) != set(manifest["layers"]):
            raise ValueError("manifest must cover exactly every target layer")
        abi = ReviewedIndependentAbi(
            manifest["generated_header_sha256"],
            NATIVE_SHARED_PROTOCOL,
            manifest["max_rows_per_source"],
            manifest["max_total_rows"],
            True,
            True,
            50,
        )
        abi.validate()
    except Exception as exc:
        local_error = exc
    _agree(local_error, device, control, "MC2 manifest validation")
    if local_error is not None:
        raise local_error
    # The group agrees both phase and the exact ordered request/position digest
    # before the reference prefix can enter its TP collective. The sequence
    # advances only after this all-rank agreement, so it is not a local guess.
    sequence = [0]

    def agree_step(accepted, step_digest, local_error):
        same = agree_decode_step(accepted, step_digest, device, control, local_error=local_error)
        if same:
            if sequence[0] >= 2**31 - 1:
                raise RuntimeError("MC2 session step counter exhausted; restart the job")
            sequence[0] += 1
        return same

    _gate.set_step_agreement(agree_step)
    sessions, adapters = [], []
    for layer in layers:
        item = manifest["layers"][str(layer.layer_idx)]
        local_error = None
        try:
            heads = int(layer.self_attn.local_num_heads)
            model_heads = int(layer.self_attn.num_heads)
            head_rank = int(layer.self_attn.tp_rank)
            local_range = (head_rank * heads, (head_rank + 1) * heads)
            weights = map_static_weights(
                layer, ep_rank=rank, ep_world=world, allow_nz=cfg.get("allow_nz") is True, expert_postprocessed=True
            )
            weights = _reuse_or_align_weights(weights, device, _check_tensor, aligned_copy)
            _check_attrs(layer, item["attrs"], 28, world)
            sc = Mc2SessionConfig(
                cfg["opapi_path"],
                cfg["generated_header_path"],
                abi,
                heads,
                model_heads,
                local_range,
                control,
                tp,
                ep,
                device,
                cfg["tp_algorithm"],
                cfg["ep_algorithm"],
                tuple(item["expert_shard_digests"]),
                head_shard_digests=tuple(item["head_shard_digests"]),
                shared_shard_digests=tuple(item["shared_shard_digests"]),
                model_shared_intermediate=item["model_shared_intermediate"],
                padding_supported=True,
                shared_tp_sharded=True,
                native_shared=True,
                consumer_readback=False,
            )
            # A process-lifetime epoch is valid only under the explicit no-refit
            # and no-group-replacement contract checked during startup.
            epoch = manifest["model_digest"] + ":" + str(layer.layer_idx)
            session = create_torch_session(sc, lambda epoch=epoch: epoch)
        except Exception as exc:
            local_error = exc
        if not _all_true(local_error is None, device, control):
            raise RuntimeError(f"group-wide MC2 model bootstrap failed; local error: {local_error}")

        def identity_factory(actual_layer, decode, *, model_digest=manifest["model_digest"]):
            digest = _gate.current_step_digest()
            if sequence[0] < 1:
                raise RuntimeError("MC2 logical step has not been agreed across ranks")
            return IndependentIdentity(actual_layer.layer_idx, sequence[0], model_digest, digest)

        backend = MegaKernelModelBackend(session, weights, item["attrs"], identity_factory)

        def coordinate(actual_layer, error):
            del actual_layer
            # Unexpected exceptions abort every rank; known unsupported input
            # permits a group-wide fallback before any MC2 collective.
            return coordinate_layer_preflight(error, _gate.is_pure_decode(), device, control)

        def execution_mode():
            return _gate.execution_mode()

        def decode_evidence(actual_layer):
            del actual_layer
            return _gate.is_pure_decode()

        benchmark = None
        if bench_cfg is not None and layer.layer_idx == bench_cfg["layer"]:

            def benchmark(actual_layer, positions, hidden_states, residual, post, comb, decode):
                if decode.rows != bench_cfg["rows"]:
                    return False
                from vllm_ascend.models.glm5next.mega_kernel_layer_bench import measure

                def consumer_ready(mode, outputs):
                    del mode
                    benchmark_consumer_ready(outputs)

                report = measure(
                    actual_layer,
                    positions,
                    hidden_states,
                    residual,
                    post,
                    comb,
                    control_group=control,
                    control_device=device,
                    consumer_ready=consumer_ready,
                    events_join_required_work=True,
                    logical_step_agreed=True,
                    expected_rows=bench_cfg["rows"],
                    warmup=bench_cfg["warmup"],
                    samples=bench_cfg["samples"],
                )
                report["consumer_timing_policy"] = "device_synchronize_only"
                write_error = None
                if rank == 0:
                    try:
                        with Path(bench_cfg["output"]).open("x", encoding="utf-8") as f:
                            json.dump(report, f, ensure_ascii=False, indent=2)
                            f.write("\n")
                    except Exception as exc:
                        write_error = exc
                _agree(write_error, device, control, "single-layer benchmark report")
                if write_error is not None:
                    raise write_error
                return True

        sessions.append(session)
        adapters.append((layer, DecodeLayerAdapter(backend, coordinate, execution_mode, decode_evidence, benchmark)))
    # All local preparation and group-level agreement succeeded. Do not leave
    # partially installed layers if validation fails before this point.
    for layer, adapter in adapters:
        adapter.install(layer)
    runner._megakernel_sessions = sessions
    return ids
