# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental GLM5Next MegaKernel startup bootstrap and cold admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.models.glm5next.mega_kernel_abi import (
    ATTR_NAMES,
    HEAD_FIELDS,
    MODEL_FIELDS,
    NATIVE_SHARED_PROTOCOL,
    SHARED_FIELDS,
    ReviewedAbi,
    StepIdentity,
    aligned_copy,
    check_abi_tensor,
    expert_bank_digest,
    tensor_sha256,
    validate_expert_bank,
    validate_projection_shapes,
)
from vllm_ascend.models.glm5next.mega_kernel_adapter import (
    DecodeLayerAdapter,
    MegaKernelModelBackend,
    StepDecodeCoordinator,
    benchmark_consumer_ready,
    coordinate_layer_preflight,
    validate_model_path_config,
)
from vllm_ascend.models.glm5next.mega_kernel_collectives import (
    HcclDomainInspector,
    admission_control,
    agree_decode_step_contract,
    agree_or_abort,
    digest_agreement,
)
from vllm_ascend.models.glm5next.mega_kernel_scheduler import V2SchedulerEvidence
from vllm_ascend.models.glm5next.mega_kernel_torch_session import (
    Mc2SessionConfig,
    SharedStepBufferPool,
    create_torch_session,
    load_torch_mc2_extension,
)
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
    eps = getattr(layer.self_attn.o_norm, "eps", None)
    if eps is None or attrs["output_norm_eps"] != eps:
        raise ValueError("output norm epsilon missing or disagrees with model")


def _opt_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    val = cfg.get(key, default)
    if type(val) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return val


def _opt_pos_int(cfg: dict[str, Any], key: str, default: int) -> int:
    val = cfg.get(key, default)
    if type(val) is not int or val < 1:
        raise ValueError(f"{key} must be a positive integer")
    return val


def _verify_layer_weights_startup(
    weights: dict[str, Any],
    manifest_item: dict[str, Any],
    *,
    heads: int,
    tp_rank: int,
    ep_rank: int,
    world: int,
    control: Any,
    device: Any,
) -> None:
    validate_projection_shapes(weights, heads)
    shared_width = int(manifest_item["model_shared_intermediate"])
    local_shared_width = int(weights["expert_down_weight"].shape[1])
    if shared_width <= 0 or shared_width > 2048 * world:
        raise ValueError("unsupported model_shared_intermediate in manifest")
    validate_expert_bank(weights, ep_rank, world, "sharded", shared_width=local_shared_width)
    if expert_bank_digest(weights) != manifest_item["expert_shard_digests"][ep_rank]:
        raise ValueError("local expert shard disagrees with reviewed manifest")
    head_fields = [(name, weights[name]) for name in MODEL_FIELDS if name in HEAD_FIELDS and weights.get(name) is not None]
    if tensor_sha256(head_fields).hex() != manifest_item["head_shard_digests"][tp_rank]:
        raise ValueError("local head weights differ from reviewed shard manifest")
    shared_fields = [(name, weights[name]) for name in MODEL_FIELDS if name in SHARED_FIELDS and weights.get(name) is not None]
    if tensor_sha256(shared_fields).hex() != manifest_item["shared_shard_digests"][tp_rank]:
        raise ValueError("local shared-expert weights differ from reviewed shard manifest")
    rep_fields = [
        (name, weights[name])
        for name in MODEL_FIELDS
        if name not in HEAD_FIELDS and name not in SHARED_FIELDS and weights.get(name) is not None
    ]
    rep_hash = tensor_sha256(rep_fields).hex()
    digest_agreement(rep_hash.encode("ascii"), device, control, "startup replicated weight contract")
    expected_rep = manifest_item.get("observed_replicated_digest")
    if isinstance(expected_rep, str) and rep_hash != expected_rep:
        raise ValueError("replicated weights differ from manifest digest")
    fa_ga_fields = [("f_g_a", weights["projection_weight"][-256:])]
    if weights.get("in_projection_scale") is not None:
        fa_ga_fields.append(("f_g_a_scale", weights["in_projection_scale"][-256:]))
    digest_agreement(tensor_sha256(fa_ga_fields), device, control, "startup replicated f_a/g_a projection")


def bootstrap(runner: Any, config_path: str):
    """Install all supported target layers with startup-only cold admission and shared pools."""
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
        repeatable_executor = _opt_bool(cfg, "repeatable_executor", True)
        sync_per_layer = _opt_bool(cfg, "sync_per_layer", False)
        consumer_readback = _opt_bool(cfg, "consumer_readback", False)
        defer_state_commit = _opt_bool(cfg, "defer_state_commit", True)
        max_cached_geometries = _opt_pos_int(cfg, "max_cached_geometries", 2)
        header_path = Path(cfg["generated_header_path"])
        if not header_path.is_absolute() or not header_path.is_file():
            raise ValueError("generated_header_path must be an existing absolute file")
        header_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
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
    agree_or_abort(local_error, device, dist.group.WORLD, "MC2 bootstrap local setup")
    if local_error is not None:
        raise local_error

    control = dist.new_group(ranks=members, backend="hccl")
    tp = dist.new_group(ranks=members, backend="hccl")
    ep = dist.new_group(ranks=members, backend="hccl")
    digest_agreement(cfg_raw + b"\0" + manifest_raw, device, control, "MC2 bootstrap config and manifest")
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
        abi = ReviewedAbi(
            manifest["generated_header_sha256"],
            NATIVE_SHARED_PROTOCOL,
            manifest["max_rows_per_source"],
            manifest["max_total_rows"],
            True,
            True,
            50,
        )
        abi.validate()
        names = HcclDomainInspector().inspect((control, tp, ep), device, rank, world)
        for group, count in ((tp, world), (ep, world)):
            ping = torch.ones(16, dtype=torch.float32, device=device)
            dist.all_reduce(ping, group=group)
            if not bool(torch.all(ping == count).item()):
                raise ValueError("HCCL domain warmup mismatch")
        op, prepare_fn = load_torch_mc2_extension()
    except Exception as exc:
        local_error = exc
    agree_or_abort(local_error, device, control, "MC2 manifest and HCCL domain admission")
    if local_error is not None:
        raise local_error

    digest_agreement(json.dumps(names[0]).encode("utf-8"), device, control, "MC2 control domain name")
    digest_agreement(json.dumps(names[1]).encode("utf-8"), device, tp, "MC2 TP domain name")
    digest_agreement(json.dumps(names[2]).encode("utf-8"), device, ep, "MC2 EP domain name")

    pool = SharedStepBufferPool(device, max_cached_geometries=max_cached_geometries)
    step_contract_scratch = pool.get_step_contract_scratch()
    sequence = [0]

    def agree_step(accepted, step_digest, rows):
        same, _ = agree_decode_step_contract(
            accepted,
            rows,
            step_digest,
            abi.max_rows_per_source,
            abi.max_total_rows,
            world,
            device,
            control,
            scratch=step_contract_scratch,
        )
        if same:
            if sequence[0] >= 2**31 - 1:
                raise RuntimeError("MC2 session step counter exhausted; restart the job")
            sequence[0] += 1
        return same

    _gate.set_step_agreement(agree_step)

    sessions, layer_backends = [], []
    prop_rounds = admission_control(1, world, True)
    tp_name, ep_name = names[1], names[2]
    for ordinal, layer in enumerate(layers):
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
            weights = _reuse_or_align_weights(weights, device, check_abi_tensor, aligned_copy)
            _check_attrs(layer, item["attrs"], 28, world)
            _verify_layer_weights_startup(
                weights,
                item,
                heads=heads,
                tp_rank=head_rank,
                ep_rank=rank,
                world=world,
                control=control,
                device=device,
            )
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
                consumer_readback=consumer_readback,
                repeatable_executor=repeatable_executor,
                sync_per_layer=sync_per_layer,
                defer_state_commit=defer_state_commit,
                max_cached_geometries=max_cached_geometries,
                tp_group_name=tp_name,
                ep_group_name=ep_name,
                ep_rank=rank,
                propagation_rounds=prop_rounds,
                generated_header_sha256=header_hash,
            )
            epoch = manifest["model_digest"] + ":" + str(layer.layer_idx)
            session = create_torch_session(
                sc,
                lambda epoch=epoch: epoch,
                layer_idx=layer.layer_idx,
                layer_ordinal=ordinal,
                static_weights=weights,
                attrs=item["attrs"],
                buffer_pool=pool,
                op=op,
                prepare_fn=prepare_fn,
            )

            def identity_factory(actual_layer, decode, *, model_digest=manifest["model_digest"]):
                del decode
                digest = _gate.current_step_digest()
                if sequence[0] < 1:
                    raise RuntimeError("MC2 logical step has not been agreed across ranks")
                return StepIdentity(actual_layer.layer_idx, sequence[0], model_digest, digest)

            backend = MegaKernelModelBackend(session, weights, item["attrs"], identity_factory)
            backend.validate_static_once(layer)
        except Exception as exc:
            local_error = exc
        if not _all_true(local_error is None, device, control):
            raise RuntimeError(f"group-wide MC2 model bootstrap failed; local error: {local_error}")
        sessions.append(session)
        layer_backends.append((layer, backend))

    coordinator = StepDecodeCoordinator(
        gate=_gate,
        pool=pool,
        control_device=device,
        control_group=control,
        layer_entries=layer_backends,
        defer_state_commit=defer_state_commit,
    )
    preflight_scratch = pool.get_preflight_scratch()

    def coordinate(actual_layer, error):
        del actual_layer
        return coordinate_layer_preflight(error, _gate.is_pure_decode(), device, control, scratch=preflight_scratch)

    def execution_mode():
        return _gate.execution_mode()

    def decode_evidence(actual_layer):
        del actual_layer
        return _gate.is_pure_decode()

    adapters = []
    for layer, backend in layer_backends:
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
                agree_or_abort(
                    write_error,
                    device,
                    control,
                    "single-layer benchmark report",
                    scratch=pool.get_agree_scratch(),
                )
                if write_error is not None:
                    raise write_error
                return True

        adapters.append(
            (
                layer,
                DecodeLayerAdapter(
                    backend,
                    coordinate,
                    execution_mode,
                    decode_evidence,
                    benchmark,
                    coordinator=coordinator,
                ),
            )
        )

    for layer, adapter in adapters:
        adapter.install(layer)
    runner._megakernel_sessions = sessions
    return ids
