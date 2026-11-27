# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a *loaded-weight observation* manifest for MC2 startup.

The digest inputs are the model's own already-loaded tensors. This is useful
for checking that they remain unchanged across admission and that ranks agree
on the observed shards; it is NOT an independently trusted checkpoint
manifest and cannot establish model authenticity or mapping correctness.
A corrupt, wrong, or consistently incorrectly mapped model can self-consistently pass.

Call on ALL ranks after vLLM post-load processing and before fusion. This
initial version requires control/TP/EP all cover the ordered default world;
no device count or decode batch size is fixed. CPU hashing copies actual model
weights and can cost substantial time and memory bandwidth. Do it once at
startup, never per step. The model must not be refitted concurrently or after
manifest creation. Any rank/device/transport failure requires job-wide abort.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.models.glm5next.mega_kernel_abi import (
    ATTR_NAMES,
    HEAD_FIELDS,
    MODEL_FIELDS,
    NATIVE_SHARED_PROTOCOL,
    SHARED_FIELDS,
    expert_bank_digest,
    tensor_sha256 as _tensor_hash,
)
from vllm_ascend.models.glm5next.mega_kernel_weights import map_static_weights


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _gather(value, group):
    size = dist.get_world_size(group)
    result = [None] * size
    dist.all_gather_object(result, value, group=group)
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a scalar numeric model attribute")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite value")
    return value


def _int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}: expected an integer model attribute")
    return value


def model_attributes(layer: Any, ep_world: int) -> dict[str, int | float]:
    """Derive supported stage-28 attrs from the actual live model configuration."""
    attn = layer.self_attn
    if type(ep_world) is not int or ep_world <= 0:
        raise ValueError("invalid EP world size")
    values = {
        "hc_mult": _int(layer.n, "hc_mult"),
        "hc_sinkhorn_iters": _int(layer.mhc_sinkhorn_iterations, "hc_sinkhorn_iters"),
        "hc_eps": _finite_float(layer.hc_eps, "hc_eps"),
        "norm_eps": _finite_float(layer.rms_norm_eps, "norm_eps"),
        "layer_rms_eps": _finite_float(layer.input_layernorm.variance_epsilon, "layer_rms_eps"),
        "stage_limit": 28,
        "conv_slot": 0,
        "lower_bound": _finite_float(attn.kda_lower_bound, "lower_bound"),
        "output_norm_eps": _finite_float(attn.o_norm.eps, "output_norm_eps"),
        "ffn_hc_eps": _finite_float(layer.hc_eps, "ffn_hc_eps"),
        "ffn_norm_eps": _finite_float(layer.rms_norm_eps, "ffn_norm_eps"),
        "ffn_layer_rms_eps": _finite_float(layer.post_attention_layernorm.variance_epsilon, "ffn_layer_rms_eps"),
        "mhc_post_mult_value": _finite_float(layer.mhc_post_mult_value, "mhc_post_mult_value"),
        "router_mode": 1,
        "expert_id": -1,
        "route_ordinal": 0,
        "down_expert_id": -1,
        "ep_world_size": ep_world,
    }
    if set(values) != set(ATTR_NAMES):
        raise ValueError("MC2 attribute ABI is not the reviewed 18-attribute contract")
    return values


def _hash_model_group(weights, selector):
    # Preserve MODEL_FIELDS order exactly as admission.py does. Filtering None
    # is part of the actual admission digest contract.
    fields = [(name, weights[name]) for name in MODEL_FIELDS if selector(name) and weights.get(name) is not None]
    if not fields:
        raise ValueError("empty model field group cannot establish shard consistency")
    return _tensor_hash(fields).hex()


def _local_layer(layer, rank, world, *, allow_nz, expert_postprocessed):
    weights = map_static_weights(
        layer, ep_rank=rank, ep_world=world, allow_nz=allow_nz, expert_postprocessed=expert_postprocessed
    )
    heads = _int(layer.self_attn.local_num_heads, "local_num_heads")
    total = _int(layer.self_attn.num_heads, "num_heads")
    tp_rank = _int(layer.self_attn.tp_rank, "tp_rank")
    if heads <= 0 or tp_rank != rank or heads * world != total:
        raise ValueError("unequal or misordered TP head partitions unsupported by current bootstrap")
    down = weights.get("expert_down_weight")
    if not isinstance(down, torch.Tensor) or down.ndim != 2 or down.shape[1] <= 0:
        raise ValueError("shared expert down projection has no local intermediate width")
    return {
        "rank": rank,
        "layer": _int(layer.layer_idx, "layer_idx"),
        "attrs": model_attributes(layer, world),
        "heads": heads,
        "model_num_heads": total,
        "shared_width": int(down.shape[1]),
        "expert": expert_bank_digest(weights),
        "head": _hash_model_group(weights, lambda n: n in HEAD_FIELDS),
        "shared": _hash_model_group(weights, lambda n: n in SHARED_FIELDS),
        "replicated": _hash_model_group(weights, lambda n: n not in HEAD_FIELDS and n not in SHARED_FIELDS),
    }


def build_loaded_manifest(
    layers: Iterable[Any],
    *,
    control_group: Any,
    tp_group: Any,
    ep_group: Any,
    generated_header_path: str,
    max_rows_per_source: int,
    max_total_rows: int,
    allow_nz: bool = False,
    expert_postprocessed: bool = False,
) -> dict[str, Any]:
    """Observe actual shards and return the same self-consistency manifest on all ranks.

    Generated header SHA is a build identity only: a human must separately
    review generated ABI types/order. `max_rows_per_source` and `max_total_rows`
    are independently reviewed build/tiling limits, not claims proved here.
    All groups must already be live. `allow_nz` explicitly permits a torch_npu
    ND conversion; `expert_postprocessed` must be attested true by the caller.
    This function never writes or overwrites any file.
    """
    if not dist.is_initialized():
        raise RuntimeError("distributed backend must be initialized")
    rank, world = dist.get_rank(), dist.get_world_size()
    members = list(range(world))
    if any(dist.get_process_group_ranks(g) != members for g in (control_group, tp_group, ep_group)):
        raise ValueError("bootstrap requires same ordered full-world control/TP/EP groups")
    local_error = None
    try:
        if not 1 <= world <= 32:
            raise ValueError("EP group exceeds the stage-28 reviewed 1..32 member limit")
        layer_list = list(layers)
        layer_ids = [_int(layer.layer_idx, "layer_idx") for layer in layer_list]
        if not layer_ids or layer_ids != sorted(set(layer_ids)):
            raise ValueError("layer list must be nonempty, sorted and unique")
        if type(max_rows_per_source) is not int or max_rows_per_source <= 0:
            raise ValueError("max_rows_per_source must be a reviewed positive limit")
        if type(max_total_rows) is not int or max_total_rows < world * max_rows_per_source:
            raise ValueError("reviewed aggregate row limit is too small")
        if type(allow_nz) is not bool or expert_postprocessed is not True:
            raise ValueError("explicitly attest expert post-load processing and allow_nz setting")
        with open(generated_header_path, "rb") as f:
            header_sha256 = hashlib.sha256(f.read()).hexdigest()
        metadata = {
            "layers": layer_ids,
            "header": header_sha256,
            "rows": max_rows_per_source,
            "total_rows": max_total_rows,
            "allow_nz": allow_nz,
            "expert_postprocessed": True,
        }
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
        metadata = None
    preflight = _gather({"rank": rank, "error": local_error, "metadata": metadata}, control_group)
    if any(x["error"] is not None for x in preflight):
        raise RuntimeError(f"group-wide manifest preflight failed: {preflight}")
    if [x["rank"] for x in preflight] != members or any(x["metadata"] != metadata for x in preflight):
        raise RuntimeError("manifest metadata, layer list or ranks disagree")

    layer_manifest = {}
    for layer in layer_list:
        local_error = None
        try:
            observed = _local_layer(layer, rank, world, allow_nz=allow_nz, expert_postprocessed=expert_postprocessed)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            observed = None
        reports = _gather({"rank": rank, "error": local_error, "observed": observed}, control_group)
        if any(x["error"] is not None for x in reports):
            raise RuntimeError(f"group-wide loaded weight mapping failed: {reports}")
        if [x["rank"] for x in reports] != members:
            raise RuntimeError("loaded-weight report rank order differs from EP/TP order")
        per_rank = [x["observed"] for x in reports]
        if any(x["layer"] != layer.layer_idx for x in per_rank):
            raise RuntimeError("loaded layer IDs disagree across ranks")
        first = per_rank[0]
        for key in ("attrs", "heads", "model_num_heads", "replicated"):
            if any(x[key] != first[key] for x in per_rank):
                raise ValueError(f"loaded model {key} differs across TP ranks")
        shared_width = sum(x["shared_width"] for x in per_rank)
        if shared_width <= 0 or shared_width > 2048 * world:
            raise ValueError("unsupported total shared intermediate size")
        layer_manifest[str(layer.layer_idx)] = {
            "attrs": first["attrs"],
            "expert_shard_digests": [x["expert"] for x in per_rank],
            "head_shard_digests": [x["head"] for x in per_rank],
            "shared_shard_digests": [x["shared"] for x in per_rank],
            "model_shared_intermediate": shared_width,
            "observed_replicated_digest": first["replicated"],
            "observed_local_shared_widths": [x["shared_width"] for x in per_rank],
        }
    basis = {
        "header_sha256": header_sha256,
        "rank_order": members,
        "layers": layer_manifest,
        "protocol": NATIVE_SHARED_PROTOCOL,
    }
    return {
        "schema": "megakernel.model_manifest.v1",
        "protocol": NATIVE_SHARED_PROTOCOL,
        "generated_header_sha256": header_sha256,
        "max_rows_per_source": max_rows_per_source,
        "max_total_rows": max_total_rows,
        "model_digest": hashlib.sha256(_canonical(basis)).hexdigest(),
        "layers": layer_manifest,
        "evidence": {
            "source": "observed_postloaded_model_tensors",
            "independent_checkpoint_manifest": False,
            "proves": "rank shard self-consistency and later change detection for these observed tensors",
            "does_not_prove": (
                "checkpoint authenticity, correct checkpoint-to-model mapping, "
                "ABI correctness, or numerical correctness"
            ),
            "weight_refit_requires_new_manifest_and_sessions": True,
        },
    }
