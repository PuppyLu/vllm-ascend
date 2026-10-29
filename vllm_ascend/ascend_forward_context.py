import math
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_config import (
    _CANN_OPS_TRANSFORMER_AVAILABLE,
    get_ascend_config,
    is_megamoe_supported_by_config,
    is_zercmoe_a3_supported_by_config,
    is_zercmoe_supported_by_config,
)
from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD,
    AscendDeviceType,
    enable_sp,
    get_ascend_device_type,
    has_layer_idx,
    is_drafter_moe_model,
    is_moe_model,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3
    ZERC_MOE = 4


_MRV2_IN_PROFILE_RUN: ContextVar[bool] = ContextVar("_MRV2_IN_PROFILE_RUN", default=False)
_MEGA_MOE_TOKENS_PER_RANK_LIMIT = 4096
_DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT = 512
_MC2_TOKENS_PER_RANK_LIMIT = 512
_A2_ZERCMOE_MIN_PREFILL_TOKENS = 64


def _is_pd_consumer(vllm_config: VllmConfig) -> bool:
    """Identify the D role independently of whether it can recompute prefill."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return False

    is_decode_bench = getattr(kv_transfer_config, "kv_connector", None) == "DecodeBenchConnector"
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    is_kv_consumer = (
        kv_role == "kv_consumer"
        if kv_role is not None
        else bool(
            getattr(kv_transfer_config, "is_kv_consumer", False)
            and not getattr(kv_transfer_config, "is_kv_producer", False)
        )
    )
    return is_decode_bench or is_kv_consumer


def _is_decode_only_node(vllm_config: VllmConfig) -> bool:
    if not _is_pd_consumer(vllm_config):
        return False

    scheduler_config = getattr(get_ascend_config(), "scheduler_config", None)
    # Actual semantics of `recompute_scheduler_enable`:
    # - Enabled: when preemption occurs on the decode node, the request is sent back
    #     to the P node to redo prefill, so the decode node only ever decodes;
    # - Disabled: prefill is executed locally on the decode node.
    return bool(getattr(scheduler_config, "recompute_scheduler_enable", False))


@contextmanager
def override_mrv2_in_profile_run(enabled: bool):
    """Override MRv2's extra profile-run marker for one forward path.

    MRv2 builds the base forward context inside upstream vLLM, so Ascend's
    platform hook cannot tell whether the current forward is the extra MC2
    profile dummy run. A ContextVar keeps this MRv2-only state scoped to the
    current forward path without adding default fallback behavior.
    """
    token = _MRV2_IN_PROFILE_RUN.set(enabled)
    try:
        yield
    finally:
        _MRV2_IN_PROFILE_RUN.reset(token)


def get_mrv2_in_profile_run() -> bool:
    return _MRV2_IN_PROFILE_RUN.get()


def use_cann_megamoe(vllm_config: VllmConfig) -> bool:
    # Disable MegaMoe pending related bug fixes
    return False
    # TODO: drop the EP-size guard when MegaMoe supports larger EP sizes.
    return (
        _CANN_OPS_TRANSFORMER_AVAILABLE
        and get_ascend_device_type() == AscendDeviceType.A3
        and get_ascend_config().enable_fused_mc2 == 1
        and is_moe_model(vllm_config)
        and vllm_config.parallel_config.enable_expert_parallel
        and 1 < get_ep_group().world_size <= 64
        and getattr(vllm_config, "lora_config", None) is None
        and is_megamoe_supported_by_config(vllm_config)
    )


def is_zercmoe_active_by_config(vllm_config: VllmConfig) -> bool:
    """Config-level ZercMoE gate without group access or backend imports.

    Safe to call from early platform hooks (cudagraph demotion) in the
    scheduler / engine-core processes. ``use_cann_zercmoe`` checks the fixed
    internal A3 W8A8 wheel or the A2 backend inside workers. A requested A3
    wheel must validate before the first forward, rather than fall back.
    """
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    if get_ascend_device_type() == AscendDeviceType.A3:
        return (
            should_prepare_zercmoe_w8a8_weights(vllm_config)
            and parallel_config.pipeline_parallel_size == 1
            and getattr(parallel_config, "prefill_context_parallel_size", 1) == 1
        )
    return (
        not _is_pd_consumer(vllm_config)
        and get_ascend_config().enable_zerc_moe == 1
        and get_ascend_config().enable_fused_mc2 == 1
        and get_ascend_device_type() == AscendDeviceType.A2
        and is_moe_model(vllm_config)
        and parallel_config.enable_expert_parallel
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.data_parallel_size == 1
        and 1 < parallel_config.tensor_parallel_size <= 64
        and model_config.dtype == torch.bfloat16
        and model_config.is_quantized is False
        and is_zercmoe_supported_by_config(vllm_config)
    )


def should_prepare_zercmoe_w8a8_weights(vllm_config: VllmConfig) -> bool:
    """Opt-in A3 weight preparation, independent of runtime eligibility.

    This does not import either backend or inspect distributed groups. P, D
    and colocated workers prepare the same weights. The caller must check the
    actual scheme: a model-level quantization flag alone does not identify W8A8.
    """
    ascend_config = get_ascend_config()
    eligible = (
        ascend_config.enable_zerc_moe == 1
        and get_ascend_device_type() == AscendDeviceType.A3
        # ModelSlim may be selected explicitly even when the HF config has no
        # quantization_config. The actual MoE scheme is validated after load.
        and (
            vllm_config.model_config.is_quantized
            or getattr(vllm_config.model_config, "quantization", None) == ASCEND_QUANTIZATION_METHOD
        )
        and vllm_config.model_config.dtype == torch.bfloat16
        and is_zercmoe_a3_supported_by_config(vllm_config)
        and is_moe_model(vllm_config)
        and vllm_config.parallel_config.enable_expert_parallel
        and getattr(vllm_config, "lora_config", None) is None
        and not ascend_config.eplb_config.dynamic_eplb
    )
    if eligible:
        from vllm_ascend.zercmoe_config import ZERCMOE_MIN_SHMEM_MB

        if getattr(ascend_config, "mix_placement", False):
            raise ValueError("ZercMoE requires independent shared experts; mix_placement is unsupported.")
        if getattr(ascend_config, "zercmoe_max_recv_token_num", None) is not None:
            raise ValueError(
                "zercmoe_max_recv_token_num is unsupported by zercmoe 1.5; remove it and use zercmoe_shmem_mb."
            )
        heap_mb = getattr(ascend_config, "zercmoe_shmem_mb", None)
        if heap_mb and heap_mb < ZERCMOE_MIN_SHMEM_MB:
            raise ValueError(
                f"A3 ZercMoE requires zercmoe_shmem_mb=0/None or at least {ZERCMOE_MIN_SHMEM_MB} MB per rank. "
                "The explicit minimum is a framework guard, not a guarantee that the heap is sufficient. "
                "Automatic sizing is determined by the wheel."
            )
    return eligible


def get_zercmoe_tokens_capacity(vllm_config: VllmConfig) -> int:
    """Size A3's shared heap from the scheduling budget, independently of MC2."""
    budget = vllm_config.scheduler_config.max_num_batched_tokens
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    if budget <= 0 or tp_size <= 0:
        raise ValueError("ZercMoE requires a positive token budget and TP size.")
    return (budget + tp_size - 1) // tp_size * tp_size


def get_zercmoe_cudagraph_mode(vllm_config: VllmConfig) -> CUDAGraphMode:
    """Keep A3 graph configuration unchanged; only its eager batches use Zerc."""
    mode = vllm_config.compilation_config.cudagraph_mode
    if get_ascend_device_type() != AscendDeviceType.A2 or vllm_config.model_config.enforce_eager:
        return mode
    if mode == CUDAGraphMode.FULL and is_zercmoe_active_by_config(vllm_config):
        return CUDAGraphMode.FULL_DECODE_ONLY
    return mode


def use_cann_zercmoe(vllm_config: VllmConfig) -> bool:
    """Worker-only backend check for the A3 1.5 W8A8 or A2 BF16 wheel."""
    if not is_zercmoe_active_by_config(vllm_config):
        return False
    from vllm_ascend.ops.fused_moe.comm_utils import a3_zercmoe_lib_available, zercmoe_lib_available

    is_a3 = get_ascend_device_type() == AscendDeviceType.A3
    world = get_ep_group().world_size
    if world <= 1 or (not is_a3 and world > 64):
        return False
    if is_a3 and vllm_config.model_config.get_num_experts() % world:
        return False
    return (
        getattr(vllm_config, "lora_config", None) is None
        and (is_a3 or not enable_sp(vllm_config))
        and not get_ascend_config().eplb_config.dynamic_eplb
        and (a3_zercmoe_lib_available() if is_a3 else zercmoe_lib_available())
    )


def validate_zercmoe_single_weight_config(vllm_config: VllmConfig) -> None:
    """Reject unsupported modes before loading a model without fallback weights."""
    ascend_config = get_ascend_config()
    if getattr(ascend_config, "zercmoe_single_weight", False) is not True:
        return
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if get_ascend_device_type() != AscendDeviceType.A3 or not is_zercmoe_active_by_config(vllm_config):
        raise ValueError("A3 ZercMoE single-weight requires an eligible A3 BF16 W8A8 MoE with EP, PP=PCP=1.")
    if not getattr(model_config, "enforce_eager", False) and (
        getattr(vllm_config.compilation_config, "cudagraph_mode", None) != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise ValueError("A3 ZercMoE single-weight requires eager or FULL_DECODE_ONLY graph mode.")
    if not getattr(model_config, "enforce_eager", False) and (
        getattr(parallel_config, "use_ubatching", False) or getattr(parallel_config, "enable_dbo", False)
    ):
        raise ValueError("A3 ZercMoE single-weight decode graph does not support microbatching.")
    if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("A3 ZercMoE single-weight does not support decode context parallelism.")
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None:
        hf_config = getattr(model_config, "hf_config", None)
        if getattr(speculative, "method", None) != "mtp" or getattr(hf_config, "model_type", None) != "glm_moe_dsa":
            raise ValueError("A3 ZercMoE single-weight currently supports speculative decoding only for GLM MTP.")
        if getattr(speculative, "draft_tensor_parallel_size", None) != getattr(
            parallel_config, "tensor_parallel_size", None
        ):
            raise ValueError("A3 ZercMoE single-weight requires MTP draft TP to match the main model TP.")
    if getattr(model_config, "enable_sleep_mode", False):
        raise ValueError("A3 ZercMoE single-weight does not support sleep mode.")
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        raise ValueError("A3 ZercMoE single-weight does not support weight transfer or reload.")
    if getattr(getattr(ascend_config, "xlite_graph_config", None), "enabled", False):
        raise ValueError("A3 ZercMoE single-weight does not support XLite.")
    if getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError("A3 ZercMoE single-weight currently requires model runner V1.")
    offload = getattr(vllm_config, "offload_config", None)
    if (
        getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0) > 0
        or getattr(offload, "offload_backend", None) == "prefetch"
    ):
        raise ValueError("A3 ZercMoE single-weight does not support CPU weight offloading.")
    if not use_cann_zercmoe(vllm_config):
        raise ValueError("A3 ZercMoE single-weight requires a usable Zerc backend and an evenly partitioned EP group.")


def _use_zercmoe_for_batch(
    num_tokens: int | None,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
    *,
    is_decode: bool,
    is_draft_model: bool,
    is_graph_mode: bool,
) -> bool:
    """A3 uses the same mathematical interface for prefill, decode and mixed batches.

    Use synchronized DP sizes for collective agreement, never a local phase
    or a token-count heuristic. Draft execution retains the original backend.
    A2 retains its separate legacy policy.
    """
    if get_ascend_device_type() == AscendDeviceType.A3:
        return (
            not is_draft_model
            and (
                not is_graph_mode
                or (
                    is_decode
                    and getattr(vllm_config.compilation_config, "cudagraph_mode", None)
                    == CUDAGraphMode.FULL_DECODE_ONLY
                )
            )
            and use_cann_zercmoe(vllm_config)
            and (num_tokens is None or 0 <= num_tokens <= get_zercmoe_tokens_capacity(vllm_config))
        )
    return (
        not is_decode
        and not is_draft_model
        and not is_graph_mode
        and not _is_pd_consumer(vllm_config)
        and (num_tokens is None or num_tokens > _A2_ZERCMOE_MIN_PREFILL_TOKENS)
        and (num_tokens is None or num_tokens <= mc2_tokens_capacity)
        and use_cann_zercmoe(vllm_config)
    )


def is_zercmoe_decode_graph(vllm_config, runtime_mode, batch_descriptor, *, is_draft_model: bool) -> bool:
    """Identify a full uniform-decode capture, including MTP verification."""
    return (
        runtime_mode == CUDAGraphMode.FULL
        and getattr(vllm_config.compilation_config, "cudagraph_mode", None) == CUDAGraphMode.FULL_DECODE_ONLY
        and bool(getattr(batch_descriptor, "uniform", False))
        and not is_draft_model
        and not getattr(vllm_config.parallel_config, "use_ubatching", False)
        and not getattr(vllm_config.parallel_config, "enable_dbo", False)
    )


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    skip_compiled: bool = False,
    max_tokens_across_pcp: int = 0,
    draft_attn_metadatas=None,
    has_sinks=False,
    input_ids=None,
    eplb_heat_collection_status: bool = False,
    request_forward: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }
    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas

        forward_context.input_ids = input_ids

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        # Use the same synchronized CPU metadata for both backend selection
        # and padding. A rank with no local tokens must choose the same backend.
        dp_meta = forward_context.dp_metadata
        if dp_meta is not None:
            max_num_tokens = int(dp_meta.num_tokens_across_dp_cpu.max().item())
        else:
            max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
        # Pass graph execution separately: eager execution does not imply
        # prefill, notably on PD consumers and MTP draft forwards.
        is_decode_graph = is_zercmoe_decode_graph(
            vllm_config, aclgraph_runtime_mode, batch_descriptor, is_draft_model=is_draft_model
        )
        moe_comm_type = select_moe_comm_method(
            max_num_tokens,
            vllm_config,
            is_draft_model=is_draft_model,
            is_graph_mode=aclgraph_runtime_mode != CUDAGraphMode.NONE,
            is_decode=is_decode_graph,
        )

        forward_context.moe_comm_type = moe_comm_type
        forward_context.zercmoe_graph_execution = is_decode_graph and moe_comm_type == MoECommType.ZERC_MOE
        if is_draft_model and getattr(get_ascend_config(), "zercmoe_single_weight", False) is True:
            forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type, is_draft_model=True)
            if moe_comm_type is not None and forward_context.moe_comm_method is None:
                raise RuntimeError("A3 ZercMoE single-weight draft MoE communication was not initialized.")
        else:
            forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
        forward_context.use_mega_moe = use_cann_megamoe(vllm_config)
        forward_context.use_zerc_moe = use_cann_zercmoe(vllm_config)
        forward_context.is_decode_only_node = _is_decode_only_node(vllm_config)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when fia merge in fiav2
        forward_context.sinks = has_sinks

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        # set for sequence parallelism, 1000 is the batch size concurrency threshold
        # for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency
        # exceeds this threshold, the performance benefits can be maximized.
        # Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.

        # main model and drafter model may have different architecture
        is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
        if is_context_moe_model:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        elif is_draft_model:
            # TODO: for dense drafter, `sp` is redundant and is not compatible with `dp` and `graph`.
            # Disable it to avoid more problems.
            flash_comm_v1_enabled = False
        else:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled

        forward_context.pad_size = 0
        if forward_context.flash_comm_v1_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model
        forward_context.is_draft_model_prefill = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            dp_meta = forward_context.dp_metadata
            max_tokens_across_dp = max_num_tokens
            if forward_context.flash_comm_v1_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = max_num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp

        forward_context.eplb_heat_collection_status = eplb_heat_collection_status
        # Set only by the request execution entry point, never by dummy runs.
        forward_context.request_forward = request_forward
        if request_forward and forward_context.zercmoe_graph_execution:
            forward_context.moe_comm_method.log_request_graph_selection(max_num_tokens)

        forward_context.mc2_num_actual_tokens = None
        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            forward_context.mc2_num_actual_tokens = num_actual_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[: forward_context.padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask
        if (
            request_forward
            and not in_profile_run
            and not is_draft_model
            and get_ascend_device_type() == AscendDeviceType.A3
            and get_ascend_config().enable_zerc_moe == 1
            and moe_comm_type != MoECommType.ZERC_MOE
        ):
            zerc_comm = get_moe_comm_method(MoECommType.ZERC_MOE)
            if zerc_comm is not None:
                zerc_comm.log_request_fallback(
                    moe_comm_type,
                    graph=aclgraph_runtime_mode != CUDAGraphMode.NONE,
                    num_tokens=max_num_tokens,
                    capacity=get_zercmoe_tokens_capacity(vllm_config),
                )
        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None
# TODO: Remove the following variables and related methods once the megamoe operator supports mixed weight formats
#  (e.g., main model quantized, draft model in floating point)
_moe_quant_mismatch: bool | None = None
_dispatch_v2_tokens_capacity: int | None = None


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return

    ascend_config = get_ascend_config()
    use_mega_moe = use_cann_megamoe(vllm_config)
    # A3 Zerc has an independent scheduler-sized heap. Enabling it must not
    # change the baseline MC2 dispatch budget used by graph/draft fallbacks.
    use_zerc_moe = get_ascend_device_type() == AscendDeviceType.A2 and use_cann_zercmoe(vllm_config)
    is_decode_only_node = _is_decode_only_node(vllm_config)

    if ascend_config.enable_prefill_mc2 or ((use_mega_moe or use_zerc_moe) and not is_decode_only_node):
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
    tp_size = vllm_config.parallel_config.tensor_parallel_size

    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    # keep the num_tokens_per_tp_rank less than fused_mc2 (mega_moe) tokens per rank limit
    if ascend_config.enable_fused_mc2:
        if use_mega_moe or use_zerc_moe:
            num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _MEGA_MOE_TOKENS_PER_RANK_LIMIT)
        else:
            num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT)
        global _dispatch_v2_tokens_capacity
        # Compute dispatch_v2 (normal MC2) tokens capacity independently from the
        # fused_mc2 path above. When enable_prefill_mc2 or use_mega_moe is true,
        # max_num_tokens is based on max_num_batched_tokens (prefill+decode unified
        # batch), which is too large for the normal MC2 decode path. Recompute the
        # token count from the decode-only configuration so that
        # _dispatch_v2_tokens_capacity follows the normal MC2 branch logic.
        if vllm_config.compilation_config.cudagraph_capture_sizes:
            dispatch_v2_max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
        else:
            dispatch_v2_max_num_tokens = max_num_reqs * uniform_decode_query_len
        dispatch_v2_tokens_per_tp_rank = (dispatch_v2_max_num_tokens + tp_size - 1) // tp_size
        _dispatch_v2_tokens_capacity = min(dispatch_v2_tokens_per_tp_rank, _MC2_TOKENS_PER_RANK_LIMIT) * tp_size

    # keep the num_tokens_per_tp_rank less than mc2 tokens per rank limit
    else:
        num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _MC2_TOKENS_PER_RANK_LIMIT)
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def get_dispatch_v2_tokens_capacity():
    return _dispatch_v2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        padded_max_tokens = (max_tokens + tp_size - 1) // tp_size * tp_size
        _reserved_mc2_mask = torch.zeros(padded_max_tokens, dtype=torch.bool, device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def _select_a2_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
    is_decode: bool = False,
    is_draft_model: bool = False,
    is_graph_mode: bool = False,
) -> MoECommType:
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = (
        vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
    )
    num_experts_per_device = num_experts // ep_world_size
    if _use_zercmoe_for_batch(
        num_tokens,
        vllm_config,
        mc2_tokens_capacity,
        is_decode=is_decode,
        is_draft_model=is_draft_model,
        is_graph_mode=is_graph_mode,
    ):
        return MoECommType.ZERC_MOE
    if num_experts > 512:
        return MoECommType.ALLGATHER
    if (
        num_experts_per_device <= 24
        and ep_world_size >= 16
        and (num_tokens is None or num_tokens <= mc2_tokens_capacity)
    ):
        return MoECommType.MC2
    return MoECommType.ALLGATHER


def _is_moe_quantized(model_instance: torch.nn.Module) -> bool | None:
    """Check if the model's MoE layers use a quantized method.

    Scans the model for the first ``routed_experts.quant_method`` and checks
    whether it is an unquantized (float) or quantized MoE method instance.
    modelslim quantization cannot be reliably detected from config, so the
    actual runtime instance type is used instead.

    Returns ``True`` if quantized, ``False`` if unquantized, ``None`` if no MoE
    layers are found (e.g. dense model or ``model_instance`` is ``None``).
    """
    if model_instance is None:
        return None
    for module in model_instance.modules():
        routed_experts = getattr(module, "routed_experts", None)
        if routed_experts is None:
            continue
        quant_method = getattr(routed_experts, "quant_method", None)
        if quant_method is None:
            continue
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        return not isinstance(quant_method, UnquantizedFusedMoEMethod)
    return None


def set_moe_quant_mismatch(
    main_model: torch.nn.Module | None,
    draft_model: torch.nn.Module | None,
) -> None:
    """Cache whether the main and draft models have inconsistent MoE quantization.

    MegaMoe is configured from the MoE-part weight format. When the main model
    and the draft model use different MoE quantization (e.g. main is quantized
    while the draft is floating-point), the MegaMoe kernel cannot handle the
    mixed weight formats and hangs.

    Should be called once after both models are loaded. The weight format is
    fixed for the lifetime of the process, so the result is cached. Only the MoE
    part is compared; a model without MoE layers (or a ``None`` draft model)
    never exercises the MegaMoe path, so its quantization is irrelevant.
    """
    global _moe_quant_mismatch
    if _moe_quant_mismatch is not None:
        return

    _moe_quant_mismatch = False
    main_quantized = _is_moe_quantized(main_model)
    if main_quantized is None:
        return

    draft_quantized = _is_moe_quantized(draft_model)
    if draft_quantized is None:
        return

    if main_quantized != draft_quantized:
        _moe_quant_mismatch = True
        logger.warning_once(
            "FUSED_MC2 disabled on draft model path due to inconsistent MoE "
            "quantization between main model (quantized=%s) and draft model "
            "(quantized=%s). Main model still uses FUSED_MC2; draft model falls "
            "back to MC2/ALLTOALL.",
            main_quantized,
            draft_quantized,
        )


def _select_a3_moe_comm_method(
    num_tokens: int,
    mc2_tokens_capacity: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
    is_decode: bool = False,
    is_graph_mode: bool = False,
) -> MoECommType:
    if _use_zercmoe_for_batch(
        num_tokens,
        vllm_config,
        mc2_tokens_capacity,
        is_decode=is_decode,
        is_draft_model=is_draft_model,
        is_graph_mode=is_graph_mode,
    ):
        return MoECommType.ZERC_MOE

    skip_fused_mc2 = is_draft_model and _moe_quant_mismatch
    if not skip_fused_mc2 and use_cann_megamoe(vllm_config):
        return MoECommType.FUSED_MC2
    if not skip_fused_mc2 and get_ascend_config().enable_fused_mc2 == 1 and get_ep_group().world_size <= 32:
        return MoECommType.FUSED_MC2

    mc2_tokens_capacity = get_dispatch_v2_tokens_capacity() or mc2_tokens_capacity
    if num_tokens is None or num_tokens <= mc2_tokens_capacity:
        return MoECommType.MC2

    return MoECommType.ALLTOALL


def _select_a5_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
) -> MoECommType:
    num_experts_per_tok = getattr(
        vllm_config.model_config.hf_text_config,
        "num_experts_per_tok",
        getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
    )
    world_size = vllm_config.parallel_config.world_size_across_dp
    if (num_tokens is None or num_tokens <= mc2_tokens_capacity) and world_size > 1:
        return MoECommType.MC2
    if world_size <= num_experts_per_tok:
        return MoECommType.ALLGATHER
    return MoECommType.ALLTOALL


def select_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
    is_decode: bool = False,
    is_graph_mode: bool = False,
) -> MoECommType | None:
    """Select the MoE communication method according to parallel settings,
    device generation, and token count.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, ZERC_MOE takes priority for bf16 models on
       eager prefill-shaped batches within capacity; otherwise pick MC2 when
       tokens fit the MC2 capacity and the DP size is large enough, or
       all-gather.
    4. On A3 with expert parallel, check the verified ZercMoE adapter gate,
       then prefer fused MC2 when enabled and the EP group
       size is small enough; otherwise use MC2 within capacity or all-to-all.
    5. On 310P, always use all-gather.
    6. On A5 with expert parallel, use MC2 when tokens fit the MC2 capacity
       and the EP size is large enough; otherwise use all-gather when
       EP size is smaller than num of topK experts or all-to-all.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode.
        is_decode (bool): Whether the caller explicitly requests a decode path.
        is_graph_mode (bool): Whether the forward uses graph execution
            (cudagraph_runtime_mode != NONE), independently of batch type.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        if getattr(get_ascend_config(), "zercmoe_single_weight", False) is True:
            raise RuntimeError("A3 ZercMoE single-weight requires an eligible MoE model; fallback is unavailable.")
        return None

    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    lora_config = getattr(vllm_config, "lora_config", None)
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif lora_config is not None and vllm_config.parallel_config.enable_expert_parallel:
        # LoRA + EP requires AlltoAll because the MC2/FusedMC2 paths
        # Ascend MoE LoRA cannot patch FusedMC2 path for dispatch_ffn_combine/mega_moe
        # is a single fused C++ op. This covers both normal model
        # forward and _dummy_run during profile_run.
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version == AscendDeviceType.A2:
        moe_comm_type = _select_a2_moe_comm_method(
            num_tokens,
            vllm_config,
            mc2_tokens_capacity,
            is_decode=is_decode,
            is_draft_model=is_draft_model,
            is_graph_mode=is_graph_mode,
        )
    elif soc_version == AscendDeviceType.A3:
        moe_comm_type = _select_a3_moe_comm_method(
            num_tokens,
            mc2_tokens_capacity,
            vllm_config,
            is_draft_model=is_draft_model,
            is_decode=is_decode,
            is_graph_mode=is_graph_mode,
        )
    elif soc_version == AscendDeviceType.A5:
        moe_comm_type = _select_a5_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity)
    elif soc_version == AscendDeviceType._310P:
        moe_comm_type = MoECommType.ALLGATHER

    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    if (
        getattr(get_ascend_config(), "zercmoe_single_weight", False) is True
        and not is_draft_model
        and moe_comm_type != MoECommType.ZERC_MOE
    ):
        raise RuntimeError(
            "A3 ZercMoE single-weight cannot fall back to another MoE backend: "
            f"selected={moe_comm_type}, tokens={num_tokens}, graph={is_graph_mode}, draft={is_draft_model}."
        )
    logger.debug(
        "MoE comm method selected: soc=%s, method=%s, num_tokens=%d, "
        "mc2_capacity=%s, zercmoe_capacity=%s, graph=%s, draft=%s",
        soc_version,
        moe_comm_type,
        num_tokens,
        mc2_tokens_capacity,
        (
            get_zercmoe_tokens_capacity(vllm_config)
            if soc_version == AscendDeviceType.A3 and getattr(get_ascend_config(), "enable_zerc_moe", 0) == 1
            else None
        ),
        is_graph_mode,
        is_draft_model,
    )
    return moe_comm_type


class _ExtraForwardContextProxy:
    """Unified forward-context access for v1/v2 model runners."""

    extra_attrs = (
        "capturing",
        "moe_comm_type",
        "moe_comm_method",
        "use_mega_moe",
        "use_zerc_moe",
        "is_decode_only_node",
        "mmrs_fusion",
        "num_tokens",
        "flash_comm_v1_enabled",
        "pad_size",
        "padded_length",
        "num_tokens_across_dp",
        "mc2_mask",
        "mc2_num_actual_tokens",
        "is_draft_model",
        "is_draft_model_prefill",
        "prefetch_mlp_gate_up_proj",
        "prefetch_mlp_down_proj",
        "model_instance",
        "layer_idx",
        "max_tokens_across_dp",
        "max_tokens_across_pcp",
        "num_accept_tokens",
        "in_profile_run",
        "request_forward",
        "zercmoe_graph_execution",
        "padded_num_tokens",
        "sinks",
        "eplb_heat_collection_status",
    )

    def check_extra_attr(self, name: str):
        if name not in self.extra_attrs:
            raise AttributeError(
                f"{name} is not extra forward context attribute, "
                "please get/set it from vllm's _forward_context directly."
            )

    @staticmethod
    def _ctx():
        return get_forward_context()

    def __getattr__(self, name: str) -> Any:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            # Unset known extras default to None so optional flags (e.g. `sinks`)
            # can be read with truthiness checks before the V2 path populates them.
            return ctx.additional_kwargs.get(name)
        return getattr(ctx, name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            ctx.additional_kwargs[name] = value
        else:
            setattr(ctx, name, value)


# usage: from vllm_ascend.ascend_forward_context import _EXTRA_CTX
_EXTRA_CTX = _ExtraForwardContextProxy()
