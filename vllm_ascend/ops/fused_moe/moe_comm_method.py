# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import _CANN_OPS_TRANSFORMER_AVAILABLE, get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import comm_utils
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
    PrepareAndFinalizeWithZercMoE,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.ops.fused_moe.zercmoe_adapter import (
    get_w8a8_operator,
    make_w8a8_padding_ids,
    prepare_w8a8_input,
    validate_w8a8_payload,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
from vllm_ascend.zercmoe_config import ZERCMOE_MIN_SHMEM_MB, is_zercmoe_draft_model_loading

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}
_DraftMoECommMethods: dict[MoECommType | None, MoECommMethod] = {}

# Default SHMEM bootstrap address for the ZercMoE transport (single node).
# Override with additional_config.zercmoe_ipport; must be identical across all
# ranks of one EP group and unique across concurrent jobs.
_ZERCMOE_DEFAULT_IPPORT = "tcp://127.0.0.1:28765"


def get_moe_comm_method(moe_comm_type: MoECommType | None, *, is_draft_model: bool = False) -> MoECommMethod | None:
    return (_DraftMoECommMethods if is_draft_model else _MoECommMethods).get(moe_comm_type)


def shutdown_zercmoe() -> None:
    """Release the ZercMoE buffer during coordinated shutdown on every EP rank.

    Never call this from a finalizer/atexit handler or a single-rank error
    fallback: the operator's documented destroy method contains a barrier.
    """
    comm = _MoECommMethods.get(MoECommType.ZERC_MOE)
    if comm is not None:
        comm.close()


def initialize_zercmoe_buffer() -> None:
    """Collective allocation before KV sizing, without launching a MoE kernel."""
    comm = _MoECommMethods.get(MoECommType.ZERC_MOE)
    if not isinstance(comm, ZercMoECommImpl):
        raise RuntimeError("ZERC_MOE communication was not initialized for this model.")
    comm.initialize_w8a8_buffer()


def initialize_zercmoe_padding_ids() -> None:
    """Create the fixed routing IDs on the eager stream before graph capture."""
    comm = _MoECommMethods.get(MoECommType.ZERC_MOE)
    if not isinstance(comm, ZercMoECommImpl):
        raise RuntimeError("ZERC_MOE communication was not initialized for this model.")
    comm.initialize_w8a8_padding_ids()


def setup_moe_comm_method(moe_config):
    draft_only = is_zercmoe_draft_model_loading()
    methods = _DraftMoECommMethods if draft_only else _MoECommMethods
    if getattr(get_ascend_config(), "zercmoe_single_weight", False) is True:
        existing = methods.get(MoECommType.MC2) or methods.get(MoECommType.ALLGATHER)
        previous = getattr(existing, "moe_config", None)
        if previous is not None:
            fields = (
                "ep_size",
                "tp_size",
                "dp_size",
                "pcp_size",
                "num_experts",
                "num_local_experts",
                "experts_per_token",
                "hidden_dim",
                "intermediate_size_per_partition",
            )
            if any(getattr(previous, field, None) != getattr(moe_config, field, None) for field in fields):
                raise ValueError("A3 ZercMoE single-weight requires consistent MoE configuration within each model.")
    if moe_config.ep_size > 1:
        methods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        methods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        methods[MoECommType.MC2] = MC2CommImpl(moe_config)
        methods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
        if not draft_only:
            methods[MoECommType.ZERC_MOE] = ZercMoECommImpl(moe_config)
    else:
        methods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context
        self.prepare_finalize.set_lora_context(lora_context)
        self.token_dispatcher.set_lora_context(lora_context)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )

        mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        if _CANN_OPS_TRANSFORMER_AVAILABLE:
            self.mega_moe_symm_buffer = None
            self.get_symm_buffer_for_mega_moe, self.mega_moe = comm_utils.load_cann_mega_moe_ops()

        if get_ascend_config().enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2(is_fused_mc2=True)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _init_mega_moe_symm_buffer(
        self,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: torch.dtype | None = None,
        *,
        is_decode_only_node: bool,
    ):
        # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where global_bs / ep_world_size live.
        # Assert it so mypy resolves those attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        group = get_mc2_group().device_group
        # The sym buffer is allocated by get_symm_buffer_for_mega_moe, a
        # collective handshake over the EP (mc2) group. Its shape params —
        # especially num_max_tokens_per_rank — MUST be identical on every EP
        # rank, otherwise ranks allocate mismatched buffers / at different
        # times and HCCL aborts (SUSPECT REMOTE ERROR 507057). So this value
        # must be derived ONLY from rank-invariant, compile-time config,
        # NEVER from the current forward's per-rank token count.
        if self.token_dispatcher.global_bs > 0:
            # global_bs = num_tokens_per_tp_rank * ep_world_size (compile-time).
            num_max_tokens_per_rank = max(
                1,
                int(self.token_dispatcher.global_bs // self.token_dispatcher.ep_world_size),
            )
        else:
            # num_tokens_per_tp_rank, set once in TokenDispatcherWithMC2.__init__
            # from scheduler/graph config — rank-invariant.
            rank_invariant_cap = getattr(self.token_dispatcher, "max_num_tokens_per_rank", 0)
            num_max_tokens_per_rank = max(1, int(rank_invariant_cap))
        num_topk = self.moe_config.experts_per_token
        num_experts = self.moe_config.num_experts
        expert_per_rank = max(1, num_experts // int(self.token_dispatcher.ep_world_size))
        absolute_safe_max_recv_token_num = max(
            1,
            num_max_tokens_per_rank * int(self.token_dispatcher.ep_world_size) * min(num_topk, expert_per_rank),
        )

        if is_decode_only_node:
            max_recv_token_num = absolute_safe_max_recv_token_num
        else:
            # P nodes and PD-mixed nodes use the configured value. This keeps
            # the existing memory/performance tradeoff for prefill workloads.
            max_recv_token_num = get_ascend_config().mega_moe_max_tokens
            logger.warning_once(
                "MegaMoe symm buffer: max_recv_token_num is set from "
                "mega_moe_max_tokens=%d (reference value) on a P or PD-mixed "
                "node. If the actual per-rank received token count after "
                "dispatch exceeds this value, precision degradation will "
                "occur. The absolute safe upper bound is %d "
                "(num_max_tokens_per_rank=%d, ep_world_size=%d, num_topk=%d, "
                "expert_per_rank=%d). Please tune mega_moe_max_tokens in "
                "additional_config based on actual expert load distribution.",
                max_recv_token_num,
                absolute_safe_max_recv_token_num,
                num_max_tokens_per_rank,
                int(self.token_dispatcher.ep_world_size),
                num_topk,
                expert_per_rank,
            )

        logger.info(
            "CANN MegaMoe sym-buffer alloc (must match across all EP ranks): ep_rank=%s ep_world=%s global_bs=%s",
            getattr(self.token_dispatcher, "ep_rank_id", "?"),
            getattr(self.token_dispatcher, "ep_world_size", "?"),
            self.token_dispatcher.global_bs,
        )

        return self.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=self.moe_config.intermediate_size_per_partition,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
        is_decode_only_node: bool,
    ):
        # TokenDispatcherWithMC2 carries global_bs (used below for the mc2_mask
        # branch); assert the subtype so mypy resolves it off the base class.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_max_tokens = self.token_dispatcher.max_num_tokens_per_rank
        if num_tokens > num_max_tokens:
            raise ValueError(
                f"MegaMoe received {num_tokens} tokens per rank, but its symmetric buffer "
                f"was allocated for at most {num_max_tokens}. Increase max_num_batched_tokens "
                "or disable fused MC2."
            )

        def to_list(x):
            return x if isinstance(x, list) else [x]

        weight1 = to_list(fused_experts_input.weights.w1)
        weight2 = to_list(fused_experts_input.weights.w2)
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
        weight_scales1 = fused_experts_input.weights.w1_scale
        weight_scales2 = fused_experts_input.weights.w2_scale

        activation_clamp = fused_experts_input.swiglu_limit if fused_experts_input.swiglu_limit > 0 else None
        x_active_mask = None
        if self.token_dispatcher.global_bs == 0 and fused_experts_input.routing.mc2_mask is not None:
            # mc2_mask comes from the reserved bool buffer in
            # ascend_forward_context.set_mc2_mask. MegaMoe wants int8 as
            # the per-token active mask, so cast only when the dtype does
            # not already match — saves the kernel launch when an upstream
            # change ever flips the reserved buffer to int8.
            raw_mask = fused_experts_input.routing.mc2_mask
            if raw_mask.dtype == torch.int8:
                x_active_mask = raw_mask.contiguous()
            else:
                x_active_mask = raw_mask.to(torch.int8).contiguous()
        # A8W4-INT precision-compensation biases B1/B2 (l1_bias/l2_bias).
        l1_bias = fused_experts_input.weights.w1_scale_bias
        l2_bias = fused_experts_input.weights.w2_scale_bias

        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = comm_utils._get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )

        if self.mega_moe_symm_buffer is None:
            self.mega_moe_symm_buffer = self._init_mega_moe_symm_buffer(
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
                is_decode_only_node=is_decode_only_node,
            )
        else:
            self.mega_moe_symm_buffer.dispatch_quant_mode = dispatch_quant_mode
            self.mega_moe_symm_buffer.dispatch_quant_out_dtype = dispatch_quant_out_dtype

        out, expert_tokens = self.mega_moe(
            fused_experts_input.hidden_states,
            topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            self.mega_moe_symm_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            x_active_mask=x_active_mask,
            activation_clamp=activation_clamp,
            weight1_type=weight_type,
            weight2_type=weight_type,
        )
        # NOTE: self.expert_token_nums is only used by the
        # mega_moe path (enable_fused_mc2 == 1) as a
        # pre-allocated in/out buffer. The MegaMoe op returns a fresh
        # expert_tokens tensor that is consumed by the caller via the
        # return value, so there is nothing to keep on the instance.
        return out, expert_tokens

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # SiTU is implemented by the generic MoE path. Keep other activations
        # on the upstream MegaMoE path, including unquantized shared experts.
        if isinstance(fused_experts_input.activation, SituActivationConfig):
            return super().fused_experts(fused_experts_input)

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == 1:
            if _EXTRA_CTX.use_mega_moe:
                out, expert_tokens = self._apply_cann_mega_moe(
                    fused_experts_input,
                    topk_ids,
                    is_decode_only_node=_EXTRA_CTX.is_decode_only_node,
                )
            else:
                assert not (
                    fused_experts_input.weights.w1_scale_bias is None
                    or fused_experts_input.weights.w2_scale_bias is None
                ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."

                out = torch.empty_like(fused_experts_input.hidden_states)
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=fused_experts_input.weights.w1,
                    weight2=fused_experts_input.weights.w2,
                    expert_idx=topk_ids,
                    scale1=fused_experts_input.weights.w1_scale,
                    scale2=fused_experts_input.weights.w2_scale,
                    bias1=fused_experts_input.weights.w1_scale_bias,
                    bias2=fused_experts_input.weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=fused_experts_input.swiglu_limit,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
                expert_tokens = self.expert_token_nums
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out,
            expert_tokens=expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )


class ZercMoECommImpl(MoECommMethod):
    """A3 ZercMoE W8A8 backend; A2 retains its BF16 ABI.

    Prefill, decode and mixed eager batches share this interface. Uniform
    EP padding and TP/SP finalization use the existing MC2 preparation. Each
    buffer is created collectively from rank-invariant capacity settings.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        self._zercmoe = None
        self._a3_zercmoe = None
        self.zerc_symm_buffer = None
        self._zerc_buffer_weight_dtype = None
        self._zerc_buffer_input_dtype = None
        self._zerc_w8a8_operator = None
        self._logged_zercmoe_launch = False
        self._logged_zercmoe_fallbacks = set()
        self._w8a8_padding_ids = None
        self._w8a8_padding_key = None
        self._w8a8_padding_inputs = None
        self._w8a8_padding_inputs_key = None

    def log_request_fallback(self, method, *, graph, num_tokens, capacity):
        """Report each fallback reason once, outside capture and dummy runs."""
        if self.token_dispatcher.ep_rank_id != 0:
            return
        reason = (
            "graph" if graph else "capacity" if num_tokens is not None and num_tokens > capacity else "configuration"
        )
        key = (method, reason)
        if key in self._logged_zercmoe_fallbacks:
            return
        self._logged_zercmoe_fallbacks.add(key)
        logger.info(
            "A3 ZercMoE request fallback: method=%s reason=%s num_tokens=%s capacity=%s. "
            "This request forward uses the original MoE backend.",
            method,
            reason,
            num_tokens,
            capacity,
        )

    def _log_request_launch(self, num_tokens):
        if (
            self._logged_zercmoe_launch
            or self.token_dispatcher.ep_rank_id != 0
            or _EXTRA_CTX.request_forward is not True
            or _EXTRA_CTX.in_profile_run
            or _EXTRA_CTX.is_draft_model
        ):
            return
        self._logged_zercmoe_launch = True
        logger.info(
            "A3 ZercMoE request launched: backend=zercmoe.mega_moe mode=%s EP=%s local_rows=%s. "
            "The asynchronous call was submitted; this is not a device completion check.",
            "graph" if getattr(_EXTRA_CTX, "zercmoe_graph_execution", False) else "eager",
            self.token_dispatcher.ep_world_size,
            num_tokens,
        )

    def log_request_graph_selection(self, num_tokens):
        """Log graph selection outside capture, without claiming device execution."""
        if getattr(self, "_logged_zercmoe_graph_selection", False) or self.token_dispatcher.ep_rank_id != 0:
            return
        self._logged_zercmoe_graph_selection = True
        logger.info(
            "A3 ZercMoE decode graph selected: EP=%s tokens=%s. "
            "Selection alone does not confirm graph replay or numerical correctness.",
            self.token_dispatcher.ep_world_size,
            num_tokens,
        )

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        if get_ascend_device_type() == AscendDeviceType.A3:
            from vllm.config import get_current_vllm_config

            from vllm_ascend.ascend_forward_context import get_zercmoe_tokens_capacity

            return TokenDispatcherWithMC2(
                is_fused_mc2=True, token_capacity=get_zercmoe_tokens_capacity(get_current_vllm_config())
            )
        return TokenDispatcherWithMC2(is_fused_mc2=True)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithZercMoE(self.moe_config)

    def prepare(
        self,
        hidden_states,
        router_logits,
        enable_shared_expert_dp=False,
        replace_allreduce=False,
        quant_type=QuantType.NONE,
        *,
        padding_top_k=None,
    ):
        factory = None
        if padding_top_k is not None and quant_type == QuantType.W8A8:
            # The runner has checked that this layer uses standard routing.
            # Prepare invokes this only after proving the entire shard is padding.
            def factory(x, rows, mask):
                return self._get_w8a8_padding_inputs_for_shape(x, padding_top_k, mask, rows)

        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
            padding_inputs_factory=factory,
        )

    def close(self):
        """All EP ranks must enter this normal shutdown path together."""
        if self.zerc_symm_buffer is None:
            return
        # Forward is asynchronous. Finish all streams before releasing the
        # buffer. The external A2 destroy also enters a cross-rank barrier.
        torch.npu.synchronize()
        self.zerc_symm_buffer.destroy()
        self.zerc_symm_buffer = None
        self._zerc_buffer_weight_dtype = None
        self._zerc_buffer_input_dtype = None

    def _ensure_zercmoe_loaded(self, *, is_w8a8=False):
        if is_w8a8:
            if getattr(self, "_a3_zercmoe", None) is None:
                self._a3_zercmoe = comm_utils.load_a3_zercmoe_ops()
        elif self._zercmoe is None:
            # A2's existing nonquantized backend stays independent of A3.
            self._zercmoe = comm_utils.load_zercmoe_ops()

    def _init_zerc_symm_buffer(self, w_dtype=torch.bfloat16):
        # ZercMoECommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where ep_world_size /
        # max_num_tokens_per_rank live. Assert it so mypy resolves those
        # attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        ascend_config = get_ascend_config()
        num_max_tokens_per_rank = max(1, int(self.token_dispatcher.max_num_tokens_per_rank))
        ep_world_size = int(self.token_dispatcher.ep_world_size)
        ep_rank = int(self.token_dispatcher.ep_rank_id)
        if ep_world_size <= 0 or not 0 <= ep_rank < ep_world_size:
            raise ValueError(f"Invalid ZercMoE EP group: rank={ep_rank}, world_size={ep_world_size}.")

        if w_dtype == torch.int8:
            ep_group = get_mc2_group()
            if ep_group.world_size != ep_world_size or ep_group.rank_in_group != ep_rank:
                raise ValueError("ZERC_MOE bootstrap identity must match the dispatcher's actual EP group.")
            if getattr(ascend_config, "zercmoe_max_recv_token_num", None) is not None:
                raise ValueError("zercmoe_max_recv_token_num is not part of the internal 1.5 API; remove it.")
            mem_size_mb = ascend_config.zercmoe_shmem_mb or 0
            if mem_size_mb and mem_size_mb < ZERCMOE_MIN_SHMEM_MB:
                raise ValueError(
                    f"A3 ZercMoE requires zercmoe_shmem_mb=0/None or at least {ZERCMOE_MIN_SHMEM_MB} MB per rank. "
                    "The explicit minimum is a framework guard, not a guarantee that the heap is sufficient. "
                    "Automatic sizing is determined by the wheel."
                )
            comm_utils.bind_zercmoe_ep_environment(ep_rank, ep_world_size)
            device_id = torch.npu.current_device()
            device_count = getattr(torch.npu, "device_count", None)
            logger.info(
                "A3 ZercMoE device visibility: EP rank=%s/%s ASCEND_RT_VISIBLE_DEVICES=%s "
                "visible_device_count=%s current_logical_device=%s",
                ep_rank,
                ep_world_size,
                os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"),
                device_count() if callable(device_count) else "unknown",
                device_id,
            )
            self._ensure_zercmoe_loaded(is_w8a8=True)
            ipport = ascend_config.zercmoe_ipport or _ZERCMOE_DEFAULT_IPPORT
            logger.info(
                "A3 ZercMoE SHMEM buffer: EP=%s rank=%s source_capacity=%s "
                "ipport=%s mem_size_mb=%s (0 delegates heap sizing to the wheel)",
                ep_world_size,
                ep_rank,
                num_max_tokens_per_rank,
                ipport,
                mem_size_mb,
            )
            if ep_rank == 0:
                local_experts = self.moe_config.num_experts // ep_world_size
                if (
                    ep_world_size * min(self.moe_config.experts_per_token, local_experts)
                    > 2 * self.moe_config.experts_per_token
                ):
                    logger.warning_once(
                        "ZercMoE 1.5 may silently truncate routed tokens under skewed expert loads: "
                        "the reported per-rank capacity is 2*m*topK. This affects eager and graph. "
                        "Validate hotspot routing against an independent reference before relying on outputs."
                    )
                logger.info(
                    "A3 ZercMoE is eligible for eager and FULL_DECODE_ONLY decode graph batches. "
                    "SHMEM initialization alone does not indicate a request called ZercMoE."
                )
            return self._a3_zercmoe.get_symm_buffer(
                moe_expert_num=self.moe_config.num_experts,
                ep_world_size=ep_world_size,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=self.moe_config.experts_per_token,
                hidden=self.moe_config.hidden_dim,
                intermediate_hidden=self.moe_config.intermediate_size_per_partition,
                ipport=ipport,
                device_id=device_id,
                w_dtype=torch.int8,
                mem_size_mb=mem_size_mb,
            )

        # A2 retains its original external-wheel bootstrap environment.
        rank_environment = {"RANK": ep_rank, "WORLD_SIZE": ep_world_size}
        for name, expected in rank_environment.items():
            value = os.environ.get(name)
            if value is None:
                continue
            try:
                actual = int(value)
            except ValueError as exc:
                raise ValueError(f"ZercMoE requires integer {name} matching its EP group; got {value!r}.") from exc
            if actual != expected:
                raise ValueError(
                    f"ZercMoE {name}={actual} does not match its EP group value {expected}. "
                    "Check the worker launch environment and EP group before initializing SHMEM."
                )
        for name, expected in rank_environment.items():
            os.environ.setdefault(name, str(expected))

        self._ensure_zercmoe_loaded()
        ipport = ascend_config.zercmoe_ipport or _ZERCMOE_DEFAULT_IPPORT
        mem_size_mb = ascend_config.zercmoe_shmem_mb or 0
        logger.info(
            "ZercMoE sym-buffer alloc (must match across all EP ranks): "
            "ep_world=%s num_max_tokens_per_rank=%s num_experts=%s topk=%s hidden=%s "
            "intermediate=%s ipport=%s mem_size_mb=%s",
            ep_world_size,
            num_max_tokens_per_rank,
            self.moe_config.num_experts,
            self.moe_config.experts_per_token,
            self.moe_config.hidden_dim,
            self.moe_config.intermediate_size_per_partition,
            ipport,
            mem_size_mb,
        )
        return self._zercmoe.get_symm_buffer(
            moe_expert_num=self.moe_config.num_experts,
            ep_world_size=ep_world_size,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=self.moe_config.experts_per_token,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=self.moe_config.intermediate_size_per_partition,
            ipport=ipport,
            device_id=torch.npu.current_device(),
            mem_size_mb=mem_size_mb,
        )

    def initialize_w8a8_buffer(self):
        """Allocate the single model-wide heap inside KV memory profiling."""
        if self.zerc_symm_buffer is None:
            self.zerc_symm_buffer = self._init_zerc_symm_buffer(w_dtype=torch.int8)
            self._zerc_buffer_weight_dtype = torch.int8
            self._zerc_buffer_input_dtype = torch.bfloat16

    def _get_w8a8_padding_ids(self, payload):
        """Reuse the small device vector across layers and repeated eager calls."""
        if (
            payload.routing.mc2_mask is None
            or getattr(self.prepare_finalize, "all_tokens_valid", False)
            or payload.topk_ids.ndim != 2
        ):
            return None
        key = (
            self.moe_config.num_experts,
            self.token_dispatcher.ep_world_size,
            self.token_dispatcher.ep_rank_id,
            payload.topk_ids.shape[1],
            payload.hidden_states.device,
        )
        if key != self._w8a8_padding_key:
            if getattr(_EXTRA_CTX, "zercmoe_graph_execution", False):
                raise RuntimeError("ZERC_MOE graph padding IDs were not initialized before graph capture.")
            ids = make_w8a8_padding_ids(
                num_experts=key[0], ep_world_size=key[1], ep_rank=key[2], top_k=key[3], device=key[4]
            )
            self._w8a8_padding_ids, self._w8a8_padding_key = ids, key
        return self._w8a8_padding_ids

    def initialize_w8a8_padding_ids(self, device=None):
        """Keep the graph's constant IDs alive outside private graph pools."""
        key = (
            self.moe_config.num_experts,
            self.token_dispatcher.ep_world_size,
            self.token_dispatcher.ep_rank_id,
            self.moe_config.experts_per_token,
            device if device is not None else torch.device("npu", torch.npu.current_device()),
        )
        ids = make_w8a8_padding_ids(
            num_experts=key[0], ep_world_size=key[1], ep_rank=key[2], top_k=key[3], device=key[4]
        )
        self._w8a8_padding_ids, self._w8a8_padding_key = ids, key

    def _get_w8a8_padding_inputs(self, payload):
        """Reuse only dummy inputs, never a previous layer's weights or routing."""
        x, ids = payload.hidden_states, payload.topk_ids
        return self._get_w8a8_padding_inputs_for_shape(
            x, ids.shape[1] if ids.ndim == 2 else None, payload.routing.mc2_mask, x.shape[0] if x.ndim == 2 else 0
        )

    def _get_w8a8_padding_inputs_for_shape(self, x, top_k, mask, rows):
        if (
            not getattr(self.prepare_finalize, "all_tokens_padding", False)
            or mask is None
            or x.ndim != 2
            or top_k is None
        ):
            # A large idle DP shard must not keep its zero-input allocation
            # alive once this rank resumes handling real or mixed tokens.
            self._w8a8_padding_inputs = None
            self._w8a8_padding_inputs_key = None
            return None
        # Rebuild on stream changes so initialization is ordered before use
        # without events or host/device synchronization. Keep one bounded entry.
        stream = torch.npu.current_stream().npu_stream if x.device.type == "npu" else None
        key = (
            self.moe_config.num_experts,
            self.token_dispatcher.ep_world_size,
            self.token_dispatcher.ep_rank_id,
            (rows, x.shape[1]),
            top_k,
            x.device,
            x.dtype,
            stream,
        )
        if key != self._w8a8_padding_inputs_key:
            # Build IDs on this stream too; do not read a vector initialized
            # by a previous stream's mixed-padding path.
            padding_ids = make_w8a8_padding_ids(
                num_experts=key[0], ep_world_size=key[1], ep_rank=key[2], top_k=key[4], device=x.device
            )
            inputs = (
                torch.zeros(key[3], dtype=x.dtype, device=x.device),
                padding_ids.repeat(rows, 1),
                torch.zeros((rows, top_k), dtype=torch.float32, device=x.device),
            )
            self._w8a8_padding_inputs, self._w8a8_padding_inputs_key = inputs, key
        return self._w8a8_padding_inputs

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        *,
        record_dispatch_event: bool = True,
    ):
        quant_type = fused_experts_input.quant.quant_type
        is_w8a8 = quant_type == QuantType.W8A8
        if get_ascend_device_type() == AscendDeviceType.A3 and not is_w8a8:
            raise ValueError("A3 ZercMoE requires W8A8 weights; the A2 BF16 backend cannot be used on A3.")
        if quant_type not in (QuantType.NONE, QuantType.W8A8):
            raise ValueError("ZERC_MOE supports A2 BF16 and A3 BF16-input W8A8 backends only.")

        if fused_experts_input.hidden_states.shape[0] == 0:
            # This flag comes from synchronized DP metadata in prepare, never
            # from a rank's local valid-token count or padding mask.
            if not getattr(self.prepare_finalize, "all_ranks_empty", False):
                raise ValueError("ZERC_MOE cannot skip an empty local rank without an all-EP-empty batch.")
            return FusedExpertsResult(
                routed_out=torch.empty(
                    fused_experts_input.hidden_states.shape,
                    device=fused_experts_input.hidden_states.device,
                    dtype=self.prepare_finalize.input_dtype,
                )
            )

        if is_w8a8:
            padding_inputs = getattr(self.prepare_finalize, "padding_inputs", None)
            if padding_inputs is None:
                padding_inputs = self._get_w8a8_padding_inputs(fused_experts_input)
            fused_experts_input = prepare_w8a8_input(
                fused_experts_input,
                all_tokens_valid=getattr(self.prepare_finalize, "all_tokens_valid", False),
                all_tokens_padding=getattr(self.prepare_finalize, "all_tokens_padding", False),
                padding_ids=self._get_w8a8_padding_ids(fused_experts_input) if padding_inputs is None else None,
                padding_inputs=padding_inputs,
            )

        # Weight packing belongs to model loading, never to a forward call.
        w1, w2 = fused_experts_input.weights.w1, fused_experts_input.weights.w2
        weight_dtype = torch.int8 if is_w8a8 else torch.bfloat16
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        if is_w8a8:
            validate_w8a8_payload(
                fused_experts_input,
                num_experts=self.moe_config.num_experts,
                ep_world_size=self.token_dispatcher.ep_world_size,
                hidden=self.moe_config.hidden_dim,
                intermediate=self.moe_config.intermediate_size_per_partition,
            )
        else:
            if not all(
                isinstance(weight, torch.Tensor) and weight.dim() == 1 and weight.dtype == weight_dtype
                for weight in (w1, w2)
            ):
                raise ValueError("ZERC_MOE A2 requires pre-packed one-dimensional BF16 weights.")
            if fused_experts_input.hidden_states.dtype != torch.bfloat16:
                raise ValueError("ZERC_MOE unquantized inputs must be BF16.")
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_max_tokens = self.token_dispatcher.max_num_tokens_per_rank
        if num_tokens > num_max_tokens:
            raise ValueError(
                f"ZercMoE received {num_tokens} tokens per rank, but its symmetric buffer "
                f"was allocated for at most {num_max_tokens}. Reduce max-num-batched-tokens "
                "or disable ZERC_MOE."
            )
        if self.zerc_symm_buffer is not None and (
            self._zerc_buffer_weight_dtype != weight_dtype
            or self._zerc_buffer_input_dtype != fused_experts_input.hidden_states.dtype
        ):
            raise ValueError(
                "ZERC_MOE cannot reuse a symmetric buffer across different weight dtypes or activation dtypes."
            )

        if is_w8a8:
            if self.prepare_finalize.input_dtype != torch.bfloat16:
                raise ValueError("ZERC_MOE A3 requires the model's BF16 output dtype from prepare.")
            self._ensure_zercmoe_loaded(is_w8a8=True)
            if getattr(self, "_zerc_w8a8_operator", None) is None:
                self._zerc_w8a8_operator = get_w8a8_operator(self._a3_zercmoe)

        # A3 preparation already resolves logical IDs and normalizes routing.
        # Keep A2's original mapping without constructing another A3 payload.
        topk_ids = fused_experts_input.topk_ids
        if not is_w8a8 and fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        # The quantized shared-expert path does not consume this event. Keep
        # it for A2 and for callers with an actual shared-stream dependency.
        before_dispatch_evt = (
            torch.npu.current_stream().record_event() if record_dispatch_event or not is_w8a8 else None
        )

        if self.zerc_symm_buffer is None:
            if is_w8a8:
                self.initialize_w8a8_buffer()
            else:
                self.zerc_symm_buffer = self._init_zerc_symm_buffer()
                self._zerc_buffer_weight_dtype = weight_dtype
                self._zerc_buffer_input_dtype = fused_experts_input.hidden_states.dtype

        # A3 uses the public 1.5 wheel API; A2 retains its existing ABI.
        if is_w8a8:
            out, expert_tokens = self._zerc_w8a8_operator(fused_experts_input, self.zerc_symm_buffer)
            self._log_request_launch(num_tokens)
        else:
            out, expert_tokens = torch.ops.npu.zerc_moe(
                fused_experts_input.hidden_states,
                topk_ids.to(torch.int32),
                fused_experts_input.topk_weights.to(torch.float32),
                [w1],
                [w2],
                self.zerc_symm_buffer.moe_expert_num,
                self.zerc_symm_buffer.ep_world_size,
                self.zerc_symm_buffer.ccl_buffer_size,
                0,  # topk_weights_type: fp32
                0,  # return_expert_tokens: counts are only consumed by dynamic
                # EPLB, which the ZERC_MOE gate excludes (mode=0 skips a per-call
                # sync: measured 3-9x lower single-call latency).
            )
        return FusedExpertsResult(
            routed_out=out,
            before_dispatch_evt=before_dispatch_evt,
            expert_tokens=expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )
