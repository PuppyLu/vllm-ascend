# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional vLLM V2 eager Decode event tracing for model A/B experiments."""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm_ascend.models.glm5next.mega_kernel_collectives import agree_decode_step


def install_decode_step_trace(runner, gate, config, *, group_agreed):
    """Install per-rank timing on real runner steps after model loading.

    The timed boundary ends at hidden states, excludes logits/sampling, and is
    complete only when the caller has independently verified the event stream
    joins the needed communication dependencies. No per-step synchronization.
    """
    from vllm.distributed.parallel_state import get_ep_group, get_pp_group, get_tp_group

    from vllm_ascend.models.glm5next.mega_kernel_step_events import V2DecodeStepHook

    if not config.decode_step_output_join_attested:
        raise ValueError("Decode timing requires verified output/communication stream joining")
    if runner.vllm_config.model_config.enforce_eager is not True:
        raise ValueError("Decode timing integration currently requires --enforce-eager")
    if not dist.is_initialized():
        raise RuntimeError("distributed group is required for Decode step matching")
    world = dist.get_world_size()
    members = list(range(world))
    if (
        get_pp_group().world_size != 1
        or dist.get_process_group_ranks(get_tp_group().device_group) != members
        or dist.get_process_group_ranks(get_ep_group().device_group) != members
    ):
        raise ValueError("Decode step matching currently requires PP1 and same ordered TP/EP world")
    if not group_agreed:
        control = dist.new_group(ranks=members, backend="gloo")
        control_device = torch.device("cpu")

        def agree_step(eligible, digest, local_error=None):
            return agree_decode_step(eligible, digest, control_device, control, local_error=local_error)

        gate.set_step_agreement(agree_step)
    hook = V2DecodeStepHook(
        type(runner),
        event_factory=lambda r: torch.npu.Event(enable_timing=True),
        stream_provider=lambda r: torch.npu.current_stream(r.device),
        rank_provider=lambda r: dist.get_rank(),
        step_id_provider=gate.step_id,
        sample_gate=lambda r: gate.is_pure_decode(),
        output_dependencies_joined_by_execute_return=True,
        target_requests=config.decode_step_target_requests,
        output_dir=config.decode_step_trace_dir,
        run_label=config.decode_step_trace_label,
    )
    hook.install()
    return hook
