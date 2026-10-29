from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import fused_moe
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, FusedMoEResult
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.zercmoe_config import (
    allow_zercmoe_shared_expert_overlap,
    use_shared_expert_overlap,
    zercmoe_draft_model_loading,
)


@pytest.mark.parametrize(
    "enabled,single,is_a3,expected",
    [(1, True, True, True), (0, True, True, False), (1, False, True, False), (1, True, False, False)],
)
def test_zercmoe_shared_expert_config_only_exempts_a3_single_weight(enabled, single, is_a3, expected):
    assert allow_zercmoe_shared_expert_overlap(enabled=enabled, single_weight=single, is_a3=is_a3) is expected


def test_zercmoe_shared_expert_draft_does_not_use_target_multistream():
    assert use_shared_expert_overlap(configured=True, has_shared_experts=True)
    with zercmoe_draft_model_loading():
        assert not use_shared_expert_overlap(configured=True, has_shared_experts=True)
    assert not use_shared_expert_overlap(configured=True, has_shared_experts=False)
    assert not use_shared_expert_overlap(configured=False, has_shared_experts=True)


@pytest.mark.parametrize("quantized,internal_router", [(False, False), (True, False), (True, True)])
def test_zercmoe_shared_stream_forks_and_joins_before_outputs(monkeypatch, quantized, internal_router):
    events = []
    state = SimpleNamespace(current="main")

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            events.append((self.name, "record"))
            return "input_ready" if sum(event == (self.name, "record") for event in events) == 1 else "router_ready"

        def wait_event(self, event):
            events.append((self.name, "wait_event", event))

        def wait_stream(self, stream):
            events.append((self.name, "wait_stream", stream.name))

    main, shared_stream = Stream("main"), Stream("shared")

    @contextmanager
    def switch(stream, enabled):
        assert enabled and stream is shared_stream
        state.current = "shared"
        try:
            yield
        finally:
            state.current = "main"

    monkeypatch.setattr(fused_moe, "npu_stream_switch", switch)
    monkeypatch.setattr(fused_moe, "shared_experts_calculation_stream", lambda: shared_stream)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: main if state.current == "main" else shared_stream)
    gate = SimpleNamespace(weight_scale=torch.ones(1), weight_scale_fp32=torch.ones(1)) if quantized else object()
    down = SimpleNamespace(weight_scale=torch.ones(1)) if quantized else object()
    if quantized:
        gate.weight = torch.ones(1)
        down.weight = torch.ones(1)
    shared = SimpleNamespace(gate_up_proj=gate, down_proj=down, act_fn=object())
    runner = SimpleNamespace(
        is_internal_router=internal_router,
        gate=SimpleNamespace(weight_fp32=torch.ones(4, 2)) if internal_router else None,
        _shared_experts=shared,
        multistream_overlap_shared_expert=True,
        quant_type=QuantType.W8A8,
        _prepare_shared_expert_input=lambda x: x,
        _finalize_shared_expert_output=lambda x: x,
    )
    runner._shared_experts_part1 = lambda x: events.append((state.current, "part1")) or x + 1
    runner._shared_experts_part2 = lambda x, part: events.append((state.current, "part2")) or part + 1

    def routed(x, logits, return_with_event):
        assert return_with_event
        events.append((state.current, "routed"))
        return FusedMoEResult(routed_out=x + 3, before_dispatch_evt=None if quantized else "dispatch_ready")

    runner.no_shared_forward_impl = routed
    runner._forward_shared_experts = lambda x, evts: AscendMoERunner._forward_shared_experts(runner, x, evts)
    if quantized:

        def dynamic_quant(x):
            events.append((state.current, "quant"))
            return x, torch.ones(1)

        def matmul(x, weight, scale, **kwargs):
            events.append((state.current, "matmul"))
            return x + 1

        def swiglu(**kwargs):
            events.append((state.current, "swiglu"))
            return kwargs["x"], torch.ones(1)

        monkeypatch.setattr(fused_moe.torch_npu, "npu_dynamic_quant", dynamic_quant, raising=False)
        monkeypatch.setattr(fused_moe.torch_npu, "npu_quant_matmul", matmul, raising=False)
        monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(npu_dequant_swiglu_quant=swiglu), raising=False)

    x = torch.ones(2, 2)
    shared_out, routed_out = AscendMoERunner.shared_forward_impl(runner, x, torch.ones(2, 4))
    torch.testing.assert_close(routed_out, x + 3)
    torch.testing.assert_close(shared_out, x + 2)
    assert events[0] == ("main", "record")
    if internal_router:
        assert events[1:4] == [("main", "record"), ("main", "routed"), ("shared", "wait_event", "input_ready")]
        assert ("shared", "wait_event", "router_ready") in events
    else:
        assert events[1:3] == [("main", "routed"), ("shared", "wait_event", "input_ready")]
    assert events[-1] == ("main", "wait_stream", "shared")
    if quantized:
        assert ("shared", "quant") in events and ("shared", "swiglu") in events
    else:
        assert ("shared", "wait_event", "dispatch_ready") in events
        assert ("shared", "part1") in events and ("shared", "part2") in events
