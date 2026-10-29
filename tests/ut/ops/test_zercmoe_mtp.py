from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe.zercmoe_adapter import validate_single_weight_mtp
from vllm_ascend.zercmoe_config import (
    is_zercmoe_draft_model_loading,
    load_zercmoe_draft_model,
    zercmoe_draft_model_loading,
)


def test_mtp_loading_role_restores_on_nested_exception():
    assert not is_zercmoe_draft_model_loading()
    with pytest.raises(ValueError), zercmoe_draft_model_loading():
        assert is_zercmoe_draft_model_loading()
        with zercmoe_draft_model_loading():
            assert is_zercmoe_draft_model_loading()
        assert is_zercmoe_draft_model_loading()
        raise ValueError("loading failed")
    assert not is_zercmoe_draft_model_loading()


@pytest.mark.parametrize("fail", [False, True])
def test_mtp_drafter_load_runs_inside_role_and_restores_it(fail):
    target = object()
    events = []

    def load_model(model):
        events.append((model, is_zercmoe_draft_model_loading()))
        if fail:
            raise ValueError("load failed")

    drafter = SimpleNamespace(load_model=load_model)
    if fail:
        with pytest.raises(ValueError, match="load failed"):
            load_zercmoe_draft_model(drafter, target)
    else:
        load_zercmoe_draft_model(drafter, target)
    assert events == [(target, True)]
    assert not is_zercmoe_draft_model_loading()


def test_mtp_comm_registry_keeps_target_and_draft_separate(monkeypatch):
    class FakeComm:
        def __init__(self, config):
            self.config = config
            self.moe_config = config
            self.initialized = False
            self.closed = False

        def initialize_w8a8_buffer(self):
            self.initialized = True

        def close(self):
            self.closed = True

    monkeypatch.setattr(comm, "_MoECommMethods", {})
    monkeypatch.setattr(comm, "_DraftMoECommMethods", {})
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(zercmoe_single_weight=True))
    for name in ("AlltoAllCommImpl", "AllGatherCommImpl", "MC2CommImpl", "FusedMC2CommImpl", "ZercMoECommImpl"):
        monkeypatch.setattr(comm, name, FakeComm)
    target, draft = SimpleNamespace(ep_size=8, model="target"), SimpleNamespace(ep_size=8, model="draft")
    comm.setup_moe_comm_method(target)
    original = comm.get_moe_comm_method(MoECommType.ZERC_MOE)
    with zercmoe_draft_model_loading():
        comm.setup_moe_comm_method(draft)
    assert comm.get_moe_comm_method(MoECommType.ZERC_MOE) is original
    assert comm.get_moe_comm_method(MoECommType.ZERC_MOE, is_draft_model=True) is None
    assert comm.get_moe_comm_method(MoECommType.MC2).config is target
    assert comm.get_moe_comm_method(MoECommType.MC2, is_draft_model=True).config is draft
    with zercmoe_draft_model_loading(), pytest.raises(ValueError, match="consistent MoE"):
        comm.setup_moe_comm_method(SimpleNamespace(ep_size=4, model="draft"))
    comm.initialize_zercmoe_buffer()
    comm.shutdown_zercmoe()
    assert original.initialized and original.closed


def _model(experts):
    model = torch.nn.Module()
    model.moe = torch.nn.Module()
    model.moe.routed_experts = experts
    return model


def _target_and_draft():
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.w13_weight_zr = torch.arange(4)
    target.w2_weight_zr = torch.arange(4)
    target._zercmoe_weights_only = True
    draft.w13_weight = torch.ones(4)
    draft.w2_weight = torch.ones(4)
    return target, draft


def test_mtp_experts_can_use_independent_legacy_weights():
    target, draft = _target_and_draft()
    validate_single_weight_mtp(_model(target), _model(draft))


@pytest.mark.parametrize(
    "invalid", ["same_layer", "aliased_storage", "zerc_packed", "missing_weights", "invalid_weights", "no_moe"]
)
def test_mtp_experts_reject_shared_or_incomplete_weights(invalid):
    target, draft = _target_and_draft()
    if invalid == "same_layer":
        draft = target
    elif invalid == "aliased_storage":
        draft.w13_weight = target.w13_weight_zr.view(-1)
    elif invalid == "zerc_packed":
        draft.w13_weight_zr = torch.ones(4)
    elif invalid == "missing_weights":
        del draft.w2_weight
    elif invalid == "invalid_weights":
        draft.w2_weight = [object()]
    draft_model = torch.nn.Module() if invalid == "no_moe" else _model(draft)
    with pytest.raises(ValueError, match="A3 ZercMoE single-weight"):
        validate_single_weight_mtp(_model(target), draft_model)
