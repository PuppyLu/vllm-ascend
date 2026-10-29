from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend import ascend_forward_context
from vllm_ascend.ops.fused_moe import zercmoe_adapter
from vllm_ascend.worker import worker as worker_module


@pytest.mark.parametrize(
    "method,args",
    [
        ("sleep", ()),
        ("wake_up", ()),
        ("init_weight_transfer_engine", ({},)),
        ("start_weight_update", ()),
        ("update_weights", ({},)),
        ("finish_weight_update", ()),
        ("update_config", ({},)),
        ("reload_weights", ()),
    ],
)
def test_zercmoe_single_weight_worker_rejects_unsupported_lifecycle_before_side_effects(monkeypatch, method, args):
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(zercmoe_single_weight=True))
    worker = object.__new__(worker_module.NPUWorker)
    with pytest.raises(RuntimeError, match="A3 ZercMoE single-weight"):
        getattr(worker, method)(*args)


def test_zercmoe_single_weight_worker_checks_configuration_before_loading(monkeypatch):
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(zercmoe_single_weight=True))
    validation = Mock(side_effect=ValueError("unsupported configuration"))
    monkeypatch.setattr(ascend_forward_context, "validate_zercmoe_single_weight_config", validation)
    worker = object.__new__(worker_module.NPUWorker)
    worker.vllm_config = object()
    worker.model_runner = SimpleNamespace(load_model=Mock())
    with pytest.raises(ValueError, match="unsupported configuration"):
        worker.load_model()
    validation.assert_called_once_with(worker.vllm_config)
    worker.model_runner.load_model.assert_not_called()
    assert not getattr(worker, "_zercmoe_single_weight_started", False)


def test_zercmoe_single_weight_worker_validates_loaded_layers_before_warmup(monkeypatch):
    config = SimpleNamespace(zercmoe_single_weight=True)
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(worker_module, "set_current_vllm_config", lambda _: nullcontext())
    events = []
    monkeypatch.setattr(
        ascend_forward_context, "validate_zercmoe_single_weight_config", lambda _: events.append("configuration")
    )
    monkeypatch.setattr(ascend_forward_context, "should_prepare_zercmoe_w8a8_weights", lambda _: False)
    model = object()

    def validate(loaded_model, *, require_single_weight):
        assert loaded_model is model and require_single_weight is True
        events.append("model")

    monkeypatch.setattr(zercmoe_adapter, "validate_w8a8_model", validate)
    monkeypatch.setattr(worker_module.torch.npu, "empty_cache", lambda: events.append("release_unused"), raising=False)
    worker = object.__new__(worker_module.NPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None
    )

    def load_model():
        events.append("load")
        config.zercmoe_single_weight = False

    worker.model_runner = SimpleNamespace(load_model=load_model, get_model=lambda: model)
    worker.load_model()
    assert events == ["configuration", "load", "model", "release_unused"]
    assert worker._zercmoe_single_weight_started is True
    config.zercmoe_single_weight = False
    for method, args in [("load_model", ()), ("reload_weights", ()), ("update_config", ({},)), ("sleep", ())]:
        with pytest.raises(RuntimeError, match="A3 ZercMoE single-weight"):
            getattr(worker, method)(*args)
    assert events == ["configuration", "load", "model", "release_unused"]


def test_zercmoe_single_weight_worker_does_not_release_cache_when_model_validation_fails(monkeypatch):
    config = SimpleNamespace(zercmoe_single_weight=True)
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(worker_module, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(ascend_forward_context, "validate_zercmoe_single_weight_config", lambda _: None)
    monkeypatch.setattr(ascend_forward_context, "should_prepare_zercmoe_w8a8_weights", lambda _: True)
    monkeypatch.setattr(zercmoe_adapter, "validate_w8a8_model", Mock(side_effect=ValueError("model invalid")))
    release = Mock()
    monkeypatch.setattr(worker_module.torch.npu, "empty_cache", release, raising=False)
    worker = object.__new__(worker_module.NPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None
    )
    worker.model_runner = SimpleNamespace(load_model=Mock(), get_model=lambda: object())
    with pytest.raises(ValueError, match="model invalid"):
        worker.load_model()
    release.assert_not_called()
    config.zercmoe_single_weight = False
    with pytest.raises(RuntimeError, match="A3 ZercMoE single-weight"):
        worker.load_model()
    with pytest.raises(RuntimeError, match="A3 ZercMoE single-weight"):
        worker.reload_weights()
    worker.model_runner.load_model.assert_called_once()


def test_zercmoe_single_weight_worker_validates_mtp_after_loading(monkeypatch):
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(zercmoe_single_weight=True))
    monkeypatch.setattr(worker_module, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(ascend_forward_context, "validate_zercmoe_single_weight_config", lambda _: None)
    events = []
    target, draft = object(), object()
    monkeypatch.setattr(zercmoe_adapter, "validate_w8a8_model", lambda model, **_: events.append(("target", model)))
    monkeypatch.setattr(zercmoe_adapter, "validate_single_weight_mtp", lambda t, d: events.append(("draft", t, d)))
    monkeypatch.setattr(worker_module.torch.npu, "empty_cache", lambda: events.append("release"), raising=False)
    worker = object.__new__(worker_module.NPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None, speculative_config=object()
    )
    worker.model_runner = SimpleNamespace(
        load_model=lambda: None, get_model=lambda: target, drafter=SimpleNamespace(model=draft)
    )
    worker.load_model()
    assert events == [("target", target), ("draft", target, draft), "release"]


def test_zercmoe_single_weight_disabled_preserves_default_worker_loading(monkeypatch):
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(zercmoe_single_weight=False))
    monkeypatch.setattr(worker_module, "set_current_vllm_config", lambda _: nullcontext())
    validation = Mock(side_effect=AssertionError("single-weight preflight"))
    monkeypatch.setattr(ascend_forward_context, "validate_zercmoe_single_weight_config", validation)
    monkeypatch.setattr(ascend_forward_context, "should_prepare_zercmoe_w8a8_weights", lambda _: True)
    model = object()
    check_model = Mock()
    monkeypatch.setattr(zercmoe_adapter, "validate_w8a8_model", check_model)
    release = Mock()
    monkeypatch.setattr(worker_module.torch.npu, "empty_cache", release, raising=False)
    worker = object.__new__(worker_module.NPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None
    )
    worker.model_runner = SimpleNamespace(load_model=Mock(), get_model=lambda: model)
    worker.load_model()
    validation.assert_not_called()
    check_model.assert_called_once_with(model, require_single_weight=False)
    release.assert_not_called()
    assert not getattr(worker, "_zercmoe_single_weight_started", False)
