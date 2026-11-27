# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM-Ascend ebaaeeec5 V2 scheduler evidence, opt-in per worker.

Usage, after imports and BEFORE worker/model initialization on every worker::

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    gate = V2SchedulerEvidence(NPUModelRunner)
    gate.install()
    adapter = DecodeLayerAdapter(backend, coordinate, gate.execution_mode, gate.is_pure_decode)
    # Install adapter on each supported actual layer, before execution.

This Python integration is eager-only, off by default, and has not been tested
on NPU; it does not implement MC2, ensure cross-rank agreement, or guarantee
that the model-layer adapter is executed. Hook once in every worker process.
Exact TP/EP admission and job-wide failure handling remain mandatory.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


class EvidenceUnavailable(ValueError):
    pass


@dataclass
class _Frame:
    owner: Any
    runner: Any
    scheduler_output: Any
    thread_id: int
    permitted_execute: bool
    prepare_calls: int = 0
    pure: bool = False
    reason: str = "unprepared"
    logical_step_id: str | None = None


_frame = contextvars.ContextVar("megakernel_v2_decode_evidence", default=None)


def _cpu_int_array(obj: Any, name: str, n: int) -> np.ndarray:
    v = getattr(obj, name, None)
    if not isinstance(v, np.ndarray) or v.ndim != 1 or v.size != n or v.dtype.kind not in "iu":
        raise EvidenceUnavailable(f"{name} must be an active-request CPU integer array")
    return v


def _cpu_bool_array(obj: Any, name: str, n: int, *, optional: bool = False) -> np.ndarray:
    v = getattr(obj, name, None)
    if v is None and optional:
        return np.zeros(n, dtype=np.bool_)
    if not isinstance(v, np.ndarray) or v.ndim != 1 or v.size != n or v.dtype.kind != "b":
        raise EvidenceUnavailable(f"{name} must be an active-request CPU boolean array")
    return v


def _int(obj: Any, name: str) -> int:
    x = getattr(obj, name, None)
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise EvidenceUnavailable(f"{name} is unavailable")
    return int(x)


def _classify(runner: Any, scheduler_output: Any, req_state: Any, desc: Any, batch: Any) -> str:
    """Return an opaque deterministic step key, or reject without inference.

    The key permits matching only after confirming TP/EP peers receive identical
    request IDs and computed-token counters; it is not a cross-rank handshake.
    NumPy fields are host metadata; no NPU tensor reads or synchronization.
    """
    if getattr(getattr(desc, "cg_mode", None), "name", None) != "NONE":
        raise EvidenceUnavailable("graph replay/capture is unsupported")
    config = getattr(runner, "vllm_config", None)
    model_config = getattr(config, "model_config", None)
    compilation = getattr(config, "compilation_config", None)
    if getattr(model_config, "enforce_eager", None) is not True:
        raise EvidenceUnavailable("service is not configured enforce_eager")
    if getattr(getattr(compilation, "cudagraph_mode", None), "name", None) != "NONE":
        raise EvidenceUnavailable("configured graph mode is not NONE")
    if getattr(runner, "pcp_manager", None) is not None:
        raise EvidenceUnavailable("PCP partition is unsupported")
    if getattr(batch, "is_dummy", None) is not False:
        raise EvidenceUnavailable("dummy or unknown batch")
    ids = getattr(req_state, "req_ids", None)
    batch_ids = getattr(batch, "req_ids", None)
    if not isinstance(ids, (list, tuple)) or not isinstance(batch_ids, (list, tuple)):
        raise EvidenceUnavailable("request IDs are unavailable")
    n = len(ids)
    if not n or not all(isinstance(s, str) and s for s in ids) or len(set(ids)) != n:
        raise EvidenceUnavailable("empty, invalid or duplicate request IDs")
    if list(batch_ids) != list(ids) or _int(batch, "num_reqs") != n:
        raise EvidenceUnavailable("batch request mapping changed")
    if _int(batch, "num_tokens") != n or _int(batch, "num_tokens_after_padding") != n:
        raise EvidenceUnavailable("padded or non-unit token count")
    counts = _cpu_int_array(req_state, "num_scheduled_tokens", n)
    if not np.all(counts == 1):
        raise EvidenceUnavailable("not exactly one scheduled token per active request")
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict) or any(scheduled.get(rid) != 1 for rid in ids):
        raise EvidenceUnavailable("scheduler request counts disagree")
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    if (
        not isinstance(spec, dict)
        or any(bool(tokens) for tokens in spec.values())
        or _int(batch, "num_draft_tokens") != 0
    ):
        raise EvidenceUnavailable("speculative or unknown scheduled tokens")
    # Do not trust GDN num_prefills: its builder can rewrite a final one-token
    # prefill as decode for scheduling purposes.
    if np.any(_cpu_bool_array(req_state, "is_prefilling_np", n)):
        raise EvidenceUnavailable("a request is still prefilling")
    if np.any(_cpu_bool_array(req_state, "prefill_runs_as_decode_np", n, optional=True)):
        raise EvidenceUnavailable("prefill disguised as decode")
    prefill_len = _cpu_int_array(req_state, "prefill_len_np", n)
    done_prefill = _cpu_int_array(req_state, "num_computed_prefill_tokens_np", n)
    if np.any(prefill_len < 0) or np.any(done_prefill < prefill_len):
        raise EvidenceUnavailable("prompt has not been fully processed")
    computed = _cpu_int_array(batch, "num_computed_tokens_np", n)
    if np.any(computed < 0):
        raise EvidenceUnavailable("invalid computed token count")
    payload = [[rid, int(computed[i]), int(counts[i])] for i, rid in enumerate(ids)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _default_capture_guard() -> bool:
    # True means execution is not tracing and the NPU stream is not captured.
    # Fail closed when either capability is unavailable or raises.
    try:
        import torch

        return (not torch.compiler.is_compiling()) and (not torch.npu.is_current_stream_capturing())
    except (AttributeError, RuntimeError):
        return False


class V2SchedulerEvidence:
    """Scoped evidence of actual vLLM scheduler phases for one model execution."""

    def __init__(self, runner_class: type, *, capture_guard: Callable[[], bool] = _default_capture_guard):
        if not callable(capture_guard):
            raise TypeError("capture_guard must be callable")
        self.runner_class = runner_class
        self.capture_guard = capture_guard
        self._originals: dict[str, Any] = {}
        self._wrappers: dict[str, Any] = {}
        self._installed = False
        self._step_agreement = None
        self._execute_lock = threading.Lock()

    def set_step_agreement(self, agreement: Callable[[bool, str | None, Exception | None], bool]) -> None:
        if self._step_agreement is not None or not callable(agreement):
            raise RuntimeError("a single group-wide step agreement is required")
        self._step_agreement = agreement

    def _current(self) -> _Frame | None:
        frame = _frame.get()
        if (
            not isinstance(frame, _Frame)
            or frame.owner is not self
            or frame.thread_id != threading.get_ident()
            or not frame.pure
        ):
            return None
        # Capture status was checked before the group step agreement; do not
        # let an independent per-rank recheck change an agreed decision.
        return frame

    def is_pure_decode(self, layer: Any = None) -> bool:
        del layer
        return self._current() is not None

    def execution_mode(self) -> str:
        return "eager" if self._current() is not None else "unsupported"

    def current_step_digest(self) -> str:
        frame = self._current()
        if frame is None or frame.logical_step_id is None:
            raise EvidenceUnavailable("step digest requested outside a verified Decode execution")
        return frame.logical_step_id

    def step_id(self, runner: Any, scheduler_output: Any, batch: Any) -> str:
        del batch
        frame = self._current()
        if frame is None or frame.runner is not runner or frame.scheduler_output is not scheduler_output:
            raise EvidenceUnavailable("step key requested outside the matching model execution")
        if frame.logical_step_id is None:
            raise EvidenceUnavailable("step key is missing")
        return frame.logical_step_id

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("already installed")
        cls = self.runner_class
        execute = getattr(cls, "execute_model", None)
        prepare = getattr(cls, "prepare_inputs", None)
        if not callable(execute) or not callable(prepare):
            raise RuntimeError("runner lacks the V2 execution interface")
        execute_sig, prepare_sig = inspect.signature(execute), inspect.signature(prepare)
        if not {"scheduler_output", "dummy_run", "is_profile"}.issubset(execute_sig.parameters):
            raise RuntimeError("unsupported execute_model interface")
        if not {"scheduler_output", "batch_req_state", "batch_desc"}.issubset(prepare_sig.parameters):
            raise RuntimeError("unsupported prepare_inputs interface")
        gate = self

        @functools.wraps(execute)
        def execute_model(runner: Any, *args: Any, **kwargs: Any):
            bound = execute_sig.bind(runner, *args, **kwargs)
            bound.apply_defaults()
            if _frame.get() is not None or not gate._execute_lock.acquire(blocking=False):
                raise RuntimeError("concurrent or nested V2 execution is unsupported; abort the job")
            frame = _Frame(
                gate,
                runner,
                bound.arguments["scheduler_output"],
                threading.get_ident(),
                bound.arguments["dummy_run"] is False and bound.arguments["is_profile"] is False,
            )
            token = _frame.set(frame)
            try:
                return execute(runner, *args, **kwargs)
            finally:
                _frame.reset(token)
                gate._execute_lock.release()

        @functools.wraps(prepare)
        def prepare_inputs(runner: Any, *args: Any, **kwargs: Any):
            frame = _frame.get()
            if (
                not isinstance(frame, _Frame)
                or frame.owner is not gate
                or frame.runner is not runner
                or frame.thread_id != threading.get_ident()
            ):
                return prepare(runner, *args, **kwargs)
            frame.prepare_calls += 1
            frame.pure = False
            if not frame.permitted_execute:
                frame.reason = "dummy or profile"
                return prepare(runner, *args, **kwargs)
            local_error = None
            batch = None
            try:
                bound = prepare_sig.bind(runner, *args, **kwargs)
                bound.apply_defaults()
                batch = prepare(runner, *args, **kwargs)
                if frame.prepare_calls != 1:
                    raise RuntimeError("multiple prepare calls in one execution are unsupported")
            except Exception as exc:
                local_error = exc
                frame.reason = "step_preparation_failed"
            if local_error is None:
                try:
                    if bound.arguments["scheduler_output"] is not frame.scheduler_output:
                        frame.reason = "scheduler object changed"
                    else:
                        frame.logical_step_id = _classify(
                            runner,
                            frame.scheduler_output,
                            bound.arguments["batch_req_state"],
                            bound.arguments["batch_desc"],
                            batch,
                        )
                        frame.pure = gate.capture_guard() is True
                        frame.reason = "pure_decode" if frame.pure else "capture_guard_failed"
                except EvidenceUnavailable as exc:
                    frame.reason = str(exc)
                except Exception as exc:
                    local_error = exc
                    frame.reason = "step_preparation_failed"
            if gate._step_agreement is not None:
                # All ranks must execute the same ordinary model step here.
                # Dummy/profile runs never enter this collective. A transport
                # failure must be handled by the distributed job watchdog.
                accepted = gate._step_agreement(frame.pure, frame.logical_step_id, local_error)
                if type(accepted) is not bool:
                    raise RuntimeError("step agreement did not return a boolean")
                frame.pure = frame.pure and accepted
                if not accepted:
                    frame.reason = "group_decode_contract_not_met"
            if local_error is not None:
                raise local_error
            return batch

        self._originals = {"execute_model": execute, "prepare_inputs": prepare}
        self._wrappers = {"execute_model": execute_model, "prepare_inputs": prepare_inputs}
        for name, wrapper in self._wrappers.items():
            setattr(cls, name, wrapper)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            raise RuntimeError("not installed")
        if self._execute_lock.locked():
            raise RuntimeError("cannot uninstall the Decode phase hook during a model step")
        for name, wrapper in self._wrappers.items():
            if getattr(self.runner_class, name) is not wrapper:
                raise RuntimeError("runner hook changed externally; refusing to overwrite it")
        for name, original in self._originals.items():
            setattr(self.runner_class, name, original)
        self._installed = False
