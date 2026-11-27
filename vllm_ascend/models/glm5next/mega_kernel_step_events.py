# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred eager-Decode device-event tracing for the verified V2 scheduler.

Samples describe a single rank's event stream after input preparation through
hidden-state return. They exclude logits, sampling and the preceding admission;
they are not end-to-end latency or cross-device makespan. Event queries never
synchronize. The caller must verify that output dependencies join the stream.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import threading
import weakref
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BOUNDARY = "prepare_inputs_return_to_execute_model_return_hidden_state_excludes_logits"


class DecodeTraceError(RuntimeError):
    """The runner or event contract does not match the instrumentation."""


class V2DecodeStepHook:
    """Measure only steps accepted by the framework's verified Decode gate."""

    def __init__(
        self,
        runner_class,
        *,
        event_factory,
        stream_provider,
        rank_provider,
        step_id_provider,
        output_dependencies_joined_by_execute_return,
        target_requests=None,
        sample_limit=256,
        max_records=2048,
        output_dir=None,
        run_label="",
        sample_gate=None,
    ):
        if output_dependencies_joined_by_execute_return is not True:
            raise DecodeTraceError("output/communication stream join must be attested")
        if any(not callable(x) for x in (event_factory, stream_provider, rank_provider, step_id_provider, sample_gate)):
            raise DecodeTraceError("event, stream, rank, step and verified Decode gate are required")
        if target_requests is not None and (type(target_requests) is not int or target_requests <= 0):
            raise DecodeTraceError("target request count must be positive")
        if any(type(x) is not int or x <= 0 for x in (sample_limit, max_records)):
            raise DecodeTraceError("record capacities must be positive")
        if output_dir is not None and (
            not isinstance(run_label, str) or not run_label or "/" in run_label or "\\" in run_label
        ):
            raise DecodeTraceError("a safe nonempty trace label is required")
        self.cls = runner_class
        self.events = event_factory
        self.stream = stream_provider
        self.rank = rank_provider
        self.step = step_id_provider
        self.gate = sample_gate
        self.target = target_requests
        self.sample_limit = sample_limit
        self.max_records = max_records
        self.path = Path(output_dir) if output_dir is not None else None
        self.label = run_label
        self._runners = weakref.WeakKeyDictionary()
        self._installed = False
        self._lock = threading.Lock()
        self._old = {}
        self._new = {}

    def _state(self, runner):
        state = self._runners.get(runner)
        if state is None:
            state = SimpleNamespace(active=None, pending=deque(), samples=[], skipped=Counter(), dumped=False)
            self._runners[runner] = state
        return state

    def install(self):
        if self._installed:
            raise DecodeTraceError("Decode event hook already installed")
        execute = getattr(self.cls, "execute_model", None)
        prepare = getattr(self.cls, "prepare_inputs", None)
        if not callable(execute) or not callable(prepare):
            raise DecodeTraceError("runner lacks the V2 execution methods")
        esig, psig = inspect.signature(execute), inspect.signature(prepare)
        if not {"scheduler_output", "dummy_run", "is_profile"}.issubset(esig.parameters):
            raise DecodeTraceError("unsupported execute_model signature")
        if not {"scheduler_output", "batch_req_state", "batch_desc"}.issubset(psig.parameters):
            raise DecodeTraceError("unsupported prepare_inputs signature")
        owner = self

        @functools.wraps(execute)
        def run(runner, *args, **kwargs):
            params = esig.bind(runner, *args, **kwargs)
            params.apply_defaults()
            if not owner._lock.acquire(blocking=False):
                raise DecodeTraceError("concurrent Decode trace execution is unsupported")
            try:
                state = owner._state(runner)
            except BaseException:
                owner._lock.release()
                raise
            dummy, profile = params.arguments["dummy_run"], params.arguments["is_profile"]
            if type(dummy) is not bool or type(profile) is not bool:
                owner._lock.release()
                raise DecodeTraceError("dummy and profile flags must be bool")
            ctx = {"eligible": not (dummy or profile), "batch": None, "ticket": None, "calls": 0}
            state.active = ctx
            if not ctx["eligible"]:
                state.skipped["dummy_or_profile"] += 1
            try:
                result = execute(runner, *args, **kwargs)
                ticket = ctx["ticket"]
                if ticket is not None:
                    execute_state = getattr(runner, "execute_model_state", None)
                    if (
                        execute_state is None
                        or getattr(execute_state, "input_batch", None) is not ctx["batch"]
                        or getattr(execute_state, "hidden_states", None) is None
                    ):
                        state.skipped["model_error"] += 1
                        state.skipped["unverified_forward_output"] += 1
                    else:
                        ticket.end.record(ticket.stream)
                        state.pending.append(ticket)
                    ctx["ticket"] = None
                owner.collect_ready(runner)
                return result
            except BaseException:
                if ctx["ticket"] is not None:
                    state.skipped["model_error"] += 1
                    ctx["ticket"] = None
                raise
            finally:
                state.active = None
                owner._lock.release()

        @functools.wraps(prepare)
        def prep(runner, *args, **kwargs):
            params = psig.bind(runner, *args, **kwargs)
            params.apply_defaults()
            batch = prepare(runner, *args, **kwargs)
            state = owner._state(runner)
            ctx = state.active
            if ctx is None or not ctx["eligible"]:
                return batch
            ctx["calls"] += 1
            if ctx["calls"] != 1:
                raise DecodeTraceError("multiple preparation calls in one step are unsupported")
            # The scheduler gate checks CPU request order, counts, prefill,
            # speculation, eager mode, and rank agreement before returning.
            if owner.gate(runner) is not True:
                state.skipped["group_step_rejected"] += 1
                return batch
            n = getattr(batch, "num_reqs", None)
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0:
                raise DecodeTraceError("verified Decode count is unavailable")
            n = int(n)
            # The runner gate validates request-state metadata; retain the
            # independent input-batch phase check for disguised prefill.
            for field, optional in (("is_prefilling_np", False), ("prefill_runs_as_decode_np", True)):
                if not hasattr(batch, field):
                    raise DecodeTraceError(f"missing CPU batch phase array: {field}")
                value = getattr(batch, field)
                if value is None and optional:
                    continue
                if not isinstance(value, np.ndarray) or value.ndim != 1 or value.size != n or value.dtype.kind != "b":
                    raise DecodeTraceError(f"invalid CPU batch phase array: {field}")
                if np.any(value):
                    state.skipped["prefill_or_mixed"] += 1
                    return batch
            if owner.target is not None and n != owner.target:
                state.skipped["decode_count_mismatch"] += 1
                return batch
            if len(state.samples) + len(state.pending) >= owner.max_records:
                state.skipped["record_capacity"] += 1
                return batch
            if len(state.pending) >= owner.sample_limit:
                state.skipped["event_capacity"] += 1
                return batch
            rank = owner.rank(runner)
            key = owner.step(runner, params.arguments["scheduler_output"], batch)
            if type(rank) is not int or rank < 0 or not isinstance(key, str) or not key:
                raise DecodeTraceError("rank or shared step id is invalid")
            stream = owner.stream(runner)
            start, end = owner.events(runner), owner.events(runner)
            start.record(stream)
            ctx["batch"] = batch
            ctx["ticket"] = SimpleNamespace(start=start, end=end, stream=stream, rank=rank, step=key, count=n)
            return batch

        self._old = {"execute_model": execute, "prepare_inputs": prepare}
        self._new = {"execute_model": run, "prepare_inputs": prep}
        for name, fn in self._new.items():
            setattr(self.cls, name, fn)
        self._installed = True

    def uninstall(self):
        if not self._installed:
            raise DecodeTraceError("Decode event hook not installed")
        if self._lock.locked():
            raise DecodeTraceError("cannot uninstall the Decode event hook during a model step")
        if any(getattr(self.cls, name) is not fn for name, fn in self._new.items()):
            raise DecodeTraceError("runner hooks were replaced by another component")
        for name, fn in self._old.items():
            setattr(self.cls, name, fn)
        self._installed = False

    def collect_ready(self, runner):
        state = self._state(runner)
        records = []
        while state.pending and state.pending[0].end.query():
            ticket = state.pending.popleft()
            ms = float(ticket.start.elapsed_time(ticket.end))
            if not math.isfinite(ms) or ms < 0:
                raise DecodeTraceError("invalid device event duration")
            records.append(
                {
                    "step_id": ticket.step,
                    "rank": ticket.rank,
                    "decode_requests": ticket.count,
                    "graph_mode": "eager",
                    "elapsed_ms": ms,
                    "boundary": BOUNDARY,
                }
            )
        state.samples.extend(records)
        return records

    def snapshot(self, runner):
        state = self._state(runner)
        if state.active is not None:
            raise DecodeTraceError("cannot snapshot an active model execution")
        self.collect_ready(runner)
        rank = self.rank(runner)
        if type(rank) is not int or rank < 0:
            raise DecodeTraceError("rank is invalid")
        return {
            "schema": "megakernel.vllm_v2_decode_step.v1",
            "run_label": self.label,
            "rank": rank,
            "target_requests": self.target,
            "boundary": BOUNDARY,
            "output_dependency_attested_not_verified": True,
            "includes_logits": False,
            "global_makespan": False,
            "pending_events": len(state.pending),
            "skipped": dict(state.skipped),
            "samples": list(state.samples),
        }

    def dump(self, runner):
        state = self._state(runner)
        if state.dumped:
            raise DecodeTraceError("trace already dumped")
        if self.path is None or not self.path.is_dir():
            raise DecodeTraceError("trace output directory must exist")
        data = self.snapshot(runner)
        path = self.path / f"{self.label}-rank{data['rank']}.json"
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        state.dumped = True
        return path
