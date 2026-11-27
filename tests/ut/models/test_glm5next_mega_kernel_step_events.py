# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Event tracing records only verified decode work and never synchronizes."""

import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace as NS

import numpy as np

_PATH = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_step_events.py"
_SPEC = importlib.util.spec_from_file_location("_step_events_under_test", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class Stream:
    def __init__(self):
        self.time = 0.0
        self.ready = False


class Event:
    def record(self, stream):
        self.stream = stream
        self.time = stream.time

    def query(self):
        return self.stream.ready

    def elapsed_time(self, end):
        return end.time - self.time


class Runner:
    def __init__(self):
        self.stream = Stream()
        self.batch = NS(num_reqs=64, is_prefilling_np=np.zeros(64, dtype=bool), prefill_runs_as_decode_np=None)
        self.state = None
        self.desc = None
        self.execute_model_state = None
        self.raise_after_prepare = False
        self.match_output = True
        self.duplicate_prepare = False

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        self.stream.time += 7
        return self.batch

    def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
        batch = self.prepare_inputs(scheduler_output, self.state, self.desc)
        if self.duplicate_prepare:
            self.prepare_inputs(scheduler_output, self.state, self.desc)
        if self.raise_after_prepare:
            raise ValueError("model failed")
        self.stream.time += 3
        self.execute_model_state = NS(input_batch=batch if self.match_output else object(), hidden_states=object())
        return "result"


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.r = Runner()
        self.h = _MOD.V2DecodeStepHook(
            Runner,
            event_factory=lambda r: Event(),
            stream_provider=lambda r: r.stream,
            rank_provider=lambda r: 0,
            step_id_provider=lambda r, s, b: s.key,
            output_dependencies_joined_by_execute_return=True,
            sample_gate=lambda r: True,
            target_requests=64,
        )
        self.h.install()

    def tearDown(self):
        self.h.uninstall()

    def test_deferred_event_boundary_and_schema(self):
        self.assertEqual(self.r.execute_model(NS(key="step")), "result")
        self.assertEqual(self.h.snapshot(self.r)["pending_events"], 1)
        self.r.stream.ready = True
        out = self.h.snapshot(self.r)
        self.assertEqual(out["pending_events"], 0)
        self.assertEqual(out["samples"][0]["elapsed_ms"], 3)
        self.assertEqual(out["samples"][0]["graph_mode"], "eager")
        self.assertEqual(out["samples"][0]["boundary"], _MOD.BOUNDARY)
        self.assertFalse(out["includes_logits"])
        self.assertFalse(out["global_makespan"])

    def test_rejected_and_mixed_are_not_timed(self):
        self.h.gate = lambda _: False
        self.h.step = lambda *_: self.fail("must not access a rejected step")
        self.r.execute_model(NS())
        self.assertEqual(self.h.snapshot(self.r)["skipped"]["group_step_rejected"], 1)
        self.h.gate = lambda _: True
        self.r.batch.prefill_runs_as_decode_np = np.ones(64, dtype=bool)
        self.r.execute_model(NS())
        self.assertEqual(self.h.snapshot(self.r)["skipped"]["prefill_or_mixed"], 1)
        self.assertEqual(self.h.snapshot(self.r)["pending_events"], 0)

    def test_failure_and_unverified_return_drop_ticket(self):
        self.r.raise_after_prepare = True
        with self.assertRaisesRegex(ValueError, "model failed"):
            self.r.execute_model(NS(key="x"))
        self.assertEqual(self.h.snapshot(self.r)["pending_events"], 0)
        self.r.raise_after_prepare = False
        self.r.match_output = False
        self.r.execute_model(NS(key="y"))
        snapshot = self.h.snapshot(self.r)
        self.assertEqual(snapshot["skipped"]["unverified_forward_output"], 1)
        self.assertEqual(snapshot["skipped"]["model_error"], 2)

    def test_dummy_and_profile_not_sampled(self):
        self.r.execute_model(NS(), dummy_run=True)
        self.r.execute_model(NS(), is_profile=True)
        self.assertEqual(self.h.snapshot(self.r)["skipped"]["dummy_or_profile"], 2)

    def test_repeated_prepare_fails_closed(self):
        self.r.duplicate_prepare = True
        with self.assertRaises(_MOD.DecodeTraceError):
            self.r.execute_model(NS(key="a"))
        self.assertEqual(self.h.snapshot(self.r)["pending_events"], 0)

    def test_target_capacity_and_dump_does_not_overwrite(self):
        self.h.sample_limit = 1
        self.r.execute_model(NS(key="a"))
        self.r.execute_model(NS(key="b"))
        self.assertEqual(self.h.snapshot(self.r)["skipped"]["event_capacity"], 1)
        with tempfile.TemporaryDirectory() as d:
            self.h.path = pathlib.Path(d)
            self.h.label = "x"
            path = self.h.dump(self.r)
            self.assertTrue(path.exists())
            with self.assertRaises(_MOD.DecodeTraceError):
                self.h.dump(self.r)

    def test_join_attestation_required(self):
        with self.assertRaises(_MOD.DecodeTraceError):
            _MOD.V2DecodeStepHook(
                Runner,
                event_factory=Event,
                stream_provider=lambda r: r.stream,
                rank_provider=lambda r: 0,
                step_id_provider=lambda *_: "x",
                sample_gate=lambda _: True,
                output_dependencies_joined_by_execute_return=False,
            )

    def test_cannot_uninstall_during_model_step(self):
        self.assertTrue(self.h._lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(_MOD.DecodeTraceError, "during a model step"):
                self.h.uninstall()
        finally:
            self.h._lock.release()


class SchedulerHookIntegrationTests(unittest.TestCase):
    def test_gate_and_trace_wrapper_order(self):
        import sys

        source = _PATH.with_name("mega_kernel_scheduler.py")
        spec = importlib.util.spec_from_file_location("_scheduler_under_test", source)
        scheduler = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scheduler
        spec.loader.exec_module(scheduler)
        runner = Runner()
        n = 64
        ids = [str(i) for i in range(n)]
        ones = np.ones(n, dtype=np.int64)
        runner.vllm_config = NS(
            model_config=NS(enforce_eager=True), compilation_config=NS(cudagraph_mode=NS(name="NONE"))
        )
        runner.pcp_manager = None
        runner.state = NS(
            req_ids=ids,
            num_scheduled_tokens=ones,
            is_prefilling_np=np.zeros(n, dtype=bool),
            prefill_runs_as_decode_np=None,
            prefill_len_np=ones,
            num_computed_prefill_tokens_np=ones,
        )
        runner.batch = NS(
            req_ids=ids,
            num_reqs=n,
            num_tokens=n,
            num_tokens_after_padding=n,
            num_draft_tokens=0,
            is_dummy=False,
            is_prefilling_np=np.zeros(n, dtype=bool),
            prefill_runs_as_decode_np=None,
            num_computed_tokens_np=ones,
        )
        desc = NS(cg_mode=NS(name="NONE"))
        runner.desc = desc
        gate = scheduler.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        gate.set_step_agreement(lambda accepted, digest: accepted and isinstance(digest, str))
        gate.install()
        hook = _MOD.V2DecodeStepHook(
            Runner,
            event_factory=lambda r: Event(),
            stream_provider=lambda r: r.stream,
            rank_provider=lambda r: 0,
            step_id_provider=gate.step_id,
            sample_gate=lambda r: gate.is_pure_decode(),
            output_dependencies_joined_by_execute_return=True,
            target_requests=n,
        )
        try:
            hook.install()
            runner.execute_model(NS(num_scheduled_tokens={i: 1 for i in ids}, scheduled_spec_decode_tokens={}))
            runner.stream.ready = True
            self.assertEqual(len(hook.snapshot(runner)["samples"]), 1)
            self.assertEqual(hook.snapshot(runner)["samples"][0]["elapsed_ms"], 3)
        finally:
            hook.uninstall()
            gate.uninstall()


if __name__ == "__main__":
    unittest.main()
