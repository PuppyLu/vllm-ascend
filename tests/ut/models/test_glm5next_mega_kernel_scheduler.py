# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_scheduler.py"
spec = importlib.util.spec_from_file_location("mega_kernel_scheduler_cpu_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["request-a", "request-b"]
        self.req = NS(
            req_ids=self.ids,
            num_scheduled_tokens=np.array([1, 1]),
            is_prefilling_np=np.zeros(2, dtype=bool),
            prefill_runs_as_decode_np=None,
            prefill_len_np=np.array([5, 7]),
            num_computed_prefill_tokens_np=np.array([5, 7]),
        )
        self.batch = NS(
            req_ids=self.ids,
            is_dummy=False,
            num_reqs=2,
            num_tokens=2,
            num_tokens_after_padding=2,
            num_draft_tokens=0,
            num_computed_tokens_np=np.array([9, 12]),
        )
        self.desc = NS(cg_mode=NS(name="NONE"))
        self.sched = NS(num_scheduled_tokens=dict.fromkeys(self.ids, 1), scheduled_spec_decode_tokens={})
        self.runner = NS(
            pcp_manager=None,
            vllm_config=NS(model_config=NS(enforce_eager=True), compilation_config=NS(cudagraph_mode=NS(name="NONE"))),
        )

    def classify(self):
        return module._classify(self.runner, self.sched, self.req, self.desc, self.batch)

    def test_one_token_decode_and_stable_step_identity(self):
        first = self.classify()
        self.assertEqual(first, self.classify())
        self.batch.num_computed_tokens_np[0] += 1
        self.assertNotEqual(first, self.classify())

    def test_prompt_tail_mixed_and_graph_rejected(self):
        self.req.num_computed_prefill_tokens_np[0] = 4
        with self.assertRaises(module.EvidenceUnavailable):
            self.classify()
        self.req.num_computed_prefill_tokens_np[0] = 5
        self.req.prefill_runs_as_decode_np = np.array([True, False])
        with self.assertRaises(module.EvidenceUnavailable):
            self.classify()
        self.req.prefill_runs_as_decode_np = None
        self.desc.cg_mode.name = "FULL"
        with self.assertRaises(module.EvidenceUnavailable):
            self.classify()

    def test_concurrent_model_execution_fails_fast(self):
        started, release = threading.Event(), threading.Event()

        class Runner:
            def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
                return None

            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                started.set()
                if not release.wait(2):
                    raise RuntimeError("test timed out")

        gate = module.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        gate.install()
        runner = Runner()
        thread = threading.Thread(target=runner.execute_model, args=(self.sched,))
        thread.start()
        try:
            self.assertTrue(started.wait(2))
            with self.assertRaisesRegex(RuntimeError, "concurrent or nested"):
                runner.execute_model(self.sched)
        finally:
            release.set()
            thread.join(2)
            gate.uninstall()
        self.assertFalse(thread.is_alive())

    def test_every_real_step_agrees_once_and_profile_does_not_vote(self):
        case = self

        class Runner:
            pcp_manager = None
            vllm_config = case.runner.vllm_config

            def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
                return case.batch

            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                self.prepare_inputs(scheduler_output, case.req, case.desc, 0)
                return gate.is_pure_decode()

        gate = module.V2SchedulerEvidence(Runner, capture_guard=lambda: True)
        votes = []
        gate.set_step_agreement(lambda accepted, digest, error: votes.append(accepted) or accepted)
        gate.install()
        try:
            runner = Runner()
            self.assertTrue(runner.execute_model(self.sched))
            self.assertFalse(gate.is_pure_decode())
            self.assertFalse(runner.execute_model(self.sched, is_profile=True))
            self.req.is_prefilling_np[0] = True
            self.assertFalse(runner.execute_model(self.sched))
            self.assertEqual(votes, [True, False])
        finally:
            gate.uninstall()


if __name__ == "__main__":
    unittest.main()
