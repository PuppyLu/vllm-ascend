# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for Torch operator argument order and bound-buffer lifecycle."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def _load_module():
    names = tuple(f"input_{i}" for i in range(47))
    names = ("x",) + names[1:]
    attrs = (
        "hc_mult",
        "hc_sinkhorn_iters",
        "hc_eps",
        "norm_eps",
        "layer_rms_eps",
        "stage_limit",
        "conv_slot",
        "lower_bound",
        "output_norm_eps",
        "ffn_hc_eps",
        "ffn_norm_eps",
        "ffn_layer_rms_eps",
        "mhc_post_mult_value",
        "router_mode",
        "expert_id",
        "route_ordinal",
        "down_expert_id",
        "ep_world_size",
    )
    modules = {
        name: ModuleType(name)
        for name in (
            "mc2_integration",
            "mc2_integration.distributed_mc2",
            "mc2_integration.projection_contract",
            "mc2_integration.stage24_admission",
            "mc2_integration.torch_acl_bridge",
            "mc2_integration.vllm_mc2_repeatable",
        )
    }
    modules["mc2_integration.distributed_mc2"]._validate_expert_bank = lambda *a, **k: None
    projection = modules["mc2_integration.projection_contract"]
    projection.MC2_INPUT_NAMES = names
    projection.validate_external_routes = projection.validate_projection_shapes = lambda *a, **k: None
    admission = modules["mc2_integration.stage24_admission"]
    for protocol in (
        "GENERAL_PROTOCOL",
        "GRID_PROTOCOL",
        "NATIVE_SHARED_PROTOCOL",
        "PADDING_PROTOCOL",
        "PROTOCOL",
        "SHARDED_SHARED_PROTOCOL",
    ):
        setattr(admission, protocol, protocol)
    bridge = modules["mc2_integration.torch_acl_bridge"]
    bridge.ATTR_NAMES = attrs
    bridge._check_tensor = bridge.initialized_outputs = lambda *a, **k: None
    modules["mc2_integration.vllm_mc2_repeatable"].RepeatableMegaKernelSession = object
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_torch_session.py"
    spec = importlib.util.spec_from_file_location("mc2_torch_session_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class BoundBuffersTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.inputs = {name: torch.tensor([i]) for i, name in enumerate(self.module.MC2_INPUT_NAMES)}
        self.outputs = [torch.zeros(1) for _ in range(50)]
        self.attrs = {
            name: i if name in self.module._INT_ATTRIBUTES else i + 0.25
            for i, name in enumerate(self.module.ATTR_NAMES)
        }
        self.groups = ("tp", "ep", "tp_algo", "ep_algo", 1, 3)
        self.sync_count = 0
        self.calls = []
        self.stream = SimpleNamespace(npu_stream=42, synchronize=self.sync)
        self.npu = SimpleNamespace(current_stream=lambda device: self.stream)

    def sync(self):
        self.sync_count += 1

    def op(self, *args):
        self.calls.append(args)
        return args[1]

    def test_operator_arguments_and_buffer_reuse(self):
        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        with patch.object(torch, "npu", self.npu, create=True):
            result = bound.run()
            bound.run()
        self.assertEqual(self.sync_count, 2)
        self.assertIs(result[0], self.outputs[0])
        self.assertEqual(self.calls[0][0], list(self.inputs.values()))
        self.assertEqual(self.calls[0][2], [int(self.attrs[n]) for n in self.module._INT_ATTRIBUTES])
        self.assertEqual(self.calls[0][3], [float(self.attrs[n]) for n in self.module._FLOAT_ATTRIBUTES])
        self.assertEqual(self.calls[0][4:], self.groups)

    def test_replaced_input_is_rejected_before_launch(self):
        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        self.inputs["x"] = self.inputs["x"].clone()
        with patch.object(torch, "npu", self.npu, create=True), self.assertRaisesRegex(RuntimeError, "changed"):
            bound.run()
        self.assertFalse(self.calls)

    def test_invalid_attributes_fail_before_launch(self):
        self.attrs["hc_mult"] = 2**63
        with self.assertRaisesRegex(ValueError, "int64"):
            self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op)
        self.assertFalse(self.calls)

    def test_integer_attributes_reject_float_and_boolean_before_launch(self):
        for value in (1.2, 1.0, True, np.bool_(True), torch.tensor(True)):
            with self.subTest(value=value):
                self.attrs["hc_mult"] = value
                with self.assertRaisesRegex(TypeError, "integer attribute hc_mult"):
                    self.module._TorchBoundBuffers(
                        self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op
                    )
        self.assertFalse(self.calls)

    def test_integer_attributes_accept_index_protocol_and_int64_bounds(self):
        class IntLike:
            def __index__(self):
                return 7

        for value, expected in ((IntLike(), 7), (np.int64(9), 9), (-(2**63), -(2**63)), (2**63 - 1, 2**63 - 1)):
            with self.subTest(value=value):
                self.attrs["hc_mult"] = value
                bound = self.module._TorchBoundBuffers(
                    self.inputs, self.outputs, self.attrs, self.groups, self.stream, self.op
                )
                self.assertEqual(bound.int_attrs[0], expected)
        self.assertFalse(self.calls)

    def test_failure_disables_reuse_and_keeps_resources(self):
        def failed(*args):
            raise RuntimeError("launch failed")

        bound = self.module._TorchBoundBuffers(self.inputs, self.outputs, self.attrs, self.groups, self.stream, failed)
        with patch.object(torch, "npu", self.npu, create=True), self.assertRaisesRegex(RuntimeError, "launch failed"):
            bound.run()
        bound.close()
        self.assertIs(self.module._FAILED_LAUNCHES[-1], bound)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            bound.validate_bindings()


if __name__ == "__main__":
    unittest.main()
