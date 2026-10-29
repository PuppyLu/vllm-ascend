# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of instance-local V2 draft identity and factory routing."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class TestDraftForwardConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[4]
        spec = importlib.util.spec_from_file_location(
            "_ascend_draft_forward_config_under_test",
            cls.repo / "vllm_ascend/worker/v2/spec_decode/forward_config.py",
        )
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def test_copy_preserves_resources_without_rerunning_constructor(self):
        class Config:
            def __init__(self):
                self.additional_config = {"enable_zerc_moe": 1}
                self.model_config = object()
                self.parallel_config = object()
                self.compilation_config = object()

            def __post_init__(self):
                raise AssertionError("post-init must not run when making the draft view")

        target = Config()
        draft = self.helper.create_draft_forward_config(target)
        self.assertIsNot(draft, target)
        self.assertIsNot(draft.additional_config, target.additional_config)
        self.assertEqual(target.additional_config, {"enable_zerc_moe": 1})
        self.assertTrue(self.helper.is_draft_forward_config(draft))
        self.assertFalse(self.helper.is_draft_forward_config(target))
        for name in ("model_config", "parallel_config", "compilation_config"):
            self.assertIs(getattr(draft, name), getattr(target, name))
        draft.additional_config["draft_only"] = True
        self.assertNotIn("draft_only", target.additional_config)

    def test_empty_and_false_markers_remain_target_contexts(self):
        for additional in (None, {}, {"_ascend_is_draft_model": False}, {"_ascend_is_draft_model": "true"}):
            with self.subTest(additional=additional):
                target = SimpleNamespace(additional_config=additional)
                self.assertFalse(self.helper.is_draft_forward_config(target))
                self.assertTrue(self.helper.is_draft_forward_config(self.helper.create_draft_forward_config(target)))
                self.assertIs(target.additional_config, additional)

    def test_each_supported_factory_branch_receives_a_private_marked_config(self):
        factory_path = self.repo / "vllm_ascend/worker/v2/spec_decode/__init__.py"
        tree = ast.parse(factory_path.read_text())
        factory_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "init_speculator"
        )
        factory_ast = ast.Module(body=[factory_node], type_ignores=[])
        namespace = {
            "torch": SimpleNamespace(device=object),
            "VllmConfig": object,
            "create_draft_forward_config": self.helper.create_draft_forward_config,
        }
        exec(compile(factory_ast, str(factory_path), "exec"), namespace)
        factory = namespace["init_speculator"]
        branches = (
            ("dspark", "dspark", "AscendDSparkSpeculator"),
            ("dflash", "dflash", "AscendDFlashSpeculator"),
            ("mtp", "mtp", "AscendMTPSpeculator"),
            ("eagle", "eagle", "AscendEagleSpeculator"),
        )
        for method, package, class_name in branches:
            with self.subTest(method=method):
                speculative = SimpleNamespace(
                    method=method,
                    use_dspark=lambda method=method: method == "dspark",
                    use_dflash=lambda method=method: method == "dflash",
                    use_gemma4_mtp=lambda: False,
                    use_step3p5_mtp=lambda: False,
                    use_eagle=lambda method=method: method == "eagle",
                )
                target = SimpleNamespace(
                    speculative_config=speculative,
                    additional_config={"enable_zerc_moe": 1},
                    model_config=object(),
                )
                seen = []

                def construct(config, device, seen=seen):
                    seen.append(config)
                    return SimpleNamespace(vllm_config=config)

                module_name = f"vllm_ascend.worker.v2.spec_decode.{package}.speculator"
                module = ModuleType(module_name)
                setattr(module, class_name, construct)
                with patch.dict(sys.modules, {module_name: module}):
                    draft = factory(target, object())
                self.assertEqual(len(seen), 1)
                self.assertIs(draft.vllm_config, seen[0])
                self.assertIsNot(draft.vllm_config, target)
                self.assertIs(draft.vllm_config.model_config, target.model_config)
                self.assertTrue(self.helper.is_draft_forward_config(draft.vllm_config))
                self.assertFalse(self.helper.is_draft_forward_config(target))
                self.assertEqual(target.additional_config, {"enable_zerc_moe": 1})


if __name__ == "__main__":
    unittest.main()
