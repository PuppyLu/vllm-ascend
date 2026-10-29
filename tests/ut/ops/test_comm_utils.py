#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend.ops.fused_moe import comm_utils
from vllm_ascend.ops.fused_moe.comm_utils import (
    _gather_along_first_dim,
    async_all_to_all,
    gather_from_sequence_parallel_region,
)


class TestDistributedCommunication(PytestBase):
    @pytest.fixture(autouse=True)
    def context(self, mocker: MockerFixture):
        mocker.patch("torch.npu.current_device", return_value="cpu")
        mocker.patch("torch.distributed.get_world_size", return_value=4)

        mocker.patch("torch.distributed.get_rank", return_value=0)

    @pytest.mark.parametrize(
        "input_tensor, output_split_sizes, input_split_sizes",
        [(torch.randn(8, 16), [2, 2, 2, 2], [2, 2, 2, 2]), (torch.randn(16, 32), None, None)],
    )
    def test_async_all_to_all(self, input_tensor, output_split_sizes, input_split_sizes, mocker: MockerFixture):
        """Test async_all_to_all"""
        mock_group = mocker.MagicMock()
        mocker.patch("torch.distributed.all_to_all_single", return_value=mocker.MagicMock())

        _, a2a_out, handle = async_all_to_all(input_tensor, output_split_sizes, input_split_sizes, mock_group)

        # Check if the output tensor is created properly
        if output_split_sizes is None:
            assert a2a_out.shape == input_tensor.shape
        else:
            total_output_size = sum(output_split_sizes)
            expected_shape = [total_output_size] + list(input_tensor.size())[1:]
            assert a2a_out.shape == torch.Size(expected_shape)

        # Ensure handle is returned from async operation
        assert handle is not None
        assert isinstance(handle, mocker.MagicMock)

    @pytest.mark.parametrize(
        "world_size, test_tensor, expected", [(1, torch.randn(8, 16), (8, 16)), (4, torch.randn(8, 16), (32, 16))]
    )
    def test_gather_along_first_dim(self, test_tensor, expected, world_size, mocker: MockerFixture):
        """Test _gather_along_first_dim"""
        mocker.patch("torch.distributed.get_world_size", return_value=world_size)

        result = _gather_along_first_dim(test_tensor, mocker.MagicMock())

        assert result.shape == expected

    @pytest.mark.parametrize(
        "input_tensor, output_split_sizes", [(torch.randn(8, 16), None), (torch.randn(8, 16), [2, 2, 2, 2])]
    )
    def test_gather_from_sequence_parallel_region(self, input_tensor, output_split_sizes, mocker: MockerFixture):
        """Test gather_from_sequence_parallel_region"""
        mock_group = mocker.MagicMock()

        result = gather_from_sequence_parallel_region(input_tensor, mock_group, output_split_sizes)

        # If output_split_sizes is not provided, result should have expanded first dimension by world size
        if output_split_sizes is None:
            expected_shape = [input_tensor.shape[0] * 4] + list(input_tensor.shape[1:])
            assert result.shape == torch.Size(expected_shape)
        else:
            # If output_split_sizes is provided, result shape is dictated by sum of output_split_sizes
            expected_shape = [sum(output_split_sizes)] + list(input_tensor.shape[1:])
            assert result.shape == torch.Size(expected_shape)


class TestA3ExternalZercmoeBackend(PytestBase):
    @pytest.fixture(autouse=True)
    def reset_backend_cache(self, mocker):
        # Loader preflight may bind/check actual EP identity; these tests
        # inspect the wheel API without distributed groups or environment edits.
        mocker.patch.object(comm_utils, "bind_zercmoe_ep_environment", return_value=None, create=True)
        comm_utils.a3_zercmoe_lib_available.cache_clear()
        yield
        comm_utils.a3_zercmoe_lib_available.cache_clear()

    @staticmethod
    def _make_ops(mocker: MockerFixture, version="1.5.0"):
        def pack_weights(w1, w2):
            raise AssertionError("loader must not pack weights")

        def pack_weights_sf(scale):
            raise AssertionError("loader must not encode scales")

        def get_symm_buffer(
            *,
            moe_expert_num,
            ep_world_size,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            ipport,
            device_id,
            w_dtype=torch.int8,
            mem_size_mb=0,
        ):
            raise AssertionError("loader must not bootstrap communication")

        def mega_moe(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            *,
            return_expert_tokens=0,
            l1_weights_sf=None,
            l2_weights_sf=None,
        ):
            raise AssertionError("loader must not launch an operator")

        return SimpleNamespace(
            __version__=version,
            pack_weights=mocker.create_autospec(pack_weights),
            pack_weights_sf=mocker.create_autospec(pack_weights_sf),
            get_symm_buffer=mocker.create_autospec(get_symm_buffer),
            mega_moe=mocker.create_autospec(mega_moe),
        )

    @pytest.mark.parametrize(
        "version", ["1.4.0", "1.5.0", "1.5.1", "1.5.0+internal", "1.5.0rc1", "1.6.0", "2.0.0", "unknown", None]
    )
    def test_load_a3_does_not_restrict_package_version(self, mocker, version):
        ops = self._make_ops(mocker, version)
        if version is None:
            del ops.__version__
        importer = mocker.patch.object(comm_utils, "import_module", return_value=ops)

        assert comm_utils.load_a3_zercmoe_ops() is ops
        importer.assert_called_once_with("zercmoe")
        ops.pack_weights.assert_not_called()
        ops.pack_weights_sf.assert_not_called()
        ops.get_symm_buffer.assert_not_called()
        ops.mega_moe.assert_not_called()

    @pytest.mark.parametrize("name", ["pack_weights", "pack_weights_sf", "get_symm_buffer", "mega_moe"])
    @pytest.mark.parametrize("missing", [False, True])
    def test_load_a3_rejects_missing_or_noncallable_api(self, mocker, name, missing):
        ops = self._make_ops(mocker)
        if missing:
            delattr(ops, name)
        else:
            setattr(ops, name, object())
        mocker.patch.object(comm_utils, "import_module", return_value=ops)

        with pytest.raises(RuntimeError):
            comm_utils.load_a3_zercmoe_ops()

    @pytest.mark.parametrize(
        "name,parameter",
        [
            ("pack_weights", "w2"),
            ("pack_weights_sf", "scale"),
            ("get_symm_buffer", "w_dtype"),
            ("get_symm_buffer", "intermediate_hidden"),
            ("mega_moe", "return_expert_tokens"),
            ("mega_moe", "l1_weights_sf"),
            ("mega_moe", "l2_weights_sf"),
        ],
    )
    def test_load_a3_does_not_inspect_signatures(self, mocker, name, parameter):
        ops = self._make_ops(mocker)
        function = getattr(ops, name)
        signature = inspect.signature(function)
        function.__signature__ = signature.replace(
            parameters=[value for key, value in signature.parameters.items() if key != parameter]
        )
        mocker.patch.object(comm_utils, "import_module", return_value=ops)

        assert comm_utils.load_a3_zercmoe_ops() is ops
        function.assert_not_called()

    def test_load_a3_accepts_documented_call_through_keyword_forwarding_wrapper(self, mocker):
        ops = self._make_ops(mocker)

        def mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, **kwargs):
            raise AssertionError("loader must not execute a wrapper")

        ops.mega_moe = mocker.create_autospec(mega_moe)
        mocker.patch.object(comm_utils, "import_module", return_value=ops)
        assert comm_utils.load_a3_zercmoe_ops() is ops
        ops.mega_moe.assert_not_called()

    def test_a3_availability_caches_success(self, mocker):
        loader = mocker.patch.object(comm_utils, "load_a3_zercmoe_ops", return_value=self._make_ops(mocker))
        assert comm_utils.a3_zercmoe_lib_available() is True
        assert comm_utils.a3_zercmoe_lib_available() is True
        loader.assert_called_once_with()
        comm_utils.a3_zercmoe_lib_available.cache_clear()
        assert comm_utils.a3_zercmoe_lib_available() is True
        assert loader.call_count == 2

    @pytest.mark.parametrize("error_type", [ImportError, AttributeError, RuntimeError, OSError])
    def test_explicit_a3_availability_failure_raises_without_a2_or_cann_fallback(self, mocker, error_type):
        loader = mocker.patch.object(comm_utils, "load_a3_zercmoe_ops", side_effect=error_type("unavailable"))
        a2 = mocker.patch.object(comm_utils, "zercmoe_lib_available", return_value=True)
        cann = mocker.patch.object(comm_utils, "cann_zercmoe_lib_available", return_value=True, create=True)

        with pytest.raises(RuntimeError):
            comm_utils.a3_zercmoe_lib_available()
        loader.assert_called_once_with()
        a2.assert_not_called()
        cann.assert_not_called()
