from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend import utils
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker import model_runner_v1 as runner_module


@pytest.mark.parametrize(
    "configured,runtime,uniform,draft,ubatch,expected",
    [
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, True, False, False, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, np.bool_(True), False, False, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, False, False, False, False),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, True, True, False, False),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, True, False, True, False),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, True, False, False, False),
        (CUDAGraphMode.FULL, CUDAGraphMode.FULL, True, False, False, False),
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, True, False, False, False),
    ],
)
def test_zercmoe_decode_graph_identifies_only_target_uniform_full_decode(
    monkeypatch, configured, runtime, uniform, draft, ubatch, expected
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=configured),
        parallel_config=SimpleNamespace(use_ubatching=ubatch),
    )
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(multistream_overlap_shared_expert=False))
    assert (
        afc.is_zercmoe_decode_graph(config, runtime, SimpleNamespace(uniform=uniform), is_draft_model=draft) is expected
    )


def test_zercmoe_decode_graph_allows_shared_expert_overlap(monkeypatch):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(multistream_overlap_shared_expert=True))
    assert afc.is_zercmoe_decode_graph(config, CUDAGraphMode.FULL, SimpleNamespace(uniform=True), is_draft_model=False)


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "decode,graph,draft", [(True, True, False), (False, False, False), (False, True, False), (True, True, True)]
)
def test_zercmoe_decode_graph_selection_is_not_fixed_to_one_ep(monkeypatch, ep, decode, graph, draft):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        parallel_config=SimpleNamespace(tensor_parallel_size=ep),
    )
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "get_zercmoe_tokens_capacity", lambda _: 128)
    result = afc._use_zercmoe_for_batch(32, config, 128, is_decode=decode, is_draft_model=draft, is_graph_mode=graph)
    assert result is (not draft and (not graph or decode))


def test_zercmoe_target_keeps_dp_metadata_synchronization(monkeypatch):
    config = SimpleNamespace()
    ascend = SimpleNamespace(enable_zerc_moe=1, get_mc2_comm_alg=lambda: "fullmesh")
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(utils, "is_moe_model", lambda _: True)
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(afc, "is_zercmoe_active_by_config", lambda _: True)
    assert not utils.should_skip_allreduce_across_dp_group(config)


@pytest.mark.parametrize(
    "computed,prompt,force,expected",
    [([0], [8], None, False), ([7], [8], None, False), ([8], [8], None, True), ([7], [8], True, True)],
)
def test_zercmoe_decode_graph_does_not_classify_last_prefill_chunk_as_decode(
    monkeypatch, computed, prompt, force, expected
):
    runner = object.__new__(runner_module.NPUModelRunner)
    runner.ascend_config = SimpleNamespace(enable_zerc_moe=1)
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner.parallel_config = runner.vllm_config.parallel_config
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array(computed),
        num_prompt_tokens=np.array(prompt),
        lora_id_to_lora_request={},
    )
    runner.speculative_config = None
    runner.uniform_decode_query_len = 1
    runner._pad_for_sequence_parallelism = lambda n: n
    dispatcher = Mock()
    dispatcher.dispatch.side_effect = lambda num_tokens, **kwargs: (
        CUDAGraphMode.NONE,
        SimpleNamespace(num_tokens=num_tokens, uniform=kwargs["uniform_decode"]),
    )
    runner.cudagraph_dispatcher = dispatcher
    monkeypatch.setattr(runner_module, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(runner_module, "enable_sp", lambda _: False)
    runner._determine_batch_execution_and_padding(1, 1, np.array([1]), 1, False, force_uniform_decode=force)
    assert bool(dispatcher.dispatch.call_args.kwargs["uniform_decode"]) is expected
