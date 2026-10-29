from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import prepare_finalize as pf
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithZercMoE
from vllm_ascend.quantization.quant_type import QuantType


def _prepare_impl(tp, rank, ctx):
    with (
        patch.object(pf, "get_tensor_model_parallel_world_size", return_value=tp),
        patch.object(pf, "get_tensor_model_parallel_rank", return_value=rank),
    ):
        return PrepareAndFinalizeWithZercMoE(SimpleNamespace(tp_group=SimpleNamespace(device_group=object())))


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("local_tokens,max_tokens", [(0, 1), (1, 1), (0, 65), (65, 65)])
@pytest.mark.parametrize("rank", [0, 1, 12, 13, 15])
def test_zercmoe_dp_uneven_tokens_have_identical_ep_rows_and_correct_masks(sp, local_tokens, max_tokens, rank):
    # Even one valid token must keep every EP rank in the collective.
    tp, hidden = 16, 4
    target = (max_tokens + tp - 1) // tp * tp
    rows = target // tp
    original = torch.arange(local_tokens * hidden, dtype=torch.float32).reshape(local_tokens, hidden).to(torch.bfloat16)
    padded = torch.nn.functional.pad(original, (0, 0, 0, target - local_tokens))
    mask = torch.arange(target) < local_tokens
    ctx = SimpleNamespace(padded_num_tokens=target, mc2_mask=mask, mc2_num_actual_tokens=local_tokens)
    impl = _prepare_impl(tp, rank, ctx)
    x = padded[rank * rows : (rank + 1) * rows] if sp else original
    logits = torch.ones(x.shape[0], 256)
    source = x.clone()
    with patch.object(pf, "_EXTRA_CTX", ctx):
        result = impl.prepare(x, logits, replace_allreduce=sp, quant_type=QuantType.W8A8)
    assert result.hidden_states.shape == (rows, hidden)
    assert result.router_logits.shape == (rows, 256)
    assert not impl.all_ranks_empty  # even for the DP rank with zero valid tokens
    assert impl.all_tokens_valid == (local_tokens >= (rank + 1) * rows)
    assert impl.all_tokens_padding == (local_tokens <= rank * rows)
    torch.testing.assert_close(result.hidden_states, padded[rank * rows : (rank + 1) * rows])
    torch.testing.assert_close(result.mc2_mask, mask[rank * rows : (rank + 1) * rows])
    torch.testing.assert_close(x, source)
    assert result.hidden_states.dtype == torch.bfloat16
    assert result.pertoken_scale is None  # the wheel quantizes activations internally

    def gather(outputs, local, group):
        assert local.shape == (rows, hidden)
        for index, output in enumerate(outputs):
            output.copy_(padded[index * rows : (index + 1) * rows])

    with patch.object(pf.dist, "all_gather", side_effect=gather) as all_gather:
        output = impl.finalize(result.hidden_states, False, result.padded_hidden_states_shape)
    if sp:
        all_gather.assert_not_called()
        torch.testing.assert_close(output, x)
    else:
        all_gather.assert_called_once()
        torch.testing.assert_close(output, original)
        assert output.shape[0] == local_tokens


@pytest.mark.parametrize("sp", [False, True])
def test_zercmoe_globally_empty_batch_needs_no_tp_collective(sp):
    ctx = SimpleNamespace(padded_num_tokens=0, mc2_mask=torch.empty(0, dtype=torch.bool))
    impl = _prepare_impl(16, 3, ctx)
    x = torch.empty(0, 4, dtype=torch.bfloat16)
    with patch.object(pf, "_EXTRA_CTX", ctx):
        result = impl.prepare(x, torch.empty(0, 256), replace_allreduce=sp, quant_type=QuantType.W8A8)
    assert impl.all_ranks_empty
    assert not impl.all_tokens_padding
    with patch.object(pf.dist, "all_gather", Mock(side_effect=AssertionError)) as gather:
        output = impl.finalize(result.hidden_states, False, result.padded_hidden_states_shape)
    gather.assert_not_called()
    assert output.shape == (0, 4) and output.dtype == torch.bfloat16


@pytest.mark.parametrize("invalid", ["mask_short", "target_unaligned", "sp_unsliced", "input_too_long"])
def test_zercmoe_inconsistent_padding_is_rejected_before_collectives(invalid):
    ctx = SimpleNamespace(padded_num_tokens=80, mc2_mask=torch.ones(80, dtype=torch.bool))
    impl = _prepare_impl(16, 0, ctx)
    sp = False
    rows = 65
    if invalid == "mask_short":
        ctx.mc2_mask = ctx.mc2_mask[:-1]
    elif invalid == "target_unaligned":
        ctx.padded_num_tokens = 65
    elif invalid == "sp_unsliced":
        sp = True
    else:
        rows = 81
    with patch.object(pf, "_EXTRA_CTX", ctx), pytest.raises(ValueError, match="ZERC_MOE"):
        impl.prepare(torch.zeros(rows, 4), torch.zeros(rows, 256), replace_allreduce=sp)


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("rank", range(8))
def test_zercmoe_all_valid_fast_path_tracks_cpu_mask_metadata_each_batch(sp, rank):
    ctx = SimpleNamespace(padded_num_tokens=16, mc2_mask=torch.ones(16, dtype=torch.bool))
    impl = _prepare_impl(8, rank, ctx)
    for actual in (16, 13, 1, 0, None, -1, 2, 3, 4):
        ctx.mc2_num_actual_tokens = actual
        ctx.mc2_mask = torch.arange(16) < (actual or 0)
        # Physical rows may already be padded. They do not prove validity.
        x = torch.ones(2 if sp else 16, 4, dtype=torch.bfloat16)
        with (
            patch.object(pf, "_EXTRA_CTX", ctx),
            patch("torch.Tensor.item", side_effect=AssertionError("must use CPU metadata")),
        ):
            impl.prepare(x, torch.ones(x.shape[0], 256), replace_allreduce=sp, quant_type=QuantType.W8A8)
        assert impl.all_tokens_valid == (actual is not None and actual >= (rank + 1) * 2)
        assert impl.all_tokens_padding == (actual is not None and 0 <= actual <= rank * 2)
        assert not impl.all_ranks_empty
