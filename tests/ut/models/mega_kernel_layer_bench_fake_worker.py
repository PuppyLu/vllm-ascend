import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as N

import torch


# Fake only the integration surfaces. No NPU, HCCL or model validation implied.
def install(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for n in ["cross_rank_prototype", "vllm", "vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.glm5next"]:
    install(n)
staged = install("cross_rank_prototype.staged_collectives")


def agree(err, dev, group, label):
    if err:
        raise err


staged._agree = agree
context = install("vllm.forward_context")
context.get_forward_context = lambda: N()
install("vllm.distributed")
kv = install("vllm.distributed.kv_transfer")
kv.has_kv_transfer_group = lambda: False
asc = install("vllm_ascend.ascend_config")
asc.get_ascend_config = lambda: N(enable_force_eplb=False)
ad = install("vllm_ascend.models.glm5next.mega_kernel_adapter")
ad.prepare_decode_inputs = lambda *a, **kw: N(state_indices=torch.tensor([[0, 0], [1, 1]], dtype=torch.int32))
ad._copy_live_kda_slots = lambda cache, indices: tuple(t.clone() for t in cache)


class Evt:
    def __init__(self, enable_timing):
        pass

    def record(self, s):
        pass

    def elapsed_time(self, e):
        return 1.0


class Npu:
    Event = Evt

    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def current_stream(device):
        return N()

    @staticmethod
    def is_current_stream_capturing():
        return False


torch.npu = Npu()
spec = importlib.util.spec_from_file_location(
    "live", Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mega_kernel_layer_bench.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# TP8 original router sees only its token partition. Verify that the comparison
# maps it back to the matching source rows and rejects an invalid mask.
prepare_cls = type("PrepareAndFinalizeWithMC2", (), {"__module__": "vllm_ascend.ops.fused_moe.prepare_finalize"})
prepare = prepare_cls()
prepare.tp_size, prepare.tp_rank, prepare.replace_allreduce = 8, 3, False
extra = N(moe_comm_method=N(prepare_finalize=prepare), padded_num_tokens=64, mc2_mask=torch.ones(64, dtype=torch.bool))
forward_context = install("vllm_ascend.ascend_forward_context")
forward_context._EXTRA_CTX = extra
full_ids = torch.arange(64 * 8, dtype=torch.int32).reshape(64, 8)
full_weights = torch.ones(64, 8)
indices = m._original_route_indices(64, 8, 3, 8)
assert indices.tolist() == list(range(24, 32))
m._assert_routes([(full_weights[indices], full_ids[indices])], [(full_weights, full_ids)], 64, indices)
extra.mc2_mask[24] = False
try:
    m._original_route_indices(64, 8, 3, 8)
except ValueError:
    pass
else:
    raise AssertionError("invalid route mask accepted")
extra.mc2_mask[24] = True
m.dist.get_world_size = lambda group: 1
m.dist.get_rank = lambda group: 0
m.dist.barrier = lambda group: None
m.dist.all_gather_object = lambda out, value, group: out.__setitem__(0, value)


class Routed:
    def _select_experts(self, *args, **kwargs):
        return torch.full((2, 8), 0.125), torch.arange(8, dtype=torch.int32).repeat(2, 1)


routed = Routed()
layer = N(
    layer_idx=5,
    self_attn=N(kv_cache=(torch.zeros(3, 3, 4), torch.zeros(3, 1, 2, 2)), _conv_state_dim_first=False),
    mlp=N(experts=N(routed_experts=routed)),
    _megakernel_stats={"hits": 0},
)


def actual(positions, hidden, residual, post, comb):
    routed._select_experts()
    layer.self_attn.kv_cache[0][0] += 1
    layer.self_attn.kv_cache[1][1] += 2
    return (hidden.clone(), None, None, None)


def fused(*args):
    result = actual(*args)
    layer._megakernel_stats["hits"] += 1
    return result


layer._megakernel_original_forward = actual
layer._megakernel_benchmark_forward = fused
before = tuple(t.clone() for t in layer.self_attn.kv_cache)
result = m.measure(
    layer,
    torch.tensor([1, 2]),
    torch.ones(2, 4),
    None,
    None,
    None,
    control_group=N(),
    control_device="cpu",
    consumer_ready=lambda mode, out: None,
    events_join_required_work=True,
    logical_step_agreed=True,
    expected_rows=2,
    warmup=1,
    samples=2,
)
assert result["status"] == "PAIRED_LIVE_LAYER_PASS" and result["summary"]["speedup_event_ms"] == 1
assert all(torch.equal(t, x) for t, x in zip(layer.self_attn.kv_cache, before))
assert "_select_experts" not in routed.__dict__
print("CPU protocol smoke: passed; no A3 or model test")


def bad_fused(*args):
    result = actual(*args)
    layer.self_attn.kv_cache[1][0] += 5
    layer._megakernel_stats["hits"] += 1
    return result


layer._megakernel_benchmark_forward = bad_fused
try:
    m.measure(
        layer,
        torch.tensor([1, 2]),
        torch.ones(2, 4),
        None,
        None,
        None,
        control_group=N(),
        control_device="cpu",
        consumer_ready=lambda mode, out: None,
        events_join_required_work=True,
        logical_step_agreed=True,
        expected_rows=2,
        warmup=0,
        samples=1,
    )
except AssertionError as e:
    assert "recurrent_state" in str(e)
else:
    raise AssertionError("bad state was accepted")
assert all(torch.equal(t, x) for t, x in zip(layer.self_attn.kv_cache, before))
assert "_select_experts" not in routed.__dict__
print("CPU failure and state-restoration smoke: passed")
