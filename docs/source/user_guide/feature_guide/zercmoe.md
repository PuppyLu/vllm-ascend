# ZercMoE on A3

The A3 integration uses the installed zercmoe wheel and its public
`pack_weights`, `pack_weights_sf`, `get_symm_buffer` and `mega_moe` APIs. It does not call the
hand-written `cann_ops_transformer.npu_zercmoe` backend. A2 retains its previous
nonquantized path.

This integration follows the supplied 1.5.0 guide: BF16 activations, INT8 expert
weights, real per-channel scales and gated `SiLU(gate) * up`. The framework does
not restrict wheel versions or inspect Python signatures. Actual operator calls
still use the interface and tensor semantics described here; check correctness
with the installed wheel and target model.

## Installation and configuration

Use the existing compatible vLLM / vLLM-Ascend environment and the complete
CANN 9.1 runtime, toolkit and built-in kernel/OPP installation on A3 aarch64.
Install the internal wheel into the Python environment used by every worker:

```bash
python3 -m pip install --no-deps /path/to/zercmoe-*.whl
python3 - <<'PY'
import torch
import torch_npu
import zercmoe
from vllm_ascend.ops.fused_moe.zercmoe_adapter import validate_w8a8_backend

validate_w8a8_backend(zercmoe)
print("ZERCMOE_API_OK", getattr(zercmoe, "__version__", "not declared"), zercmoe.__file__)
PY
```

This checks package loading and that the required API names are callable; it does not
build a communication domain or execute a MoE kernel. Importing the wheel
loads its bundled libraries. Keep the normal CANN environment sourced.

Merge these fields into each participating worker's `--additional-config`.
P, D and colocated workers use the same A3 policy:

```json
{
  "enable_zerc_moe": 1,
  "zercmoe_ipport": "tcp://127.0.0.1:28765",
  "zercmoe_shmem_mb": 0
}
```

The backend is disabled by default. Remove the previous
`zercmoe_max_recv_token_num`: it belonged to the retired hand-written KFC ABI
and has no equivalent in this wheel. Eligible workers reject it rather than
silently ignoring it. `enable_fused_mc2` independently chooses the existing
backend for draft models and workers without Zerc; it is not required to enable Zerc. A requested
eligible A3 backend with a missing package or required API raises an error; it
cannot silently substitute the old CANN operator. Incompatible arguments or
operator behavior are detected only when exercised and validated.

Enabling A3 Zerc automatically packs each target-model routed-expert layer and
releases its original W1/W2 instead of also storing the legacy NZ weights.
No separate single-weight option is supported: remove the retired
`zercmoe_single_weight` setting from existing configurations. Dual weights
and target-model fallback are unsupported.
Weight packing still
temporarily needs the original and packed tensors. After successful loading,
unused PyTorch NPU allocator blocks are returned before SHMEM initialization.
The SHMEM heap and its minimum size are unchanged; actual peak memory and fit
must be measured on NPU.

Single-weight mode requires model runner V1, A3, BF16 activations,
W8A8_DYNAMIC routed experts, expert parallelism and PP=PCP=DCP=1. Use either
`--enforce-eager` or `FULL_DECODE_ONLY` graph mode without microbatching. GLM MTP
is supported when draft TP matches target TP: the
target's routed experts use Zerc only and the independent MTP routed experts
retain only their original backend format. Other speculative methods, LoRA, dynamic EPLB, XLite,
CPU weight offloading, sleep, weight updates and in-place reload are not
supported. A target batch that cannot use Zerc fails explicitly instead of
falling back to a backend whose weights are absent; MTP uses its own MoE
communication objects and legacy backend. MTP increases memory for its own
weights and runtime state. This MTP combination still requires
validation on the target NPU and model checkpoint.

For V1 decode graphs use `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
without `--enforce-eager`. Uniform target decode graphs use Zerc; prefill stays
eager and the MTP draft keeps its original backend. Other target graph modes
fail rather than falling back. Model-level BF16 graph replay,
multi-rank scheduling and wheel workspace pointer lifetime have not yet been
validated on NPU. Use isolated devices for the first correctness tests.
Shared-expert multistream overlap can be enabled for A3 Zerc-only target
weights; MTP draft experts retain the legacy single-stream path. Cross-stream
graph capture/replay and output correctness require NPU validation.

An operator test reported that wheel 1.5 can silently truncate a receiving
rank's routed rows when they exceed `2 * m * topK`; this affects eager and
graph, including zero-weight padding routes. Increasing the SHMEM heap is not
known to change that internal limit. Compare hotspot routes with an independent
golden result, not just graph against eager, and do not infer correctness from
successful execution.

## SHMEM identity, memory and lifetime

The wheel bootstraps through TCP/SHMEM, independently of HCCL collectives.
The framework uses the actual MC2/EP group's rank and size. When PP=PCP=1,
this group spans DP times TP; FlashComm1 does not add ranks.

The 1.5 interface reads `RANK` and `WORLD_SIZE` from the worker environment.
Missing values are populated from the actual EP group before importing or
packing with the wheel. Existing conflicting values are rejected before
initialization; the framework does not overwrite another rank domain.
`LOCAL_RANK` and `MASTER_*` remain unchanged. The framework currently passes
the current NPU logical device as `device_id`; this need not equal the EP rank.
Before initializing the A3 heap, each rank logs its EP rank and size,
`ASCEND_RT_VISIBLE_DEVICES`, NPU device count and current logical device. Use
the worker logs to check effective visibility; the wheel's requirements for
peer-device visibility and device-number conventions must be verified for the
actual deployment. Launchers with conflicting
inherited values need corrected worker identity or a wheel API with explicit
EP-rank parameters.

All ranks of one EP group must use the same reachable `zercmoe_ipport` and
capacity settings. Independent EP groups and jobs need distinct endpoints;
do not allocate different ports to DP ranks within one EP group. Multi-node
runs need an explicit reachable host address, not loopback. The wheel's
supported topology and EP sizes still require device validation.
Separate P and D groups must use separate endpoints, even on the same host.

`zercmoe_shmem_mb=0` or omission delegates sizing to the wheel. The 1.5 guide
describes automatic sizing as at least **18000 MB per rank/die**, potentially
more for larger shapes; setting it to `0` does not eliminate that memory cost.
Explicit values of at least **1024 MB per rank** pass the framework guard and
are forwarded to the wheel. This guard is not a guarantee that the requested
size is sufficient: the required size depends on the model and EP configuration,
and the guide reports `507015` AICore exceptions with 600 MB and 2 GB heaps.
Validate reduced sizes with the actual wheel and target workload. One
model-wide heap is created collectively inside memory profiling, before automatic KV Cache
sizing, and reused by sequential MoE layers. It is also created when profiling
uses short or fallback batches, so it cannot first appear after KV Cache has
consumed the free memory.

Zerc source capacity is `ceil(max_num_batched_tokens / TP)` per rank, covering
the aligned scheduling budget in every role. It is independent of the MC2
dispatch capacity and does not inherit MegaMoE's 4096-token limit. A 256K
request normally runs as multiple prefill chunks; its total input length is
not the number of rows passed to a single MoE call. Actual wheel limits still
require validation at the requested shape.

Budget the heap together with model weights, additional packed expert weights,
legacy NZ weights, KV Cache and other workspaces. Device capacity alone does
not establish that a model will fit. Explicit KV-cache budgets must still
leave room for these allocations.

All EP ranks must stop submissions, synchronize their devices and enter
`shutdown_zercmoe()` in the coordinated shutdown path. It calls the wheel's
collective `sym.destroy()`. Do not invoke it from a rank-local error handler,
destructor or `atexit`. Bootstrap and forward errors propagate without local
retries or fallback.

## Tensor boundary and execution

For E local experts, hidden size K and expert intermediate size I:

| Tensor | Shape / dtype |
| --- | --- |
| Activations | `[M,K]`, BF16 |
| Expert IDs | `[M,topK]`, INT32 global IDs, distinct per token |
| Routing probabilities | `[M,topK]`, FP32 |
| W1 before packing | `[E,K,2I]`, ND INT8, gate then up |
| W2 before packing | `[E,I,K]`, ND INT8 |
| Packed W1/W2 | Separate contiguous 1D INT8 tensors from `pack_weights` |
| S1 before encoding | `[E,2I]`, raw FP32 calibration scales |
| S2 before encoding | `[E,K]`, raw FP32 calibration scales |
| S1/S2 at forward | Contiguous INT64 output from the wheel's `pack_weights_sf`, with `E*2I` / `E*K` slots |
| Output | `[M,K]`, BF16, matching the input dtype |
| Expert token counts | `[E]`, INT32 |

Weight scales load directly into FP32 storage. Before the original weights are
released, the framework packs weights and calls the wheel's own
`pack_weights_sf` once for each scale. It retains the raw FP32 scales separately
and passes the same cached INT64 tensors to every forward. The encoder's shape
and slot order are preserved; existing fused-MC2 INT64 scales are not reused.
The 1.5 Python wrapper otherwise converts FP32 scales through CPU memory on
every call, so raw FP32 scales are rejected at the forward boundary. Normal
weight reloading rebuilds both encoded tensors. The wheel still handles input
activation quantization and SwiGLU requantization inside each forward.
Symmetric expert weights are required; nonzero offsets are rejected.

K and I must be divisible by 16. The former hand-written kernel's UB limits
and EP/expert-count caps do not describe this wheel. The framework requires
more than one EP rank and equal expert partitions; the wheel must validate its
remaining kernel and topology limits.

Eligible **eager prefill, decode and mixed** main-model batches enter this
backend in P, D and colocated deployments, including a single valid token.
There is no 64-token threshold and no inference of phase from token count or
server role. Selection uses synchronized token counts and the dedicated Zerc
capacity. Out-of-capacity target batches fail instead of falling back to a
backend whose expert weights are absent. Existing DP padding
and FlashComm1 TP-shard preparation are retained. Padding uses legal distinct
IDs, zero probabilities and zero BF16 activations. Ranks with no valid local
tokens still participate; only a synchronized all-EP-empty batch may skip.

Padding IDs prefer the source EP rank's local physical experts. When topK
exceeds the local expert count, the remaining distinct IDs wrap to other ranks.
The small device vector is cached across layers and batch sizes. Real routes
are mapped through `log2phy` before inserting physical padding IDs, so static
placement does not remap padding away from its intended rank. This removes
the framework's fixed-expert-0 padding hotspot; it does not make the wheel
skip zero-probability routes or establish a performance improvement.

Identified MTP/draft batches keep their configured fallback path;
MTP is not covered by this integration's NPU validation. Shared experts stay
separate; `mix_placement=true` is rejected. LoRA, dynamic EPLB, PP and PCP are excluded. Existing
model rotation processing is retained; this adapter adds no QuaRot rotation.
This does not validate every online-rotation model variant. C8 attention/cache
quantization is unchanged. The wheel also documents FP16 support; this
framework integration targets the existing BF16 GLM path.

The forward calls the public API:

```python
y, expert_tokens = zercmoe.mega_moe(
    x_bf16, topk_ids_int32, topk_weights_fp32,
    packed_w1, packed_w2, sym,
    return_expert_tokens=0,
    l1_weights_sf=scale1_int64_cached,
    l2_weights_sf=scale2_int64_cached,
)
```

The confirmed 1.5 asynchronous contract returns an empty INT32 `expert_tokens`
tensor with shape `(0,)`. The adapter validates this placeholder and the output
tensor without copying either to the CPU. Dynamic EPLB is excluded from this
path, so no consumer needs expert counts. Setting `return_expert_tokens=0`
skips the wheel's post-dispatch stream synchronization and D2H/H2D copies for
count collection. It does not skip device computation. This adapter does not
use the retired hand-written error-count protocol. Validate asynchronous
continuous calls and end-to-end performance with the deployed wheel.

## Verify actual backend use

With model runner V1, the first eager request submitted to the wheel logs
`A3 ZercMoE request launched` at INFO on EP rank zero. A graph request logs
`A3 ZercMoE decode graph selected` once; this records selection and is not proof
of replay. Unsupported target graph or capacity selections fail before execution.
These messages
exclude dummy/profile forwards. A launch message means the asynchronous
call was submitted, not that device execution or numerical accuracy was
verified. Model runner V2 retains the DEBUG selection logs below; its
warmup forwards are not labeled as real requests.

Set `VLLM_LOGGING_LEVEL=DEBUG` and check for both
`A3 ZercMoE SHMEM buffer` and the actual `ZERC_MOE` communication selection.
Service startup or throughput figures alone do not prove a batch called the
wheel: graph selection alone is not a device execution check. Inspect request-time
`method`, `num_tokens`, `zercmoe_capacity`, `graph` and `draft` log fields.
With `--enforce-eager`, verify `method=MoECommType.ZERC_MOE` for prefill and
single-token decode, including on D workers. Heap creation alone is not proof
that an inference batch called the wheel.

First run the wheel's supplied BF16 demo on reserved devices, independently
of the full framework:

```bash
export IPPORT=tcp://<reachable-host>:<unused-port>
export ASCEND_RT_VISIBLE_DEVICES=<reserved-device-0>,<reserved-device-1>
torchrun --standalone --nproc-per-node=2 "$(python3 -c \
  'import zercmoe,os; print(os.path.join(os.path.dirname(zercmoe.__file__),"tools","example_w8a8.py"))')" bf16
```

Then verify framework weight loading, backend selection, model outputs,
repeated calls and coordinated shutdown before comparing the original path
at identical workload settings.

CPU adapter and selection tests use a mock wheel to verify this boundary. The
private 1.5 wheel was not available in the development environment; its actual
NZ packing, SHMEM communication, EP16/EP32 topology, GLM accuracy and end-to-end
performance remain unverified by this change. Results from the separate
hand-written operator do not validate this backend.
