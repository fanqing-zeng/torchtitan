# Kimi K3 NPU fused operators

This branch builds on `npu-moonep-kimik3` (`aa9172f8`). Kimi model flavors
enable the following implementations on NPU. CPU and CUDA retain their existing
reference paths; other model families keep their original configuration defaults.

| Operation | NPU implementation | Training gradients |
| --- | --- | --- |
| SiTU-GLU | `cann_ops_nn.situ_glu` / `situ_glu_grad` | Packed gate/up input, including `linear_beta` |
| RMSNorm | `torch_npu.npu_rms_norm` / `npu_rms_norm_backward` | Input and affine weight |
| Gated RMSNorm | `fla.modules.fused_norm_gate.rms_norm_gated` | Input, sigmoid gate, and weight |
| Short convolution / causal_conv1d | Public `fla_npu.ops.ascendc.causal_conv1d` / `causal_conv1d_bwd` | Input, filter, and optional history |
| KDA | Public `fla_npu.ops.ascendc.chunk_kda_fwd` / `chunk_kda_bwd` | Q/K/V, raw gate, beta, `A_log`, `dt_bias` |
| Attention residual | Local Triton-Ascend kernels | Partial block, all history blocks, projection, norm weight |

ShortConvolution and causal_conv1d share one AscendC adapter. Q/K/V filters
retain their checkpoint ownership and are concatenated only for execution.
The convolution backward API requires its preactivation, so forward saves the
AscendC convolution result and applies SiLU; backward fuses the SiLU derivative.
No Triton short-convolution path is selected on NPU by these flavors.

The FLA `fused_attnres` migration is available as an explicit comparison backend.
The default local implementation reads strided `[tokens, blocks, hidden]`
history directly, accumulates in FP32, and reduces parameter gradients in a fixed
order without atomics.

## Dependencies

Use a Linux Ascend environment with mutually compatible PyTorch, torch-npu,
CANN and Triton-Ascend builds. Install the model's existing dependencies, then
the NPU-capable FLA package, fla-npu and ops-nn built for the target chip.
Installing upstream CUDA-only Triton or FLA is insufficient for the NPU paths.

The AscendC calling conventions are adapted from MindSpeed-MM commit
`901bf0db2b66322d3c1c21ca6698881a04134a02`, specifically
`mindspeed_mm/fsdp/ops/kda/ascendc/chunk_kda_ascendc_kimi.py` and
`mindspeed_mm/fsdp/ops/gdn/causal_conv1d_ascendc.py`. Its Kimi installation guide
references fla-npu revision `3167fe36bd1`; the required public API, rather than
a guessed wheel version, is checked by the test runner. Rebuild the matching
operator package when changing fla-npu; old private `_aclnn_ctypes` APIs are no
longer used here.

Required interfaces are listed in `scripts/run_kimi_fused_ops_tests.py`.
The optional FLA attention-residual backend also requires the
`FusedAttnresFunction` contract documented in `attn_res.py`, including the
two-result internal forward used by its SPMD reference decomposition.
Explicitly selected fused backends fail on missing dependencies.

## Select implementations

The factory enables fused NPU defaults before applying model converters:

```python
from torchtitan.models.kimi_k3 import build_model_config, configure_fused_ops

config = build_model_config("debugmodel", seq_len=128)

# Override individual operators before constructing the model.
configure_fused_ops(
    config,
    situ_glu="ascendc",       # eager / ascendc
    rms_norm="ascendc",       # eager / ascendc
    gated_rms_norm="fla",     # eager / fla
    causal_conv1d="ascendc",  # eager / ascendc
    kda="auto",              # auto / ascendc / fla
    attn_res="triton",        # eager / triton / fla
)

# Original configuration for controlled comparisons. Its existing KDA backend
# still uses auto selection; the other operators use their eager definitions.
reference = build_model_config("debugmodel", use_npu_fused_ops=False)
```

The MoonEP transform preserves the configured SiTU activation in its existing
activation callback. Parameter names, shapes, initialization and checkpoint
ownership remain the same. Shared operators retain their existing SPMD type
semantics through registered PyTorch decompositions.

## Supported contracts

- AscendC KDA training supports Q/K head dimension 128, value dimension 128 or
  256, equal Q/K/V head counts, FP16/BF16 operands, and chunk size 64. Packed
  input uses batch size 1. Fused gate backward supports at most 1024 packed
  chunks, matching the source adapter's restriction.
- `auto` uses FLA for the existing head-dimension-64 debug path and on non-NPU
  devices; on NPU, head dimension 128 selects AscendC. Choosing `ascendc`
  explicitly requires its supported shapes.
- AscendC KDA saves its intermediates (`disable_recompute=True`). Initial
  recurrent state, final-state output/gradients and state-layout transposition
  are unsupported. Unsupported options are rejected instead of discarding
  gradients. Outer activation checkpointing can replay the complete operator.
- AscendC KDA does not provide the recurrent-state gradients required by CP.
  Use context parallel degree 1 for the fused KDA route. The CP override rejects
  NPU `auto`/`ascendc` selection instead of silently using a different kernel.
  Existing Attention Gym CP remains a separate path.
- Short convolution supports packed document boundaries and differentiable
  history `[sequences, channels, width]` or `width-1`; the Kimi training path
  requests no final state. It does not cross document boundaries.
- Attention residual supports optional partial blocks, empty completed-block
  history when a partial block is present, noncontiguous inputs, FP32 parameter
  weights with low-precision activations, and all first-order training gradients.
  At least one residual source is required. Kernels preserve the supplied eps.

## Tests and validation

Run the CPU references, public-API mocks, gradients, SPMD contracts, checkpoint
and model-configuration tests:

```bash
python scripts/run_kimi_fused_ops_tests.py -q
```

Run the same tests plus real device kernels and model composition on Ascend:

```bash
python scripts/run_kimi_fused_ops_tests.py --npu -q
```

`--npu` requires an accessible device and all required APIs before running pytest,
so an absent runtime cannot produce an all-skipped success. Direct pytest
collection on a CPU host skips hardware-only modules.

The CPU tests are under `tests/unit_tests/cpu`, which the existing CPU CI job
collects. Hardware tests are under `tests/unit_tests/npu`; they require a separate
Ascend runner. CPU mocks verify translation and gradient plumbing, not device
kernel arithmetic. The small model test isolates the composition of these fused
operators; its MoonEP callback case does not validate multi-rank communication.

The handwritten attn_res tests compare a FP64 oracle, the same-input-dtype eager
reference and the real kernel. They check all gradients, tile boundaries,
production hidden dimension 7168, extreme scores, near-zero variance,
cancellation, strides, checkpointing and repeatability. Error thresholds are
declared before execution; ill-conditioned comparisons are reported separately.

Before accepting performance results on a target stack, run device numerics,
then compare training loss and grad_norm with seed 42 and deterministic mode
using `scripts/loss_compare.py`. Verify convergence on representative data for
these computation changes. Measure operator forward/backward time, peak memory
and training throughput after warmup; training benchmarks must run at least
10 steps. Record chip, CANN/torch-npu/FLA/fla-npu/Triton revisions, shape, dtype,
packing and parallelism in the results. This change has no measured NPU speedup
or device numerical claim until those runs are performed.

### Local verification record (2026-10-16)

On Windows with Python 3.11 and PyTorch 2.14.1+cpu:

- The focused runner passed all 225 CPU tests, including cold fullgraph
  registration, independent gradients, packed boundaries, SPMD annotations,
  recomputation, configuration propagation and checkpoint compatibility.
- Existing RMSNorm, gated RMSNorm, Kimi pipeline and MoonEP tests produced
  114 passes and three failures. Two pipeline tests call the existing
  `_make_forward_chunk_state` method absent from this PyTorch build; the stage
  and test files are unchanged from the base branch. One recipe-import test
  requires the unavailable `grain` dependency. Broader feed-forward/offset-norm
  test collection also requires missing CUDA/CuTe or data dependencies.
- Changed files passed the applicable pre-commit hooks (formatting, flake8,
  pydoclint, spelling, license and syntax checks). Targeted Pyrefly with the
  test environment's interpreter reported zero errors. The full
  `pre-commit run --all-files` did not pass: existing files have formatting,
  license and lint issues, global type checking lacks required dependencies,
  and the link-checker hook requires `/bin/bash`, unavailable on this host.
  Hook changes unrelated to this feature were reverted.
- Direct device-test collection skipped on this CPU host. The `--npu` runner
  failed as expected on missing `torch_npu`. Real kernel compilation, device
  numerics, distributed training, convergence and throughput remain unverified.
