# Matched 15-launch versus 19-launch chunk timing

CPU-prepared only. No GPU timing has run in this cycle; another agent owns GPUs.
No production source, reference, tolerance or workspace changes.

`chunk_fusion_performance.cu` reuses the existing chunk continuation fixture
buffers, events, guards and stream helpers. Both headers compile into the same
CUDA12.4/sm86 binary. The archive is test-only `chunk_unfused.cuh`, SHA256
`2182f58b6f5543d681dc6fbd1873bb0d4de1e27a3b4e265f85af001af1c5d0f7`.
It contains the final prefix-window, double-to-F32 query-scale, triangular and
split-K fixes. It is not an earlier incorrect kernel. Preparation verifies
unchanged normalization/prepare/prefix/matmul/triangular functions, and pins the
current header against the accepted explicit-M1 backend evidence. Timing invokes
only the archived chunk entry and current prefill chunk entry, never scan.

## Matched design

- Actual5, dense64 and dense129; all Hk16/Hv32/Dk128/Dv128.
- Original retained inputs and independent HF output/state hashes verified;
  references are copied/converted losslessly, not regenerated.
- Same input, state, output, scratch addresses and nonblocking owner stream for
  both variants. State reset before each invocation; scratch remains dirty.
- Two untimed complete correctness gates before warmup (one per variant).
- Ten independently reset warmups per variant/case, alternating order.
- Thirty AB/BA pairs per case: 90 pairs, 180 individually timed invocations total.
  Each variant runs first15 times per case; initial order alternates by case.
- CUDA events enclose submission only. State D2D reset precedes the start event;
  every output/state word is checked against the same independent HF reference
  after the stop event. Input immutability and guards are checked every sample.
- Runtime register/local/shared/max-thread attributes for18 old/new kernels and
  raw PTXAS stack/spill/register output retained. Timing summary includes raw
  samples, paired deltas, median and p95; no throughput projection.

There are three GPU processes, each bounded60seconds. The existing authenticated
supervisor wrapper is SHA-pinned, as are its subreaper, CUDA libraries, compiler,
source dependencies, fixture files and executable. Before any GPU child it
requires all eight devices idle. After authenticated custody/ECHILD it takes
three idle samples and records release. The outer wrapper wait is bounded240s;
on any failed launch/receipt/timeout, stop and inspect custody before retrying.

## Ready command — sole GPU owner only

Run from the repository root **after explicit ownership is granted**. The receipt
directory must not exist; reruns require a fresh receipt directory.

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export NVIDIA_VISIBLE_DEVICES=all
/opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_fusion_performance.py run target/validation/gdn-chunk-fusion-performance-20261007/prepared --evidence target/validation/gdn-chunk-fusion-performance-20261007/run --execute-cuda
```

Sources are pinned. If a dependency changes, do not relax the pins: prepare a
fresh directory with GPUs hidden and review the new source equivalence check.

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
/opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_fusion_performance.py prepare target/validation/gdn-chunk-fusion-performance-rerun/prepared
```

## CPU readiness checks

- Six dedicated benchmark pin/coverage/order/ownership-guard tests pass.
- Six current backend chunk-plan tests pass; four GPU tests remain ignored.
- CUDA12.4/sm86 benchmark builds without warnings; host-only native ABI/contracts
  execute without device access.
- Current backend CUDA all-tests clippy passes with `-D warnings` (zero warnings).
- No GPU query, workload, ownership acquisition, drain or release in preparation.

Evidence: `target/validation/gdn-chunk-fusion-performance-20261007/READY.json`.
The existing49/120 fusion gates are not rerun here; this adds only the three-case
matched performance experiment.
