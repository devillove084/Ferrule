# Explicit BF16-output GemmEx backend

## Typed API and admission

This is a backend operator identity, not a model precision variant. No existing
numeric profile, default, model plan, chunk or recurrent implementation changed.

```rust,ignore
use ferrule_backend::cuda::operators::linear::Bf16GemmExPlan;

let plan = Bf16GemmExPlan::new(
    layout, rows, admitted_storage_bytes, admitted_external_provider_bytes,
)?;
let mut prepared = op.prepare_bf16_gemmex(plan)?; // outside capture, once
prepared.execute(&op, &numeric_fp8_weight, &bf16_input, &mut bf16_output)?;
```

- Packed BF16 activation/output; numeric FP8 weights with BF16 128x128 scales.
- M in1..=32; N/K positive multiples of128; origins zero. No expert ID or role
  appears in the native/provider arithmetic. Overflow, span, alignment and
  read/write overlap are rejected before native submission.
- BF16 A/B/**C**, F32 compute, GemmEx algorithm99, math16, atomics disabled,
  host alpha1/beta0. Output is native BF16, never rounded F32 GEMM output.
- CUDA12.4 build, actual sm86 device, and cuBLAS120405 runtime only. Other builds
  or runtime versions fail closed; no implicit fallback or toolkit change.

The plan separates these charges:

```
staging_bytes = 2 * ((M + N) * K + M * N)
cublas_workspace_bytes = 33,554,432
storage_bytes = staging_bytes + cublas_workspace_bytes
external_provider_reserve = 16,777,216  // additional, not scratch
```

Staging contains packed input, the **full decoded weight**, and BF16 warmup
output. Runtime writes directly to the caller's separate BF16 output. Immutable
encoded artifact storage, caller input/output, and allocator segment/grant
rounding are additional existing owner costs, not hidden inside this formula.
Budgets apply **per live prepared object**. Cache/reuse by exact owner and plan;
do not prepare a new handle on every projection call.

Example M1 down N2048/K512:2,102,272 staging bytes +32MiB workspace.
Example M1 concatenated gate-up N1024/K2048:4,200,448 staging bytes +32MiB.
M5 QKV N8192/K2048:33,656,832 staging bytes +32MiB.

Validated process setting: `CUBLAS_WORKSPACE_CONFIG=:0:0` **before cuBLAS handle
creation**. This disables the library-default pool, not this operator's explicit
32MiB workspace. With inherited `:4096:8`, the first bounded attempt exceeded the
16MiB external reserve and correctly failed closed. That failed attempt is
retained; its exact growth was not logged. The final code reports measured
admission growth in the error. Do not widen the reserve or silently change the
process environment. The admitted setting measured2MiB or10MiB external growth,
with no further growth after fixed-shape warmup.

## Ownership, capture and failure behavior

The private provider owns one handle, an exact stream lease and the complete
caller-budgeted storage allocation. Workspace is bound once, after stream setup,
and never rebound at launch. Initialization performs two fixed-shape warmups
outside capture, with a16MiB growth ceiling and a second-pass no-growth check.
Native launch performs BF16 input packing, full weight decode and GemmEx only;
there is no launch allocation, free, synchronization or handle construction.

Normal destruction proves stream completion and destroys the handle **before**
releasing its bound storage. Failed initialization is covered by the same owner.
Submission failure poisons the provider and marks existing context completion
unknown. Unknown completion/capture/destruction failure retains the bound
storage and stream rather than releasing a pointer cuBLAS may still use.

For capture, use the retained typed graph:

```rust,ignore
let graph = prepared.capture(&op, &weight, &input, &mut output)?;
graph.upload(&op)?;
graph.launch(&op)?;
```

The graph retains the provider and all operand allocations and can outlive their
original wrappers. Retiring the last graph retires its leases only after known
completion. Replays require the exact owner. Unknown replay/upload completion
is terminal and retains operands as well as workspace. Ordinary `execute`
inside an arbitrary unretained capture is deliberately rejected; this is not
an unowned pointer escape into a global model graph. A future larger graph
integration must retain equivalent leases, not bypass this guard. Surviving
aliases still require normal ordering against the last consumer.

## Frozen arithmetic gates

Final evidence: `target/validation/bf16-gemmex-backend-20261006/verified/`.
All payloads were checked against the original independent HF fixture. Inputs
and GPU-decoded BF16 weights were downloaded and compared word-for-word before
accepting output comparisons. Expected values remain original BF16 outputs.

- Eight blockers,16 repetitions each: **0/14,336 words differ per pass**, no drift.
- Original114 projection coverage: **0/209,920 BF16 words differ** using78 native
  calls: six dense projections,36 full-width concatenated routed gate-up, and36
  routed down projections. The final matrix is an explicit zero-difference gate.
- Concatenated expectations are only the original gate/up halves, interleaved
  per row for M2. Weights/scales are vertically concatenated; inputs must equal.
  No reference recapture, synthesized expectation or reference rewrite.
- The isolated-N512 diagnostic matrix was112/114 exact. Its M2 up differences
  were107/index432 `bc39` vsHF`bc38`, and112/index59 `39c8` vsHF`39c9`.
  Both disappear at HF's actual N1024 concatenated geometry. Thus integration
  must preserve native full width, not split gate/up into separate GEMMs.

This establishes the selected arithmetic on the frozen cases, not universal
bit-equivalence for all admitted shapes. No full40 or model-throughput claim.

## Validation and durable protocol

- CUDA12.4/sm86 build with source snapshot pinned **before compilation**.
- cuBLAS, cuBLASLt, cudart, binary, supervisor and fixtures pinned before GPU runs.
- Eight CPU tests: three new admission/native-policy tests, native pointer
  preflight without device calls, and four existing GEMV/profile regressions.
- Actual synthetic M1/2/5/16/32 across several N/K shapes; exact analytic results.
- GPU rejection: short/misaligned/aliased output, foreign owner, prepare during
  capture, and unretained capture. Checked no allocation/sync on execution and
  capture; no postwarmup memory growth on every frozen case.
- Retained graph survives dropping original provider and operand wrappers;
  three replays/shape pass and the last graph releases its provider lease.
- Injected graph-launch unknown state retains allocations, poisons reuse and
  rejects preparation. This is controlled failure injection, not a fabricated
  claim of provoking an asynchronous hardware fault.
- Isolated pinned cuBLAS120901 **negative** process rejects preparation. Ferrule
  was compiled with CUDA12.4, not CUDA12.9.
- Seven existing actual GPU graph lifecycle regressions passed. An earlier
  zero-test filter is preserved as historical evidence and not counted.
- `rustfmt`, native clang-format and diff-whitespace checks for changed code.

Each GPU plan used authenticated process custody and three idle checks after
execution. Fresh labels/directories are mandatory; existing evidence is not
reused as a new run. `backend_run.py` is the permission-gated bounded executor.
Final release and aggregate report are in `GPU-RELEASE.json` and `FINAL.json`.

## Reviewed shared-pool admission and terminal graph retirement

`Bf16GemmExPlan::new` now admits **physical allocator grant bytes**, exposed by
`storage_grant_bytes()`, rather than only `storage_bytes()` logical payload.
The additional 16 MiB external reserve is charged in a persistent stream-owner
ledger, not inferred from `cudaMemGetInfo`. Each stream owner admits one backing
pool and its physical grant/external reserve; graphs retain that reservation.
Another prepared object on the same owner is rejected until every graph and
backing lease retires. Independently budgeted owners may coexist on the same
GPU or different GPUs. Unknown retains only the affected owner's allocation,
handle and ledger entry; it is not a process-wide circuit breaker.

A short per-device mutex coordinates GemmEx preparation, including the complete
shape warmup/revisit. It does not live in the provider and is released on error
or unwind. It cannot serialize unrelated CUDA work or allocations. Admission
fences its owner stream and checks global free headroom against the new physical
grant plus all outstanding GemmEx external reservations on that device. Existing
physical allocations are already excluded from free bytes. Global memory deltas
remain diagnostics, not attributable provider accounting or a hard library cap.

The native startup snapshot requires literal `CUBLAS_WORKSPACE_CONFIG=:0:0`
before process execution; a later environment change cannot rehabilitate an
invalid startup. Both explicit preparation APIs reject invalid startup/current
configuration before allocating storage or creating a handle. The launcher
sets this environment on the child, never during preparation.

Graph result-copy/completion errors after preflight mark terminal Unknown.
Wrong-owner/capture/extent preflight remains safe rejection. Graph executable
and graph destruction errors now mark context Unknown before operand/provider
leases retire. Tests inject copy, completion, graph-destroy and exec-destroy
failures in isolated children and assert retained allocations plus blocked
replay/execute/preparation. No GEMM/FP8 arithmetic was changed by this review.

## Completion review: CPU-ready, GPU tests pending a separate grant

`CudaStream::synchronize` (including `CudaOperators::sync_stream`) and owned
`CudaEvent::synchronize/query` mark context Unknown on driver completion errors.
Owner/capture/binding preflight and `CUDA_ERROR_NOT_READY` do not poison healthy
owners. Existing graph destruction and retained-operand fixes remain in place.
Native initialization and shape warmups return distinct status **9** when their
synchronization fails. Rust consumes that status before attempting another
fence, retaining custody even if a subsequent fence would succeed.

New CPU tests: per-owner/per-device reservation coexistence, retained charges,
checked overflow, completion status classification and sticky native status.
New ignored GPU tests in `bf16_gemmex_owner_completion.rs`: independent same-GPU
owners, two-device coexistence (requires two visible GPUs, never silently skips),
external operator/stream/event-sync/event-query failure after graph launch,
pending/preflight retryability, and initial/shape warmup Unknown. The standalone
`bf16_gemmex_warmup.cu` injects failure into the **actual native sync call sites**
and proves the next real fence succeeds; Rust tests separately verify quarantine.
These tests have been prepared, not executed under the CPU-only task.
