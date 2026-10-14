# Qwen3.5-35B numeric CUDA PP2 × EP

This is the TP1 segment-provider leaf of shared parallel composition, using
**thread** owners, not an independent whole-factory construction algorithm or an
extension of the dense process Boot protocol or a generic `PP * TP * EP` mesh. It accepts the strict full-depth
Qwen3.5-35B-A3B numeric FP8 checkpoint. It does not execute vision/MTP, CPU MoE,
TP, process PP×EP, partial `--max-layers`, restart/replay, or local CUDA expert
cache overrides. Unsupported combinations are rejected rather than falling back.

TP uses a tensor-rank provider leaf with its own geometry and budget contract;
binding, admission and assembly share the same composition flow. See
[CUDA TP×EP commands and acceptance](../crates/ferrule-runtime/tests/support/hybrid_cuda_tensor.md);
its full35B serving commands explicitly require `--max-tensor-mb 2048`.
That provider supports thread PP1/2 TP2/4 EP2; its full-model strict GPU evidence
currently covers TP2 EP2 only (PP1 and PP2). TP4 support and W8A16 experimental
opt-in must not be presented as strict quality acceptance. The TP1 PP/EP evidence
in this document is independent; it does not expand the TP×EP capability gate.
See also the [README support matrix](../README.md#support-matrix).

## Ownership and admission

- Two root/KV owners: global layers `0..20` and `20..40`. Each has five compact
  full-attention KV planes and fifteen recurrent layers.
- Each stage has EP2, EP4 or EP8 **additional** expert owners. Root IDs are 0/1;
  child IDs follow in stage order. Only the two roots enter the existing KV
  transaction cohort. Expert tickets carry its transaction, root source, stable
  session IDs and global layer/expert coordinates.
- `--devices` is required and has `2 + 2*EP` entries: root0, root1, stage0's
  experts, then stage1's experts. Repeated ordinals are allowed; every root and
  child context, resident weight, scratch reservation and allocator margin is
  charged to the same physical-card ledger before any uploads.
- One immutable full-model binding and one fully prewarmed compressed host image
  are shared. Roots do not rebind a partial model or renumber experts. Source
  freshness is checked before preparation. Actual child Ready statistics must
  match admitted ownership and residency; forward checks reject uploads,
  evictions or changed residency.
- Host admission includes simultaneous root prepared bindings/upload temporaries,
  child metadata, shared compressed weights, worker stacks and checked full-vocab
  F32 activation/logit transport holds. KV uses logical L / physical 2L capacity.
- The current model estimator has no routed-segment weight/workspace ledger.
  Consequently **each root is conservatively charged the complete external-root
  weight/workspace upper bound**. State and KV are stage-exact. This may reject a
  tighter placement that could fit with a future model-owned segment estimator;
  it does not undercharge that placement.
- Existing `PipelineParallelExecutor`, `PipelineStageWorker` and
  `DistributedTransaction` still own forward/commit/rollback. Rollback and abort
  cannot proceed before root **and** child quiescence. Pending retains the worker
  journal; Unknown is quarantined, not converted to successful cleanup.

Numeric storage is E4M3FN plus original numeric scales. Omitted
`--numeric-fp8-precision` means **F32Tf32x3**. `w8a16-bf16` is explicit opt-in;
model, activation, KV, state and output boundaries stay F32 for both profiles.

## Serving commands

Run from the repository root. Replace the example checkpoint directory with the
actual strict checkpoint. Build for the real GPU architecture; `sm_80` below is
an Ampere example, not a hardware auto-selection policy.

```sh
FERRULE_CUDA_ARCH=sm_80 cargo build --release -p ferrule-cli --features cuda

# Four cards: root0+stage0/expert0 on CUDA0, root1+stage1/expert0 on CUDA2.
# Six logical owners, two KV participants. F32 default.
target/release/ferrule serve /path/to/Qwen3.5-35B-A3B-FP8 \
  --backend cuda --engine pipeline --rank-backend thread \
  --pipeline-parallel 2 --tensor-parallel 1 --expert-parallel 2 \
  --devices 0,2,0,1,2,3 \
  --ctx-size 1024 --max-active-sequences 2 \
  --max-batch-tokens 4 --prefill-chunk-size 4 --kv-cache-mb 1024 \
  --expert-prewarm full --max-tensor-mb 1024 \
  --served-model-name qwen35-pp-ep --port 8000
```

For explicit W8A16, add `--numeric-fp8-precision w8a16-bf16` to the same command.
An eight-card PP2×EP4 placement is
`--expert-parallel 4 --devices 0,4,0,1,2,3,4,5,6,7`.
Placement length is not the number of distinct cards. Live RAM/VRAM admission
must pass; these examples do not guarantee capacity on an arbitrary card.
Thread mode retains the existing default `--rank-timeout-ms 30000`; custom thread
timeouts are rejected. Do not use `--rank-backend process` for this composition.

After the server reports ready:

```sh
curl --fail http://127.0.0.1:8000/readyz
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-pp-ep","prompt":"Hello","max_tokens":8,"temperature":0,"stream":false}'
```

## Tests for the GPU owner

The three original real-GPU smoke/tiny tests are ignored by default. They fail on missing hardware,
checkpoint, admission failure or numerical mismatch; they do not skip into CPU.
Do not hide GPUs for these commands.

```sh
# Two GPUs; four-layer GDN/GQA numeric fixture. Both profiles, full logits
# compared with the monolithic resident EP2 path, then fork/cancel/release/close.
FERRULE_CUDA_ARCH=sm_80 cargo test -p ferrule-composition --features cuda --lib \
  pp_ep_gpu_tiny_matches_monolithic_ep_full_logits_and_custody \
  -- --ignored --nocapture --test-threads=1

# Full 40-layer public factory, generation, retained-session reset,
# cancellation and shutdown. Both default F32 and opt-in W8 tests are selected.
FERRULE_QWEN35_PP_EP_MODEL=/path/to/Qwen3.5-35B-A3B-FP8 \
FERRULE_QWEN35_PP_EP_DEVICES=0,2,0,1,2,3 FERRULE_CUDA_ARCH=sm_80 \
  cargo test -p ferrule-composition --features cuda --lib \
  pp_ep_gpu_full35b_public_factory_ -- --ignored --nocapture --test-threads=1
```

The full35B tests are real factory/lifecycle smoke tests, **not an independent
full35B numerical oracle certification**. The tiny test compares full logits
against the existing monolithic EP implementation. Only a matching GPU-owner
receipt establishes execution; compiling ignored tests is not numerical acceptance.
The later pinned strict/tiny run below does not turn a lifecycle smoke into an
independent oracle or validate a subsequently rebuilt binary.

## Maintained full40 F32 strict acceptance

The actual `pp-ep-owner` PP2 TP1 EP2/EP4 and `runtime-tp-ep-owner` PP1/2 TP2 EP2
snapshot harnesses have been migrated into a shared composition-test suite:
[full40 strict tests and exact commands](../crates/ferrule-composition/tests/unit/engine/model_factory/full40_strict/README.md).
The private `#[path]` adapters invoke the real factory admission/executor, with
one common independent-oracle reader/comparator and one lifecycle harness.

The maintained tests require manual opt-in, explicit model/reference/device/report
paths and `--ignored`. All use explicit `ServingModelOptions` 2048 MiB tensor
budget, unchanged F32 `atol=rtol=2e-4`, all vocabulary/all rows, and retained
fork/recreate/pre-cancel/release/shutdown assertions. PP routes/raw state tensors
are not directly compared; the existing resident EP8 strict tests retain those
trace comparisons through the same helper. No W8 gate or production capability
was changed. The maintained four-topology suite subsequently passed in the
[pinned post-refactor GPU-owner run](../target/validation/parallel-refactor-readiness-20261007/gpu-owner-current-20261007/ACCEPTANCE.json).
See [current acceptance](CURRENT_ACCEPTANCE.md) for counts and exclusions. That
receipt validates its pinned sources/binary, not later edits or every admitted degree.

## CPU / hidden-GPU validation

```sh
cargo test -p ferrule-composition --lib pp_ep_
cargo test -p ferrule-composition --features cpu-reference --lib model_factory
cargo test -p ferrule-runtime --features cpu-reference \
  --test hybrid_pipeline_state --test pipeline_program
cargo test -p ferrule-composition --features cpu-reference --test hybrid_pipeline
cargo test -p ferrule-cli --features cpu-reference --bin ferrule commands::serve::tests

# With CUDA 12.4 nvcc and its development/runtime libraries on the environment:
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 \
  cargo check -p ferrule-runtime -p ferrule-composition -p ferrule-cli --features cuda --tests
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 \
  cargo test -p ferrule-composition --features cuda --lib pp_ep_
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 \
  cargo test -p ferrule-cli --features cuda --bin ferrule commands::serve::tests
```

Use a sufficiently long build timeout (up to 1,800,000 ms). No GPU execution is
performed by the hidden-GPU admission/CLI tests. They cover policy selection,
immutable profiles, root/child identity, wire overflow, checkpoint freshness,
aggregate colocation, and preservation of Pending/Unknown cleanup custody.
