# Mixed BF16 execution-boundary fault component

## Status and scope

The two ignored selectors below are **implemented and CUDA-build validated, not
GPU-executed in this change**. CPU execution cannot qualify them. Each selector
runs Blocking and Lookahead, giving four eventual GPU cells. There is no new
public flag, CLI path, runtime/composition hook, or production fault policy.

This is an actual `GenericDecoderRunner<HybridCudaDecoder>` /
`CudaHybridModule` component test, not an integrated CLI E5 experiment or a
35B production-serving fault guarantee. It does not change or requalify any
candidate/quality inventory, full-model numeric reference, or tolerance.

The old `bf16_chunk` lifecycle selectors and the F32
`hybrid_unknown_finish_keeps_transaction_pins_despite_independent_kv_fence`
regression remain unchanged. They are useful narrower regressions, **not** the
new Mixed BF16 expert-read fault cells.

### Fixture and actual work

- Reuses `fixture.rs` / `hf_numeric_profile.json`, seed **3501635**; no user input
  or production weights. Hidden/intermediate width 16, vocabulary 32, two layers
  (GatedDeltaNet then GQA), four experts/layer, top-3, shared expert.
- `Bf16ModelMixedF32Accumulate`; numeric FP8 expert weights and BF16 scales,
  genuine BF16 history/KV/activations, F32 recurrent SSM, 512 state bytes per
  sequence. It uses the existing CUDA operator/kernel selection, not CPU math
  or a replacement model. These tiny shapes/weights are not SOTA accuracy or
  performance evidence; they need not select every kernel used at 35B scale.
- Real bounded cache: one expert, 64 KiB logical cache budget, 4096-byte
  persistent numeric scratch reservation. Real `ExpertReadPipeline` in both
  modes: 2 MiB host bound, 256 KiB metadata, 256 KiB requested worker stack.
- Transaction 1 executes and publishes token `[1]`, establishing frontier **1**.
  Transaction 2 continues with `[2, 3]` and real KV COW. A native borrowed
  diagnostic callback raises an ordinary error after the first routed **down**
  projection in layer 1 submits. Both attention layers have been reached.
- The callback verifies BF16 output storage, an increase in actual compute
  launches across that projection, and the matching source-verified worker
  receipt. The admitted expert identity is joined to the read's image,
  layer/expert, and operation ID. CUDA upload/cache counters are also recorded.
  Owner-wide counter deltas are not claimed as per-ticket syscall attribution.

The fault itself is **scheduled/synthetic**. In the Unknown cell, the private
`cfg(test)` observer uses the existing consumer-proof failure seam. The real
compute event wait and upload sync run first; their results are recorded, then
the successful proof is deliberately discarded. The assertions require both
real waits to have succeeded. This models unavailable evidence, not a hardware
crash, illegal kernel, stalled GPU, or real transport-induced loss.

### Assertions and limitations

Clean:

1. Ordinary error, no typed Unknown/quarantine; host sequence teardown joins
   the worker and consumes/cancels its receipts.
2. The actual transaction abort retires working state and pending KV/COW pages.
   Original topology/generation/frontier remain unchanged.
3. Explicit, separately logged D2H/preempt/restore actions compare committed
   GDN and KV bytes bitwise with the frontier-1 baseline. Passive samples do
   **not** download state.
4. Operation holds and pending upload/Unknown charge retire. Persistent image
   scratch and default + committed state legitimately remain. Retry succeeds.
5. Explicit page/state release, shutdown and final owner Drop precede a logged
   allocator trim/measurement. Reclamation is compared with this fixture's
   pre-image allocator baseline, not device-wide free memory.

Unknown:

1. Forward and repeated abort return typed `CompletionUnknown` after actual
   read/upload/compute work. Host cleanup may independently finish.
2. Working state, active transaction, KV pins, cache charges, scratch and held
   BF16/route operands remain. Unknown bytes are a **subset**, not another
   additive cache bucket.
3. Independent compute/upload events can complete but cannot discharge owner
   custody. Ordinary reset, new sequence admission, KV release, publish and
   shutdown cannot clear the sticky owner state.
4. Topology/generation/frontier and retained state-byte counts are checked.
   **Post-Unknown KV/SSM contents are unobserved**: a fresh device read is not
   authorized by this owner's lost proof. No bitwise claim is manufactured.
5. Runner, states and device are deliberately retained until test-process exit.
   No forced-zero allocation assertion and no post-Drop release claim.

Snapshots call only existing passive observations; repeated samples must be
identical and leave CUDA counters unchanged. They never poll/join a worker,
query an event, trim an allocator, or drive cancellation. Operational actions
and receipt evidence have separate JSONL record kinds. Asynchronous worker
progress is not inferred to be GPU overlap.

The accounting equations are scoped:

- Cache: `budget = free + resident + pending + scratch`.
- `unknown_subset <= resident + pending + scratch` (never added again).
- Host: `host_charged = packed + artifacts + metadata + requested_stack`.

There is **no runtime broker** in this fixture. Five-bucket broker accounting,
allocator-grant conservation, full-process RSS, cross-request admission pressure,
and integrated serving cancellation are not reconstructed from these ledgers.

## Exact ignored selectors

Both use `-p ferrule-model --lib --no-default-features
--features cuda,ferrule-common/nvtx-profile`:

```text
transformer::forward::tests::cuda::mixed_fault::mixed_bf16_clean_fault_after_real_expert_submit_both_schedules
transformer::forward::tests::cuda::mixed_fault::mixed_bf16_unknown_after_real_expert_submit_both_schedules
```

For a separately authorized sole GPU owner, run **one exact selector per
process**, with `-- --ignored --exact --nocapture --test-threads=1` and an
external bounded timeout. Do not run them during another profile owner's GPU
reservation. Unknown intentionally retains memory until that process exits.
`CUBLAS_WORKSPACE_CONFIG=:0:0` must be set before initialization. This handoff
does not select or authorize a physical GPU.

### JSONL evidence

Set the experimental **test-only** environment variable
`FERRULE_COMPONENT_FAULT_REPORT_DIR` to an existing empty output directory.
The files `mixed_clean.jsonl` and `mixed_unknown.jsonl` are opened with
`create_new`; an existing file is an error, never overwritten. Without the
variable, the same records go to stderr with `COMPONENT_FAULT` prefixes.

Records include header/config, running case, physical action, passive snapshot,
pre-consumer boundary + actual wait results, host cleanup receipt, case outcome
and final seal. A header starts with `gpu_execution: unobserved`; a running case
is not a pass. Only both completed schedule cells permit a seal. An assertion
failure or interrupted process leaves no successful seal. The seal is an
append-only completion marker, **not a cryptographic attestation**. Report
construction/CPU tests never emit observed GPU case outcomes.

## CPU-only validation and hidden CUDA build

CPU commands used (all with `CUDA_VISIBLE_DEVICES=''`,
`NVIDIA_VISIBLE_DEVICES=none`, `CARGO_INCREMENTAL=0`, four build jobs):

```sh
cargo test --locked -p ferrule-model --no-default-features --test hybrid_bf16 --test expert_read_constructor --test mixed_decoded_budget -- --test-threads=1
cargo test --locked -p ferrule-model --no-default-features --lib transformer::forward::tests -- --test-threads=1
cargo test --locked -p ferrule-model --no-default-features --lib transformer::expert_reads::tests -- --test-threads=1
cargo test --locked -p ferrule-model --no-default-features --lib checkpoint::read_pipeline -- --test-threads=1
```

The exact shared GPU fixture options have a CPU admission test, including both
schedules and identical Mixed memory plans. This caught an initially undersized
64 KiB metadata allowance; the test fixture now uses the admitted 256 KiB bound.
Other new CPU contracts reject over-budget/overflow/double-counted ledgers,
missing GPU evidence, and overwriting existing reports. Existing real-worker
CPU cancellation and forward mock regressions are tested without relabelling
them as CUDA execution.

Hidden compile/link command (CUDA **12.4.99**, explicit `sm_86`, NVTX feature
consistent with common; no test execution):

```sh
env \
  PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin \
  CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao \
  NVCC=/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc \
  LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64 \
  LIBRARY_PATH=/home/admin/luhuanbing.lhb/Ferrule/target/validation/composition-refactor-cuda124-link:/usr/lib64 \
  FERRULE_NVTX_INCLUDE_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nvtx/include \
  FERRULE_PROFILE_NVTX=0 FERRULE_CUDA_TEST_ORACLE=0 FERRULE_CUDA_ARCH=sm_86 \
  CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none CUBLAS_WORKSPACE_CONFIG=:0:0 \
  CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=4 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test --locked -p ferrule-model --no-default-features \
  --features cuda,ferrule-common/nvtx-profile --lib --no-run
```

The pre-existing link directory points to the external trusted CUDA 12.4 runtime
libraries, not a fake provider. `CARGO_PROFILE_TEST_DEBUG=0` reduces test-binary
debug information; it does not alter fixture arithmetic. The CUDA binary can be
listed (`--list`) and its `mixed_fault::contract` CPU tests run with GPUs still
hidden. Neither action executes the ignored GPU cells.
