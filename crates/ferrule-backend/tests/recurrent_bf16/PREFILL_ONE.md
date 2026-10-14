# Explicit one-row prefill chunk arithmetic

The actual Hello first-prefill fixture at layer9 isolates a semantic schedule
mismatch, not a recurrent-kernel defect. Inputs/projections and initial F32 state
are exact. Cached recurrence matches pinned HF recurrence, but differs from the
original HF prefill chunk output at two of4096 BF16 words (1575/1596). The new
explicit chunk entry matches all4096 output and524288 F32 state words. Layer and
indices are test provenance only, never dispatch conditions.

## Integrator API

```rust,ignore
use ferrule_backend::cuda_plan::{Bf16ChunkExecution, Bf16ChunkPlan};

// Select using the actual execution phase, never row count or state contents.
let prefill = Bf16ChunkPlan::new_prefill(layout, scratch_budget_bytes)?;
assert_eq!(prefill.execution(), Bf16ChunkExecution::PrefillChunk);
// Allocate/retain scratch in the existing image/owner lifecycle, not submission.
ops.gated_delta_net_bf16_chunked_into(
    buffers,
    Some(CudaBf16ChunkWorkspace::new(&mut scratch, prefill)?),
    prefill,
)?;

// Backward-compatible cached single-row behavior:
let cached = Bf16ChunkPlan::new(single_row_layout, 0)?;
assert_eq!(cached.execution(), Bf16ChunkExecution::CachedRecurrent);
ops.gated_delta_net_bf16_chunked_into(cached_buffers, None, cached)?;
```

- `new_prefill` admits rows>=1 with the existing chunk dimension/overflow limits.
  At Hv32/Dk128/Dv128 it requires4,462,592 F32 scratch elements (17,850,368 bytes)
  even for one row, and reports15*ceil(rows/64) launches. It does not zero state.
- Existing `new(rows=1)` remains zero scratch/one recurrent launch; existing
  `new(rows>1)` remains the same chunk plan as `new_prefill`.
- Plan equality includes execution kind. A cached plan cannot borrow chunk scratch
  or submit a prefill workspace wrapper; absent/short/foreign/mismatched workspace
  is rejected before submission. Existing writable alias/owner checks remain.
- Native `ferrule_core_bf16_chunk_prefill_launch` admits rows>=1. The old
  `ferrule_core_bf16_chunk_launch` still rejects rows1. Both call the same checked
  15-launch implementation. Descriptor size104, offsets, reserved=0 and ABI
  layout are unchanged; no flags or unchecked raw-buffer public API were added.
- The caller must budget scratch for possible one-row prefill, select by phase,
  and retain operands/scratch through completion or Unknown quarantine. Partial
  failure is not rollback. No added host allocation/copy/fence in submission.
- No precision profile/default model switch, recurrent/norm/math edit, per-index
  correction, reference substitution, tolerance widening or CPU fallback.

## Dedicated evidence

`target/validation/gdn-prefill-one-20261007/FINAL.json` records source pins,
commands, receipts and release. CUDA12.4/sm86 native and the pinned
Torch2.13cu129/Transformers5.5.3 oracle run in separate authenticated processes.

Before implementation, all prepared 19-to-15 fusion gates passed:10 baseline,
7 continuation/reset,14 matrix,120 independent stages,49 fused-panel and3 Rust
resource/API cases. The original orchestration used a Python without Torch;
GPU children had passed and drained. Comparison was completed with pinned Python.
A redundant HF retry encountered an existing output manifest and was drained;
neither issue was a numerical failure or a reference change.

New gates:
- Actual layer9 output/state exact; cached execution preserves the independently
  verified two-word recurrence-vs-chunk distinction, not an altered cached result.
- Five nonzero-F32-state synthetic geometries (Dk1/17/128/511/512) each execute
  chunk1 -> chunk5 -> chunk1, same dirty scratch/state, no inter-call host fence.
  All15 calls match the unmodified pinned HF chunk function exactly.
- Four retained cached controls: Hello and Capital decode0 layers0/1, output and
  F32 state exact. Independent HF replays must reproduce original references.
- Eight typed Rust GPU tests pass, including exact actual prefill,15 vs1 resource
  counters, missing/short/foreign/mismatched scratch rejection, cached delegation,
  actual Hello decode, and all-operand owner/extent validation.
- Final source rerun:49 fused-panel cases and120 independent HF stages exact,
  including dense129 production geometry and Dk511/K512 matrices.

The new native kernels use exactly the previously verified fusion arithmetic;
only checked admission and dispatch changed. No full-model run. A matched
unfused/fused timing run was not available as a prepared gate (the existing timer
compares scan/chunk), so no new latency or throughput claim is made.

To prepare a fresh bounded rerun (receipt/output directories must not exist):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/prefill_one.py prepare target/validation/gdn-prefill-one-rerun
# Sole GPU owner only; each child bounded120seconds, authenticated drain/idle.
/opt/conda310/bin/python -B target/validation/gdn-chunk-splitk-20261006/owned.py target/validation/gdn-prefill-one-rerun/jobs.json target/validation/gdn-prefill-one-rerun-receipts
```
