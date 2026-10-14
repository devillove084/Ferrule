# Explicit Mixed GemmEx integration gates

Only `Bf16ModelMixedF32Accumulate` selects native BF16-output GemmEx. All other
profiles and defaults are unchanged. `linear_plan` remains the F32-output
fallback; the BF16 owner selects `gemmex_plan` first. Admissibility is not a
universal Hugging Face bit-equivalence claim.

## Ownership and budget

One eager image-owned provider admits a finite set of M/N/K layouts, prepares
all shapes before invocation, and revisits the complete set to reject further
provider growth. The workspace is bound once at the largest staging offset;
execute never rebinds or allocates provider storage. The separate external
16 MiB ledger reservation covers the entire shape set; device-global free-memory
deltas are diagnostics, not attributable cuBLAS allocation accounting. `CUBLAS_WORKSPACE_CONFIG=:0:0`
must be set by the process before preparation; the model never sets it.
The backend now retains physical/external charges per exact stream owner, with
short per-device preparation coordination rather than process-wide lifetime
exclusion. Independent owners/devices coexist; an Unknown owner retains its
own charge without blocking other healthy owners. GPU tests for this revision
remain pending the next explicit grant.

The 35B M<=8 metadata gate charges 67,272,704 logical storage bytes,
69,206,016 allocator-grant bytes and 16,777,216 external bytes once. The replaced
M1 GEMV provider is not prepared or charged for selected projections. Current
workspace demand is 817,594,444 bytes against 2,155,872,256; the unchanged
7 GiB admission reserves 6,846,743,296 bytes, leaving 669,449,472 bytes with the
original 536,870,912-byte margin already included. Unsupported shapes and the
output head/router retain their old path and cannot inflate this pool.

Routed gate/up retain the original separately validated numeric artifacts and
canonical metadata/cache lease. Both decode into adjacent BF16 staging rows;
one native-width GEMM writes BF16, followed by a word-preserving split. No
host payload concatenation, expert-ID selection, independent half-width GEMM,
or F32-output-and-cast substitution is used. Same-stream sequential reuse is
covered by the existing synchronous/lease consumer proof. Unknown retains the
pool, artifacts and input/output holds; it cannot be cleared by a later sync.

## Required GPU order (not run by the implementation task)

Use fresh authenticated supervisor evidence, CUDA12.4/sm86/cuBLAS120405,
`CUBLAS_WORKSPACE_CONFIG=:0:0`, and explicit sole-GPU permission. Never invoke
these ignored tests with a broad filter. Each initial gate is capped at 120s:

1. Backend: `cuda::providers::bf16_gemmex::view_tests::bf16_gemmex_shared_shapes_resource_gate`
2. Model prefix `transformer::standard::cuda::bf16_hybrid::gemmex_tests::`:
   - `mixed_gemmex_prepare_failure_clean_retry`
   - `mixed_gemmex_clean_failure_reuses_pool`
   - `mixed_gemmex_unknown_retains_pool_and_operands` (isolated process)
   - `mixed_gemmex_synthetic_256_model_paths`
   - `mixed_gemmex_frozen_114_production_paths`
3. Maintained Mixed tiny full-state/COW/abort/restore acceptance.
4. Only after all preceding gates pass: one maintained Mixed full40, original
   references/tolerances/cap, default Hello/prefill trace. No extra captures.

The frozen test uses `FERRULE_GEMMEX_FIXTURE` pointing to the original separate
114-case `bf16-gemmex-backend-20261006/matrix-fixture`, NOT the host-concatenated
78-case backend fixture. It invokes production resident-linear and admitted
metadata-pair paths and gates all 209,920 original BF16 words across 78 native
calls. `FERRULE_GEMMEX_REPORT` must name a fresh JSONL. References are never
regenerated. The 256x256 synthetic expectation is analytic 256*1, explicitly
not an independent HF oracle or full-model acceptance.

Readiness, logs and source/binary/reference pins are under
`target/validation/bf16-gemmex-integration-20261006/`. The prepared runner checks
prerequisite receipts, source/binary/fixture hashes, authenticated custody and
all-GPU quiescence. It refuses full40 until preceding receipts pass. Source
changes by another agent require rebuild/repin, not ignoring a pin mismatch.
