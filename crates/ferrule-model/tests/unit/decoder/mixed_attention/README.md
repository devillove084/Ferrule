# Mixed prepared-attention integration

Implemented for the existing Mixed profile only. Other profiles retain their
previous attention dispatch. No new precision profile, CPU fallback, reference,
threshold, cap, or allocator-margin change.

## Ownership and phase authority

- `HybridMemoryPlan::bf16_attention_plan` builds one `from_shapes` plan for the
  image. It warms per-sequence (M,KV) shapes; legal ragged combinations need no
  additional preparation. H/Hkv/D must be uniform across GQA layers because the
  backend fixes those dimensions. Unsupported geometry rejects during admission.
- The existing preparation claim owns one provider allocation/handle, external
  reserve and shared status word. All GQA layers reuse it sequentially.
- `MixedAttentionCapacity` enforces context <=1024 at initial admission, and
  validates the actual packed batch before execution. `CudaHybridModule::validate`
  also calls `prepared_attention_view` through each authenticated typed KV guard
  before any embedding/layer submission. Scheduler labels grant no authority.
- `begin_bf16_attention` runs once before embedding. The image records the exact
  invocation transaction ID; it is not a sequence-history flag. The KV binding
  checks that ID and calls `append_and_attend_prepared`. It cannot expose or
  reconstruct private KV planes, slot tables, epochs or COW write authority.
- The adapter never clears/consumes status. Finish consumes it before output
  projection/logits publication. Abort also consumes active status; a completed
  semantic error permits the existing journal rollback. Unknown remains terminal.
- Provider/status and operand roots remain retained by existing operator/module
  quarantine on Unknown; admission/work custody stays in the existing protocol.
  There is no new coordinator. The adapter adds no per-layer status fence or
  handle retirement; existing synchronous model operator fences remain unchanged.

## Admission (H16/Hkv2/D256, batch8/context128/sequences2, ten GQA layers)

| Resource | Bytes |
|---|---:|
| Attention logical panels + workspace | 35,848,192 |
| Attention dedicated physical grant | 37,748,736 |
| Attention external reserve | 16,777,216 |
| Ten layers of potentially pending metadata + one status word | 13,056 |
| Total incremental attention workspace demand | 54,539,008 |
| Existing projection physical / external | 69,206,016 / 16,777,216 |
| Existing GDN chunk physical | 18,874,368 |
| Updated capacity8 / capacity1 workspace demand | 872,133,452 / 649,638,732 |
| Already admitted workspace cap | 2,155,872,256 |
| Total device admission / cap | 6,846,743,296 / 7,516,192,768 |
| Device headroom / included allocator margin | 669,449,472 / 536,870,912 |

Physical metadata grants are computed from maximum rows, sequence count, page
size, context and GQA layer count (not hardcoded to ten). The status and all
potentially pending layer metadata are charged, even though ordinary fences can
retire them earlier. Panel resources are charged once, not per layer or shape.
Global device-memory deltas are diagnostics, not an attributable provider cap.
The 16 MiB external reserve is not a guarantee of a hard vendor allocation cap.

## Tests and durable evidence

`target/validation/bf16-attention-model-current-20261007/FINAL.json` records source,
binary and reference hashes, exact commands, timings, metrics and release.
`validated/PREPARED.json` and authenticated supervisor receipts describe the
GPU-tested revision. Later changes are limited to CPU budget-test expectations
and this documentation; production source remains the GPU-tested revision.

CPU checks (hide GPUs):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void cargo test -p ferrule-model --lib decoder::mixed_attention::tests
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void cargo test -p ferrule-model --features cuda --test hybrid_bf16
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void CUBLAS_WORKSPACE_CONFIG=:0:0 cargo test -p ferrule-composition --features cuda --test qwen35_memory_plan mixed_attention -- --nocapture
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void cargo clippy -p ferrule-model -p ferrule-composition --features cuda --lib --tests -- -D warnings
```

GPU tests, each authenticated and bounded, only after explicit sole-owner grant:

- `transformer::standard::cuda::bf16_attention::tests::mixed_attention_frozen_and_growing_views_exact`:
  original actual HF raw/gated4096 exact through the retained model provider and
  BF16 sigmoid path, plus 128 changing KV views. No execute allocations/sync;
  one retained owner and clean allocator reclamation. Internal score/probability
  stages were validated by the independent backend gates, not recaptured here.
- `...::mixed_attention_sticky_status_abort_reset`: a real invalid-query device
  latch survives a later valid execute, publication fails, and consumed status
  permits clean reset/reuse.
- `...::mixed_attention_unknown_retains_actual_custody`: actual attention submitted,
  injected lost model consumer proof, reuse/preparation boundary poisoned and
  provider/status/operand storage retained after Drop/allocator trim. This is
  injected evidence loss, not an induced hardware fault or a backend D2H fault.
- Existing `cuda::history_gpu::mixed_m1_history_survives_abort_fork_and_prefill_continuation`
  and Mixed tiny exercise real model cached steps, fork/COW/abort/reset and clean
  shutdown with the new adapter. The full-state tiny oracle remains unchanged.
- Frozen114 projection production paths: 0/209920 unequal words.
- One original full40: all804 metrics pass, no state/route failures.
- Original fresh oracle `102c798b...`: all27 common-history predictions and
  unchanged full-vocabulary bounds pass; independent free-greedy all3, QA
  fork/reset, and separate QA cold-owner replay pass. No HF regeneration.

One initial synthetic test incorrectly assumed uniform attention always yields
exactly the constant V: BF16 probability materialization can change that result.
The test was corrected to include that documented boundary; the original HF
fixtures, all numerical thresholds and production arithmetic were unchanged.
No additional whole-model capture or repeated full40 was needed.
