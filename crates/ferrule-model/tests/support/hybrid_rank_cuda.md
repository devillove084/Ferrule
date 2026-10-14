# Model CUDA hybrid rank owner

This **model API** is used by the separately registered, restricted
[Qwen3.5 CUDA thread TP×EP factory](../../../ferrule-runtime/tests/support/hybrid_cuda_tensor.md).
Model geometry support alone is not runtime capability or quality acceptance;
use the runtime's typed profile validation and documented evidence boundaries.
CUDA module, PP segment and rank preparation share the same exclusive device/image
claim.

## Runtime interface

All types below are re-exported from `ferrule_model::transformer`:

- `CudaHybridRankMemory::for_rank(resources, segment_plan, parallel_plan,
  rank_layout, options, physical_pages)` is host-only admission. F32 numeric
  arithmetic is the default; W8A16 requires explicit numeric options.
- `CudaHybridRankSegment::prepare_rank(resources, segment_plan, parallel_plan,
  Arc<HybridRankLayout>, options, HybridCudaDevice, Box<dyn
  StandardTensorCollective>, Option<HybridCudaRoutedExperts>)` prepares the image
  inside its persistent owner thread.
- `new_lifecycle()` returns `CudaHybridRankLifecycle`, implementing
  `DecoderSequenceLifecycle<CudaHybridRankState>`. State is deliberately not
  Clone. Multiple lifecycle handles share physical credits. Checkout/fork and
  reset replacement allocations are charged independently. Unknown completion
  retains credits; failed release returns the exact state.
- Use `TypedCudaPagedKvPool<CudaHybridRankState>` with
  `owner.layout().kv_planes(options.page_size(), options.max_positions())` and
  the **same physical page capacity** admitted above, including shadow slots.
- Execute the caller's already-entered transaction with
  `execute_cuda_bound(batch, DeviceSegmentInput, working_states, entered_kv,
  sequence_ids, check_active) -> DeviceSegmentOutput`, or `execute` for explicit
  host input/output boundaries. `sequence_ids` is in packed-sequence order.
- `validate_frontier`, `poll_quiescence`, `needs_quarantine`, `is_failed`,
  `retains_expert_custody`, `poll_shutdown` and `shutdown` provide journal hooks.
  This owner never publishes state/KV, allocates transaction IDs or acknowledges
  a runtime commit. False/pending or unknown means retain the journal/custody.

## Executable path

The owner invokes `HybridDecoderSegment::execute_bound_inner` exactly once.
`LayerOperators::Cuda` enters existing root submission scopes and the existing
GQA/GDN/FFN loop. Rank hooks select local projections/heads/state and existing
CUDA kernels; they do not implement another forward or CPU model math.

Numeric preparation uses paired-source verified rectangle reads. Raw E4M3
payload and the minimal paired scale grid retain global row/column origins.
GDN Q/K/V segments remain distinct through numeric execution, then GPU D2D copies
pack them into the local QKV tensor. Local conv/a-log/dt-bias vectors follow the
same segmented head geometry. Rank execution never reopens a checkpoint or
loads full projections. Dense F32/BF16 rectangles are supported too. Embedding
and normalization weights are intentionally replicated.

The owner consumes the model's V2 schedule through the existing
`StandardTensorCollective`. TP1 consumes logical sites without wire calls.
TP2/4 host-stage transport only. Agreement binds logical input/session identity;
owner-local generations remain independently validated against entered KV.
Attention/dense/shared down partials reduce; the ragged vocabulary head gathers.
Only the attached source dispatches external EP, once per routed layer. Peers
contribute exact transport zeros, then receive the reduced GPU result. Expert
storage/dispatch/combine belongs to the supplied existing CUDA executor.

## Admission and integration boundaries

- Weight, paired scales, per-binding numeric workspace, K/V planes and each
  convolution/recurrent root use actual allocator grant classes. Preparation
  checks actual allocation grants against the admitted image.
- Operation rows use a **conservative envelope**, not an allocator reservation.
  Recurrent growth roots, six GQA metadata roots, and source-side external holds
  are separately included. `max_collective_elements` and `pinned_host_bytes`
  expose transport admission inputs. Runtime must still charge its host
  collective storage, host prepared image, allocator segment slack and external
  EP owners (including colocated owners), and retain capacity for output handles
  that it keeps beyond a call. The model budget is not a global device ledger.
- Device/image exclusivity now uses the shared, opaque
  `ferrule_model::decoder::HybridCudaPreparationClaim`. `prepare_rank(...)`
  keeps its signature and performs its own admit/finish. A full runtime factory
  should instead own the token through pool/lifecycle/state construction:

  ```rust,ignore
  let memory = CudaHybridRankMemory::for_rank(
      resources, &segment_plan, &parallel_plan, &layout, options, physical_pages,
  )?;
  let claim = device.admit(memory.admission_estimate()?, memory.max_sequences)?;
  let result = (|| {
      let owner = CudaHybridRankSegment::prepare_rank_with_claim(
          resources, segment_plan, parallel_plan, layout, options,
          device.clone(), collective, experts, &claim,
      )?;
      let mut lifecycle = owner.new_lifecycle()?;
      let state = lifecycle.create()?;
      let pool = TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(
          device.operators().clone(),
          &owner.layout().kv_planes(options.page_size(), options.max_positions())?,
          physical_pages,
      )?;
      Ok((owner, lifecycle, state, pool))
  })();
  claim.finish(result)
  ```

  Never call plain `prepare_rank` inside an existing claim: it correctly rejects
  a second admission. The token cannot be cloned or constructed externally;
  only a real model prepare can mark its image ready. It verifies owner identity,
  one-image use and rank admission ceilings. `finish(Ok(...))` without a prepared
  image is rejected. Final proof failure retains the successful result rather
  than dropping potentially live resources. `Err` refunds only after temporary
  owners/states have been dropped and completion is proven; panic/Unknown never
  clears ownership. A successful claim permanently consumes the device factory,
  just as the original full-module factory did.
- `CudaHybridModule::prepare_with_claim(..., &claim)` is the full-module
  composition seam used by the existing decoder factory, so it does not nest
  claims. Public module/PP/rank convenience constructors acquire their own claim.
  PP's existing constructor has no sequence/page-layout arguments: its established
  caller-owned state/KV/colocation admission remains unchanged; it claims the
  supplied root weight/workspace ceilings rather than inventing missing geometry.
- `HybridCudaRoutedExperts` authenticates no roster by itself. Runtime must verify
  the actual child owners and scope against `HybridParallelPlan` before handing
  off custody. Source rank and attachment presence are checked locally.
- No runtime registration, scripts, HostKV append bridge, support flag or
  distributed journal orchestration was changed. GPU numerical, decode/COW,
  cancellation and multi-device acceptance still belong to the GPU/runtime
  owners. This compile-tested path alone is not a production acceptance claim.

## Validation fixtures

`hybrid_rank_cuda` contains host tests for numeric origins, no-full-read policy,
local dense GDN vectors and physical admission. `gpu::tiny_numeric_hybrid_*`
are ignored fixtures with actual CUDA GQA/GDN/shared-expert projections and
preprepared CUDA routed experts, using the runtime's existing host collective.
They cover TP1/2/4 on independent owner contexts, source-only dispatch, ragged
head output, physical lifecycle credits and rollback. F32 compares TP2/4 with
TP1; W8A16 is explicit. They were compiled with CUDA 12.4 and GPUs hidden, not
executed. No CPU model oracle or fallback is used by these GPU fixtures.

## Claim tests

`tests/unit/cuda_preparation_claim/mod.rs` contains host token-state/admission
tests and ignored CUDA cleanup/foreign-owner/final-proof-loss tests.
`tests/hybrid_rank_cuda/preparation_claim.rs` contains the ignored actual
TP1/2/4 factory test. The latter prepares a rank, creates its pool and default
state inside the token scope, exercises a late clean failure/retry and verifies
that module/PP constructors cannot acquire a second image on the same device.
