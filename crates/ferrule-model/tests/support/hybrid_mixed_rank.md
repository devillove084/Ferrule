# Model-only Mixed BF16 TP2 rank slice

This is a deliberately bounded, executable model slice. It is **not** full40
Qwen3.5 decoder admission and it does not enable the production Mixed profile.
Composition, runtime transport ownership, backend ownership, and CLI remain
outside this model change. Existing F32 TP1/2/4, W8A16, and single-owner Mixed
profiles remain unchanged.

## Historical five-gate validated status

The five-gate acceptance below records an earlier source/binary snapshot. See
the final P1 continuation section for current-source validation; this historical
record does not certify the current binaries or full-model admission.

The current acceptance record is the `final2-*` sweep under
`target/validation/qwen35-35b-current/model-mixed-tp2-gates/`. It supersedes the
earlier compile-only handoffs and the provisional four-site data schedule.

All five exact gates passed with the CUDA 12.4 / `sm_86` toolchain on GPUs 0
and 1, with `CUBLAS_WORKSPACE_CONFIG=:0:0`, `FERRULE_RUN_MIXED_RANK_GPU=1`, and
no relaxed numerical tolerance:

1. `hybrid_rank_precision::frozen_tp2_native_bf16_rank_row_partials`
2. `hybrid_rank_precision::frozen_tp2_numeric_gemmex_rank_row_partials`
3. `hybrid_rank_precision::frozen_tp2_numeric_w8a16_rank_row_partials`
4. `hybrid_rank_precision::frozen_tp2_local_experts_shared_late_add`
5. `hybrid_mixed_layer::gpu::frozen_tp2_complete_gqa_dense_bf16_journal_block`

Each gate independently discovered exactly one test and zero benchmarks, then
ran exactly one test with `--exact --ignored --nocapture --test-threads=1`.
The four projection/expert gates each passed one test. The complete-block gate
passed one test on both rank lanes and printed, per lane:

- 5 successful block invocations;
- 3 physical COW append/abort transactions;
- 48 exact frozen BF16 output-word comparisons;
- 2-byte data payload elements;
- 1,248 bytes of native BF16 weight/norm/RoPE payload;
- 71,307,776 warmed live-grant bytes and 71,311,104 peak live-grant bytes;
- 138,412,032 reserved and peak-reserved bytes;
- zero active transactions at drain.

The complete-block gate uses the production `MixedRankCollective` adapter,
not a test-only collective wrapper. Every `final2-*` record reports a reaped
child, `echild: true`, unchanged source pins, and a fully idle post-snapshot:
all GPUs 0--7 reported 0 MiB, 0% utilization, and no compute applications.
The temporary runner is
`target/validation/qwen35-35b-current/model-mixed-tp2-gates/run.py`; its
machine-readable summary is `final2-summary.json` in the same directory.

The saved binary pins are:

- precision gate binary:
  `e31edaa1b819671970ab5fe8f2e655f5238af7e0c822fa703ba37c004c6e90c3`;
- complete-block binary:
  `59338313e784540f928bff8b74e61f12187743cfd91cbdfe8edb8816377fbd2b`;
- `hybrid_rank_mixed_oracle.json`:
  `5a671d220bd4cecfee92a1614f637c0d59bb5feeff543240fd377bdb81a051af`;
- `hybrid_mixed_layer_oracle.json`:
  `cde7d207d2e7c915f1eb33484e6ef98406efbea51ce889ce66d01a36c8617af2`.

## Implemented model contracts

- `HybridRankPrecision::MixedBf16F32` selects BF16 activation/KV/conv storage,
  F32 recurrent state, and typed `Bf16PartialsAndReduction` boundaries. The
  legacy F32 plan constructor and profiles are unchanged.
- `HybridMixedBlockPlan` admits only complete resident BF16/numeric images. It
  rejects dense F32 resident weights, non-BF16 scales, incomplete physical
  grants, missing routed gate/up/down shards, and unsupported FP8 scale
  geometry. Every routed expert in the selected layer is resident on each TP
  rank as an EP=1 local tensor shard. There is no CPU fallback, offload cache,
  or resident F32 mirror.
- `prepare_mixed_rank_block` validates generation-bound handles, paired FP8
  source identity, physical payload grants, and publication ordering.
- Local linear execution uses resident BF16 GemmEx, numeric FP8 GemmEx, or the
  actual BF16-input W8A16 path. BF16 output is written back before a row
  reduction; unreduced row partials are not promoted to an F32 transport
  mirror. Unsupported geometry fails closed.
- `mixed_rank_swiglu_local` supports concatenated gate/up execution when the
  complete-width geometry is admitted, otherwise explicit local projections.
  Down projections produce local BF16 partials.
- `mixed_rank_moe_local_add` implements the late-add shape: local shared and
  routed outputs are combined first, then one typed BF16 TP reduction is
  required. The early/separately-reduced branch is rejected rather than
  silently changing operation order. Routed weighting is explicitly the
  HF-eager BF16 product/add contract; no fused-vLLM numerical-equivalence
  claim is made.
- Local FP8 execution may use a packed zero-origin view only when original
  origins are complete 128x128 scale-block multiples. Bytes and original
  binding provenance remain unchanged. Misaligned native GemmEx fails closed.
- `CudaMixedRankLinear` carries image generation and slot identity. Stale or
  foreign tokens are rejected, and resident images participate in model memory
  accounting and quarantine rules.

## Complete stateful GQA+dense block

The model-owned entry points are on the existing
`CudaStandardDecoderOperators`:

- `prepare_mixed_rank_gqa_block` prepares one TP2 GQA+dense block and creates
  the existing typed `CudaHybridRankLifecycle` with BF16-compatible state;
- `execute_mixed_rank_gqa_block` invokes the existing
  `execute_standard_layer_inner` exactly once, rather than creating a second
  forward loop.

The admitted range is one GQA+dense block, TP2, M1..32, KV<=1024, with warmed
attention views. The existing full-rank `CudaHybridRankSegment` Mixed
admission remains closed.

Dispatch points and state behavior:

- `standard/cuda.rs::linear` selects the mixed block path for rank-local Q/K/V/O.
  O is BF16 before one `AttentionOutput` BF16 all-reduce and the attention
  residual.
- `standard/cuda.rs::dense_swiglu` selects local gate/up/down. Down is BF16
  before one `DenseMlpDown` BF16 all-reduce and the MLP residual.
- `attention_heads` uses the generic `HybridRankLayout::attention(global_layer)`.
  Compact KV-layer mapping, whole-head partitioning, GQA K/V replication,
  interleaved query gates, RoPE, native BF16 norms, and residuals use existing
  model operators; no Qwen-specific layer-index hack is introduced.
- `CudaStandardKvBinding::with_prepared_attention` binds prepared BF16
  attention and the existing transactional journal adapter. View misses fail
  closed rather than selecting eager or F32 attention.
- `CudaHybridRankLifecycle::with_precision` preserves BF16 KV, F32 SSM state,
  fork/checkout credits, append validation, rollback, and unknown-completion
  quarantine. No model commit or sequence-frontier advance is hidden inside
  the block call; the caller owns those lifecycle operations.

## Typed control/data boundary

`HybridBf16Collective` has separate control hooks:

- `validate_execution(transaction, batch, sessions)` rechecks the sealed
  execution identity;
- `complete_owner(transaction)` records control completion only after data
  consumers are fenced;
- `exchange_bf16(transaction, site, kind, values)` carries BF16 data only.

The complete block has exactly two BF16 data sites:

1. `AttentionOutput`: BF16 sum;
2. `DenseMlpDown`: BF16 sum.

Agreement and Completion are **not** BF16 arrays. Transaction IDs, session
identity, shape/control state, sealing, and completion stay in the runtime's
typed control protocol. The production runtime adapter validates the sealed
kind/site and calls its control checks; the model does not encode control
identities as floating-point values or invent a model coordinator.

The independent Torch fixtures establish that BF16-per-rank projection
writeback and deferred F32 partial accumulation differ, and that local
shared+routed late-add differs from separately reduced branches. The selected
policy is the explicit BF16 producer-output/BF16 reduction contract exercised
by the frozen gates, not an assertion derived from metadata or a blanket claim
about every vLLM quantization kernel.

## CPU, compile, and GPU evidence

With devices hidden after sourcing the supplied environment, the following
checks passed:

```sh
source target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES=''
cargo clippy -p ferrule-model --features cuda \
  --test hybrid_rank_precision --test hybrid_mixed_layer --offline -- -D warnings
cargo test -p ferrule-model --features cuda \
  --test hybrid_rank_precision --test hybrid_mixed_layer --no-run --offline
```

The CUDA-feature CPU suite passed 13 `hybrid_rank_precision` tests and 3
`hybrid_mixed_layer` tests. Additional model checks passed the existing numeric
shard, TP/EP, V2 channel, lifecycle-credit, and CUDA rank CPU tests. The
complete-block binary compiled with CUDA 12.4 and `sm_86` before the final
GPU sweep. `git diff --check` passed for the touched model slice.

The complete-block GPU gate uses real journal/backend/lifecycle state: a
prefill rollback, a committed source, a logical fork, then three physical
page-1-to-page-2 COW decode append/aborts. It compares each result with the
independent frozen whole-sequence oracle, verifies source/fork frontier and
source physical-slot stability, restores free slots, holds resident weights
constant, observes stable warmed allocator grants, records zero artifact
uploads, releases all pages/states, and quiesces each owner. The allocator
figures are exact for Ferrule allocator-owned storage; sampled driver memory
also includes CUDA context/cuBLAS private memory and is not an exact peak
ledger.

## Explicit remaining seams; no full40 claim

The following are intentionally unsupported and must remain fail-closed:

1. GDN vectors/norm/state/chunk preparation and execution, including BF16
   convolution with F32 SSM lifecycle integration.
2. Routed MoE dispatch inside the complete layer loop. The local routed-expert
   primitive is real and tested, but it is not silently connected to the
   complete GQA+dense path. Full-model MoE requires the shared+routed late-add
   hook and a pinned fused-vLLM accumulation oracle.
3. Multi-layer ranges, embedding/head ownership, biased/clamped projections,
   cold/unadmitted attention views, and production model boot/admission.
4. Unified whole-model operation, attention, GemmEx/provider, state/KV,
   collective, and peak-grant accounting. The block grant checks are not a
   full40 memory grant.
5. Caller-side production layer-loop dispatch, commit/frontier lifecycle, and
   final endpoint wiring owned by runtime/composition agents.

Consequently this change proves a real model-executing TP2 BF16 GQA+dense
block plus local tensor-sharded expert primitives. It does **not** enable or
claim whole-model Qwen3.5 40-layer Mixed TP2/EP1 readiness, vLLM fused-MoE
parity, or cached-decode parity beyond the tested journal COW block.

# Follow-up: bounded GQA+GDN recurrent island

The model-only extension adds `HybridMixedGqaBlockPlan::new_gqa_gdn_pair`.
It admits exactly two layers in the existing rank layout: a GQA+dense layer
followed by a GDN+dense layer. It does not admit arbitrary ranges or attach to
the full model loop. The existing `execute_standard_layer_inner` is invoked
for both layers in one owner transaction; the caller still owns KV commit and
sequence-frontier publication.

The GDN rank contract is explicit:

- query/key/value heads are partitioned by the existing generic rank layout;
  each rank retains whole local key groups and their value heads;
- convolution history is BF16 and F32 recurrent SSM state is rank-local;
  there is no replicated F32 mirror, CPU fallback, or recurrent collective;
- local GDN QKV/Z/A/B/output and dense MLP projections use the native rank
  image; the output projection uses the existing typed BF16 reduction site;
- vector parameters are admitted as native local BF16 rank payloads, and the
  rank plan accounts for local state, chunk workspace, and state-copy grants;
- the existing BF16 chunk kernel is selected for fresh multi-row prefill,
  including the explicit one-row fresh-prefill plan. A single-row cached
  continuation uses the existing direct recurrent kernel. Selection is based
  on validated state frontier/history, never allocation presence or scheduler
  labels;
- checkout, logical fork, COW rollback, reset, and release use the existing
  typed CUDA hybrid lifecycle. State identity remains bound to the rank image.

CPU admission tests cover local head geometry, 30 BF16 convolution-history
words and 12 F32 SSM words per rank in the synthetic TP2 fixture, nonzero
state/workspace budgets, chunk fresh-vs-cached selection, wrong-order ranges,
MoE rejection, and capacity failure. The independent Torch oracle is
`tests/support/hybrid_mixed_gdn_oracle.py` with frozen output and per-rank
state words in `hybrid_mixed_gdn_oracle.json`; it pins Transformers 5.5.3
source SHA256
`aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048`.

## GDN GPU gate result: intentionally not a parity pass

The prepared exact gate is:

`hybrid_mixed_layer::gpu::frozen_tp2_complete_gqa_gdn_bf16_journal_pair`

The final run is recorded under
`target/validation/qwen35-35b-current/model-mixed-tp2-gates/gdn-final-gpu_frozen_tp2_complete_gqa_gdn_bf16_journal_pair/`
and summarized in `gdn-summary.json`. It discovered exactly one test and
executed it with the production runtime adapter on both TP lanes. The gate
**failed closed** because exact F32 SSM state parity was not achieved; no
numerical tolerance was relaxed and no profile was enabled.

Observed exact results before the deliberate final assertion:

- 112/112 BF16 output words matched the independent Torch oracle;
- 360/360 BF16 convolution-history words matched;
- 128/144 F32 SSM words matched; 16 differed by at most 2 ULP, with maximum
  absolute difference `4.656612873077393e-10`;
- fresh one-row state, cached execution output, reset output, fork state copy,
  source-state isolation, and all three COW aborts completed;
- each rank performed six invocations, three COW append/aborts, and one reset;
- all pages were reclaimed, state credits were released, and both owners
  quiesced. The gate child was proven drained with `ECHILD`.

Per-rank GDN image metrics in that run were 1,992 resident weight/vector
bytes, 71,574,528 warmed live-grant bytes, 71,578,880 peak live-grant bytes,
and 138,412,032 reserved/peak-reserved bytes. These allocator values exclude
CUDA context and cuBLAS-private allocations.

The narrow remaining blocker is the F32 SSM state bit contract for fresh
multi-row chunk prefill and one rank's cached continuation. The existing CUDA
operator arithmetic and the pinned CPU Torch reference differ by 1--2 ULP in
those state words while the BF16 outputs remain exact. This is not being
silently redefined as success; resolving it requires an explicitly chosen
cross-device recurrent arithmetic/reference contract (or a backend-owned math
change), outside this model-only ownership boundary. The GQA+GDN pair therefore
remains an explicitly invoked experimental API, while full-model production
admission stays fail-closed. The constructor proves geometry, not parity.

The original five GPU gates were rerun against the rebuilt binaries after this
extension and all five still passed. Final ownership record:
`target/validation/qwen35-35b-current/model-mixed-tp2-gates/gdn-release.json`.
All eight GPUs were idle with no compute applications; model GPU ownership was
released.

## Current numerical-debug update

The preceding CPU-reference gate result is superseded for numerical diagnosis
by the profile-matched replay under
`target/validation/qwen35-35b-current/model-mixed-tp2-gates/gdn-replay-profile-cuda-custody.json`.
The independent CPU fixture remains immutable. A separate
`hybrid_mixed_gdn_cuda_oracle.json` stores only the F32 SSM words obtained by
running the same captured rank-local BF16 operands through the established
pinned HF CUDA policy. Its CPU-produced BF16 outputs and convolution history
are required to equal the original fixture.

Same-input evidence is exact:

- rank-local QKV/A/B/Z operands, A_log, dt_bias, norm, incoming BF16 history,
  and incoming F32 state all match the independent fixture bit-for-bit;
- all captured BF16 convolution results match;
- pinned HF CUDA replay matches Ferrule's GDN output and F32 SSM state bit-for-
  bit for all six rank/case combinations;
- the first CPU/CUDA reference differences are in F32 `exp(A_log)`, softplus,
  and decay. BF16 beta and normalized Q/K match;
- chunk intermediates (`q`, `k`, `k_beta`, `v_beta`, gate prefix, values,
  attention, and state-intermediate inputs) match exactly. The comparison is
  retained in `gdn-chunk-intermediate-comparison.json`.

Therefore the 16 prior 1--2 ULP differences were a CPU-versus-CUDA Torch
reference-device mismatch, not TP state ownership, frontier, BF16 conversion,
collective arithmetic, or an SSM collective. No per-index correction and no
backend arithmetic change was made. The exact state contract is now explicitly
F32 state against the pinned HF CUDA reference while BF16 outputs/history
remain against the independent CPU fixture.

The new immutable profile fixture is validated by
`frozen_gdn_cuda_state_oracle_is_profile_matched_and_output_stable`. The
four projection/expert GPU gates still pass on the rebuilt model. The two
production-adapter block gates—both the existing GQA+dense block and the new
GQA+GDN pair—are currently blocked before preparation by the runtime-owned
control seam:

`Mixed collective does not expose an authenticated schedule binding`

`HybridBf16Collective::sealed_binding()` now fails closed by default, but the
current runtime `MixedRankCollective` implementation does not yet implement
that hook. This model-only task intentionally did not edit runtime-owned code
or fabricate a binding in the test. Once the runtime owner supplies the sealed
binding implementation, rerun both block gates; no further numerical model
change is indicated by the same-input evidence. Full-model admission remains
closed.

## P1 model-only continuation: preparation and ownership seams

The current model branch adds the following without changing runtime, composition,
CLI, or backend ownership:

- `TypedSourceVectorPolicy` restricts Mixed F32-source conversion to
  `LinearAttentionALog` and `LinearAttentionNorm`. The source binding remains
  F32; the only resident representation is one-time BF16 RNE words. Native
  projection/non-GDN F32 sources still fail closed. The CPU tests include exact
  tie-to-even words and assert source identity/dtype preservation.
- Replicated F32 GDN norms keep the existing full-vector materialization path
  and use typed preparation conversion; they are not rank-matrix shards. The GPU fixture preserves F32 A_log/norm source
  tensors while projection weights remain BF16.
- Mixed rank image publication is atomic after projection/vector/workspace
  preparation and owner fences. Known-complete preparation errors clear only the
  attempt's resources and permit bounded retry; typed Unknown completion keeps
  the image unpublished, quarantines the owner, and retains custody.
- W8A16 Mixed numeric submission checks workspace poison immediately after the
  backend call. Poison latches owner quarantine and preserves resident/operand
  leases; ordinary errors retain the existing reusable path.
- GPU-ready fault gates cover failure injection after projection, vector, chunk,
  attention, and lifecycle preparation, plus clean/Unknown numeric completion.
  They are compile-only/ignored until a separate GPU grant.

No GPU gate was run in this continuation. The currently ready ignored IDs are:
`mixed_prepare_known_failures_rollback_and_retry`,
`mixed_prepare_unknown_never_releases_or_retries`,
`mixed_w8_completion_bridge_quarantines_and_retains`, the two complete block
adapter gates, and the four existing projection/expert gates. Full40 admission,
shared+routed production layer-loop dispatch, embedding/output integration, and
runtime control lifecycle remain fail-closed and are not claimed here.

Current CPU validation: `hybrid_mixed_layer` 9 passed / 2 GPU ignored;
`hybrid_rank_precision` 13 passed / 4 GPU ignored; three focused library CPU
checks passed (source rounding, preparation fixture, typed Unknown). All nine
ignored GPU gates were discovered (3 ownership + 2 blocks + 4 projection/expert),
not executed. CUDA 12.4/sm_86 compile/link and targeted Clippy `-D warnings` passed
with `CUDA_VISIBLE_DEVICES=''`; no GPU query or context was requested.

The runtime owner has now implemented authenticated `sealed_binding()`.
`runtime_adapter_exposes_authenticated_binding_before_model_preparation` verifies
exact binding equality and model admission without a GPU. This supersedes the
missing-hook blocker in the preceding historical section; complete block GPU
reruns remain pending a new grant. No runtime source was changed here.

The W8 fault gate injects the backend-reported poison status into the exact model
completion bridge with real resident allocations and operand leases. It does not
induce a native backend workspace failure; that provider-level failure mechanism
is backend-owned. Unknown tests intentionally retain allocations until process
exit and must run in isolated, reaped gate processes. Known-failure retry uses a
fresh preparation transport; it does not revive an aborted sealed runtime cohort.

## Generic rank-program slice (2026-10-09)

The model-only Mixed execution slice now lowers a typed bound rank program rather
than a production GQA/pair combination. `HybridMixedRankProgramPlan::from_layout`
iterates the layout's declared layer range and composes GQA or GDN attention with a
dense SwiGLU leaf. `HybridMixedGqaBlockPlan` remains only as a compatibility type
alias; its constructors validate and wrap the generic program. The authoritative
`execute_standard_layer_inner` loop is used once per bound layer in
`standard/cuda/hybrid_rank/mixed_layer.rs`.

Validated current-source CUDA 12.4/sm86 gates under authenticated custody:

- 10/10 passed: the nine preparation/precision/block gates plus
  `gpu::frozen_tp2_generic_gqa_gdn_gqa_bf16_program`.
- Generic GPU range is three layers `GQA -> GDN -> GQA`, with two compact physical
  KV layers and one local GDN state slot per rank. Both ranks match the immutable
  independent reference exactly: 112 output BF16 words, 360 BF16 convolution/
  history words, and 144 F32 SSM words; cached decode, COW abort, and reset pass.
- Pair gate remains exact at the same 112/360/144 counts. Existing native BF16,
  GemmEx, W8A16, and local shared/routed arithmetic gates remain passing.
- Preparation known-failure retry, Unknown retention, and W8 completion poison
  all pass. The W8 test injects the model completion bridge; it is not native
  backend workspace fault injection.
- Generic fixture measured per-rank peak live grant 71,582,720 bytes and peak
  allocator reserve 138,412,032 bytes. These are fixture observations, not a
  full-model budget or admission permission. Post-run all eight GPUs were idle,
  with no compute applications and zero reported memory/utilization.

CPU tests cover arbitrary dense GQA/GDN ordering, multiple recurrent state slots,
compact-KV indexing, shared maximum recurrent workspace, and fail-closed routed
semantics. The source-vector contract remains original F32 binding with one typed
BF16 resident conversion for Mixed GDN vectors; no F32 mirror was added.

### Explicit remaining seams

- Routed FFN preparation and dispatch are still fail-closed. The local resident
  expert arithmetic exists and is tested in isolation, but production layer
  execution does not yet connect replicated BF16 router logits/full-softmax F32
  top-k route identity, local shared+routed HF-eager addition, and one late BF16
  reduction. Do not enable MoE from metadata.
- The runtime mapping for a new combined `FeedForwardOutput` data site is not
  changed here. Shared/routed separate reductions remain forbidden.
- Embedding/output-head execution, full aggregate operation/provider/KV/state
  budget admission, production head-128 independent fixture, full40 quality gate,
  and whole-model admission remain closed. No full-model TP readiness is claimed.

### Evidence and next-owner contract

Evidence root: `target/validation/qwen35-35b-current/model-mixed-tp2-gates/`.
`generic-final-summary.json` aggregates ten nonzero exact-discovery runs. Each
`generic-final-*` directory records discovery, binary/source/reference hashes,
authenticated subreaper custody (`proven_drained`, `echild=true`), GPU timeline,
and all-eight-device pre/post-idle snapshots. Startup was devices 0/1,
`CUBLAS_WORKSPACE_CONFIG=:0:0`, CUDA 12.4, `sm_86`. Ownership is released.

Two stale test assumptions were corrected, not production arithmetic:

1. Unknown fault stages had reused one stream owner even though the first case
   intentionally retained that owner's reserved provider pool. Each independent
   Unknown case now owns a fresh device/operator, with failed quiesce/retry/Drop
   retention asserted unchanged.
2. GPU collective fixtures hardcoded protocol version 1. They now use
   `HYBRID_AGREEMENT_VERSION` (2), as the sealed runtime binding requires.

Dispatch/API paths (relative to `crates/ferrule-model/src/transformer`):

- `hybrid_parallel/mixed_layer.rs`: `HybridMixedRankProgramPlan::from_layout`,
  `layers`, `recurrent_shapes`, and the typed data-only schedule.
- `standard/cuda/hybrid_rank/mixed_layer.rs`:
  `prepare_mixed_rank_program` atomically prepares all declared layers;
  `execute_mixed_rank_program` uses each global layer's local KV/state mapping
  and calls the existing standard executor. No model-specific index/count branch.
- `tests/support/hybrid_mixed_program_oracle.py` (relative to model root): new
  independent Torch layer-3 suffix over SHA-pinned immutable HF/Torch prefix
  outputs. Original prefix and CUDA state oracle files were not rewritten.

For the runtime owner: no new model collective semantic or runtime API was
introduced in this slice. The next MoE change needs an authenticated typed
route-ID/weight-bit agreement keyed by transaction/global layer/ordered rows;
this must not reuse BF16 data packets. The FFN boundary then needs one combined
output site after local gated shared plus routed accumulation. Do not relabel a
shared-only reduction as a combined result. Existing control binding, owner
validation, and completion remain unchanged.

Final hidden-device CPU validation: 11 layer tests and 13 rank-precision tests
passed, plus all three focused source/preparation tests. Full library CPU run:
589 passed, 103 ignored. CUDA compile/link, targeted Clippy `-D warnings`, owned
Rustfmt checks, and `git diff --check` passed. Four-layer and GDN-first layouts
are CPU-tested; the new numerical GPU gate is three layers with Dk=3/Dv=2,
not head-128 certification. Recurrent-only ranges remain rejected upstream by
the existing rank state schema. Different GQA provider geometries in one program
are explicitly rejected. No runtime/composition/CLI/backend file was edited.


## Routed Mixed TP2 model slice (2026-10-09)

The generic `HybridMixedRankProgramPlan` now admits routed FFN layer specs
without adding a pair constructor or model/layer factory. For each routed layer
it lowers the replicated router, shared expert and every local tensor-sharded
expert gate/up/down projection. `StateDictMaterializer::expert_parameter`
uses the CUDA rank path for expert bindings, so all expert shards and their
original FP8 payload/scale provenance are resident before execution; routed
execution does not enter a provider, checkpoint reader, CPU cache, or offload
path.

The model contract is explicit:

- Router operands are BF16 resident; router logits are produced at the BF16
  model boundary and normalized by full F32 softmax plus selected-probability
  renormalization. Raw-logit descending selection uses lower expert ID only for
  exactly equal raw logits.
- `HybridMixedRouteAgreement` carries ordered expert IDs and exact F32 weight
  bits as typed control metadata. It is never serialized into BF16 data words.
- Each rank computes local gate/up/down expert math and the shared branch with
  HF-eager BF16 expert order. Shared and routed contributions are added locally.
- One `RoutedExpertOutput` BF16 data site reduces the combined FFN output.
  Separate shared/routed reductions are rejected.
- Route agreement must occur after sealed execution validation and before local
  expert execution. Duplicate/malformed/reordered route control fails closed.

### Runtime handoff

The model trait exposes `sealed_route_sites()` and
`agree_routes(transaction, agreement)`. The production runtime adapter now
publishes the sealed route sites and compares exact expert IDs/F32 weight bits
under its typed control protocol. The model GPU fixture consumes that actual
`MixedRankCollective` adapter; no private route transport remains, and no route
identity or transaction/control field is encoded as BF16 data. Control
completion and unknown-custody handling remain separate from BF16 data-site
completion.

No runtime, composition, CLI, or backend files were edited by this model slice.
There is no vLLM arithmetic-equivalence claim; this is the Ferrule HF-eager
model-artifact contract.

### Final GPU evidence

Final current-source run: **12/12 exact gates passed** under CUDA 12.4,
`sm_86`, devices 0/1, `CUBLAS_WORKSPACE_CONFIG=:0:0`, authenticated subreaper
custody, and all-eight-device pre/post idle checks. The suite includes the
runtime-backed endpoint gate
`gpu::frozen_tp2_tokens_moe_logits_bf16_program`, which covers token embedding,
the generic layer segment, ragged output-head BF16 gather, cached decode, and
MoE. The MoE gate reports across both ranks:

- 12 rank-local route agreement calls and 12 combined late BF16 reduction
  submissions (six two-rank agreements and six two-rank MoE allreduces);
- 112 route IDs, 112 exact F32 route-weight words, 112 local BF16 partial
  words, and 112 final BF16 output words;
- 16 resident experts per rank, top-k 8;
- unchanged GDN coverage: 360 BF16 history words and 144 F32 SSM words,
  including prefill, cached decode, COW/fork abort, and reset.

MoE fixture per-rank peak live grant was 71,618,560 bytes and peak allocator
reserve 138,412,032 bytes; resident weight bytes were 7,864. These are bounded
fixture measurements, not full-model budget admission. The final post snapshot
shows zero memory/utilization and no compute applications on all eight GPUs.

Independent route/output reference is
`tests/support/hybrid_mixed_moe_cuda_oracle.json`, generated by an authenticated
Torch CUDA process. CPU Torch differed by one F32 ULP in selected renormalization
on two cases; raw logits, IDs, BF16 partials, and output were unchanged. The
CUDA reference is used because the declared contract is CUDA F32 reduction order,
not a relaxed tolerance or production-derived expected value.

### Model-only estimate and endpoint evidence

The model now exposes `HybridMixedRankOperationEstimate` before payload
materialization and composition can consume it through its existing callback.
The real 35B header-only probe, with GPUs hidden, reports per rank:

- storage ledger: `18,573,937,152` bytes;
- GemmEx storage/provider: `543,162,368` / `16,777,216` bytes;
- attention storage/provider: `44,040,192` / `16,777,216` bytes;
- recurrent workspace: `10,485,760` bytes;
- operation peak: `1,138,176` bytes;
- boundary device grant: `6,291,456` bytes;
- total model estimate: `19,212,773,632` bytes;
- with an explicit 128 MiB owner margin: `19,346,991,360` bytes.

The probe also verified the known storage totals, 30 recurrent layers, 40 route
sites, 81 data sites including the output-head gather, and one-byte-underbudget
rejection. This is a pure header/geometry estimate, not a measured full40
allocation or quality/canary result. FP8 scale dtype and complete 128x128
origin geometry are rejected before GemmEx planning; no scale relabeling or
F32 mirror is introduced.

Embedding/output-head execution is now exercised by the small runtime-backed
boundary fixture with exact independent Torch BF16 words and ragged vocabulary
ordering. A CPU transport test covers zero-padded BF16 column packing and
unpacking. Production head-128 parity, full40 admission, and full-model
quality/canary gates remain fail-closed; whole-model readiness is not claimed.

The complete MoE fixture uses native BF16 projection weights. Encoded FP8
GemmEx/W8A16 projection gates also passed, but this does not certify a complete
FP8 MoE block or production head-128 geometry. FP8 stays compressed with scales
in resident GPU storage; the existing GemmEx backend still performs its bounded
**device-side** operand staging. No claim of zero device decode/staging is made.
The MoE fixture removes checkpoint payload files after both ranks finish
preparation, then executes every transaction with zero artifact uploads, proving
there is no execution-time checkpoint/offload fallback in the tested path.

The final suite was rerun after the estimator, endpoint, transport-test, and
metric changes. Its authenticated post-snapshot had zero memory/utilization and
no compute applications on all eight GPUs; GPU ownership has been released.
Do not reuse the grant implicitly.

## Full40 Hello layer-0 diagnostic (2026-10-09)

A single authenticated TP2 GPU diagnostic loaded the full resident image once,
ran the same Hello prefill input (`[9419]`), and stopped from the existing
observer immediately after layer 0. It rolled back the journal transaction,
released the initial state, quiesced both owners, and left zero active
transactions/live state. The all-eight-device post-custody snapshot was idle;
this was not a full40 acceptance run.

The capture compared only boundaries present in the immutable independent
archive. The archive contains 2,178 tensors/case; the bounded layer-0 capture
compared 24 distinct mappings and explicitly reported 2,154 uncovered tensors.
It did not synthesize expected QKV/GDN intermediates from Ferrule output.

Earliest compared boundary:

- replicated token embedding and layer input: exact on both ranks;
- layer-0 GDN BF16 convolution state: first difference, rank 0 `5/16,384`
  words and rank 1 `3/16,384` words;
- subsequent F32 recurrent state: rank 0 `384/262,144` unequal words,
  rank 1 `128/262,144`;
- attention partial/reduced outputs, route IDs, shared/routed partials, late
  MoE reduction, and layer-0 hidden output remained within the unchanged
  reference tolerance;
- selected route IDs were exact. Route-weight differences were only small
  F32 rounding differences.

The exact diagnostic receipt is under the ignored validation output
`target/validation/tp2-mixed-claim-gates/layer0-boundary-02/`, with
`events.jsonl`/`*.f32` captures per rank and
`boundary-comparison.json`. The parser is
`tests/support/hybrid_mixed_production_compare.py` and is CPU-only.

This does not prove whether the first GDN difference originates in the local
QKV GEMM output, the BF16 conversion staging, convolution arithmetic, or the
independent reference device policy: the independent archive has no expected
same-input QKV/convolution intermediate. Therefore no arithmetic change was
made and no tolerance was widened. The next minimal operator gate is a pinned
same-input GDN QKV/convolution fixture with expected pre-convolution QKV and
post-convolution state, before any full40 rerun.

## Claim-aware production seam (CPU-validated, GPU pending)

`CudaStandardDecoderOperators::prepare_mixed_rank_program_with_claim` is now
available for the runtime factory. It accepts the existing
`HybridCudaPreparationClaim` rather than creating a second reservation:

```text
prepare_mixed_rank_program_with_claim(
    resources,
    program,
    device,
    max_parameter_bytes,
    collective,
    attention_shapes,
    claim,
) -> Result<CudaHybridRankLifecycle>
```

The runtime must obtain the same claim inputs from
`HybridMixedRankProgramPlan::cuda_claim_inputs(resources, physical_pages,
device_ordinal, max_parameter_bytes, attention_shapes)`, then call
`HybridCudaDevice::admit_with_context(estimate, context)`. The context seals:

- the complete program/layout/segment and collective binding, including
  max-rows and max-sequences;
- physical KV pages, device ordinal, parameter read limit, source-bound
  parameter identities, attention shapes, and provider limits;
- the storage-ledger state/KV/weight grants and the complete compiled workspace
  pool, including GemmEx, attention, recurrent, state-copy, operation, and
  boundary grants.

Preparation validates exact context and estimate equality before allocation.
`begin_image` occurs only after model validation/source binding and before the
first upload. `image_prepared` occurs only after all projection, vector, chunk,
attention, and lifecycle construction has completed and the outer completion
fence has succeeded. The model then publishes the image atomically. The caller
still owns pool/default-state construction and must call `claim.finish(result)`;
that final fence is the only successful claim commit.

Ordinary preparation errors clear reusable temporary image state and allow the
claim's normal proven rollback. Unknown completion retains/quarantines the
claim-owned CUDA resources. No claim phase is marked prepared on an ordinary
error or before completion proof. Existing F32 claim APIs and profiles are
unchanged.

CPU tests cover wrong device/page/state/weight/workspace budgets, mismatched
source bindings, changed attention geometry, changed provider context, exact
estimate mismatch, and missing parameter read budget. Existing ignored CUDA
fault tests now also exercise the claim path across projection/vector/chunk/
attention/lifecycle stages; they were compiled but not run in this pass. The
production GPU claim gate remains pending a separate explicit grant.

## Same-input layer-0 HF CUDA replay: divergence is before GDN

The additive replay `hybrid_mixed_gdn_full40_replay.py` consumes the retained
`layer0-boundary-02/rank-{0,1}/events.jsonl` and binary F32-expanded observations.
It never treats Ferrule output as an expected answer: captured inputs are sent
through the unchanged pinned HF chunk function and RMSNormGated, and checkpoint
projections are independently evaluated with Torch `F.linear`. Six CPU parser
tests reject malformed BF16 expansions, duplicate operands, nonzero initial
state, escaped paths, and later-layer captures.

The authenticated HF continuation is saved at:

```text
target/validation/tp2-mixed-claim-gates/layer0-boundary-02/hf-replay/
```

- `numerical/report.json`: exact comparisons, all unequal indices, source and
  capture hashes, local/global head ranges, and original source dtypes.
- `numerical/hf-intermediates.safetensors`: independent projection outputs,
  conv pre-SiLU/output/history, Q/K normalization, decay/beta, chunk locals,
  F32 state, and gated output. SHA256:
  `41326c094e9e622ff25d93d38d2f666922968d16f7d12b0d019a531d982f7c7a`.
- `qkv-f64-diagnosis.json` and `qkv-f64-all-outputs-diagnosis.json`: CPU-only
  high-precision rounding diagnostics, explicitly **not acceptance references**.
- `result.json`, `supervisor.json`, `before.json`, `after.json`, `release.json`:
  authenticated custody, reaped descendants (`echild=true`), all eight GPUs
  idle before/after, and owner-lock release.

Only one new HF CUDA child was run, reusing the already-drained CUDA12.4 native
capture (binary SHA256
`35db801704b18b2543b6dbb9da4890a12f96430454d7b664db87cfcbd36c8960`).
All recorded Rust/native source pins still matched at diagnosis. Ferrule was
not rerun; no layer 1+ or full40 forward was executed in this continuation.
The base Python3.12 `-S` interpreter supervised custody; the numerical child
used `/opt/conda310/bin/python`, where the pinned Torch `2.13.0+cu129` and
Transformers `5.5.3` actually reside. Source hashes and TF32/reduced-BF16-off,
deterministic settings are checked by the unchanged reference initializer.
HF retained its archived `CUBLAS_WORKSPACE_CONFIG=:4096:8` policy; the original
Ferrule run used `:0:0` plus its explicitly prepared workspace. No claim that
these library/version/dispatch policies are identical is made.

Exact comparisons, **per rank**:

| Boundary | Rank 0 unequal/words | Rank 1 unequal/words |
| --- | ---: | ---: |
| Independent same-normalized-input QKV projection | 5/4,096 | 3/4,096 |
| A, B projections | 0/16 each | 0/16 each |
| Z projection | 0/2,048 | 0/2,048 |
| Same-input BF16 convolution history | 0/16,384 | 0/16,384 |
| Same-input BF16 convolution output | 0/4,096 | 0/4,096 |
| Same-input BF16 chunk output | 0/2,048 | 0/2,048 |
| Same-input F32 SSM state | 0/262,144 | 0/262,144 |
| Same-input BF16 gated output | 0/2,048 | 0/2,048 |

The source conv weights (16,384 words), A_log/dt_bias (16 each), and norm (128)
match their independently selected resident BF16 values exactly on both ranks.
A_log/norm retain F32 checkpoint provenance and undergo the established single
BF16 conversion. Rank 0 owns key heads 0..8/value heads 0..16; rank 1 owns key
heads 8..16/value heads 16..32, each dimension 128. Both initial histories and
F32 SSM states are exactly zero. Fresh HF uses padded Conv1d and chunk initial
state `None`; the separately checked update convolution with captured zero
history is bit-exact to that fresh branch for this one-token case. This does
not generalize the fresh/cached equivalence to nonzero history.

All eight unequal QKV words, as F32-expanded BF16 bit patterns:

| Rank | Local index | Ferrule | HF | Semantic source row |
| ---: | ---: | --- | --- | ---: |
| 0 | 581 | `bfec0000` | `bfeb0000` | Q 581 |
| 0 | 919 | `b9cd0000` | `b9ce0000` | Q 919 |
| 0 | 1359 | `bda70000` | `bda80000` | K 2383 |
| 0 | 1547 | `bd170000` | `bd180000` | K 2571 |
| 0 | 3834 | `bfad0000` | `bfac0000` | V 5882 |
| 1 | 710 | `39c30000` | `39c40000` | Q 1734 |
| 1 | 1723 | `3daa0000` | `3da90000` | K 3771 |
| 1 | 2696 | `3ccf0000` | `3cd00000` | V 6792 |

Independent Torch QKV matches **all 4,096** archived conv-history tail words on
each rank. HF replay from Ferrule's QKV reproduces Ferrule's complete F32 SSM
state exactly, while retaining the original archived differences of 384/128
state words. Thus those state differences are explained by differing upstream
QKV, not a demonstrated convolution, state-frontier, normalization, or chunk
recurrence error. No GDN arithmetic fix or oracle rewrite is justified.

CPU `math.fsum` of checkpoint-decoded BF16 products finds Ferrule agrees with
the rounded high-precision dot on 7/8 disputed indices; HF agrees at rank 1
index 2696 instead. Across all outputs, rounded-F64/Ferrule differences are
0/4,096 and 1/4,096; rounded-F64/HF differences are 5/4,096 and 2/4,096. This
supports investigating F32 accumulation/dispatch near BF16 rounding boundaries,
but F64 is not the chosen F32-accumulator oracle and cannot replace it.

Next minimal operator gate (not yet run): use the same pinned BF16 input and
FP8/scales for the local QKV projection only, capture decoded weight words,
and compare the production three GemmEx rectangles (N=1024,1024,2048; M=1,
K=2048) against one concatenated native N=4096 projection. The source dispatch
is `standard/cuda/hybrid_rank/mixed.rs::mixed_linear_inner`; preparation chooses
`NativeBf16GemmEx` in `mixed_layer.rs`. The reference selects/concatenates the
three semantic ranges before one `F.linear`. Segment geometry, actual decoded
operands, and CUDA12.4/12.9 library accumulation must be distinguished before a
general implementation change. Do not apply per-index adjustments or widen
state tolerances.

HF replay sampled peak device memory was 426,770,432 bytes per selected GPU;
this is a sampled driver high-water mark, not an exact allocator budget claim.
The monitor recorded `observed_within_targets` and no errors, but marked
continuous coverage and exact peaks false and process GPU attribution unknown.
All nonselected device memory deltas were zero. The original full resident
capture's peak reserved allocation remains 19,086,180,352 bytes per rank; no
new full-image allocation occurred here. Original references, numerical budgets,
F32 profiles, and full40 production fail-closed status remain unchanged.

Validation for this continuation: six Python parser tests passed; CUDA12.4
`sm_86` hidden-device compilation/test execution passed 15 Mixed-layer and 13
rank-precision CPU tests (11 GPU/header tests ignored); targeted `--no-deps`
Clippy for the library and both test targets passed with `-D warnings`, and
`cargo fmt -p ferrule-model --check` passed. An initial command used invalid
`FERRULE_CUDA_ARCH=sm86` and failed at build-script validation; rerunning with
`sm_86` passed without source changes. No native numerical code changed.

## Combined native Mixed QKV implementation

The fixed-input gate was completed before changing production preparation. With the
retained rank-local BF16 input and original FP8/scales, CUDA12.4 Ferrule and pinned
CUDA12.9 HF both produced the same result for one concatenated `N=4096` QKV
projection and its downstream BF16 convolution/F32 SSM. The prior segmented
`N=1024,1024,2048` path matched segmented HF, but differed from the independent
concat reference at 5/4096 rank-0 and 3/4096 rank-1 QKV words. The concat path
matched HF and the archived QKV/SSM reference at 0 words unequal. This is a
geometry/accumulation result, not a tolerance or GDN correction.

Production Mixed preparation now exposes `NumericFp8RankPlan::mixed_gemmex_qkv_layout`
and `PreparedNumericFp8RankLinear::packed_mixed_gemmex_qkv`. It proves every
semantic Q/K/V read is complete 128-row/128-column BF16-scale geometry, retains
all source identities and segmented checkpoint reads, then publishes one validated
encoded FP8/scales payload in Q→K→V order. `HybridMixedBlockPlan` admits the
combined native GemmEx view and charges the exact packed device grants; the
header estimator includes the largest combined GemmEx workspace and the bounded
encoded host packing overlap. W8A16, F32, non-QKV numeric segments, and the
single-owner profile remain unchanged and partial scale geometry fails closed.

Fresh authenticated production receipt:
`target/validation/tp2-mixed-claim-gates/qkv-production-regressions-01/`.
The model-owned production gate passed on both ranks with 4096/4096 words exact,
one resident numeric segment per rank, zero execute-time artifact uploads, and
no checkpoint access after both rank images were prepared. Existing generic
GQA/GDN, top-8 local-sharded MoE, and token-to-logits MoE endpoint gates also
passed. The supervisor drained (`echild=true`) and recorded all eight GPUs idle
and released. The endpoint telemetry record is marked `blocked`: its only
observer sample saw zero bytes on rank 1, failing the observer's positive-peak
requirement. Its test exit was zero with no reported monitor errors, and
custody/release verification passed. All resource samples lack continuous
coverage and exact process GPU attribution; some regression samples also saw
1 MiB nonselected-device transients. Do not interpret them as full-model peak
measurements or claim all nonselected deltas were zero.

Fresh staged binary pins:
- rank production gate: `c6b34bd3c06d79851fef9897edde1a55e0d9f548f420a8d6badff435a67cbc30`
- generic layer gates: `cbbaa082e036d03b1124618d19e942eb2e19ea0e9405750986b06b3f544a5732`

Exact fresh-regression counts across both ranks:

| Gate | BF16 output words | BF16 conv-state words | F32 SSM words |
| --- | ---: | ---: | ---: |
| Production numeric QKV | 8,192 | not part of this gate | not part of this gate |
| Generic GQA/GDN/GQA | 112 | 360 | 144 |
| Generic top-8 MoE | 112 | 360 | 144 |
| Token-to-logits MoE endpoint | 154 | 360 | 144 |

Each generic gate exercised six invocations per rank, three COW append aborts
and reset, with the existing production runtime route/data/control adapter.
The earlier geometry receipt independently checked 262,144 F32 SSM words and
16,384 BF16 history words per rank for every candidate. The new QKV gate uses
CPU exports of `hf_concat.qkv`, never Ferrule outputs as expected values.

Hidden CUDA12.4 compilation and strict targeted Clippy passed. The targeted
suites passed 31 CPU tests, plus the explicit real-checkpoint header test.
For the existing header test's admitted capacities, per-card total model grants
remain 19,212,773,632 bytes; including the test's explicit 128 MiB owner margin,
19,346,991,360 bytes fits 22 GiB. The largest existing output-head GemmEx view
already dominates the combined QKV workspace. Packed-QKV host staging is now
included (aggregate host staging 20,752,384 bytes). These are capacity-specific
estimates, not new measured full40 peaks. Source and binary pins are summarized
in `qkv-production-regressions-01/model-handoff.json`.

These are model-slice receipts, not full40 admission. The owner must rebuild its
claim-aware full40 numerical gate against these model sources and re-pin it;
that binary was not rebuilt or run here. No runtime/composition/backend/CLI
source was changed and no full40 profile was enabled.

## Capital all-layer diagnosis: shared gate/up geometry fixed

One complete Capital prefill was captured from a single claim-aware resident
image, not forty per-layer reloads. Receipt:
`target/validation/tp2-mixed-claim-gates/capital-all-boundaries-01/`.
The observer completed all collectives and output-head work before rollback;
no numerical comparison or first-difference exception ran inside the collective
schedule. Capture cap is configurable up to 512 MiB total (half per rank).
Chunk workspace is intentionally excluded; every layer's hidden/residual/norm,
attention partial/reduction, GDN state, router, shared/routed partial, and
available projection tensors are retained. Paired gate/up operands/results and
GQA Q/K/V/core observations were added as disabled-by-default diagnostics.
There are 7,930 observations and 172,075,040 tensor bytes per rank, excluding
the separately saved final logits; the cap was not reached.

The offline mapper compared 925 distinct immutable archive fields in 1,128
observations, not the full 2,178-field two-call acceptance inventory. Unmapped
fields are explicitly diagnostic-only. It reproduces the reported Capital
prefill result exactly: 755,393/1,241,600 tolerance failures, maximum 0.5625.
The first mismatch is layer 2's shared-expert partial, after exact layers 0/1
and 74 preceding exact archive observations. Layer-2 GDN state, attention
partial/reduction, router IDs/weights, and routed partial are exact. Shared
output differs at 2/10,240 rank-0 words and 7/10,240 rank-1 words; one difference
survives the late reduction and residual into layer-2 hidden. Later differences
therefore cannot be used to blame GQA or the output head without same-input
operator evidence.

Supplemental pinned HF replay, without any full-image reload:
`capital-all-boundaries-01/shared-geometry/numerical/`.
This uses the unchanged independent logical-TP2 operand partitioning and HF
source. Archive previous hidden plus attention reproduce the captured residual
and post-attention norm input/output exactly. Native HF shared MLP executes
separate `gate_proj` and `up_proj`; native routed experts execute one
`gate_up_proj`. The old Mixed shared path incorrectly reused the routed
concatenated GEMM shape. Exact first differing upstream BF16 values (expanded
F32 bits) are:

| Rank | Shared up local index | Old paired GEMM | Independent separate HF |
| ---: | ---: | --- | --- |
| 0 | 1112 | `b9610000` | `b9620000` |
| 1 | 832 | `bc250000` | `bc260000` |

Gate outputs are exact. Each up discrepancy causes one SwiGLU product word to
differ. HF concat controls reproduce every old gate/up/product/output word;
same-input SiLU/product and down projections are also exact. Independent native
separate-projection MLP outputs reproduce the archive exactly. No scale-origin,
state, BF16 conversion, or threshold correction is justified.

General model fix:
- `hybrid_rank/mixed.rs`: pair numeric native gate/up only for
  `TensorRole::RoutedExpertGate`. Shared and dense native Mixed MLPs use their
  existing independent linear projections for every M and every layer.
- `hybrid_parallel/mixed_block.rs` and `mixed_estimate.rs`: warm/charge the
  doubled gate/up view only for routed experts. Combined GDN QKV is unchanged.
- No backend/runtime/composition/CLI changes, profile enablement, tolerance
  changes, per-layer/expert hardcode, CPU fallback, or F32 weight mirror.

Fresh production replay and regression receipt:
`target/validation/tp2-mixed-claim-gates/capital-shared-fix-regressions-01/`.
All three exact selectors passed:
- `frozen_tp2_capital_shared_mlp_source_geometry`: both prepared rank images
  resident together; checkpoint fixture removed before execute; gate 1,280,
  up 1,280 and shared output 10,240 words per rank exact; zero execute uploads;
  allocator live requested bytes zero after owner retirement.
- `frozen_tp2_local_experts_shared_late_add`: routed concat and late-add
  regression passed.
- `gpu::frozen_tp2_tokens_moe_logits_bf16_program`: production runtime adapter,
  154 output words, 360 conv words and 144 F32 SSM words exact across both ranks;
  12 invocations, COW/fork/reset preserved.

Hidden CUDA12.4/sm_86 checks passed 33 targeted CPU tests and strict Clippy for
model lib/three test targets. Five full-boundary mapping tests passed. The
full-image, supplemental HF, and fixed replay supervisors each drained with
`echild=true` and all eight GPUs idle/released. Resource monitors do not establish
continuous/exact process peaks. Full-image allocator peak reserved bytes were
19,086,180,352/card; fixed replay is not a new whole-model budget measurement.

Fresh binaries for rebuild handoff:
- rank gate: `dc11bd077543ec69cedac059da5181d02fefaf54b0f8483a646426d5a2493d89`
- layer gate: `ba9dedee3019b35b4d9e75ec37c8772d5b6a29f77a696fc61e5eefce3b29c9ea`
- independent shared expected: `e6ef06f6dac38aaca9496c2658075f5d8aac44b508f073a58705c760df65c51f`

Exact source pins and receipts are in
`capital-shared-fix-regressions-01/model-handoff.json`. One rebuilt full-quality
rerun is now the next gate. It has NOT been run after this fix; later independent
causes are not excluded and no full40 numerical-pass claim is made.

## Native source trace / committed KV export (CPU-only handoff)

No new profile/decoder loop or changed oracle inventory/tolerance. Runtime,
composition and paper are untouched. Existing `set_diagnostic_trace` remains.

### Runtime API

`CudaStandardDecoderOperators::set_native_diagnostic_trace(limits, filter, callback)`:

- `limits: CudaDiagnosticLimits`, constructed with `new(max_tensor_f32_bytes)?`.
- `filter: FnMut(&CudaDiagnosticDescriptor) -> bool + 'static`.
- `callback: FnMut(&CudaNativeDiagnosticEvent<'_>) -> Result<()> + 'static`.
- These types are exported from `ferrule_model::transformer`.
- Replaces, rather than combines with, the legacy callback. Filter runs once
  before any trace-only GPU allocation/callback, including over-cap metadata so
  the writer can report rejection. A true filter result cannot bypass the cap.
- Descriptor includes native dtype, ND geometry/element strides, global layer,
  source parameter partitions and local attention heads/replication owners.
  `event.parameter` retains checkpoint dtype/shape/source identity/transforms.
  Replicated norm partition coordinates are `[elements,1]`, not activation axes.
- GDN cohort metadata includes sequence index, packed row IDs, logical positions
  and input frontier. Join to runtime invocation/session/phase metadata. All
  execution-time observations remain provisional, not committed.
- Download F32 with `download_f32_buffer`; BF16 with `download_bf16_buffer` (native
  two-byte D2H, exact host-only expansion for serialization; no GPU expansion).
- `router.probabilities`: actual `[rows,experts]` F32 values stored by the same
  full-softmax kernel before top-k. No replay/recomputation. Native `router`
  remains BF16 logits with selected IDs/F32 weights exposed as controls.
- Optional probability ABI pointer requires rebuilding backend consumers. The
  disabled kernel specialization has no additional probability store.

### Admission

Native observation and committed-KV export need zero expansion/device scratch.
Only the optional router sink allocates trace-only GPU memory; the existing
owner holds it through completion/Unknown. `router_scratch_grant_bytes(M,E)` gives
its exact allocator grant (M32/E256 = 32,768 bytes), rejecting a source larger
than the cap. `device_scratch_grant_bytes()` is the conservative cap-based bound
(8 MiB at an 8 MiB export cap), not an arbitrary 64 MiB reserve. `host_tensor_bytes()`
bounds native download plus F32 view (12 MiB at that cap). Writer/index/byte-file
buffers, descriptor metadata, all-rank duplication and shared allocator segment
reservation remain caller-owned budgets. The legacy callback is not covered by
these native-mode bounds.

### Post-publication retained KV + append

The existing `PagedKvBackend<TypedCudaPagedKvPool<S>>` exposes:

```rust,ignore
pub fn export_committed_bf16_kv(
    &mut self,
    source: &ferrule_common::host_kv::CommittedKvSource,
    local_kv_layer: usize,
    max_chunk_words: usize,
    observe: impl FnMut(
        ferrule_backend::cuda::operators::kv::CudaCommittedBf16KvChunk<'_>,
    ) -> Result<()>,
) -> Result<()>;
```

Use existing `bind_committed_logical_owner(manager_owner)` first. AFTER actual
journal/global publication, acquire/revalidate the live logical source lease,
with `positions=0..committed_tokens` and all retained + appended page IDs. Keep
that authority lease alive throughout export. A caller-constructed descriptor
alone is not runtime session authority; the backend checks physical custody and
bound-manager identity, not the runtime's session registry.

Derive `local_kv_layer` from `plan.layout().state_schema().entry(global_layer)`.
For GQA, `plan.layout().attention(global_layer)` supplies local KV head range,
head dimension and replica owners. Each callback contains native `&[u16]`,
positions, plane, compact local KV layer, page/slot/epoch, token width, and the
source's owner/state slot/generation/incarnation. Logical ND shape is
`[chunk.positions.len(), kv_heads.len(), head_dim]`, native strides
`[kv_heads.len()*head_dim, head_dim, 1]`; physical page slots are not logical rows.
Join transaction/session/phase to the successful parent publication receipt, not
to execution_end. Runtime may stream chunks into the full expected tensor in
logical position order; unused tail words are never exported.

Export rejects active/prepared/Unknown custody, foreign manager, nonresident or
repeated pages, partial frontier, wrong schema/layer/dtype, and offloaded/chunked
storage. No new device or pinned allocations. Backend helper
`committed_bf16_kv_trace_host_bytes(source, max_chunk_words)` computes native host
payload plus the exact per-page mapping table. Add caller's F32 serialization
buffer separately. D2H failure poisons physical custody; model returns typed
CompletionUnknown and cannot clear it with a later fence. A callback/storage
error after a completed read does not relabel data as a commit.

### Validation / pending runtime work

CPU CUDA12.4 sm_86 build and targeted strict library Clippy passed with
CUDA_VISIBLE_DEVICES empty. New CPU trace tests: 4 pass; committed-KV validation:
2 pass; backend ABI layout and router policy contract: 1 pass each. No GPU query
or execution. Prepared, compiled, ignored GPU gates:

- `transformer::standard::cuda::diagnostic::source_tests::native_bf16_trace_filter_has_no_device_expansion`
- `cuda::operators::kv::transaction::diagnostic::gpu_tests::committed_bf16_native_prefix_append_and_active_rejection`
- `gpu::full_softmax_actual_probability_sink_uniform_and_rejections`

Runtime still must switch its owner-local observer to the native API, serialize
these fields, admit calculated grants, and call export with logical authority
only after publication. The native API is not wired by this model-only change.
Do NOT mark the 2,178-tensor required inventory complete until the real captured
candidate has been checked against the unchanged oracle inventory.

### Lifecycle review for runtime native-trace integration

- Metadata filters now also see over-cap tensors; record rejection and leave
  required inventory incomplete. The same hard cap still prevents callback and
  source-sink allocation. CUDA readback failures must propagate; only host writer
  failures may be recorded as ordinary capture failures.
- Physical export cannot authenticate publication or a logical lease from
  `CommittedKvSource` alone. After real publication, use the existing manager's
  `committed_kv_source`, `acquire_committed_read`, `validate_committed_read`, then
  export using `lease.source()`. Only release via `release_committed_read` after
  proven physical completion. Unknown retains the lease and owner resources;
  ordinary writer failure after completed D2H permits normal lease release.
- Generation zero is valid in the existing page manager (including HostKV live
  tests). Export no longer invents a nonzero-generation requirement. The manager
  must still reject stale generation/incarnation/frontier/page mappings exactly.
- Existing committed-read authority rejects shared/aliased pages and pending
  reservations. If it cannot issue a lease (e.g. a still-shared fork prefix),
  do not bypass it or fabricate authority for tracing; report that capture seam.
- Install/clear only outside an observer callback; neither the native callback
  nor the committed-KV host callback may re-enter its owner.
