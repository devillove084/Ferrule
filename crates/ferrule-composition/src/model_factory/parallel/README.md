# Parallel composition

This directory replaces independent PP×EP and TP×EP construction algorithms with
validated geometry, provider budgets and shared pipeline assembly. The public
Resident/Pipeline selection and frozen builder contracts are unchanged. Engine
kind, PP/TP geometry, expert attachment, transport and numeric profile are
separate axes; an EP attachment does not turn a resident root into a pipeline.

## Catalog and profile flow

`model_factory::MODEL_IMPLEMENTATIONS` remains the single built-in catalog.
`SelectionRequest` normalizes explicit serving/profile intent; each catalog
entry's selector returns a `ResolvedStrategy` carrying construction, backend,
option policy and topology policy together. `ResolvedConstruction` freezes the
Resident/Pipeline builder. The resolved request/build path does not rediscover a
constructor by matching arithmetic or parallel degrees.

Family-owned selectors still enforce checkpoint/profile support and option
consumption. Named profiles, model precision, numeric FP8 arithmetic and KV
policy are not interchangeable. `selection.rs`, `bf16_serving`, `mixed_serving`,
`dense_f32_kv_policy` and `qwen35_composition` own those decisions; this directory
does not add a second catalog or upgrade a quality label.

`execution_policy.rs` supplies pure typed topology, transport, capability,
placement and explicit-override validation. Family/provider adapters supply the
supported degrees and placement rules. This structural validation neither opens
a device nor proves checkpoint compatibility, live capacity or numerical
quality. Strict family binding and owner-local admission remain necessary.

## Responsibilities

| Module | Responsibility | Boundary |
|---|---|---|
| Family adapter [binding](../adapters/qwen35/binding.rs) and [composition](../adapters/qwen35/composition.rs) | Bind the strict family checkpoint/profile, validate family capability, and lower provider inputs | Family recipe/support policy stays outside shared provider planning |
| [inputs.rs](inputs.rs) | Typed numeric provider inputs, capacity limits and explicit expert placement | Inputs are lowered by an admitted family adapter; model identity is provenance, not a constructor selector |
| [bindings.rs](bindings.rs) | Check already-bound resources, full depth, source freshness, parameter limits and consumed/default-only provider options | Does not bind a family recipe or grant family/profile capability |
| [plan.rs](plan.rs) | Payload-free validated `Geometry`: root/expert roster, layer partition and placement; numeric `Plan` composes existing `HybridParallelPlan`, `HybridExpertTopology`, KV capacity and host preparation | Ordinary and dense hybrid pipelines also consume `Geometry`; experts are not an extra KV mesh dimension |
| [budget.rs](budget.rs) | Provider representations, pipeline expert/stage estimates and aggregate physical-card admission | Every colocated owner is charged; TP1 and TP-rank accounting remain distinct, conservative contracts |
| [budget/common.rs](budget/common.rs) | Checked decoder options, expert payload envelopes, process overhead and host-startup calculations | Shared arithmetic, not family checkpoint acceptance |
| [budget/local.rs](budget/local.rs) | Bound-resource resident memory plan and admission | Actual bound-resource accounting remains distinct from family schema estimates |
| [budget/resident.rs](budget/resident.rs) | External-routing resident root and colocated expert physical-card admission | Charges each owner's weights, state, workspace and margins; not a physical reservation |
| [experts.rs](experts.rs) | `OwnerPlan`, stage ownership, eager numeric thread startup, immutable thread/process Ready validation and attachment preparation | Shared with resident EP; does not own process boot/frame preparation or parent KV publication |
| [provider.rs](provider.rs) | Owner-local device and optional expert preparation | Only a stage's designated source starts experts, also with TP2/TP4 |
| [assembly.rs](assembly.rs) | `PipelineAssembly`, `build_bound` and `start_admitted`: admission → host preparation → persistent startup/Ready → engine wrapping | Ordinary, dense hybrid and numeric pipelines share the sequence; typed stage/collective and transport constructors are leaves |
| [resident.rs](resident.rs) | Bound local/resident root construction, external expert preparation and runtime engine wrapping | Resident execution retains root-only KV authority; it is not a pipeline transaction owner |
| [process.rs](process.rs) | Numeric expert boot/frame planning, owned-partition host preparation, process admission and startup | Model-neutral planning does not expand the production numeric binder's accepted families |
| [reporting.rs](reporting.rs) | Wrap runtime engine reports and shutdown observations | Reporting only; the wrapped engine retains execution, publication and custody |

TP1 retains its conservative full routed-root envelope per segment; TP ranks
retain allocator-grant accounting, replicated KV, state shadows and pinned
boundaries. The shared expert ledger takes an explicit representation:

- `PayloadEnvelope`: exact payload weights; workspace `64MiB + activation_bytes`.
- `PhysicalAllocationGrants`: separately rounded weight/scale grants; workspace
  `grant(64MiB) + 16 * grant(activation_bytes)`.

Both retain zero expert state/KV and 64MiB margin per owner. These representations
are not interchangeable estimates. Shared source-cache/warm accounting is not
silently expanded or reduced, and admission is not a memory reservation.

`PipelineParallelExecutor`, `PipelineStageWorker`, the existing expert adapter
and process owners retain execution, commit, publication, Unknown custody,
shutdown and reaping. There is no new coordinator, registry or transaction loop.
Resident EP keeps its resident executor and root-only KV authority while sharing
expert planning/preparation/Ready invariants. Process roots never share a thread
cache or warm unowned expert partitions. Ready failure is logged before cleanup;
ordinary cleanup errors retain primary context and typed Unknown panic custody
is not converted to an ordinary error.

## Minimum family adapter contract

Adding a model requires three pieces, not another per-model PP×EP/TP×EP factory:

1. A model recipe and strict binder for checkpoint schema, storage/precision,
   state/parameter requirements, source identity and supported attachments.
2. One built-in catalog entry with matching family/backend/topology/option policy,
   resolving construction and lowering typed inputs into these shared providers.
3. Operators and state/KV/precision/completion contracts compatible with the
   selected executor. Reuse existing execution only where those contracts fit;
   otherwise implement and validate the missing capability before admitting it.

The public `model_factory::qwen35_composition` API is still mounted by
[model_factory/mod.rs](../mod.rs) from the adapter's composition file. Physical
source extraction is not API removal or a new public plugin-registration API.

In particular, [process.rs](process.rs) shares planning, not universal numeric
checkpoint support: [process_binding.rs](../../process_binding.rs) still binds
only strict Qwen3.5-35B-A3B FP8 for numeric expert boots. Another family requires
explicit binder/boot capability and validation. Process TP, numeric hybrid
process PP×EP and BF16/Mixed expert wire remain unsupported; neither structural
metadata nor code reuse proves numerical acceptance. The canonical
[architecture](../../../../../docs/ARCHITECTURE.md) describes model extension,
and [current acceptance](../../../../../docs/CURRENT_ACCEPTANCE.md) owns the
source-pinned evidence rather than duplicating a status history here.

## Engine, topology and transport support

The table describes guarded construction paths, not new numerical acceptance.
All rows remain subject to family/profile, compiled-backend, source, capacity,
explicit-option and process-platform checks. DP/SP/CP stay 1; automatic
restart/replay is unsupported.

| Path | Engine / topology | Thread owners | Process owners |
|---|---|---|---|
| Ordinary standard pipeline | Pipeline; TP1 PP with optional per-stage EP for supported MoE | CPU-reference or CUDA | Existing standard decoder process transport; required launch/deadline/frame/capacity validation |
| Ordinary dense tensor pipeline | Pipeline; CUDA TP2/4, EP1, supported PP partition | Yes; distinct root devices | No TP process support |
| Strict dense Qwen3.5 hybrid | Pipeline; PP2 TP1 EP1, F32 | CPU-reference or CUDA | Existing dense hybrid process transport; bounded full-vocabulary frames |
| Strict numeric Qwen3.5 resident EP | Resident; PP1 TP1 EP2/4/8; explicit distinct expert devices, root shares expert0 card | Full shared prewarm | Separate resident EP process path with owned-partition prewarm; not process PP×EP |
| Strict numeric Qwen3.5 segment provider | Pipeline; CUDA PP2 TP1 EP2/4/8 | Full shared prewarm; explicit root-then-stage-EP placement and aggregate colocation admission | Unsupported |
| Strict numeric Qwen3.5 tensor-rank provider | Pipeline; CUDA PP1/2 TP2/4 EP2 | Same attachment/source rules; explicit PP-major TP roots then stage experts | Unsupported |

Numeric F32 remains the default and W8A16 remains explicit opt-in with its
quality-unaccepted label. Metadata support for TP4/EP8 is not full40 acceptance.
CPU-reference is a test/reference capability, not production GPU evidence.
Named BF16/mixed/KV-specialized profiles retain their own narrower policies;
they do not inherit every row in this table.

## Remaining dispatch is intentional

Family capability and strict binding are private leaves because supported
checkpoint schemas and quality policies differ. Dense versus numeric host
preparation is a provider choice. TP1 segment versus TP rank selects typed stage
construction, collective wiring and the matching budget representation, not a
second admission/prewarm/startup algorithm. Standard/dense thread versus process
selects transport constructors and owner lifetime handling. Resident EP remains
a separate executor, not a call through the pipeline factory.

## Test responsibilities and stable selectors

Tests live under `tests/unit/engine/model_factory/`:

- `parallel_segment.rs` → `model_factory::parallel::segment_tests`.
- `parallel_tensor.rs` → `model_factory::parallel::tensor_tests`.
- `parallel_tensor_fixture.rs` → `tensor_tests::fixture`.
- `parallel_assembly.rs`, `parallel_expert_budget.rs` and
  `parallel_expert_cleanup.rs` cover shared ordering, budget representations and
  Ready/cleanup diagnostics.

The former combination-factory test filenames have been retired. Module IDs,
the two tiny GPU selectors and four full40 strict selectors are unchanged by
this file-only migration. See the [strict harness README](../../../tests/unit/engine/model_factory/full40_strict/README.md)
for exact GPU-owner commands, reference identities, placement and tolerances.
Historical receipts retain historical names and hashes. The successful pinned
GPU matrix predates this test-path cleanup: it remains evidence for that pinned
binary, not a claim that the renamed tree was GPU-rerun.

## Validation without GPU access

Use an isolated target directory when another writer owns Cargo's normal lock.
Never use `--ignored` or device probes for this validation.

```sh
CUDA_VISIBLE_DEVICES='' cargo test --locked -p ferrule-composition --no-default-features --features cpu-reference --lib model_factory::parallel --target-dir /tmp/ferrule-composition-refactor-cpu
CUDA_VISIBLE_DEVICES='' cargo test --locked -p ferrule-composition --no-default-features --features cpu-reference --test architecture --test hybrid_parallel_policy --test parallel_composition_contract --target-dir /tmp/ferrule-composition-refactor-cpu
```

For this CUDA12.4/sm_86 host, configure the real compiler and all link libraries
explicitly rather than sourcing an environment that selects a GPU:

```sh
export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao
export NVCC=/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc
export PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin
export FERRULE_CUDA_ARCH=sm_86
export FERRULE_CUTLASS_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/vendor/cutlass
export LIBRARY_PATH=/home/admin/luhuanbing.lhb/Ferrule/target/validation/composition-refactor-cuda124-link:/usr/lib64
export LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64
export CARGO_TARGET_DIR=/tmp/ferrule-composition-refactor-readiness
cargo test --locked -p ferrule-composition --features cuda --lib --no-run --message-format=json
cargo test --locked -p ferrule-composition --features cuda --lib model_factory::parallel
```

The link directory must provide libcudart, libcublas and libcublasLt. Obtain the
test executable from the compiler-artifact JSON; check each exact selector using
`<emitted-binary> <exact-test-ID> --list --exact` with GPUs still hidden. A zero
match is a failure. CUDA-enabled filtered execution runs metadata/synthetic
checks and leaves GPU tests ignored; it is not GPU numerical acceptance.


## Internal Mixed resource requirements and routed control

`ParallelExecutionPlan::resource_requirements` combines model-owned rank storage
with a per-rank `ParallelOperatorEstimate` callback. Missing estimates are `None`,
not zero. `known_device_bytes` is a lower bound; `device_bytes` is absent until
all categories and the exact model data/route schedule are present. `admit`
rejects incomplete ledgers. `prepare_internal` only accepts a source-bound model
estimate and distinct PP1/TP2/EP1 placement before invoking experimental owner
construction. Synthetic callbacks cannot authorize startup. The public execution
quality gate and CLI Mixed capability remain closed.

The ledger charges compressed FP8 weights plus scales and native vector weights,
`2*S+1` physical state copies, the model's maximum state-copy workspace, physical
shadow/COW KV slots (including replicated local KV heads), and explicit context
and allocator margins on **each owner**, including colocated owners. Model
provider views feed `ParallelProviderPools::from_plans`: one maximum GemmEx
storage grant and one maximum attention storage grant, with one distinct external
reserve for each family. Forty layers do not create forty provider contexts.
Derived RoPE/attention metadata/recurrent work, operation roots, and pinned/device
boundary storage must be supplied separately; native norm/GDN parameter roots
already included in the weight ledger must not be charged again.

For a routed rank program, construct the existing `MixedRankCollectivePlan` from
`program.sites()`, then call `.with_route_sites(program.route_sites())` before
`MixedRankCollective::prepare`. Pass that capability directly as the model's
`HybridBf16Collective`. `sealed_route_sites` reports the runtime plan's own
metadata; `agree_routes` uses the existing coordinator and compares exact expert
IDs and F32 weight bits between attention output and the single late combined
`RoutedExpertOutput` reduction. No route identities are BF16 tensors. Its explicit
`route_control_budget()` is included once in collective `total_host_bytes`.

`tests/tp2_mixed_contract.rs::real_tp2_mixed_header_storage_probe` is an explicitly
ignored **CPU header-only** test, not a GPU gate. With GPUs hidden and the real
35B source directory explicitly selected, its exact known totals are reportable;
both storage-only gaps and complete model-estimated totals are reported separately.
This is the exception to the generic no-ignored-test validation guidance above.


### Full-segment Mixed assembly handoff

`ParallelExecutionPlan::mixed_programs(capacity)` now derives one model-owned
`HybridMixedRankProgramPlan::for_segment` per root. Endpoint ownership is the
ordinary `LayerSegmentPlan`: the final `OutputHead` BF16 gather is part of the
sealed schedule and transport budget. `mixed_collective_plans(capacity, limits)`
produces the production `MixedRankCollectivePlan` including exact typed route
sites; it does not substitute the legacy F32 agreement/completion packets.
`resource_requirements` rejects hidden-only schedules for a serving segment.

`resource_requirements_from_model` consumes each program's pure
`operation_estimate`, using `ParallelOperatorEstimate::from_program`. It checks
storage and copy grants against the storage ledger, and charges them once. The
model currently reports pageable boundary vectors and zero pinned-transfer-ring
bytes; composition preserves that distinction rather than inventing a pinned
allocation. These are model-estimated envelopes, not measured allocations.

The existing `new_hybrid_cuda_tensor` accepts `HybridCudaTensorStage<P>` over
CUDA rank state/KV types. The sealed `HybridCudaTensorLeaf` admits the legacy
F32 leaf or the experimental `MixedCudaTensorProgram`, not arbitrary BF16
metadata. `HybridExpertTopology::bind_tensor_scopes` authenticates absent or
external EP groups before workers start. No new coordinator or journal exists.

`ParallelExecutionPlan::start_internal` in shared `assembly.rs` consumes the
complete model ledger and the existing provider device leaf. `MixedTensorExecutionPlan`
binds both complete rank programs, typed data/route sites and every legal
attention `(rows,tokens)` view for the actual configuration. Batch8/context128
prepares 996 views; it never reuses a rows2/context1024 claim. The real header
probe reports `19,354,480,640` bytes per root and `55,628,984` host bytes for
that serving capacity. Context and provider
configuration are validated before creating either device, then rechecked on
each persistent owner before payload materialization.

`HybridCudaTensorStage<MixedCudaTensorProgram>::prepare_mixed` calls
`cuda_claim_inputs`, `admit_with_context`, and the model's
`prepare_mixed_rank_program_with_claim`. It assembles the existing BF16 paged KV
pool and verifies initial lifecycle state creation/release before `claim.finish`.
A successful stage is not published before that final fence. Unknown completion
uses the existing retention/panic boundary; the worker retains state/KV and
owns rollback, fork, cancellation, commit, and final publication.

`TensorScopedStateProgram` validates the sealed transaction before physical KV
enter. The model owns the one layer loop and BF16 word exchanges; only the final
logits boundary expands BF16 results into the existing `DenseLogits` F32 host
API. Input is tokens, output is logits; PP>1/hidden activation transport is not
silently enabled. All local expert shards are prewarmed by model preparation;
no external children, lazy cache, CPU offload, or full decoded mirror is added.

## Experimental GPU-owner handoff (not run here)

Build with the CUDA12.4/sm86 environment above, GPUs hidden:

```sh
cargo test -p ferrule-composition --no-default-features --features cuda --test tp2_mixed_execution --no-run
```

Only the GPU owner may run the ignored selectors, with a fresh two-device grant,
`FERRULE_MIXED_GPU_RUN=1`, explicit `CUDA_VISIBLE_DEVICES` (mapped to local 0,1),
and launcher `CUBLAS_WORKSPACE_CONFIG=:0:0`:

- `tiny::mixed_claim_workers_prefill_fork_decode_release`: actual generic
  assembly, resident native Mixed GQA/GDN/MoE image, journal fork/decode/release.
- `full40_independent_logits_prefill_one_decode`: original full40 FP8 source,
  independent serialized paper oracle, full-vocabulary prefill plus exactly one
  recorded decode token per case. Set `FERRULE_MIXED_MODEL`,
  `FERRULE_MIXED_ORACLE_DIR`, and `FERRULE_MIXED_ORACLE_SHA256`. The reader checks
  the manifest/protocol, artifact hashes and original checkpoint shard hashes
  before owner startup. It never imports or executes oracle scripts.

This is a logits/lifecycle gate, NOT the all-boundary/state acceptance canary.
CLI/catalog/profile behavior and advertised quality remain unchanged. Owner
startup logs report model claim categories and observed allocator live grants;
no device allocation measurement is claimed from the CPU metadata tests.
