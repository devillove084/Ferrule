# Qwen3.5 CUDA hybrid TP × EP runtime

[README support matrix](../../../../README.md#support-matrix) ·
[TP1 PP×EP scope and evidence](../../../../docs/QWEN35_PP_EP.md) ·
[Model rank API](../../../ferrule-model/tests/support/hybrid_rank_cuda.md)

## Supported production factory

- Strict full-depth Qwen3.5-35B-A3B numeric FP8 checkpoint.
- CUDA **thread** PP1/2, TP2/4, EP2; DP/SP/CP1. Process TP is rejected.
- Existing PP2 TP1 EP2/4/8 and resident TP1 EP routes are unchanged.
- F32Tf32x3 is default. `--numeric-fp8-precision w8a16-bf16` is explicit opt-in,
  **quality-unaccepted**, not a same-precision replacement or CPU fallback.
- Full shared host expert prewarm; every child eagerly owns its complete scoped
  expert set. No prefix cache, local expert-cache overrides or rank replay.

The execution path is `PipelineParallelExecutor` → existing owner worker/journal
→ `TensorScopedStateProgram` → `CudaHybridRankSegment`. There is no nested TP
coordinator. The decorator registers the model-owned V2 schedule and forwards
root/expert quiescence; Pending cannot acknowledge Ready/Rollback/Abort, and
Unknown retains custody. The CUDA host-boundary wrapper calls
`execute_cuda_bound`; all layer math is on CUDA.

`HybridCudaDevice::admit` and the model's opaque non-cloneable preparation claim
cover rank image, typed local KV pool and lifecycle construction.
`prepare_rank_with_claim` authenticates the token; `finish` consumes success.
Each rank gets a dedicated owner-local device/context, including colocated ranks.
No model/backend/HostKV changes are required by this runtime integration.

## Typed capability and production validation

`HybridExpertTopology::new` accepts general/reference geometry; it is not an
execution permit. The former no-argument, unconditional production rejection is
replaced by an explicit `HybridExecutionProfile` containing `Qwen35Profile`,
`ModelExecutionBackend`, `PipelineRankBackend`, `ParallelismPlan` and
`NumericFp8Profile`. API definitions live in
[`model_factory::qwen35_composition`](../../../ferrule-composition/src/model_factory/adapters/qwen35/composition.rs).
The `Qwen35CompositionAdapter` owns family/backend selection and evidence labels;
`HybridExpertTopology` retains only model-neutral partition, roster, source and
ownership checks. Neither is another coordinator.

| Declared profile, PP1/2 TP2/4 EP2 DP/SP/CP1 | `execution_capability(profile)` | `validate_production_execution(profile)` |
|---|---|---|
| CUDA thread F32, TP2 | `F32WithFullModelStrictEvidence` | `Ok(())` |
| CUDA thread F32, TP4 | `F32FactoryOnly` | `Ok(())`; no full-model strict evidence |
| CUDA thread W8A16 | `W8A16QualityUnaccepted` | Typed `W8A16QualityUnaccepted` error |
| CPU, CUDA process TP, unsupported degrees/profile, sparse/misordered KV owners | Typed rejection | Typed rejection |

Only F32 **TP2 EP2**, PP1 and PP2, has the full-model strict evidence recorded
below. The capability label describes that frozen profile-level evidence, not a
fresh validation of the caller's checkpoint, GPU, context or placement. TP4 is
implemented and admitted but must not inherit TP2's strict acceptance. TP1 PP/EP
has independent evidence and is outside this validator's scope.

Both queries are metadata-only and run in CPU builds. `Ok` is not CUDA feature
availability, device readiness, a memory reservation or general production
readiness. The actual factory still requires strict full-depth text-only binding,
immutable source identity, full prewarm, explicit devices and aggregate admission.
It uses `Qwen35CompositionAdapter::execution_capability` so the existing explicit experimental W8A16 route
remains available without passing the production-quality gate. There is no CPU
fallback or process TP enablement.

The accepted KV roster is exactly contiguous PP-major TP roots in replica zero;
EP2 children follow all roots in stage order, with each stage's first TP root as
its source. EP owners never join the KV cohort. Every stage must expose a
full-attention KV plane. General sparse/nonzero-source model fixtures do not
establish factory support for those layouts. Rejections retain
`HybridCapabilityError` in `ferrule_common::Error::ModelSource`.

For existing API callers, import the profile/capability/error types from
`model_factory::qwen35_composition` and call
`Qwen35CompositionAdapter::new(&topology).validate_production_execution(profile)`
(or `.execution_capability(profile)`), not a method on the topology. The similarly
named `HybridTensorCpuSegment` reference-only rejection is unchanged. CUDA owners
remain feature-gated; pure metadata policy and its public API do not require CUDA.
Unknown completion/custody is not inferred from capability and remains unchanged.

Default-feature, payload-free API coverage:
`cargo test -p ferrule-composition --test qwen35_composition`.
The existing `tp_ep_capability` unit tests also compare this adapter against the
actual factory metadata admission, including process/CPU/degree/roster rejection.

## Placement and accounting

Device-list order is **all PP-major TP roots, then stage-ordered EP children**.
EP is not a multiplying mesh axis. Logical owner IDs never alias, even when
physical device ordinals repeat.

| PP/TP/EP | KV roots | EP children | Device entries |
|---|---|---|---|
| 1/2/2 | 0,1 | 2,3 (source 0) | 4 |
| 2/2/2 | 0,1; 2,3 | 4,5 (source 0); 6,7 (source 2) | 8 |
| 2/4/2 | 0..3; 4..7 | 8,9 (source 0); 10,11 (source 4) | 12 |

For PP2 TP2 EP2, `0,1,2,3,4,5,6,7` uses eight GPUs. The alternative
`0,1,2,3,0,1,2,3` colocates roots/children on four GPUs; it is admitted only if
**all** context charges fit. For PP1 TP2 EP2, `0,1,0,1` colocates on two GPUs.

Admission separately accounts for:

1. Full model logical KV pages, including all configured full contexts.
2. Sum of actual rank-local physical KV payloads, including GQA replicated heads
   and 2L shadow/COW slots. `--kv-cache-mb` limits this sum, not logical KV.
3. Model allocator-rounded weight/workspace/KV grants and `(2*S+1)` recurrent
   state grants per rank; separate rounded expert weights/scales, conservative
   activation grants and per-context 64 MiB allocator/context headroom.
4. Full-vocabulary/activation wire copies, service collective inputs/outputs plus
   caller-held results, bounded pinned slots, prepared host images, shared expert
   payload, all owner metadata and stacks.

A single immutable full `Arc<BoundDecoderResources>` seals root slices and global
expert identities. Source freshness and actual child Ready/residency are checked.
Budgets fail closed; admission never shrinks contexts, slots or workspace to fit.
Live memory admission is not a reservation against other processes allocating later.

## CLI examples for GPU owners

Replace `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8` with the actual local checkpoint.
These are existing parsed flags; no new execution-enable switch is required.
The explicit **`--max-tensor-mb 2048` is required for this TP fixture**: the
replicated embedding is 248320 × 2048 × 4 = 2,034,237,440 bytes (1940 MiB) in F32,
even though its checkpoint BF16 payload is 970 MiB. The unchanged 1024 MiB global
serving default correctly fails `rank dense parameter exceeds local read budget`.
This is a per-parameter read/materialization limit, not total VRAM or KV capacity.

For API callers, set `ServingModelOptions.max_tensor_mebibytes = Some(2048)`.
Setting only `ModelFactoryOptions.max_tensor_mebibytes` is insufficient: absent
explicit serving intent, Qwen35 resolution restores the 1024 MiB default. The
full35B public tests now assert the resolved `max_tensor_bytes == 2048 << 20`.
No default/admission bound was loosened.

```sh
cargo run --release -p ferrule-cli --features cuda --bin ferrule -- serve /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --backend cuda --engine pipeline --rank-backend thread --pipeline-parallel 1 --tensor-parallel 2 --expert-parallel 2 --devices 0,1,2,3 --expert-prewarm full --max-tensor-mb 2048 --ctx-size 32 --max-active-sequences 2 --max-batch-tokens 4 --prefill-chunk-size 4

cargo run --release -p ferrule-cli --features cuda --bin ferrule -- serve /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --backend cuda --engine pipeline --rank-backend thread --pipeline-parallel 2 --tensor-parallel 2 --expert-parallel 2 --devices 0,1,2,3,4,5,6,7 --expert-prewarm full --max-tensor-mb 2048 --ctx-size 32 --max-active-sequences 2 --max-batch-tokens 4 --prefill-chunk-size 4
```

For W8A16 add exactly `--numeric-fp8-precision w8a16-bf16`. Retain its
quality-unaccepted label until independent GPU quality acceptance.

## Maintained independent full40 F32 strict tests

The PP/TP GPU-owner snapshot harnesses now live in
[`tests/unit/engine/model_factory/full40_strict`](../../../ferrule-composition/tests/unit/engine/model_factory/full40_strict/README.md).
PP2 TP1 EP2/4 and PP1/2 TP2 EP2 share one oracle/comparator and lifecycle harness,
with thin private factory adapters. Use the exact IDs, explicit model/reference/
device/report environment and manual opt-in commands in that README.
The new suite compares every vocabulary element with unchanged `atol=rtol=2e-4`;
PP route/raw state tensor traces are **not** directly compared. W8 remains
quality-unaccepted. Migration compile/Clippy is **not** a current-tree GPU pass.

## Ignored real GPU tests

Tiny: two GPUs, actual runtime expert children, PP1/2 TP2/4 against TP1 CUDA EP,
full logits over prefill/decode, fork/COW, pre-cancel, publication and release.
Both numeric profiles run, without a CPU oracle.

```sh
CUDA_VISIBLE_DEVICES=0,1 cargo test -p ferrule-composition --features cuda --lib tp_ep_gpu_tiny_matches_tp1_ep_full_logits_and_custody -- --ignored --nocapture --test-threads=1
```

Full35B: public planner/build factory, generation, retained session reset, cancel,
release and shutdown. These are runtime smoke/custody tests, not full35B numeric
quality acceptance. They explicitly request the 2048 MiB serving tensor budget.
Use the exact owner count matching the selected test.

```sh
FERRULE_QWEN35_TP_EP_MODEL=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 FERRULE_QWEN35_TP_EP_DEVICES=0,1,2,3 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::tensor_tests::tp_ep_gpu_full35b_public_factory_pp1_f32_default -- --exact --ignored --nocapture --test-threads=1

FERRULE_QWEN35_TP_EP_MODEL=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 FERRULE_QWEN35_TP_EP_DEVICES=0,1,2,3,4,5,6,7 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::tensor_tests::tp_ep_gpu_full35b_public_factory_pp2_f32_default -- --exact --ignored --nocapture --test-threads=1
```

Explicit W8 tests end in `pp1_w8a16_opt_in` / `pp2_w8a16_opt_in` instead of
`pp1_f32_default` / `pp2_f32_default`. GPU visibility must include every listed
ordinal; ordinals are relative to `CUDA_VISIBLE_DEVICES`.

## No-GPU validation

Use command timeouts of 1,800,000 ms, not 120 seconds. In this workspace CUDA
12.4 is at `/home/admin/wxw403883/anaconda3/envs/tao`.

```sh
CUDA_VISIBLE_DEVICES='' PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin:/usr/local/bin FERRULE_CUDA_ARCH=sm_86 cargo check -p ferrule-runtime -p ferrule-composition -p ferrule-cli --features cuda --tests

cargo test -p ferrule-composition --features cpu-reference --lib model_factory
cargo test -p ferrule-runtime --features cpu-reference --lib parallel::pipeline
```

For linked hidden-GPU tests on this machine, additionally set
`LIBRARY_PATH=/home/admin/luhuanbing.lhb/Ferrule/target/validation/composition-refactor-cuda124-link:/usr/lib64`
(the directory must provide libcudart, libcublas and libcublasLt linker names) and
`LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64`.
Run `cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::tensor_tests`
with those paths, `CUDA_VISIBLE_DEVICES=''` and `FERRULE_CUDA_ARCH=sm_86`, without
`--ignored`, to execute metadata admission and compile the ignored GPU tests.
The mount is `parallel_tensor.rs`, not a standalone TP/EP production factory;
see the [composition responsibility map](../../../ferrule-composition/src/model_factory/parallel/README.md).
Hidden validation does not rerun or replace the frozen GPU evidence below.


## GPU owner acceptance evidence (frozen snapshot, not rerun here)

Verified against `target/validation/qwen35-35b-current/runtime-tp-ep-owner/SUMMARY.json`,
per-run commands and the pinned-source diff:

- F32 full40 PP1 TP2 EP2 and PP2 TP2 EP2 both passed. Each compared 1,986,560
  base logits (8 rows); 3,973,120 elements including fork/recreate checks.
  `max_abs=3.647804260253906e-5`, `atol=rtol=2e-4`, zero failures/nonfinite.
- v4 public-factory smoke passed with explicit 2048 MiB serving intent. Original
  and v2 runs failed at the unchanged 1024 MiB rank dense-read limit.
- v5 strict binary SHA256:
  `dee47f80503acbcfdc8ed12b0f812598be20a2deaeead1ecde83367aa23485fa`.
- Exact strict test ID in that frozen binary (not registered in the worktree):
  `model_factory::qwen35::tensor_pipeline::tests::owner_strict::tp_ep_owner_full40_f32_all_tokens_strict`.
  The implementation has since been migrated to the maintained suite above with
  four distinct topology-specific IDs. This **old snapshot ID** is not a worktree
  Cargo filter; do not mistake zero matching tests for acceptance.
- Snapshot tiny placement used distinct GPUs (with selected TP4 colocation),
  unlike the worktree's two-device colocated tiny fixture; that placement change
  was not imported. The strict evidence covers TP2, not full35B TP4 or every
  colocated mapping. W8 quality remains failed/unaccepted; no throughput claim.

For current-source validation use the maintained strict tests above, not the
frozen binary. Original commands/hashes remain in the owner artifacts for
provenance only. The migration was checked with GPUs hidden; a sole-owner GPU
rerun is required to attach a pass to the maintained tests/current source.
