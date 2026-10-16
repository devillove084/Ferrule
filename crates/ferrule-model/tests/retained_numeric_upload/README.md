# Retained standard numeric-FP8 upload contracts

Production leaf: `CudaStandardDecoderOperators::enable_retained_numeric_fp8_upload(policy_headroom_bytes, ready_timeout)` on an existing, unused bounded numeric TP1 owner. The maintained runner forwards the model-neutral `ExpertTransferPolicy` through `ExpertTransferControl`; the leaf rejects unsupported owners. Defaults remain synchronous. The existing state-dict materializer must have explicit `ExpertReadPipelineOptions`. Both root and worker routing preflight reject metadata-less providers before `expert()` when retained transfer is selected. No environment variable enables this path.

The existing `max_host_bytes` must cover the concurrent packed read reservation, prepared immutable sources, requested pinned slab, actual backend ticket inline size, selected metadata bound, requested worker stack, and caller-supplied policy headroom. A matching accepted lookahead shares its existing packed charge. Source reservation is shared with prepared artifacts, not refunded/re-added while transferring custody. Observations distinguish packed bytes reserved before submission from actually charged read-pipeline bytes, without double counting their handoff.

Headroom is a named caller policy constraint, not measured allocator/driver/page overhead or certified physical RSS/memlock. Choose and qualify it for the deployment; a positive value alone does not establish physical safety. CUDA allocation/pinning may still fail. Such failures conservatively quarantine the owner and host reservation.

The first stage is serialized: six upload copy fences are replaced by a triple ticket and host-observed readiness polling. The existing consumer event/upload cleanup remains. There is **no upload/compute overlap or performance qualification claim**. Polling has an explicit timeout; timeout cancellation performs one synchronous exact-ticket drain, whose driver wait itself is not time-bounded. Cancellation never publishes. Unproved errors/unwinds retain custody and quota, and cannot replace the worker to recover admission.

- `ledger.rs`: production host ledger, exact cap/minus-one before IO, packed reservation handoff, source aliases/weak-witness retirement, pending/cancel proof seam, sticky Unknown, duplicate/stale/overflow/release-once, and worker reuse after provider teardown.
- `provider.rs`: actual private state-dict provider reservation/materialization/custody bridge and disabled legacy capability.
- `owner.rs`: CPU-only exact cache lease/image/ticket generation guards and owner duplicate/Unknown gates; also one ignored real production CUDA miss/consume/evict/reload test against default execution. The ignored test is a tiny integration check, not model-quality qualification.

CPU: `cargo test -p ferrule-model --lib --no-default-features retained_upload`.
The two owner-guard tests require the CUDA feature to compile/link but create no CUDA context and issue no driver calls. Filter to their exact module and skip `real_owner`. The real GPU tests must remain ignored until the GPU owner authorizes qualification. The two new routing tests call the actual root/worker routing entrypoints with a metadata-less provider whose `expert()` panics and counts reads; neither is a CPU execution claim.

## Maintained Qwen3.5 opt-in quality registrations

The existing `NUMERIC_BF16_TEST_RUNNER` mount configures the real owner before the unchanged full40/fresh loops. `bf16_full40.rs` adds only two explicit profiles: `StorageMiddle64RetainedBlocking` and `StorageMiddle64RetainedLookahead`. Their manifest and effective config bind `ExpertTransferPolicy::Retained { policy_headroom_bytes: 16 << 20, readiness_timeout: Duration::from_secs(30) }`. Default profiles keep their historical manifests and synchronous policy. Callback assertions reject duplicate configuration and disabling an installed policy, and require positive retained aggregate reservation evidence plus actual cache uploads after execution. These are assertions/registrations, not GPU qualification.

Unchanged host cap = 64 MiB, metadata = 8 MiB, requested stack = 2 MiB; device/cache/workspace/scratch/scheduler quotas are unchanged. Original Qwen3.5 expert headers contain three 1,048,576-byte FP8 weights and three 128-byte BF16 scale tensors: `P = 3,146,112`. The serial upload reserves `3P + size_of::<CudaNumericFp8UploadTicket>() + 16 MiB + 8 MiB + 2 MiB = 36,701,312 + ticket_inline` bytes (36,701,896 bytes with the pinned x86_64 build's 584-byte inline ticket; 30,406,968 bytes logical slack). Even another packed P fits the logical cap, but production forbids that submission while transferred upload custody remains unresolved. The snapshot reports only the reserved packed floor not already covered by actual read quota. Exact-cap tests cover reservation, demand, materialization, transfer, alias retirement and release for both schedules, including accepted matching lookahead.

The 16 MiB reserve is a named policy constraint, not measured allocator headers, page rounding, memlock, driver allocation or RSS. The 30-second deadline bounds readiness polling after submission, not staging or synchronous timeout cleanup. No upload/compute overlap claim.

All eight ignored full-model selectors use the existing library test target `ferrule-composition --no-default-features --features cuda --lib` and unchanged ATOL `0.00025`, RTOL `0.012`, exact ordered routes and greedy token checks. Prefix for the full40 tests:

`model_factory::dense_bf16_tests::full40::`

- `numeric_bf16_mixed_storage_middle64_retained_blocking_public_factory_full40_logits_routes_states_match_profile_reference`
- `numeric_bf16_mixed_storage_middle64_retained_lookahead_public_factory_full40_logits_routes_states_match_profile_reference`

Prefix for fresh/no-observer/cold tests:

`model_factory::dense_bf16_tests::full40::fresh_quality::`

- `mixed_storage_middle64_retained_blocking_fresh_quality_retained_generation`
- `mixed_storage_middle64_retained_lookahead_fresh_quality_retained_generation`
- `mixed_storage_middle64_retained_blocking_fresh_quality_no_observer_equality`
- `mixed_storage_middle64_retained_lookahead_fresh_quality_no_observer_equality`
- `mixed_storage_middle64_retained_blocking_fresh_quality_cold_owner_restart`
- `mixed_storage_middle64_retained_lookahead_fresh_quality_cold_owner_restart`

Cold requires a retained diagnostic primary with the identical schedule, exact policy and reference pin. No-observer equality requires a newly pinned same-policy diagnostic primary and retains existing 64-call actual-logit fingerprints. Existing v3 read-report controls are unchanged. CPU tests in `bf16_fresh_quality/retained_transfer.rs` reject cross-policy, omitted and modified policy receipts in both directions.

## CPU-only validation commands

No GPU operations are needed for these commands. CUDA-feature commands require the pinned CUDA12.4/sm86 compile/link environment; keep `CUDA_VISIBLE_DEVICES=''` and `CUBLAS_WORKSPACE_CONFIG=:0:0` even for the metadata-only composition tests.

```sh
CUDA_VISIBLE_DEVICES='' cargo test --offline --locked -p ferrule-model --lib --no-default-features --target-dir target/numeric-fp8-residency-audit-cpu retained_upload -- --test-threads=1
CUDA_VISIBLE_DEVICES='' cargo test --offline --locked -p ferrule-model --lib --no-default-features --target-dir target/numeric-fp8-residency-audit-cpu -- --test-threads=1
CUDA_VISIBLE_DEVICES='' cargo clippy --offline --locked -p ferrule-model --lib --no-default-features --target-dir target/numeric-fp8-residency-audit-cpu -- -D warnings
```

Process-local CUDA environment (compilation/linking and CPU tests only):

```sh
export CUDA_VISIBLE_DEVICES=''
export CUBLAS_WORKSPACE_CONFIG=:0:0
export PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin
export CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao
export NVCC=/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc
export FERRULE_CUDA_ARCH=sm_86
export FERRULE_CUTLASS_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/vendor/cutlass
export RUSTFLAGS='-L native=/home/admin/wxw403883/anaconda3/envs/tao/lib -L native=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-35b-current/toolchain/link'
export LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64
cargo check --offline --locked -p ferrule-model -p ferrule-composition --no-default-features --features cuda --lib --tests --target-dir target/numeric-fp8-residency-cuda124
cargo clippy --offline --locked -p ferrule-model --features cuda --lib --tests --target-dir target/numeric-fp8-residency-cuda124 -- -D warnings
cargo test --offline --locked -p ferrule-model --features cuda --lib --target-dir target/numeric-fp8-residency-cuda124 transformer::standard::cuda::residency::retained_upload::tests -- --test-threads=1 --skip real_
cargo test --offline --locked -p ferrule-composition --no-default-features --features cuda --lib retained_transfer --target-dir target/numeric-fp8-residency-cuda124 -- --test-threads=1
cargo test --offline --locked -p ferrule-composition --no-default-features --features cuda --lib model_factory::dense_bf16_tests::full40:: --target-dir target/numeric-fp8-residency-cuda124 -- --test-threads=1
cargo test --offline --locked -p ferrule-composition --no-default-features --features cuda --lib expert_read_reporting --target-dir target/numeric-fp8-residency-cuda124 -- --test-threads=1
cargo test --offline --locked -p ferrule-composition --no-default-features --features cuda --lib storage_middle64_retained_ --target-dir target/numeric-fp8-residency-cuda124 -- --list --ignored
```

Strict composition test clippy is blocked by unchanged `expert_read_reporting.rs:336` (`clippy::too_many_arguments`, 9/7). This stage does not alter that v3 reporting implementation. A separate check may allow only that lint using `-- -D warnings -A clippy::too_many_arguments`; do not label it an unqualified strict pass.

The model library also registers these exact GPU routing regressions under
`transformer::standard::cuda::residency::retained_upload::tests::`:

- `real_root_route_rejects_metadata_less_provider_before_payload_read`
- `real_worker_route_rejects_metadata_less_provider_before_payload_read`

Run only after GPU-owner authorization, with one exact selector and `--ignored --exact --test-threads=1`. Compilation/listing does not establish that their zero-read assertions passed on a real owner.
