# Maintained full40 F32 pipeline acceptance

These ignored **composition library tests** migrate the successful GPU-owner
`pp-ep-owner` and `runtime-tp-ep-owner` snapshot harnesses into the repository.
No test source is loaded from `target/`, no production API is widened, and no
P1/capability gate, precision default or admission limit is changed.

The four maintained tests passed in the pinned
[post-refactor GPU-owner run](../../../../../../../target/validation/parallel-refactor-readiness-20261007/gpu-owner-current-20261007/ACCEPTANCE.json):
PP2TP1EP2, PP2TP1EP4, PP1TP2EP2 and PP2TP2EP2; 3,973,120 comparisons each
including fork/recreate, **15,892,480 total**. This supersedes the earlier
compile-only migration status, not the requirement to rerun after source changes.
The receipt does not directly compare router/GDN/KV tensors or validate process
transport, TP4, broader workloads or performance. See
[current acceptance](../../../../../../../docs/CURRENT_ACCEPTANCE.md) for scope.
W8 remains quality-unaccepted; this suite only tests the F32 default.

## Structure and coverage

- `oracle.rs`: shared independent F32 manifest/report/checksum reader and scalar
  comparison. The existing resident EP8 thread/process strict tests also consume
  this module, including their ordered-route and GDN state trace comparisons.
- `pipeline.rs`: one parameterized public-planner / private-factory acceptance and
  lifecycle harness for all four PP/TP/EP configurations. `pp.rs` and `tp.rs` only
  adapt private admission/executor construction and validate model-local layout.
- Shared `parallel::segment_tests` / `parallel::tensor_tests` mounts expose the
  adapters to the shared `Admission` and `start_admitted`; the common mount is
  `#[cfg(test)]` only.

Preserved strict contract:

1. Complete 40-layer Qwen3.5-35B-A3B numeric FP8 checkpoint; no partial layer run.
2. Independent oracle schema `ferrule.qwen35-fp8-reference.f32.v1`, complete F32
   reference, TF32 disabled in the reference; report/manifest SHA256 equality and
   actual safetensors SHA256 checked via `sha256sum`.
3. Both `hello` and `capital`, all rows of prefill and `decode.0`, **all 248320
   vocabulary values per row**. Exactly 1,986,560 base logits; including fork and
   recreate comparisons exactly 3,973,120 elements. No top-k/argmax substitute.
4. F64 comparison of F32 operands; both finite;
   `abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)`. Unchanged tolerance.
5. Committed token frontier, fork decode vs independent oracle and identical
   original-session logits, pre-admission cancellation preserving publication and
   pages, release/recreate/recompute, zero allocated pages/outstanding work, no
   quarantine, and successful shutdown.
6. Explicit `ServingModelOptions.max_tensor_mebibytes = Some(2048)` plus resolved
   byte-limit assertion. The global 1024 MiB default and admission are untouched.
7. Stage-local source/child identities, compact ten GQA layers, physical KV bytes
   matching admission, and PP*(TP+EP) budgets rather than an EP-multiplied KV mesh.

**Coverage boundary:** these PP/TP owner APIs do not expose router traces or raw
KV/GDN state snapshots. Neither snapshot pipeline harness directly compared
ordered routes or state tensors. The migrated pipeline report explicitly says
these are **NOT compared**. Shared ordered-route comparison is exercised by the
existing resident EP8 strict tests, not claimed for these four pipeline tests.
Cancellation here is pre-admission, not an injected in-flight kernel failure.

## Explicit manual GPU invocation

Run from the repository root, only after the sole GPU owner allocates the devices.
Model/reference paths are mandatory: no NAS or `target` fallback in test code.
The example paths are the existing owner reference locations; substitute local
paths explicitly if needed. CUDA 12.4 and runtime/link libraries must be set up.
RTX 3090 runs should be compiled with `FERRULE_CUDA_ARCH=sm_86`.

Common environment (new report directory per rerun):

```sh
export FERRULE_QWEN35_STRICT_RUN=1
export FERRULE_QWEN35_STRICT_MODEL=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8
export FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-fp8-reference-process-20260930
export FERRULE_QWEN35_STRICT_REPORT_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-full40-maintained-rerun
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FERRULE_CUDA_ARCH=sm_86
```

`--ignored` AND `FERRULE_QWEN35_STRICT_RUN=1` are required. Each command selects
one exact test rather than a GPU test matrix. Verify `running 1 test` and
`1 passed`; a zero-matching-test invocation is never acceptance.
Devices are visible ordinals: PP-major TP roots first, stage EP children second.
Repeated ordinals in the TP1 mappings reproduce the owner's colocated placement;
all contexts still enter admission. Insufficient RAM/VRAM fails, never skips.

```sh
# PP2 TP1 EP2: six owners, four physical GPUs.
FERRULE_QWEN35_STRICT_DEVICES=0,2,0,1,2,3 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::segment_tests::full40_strict::pp2_tp1_ep2_full40_f32_strict -- --exact --ignored --nocapture --test-threads=1

# PP2 TP1 EP4: ten owners, eight physical GPUs.
FERRULE_QWEN35_STRICT_DEVICES=0,4,0,1,2,3,4,5,6,7 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::segment_tests::full40_strict::pp2_tp1_ep4_full40_f32_strict -- --exact --ignored --nocapture --test-threads=1

# PP1 TP2 EP2: four owners/GPUs.
FERRULE_QWEN35_STRICT_DEVICES=0,1,2,3 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::tensor_tests::full40_strict::pp1_tp2_ep2_full40_f32_strict -- --exact --ignored --nocapture --test-threads=1

# PP2 TP2 EP2: eight owners/GPUs.
FERRULE_QWEN35_STRICT_DEVICES=0,1,2,3,4,5,6,7 cargo test -p ferrule-composition --features cuda --lib model_factory::parallel::tensor_tests::full40_strict::pp2_tp2_ep2_full40_f32_strict -- --exact --ignored --nocapture --test-threads=1
```

Reports are named `pp2-tp1-ep2-full40-f32.json`, `pp2-tp1-ep4-full40-f32.json`,
`pp1-tp2-ep2-full40-f32.json`, and `pp2-tp2-ep2-full40-f32.json`. Files are created
exclusively before device preparation: existing reports are never overwritten.
Only a completed run writes `passed`; startup/assertion failures can leave an
empty reserved file and are not acceptance. Nonfinite/tolerance violations are
retained in the report and fail the test after normal shutdown.

## Composition mount migration

The PP/EP and TP/EP production factories were retired in favor of shared
`model_factory/parallel/{plan,budget,experts,assembly}.rs`. The exact commands
above use `parallel::segment_tests` and `parallel::tensor_tests`; the four test
names, reference IDs, comparator/oracle logic and tolerances are unchanged.
The responsibility-named files `../parallel_segment.rs` and
`../parallel_tensor.rs` retain those module IDs; the tensor fixture is
`../parallel_tensor_fixture.rs`. Renaming these files and their `#[path]` mounts
does not rename any test selector. Historical pinned GPU acceptance remains
proof for its recorded pre-rename source/binary, not a GPU rerun of the renamed
tree; hidden compilation and exact test listing validate the new mounts.
The shared `start_admitted` path still performs the same host preparation and
persistent owner startup after the harness's explicit live admission.

## No-GPU validation

Keep GPUs hidden and do **not** use `--ignored`. Use command deadlines up to
1,800,000 ms for CUDA compilation.

```sh
cargo clippy -p ferrule-composition --lib --tests --no-deps -- -D warnings
cargo test -p ferrule-composition --lib full40_strict
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_86 cargo check -p ferrule-composition --features cuda --tests
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_86 cargo clippy -p ferrule-composition --features cuda --lib --tests --no-deps -- -D warnings
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_86 cargo test -p ferrule-composition --features cuda --lib full40_strict
```

For a fixed-binary run, build with `cargo test --no-run --message-format=json`,
extract the emitted library-test executable (do not guess a `target/debug/deps`
path), copy it to an exclusive validation directory, record its SHA256, and verify
each exact test ID with that binary's `--list --exact`. Rebuild after policy or
other compiled-source changes; a stable earlier binary is not current-tree proof.
Pin the F32 manifest, both report JSON files and both tensor files externally.
The F32 reader itself checks tensor hashes against the manifest, not an independent
manifest hash. Run through the authenticated repository subreaper and retain full
stdout/stderr until descendant drain is proven. Subreaper drain is not GPU
quiescence proof; GPU-owner pre/post checks remain mandatory.

The last command runs comparator/topology checks and lists the four GPU tests as
ignored; it compiles them but performs no GPU acceptance. The test helpers also
cover threshold, shape, nonfinite and ordered-route mismatch detection.
