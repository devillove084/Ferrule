# CPU reference feature boundary

`cpu-reference` is an explicit, non-default testing feature. It does not
activate CUDA. Enable it on the package whose tests you run; for workspace
validation, `--features ferrule-cli/cpu-reference` forwards through CLI and
server to runtime, model and backend. No self-dependency or dev-dependency
turns it on automatically. `cuda-test-support = ["cuda"]` remains independent.

## CLI compatibility

`chat`, `serve` and `bench-interactive` select CUDA when `--backend` is omitted
or `auto`. They never silently select CPU by model family. Explicit `cpu`,
`cpu-reference` and `reference` are accepted only in a `cpu-reference` build.
`bench-interactive` now accepts the same `--backend` override. Production
inference builds should use `cargo build --locked -p ferrule-cli --features cuda`.
A default build still supports host-only tools, not a CPU inference fallback.
Library `BackendSelection` parsing, variants and family-dependent Auto remain
compatible in this first migration batch.

## Test-owned APIs

- `model/src/moe/executor.rs` retains `ExpertExecutor`. Its reference
  implementation and `reference_linear` now live in
  `model/tests/support/expert_reference.rs` (under `crates/ferrule-model`).
- `runtime/src/io/testing.rs` moved to
  `runtime/tests/support/materialization.rs`.
- Existing public reference paths, `io::testing`, `with_testing_resources`
  and `PhysicalResourceBroker::testing_default` are available only under
  `cfg(test)` or `cpu-reference`. Integration tests link a normal library, so
  they need the feature; `cfg(test)` alone is insufficient.
- Model `cpu_reference_api`, runtime `io_registry`/`io_physical`, CLI
  `serve`/`bench_kv_trace` targets require `cpu-reference`. The two existing
  GPU CLI serve tests moved to `serve_cuda`, requiring only `cuda`; shared
  fixtures live in CLI `tests/support/serve_fixture.rs`. Tests were not deleted.
- Process-child build helpers forward the caller's reference feature; the
  general process helper also forwards CUDA. Network linear fake children
  remain deliberately CPU-only.

## Runtime host metadata and runner adapter

The canonical host-memory probe and its counter/ancestry tests live in
`crates/ferrule-backend/src/resources/host_memory.rs`. Application error mapping
is in `crates/ferrule-composition/src/model_factory/host_memory.rs`; its test
checks the model-source error boundary without duplicating the OS probe.
The executor and its HostKV dispatch have a single implementation under
`crates/ferrule-runtime/src/engine/runner_executor.rs` and
`crates/ferrule-runtime/src/engine/runner_executor/kv_dispatch.rs`.
`RunnerMultiSessionExecutor` is the canonical public type;
`NativeMultiSessionExecutor` remains a warning-free compatibility export of that
same type.

Policy, profiles, source/checkpoint metadata and host admission tests do not
require a CUDA context. Generic HostKV dispatch registration is backend-neutral;
concrete CUDA runner installation and execution still require `cuda`. Compiling
metadata does not enable CPU reference execution or a fallback provider.

## Validation lanes

```
cargo check --locked --workspace --all-targets
cargo test --locked --workspace --doc
cargo check --locked --workspace --all-targets --features ferrule-cli/cpu-reference
cargo test --locked -p ferrule-runtime --features cpu-reference --test io_registry --test io_physical
cargo test --locked -p ferrule-model --features cpu-reference --test cpu_reference_api --test architecture_boundaries
cargo test --locked -p ferrule-cli --features cpu-reference --bin ferrule --test backend_policy --test serve --test bench_kv_trace
```

CI runs `just ci-cpu` and `just ci-cpu-reference` separately, with distinct
JUnit reports. Coverage accumulates both lanes; clippy checks both. CUDA
compile-only checks both CUDA-only and CUDA+reference without executing tests.
Do not use `--all-features` for a CPU-only lane. `required-features` skips a
missing-feature target; it never activates the feature. Keep the reference
lane required so those tests are not silently lost. Branch protection must
recognize `CPU tests (default)` and `CPU tests (reference)`; YAML alone cannot
change the remote required-check configuration.

## Deliberately not removed

This batch isolates public test helpers and production CLI policy, not the
whole low-level CPU backend. `NativeCpuProvider`, native CPU compilation,
standard/hybrid CPU runners and library factories remain for a later typed
host/reference split. In particular CUDA streaming attention consumes CPU
partials, CUDA KV transactions use shared CPU-module descriptors, and GPU EP
construction calls `GenericDecoderOptions::standard_cpu` for common metadata.
Do not gate these whole modules by name. Tokenization, routing, scheduling,
host KV, transport, owner journals and all contract tests remain intact.
No synthetic harness was deleted: no proven redundant call graph justified it.
