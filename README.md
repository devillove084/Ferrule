<p align="center">
  <img src="docs/assets/ferrule-logo.svg" alt="Ferrule" width="480" />
</p>

<p align="center">
  <strong>A Rust-native runtime for exact, resource-aware LLM inference.</strong>
</p>

<p align="center">
  Ferrule makes execution transactions, paged KV state, expert residency, asynchronous I/O,
  and kernel selection explicit parts of one inference runtime.
</p>

<p align="center">
  <a href="https://github.com/devillove084/Ferrule/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/devillove084/Ferrule/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/devillove084/Ferrule"><img alt="Coverage" src="https://codecov.io/gh/devillove084/Ferrule/graph/badge.svg?branch=main" /></a>
  <img alt="Rust" src="https://img.shields.io/badge/Rust-2024-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-CUTLASS-22c55e?style=flat-square" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2563eb?style=flat-square" />
</p>

<p align="center">
  <a href="#why-ferrule">Why Ferrule</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#serving-and-benchmarking">Serving</a> ·
  <a href="docs/ARCHITECTURE.md">Documentation</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

---

## Why Ferrule

Modern inference systems are constrained by more than kernel latency. Large sparse models also
need deterministic transaction ownership, bounded KV state, expert placement, storage scheduling,
and failure-safe cancellation. Ferrule treats those concerns as one runtime rather than a set of
independent caches and callbacks.

Ferrule currently focuses on exact DeepSeek-V4 Flash inference with a checkpoint-native proposal attachment and provides:

- **Exact execution transactions** — packed proposal verification, per-sequence acceptance,
  correction or bonus staging, and commit/rollback share explicit identities.
- **Runtime-owned materialization** — a global `LoadRegistry` coordinates cross-cohort
  single-flight, typed generations, targeted wakeups, cancellation, and retirement.
- **Asynchronous expert streaming** — `O_DIRECT`/`io_uring` reads, registered pinned slabs,
  CUDA upload events, stable slot generations, and publish-time residency.
- **Hard resource admission** — queue entries, slabs, read/upload bytes, frames, leases, arenas,
  KV state, continuations, and ready cohorts are bounded before dispatch.
- **Paged KV transactions** — reservation, fork, copy-on-write, prefix commit, rollback, and
  retirement are validated independently from packed execution slots.
- **Kernel-provider boundary** — semantic execution plans are separated from CUDA/CUTLASS
  provider selection and launch descriptors.
- **Auditable performance** — stage counters, resource high-water marks, acceptance accounting,
  and uncovered critical-path time are first-class output.

Ferrule is under active systems validation. The current CUDA-profile concurrency acceptance gate is still open;
see [Roadmap](docs/ROADMAP.md) for the exact status and [Throughput methodology](docs/THROUGHPUT.md)
for the difference between targets, roofline models, and measured results.

## Architecture

<p align="center">
  <img src="docs/assets/ferrule-arch.svg" alt="Ferrule target architecture" width="100%" />
</p>

The runtime owns scheduling, transactions, global expert loading, hard credits, waiter indices,
and retirement. Model code owns exact artifact, router, and residency semantics behind a
model-neutral execution boundary. Providers own hardware-specific allocation and launch details
solely through versioned POD command/completion. No layer may publish readiness without the
identities and resource custody required by the layer above it.

For the full ownership model, see [Architecture](docs/ARCHITECTURE.md). Provider contracts are
specified in [Kernel Provider ABI](docs/KERNEL_PROVIDER_ABI.md) and [CUTLASS integration](docs/CUTLASS.md).

## Support matrix

| Area | Current status |
|---|---|
| Model profile | DeepSeek-V4 Flash with its proposal attachment is the active end-to-end validation target |
| Execution | Native Rust resident runtime with packed exact verification |
| Expert I/O | Direct `io_uring` reads into registered pinned slabs, then CUDA event-gated upload |
| KV cache | Runtime-owned paged reservations, COW forks, prefix commit, rollback, retirement |
| CUDA profile | Current CUDA profile with an explicitly detected and validated target; concurrency acceptance remains open |
| Kernels | Semantic provider registry with CUDA/CUTLASS implementations |
| Serving | OpenAI-compatible HTTP and SSE path with official vLLM benchmark integration |
| Other accelerators | Not yet validated; results and capabilities are hardware-profile specific |

Do not extrapolate current CUDA-profile measurements or kernel availability to another device. A backend is
supported only after its provider capabilities, correctness suite, and end-to-end evidence pass.

## Quick start

### Prerequisites

- Rust toolchain pinned by `rust-toolchain.toml`;
- CUDA toolkit and a supported NVIDIA environment;
- `cargo-oxide` for the current CUDA build path;
- `just` for repository workflows;
- a local DeepSeek-V4 Flash checkpoint and proposal attachment for model commands.

Inspect the local CUDA setup:

```bash
just cuda-info
just oxide-doctor
```

Build the current CUDA profile with its detected target:

```bash
just build-cuda sm_121a
```

Run an interactive session:

```bash
just chat models/DeepSeek-V4-Flash-0731
```

Run the resident driver benchmark:

```bash
just dsv4-runtime-driver-bench
```

The model checkpoint is large and is never downloaded implicitly by Ferrule.

## Serving and benchmarking

Start the OpenAI-compatible server:

```bash
just dsv4-serve
```

In another terminal, run the serving smoke workload:

```bash
just dsv4-vllm-bench smoke
```

Or run the configured concurrency sweep:

```bash
just dsv4-vllm-bench sweep
```

Benchmark artifacts are evidence inputs, not release claims. Report externally committed output
tokens, successful and failed requests, acceptance, physical I/O, resource high-water marks, and
uncovered critical-path time together. The complete measurement contract and equations live in
[docs/THROUGHPUT.md](docs/THROUGHPUT.md); serving details live in
[docs/serving.md](docs/serving.md).

## Validation

Platform-independent checks:

```bash
just check
just test-runtime
just test-model
just test-server
just test-cli
```

CUDA-required checks:

```bash
just test-cuda-required
just test-cutlass-provider sm_121a
```

The default `just test` workflow also requires the repository's configured CUDA environment.
Local model tests may require the official checkpoint and are intentionally separate from small,
hermetic unit tests.

## Documentation

| Document | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Runtime ownership, transactions, I/O, wake, and cleanup invariants |
| [Throughput methodology](docs/THROUGHPUT.md) | Metric definitions, rooflines, I/O budget, counters, and reproducibility |
| [Roadmap](docs/ROADMAP.md) | Current acceptance state, verified evidence, and remaining gates |
| [Serving](docs/serving.md) | OpenAI-compatible server and official benchmark workflow |
| [Kernel Provider ABI](docs/KERNEL_PROVIDER_ABI.md) | Provider registry, capability, and launch-descriptor contracts |
| [CUTLASS](docs/CUTLASS.md) | CUTLASS provider build and validation notes |
| [Expert memory architecture](docs/expert-memory-architecture.md) | Expert artifact and residency design |
| [Scheduler architecture](docs/scheduler-architecture.md) | Scheduling and resource admission model |

## Project status

Ferrule is not yet release-ready. The typed I/O and scheduler architecture is implemented and has
broad deterministic coverage, but the current real-CUDA multi-request path still requires the
continuous `n8 → c1/c2/c4` acceptance chain documented in the roadmap. No SOTA or production
throughput claim is made from partial, historical, or single-request evidence.

## Contributing

Changes should preserve explicit ownership and fail-closed validation:

1. add deterministic tests for every new state transition and cleanup branch;
2. keep hardware-specific behavior behind provider capabilities;
3. report targets, models, and measurements as separate quantities;
4. include the narrowest relevant test command and any required hardware assumptions;
5. do not weaken identity, generation, credit, or retirement checks to improve a benchmark.

Run `just fmt`, the relevant package tests, and the applicable CUDA lane before opening a change.

## License

Ferrule is licensed under the [Apache License 2.0](LICENSE).
