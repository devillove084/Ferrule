<p align="center">
  <img src="docs/assets/ferrule-logo.svg" alt="Ferrule" width="480" />
</p>

<p align="center">
  <strong>A Rust-native runtime for exact, resource-aware LLM inference.</strong>
</p>

<p align="center">
  Ferrule makes execution transactions, paged KV state, expert residency,
  asynchronous I/O, and kernel selection explicit parts of one runtime.
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

Large sparse models are constrained by more than kernel latency. Exact
inference also requires deterministic transaction ownership, bounded KV state,
expert placement, storage scheduling, numerical fidelity, and failure-safe
cancellation. Ferrule treats these concerns as one runtime instead of a set of
independent caches and callbacks.

Ferrule currently focuses on exact DeepSeek-V4 Flash inference with its native
proposal attachment:

- **Exact execution transactions** — proposal, verification, publication, and
  rollback share explicit identities and physical terminalization rules.
- **Runtime-owned materialization** — a global `LoadRegistry` coordinates
  exact-key deduplication, typed generations, targeted wakeups, cancellation,
  and retirement.
- **Asynchronous expert streaming** — direct reads, registered pinned staging,
  CUDA upload events, stable slot generations, and publish-time residency.
- **Hard resource admission** — I/O entries, slabs, bytes, frames, leases,
  arenas, KV state, continuations, and ready cohorts are bounded before use.
- **Paged KV transactions** — reservation, fork, copy-on-write, prefix commit,
  rollback, and retirement are independent from packed execution slots.
- **Provider-neutral execution** — semantic plans are separated from CUDA and
  CUTLASS implementation selection.
- **Auditable correctness and performance** — BF16 writeback boundaries, stage
  counters, resource high-water marks, and uncovered critical-path time are
  explicit evidence.

## Architecture

<p align="center">
  <img src="docs/assets/ferrule-arch.svg" alt="Ferrule architecture" width="100%" />
</p>

One Rust owner controls requests, scheduling, transactions, materialization,
expert residency, paged KV state, CUDA resources, and retirement. Model code
resolves exact semantic stages. Native providers receive same-checkout POD
descriptors and temporarily borrow Rust-owned streams, pointers, and workspace.
No layer may publish readiness without the exact identity, generation, and
resource custody required by the layer above it.

See [Architecture](docs/ARCHITECTURE.md) for the ownership, scheduling,
provider, serving, and cleanup contracts.

## Support matrix

| Area | Current scope |
|---|---|
| Model profile | DeepSeek-V4 Flash 0731 with its proposal attachment |
| Execution | Native Rust runtime with packed exact verification |
| Expert I/O | Direct asynchronous reads into registered pinned staging, followed by event-gated CUDA placement |
| KV cache | Runtime-owned paged reservations, COW forks, prefix commit, rollback, and retirement |
| CUDA | Runtime target detection; local validation on `sm_86`; compile-only validation for `sm_103` |
| Kernels | Provider-neutral semantic plans with portable CUDA and pinned CUTLASS implementations |
| Serving | OpenAI-compatible HTTP and SSE with deterministic greedy decoding |
| Other accelerators | Not yet validated |

Support and numerical evidence are hardware-profile specific. In particular,
successful `sm_86` tests and an `sm_103` cross-build must not be presented as a
B300 full-model parity result.

## Quick start

### Prerequisites

- the Rust toolchain pinned by `rust-toolchain.toml`;
- CUDA Toolkit and a supported NVIDIA driver;
- `just` for repository workflows;
- a local DeepSeek-V4 Flash checkpoint for model commands.

Ferrule uses standard Cargo plus NVCC. The repository pins CUTLASS v4.6.1 at
commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`.

Inspect the CUDA environment, prepare CUTLASS, and build for the detected GPU:

```bash
just cuda-info
just cutlass-setup
just build-cuda
```

Compile an explicit target without requiring matching local hardware:

```bash
just check-cuda-arch sm_103
```

Run an interactive greedy session:

```bash
just dsv4-chat
```

Run the resident driver benchmark:

```bash
just dsv4-runtime-driver-bench
```

Ferrule never downloads the model checkpoint implicitly.

## Serving and benchmarking

Start the OpenAI-compatible server:

```bash
just dsv4-serve
```

In another terminal, run a smoke workload or concurrency sweep:

```bash
just dsv4-vllm-bench smoke
just dsv4-vllm-bench sweep
```

The current DeepSeek-V4 serving path is greedy only. Unsupported sampling
options are rejected rather than silently approximated. Benchmark artifacts
are evidence inputs, not release claims: report committed tokens, failures,
acceptance, physical I/O, resource high-water marks, and uncovered critical
path together. The complete reporting contract is in
[Throughput](docs/THROUGHPUT.md).

## Validation

Platform-independent checks:

```bash
FERRULE_NO_CUDA=1 just lint
FERRULE_NO_CUDA=1 just test
just deny
```

CUDA checks on the detected GPU and compile-only validation for another target:

```bash
just test-cuda-required
just check-cuda-arch sm_103
```

Local checkpoint tests are intentionally separate from small hermetic tests.
See [Computation](docs/COMPUTATION.md) for numerical boundary tests and their
limits.

## Documentation

| Document | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Runtime ownership, transactions, materialization, scheduling, providers, serving, and cleanup |
| [Computation](docs/COMPUTATION.md) | DeepSeek-V4 numerical precision, indexing, quantization, and parity contracts |
| [Throughput](docs/THROUGHPUT.md) | Metric definitions, rooflines, I/O budgets, overlap, and reproducibility |
| [Roadmap](docs/ROADMAP.md) | Unfinished work only |

## Project status

Ferrule is not release-ready. Deterministic CPU tests, RTX 3090 CUDA tests, and
`sm_103` compile validation cover important contracts, but full-model B300 token
parity and multi-request acceptance remain open. No production-readiness or
state-of-the-art throughput claim is made from partial or cross-device evidence.

## Contributing

Changes should preserve explicit ownership and fail-closed validation:

1. add deterministic tests for numerical boundaries and state transitions;
2. keep hardware-specific behavior behind provider capabilities;
3. report targets, models, and measurements as separate quantities;
4. include the narrowest relevant test command and hardware assumptions;
5. never weaken identity, generation, credit, precision, or retirement checks
   to improve a benchmark.

Run `cargo fmt --all -- --check`, the relevant package tests, and the applicable
CUDA lane before opening a change.

## License

Ferrule is licensed under the [Apache License 2.0](LICENSE).
