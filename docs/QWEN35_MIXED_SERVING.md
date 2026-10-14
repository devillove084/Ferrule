# Experimental single-owner Mixed BF16 serving

`ferrule serve --model-profile qwen35-mixed-bf16` explicitly selects the
existing `ResidentModelPlanner::prepare_numeric_fp8_bf16_serving_with_precision`
factory with `ModelPrecision::Bf16`, `NumericStoragePolicy::E4M3FnBf16Scales`
and `NumericFp8Precision::Bf16ModelMixedF32Accumulate`. It uses the ordinary
resident model worker, scheduler, transactions, cancellation and shutdown.
There is no alternate inference loop or CPU fallback.

## Precision and quality boundary

- Exact full-depth text-only Qwen3.5-35B-A3B FP8 checkpoint; E4M3FN weights and
  BF16 scales. No BF16 checkpoint substitution, vision or MTP execution.
- BF16 activation/residual/device logits, KV and convolution state;
  F32 accumulation and recurrent SSM. Host logits expand to F32.
- Mixed role-dependent arithmetic is not legacy `w8a16-bf16`, native FP8 W8A8,
  or a claim of Marlin/HF bitwise equivalence.
- The latest pinned [prepared-attention result](../target/validation/bf16-attention-model-current-20261007/FINAL.json)
  passes **804 original full40 metrics / 1,986,560 logits** and bounded fresh
  QA/arithmetic/Chinese quality: **27 logits metrics / 9,187,840 elements**,
  within 5,427 total metrics. Independent free-greedy, QA fork/reset and a
  separate cold-owner QA replay passed against unchanged references/bounds.
  The earlier attention6 failure is superseded for this evaluation, not erased.
  Logits are not bit-exact; arithmetic completion is incomplete within the fixed
  token budget. Prepared attention requires uniform H/Hkv/D and context <=1024.
- The earlier [public serve receipt](../target/validation/mixed-public-serve-owner-20261007/admission-deadline-fix/FINAL.json)
  records **Hello lifecycle PASS**, not fresh-quality or performance acceptance.
  Its binary predates the latest attention changes. These separate receipts do
  **not** establish public-serving acceptance for the latest quality-tested source.
  Full source/binary scope and remaining gates live in [current acceptance](CURRENT_ACCEPTANCE.md).
  Neither result establishes universal quality, distributed BF16 or fair vLLM performance.

## Historical pinned handoff and bounded owner smoke

The following paths/hash describe the original build-only handoff, not a newly
validated current binary. The later Hello lifecycle receipt above records its
own binary/plan identities. Rebuild and prepare a fresh handoff after source
changes; do not reuse these historical plans as current acceptance.

The original Cargo artifact resolved from `compiler-artifact.executable` is:

`target/validation/qwen35-35b-current/toolchain/cargo-cuda124/debug/ferrule`

The preserved copy is:

`target/validation/mixed-public-serve-ready-20261007/ferrule-cuda124-mixed`

SHA256: `9d8eb64f78b0b38f0a54837c2e29247bea6de50a17ea5d85969a1006510bd812`.
This is Cargo's optimized **dev** profile, not an assumed release binary.
`build-final-report.json`, `build-final-supervisor.json`, and
`build-environment.json` in that directory record the resolved executable,
source snapshots, verified child drain and literal CUDA12.4/sm86 environment.
CUDA feature check/build, loader resolution and `serve --help` passed with GPUs
hidden in that build-only step. The separate, later owner receipt above records
actual public-serving lifecycle execution; this build receipt alone does not.

The dedicated test entry [mixed_serve_smoke.py](../scripts/tests/mixed_serve_smoke.py)
reuses the existing authenticated owned subreaper, HTTP/SSE parser and
readiness/graceful-shutdown helpers. It does not change benchmark semantics.
Its [configuration](../crates/ferrule-cli/tests/support/mixed_serve_smoke.json)
is also parsed by a CPU Rust test through the actual CLI/factory dispatch.
The configuration fixes 7 GiB, context128/batch8/sequences2, lazy experts,
resident BF16 KV, F32 SSM, CUDA0 and startup cuBLAS `:0:0`.

CPU-only recheck (does not launch a server, even if GPUs are visible):

```sh
CUDA_VISIBLE_DEVICES='' target/vendor/vllm028-py312/bin/python -B \
  scripts/tests/mixed_serve_smoke.py run \
  --plan target/validation/mixed-public-serve-ready-20261007/serve-smoke-plan.json
```

**Historical invocation template, GPU owner only.** After rebuilding, use the
newly prepared plan and a fresh output path, with explicit authorization for this
limited lifecycle smoke (the paths below belong to the old handoff):

```sh
CUDA_VISIBLE_DEVICES='' target/vendor/vllm028-py312/bin/python -B \
  scripts/tests/mixed_serve_smoke.py run \
  --plan target/validation/mixed-public-serve-ready-20261007/serve-smoke-plan.json \
  --output target/validation/mixed-public-serve-ready-20261007/owner-smoke \
  --execute --sole-owner-permission
```

The parent stays GPU-hidden. Only the owned server receives the plan's GPU0,
CUDA12.4 library path and `CUBLAS_WORKSPACE_CONFIG=:0:0` environment.
Without both execution flags, no server is launched. The output must be new.

Each round has bounded startup (180s), requests (120s each), cancellation
reclamation (30s) and graceful teardown (45s), under a 720s outer custodian.
The two rounds perform:

1. Real public `serve` startup; `/health`, `/readyz`, `/v1/models` and Mixed
   effective-profile log checks.
2. Raw **Hello**, greedy, two output tokens; validate real SSE usage,
   `finish_reason=length` and `[DONE]` with the existing parser.
3. Raw Hello with a four-token ceiling; close the response after first content,
   require the matching `client_disconnect` trace and zero admission leases.
   A completed request racing disconnect is **not** cancellation success.
4. Another Hello/two-token completion verifies post-cancellation usability.
5. SIGTERM to the unreaped owned root; require normal exit, released port,
   authenticated subreaper identity/launch ID and kernel ECHILD before restart.
6. One fresh-process/model restart repeats the lifecycle on the same port.
   OS/page cache is not flushed or claimed cold.

Startup, actual text/usage, errors, teardown and bounded server-log tails are
printed as well as saved. Helper loss, timeout, unknown descendants or failed
work stops further launches; no process-name/PID-group cleanup or foreign PID
signalling is permitted. Process drain is **not** a GPU-quiescence measurement.
There are no GPU queries, vLLM launches, installs or numeric-reference changes.

**Concurrent attention edits invalidate the source pin. Rebuild and reprepare
before GPU execution; do not edit the plan hashes to bypass this gate.** The
plan pins the binary, source tree, workflow, checkpoint/tokenizer metadata,
Python executable and resolved CUDA runtime/cuBLAS/driver library files. An
unmodified plan has no power to establish fresh quality acceptance.

To prepare a new handoff, retain the owner-selected literal environment in
`build-environment.json`; collect a successful bounded Cargo CUDA build's JSON
artifact into `build-final-report.json` and its authenticated supervisor receipt
into `build-final-supervisor.json`, with before/after source snapshots. Copy the
resolved executable as `ferrule-cuda124-mixed`, then run:

```sh
CUDA_VISIBLE_DEVICES='' target/vendor/vllm028-py312/bin/python -B \
  scripts/tests/mixed_serve_smoke.py prepare \
  --directory target/validation/mixed-public-serve-ready-20261007
```

The shown directory already has its plan; preparation refuses overwriting it.
Use a fresh owner-selected handoff directory after rebuilding.

`CUBLAS_WORKSPACE_CONFIG` must be **literally `:0:0` at process startup**.
Missing, empty and other values fail before model-file access; Ferrule never
changes the environment after startup. Composition routing checks it too.

The owner always uses visible device 0; select the physical card through
`CUDA_VISIBLE_DEVICES`, not `--devices`. Supported topology is PP/TP/EP=1,
thread/default rank policy, engine auto/resident, resident KV, lazy experts.
Full prewarm, prewarm workers, explicit host/pinned cache/hotset overrides,
HostKV, pipeline/process/rank placement, timeout/restart overrides are rejected.
`--expert-prewarm lazy` is optional confirmation of the intrinsic policy.
`--model-precision` and `--numeric-fp8-precision` conflict with this profile.
Omitting the profile preserves all previous defaults and CPU-reference behavior.

## Budget contract

| CLI input | Existing factory capacity |
|---|---|
| `--cuda-expert-device-entries` / `--cuda-expert-device-bytes` | Global expert entries and cache bytes, including reserved scratch |
| `--cuda-expert-scratch-mb` | Scratch portion inside expert cache ceiling |
| `--cuda-workspace-mb` | Operator workspace grant, excluding expert scratch |
| `--cuda-device-budget-mb` | Total admitted device ceiling, including unchanged allocator margin |
| `--kv-cache-mb` | Physical KV-plane budget |
| `--ctx-size` / `--max-active-sequences` / `--max-batch-tokens` | Context, sequence/state copies, packed rows |

The three new MiB overrides require `--model-profile`; absent overrides use
existing factory limits, **not hardcoded 7 GiB defaults**. The smoke configuration retains
512 MiB allocator margin and explicitly supplies the 7 GiB test ceiling.
This is not a physical VRAM reservation, total host-memory limit, or vLLM
utilization equivalent. Source-header admission includes state, workspace,
weights, typed KV and margin; live free-memory admission stays on the owner.
Undersized budgets fail rather than shrinking scratch/state or falling back.

Mixed requires explicit `--max-batch-tokens <= min(32, ctx-size)` (the legacy
512 default is not silently reduced for this profile). Existing metadata
validation enforces nonzero context/sequence capacities, full depth, strict
context maximum and tensor-read limits. Prefill chunks retain the existing
reported clamp to the admitted batch. Prefix caching and proposals remain off.
The 128-position smoke configuration cannot serve a 128-input + 32-output workload unchanged.

## CPU-only regression commands

```sh
CUDA_VISIBLE_DEVICES='' cargo test --offline -p ferrule-composition \
  --no-default-features --lib mixed_serving
CUDA_VISIBLE_DEVICES='' cargo test --offline -p ferrule-cli \
  --no-default-features --bin ferrule model_precision_tests
CUDA_VISIBLE_DEVICES='' cargo test --offline -p ferrule-cli \
  --features cpu-reference --bin ferrule model_precision_tests
```

Tests use sparse synthetic source headers (never weight payloads or CUDA),
compare dispatch with the existing explicit factory, freeze effective budgets,
reject unsupported policy/underbudget cases and check child-local startup
variables without unsafe process-global environment mutation.
