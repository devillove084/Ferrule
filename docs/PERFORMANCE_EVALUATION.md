# Performance evaluation protocol

<!-- markdownlint-disable MD013 MD060 -->

## 1. Status, question, and deliverable

**Maintained status entry:** [Current acceptance](CURRENT_ACCEPTANCE.md) separates
latest Mixed quality, post-refactor F32 parallel acceptance and earlier public
Hello lifecycle receipts. Those are not end-to-end performance results. A fair
current Ferrule/vLLM comparison remains **not completed**; the historical actual
vLLM rows disclose precision/topology mismatches. Operator timings, correctness
run wall times and serving smoke cannot establish matched throughput or speedup.
The dated evidence sections below retain their original scope and failures.

This protocol complements [THROUGHPUT.md](THROUGHPUT.md). It defines frozen
comparison controls, request/failure accounting, independent quality gates,
supervisor custody and statistical interpretation. Historical measurements
below retain their original source/binary scope; they are not current-checkout
acceptance. A slower, blocked or infeasible configuration remains a result.

### 1.1 Historical aligned evidence snapshot through v4

The archived sequence is [baseline](../target/validation/perf-alignment-current/ferrule-baseline/REPORT.md),
[v1](../target/validation/perf-alignment-current/ferrule-optimized-v1/REPORT.md),
[v2](../target/validation/perf-alignment-current/ferrule-optimized-v2/REPORT.md),
[v3](../target/validation/perf-alignment-current/ferrule-optimized-v3/REPORT.md),
and [v4](../target/validation/perf-alignment-current/ferrule-optimized-v4/REPORT.md).
All rows use the same 15 measured requests / 480 actual completion tokens:
`Hello`, input usage 1, max 32, `c=1`, context 2048, active 4, warmup 2,
repeat 3, requests 5 per repeat, fixed-work, no errors/EOS-shortening.

| Run | process EP8 tok/s | thread EP8 tok/s | Reported change |
|---|---:|---:|---:|
| Initial | 2.587837 | 5.589628 | reference |
| v1 | 2.605017 | 5.690816 | +0.664% / +1.810% |
| v2 | 3.248592 | 10.035088 | v1→v2: +24.705% / +76.338% |
| v3 | 5.619765 | 10.188503 | v2→v3: +72.991% / +1.529% |
| v4 | **5.658288** | **10.107201** | v3→v4: **+0.685% / -0.798%** |

These are fixed-work measurements, not significance tests; no independent
confidence intervals or statistical significance claim is made. v3→v4 is a small
change near measurement noise: process `+0.685%`, thread `-0.798%`. v3 thread
TTFT p95 had already regressed about 1.25% (`118.135→119.615 ms`).
The client reports TTFT to the first nonempty SSE content event and a TPOT proxy
from first to last content event divided by `N-1`; raw SSE gaps can contain multiple
tokens and are not token ITL.

The vLLM reference is **83.817152 tok/s** on eight GPUs, but it is `vLLM 0.28.0+cu129`,
TP4/DP2/EP8, Marlin W8A16/BF16—not the Ferrule TP1/EP8 F32 fused-M1 /
TF32x3 multirow profile. Ferrule therefore is not an equal precision/topology
speed comparison: the vLLM/v4 rate ratios remain 14.813× (process) and 8.293× (thread).
The vLLM artifact is [`result.json`](../target/validation/vllm028-py312/jit-cuda12/result.json)
with its [environment README](../target/validation/vllm028-py312/jit-cuda12/README.md).

Full40 v4 process-EP8 passed on 1,986,560 logits at `atol=rtol=2e-4`,
`max_abs=4.386901855e-5`, tolerance ratio `0.163209953`; each owner held 1,280
experts, upload/eviction/residency deltas during requests were zero, and all
owned subprocesses were reaped with GPUs idle after normal shutdown. Ten targeted GPU tests
and one selector CPU test also passed. This is numerical/custody evidence, not a
benchmark significance claim, a multi-host result, or proof of host-KV live DMA.

### 1.2 Runtime attribution, provenance and limits

| Change group | Observed runtime evidence | Attribution limit |
|---|---|---|
| v1 IPC ownership, scratch/prefix reuse, completion notification | Root nanosleep disappears; root memset calls 36,789→28,869; formal process/thread rates improve only 0.664%/1.810% | Waiting moves to futex; three changes were not independently ablated |
| v2 fused M1 and numeric consumer scope | 39,930 dequant launches replaced by fused M1; total launches 113,619→73,689; owner EventSync 105,600→31,680; real consumer proof 8→2 | Proof hook counts are correctness evidence, not a microbenchmark-derived serving gain |
| v3 root scope/boundary and process validation/notification path | Historical v3 profile: root EventSync 43,762→35,842, proof 7→1; process +72.991%, thread +1.529% over v2 | Boot-cache validation counts are CPU-unit evidence, not production per-request instrumentation; no independent four-change ablation |
| v4 GDN direct-copy guard | `cfg(test)` direct-copy count 0 versus general 45; 4,950 per-request count is `5×30×33` workload derivation | Derived from validated row counts, not a production trace or time saving; do not promise TP/BF16 improvement |

v1–v3 used separate short production profiles plus unprofiled HTTP runs; v4 ran
no Nsight profile. The v4 copy-count assertion uses rows 4,1,3,1 and verifies direct
0 versus general 45 copies; 4,950 per request is a workload derivation, not measured
production copies or saved time. KeepAll proof/sync 0/2 and bounded 1/3 remain intact.
These results are not sums of microbenchmark benefits. Cross-thread/device inclusive waits cannot be added
as wall time. Nsight retained `NumTpcs` metadata errors (`trace_import_valid=false`)
and BPF permissions blocked exact wake counts; use matched activity counts only,
not occupancy/TPC claims. Earlier optional-profile cleanup problems remain in v1/v2
reports; v3 normal test/bench/profile shutdown was verified, including empty owner
children, released port, zero GPU used bytes and no compute processes.

v3 source manifest SHA-256 (including untracked source):
`16a4903daec42f89ebf89cd2cd113af4c51abbd02b61098aa33dfc627c37a409`;
v3 CLI: `a579df454a2df76b4694c8f22f2e673bc2a161c81059ec5a1a80694d979e04f0`.
v4 full fingerprints, read from the report-linked
[result](../target/validation/perf-alignment-current/ferrule-optimized-v4/result.json) and
[build manifest](../target/validation/perf-alignment-current/ferrule-optimized-v4/build-manifest.json):
source `d00fc888d4427577fefffd7e43fd8f856cea5e4787c6188a82fdca68640678a9`;
CLI `a0ed88cfd883ae7f9ba667447643b93f8a878728b341cc7a39e0372c47ec7405`.
The v4 build/run source did not drift, and older artifacts were unchanged.
Shared client SHA-256:
`a9aac55c7151d22f0d3cb53844faeb7f09c09eb59ae0407c7f22c6c7d02f6257`.
These identify archived runs, not later worktrees/binaries. This docs-only update
occurs after the frozen manifest; it no longer identifies the whole edited worktree.

The v4 report includes selected CPU/GPU and strict-oracle checks, not an
unfiltered full-suite result. Its first strict attempt stopped before numerical
comparison because NVML/local PID namespaces did not match; the rerun passed.
The aborted run remains evidence, not a pass. Cross-namespace mapping remains
unvalidated; only identity-verified owned processes may be cleaned up.

All benchmark rows use startup/warmup-excluded summed fixed-work window time;
TTFT and TPOT proxy p95 are pooled request quantiles, linearly interpolated at
`(count-1)*p`, not independent-run confidence intervals. The initial report's
thread TPOT p95 `177.8 ms` was content-event ITL; corrected TPOT proxy p95 is
`174.395 ms`. v4 TTFT mean/p95: process `187.259/195.636 ms`, thread
`115.645/118.678 ms`; TPOT proxy mean/p95: `171.722/173.487 ms` and
`96.145/96.778 ms`. Three windows share one server per configuration; they are not
independent cold restarts. vLLM's eight-GPU measured requests all reached 32;
one warmup ended at 25 and is excluded. Its four-GPU row had 456 actual tokens
and is not the matched 480-token comparison.

Outstanding at the v4 snapshot: full long-prompt/concurrency matrix, independent cold-restart/randomized
pairs, confidence intervals/statistical significance, matched precision/topology,
per-change ablations, untested hardware and network/multi-host validation. Rates
remain far from vLLM, not “approaching parity.” Small v4 changes are close to noise;
further measurement should decide larger changes, not promises that TP or BF16
must be faster. Resource safety and precision remain candidate paper mechanisms,
not new novelty claims. Artifacts in ignored `target/`
need checksum-preserving archival/versioning before publication; document links
are a local historical snapshot, not a portable or future-worktree validation.

Research leads come from `temp_ferrule_refactor.md` (11700–13047,
13063–13856), `temp_ferrule_impl.md` (82640–82780, 84285–84615,
88600–88724, 89654–90610), and `temp_kv_set.md` (1709–2068,
2154–2238). These are private historical leads, not public artifacts or verified
citations. Before claiming novelty, verify primary sources for serving/offload,
expert caching, admission, cancellation, and asynchronous ownership; compare
properties and assumptions, not framework names. Single-flight, leases, LRU,
transactions, and Rust are not themselves contributions.

### 1.3 Historical baseline audit and current comparability boundary

The [prepared baseline audit](../target/validation/perf-alignment-current/bench-stage/prepared/exact-baseline.json)
and [2026-10-03 candidate audit](../target/validation/fair-vllm-audit-20261003/unavailable.json)
are CPU/text-only evidence, not server startup, memory-fit or quality acceptance.
The archived vLLM 0.28.0+cu129 deployment used TP4/DP2/EP8 and Marlin/BF16;
Ferrule TP1/EP8 F32 and W8A16 retain F32 model/KV/state/output boundaries.
The installed FP8 quantizer admits FP16/BF16 model activations, not F32. Its EP
degree is derived from TP×DP×PCP; PP does not multiply per-stage EP. There is no
independent `--expert-parallel-size` selector making TP1/DP1/PCP1 into EP8.

Equal PP/TP/EP degrees alone do not establish equal logical shards or resources.
Ferrule separate TP roots and expert owners may represent the same logical
topology as shared vLLM workers, but canonical tensor/expert assignments, physical
placement and admitted budgets must be checked. Owner count alone is neither
proof of equivalence nor a reason to reject an otherwise matched topology.

The old candidate drafts remain blocked historical artifacts, not launch
recipes. Current F32 parallel support and single-owner Mixed bounded PASS are
listed in [current acceptance](CURRENT_ACCEPTANCE.md); neither makes the old
BF16-boundary audit current or establishes a fair vLLM pair. W8A16 strict FAIL
remains independent. Prepare fresh owner-reviewed profiles with the retained
[matrix CLI](#reproducible-long-contextconcurrencycold-restart-matrix-runner),
not historical helper-generated argv or edited acceptance hashes.

### 1.4 Current-owner matched-client results — 2026-10-02

The [current-owner report](../target/validation/qwen35-35b-current/current-owner/REPORT.md)
measures the optimized artifact against the
[prior CUDA12.4 artifact](../target/validation/qwen35-35b-current/gpu-owner-final-cuda124/REPORT.md).
The [matched-client audit](../target/validation/qwen35-35b-current/current-owner/matched-comparison-audit.json)
records `server_settings_equal`, `client_settings_equal` and `client_hash_equal`
as true for all four profiles. It records 15 requests and 480 output tokens
per current profile; the report confirms all prior rows also used 480 tokens.
The audit's text-hash field is `not exposed`, not proof of output identity or
numerical equivalence.

All rows below use eight RTX 3090 GPUs, CUDA12.4, EP8/TP1/PP1, full owned-expert
prewarm, context 2048, active/cohort 4, batch/prefill chunk 32, KV 1024 MiB and
host expert cache 40960 MiB/10240 entries. The literal prompt is `Hello`, actual
input usage 1, greedy output cap 32, concurrency 1, seed 0, warmup 2 per window,
three windows × five measured requests, request timeout 120 s and fixed-work
safety limit 600 s. Every current row completed **15/15 requests, 480 actual
output tokens**, all finishing at length, with normal owned shutdown. Rates
exclude startup and warmup and divide actual successful usage output tokens by
summed measured-window wall time.

| Profile | Prior artifact tok/s | Current optimized tok/s | Descriptive current/prior ratio |
| --- | ---: | ---: | ---: |
| F32 process | 5.585670 | **5.684965** | 1.0178× |
| F32 thread | 10.386380 | **10.351618** | 0.9967× |
| W8A16 process | 2.544555 | **5.647071** | 2.2193× |
| W8A16 thread | 3.153882 | **9.966111** | 3.1600× |

The W8A16 before/after observation is therefore about **2.545→5.647 tok/s**
(process) and **3.154→9.966 tok/s** (thread), alongside current F32
**5.685/10.352 tok/s**. Preserve the slower prior W8A16 result; it is not an
invalid measurement erased by the optimization. Three windows on one server
are not independent randomized cold restarts. These small samples establish
neither statistical significance nor stable parity with F32, long-context
scaling, or full40 W8A16 quality.

**Matched-client provenance.** The table uses the historical client SHA-256
`e3faaffe49dad7b04cb8a74bb711e89d51cea2b29da4ef97031ab956223820ca`
on both artifacts; current results are in
[`hello32-matched-client/`](../target/validation/qwen35-35b-current/current-owner/hello32-matched-client/).
The current-client supplemental runs used
`ea65f94072d543085c71a1f1a557e9ee154bfc61d3f838f22bb87695329701e6`
and also completed 15/15 each, but are **not** substituted into this table:
the client includes later checkpoint/reporting changes. The tested source
manifest is `700678b9377301f45006376472f86dde55535d32d2e3cd28750f216626b61f6f`,
and CLI is `1f610139c90ee89dbc913571b1d53635619595f371e2ab6c3ef3c7b2669376fb`.
The owner built an immutable 784-file snapshot; these hashes do not identify
a later working tree. Model metadata was recorded but weight shards were not
rehashed. See the [build manifest](../target/validation/qwen35-35b-current/current-owner/build-manifest.json)
and [source drift](../target/validation/qwen35-35b-current/current-owner/source-drift.json).

**Kernel versus serving evidence.** The
[kernel optimization report](../target/validation/qwen35-35b-current/w8a16-kernel-opt/REPORT.md)
selects the dedicated four-warp BF16-RNE/F32-boundary vector path for `M<=8`
and the cooperative-loader BF16 WMMA path for `M>8`, without decoded-weight
materialization or an F32 fallback. Its CUDA-event graph microbenchmark reports
all 140 W8A16 cases faster, minimum observed ratio 1.82×, using three warmups
and seven measured graph replays of 100 launches. Those kernel timings exclude
uploads/setup and cannot be extrapolated to serving tok/s. W8A16 M1 is still
slower than the same-run F32 M1 companion in the reported example. Memcheck and
racecheck were clean in the backend owner's run; they were not rerun in the
current serving pass. Rejected variants remain archived. Nsight Compute was
blocked by driver/tool incompatibility, so there is no stall-counter claim.
The kernel report's then-pending model dispatch expectation was subsequently
updated by the current owner and checked at the M8/M9 boundary.

**Current scoped validation, not blanket acceptance.** The current owner reports
backend W8A16 12/12, model precision 4/4, CUDA root/workspace 19/19, root
integration 11/11, tiny W8A16/Torch 1/1, and current HostKV live fixture 2/2
passing. The repaired root import and current root/workspace retest supersede
the earlier compile blocker only for the identified current snapshot; retain
the prior failed attempt. The full40 **F32** process-EP8 oracle passed
1,986,560 logits with maximum absolute error `4.38690185546875e-5`, no failures
or nonfinite values. **At that owner snapshot**, an independent precision-matched
full40 W8A16 oracle was missing; tiny/kernel numerical tests and successful
serving did not supply one. That status is historical: Section 1.5 records the
subsequent independent oracle and failed W8A16 strict comparison. The prior
tiny-run `cuInit` failure/retry and prior staged 8k timeout remain
in their original reports. Current HostKV results cover the named live fixture,
not all serving/host-offload compositions.

**Dense process PP2 is now an actual smoke.** The
[dense 0.8B result](../target/validation/qwen35-35b-current/current-owner/dense-pp2/result.json)
used `/mnt/nas1/hf/Qwen3.5-0.8B` on GPUs 0/1, process PP2/TP1/EP1, BF16
checkpoint storage and logged F32 arithmetic. Readiness took 10.2749 s; one
non-streaming `Hello` request took 1.1734 s and returned input usage 1/output
usage 8, finish `length`. Exit 0, released port 18356 and idle GPUs were
recorded. This is a live end-to-end smoke, **not a dense performance benchmark
or quality oracle**, and it does not enable process profiles in the separate
matrix supervisor.

**Retained limits.** No 8k rerun occurred in this owner pass; the interrupted
trial remains incomplete. The current subreaper contract supersedes the old
process-profile harness blocker, not the historical GPU evidence. Host-only
reporting, lint and custody checks do not validate a rebuilt numerical artifact.

The historical vLLM 83.817152 tok/s row remains a **system-level deployment
reference**, not a matched before/after or equal-arithmetic baseline. vLLM is
TP4/DP2/EP8 with BF16 model activation/KV; both Ferrule profiles remain TP1
with F32 model activation/KV/state/output, even when numeric operands use BF16.
An exact same-precision/same-topology installed-vLLM baseline remains
**unavailable**. Neither matched-client settings nor optimized W8A16 kernels
remove that distinction.

### 1.5 Independent full40 W8A16 oracle exists; strict acceptance failed

The subsequent independent
[summary](../target/validation/w8a16-full40-independent-20261002/SUMMARY.json)
and [quality audit](../target/validation/w8a16-full40-independent-20261002/quality.json)
replace **missing oracle** with **`STRICT_ACCEPTANCE_FAILED`**. The accepted
reference generation is `qwen35-fp8-reference-full40-w8a16-bf16-rne-v2`, manifest
SHA-256 `0bafdc340c54712e3db14a07398a16377cbe4194ae2fb5de6ef4f69b3b97233b`,
using native CPU Torch 2.13.0+cu129, audited Transformers 5.2.0 and tokenizers
0.22.2. The provisional v1 generation with a dependency-check override is
excluded. The arithmetic contract is BF16-RNE numeric weights/inputs with
original multiplicative scales and **F32 model/KV/state boundaries**; it is
neither an all-BF16 oracle nor the separate F32 reference profile.

Scope: **single-GPU full40**, `Hello` and capital prompts, all prefill logits
and one cached decode per case—eight vocabulary rows total. This is not an EP8,
serving, long-context or general language-quality acceptance run. At unchanged
`atol=rtol=2e-4`, the matching W8A16 comparison reports:

| Check | Result |
| --- | --- |
| Logits outside strict tolerance | **1,837,493 / 1,986,560** |
| Maximum absolute logit error | **0.2965252995491028** |
| Ordered route-ID mismatches | **4**, capital prefill row 1, layers 13/20/29/30 |
| Routing weights and convolution/recurrent state | Strict failures retained in `quality.json` |
| Argmax agreement | **8/8 rows**, not quality acceptance |
| Minimum top-10 overlap | 9/10, diagnostic only |
| Maximum KL(reference→actual) / total variation | 0.000823801449 / 0.012990306963, diagnostic only |

Both full40 W8A16 tests failed (CUDA test command exit 101); the quality audit
returned exit 1. The same evidence records a passing tiny W8A16 test and two
passing F32 regressions, which do **not** cancel the full40 failures. No approved
distribution criterion replaces strict allclose/route/state acceptance, and
neither preserved argmax nor small distribution differences prove harmlessness.
There is **no tolerance relaxation or post-hoc threshold**. Payload provenance
records 36,080 shared payloads unchanged against the fixed F32 reference;
unused experts are not certified, and `source_freeze_claim` is false. Preserve
its own source/binary hashes rather than relabeling the older serving snapshot
as this validation input.

The [first-divergence diagnosis](../target/validation/w8a16-first-divergence-20261002/FIRST_DIVERGENCE.json)
retains strict failure while supporting an **early numerical mechanism**:
CUDA/Torch F32 reduction-order and norm differences are amplified when later
numeric activations cross discontinuous BF16 rounding bins. It inspects both
prompts' first four prefill layers and same-input native GDN for the first three
GDN layers—not every layer and not the first cached-decode divergence.

- Hello first differs bitwise at layer-0 QKV, initially within tolerance; the
  first strict native boundary is layer-2 shared-expert gate projection.
- Capital first differs bitwise at input RMSNorm, but this changes **no** BF16
  input bits for the first projection. Its QKV also differs on the same rounded
  operands due to reduction order; norm alone does not explain the drift.
  The first strict native boundary is layer-1 QKV.
- Same-input projection checks (135 Hello, 441 capital), ten norm checks and
  fifteen GDN checks per case had no outside-tolerance elements in this early
  scope. CPU replay of the documented vector reduction reproduces selected
  captured outputs bitwise. That replay is a diagnostic, **not a replacement
  independent acceptance oracle**. Tracing retained all 654 original tensors
  bitwise per case while adding 182 trace tensors.
- The policy audit found no evidenced scale/profile mismatch requiring a local
  production fix; no production fix was applied. This does **not** establish
  that the entire model is defect-free or all later divergence is harmless.
  The full40 logit/route/state failures remain authoritative.

W8A16 remains an **explicit opt-in** (`--numeric-fp8-precision w8a16-bf16`),
not a validated replacement for default F32. The Section 1.4 timings remain
scoped observations, not quality-qualified W8A16 performance acceptance.
The exact same-precision/topology vLLM baseline is still unavailable: the
independent CPU oracle does not change the TP1/F32-boundary versus
TP4/DP2/BF16 deployment distinction or permit a fair-vLLM claim.

### 1.6 Historical polling backoff result

The [p128/o32/c1 F32 process-EP8 report](../target/validation/qwen35-35b-current/polling-ipc-p128-owner/REPORT.md)
compared ordinary optimized binaries with IPC instrumentation disabled:
query/yield baseline **2.469859 tok/s**, backoff candidate **2.265026 tok/s**
(-8.29%). Each variant had **one server start and three measured requests**.
This descriptive non-improvement is not a general regression, confidence claim
or causal attribution. Matching response text/usage is not a numerical oracle;
query counts were unobserved and CPU totals were lifetime observations.

The default sleep/backoff branch was removed; the current exact-ticket
[query/yield contract](PARALLELISM.md#exact-ticket-synchronous-boundary-wait)
retains completion proof and custody. The experiment does not measure an
observer-cost benefit from frame-aggregated instrumentation or prove a speedup
for a subsequently edited tree. Original reports and binaries remain unchanged.

## 2. Implemented mechanisms and evidence boundaries

| Static source evidence | Experiment it can motivate | Boundary to preserve |
| --- | --- | --- |
| [Qwen strict adapter](../crates/ferrule-composition/src/model_factory/adapters/qwen35/mod.rs) / [shared local budget](../crates/ferrule-composition/src/model_factory/parallel/budget/local.rs) | Full40 single-GPU numeric-FP8, bounded expert cache, host prewarm, physical admission | GPU serving; bounded token-serial prefill; prefix reuse/speculation disabled |
| [Qwen EP binder](../crates/ferrule-composition/src/model_factory/adapters/qwen35/ep.rs) / [shared resident assembly](../crates/ferrule-composition/src/model_factory/parallel/resident.rs) | Resident EP, full owned-expert prewarm and all owned experts GPU-resident | Not generic TP; local-cache overrides rejected; not request-time storage streaming |
| `crates/ferrule-runtime/src/io/registry.rs`, `scheduling/resources.rs`, `engine/driver/resident.rs` | Exact-key operation sharing, held/submitted credits, cancellation versus physical Publish | Trace actual call-path membership; do not assume every dedicated Qwen expert operation traverses the generic registry |
| `crates/ferrule-runtime/src/cache/page_manager.rs`, model decoder transactions | KV reservation/publication/retirement and provisional-state isolation | Logical terminal is not physical quiescence; ledger scope is not all process memory |
| `crates/ferrule-composition/tests/unit/engine/model_factory/qwen35_ep_strict_tests.rs`, `crates/ferrule-composition/tests/unit/engine/model_factory/qwen35_ep_strict_process.rs` | Full-vocabulary, route and recurrent/convolution-state checks | v4 executed the process EP8 full40 oracle; other profiles still require their own acceptance |
| `crates/ferrule-model/tests/expert_parallel/bounded_cuda.rs`, `cuda_materialization_registry.rs`; `crates/ferrule-runtime/tests/io_registry.rs`, `distributed_transactions.rs` | Bounded materialization and protocol-specific controls/faults | Fixture evidence must not be relabeled Qwen HTTP serving or hardware recovery |

The original first comparison is the archived eight-GPU **thread/process EP8**
fixed-work run. Process EP now has a full40 oracle and lifecycle evidence in the
v4 report; this does not make it an equal vLLM topology. Qwen35 TPxEP/PPxEP,
live host-KV DMA/decoder composition, dense process hybrid, adaptive CPU/GPU
attention, multi-host and speculative serving remain outside the **v4** snapshot.
Section 1.4 adds the current HostKV live fixture and real dense 0.8B process-PP2
smoke; neither is a general serving-composition or quality/performance claim.
KVSET/trace and fixed-split streaming components do not establish integrated
serving speed or a new cache algorithm. The CPU reference lane and benchmark
harness are landed first-batch infrastructure, not a claim that CPU serving
backends were removed or that CPU is a performance competitor.

## 3. Frozen comparison contract

Create a versioned manifest **before measured runs**; unresolved fields block
that cell. Do not silently change a profile to make it feasible.

| Control | Required match or explicit disclosure |
| --- | --- |
| Checkpoint | Exact `Qwen3.5-35B-A3B-FP8` text checkpoint, full 40 layers; source revision, config/index/shard hashes and numeric scales. Historical local locator: `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8`; a path alone is not identity. No BF16 model substitution, repacking, requantization, vision or MTP |
| Tokenizer/input | Identical tokenizer assets/revision, chat template bytes/hash, special tokens, thinking/template options, UTF-8 requests and rendered input token IDs; same ordered dataset and shuffle seed |
| Output policy | Greedy `temperature=0`, no sampling/penalties/tools/stop strings/speculation; fixed seed, exactly 32 generated tokens with EOS ignored in the performance corpus; verify the effective request rather than assuming flags were accepted |
| Hardware | Same physical RTX 3090 cards, SM86, nominal 24 GiB/card; 1-card and 8-card strata; UUIDs, usable bytes, GPU0/root placement, PCIe/NUMA/NVLink/P2P topology, power/clocks/thermal state and unrelated load |
| Host/storage | Same CPU/thread affinity, NUMA placement, RAM/cgroup and pinned limits, storage/filesystem/NAS path and mount identity; no hidden extra host, CPU compute, swap or offload budget |
| Arithmetic | Record FP8 format, scale dtype/interpretation, dequant rounding, activation/accumulation/output/KV dtype, TF32 flags, kernel and attention implementations; see precision classes below |
| Runtime | Pin Ferrule source and release binary; vLLM server AND client version/commit, wheel/container hash; Python, PyTorch, Transformers, tokenizer, CUDA toolkit/runtime/driver, CUTLASS, compiler and communication library versions. Record resolved options, not only defaults |
| Serving | Real TCP streaming HTTP for both, same client host/network, endpoint `/v1/chat/completions`, payload contract, client concurrency, context ceiling and arrival policy; no in-process Router time mixed into the table |
| Resources | Explicit per-card and total limits for weights/experts, scratch, KV/recurrent state, allocator reserve and upload headroom; host/pageable/pinned/frame capacities and high-water marks. Distinguish configured logical limits from physical allocations |

The current vLLM environment is isolated `vLLM 0.28.0+cu129`,
Python 3.12, Torch `2.13.0+cu129`, FlashInfer `0.6.16.post3`, process-local
CUDA 12.9.86 JIT toolchain, and driver API `12090` / driver `575.57.08` on RTX 3090 (SM86), as recorded by the artifact loader.
The CUDA 12.9 JIT repair is an isolated toolchain/workspace fix, not a source or
third-party patch. The default CUDA 13 path failed because FlashInfer JIT used
`/usr/local/cuda` 13.2 and cached `sampling.so` required `libcudart.so.13`;
Python 3.10's `array.array[int]` annotation in FlashInfer `comm/fd_exchange.py`
raised `TypeError: 'type' object is not subscriptable` on import, a separate root
cause ([retained trace](../target/validation/flashinfer-py310-audit/import-current.out)). The passing vLLM artifact loads CUDA 12 runtime under the
process-local environment. These facts do not make TP4/DP2/EP8 BF16 equivalent
to Ferrule TP1/EP8 F32 fused-M1/TF32x3 multirow.

### Precision and compatibility classes

Ferrule v4 retains CUDA 12.4, FP8 E4M3FN storage with numeric BF16 block scales:
rows=1 selects `F32FusedM1` (IEEE F32 FMA), multirow selects `F32TiledTf32x3`.
The historical `precision()` label `F32Tf32x3` is not the selected kernel for every
shape. This is neither native FP8 MMA on RTX 3090 nor a bitwise oracle match;
scale names do not imply reciprocals/E8M0. Checkpoint names do not establish
arithmetic equivalence.

- **P-match:** the same checkpoint and arithmetic contract, independently
  quality-gated, with implementation/reduction-order differences disclosed.
  vLLM availability in this class is pending; the measured Marlin/BF16 row is P-different.
- **P-different:** identical checkpoint/workload but vLLM uses BF16/FP16 or
  different dequantization, accumulation, KV or output rounding. Publish in a
  separate deployment table after its own oracle checks. Label **non-equivalent
  precision**; a quality pass does not make arithmetic or compute cost identical.
- **Blocked:** pinned vLLM cannot execute the model/precision on SM86 or within
  the budget. Keep error/capability evidence; speedup is N/A, not zero or infinity.

A CPU/GPU hybrid run with different precision is likewise non-equivalent and
outside the primary GPU table; it needs its own resource and numerical contract.
Do not use CPU oracle latency as serving performance. Do not silently dequantize
to another stored checkpoint to fill a missing matched baseline.

### Single card, eight cards, EP and TP

| Stratum | Ferrule | External baseline | Permitted interpretation |
| --- | --- | --- | --- |
| D1 | One GPU, bounded local expert cache, explicit host residency | Actual vLLM server on the same one GPU; supported offload only if declared and budgeted | Same-resource deployment comparison; capacity failure remains a result |
| D8 | Eight GPUs, thread/process TP1 EP8, full owned-expert residency | Measured vLLM eight-card TP4/DP2/EP8 Marlin W8A16/BF16 | P-different end-to-end strategy comparison, not isolated EP-versus-TP or equal-precision speed |
| D1 versus D8 | Different total VRAM and possibly residency | Compare within each engine only with caveats | Deployment/resource curve, not controlled parallel scaling |

Record TP/EP/PP/DP degree, rank/thread/process placement and collective transport
separately; vLLM TP8 must not be described as Ferrule EP8 equivalent. A secondary
vLLM best-supported strategy may be reported, but use a preregistered equal tuning
budget (at most three candidate configurations per engine/stratum on a separate
pilot corpus), archive all attempts, and freeze before measurement. No eight
single-GPU replicas masquerading as one eight-GPU model group.

Show GPU count, GPU-seconds per successful output token and aggregate tok/s/GPU
alongside raw rates; host cost remains separate. If residency/capacity cannot be
held constant, do not report parallel efficiency. Only a future matched-residency
experiment can use `speedup = Rn/R1`, `efficiency = speedup/n`; never divide by
an OOM, unsupported or missing baseline. More VRAM enabling residency is not
proof of faster parallel compute.

## 4. Workload and run protocol

### Minimal matrix and prerequisites

1. Preflight both actual servers with full model load, one greedy request and
   bounded shutdown; run the strict oracle in Section 6 on the same source,
   precision and placement. An HTTP smoke is not strict validation.
2. Freeze the manifest-driven synthetic/deterministic corpus with **128, 512,
   2048, 4096 and 8192 actual prefill tokens**, output caps **32, 128 and 256**,
   concurrency **1, 2, 4 and 8**. Each declared deployment profile has 60 cells;
   context is **prefill plus output**, up to 8448. Unsupported exact lengths,
   context or active-request capacity remain blocked, never silently truncated.
   The older Hello snapshot and proposed chat/ignore-EOS study are separate
   protocols; this executable matrix keeps raw completions and natural EOS.
3. Each cell uses **5 independent fresh-server repetitions**, each with
   **100 measured requests** and **8 fixed warmup requests**, using the same
   within-repetition prompt order across systems. Prefix cache remains off in
   both in the primary profile. Explicit prefix-enabled controls are separate
   profiles. Use varied prompts rather than a single repeated Hello prompt;
   do not claim route coverage from synthetic text alone.
4. Primary arrival policy is finite closed-loop saturation: issue immediately
   when a client concurrency slot is available. Record actual send times and
   server active-request limits. `request_rate=inf` with a concurrency cap is
   not an open-loop latency/SLO study. Long-context cells are now scheduled by
   the runner, not measured here; fixed-rate Poisson overload and soak studies
   remain separate protocols.

Predeclare startup timeout **900 s**, request deadline **300 s from client send**,
measured-run wall limit **3600 s from first measured send**, and teardown limit
**120 s**. These are experimental limits, not performance targets. A timeout
remains data; any later longer-deadline experiment gets a new protocol ID and
retains the first attempt. Do not repeatedly rerun until success. If a run is
cut short, archive all scheduled-but-unissued requests separately.

### Cold, warmup, repetitions, interleaving

For each repetition: verify prior owners have drained and GPUs are available;
start a new server; timestamp spawn, load, prewarm and Ready; snapshot caches;
send **one separately labeled first-after-Ready request**; execute the 8 warmup
requests; snapshot again; then start the 100-request measured window. Warmup
and first-request tokens do not enter steady-state throughput. A full-prewarmed
EP8 Ready state is already expert-resident: this first request is process-cold,
not device-expert-cold. Stop and record a warmup failure; do not replace it.

Run engines sequentially on the same hardware, never concurrently. Use five
paired blocks, each containing all eligible configurations; generate and archive
a seeded balanced/randomized order of engine, GPU stratum and workload. Use the
same request-order seed within each pair. Publish the schedule and temperatures,
clocks and cache carryover; do not always run Ferrule after vLLM or cold before
warm. Profiling and tuning runs are separate from measured replicas.

Do not clear global page caches or disrupt other jobs implicitly. Cold-storage
experiments require an explicit isolated-host cache-reset procedure and evidence;
otherwise label storage state `uncontrolled` or `warm`, never `cold`. Independent
processes do not reset OS/NAS caches. Startup cost is reported separately, with
an optional lifecycle rate including spawn through proven drain, never silently
folded into or removed from the steady-state denominator.


### Supervisor custody and restart budget (current contract)

The matrix still uses `scripts/test_vllm_qwen35.py:run_process(argv, timeout,
payload)`. Its internal `scripts/_owned_subreaper.py` is a small process custodian,
not another benchmark harness or a server/vendor patch. Each call launches a
fresh, single-thread Linux helper. **Before forking the application root**, the
helper installs a non-reaping SIGCHLD handler (clearing inherited SIG_IGN and
SA_NOCLDWAIT), sets `PR_SET_CHILD_SUBREAPER`, reads it back with
`PR_GET_CHILD_SUBREAPER`, and verifies `waitid` support. The parent checks the
private socket handshake, launch identity, helper PID, and method before sending
launch authorization. Matrix capability probing launches only this helper; every
actual run independently repeats the handshake. Owner JSON cannot supply it.

The root starts in a fresh session, behind a launch pipe until its initial
birth/root credentials have been journaled. A descendant's `setsid`, double fork,
thread-originated fork, or earlier unobserved reparent does not escape the kernel
subreaper ancestry. Nested subreapers are terminated in turn, causing their
children to be adopted. Only the helper performs application-tree reaping; the
calling process and its HTTP/client threads do not compete for these children.

The helper enumerates **its own direct kernel child list**, not arbitrary procfs
PIDs. Before each signal it calls `waitid(P_PID, ..., WNOWAIT | WNOHANG | WEXITED
| __WALL)`, then checks procfs birth ticks and PPID. Only a successfully verified,
unreaped direct child may receive a numeric-PID signal. Because no other waiter
or SIGCHLD handler reaps in that helper, an exit between validation and signaling
leaves a zombie holding the same PID. This is not a check-then-kill recipe for a
threaded generic parent. Names, executable paths, groups, NVML PIDs, old journal
rows and owner attestations never authorize a kill. No `killpg`, `pkill`, cgroup
privilege, pidfd API, global scan, or application modification is required.

Cleanup first stops direct fork sources, then requests TERM/CONT, escalates to
STOP/KILL after a short grace period, and repeatedly adopts/terminates/reaps new
direct children. Reaping uses WNOWAIT observation followed by the single explicit
reap. **Only kernel ECHILD with no unreconciled child records means
`proven_drained`**; an empty procfs list or root exit does not. Cleanup has a
five-second deadline. Permission errors, unkillable/D-state children, unexpected
identity changes, helper failure, truncated evidence, or deadline without ECHILD
remain unknown/error and halt the matrix before the next restart. Parent owner
interruption is rethrown with its original primary error and helper cleanup
receipt attached. Durable partial HTTP/lifecycle/matrix reports remain intact;
`custody.json` adds launch/boot/root/helper/child credentials and reap outcomes.
A journal error does not suppress cleanup; it remains a reporting failure.

**Scope, not sandbox.** This contains ordinary descendants while the helper is
alive, including setsid/doublefork; it does not stop hostile same-UID code from
attacking the helper or tampering with its files, ptrace, privileged namespace
manipulation, resource-exhaustion fork bombs, or launching work through an
external service (systemd/Docker/SSH/MPS daemon, etc.). Such external launchers
are outside the benchmark contract. No claim is made that every historically
short-lived process is individually logged: children reaped by their own parents
need not be adopted. Helper loss yields unknown, never a parent-side hunt for
remaining PIDs. ECHILD proves process-tree termination, **not GPU fences, driver
recovery, device-memory drain, or remote work completion**. Owner GPU/port checks
remain necessary; `gpu_quiescence` stays `unverified`.

Process profiles require a successful real helper handshake, with capability,
workload, foreground-root and exclusive-resource checks retained.
`all_workers_in_process_group` is legacy metadata, not custody proof;
`no_daemonize:true` describes the foreground root, not all descendants.

CPU-only regression commands (real bounded subprocesses and local fake HTTP,
no model/GPU imports):

```sh
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_owned_descendants.py'
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_vllm_supervisor_failures.py'
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_bench_matrix_recovery.py'
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_*.py'
./target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests -p 'test_*.py'
```

Plan a small explicit subset before scheduling traffic. Review the emitted
supervisor/cleanup budgets, including capability probing and helper reserves;
bounded userspace waits do not guarantee progress through a stalled kernel or
filesystem. The default full grid is not an instruction to launch a long sweep.

Use `matrix-corpus` with pinned local tokenizer assets and `matrix-plan` with an
owner-authored profile list as documented below. Planning is offline; only
`matrix-run --execute` sends traffic, after real custody and owner checks.
Never overwrite old plans or change their hashes to bypass source/binary drift.
The historical 8k interruption remains failed/partial, not a long-context PASS.

#### Stage progression and preregistered statistical interpretation

1. Keep the historical 8k interruption and its adjudication unchanged. First run
   CPU custody fixtures on the actual host; absent containment is a blocked
   prerequisite, not permission to try one long GPU run.
2. Once containment and owner checks exist, approve one randomized feasibility
   stage at a time: smoke, concurrency-2, context-2k, context-8k. Each stage uses
   two independent restarts per approved profile, two measured requests and two
   warmups. Use the same seeded block across profiles; never concatenate
   independently run single-profile files and call that paired randomization.
3. Freeze a new confirmatory protocol before inspecting confirmatory outcomes:
   all 60 required cells/profile, five restart blocks, 100 measured requests,
   eight warmups, default limits, immutable profiles, and no success-only retries.
   Chunk its *execution* into owner-approved bounded groups without changing the
   preregistered block/order or deleting blocked cells. The runner has no resume
   claim; safe chunk/resume orchestration is still a prerequisite for a study
   of this size, not a reason to launch the full grid automatically.
4. Existing CIs are seeded percentile bootstrap over restart-level values
   (10,000 resamples), not individual requests. Matched paired differences
   require all scheduled pairs, length/workload/quality/precision/topology gates,
   and complete process runs; missing/failed/unknown slots suppress throughput
   CIs. Scheduled success fractions retain failures and unknown-request budgets.
   Two-run feasibility CIs are exploratory; five blocks are a protocol minimum,
   not a power or statistical-significance guarantee.
5. The current CIs are pointwise, not simultaneous over 60 cells or all profile
   pairs. Preregister primary cells/contrasts, an effect-size target, and a
   multiple-comparison/power plan before making significance claims. Do not pool
   within-server windows as restarts, treat process-cold as storage-cold, or
   promote unmatched F32/W8A16/vLLM rows into a fair speedup. No such statistical
   acceptance is claimed by this CPU repair.

The isolated helper, direct-child checks and credential fsyncs add supervisor
CPU/storage work. Archive both supervisor/helper hashes and this changed measurement
contract; do not silently attribute old/new-client differences to GPU kernels.

### Reproducible long-context/concurrency/cold-restart matrix runner

The shared client now has an opt-in, manifest-driven matrix lane. The legacy
`ferrule.http_bench.v1` Hello invocation and report shape remain unchanged; the
matrix report is `ferrule.http_matrix.v1` and references unmodified v1 child HTTP
reports rather than changing old artifacts. This lane is an execution protocol,
not measured evidence: no GPU, server or driver command is run while
creating a manifest or in plan mode.

The full canonical grid is below. `matrix-plan --cell p128-o32-c1` (repeatable)
selects a frozen subset; `--profile-id` selects one profile. `--limits-json`
can shorten, but not enlarge, protocol deadlines. Run mode cannot override
these choices. `matrix-run --execute` now refuses more than ten server starts
by default; an owner must explicitly set `--max-server-runs` for a larger
schedule. Planning never launches the full grid automatically.

The full canonical cells are exactly:

- prefill lengths `128, 512, 2048, 4096, 8192`;
- output caps `32, 128, 256`;
- client concurrency `1, 2, 4, 8`;
- context length is `prefill + output`, so output is included in the context
  ceiling (for example, prefill 8192/output 256 has context 8448);
- five independent fresh-server process runs per cell by default, 100 scheduled
  requests and 8 warmups per process run.

A corpus is an owner-exported JSON object containing tokenizer `revision`,
asset `files_sha256`, encoding/template flags, and literal UTF-8 prompts with
both exact `input_ids` and `input_tokens`. The runner does not retokenize a
prompt, infer a count from text, or silently truncate. The optional local export
uses only an already-installed `tokenizers` module; it never installs or
updates a package:

The tokenizer directory and revision below are explicit owner inputs, not
a guessed model revision; replace them with the exact locally pinned identity.
Create the artifact parent directory before writing a new corpus/manifest.

```sh
mkdir -p target/validation/http-matrix
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py matrix-corpus \
  --tokenizer-dir /path/to/pinned/tokenizer-assets \
  --revision OWNER_PINNED_REVISION \
  --output target/validation/http-matrix/corpus.json
```

A profile file is a JSON list. Each profile supplies an `id`, `engine`, exact
loopback `/v1/completions` URL, served model ID, absolute `argv` (no shell
expansion; `{context_tokens}` and `{concurrency}` are replaced), `ownership`
attestations, and complete operator metadata. Metadata must declare tokenizer,
checkpoint, precision, topology, engine capability, cache, resources and the
stateful quality gate. The precision contract includes weights/activation/
accumulation/KV/output dtypes, numeric operand and state-boundary dtypes,
rounding and scale profile. Admitted resources are compared as a normalized
physical GPU/budget contract. Logical topology compares DP/TP/EP/PP/PCP,
canonical groups, PP layer ranges, tensor slices and expert assignments.
Owner layout, worker count and expert transport remain independently audited.
Capability and cache policy are explicit. Unknown or unavailable fields block
comparability; they do not default to a match.

**Owner profile input (`profiles.json`).** Do not copy historical launch flags
without checking the current pinned binary. No server argv, GPU allocation or
quality PASS is invented by this client. Supply a list of profile objects with:

| Field | Required shape / meaning |
| --- | --- |
| `id`, `engine`, `model` | Unique alphanumeric/hyphen/underscore ID; `ferrule` or `vllm`; actual served model ID |
| `url` | `http://127.0.0.1:PORT/v1/completions`, reserved unused port, no auth/query/redirect |
| `argv` | JSON string array beginning with the absolute executable path; existing owner-reviewed launcher may be used, but no daemonization. Only `{context_tokens}` and `{concurrency}` are substituted; the owner must map these to real server flags and archive resolved settings |
| `ownership` | `exclusive_resources:true`, `no_daemonize:true` for a foreground server root. Descendant sessions/forks are allowed only under the verified subreaper. `all_workers_in_process_group` is retained legacy metadata, not authority. External launch services are outside the contract |
| `metadata.tokenizer` | Exact copy of `corpus.json`'s tokenizer object; includes `revision`, asset `files_sha256`, and `encoding` (`add_special_tokens:false`, raw `template:none`, thinking disabled for the built-in exporter) |
| `metadata.model` | `checkpoint_sha256` identifying the complete pinned checkpoint closure; not a directory path alone |
| `metadata.precision` | `weights_dtype`, `activation_dtype`, `accumulation_dtype`, `kv_dtype`, `output_dtype`, `numeric_operands_dtype`, `state_boundary_dtype`, `rounding` (include actual RNE/dequant and shape-dependent arithmetic, not just checkpoint storage), `scale_profile`, `admitted_resources` |
| `precision.scale_profile` | At least `dtype`, `interpretation`, `block`; include scale asset hashes and kernel/TF32 policy as additional fields. Nested unknowns are not matches |
| `precision.admitted_resources` | `gpu_uuids`, `per_gpu_limit_bytes`, `host_limit_bytes`, `pinned_limit_bytes`; explicit equal actual admitted budgets, not only requested CLI limits |
| `metadata.topology` | Positive integer `dp`, `tp`, `ep`, `pp`, `pcp`, `model_worker_count`; `rank_layout` (`separate-tp-ep` or `shared-tp-dp-pcp`); `expert_policy` containing `residency`, `partition`, `placement`, `rank_backend`, `transport` |
| `topology.canonical` | `schema:"ferrule.logical_topology.v1"`, `num_layers`, `moe_layers`, `logical_groups`, `pp_layer_ranges`, `tensor_shard_map`, `expert_shard_map`; optional `device_permutation_policy` defaults to `"exact_canonical_slot"` |
| `topology.physical_placement` | One row per engine owner: `owner`, `pp`, `role`, `lane`, `cuda_ordinal`, `physical_ordinal`, `gpu_uuid`; UUIDs must identify the actual device, not owner-local numbering |
| `metadata.engine_capability` | `max_context_tokens`, `max_concurrency` (positive integers), `streaming:true`, `raw_completions:true`, `eos_policy:"permitted"`, `prefix_cache` (explicit boolean); smaller/unknown capacity produces blocked cells, never silent truncation |
| `metadata.cache` | `prefix_mode:"off"` or `"first-cold-then-warm"`, `os_cache`, `state_reset:"per_request"`; the latter prefix mode also requires `fresh_process_empty_prefix:true`, supported prefix cache, and enough warmups to prime every prompt. `os_cache:"uncontrolled"` is honest but blocks fair comparison |
| `metadata.quality_gate` | `status:"pass"`, `reference_sha256`, `coverage_sha256`, and `logits`, `routes`, `recurrent_state`, `convolution_state` all `true`; link real external oracle evidence. Do not mark unsupported/stateful checks passed to unlock a label |

`state_boundary_dtype` must describe the actual boundary contract (including
convolution/recurrent state), not merely say "mixed"; use `unknown` until
verified. `numeric_operands_dtype` is not a substitute for activation/KV dtype.
`model_worker_count` counts model execution owners, not unique GPUs or HTTP
coordinators. The shared layout checks EP1 (disabled) or EP=TP×DP×PCP and
workers=PP×TP×DP×PCP. The separate layout requires DP1/PCP1 and
workers=PP×(TP+EP) for EP>1. Unknown layouts fail closed.

**Canonical comparison schema (CPU-only normalization, 2026-10-03).**
`comparison_contract` returns independent `same_logical_topology` (also exposed
as the compatibility alias `sametopology`), `owner_organization_match`, and
`physical_resources_match`. Different owner organizations alone do not block
`contract_match`. Unknown/invalid owner declarations still block it. The
engine-specific `launch_contract` remains strict about exact flags, owner
counts/roles/order and visible-device placement; normalization does not rewrite
or relax a launch contract.

The minimal v1 schema uses zero-based indices and half-open intervals:

| Canonical field | Row shape and validation |
| --- | --- |
| `num_layers`, `moe_layers` | Positive model layer count; nonempty unique list of routed-MoE layer indices |
| `logical_groups` | `{kind, pp, dp, pcp, members}`; `kind` is `tensor` or `expert`, and `members` lists lane IDs. Tensor groups enumerate each PP/DP/PCP combination with TP members; expert groups enumerate each PP stage with EP members and `dp=pcp=0` (the flattened EP group, not a DP worker coordinate) |
| `pp_layer_ranges` | `{pp,start,end}`; exactly one nonempty range per stage, covering `[0,num_layers)` without gaps/overlap |
| `tensor_shard_map` | `{pp,dp,pcp,lane,layer,tensor,axis,start,end,size}`; canonical tensor name and numeric axis; one contiguous slice per TP lane for each declared tensor, covering `[0,size)` without gaps/overlap; every layer/group needs tensor evidence |
| `expert_shard_map` | `{pp,lane,layer,experts,total_experts}`; explicit expert IDs for every EP lane of every declared MoE layer, covering `[0,total_experts)` once; an empty lane is allowed, missing lanes are not |

Tensor names must be translated to the same model-level names, not compared
using engine-specific packed parameter names. v1 represents partitioned tensor
slices, not arbitrary packing/replication transforms. Unsupported mappings stay
unknown pending a reviewed adapter. The supplied inventory is a declaration:
structural validation cannot prove that it lists every real model tensor or
matches actual runtime materialization. PP sizes and expert IDs are not inferred
from degrees, partition-policy names or worker counts. The owner must supply
reviewed model-layer, MoE-layer and shard-map evidence; two identical unknown
declarations never establish a logical match.

The normalizer adapts separate TP/EP owners and shared TP/DP/PCP workers to
`shard_to_device` rows keyed by `(kind,pp,dp,pcp,lane)`, joined to tensor/expert
rows by their group and lane. Each row contains physical ordinal and UUID, not
owner ID or local CUDA ordinal. It separately emits `unique_gpu_count`, unique
`gpus` with `limit_bytes`, host/pinned limits and expert residency. Repeated
colocated owners count once toward GPU budgets. `gpu_uuids` and
`per_gpu_limit_bytes` are parallel arrays with one positive byte budget per
unique GPU; host/pinned budgets are nonnegative integer bytes. Duplicate UUIDs,
inconsistent ordinal/UUID identities, and unbudgeted or unused admitted GPUs
fail closed. Budget-array order is irrelevant after pairing limits with UUIDs.

**Device permutation policy:** omission means `exact_canonical_slot`; this is
the only supported v1 policy. Corresponding logical slots must map to the same
physical ordinal and UUID. List ordering and consistent local CUDA renumbering
are normalized, but swapping physical devices between logical slots is a
physical-resource mismatch, even with the same GPU set and logical shards. No
implicit graph isomorphism, UUID renaming or budget-equivalent GPU substitution
is allowed; an unsupported policy blocks comparability.

For PP1/PP2 TP2 EP2 DP1 PCP1, Ferrule's four/eight separate colocated owners
and vLLM's two/four shared workers can therefore have
`same_logical_topology=true`, `physical_resources_match=true`, and
`owner_organization_match=false`. Different PP boundaries, tensor slices or
expert assignments instead produce a logical mismatch. Extra/noncolocated GPUs
or different budgets produce a physical/resource mismatch. Neither label can
override precision/rounding, KV/GDN/model boundary, runtime-evidence or stateful
quality-gate failures. Existing real candidates remain blocked; synthetic CPU
fixtures are not runtime, memory-fit, quality or performance evidence.

Older profiles missing canonical fields still load for separately labeled
planning, but cannot earn `same_logical_topology`, `sametopology` or `fairmatch`.
Precision and owner labels remain independently reportable. Runtime evidence
is additionally mandatory in `comparison_eligibility`; declaration-only
`contract_match` is not a fair-comparability verdict.

CPU validation uses the existing audited oracle overlay (no installation,
GPU access or service launch):

```sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=target/validation/w8a16-full40-independent-20261002/python-overlay:scripts
./target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests -p 'test_*.py'
```

Oracle version requirements and acceptance gates remain independent of the
HTTP client; an unavailable reference environment is not a passing check.

Keep the existing `source`, `binary`, `dependencies`, `resources` and oracle
artifact provenance alongside these fields. Precision/topology unknowns permit
a separately labeled deployment run, but never a fair comparison. Missing
capability, inconsistent tokenizer, unsupported prefix/state policy or missing
ownership attestations block launch. Additional strata (single/eight-card,
thread/process, prefix-off/prefix-enabled) are separate explicitly named
profiles, not silently mixed samples. The small exported corpus is synthetic,
not representative of all routes/tasks, and exporter text/token IDs must be
checked against each server's actual tokenizer/encoding contract. HTTP usage
can validate lengths but cannot independently attest the remote token IDs.

Create and review a small seeded feasibility schedule without traffic, using
the owner-authored `profiles.json` schema above. These paths are new output
examples, not permission to reuse or overwrite archived artifacts:

```sh
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py matrix-plan \
  --corpus target/validation/http-matrix/corpus.json \
  --profiles target/validation/http-matrix/profiles.json \
  --cell p128-o32-c1 --seed 20261001 --process-runs 2 --requests 2 --warmup 2 \
  --output target/validation/http-matrix/manifest.json
```

This example has four fresh-server runs for two profiles; it is a feasibility
check, not confirmatory statistics. Omitting the subset and using default counts
would produce 60 cells/profile and 600 starts for two profiles. The schedule
shuffles cells and profile order with the recorded seed. `matrix-run` without
`--execute` is also plan-only and cannot start a server:

```sh
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py matrix-run \
  --manifest target/validation/http-matrix/manifest.json \
  --output target/validation/http-matrix/plan-review
```

Only the server owner may later opt in to inference traffic with `--execute`.
The runner then reuses the existing `test_vllm_qwen35.py` bounded
`run_process` supervisor. Every process run starts one owner command in a fresh
process group, waits for a bounded HTTP Ready phase, sends a separately labeled
first-after-Ready request, runs warmup, then measures the fixed-work window.
Startup, first request, warmup, measure, and shutdown are separate lifecycle
fields. The command is never run here; the following full-grid form explicitly
allows 600 starts (two profiles × 60 cells × five restarts). Prefer the staged
commands above; do not run this large experiment automatically:

```sh
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py matrix-run \
  --manifest target/validation/http-matrix/manifest.json \
  --execute --max-server-runs 600 --output target/validation/http-matrix/run
```

The supervisor signals only identity-verified owned descendants; a process
group is not treated as containment. It does not use a PID namespace as authority to kill
foreign processes, does not kill an existing listener (the reserved loopback
port is checked before spawn), and does not use `killall`, global process
matching, or global page-cache/drop-caches operations. A shutdown timeout or
unreaped child halts subsequent schedule entries; the remaining entries stay
`not_run` with their fixed scheduled-request budget. GPU quiescence remains
`unverified` unless the owner supplies separate evidence.

Prefix policy is process-local: `off` or an explicitly declared
`first-cold-then-warm` profile; off is required when prefix cache is unsupported. The latter records first-after-Ready separately
and requires an owner declaration of fresh-process empty-prefix state plus explicit priming; it never
clears global caches or silently removes a cold prefix from the report. OS/NAS
cache state is `warm` or `uncontrolled`, never relabeled `cold` by process
restart.

Every offered request remains in the elapsed-time/request denominators. Startup failure/OOM has
zero offered requests and an unavailable throughput, while measured timeout,
HTTP error, protocol error, early EOS and partial output remain recorded. Early EOS is protocol success,
but not length-matched success; it blocks a fixed-length fair comparison. Scheduled, offered, not-issued, success, failure classes and lifecycle
statuses are retained. `offered_known`, `not_issued_known` and
`unobserved_request_budget` distinguish known counts from a missing terminal
artifact. Scheduled-request success is an observed lower bound when a hard
kill loses a terminal report, not an assertion that no requests ran. Failure,
timeout, OOM, blocked and unrun slots stay in the preregistered denominator;
startup serving throughput remains N/A rather than an invented zero. Summary CIs are percentile
bootstrap over independent process-run values (seeded, 10,000 resamples when
available); fewer than two values yields no CI. Missing/failed process runs suppress the
throughput CI instead of filtering the sample to successful runs; success-fraction
CI uses observed successes divided by every scheduled request. A singleton
can still have a descriptive mean. No within-server window is an independent
bootstrap unit. Paired cell comparisons use the same independent
repetition slots and never emit a speedup automatically.

A comparison is `fairmatch` only when every required contract is known and
matches: tokenizer/checkpoint, weights/activation/accumulation/KV/output dtype,
RNE/rounding, scale profile, admitted resources, DP/TP/EP/PP, expert policy,
engine capability, cache/state isolation, and the complete stateful quality
gate. BF16/W8A16 versus F32 is explicitly non-fair even when request lengths
match. Equal request lengths, protocol success, or text resemblance do not
make a precision/topology comparison valid; `speedup` remains null pending the
full audit.

### Interrupted-stage reporting correction and custody limit

The GPU-owner [report](../target/validation/qwen35-35b-current/gpu-owner-final-cuda124/REPORT.md)
and [8k adjudication](../target/validation/qwen35-35b-current/gpu-owner-final-cuda124/context8k-adjudication.json)
are the evidence for this harness correction. The F32 8k stage's outer 1600-s
bound interrupted the second start. The old parent checkpoint still said
`planned`/`not_run`, although `lifecycle.json` showed `warmup_measure`; an empty
HTTP file and secondary `KeyError: summaries` did not establish zero issued
requests or a model failure. Those raw files and their separate adjudication
are **unchanged**. This correction runs only CPU/mock tests, not a GPU retest.

New executions now behave as follows:

- Parent `matrix.json` has `summaries`, `comparisons` and `reporting_errors`
  from the first checkpoint. Before dispatch, both the parent and the active
  slot are durably recorded as `in_progress`, with the slot directory and
  supervisor budget. `not_run` is reserved for undispatched slots.
- Checkpoints use write/flush/fsync, atomic replacement, and directory fsync.
  Catchable SIGTERM/SIGINT records an incomplete finalized result, captures
  available child artifacts and bounded group-cleanup evidence, and halts the
  remaining schedule. Previous signal handlers are restored. An uncatchable
  SIGKILL, machine loss, or storage error can leave `finalized:false`; that is
  **last-known state**, not proof the child stopped or the GPUs drained. No
  checkpointing scheme can finalize after its process is killed.
- The matrix HTTP lane checkpoints first-after-ready, warmups, entry to the
  measured phase, and each observed terminal request. `active_requests` are
  durable observations, not a complete interrupted window. `offered` in a
  partial HTTP checkpoint is a terminal-observation lower bound; unknown
  in-flight outcomes are `unobserved_request_budget`, not fabricated
  `not_issued` or zero failures. Interrupted-window rates remain unavailable;
  completed earlier process runs remain visible but cannot supply a complete
  multi-restart throughput CI.
- Empty/truncated/schema-incomplete child JSON is retained at its original
  path and recorded as `artifact_errors`. Secondary summary/comparison/log
  errors are recorded separately, never substituted for a supervisor timeout
  or owner interruption. Worker `work_status`/`error` preserve a primary
  startup/measurement failure when cleanup subsequently fails. The generic
  supervisor rethrows owner interruption with `supervisor_result` evidence.

**Measurement impact:** synchronous durable per-terminal checkpoints add client
work to the matrix window, and that cost is included in its denominator. The
HTTP `client.checkpointing` field declares this; legacy client mode has no
per-terminal checkpoint in its measured window. Freeze the changed client and
supervisor hashes and use a new output/protocol artifact. Do not silently
attribute a change from the old client to model/kernel speed. Source, binary,
checkpoint, numeric precision, topology, capacity and quality gates remain
independent requirements; the reporting fix grants no acceptance or fairmatch.

**Custody decision superseded by the isolated subreaper contract above.** The
old polling/group-only lane could not safely cover process ranks and correctly
kept them blocked. Current executions require a verified helper handshake before
root launch and kernel ECHILD plus a valid final helper receipt before the next
restart. This supports separate-session and reparented descendants without
owner-supplied containment claims. Failure remains unknown/fail-closed, and
independent GPU-quiescence evidence is still the execution owner's job. The
historical 8k timeout/partial artifacts are unchanged and were not rerun on GPUs.

The reporting/custody correction passed **91 CPU script tests** in the recorded
repair pass. The newer CPU validation above supersedes that test count, not the
interrupted raw evidence. CPU-only regression commands (fake HTTP fixtures, no GPU tests):

```sh
./target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests -p 'test_bench_matrix_recovery.py'
./target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests -p 'test_vllm_supervisor_failures.py'
./target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests -p 'test_*.py'
```

### Executable HTTP alignment protocol (`ferrule.http_bench.v1`)

This is a **bounded preliminary alignment protocol**, not a claim that the paper
long-prompt/concurrency matrix or matched-precision study has passed. The current
Hello/32 compatibility, oracle and profile results are separately recorded in 1.1 and 1.4.
It deliberately uses **raw `/v1/completions`** and EOS-permitted generation instead
of the proposed chat/ignore-EOS contract in Section 3. Do not merge these rows into
that matrix without preregistering the changed contract. The same client is
[`scripts/bench_ferrule_vllm.py`](../scripts/bench_ferrule_vllm.py) for both engines.
The existing `bench_vllm_serve.sh` is the official-vLLM-client wrapper with different
random/chat/ignore-EOS assumptions; `test_vllm_qwen35.py http` remains a sequential
smoke/diagnostic, not this benchmark. Neither old script is changed here.

**Ownership and preconditions.** Only the installation/server owner may probe or
reserve GPUs, start/stop production servers, install/change packages, or run a
profiler. The legacy client-only mode does none of those things and imports no inference
packages. The matrix mode above starts owner-supplied servers only with explicit
`--execute`; its optional corpus-export mode uses an existing local tokenizer. Wait for explicit ownership handoff before issuing HTTP traffic (HTTP
requests do execute inference). Run the engines sequentially on the same reserved
hardware; never run them simultaneously for a matched baseline. Ready/first-after-
Ready/load/prewarm/shutdown and physical drain evidence belong in a separate
startup/lifecycle record, not in request latency. The client cannot prove drain.

The historical successful launch is archived in
`target/validation/qwen35-35b-cuda124/commands.txt`, `ep8.json` and `REPORT.md`:
`target/validation/qwen35-35b-current/toolchain/cargo-cuda124/release/ferrule serve`
with `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8/`, served ID `qwen35-current`, port **18355**,
process EP8, devices 0..7, TP=PP=1, context 1024, active sequences/decode cohort 1,
batch/prefill chunk 32, KV 1024 MiB, full prewarm with 4 workers and host expert cap
40960 MiB. Its CUDA12.4 environment/link paths are in that command archive; do not
replace it with an arbitrary `target/release/ferrule`, source the environment into
the vLLM installation, or infer binary identity from the current source tree.
The archived raw `Hello` request has usage input=1/output=2. This proves a short
smoke only; the archived full40 strict run timed out and is **not** oracle PASS.
Source inspection of today's HTTP API does not prove an older binary implements
new fields, so this client never requires `ignore_eos`, `min_tokens` or token-ID
prompts. Existing production serving/runtime paths are reused without a fake
inference runner.

**Exact client commands, after the owner has started the matching server.**
The first URL/model below is the archived Ferrule contract. The second defines a
handoff convention: owner must expose vLLM at `127.0.0.1:18356` with served ID
`qwen35-vllm`, or replace those two arguments with its actual endpoint/model ID.
The client does not discover or assume a live server. Both commands use the
existing stdlib interpreter, not the vLLM environment, and send identical prompt
bytes and generation fields except for the served model ID:

```sh
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py \
  --engine ferrule --url http://127.0.0.1:18355/v1/completions \
  --model qwen35-current --prompt Hello --input-tokens 1 --max-tokens 2 \
  --seed 0 --warmup 2 --repeat 3 --requests 20 --concurrency 1 \
  --timeout 30 --measurement-seconds 60 \
  --output target/validation/http-alignment/ferrule-hello2-c1.json

/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py \
  --engine vllm --url http://127.0.0.1:18356/v1/completions \
  --model qwen35-vllm --prompt Hello --input-tokens 1 --max-tokens 2 \
  --seed 0 --warmup 2 --repeat 3 --requests 20 --concurrency 1 \
  --timeout 30 --measurement-seconds 60 \
  --compare target/validation/http-alignment/ferrule-hello2-c1.json \
  --output target/validation/http-alignment/vllm-hello2-c1.json
```

These are short-length alignment/latency baselines, not representative long-context
performance. Output files are exclusive-create: use a new filename for each new
experiment, never overwrite failed attempts. The commands above intentionally use
legacy-compatible `--window-kind fixed-time`: they measure offered-load delivery
against the full 60-second denominator, so a finite 20-request cap can leave
intentional idle in W and must not be called peak throughput. For the same request
count's real completion window, use the following fixed-work variants instead:

```sh
/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py \
  --engine ferrule --url http://127.0.0.1:18355/v1/completions \
  --model qwen35-current --prompt Hello --input-tokens 1 --max-tokens 2 \
  --seed 0 --warmup 2 --repeat 3 --requests 20 --concurrency 1 \
  --timeout 30 --window-kind fixed-work --measurement-seconds 600 \
  --output target/validation/http-alignment/ferrule-hello2-c1-fixed-work.json

/opt/conda310/bin/python -B scripts/bench_ferrule_vllm.py \
  --engine vllm --url http://127.0.0.1:18356/v1/completions \
  --model qwen35-vllm --prompt Hello --input-tokens 1 --max-tokens 2 \
  --seed 0 --warmup 2 --repeat 3 --requests 20 --concurrency 1 \
  --timeout 30 --window-kind fixed-work --measurement-seconds 600 \
  --compare target/validation/http-alignment/ferrule-hello2-c1-fixed-work.json \
  --output target/validation/http-alignment/vllm-hello2-c1-fixed-work.json
```

For `fixed-work`, `--measurement-seconds` is only a per-window safety cutoff;
there is no post-completion sleep. The JSON records `measurements.window_kind`,
`denominator`, `denominator_seconds`, `time_to_complete_seconds`,
`all_scheduled_requests_terminal`, `work_completed`, `all_requests_successful`,
and `intentional_idle_seconds`. The denominator starts at the first actual measured HTTP send and ends at the
last client-observed terminal/deadline for that request count; failures and
timeouts remain in it. `time_to_complete_seconds` is present when every scheduled
request reached terminal/deadline, even if some failed; it is null when the cutoff
left requests unissued. `work_completed` means all scheduled requests reached
terminal/deadline, not that all requests succeeded; use `success`, `failed`,
`not_issued` and `all_requests_successful` separately. Concurrency remains a real
closed-loop in-flight cap, and warmup is outside every measured completion window.
Do not compare a fixed-time row to fixed-work, or call either one a peak rate unless
the request cap and arrival policy are explicitly saturated.

`--repeat 3` means three windows on
one existing server, with two sequential warmups before **each** window, not three
independent server replicas. Warmup failure stops measurement. After a request
client-timeout/client failure, new submissions/repetitions stop; retain the current
window and not-issued requests, and require owner-confirmed drain before retry.
Warmups, CPU-client process startup and owner-measured cold startup are outside W.
HTTP/SSE errors are retained per request, with no automatic retries.

**Length and template matching.** Greedy payload is `temperature=0`, `n=1`, seed,
`max_tokens`, literal `prompt`, streaming and `include_usage`; there is no chat
rendering, server thinking switch, stop string, penalty or ignore-EOS override.
Raw completions may naturally generate `<think>` text; it is ordinary output on
both sides, not silently filtered. Disable/record server generation defaults,
prefix caching and reasoning/template transforms in the owner manifest. Matching
string hashes does not by itself prove matching tokenizer IDs/BOS behavior.

For 128/1024-token prompt cells, replace `--prompt`/`--input-tokens` with
`--prompts-json` pointing to a frozen JSON list of objects shaped as
`{"id":"p0","prompt":"literal pre-rendered text","input_tokens":128}`.
The number is an expected **usage** count, not an estimate based on characters.
Prepare/verify the corpus with the already-installed tokenizer or existing
`/tokenize` API separately; this client neither imports a tokenizer nor contacts
extra endpoints. It archives literal text, hashes, expected counts, and actual
`usage.prompt_tokens`/`completion_tokens`. For pre-rendered chat, freeze one
no-thinking template offline and replay that exact text through raw completions;
do not compare independently rendered chat APIs. The archived ctx=1024 server
cannot serve 1024 input plus output without a separately declared context change.

Early EOS is a valid successful response, **not** proof of max-length generation.
Inspect `input_length_matches`, `output_length_matches`, finish reason and usage.
If two engines produce different lengths, report their individual results but no
blind latency/rate speedup. Preflight both, choose a common shorter output cap
that both actually reach, freeze it, and rerun the **whole** paired workload;
otherwise publish a separate natural-EOS stratum. Do not drop short/error requests
post hoc. `--compare` checks identical workload/corpus and actual per-request
input/output counts, incomplete runs and not-issued requests; it always leaves
speedup null pending the independent resource/precision/quality audit.

**Measurement and failures.** `fixed-time` has an explicit fixed `W =
--measurement-seconds`, starting immediately before measured dispatch, not a sum
of successful-request times. `fixed-work` uses the actual completion denominator
described above; its `--measurement-seconds` is only a cutoff. Both modes keep
`--requests` as a finite scheduled cap, and
`--concurrency` a closed-loop in-flight cap. In fixed-time only, if the cap finishes early, the client
waits out the window and **idle time stays in W**. Thus the commands above measure
fixed-budget delivery and latency, not peak saturation throughput. For a separate
saturation experiment, preregister a sufficiently high request cap and one window;
requests unfinished at the cutoff are right-censored timeouts, not successes, and
owner-confirmed drain is needed afterwards. Do not mix finite-cap and saturation
cells or reinterpret a capped rate as a peak rate. Unstarted windows have no
measured throughput. Scheduled/offered/not-issued and every terminal status are
recorded; timeout/failure/partial requests never disappear from the denominator.

TTFT is send-to-first **nonempty content event**, excluding headers, role-only,
empty and keepalive events. Send means client HTTP-operation start and includes
connect/TLS; client workers are reused but each request uses a new connection.
Raw monotonic event times, finish, usage, DONE, EOF/error/deadline and per-request
status are retained. Parsing requires HTTP 200 + SSE MIME, one finish, consistent
nonnegative integer usage and exactly one `[DONE]`, and consumes through EOF to
reject duplicate/trailing data. Partial text/events survive timeout/error; absent
usage is null, never inferred from text/chunk count. `content_event_itl_seconds`
is **not token ITL**: chunks can hold multiple tokens. Exact token ITL/TPOT and
visible-committed-token rate remain unavailable. The explicitly named
`mean_tpot_usage_proxy_seconds = (last content event - first content event)/(N-1)`
is provided only for successful requests with known usage N>1, at least two
content events and no observed separate reasoning. Multi-token first/last chunks,
EOS/hidden tokens and buffering bias it; it is not a token-timestamp measurement.

JSON includes success/failure distributions separately, p50/p95 with linear
interpolation at `(count-1)*p`, sample counts, actual-length histograms and
successful-output/length-matched delivery rates over the **same W**. Known usage
on partial failures is reported separately, never counted as successful output.
Optional `--ttft-slo` and `--e2e-slo` must be preregistered for SLO-goodput; without
them that metric is null. Quality/semantic validity is an external gate; protocol
success alone is not numerical quality. Aggregate latency quantiles are pooled
request summaries, not confidence intervals across independent server replicas.

**Provenance and profile handoff.** Add `--metadata` with the owner-produced JSON
manifest for every publishable run. Top-level manifest fields are `source`,
`model`, `binary`, `dependencies`, `precision`, `topology`, `resources`, `cache`,
`startup`, and `quality_gate`; each may contain detailed evidence and artifact
references. Record source commit+dirty closure, binary/wheel SHA256 and exact launch
argv/environment allowlist; checkpoint/tokenizer hashes and revision; actual vLLM,
Torch/CUDA/runtime/driver versions (the pinned cu129 result verifies the current
row, not future installations); storage/activation/accumulator/kernel precision;
GPU UUIDs/interconnect/TP/EP/PP/rank backend; per-card and host budgets/high-water;
cache/prefix/prewarm/page-cache state and before/after counters. Never include
credentials. Metadata is operator-supplied, not remotely verified by this client.
Missing categories explicitly remain `unavailable`, with a machine-readable
`metadata.unavailable_categories` list (absent/null/empty or declared unavailable;
not a completeness audit of fields inside supplied categories). The runnable commands above
without a manifest are not a provenance-complete performance claim. Artifacts
contain prompts, output and raw server SSE/errors, so use non-sensitive inputs
and restricted storage.

After actual alignment and strict validation, the owner profiles a **separate**
run of the same production binary/workload with queue/prefill, attention/GDN,
router/experts, GEMM/dequant, H2D/read/communication and retirement evidence. Keep
instrumentation overhead and profile samples separate from uninstrumented
baseline windows. Client-only code does not justify runtime optimization; the
current v1–v3 reports supply separate runtime evidence. For new changes, measure
the bottleneck, rerun the oracle, and rerun the frozen baseline.

CPU-only validation (no server/GPU needed):

```sh
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -v
/opt/conda310/bin/python -B -m py_compile scripts/bench_ferrule_vllm.py scripts/tests/test_bench_ferrule_vllm.py
```

### Residency and host I/O controls

Record at spawn/Ready/post-warmup/post-measure/drain, per owner:

- GPU expert entries/bytes, hits/misses, upload counts/bytes, evictions, scratch,
  pending uploads, KV/recurrent state and allocator allocated/reserved bytes.
- Host compressed expert bytes, cache mode/capacity/hits, full/lazy prewarm,
  pageable/pinned/staging/RSS/cgroup peaks; host cap is not total RSS.
- Logical demand versus physical read operations/bytes, read latency, H2D/D2H,
  install and communications; storage client/NAS or block-device counters where
  available. `/proc` read counters of zero do not prove zero NAS traffic.
- Prefix state (off here), OS cache state, route/expert working-set reuse and
  counters missing from the implementation, with explicit `unavailable` reasons.

Single-card host-warm GPU-cache misses demonstrate host-to-GPU materialization,
not cold disk streaming. EP8 full residency provides a normal-path control, not
out-of-core evidence. Only measured request-time storage movement under stated
cache controls supports a storage-out-of-core claim. Counter deltas, not startup
cumulative totals, must accompany every measured window.

## 5. Timing, denominators and statistics

Use client monotonic timestamps: `s` = request sent, `t1..tn` = receipt of each
externally committed output token, `e` = terminal response/error/deadline.
Record server admission/queue time separately if instrumented; do not subtract
it from client latency. Role-only/empty SSE, HTTP headers and `[DONE]` are not
first tokens. SSE chunks may contain multiple tokens: if exact token boundaries
cannot be observed, report chunk gaps and mark token ITL unavailable rather
than dividing a chunk interval into invented token latencies. Retokenized text
alone is not guaranteed to reconstruct generated token IDs.

| Metric | Definition and population |
| --- | --- |
| TTFT | `t1 - s`; requests receiving a token, split successful and unsuccessful; no-token requests retained as failures/censored, not zero |
| E2E latency | `e - s`; success distribution plus separate time-to-failure/timeout distributions |
| Per-request TPOT | `(tn - t1)/(n - 1)` for successful requests with `n >= 2`; otherwise N/A |
| ITL | `tk - t(k-1)`; report token-weighted pooled p50/p95 and sample count separately from request-weighted TPOT p50/p95 |
| Per-request E2E output rate | `n/(e - s)` for success; mean/median of this rate is not aggregate throughput |
| Per-request decode rate | `(n - 1)/(tn - t1)` for success with `n >= 2`; explicitly excludes TTFT |
| Aggregate committed output rate | All caller-visible committed measured tokens, including partial failed/cancelled requests, divided by `W` |
| Successful-output throughput | Only tokens of successfully completed requests divided by the same `W`; show alongside committed rate so partial failures cannot look like useful success |
| Request throughput/success rate | Successful requests divided by `W`; successful requests divided by offered requests |

`W = latest measured client terminal/deadline - first measured send`, including
queueing, errors and stalled requests. Report server drain separately; a client
timeout does not prove execution stopped. Do not sum per-request rates, divide by
GPU kernel time, or count prompts/drafts/verification rows/warmup as output.
Optional total input-plus-output throughput must be explicitly named, not mixed
with output throughput. SLO-goodput is deferred until thresholds are preregistered;
if added, failed/time-out/quality-failing requests contribute no goodput.

Require the disjoint accounting identities:

```text
scheduled = not_issued + offered
offered = success + rejected + timeout + cancelled + error
error includes OOM, transport, server, and protocol errors (one terminal class)
```

Keep cause/phase separately to avoid counting an OOM that becomes a timeout twice.
All offered requests, including OOM/timeouts/errors, remain in the success-rate
request denominator, and their elapsed time remains in the shared wall-time
throughput denominator. Partial tokens remain explicitly
counted only in the committed metric. Startup OOM has no serving window or
offered requests: throughput N/A, configuration failure in the deployment table.
A live measured server producing no tokens can have a measured zero rate; that
is different from never starting. Run-level timeouts preserve failure records,
partial output, right-censoring and remaining `not_issued`; no silent deletion.

Report per-run p50/p95, counts and completion fractions, plus medians and 95%
confidence intervals across the five independent runs. Use seeded paired
run-level bootstrap (10,000 resamples) for aggregate-rate ratios/differences;
requests within a server run are not independent replicas. With five blocks,
intervals are preliminary; no p99 or strong tail/convergence claim. Missing,
failed or incomparable pairs are not dropped to produce a headline speedup;
report availability and conditional summaries, and leave the overall ratio N/A.
Never impute timeouts as successful latency samples or claim all-request p95
from a success-only distribution. Archive estimator and percentile conventions.

## 6. Quality gate: independent oracle, not CPU serving

Use [the FP8 reference contract](../scripts/qwen35_fp8_reference.md) and
[`qwen35_fp8_reference.py`](../scripts/qwen35_fp8_reference.py), explicitly
`--precision f32` for current Ferrule `F32Tf32x3`. The default `bf16_rne`
reference is a different contract and not an acceptable silent replacement.
Require manifest `status=complete`, `full_reference_complete=true`, schema
`ferrule.qwen35-fp8-reference.f32.v1`, matching payload hashes and self-checks.
Tiny-only, ignored, filtered, compile-only and old historical PASS are not full40
acceptance. CPU/F32 reference is solely an independent numerical oracle.
For the opt-in W8A16 contract, the separate audited `bf16_rne` v2 reference now
exists: its matching full40 comparison **failed**, as detailed in Section 1.5.
Do not keep labeling that blocker “missing oracle,” substitute the F32 reference,
or replace strict acceptance with kernel-order replay or argmax agreement.

On frozen single-card and thread EP8 paths, verify all vocabulary logits at all
reference positions, ordered route IDs, routing weights, and convolution/recurrent
state for prefill and cached decode. Existing strict EP comparison uses
`abs(actual-reference) <= 2e-4 + 2e-4 * abs(reference)` for numeric arrays; preserve
this preregistered gate, zero nonfinite values, zero violations and exact route
IDs, recording max absolute/RMS/tolerance ratio and first mismatch. TF32x3 need
not be bitwise equal. Revalidate after each kernel, dtype, residency or batching
change; do not widen tolerances after seeing results.

The small Hello/capital reference is a minimum gate, not coverage of the entire
performance corpus. Add a frozen canary subset covering both lengths, concurrency,
cache misses and 32-token greedy continuation; record ordered generated IDs,
EOS/stop reason and mismatch position. Full-model oracle extensions and vLLM
logit/route/state extraction are **pending**; text resemblance is not a substitute.
Separate oracle-compatible plain-token fixtures from chat workloads and match
actual rendered chat IDs for canaries. Benchmark runs need not compute full
logits in the timed path, but must link to the exact source/binary/profile oracle
run and its coverage; canary success alone is not strict state validation.

For a vLLM P-different profile, preregister an independent reference with its
actual rounding/precision before measurement; the legacy BF16-RNE script is not
a generic all-BF16 oracle. If exact greedy equality fails, report the mismatch
and stop exact-equivalence claims. Broader quality equivalence would require a
separate justified task/logit evaluation, not a post-hoc tolerance change.
A quality failure remains in the artifact and excludes that cell from validated
performance claims, even if HTTP throughput is high.

### Retired task/logit diagnostic workflow

The one-off KL/JS/NLL and fixed-corpus task-audit workflow is retired. The retained
HTTP matrix client is **not an equivalent replacement**: request timing, text
agreement and performance bootstrap cannot produce those metrics or certify
full-model/task quality. No new task-quality command or acceptance threshold is
provided here.

The preserved [three-gate report](../target/validation/w8a16-scalar-reference-20261002/full40-quality-final.json)
records `operator_math=passed`, `implementation_equivalence=failed`,
`task_quality=not_run`, `full_model_quality_accepted=false`. Those W8A16 results
are not reversed by the separate Mixed PASS. Future task studies need independent
reference/candidate outputs, pinned inputs and state-reset evidence, and
preregistered metrics, equivalence margins, sampling and multiplicity rules.

## 7. Minimal attribution, profiling and safety evidence

### Baselines and ablations

The external baseline is the running **vLLM 0.28+cu129 server** recorded in the
current artifact, not the vLLM client, Transformers CPU forward, NCCL microbenchmark,
or in-process SSE fixture. Use the same serialized workload and measurement client
against both servers. The current vLLM row is a measured reference with disclosed
precision/topology mismatch, not a matched same-compute baseline.
[`scripts/bench_vllm_serve.sh`](../scripts/bench_vllm_serve.sh) currently targets
Ferrule with vLLM client tooling, defaults to DeepSeek and random prompts,
`temperature=0` and `ignore-eos`. Its existence does not implement this protocol:
pinned request replay/token IDs, per-token timing, deadlines, interleaving,
fingerprints and all failure accounting must be checked or provided separately.
The client also needs to distinguish advertised input length from rendered chat
length. The first shared-harness and opt-in CPU-reference lane are already
implemented; the matrix runner adds opt-in execution infrastructure, not GPU experiment
results or wholesale deletion of CPU internals.

| Required control | Safe experiment | Status/boundary |
| --- | --- | --- |
| Residency sensitivity | Single-card cache budget B versus a smaller legal budget (target B/2), same full host prewarm, prompt order and arithmetic; record actual expert entries/bytes and misses | Configuration mechanism exists; choose both values in preflight respecting full-route capacity and scratch, then freeze. If infeasible, retain failure, not an invented switch on EP8 |
| One profile-led optimization | Frozen pre-optimization versus optimized binary; same resources/quality gate, modify only the measured bottleneck | v1–v4 provide runtime A/B evidence; v4 +0.685%/-0.798% is near noise, not stable speedup |
| Protocol cost/control | Resident/no-fault path: exclusive CPU scheduler/admission/commit/retirement times and metadata; compare a safe serialized load/execute/drain reference against full protocol on the same connected provider | Simpler baseline wiring/timing may be pending; fixture-only implementation must remain a separately labeled protocol experiment |

The safe serialized control retains complete-key identity, capacity checks,
leases, commit semantics and last-consumer fences; it removes concurrency or
sharing, not correctness. For a claimed dedup benefit, use identical expert
demands with controlled repeated versus diverse keys and measure physical reads/
uploads against that control. Do not disable admission, quarantine or fences to
manufacture an unsafe fast baseline. If the Qwen path does not traverse the
mechanism, a fixture ablation cannot explain its HTTP speed. No required sweep
of every subsystem; prefetch, wakeups, process owners and KVSET are follow-ups
only when connected and relevant to a measured bottleneck.

After the first baseline, profile separate diagnostic runs: queue/prefill,
attention/GDN, routed experts/router, dequant/GEMM, host read, H2D/install,
communication/synchronization and commit/retirement. Align CPU/GPU causal events
and report exclusive versus nested intervals; high utilization is not overlap
proof. Identify one bottleneck before optimizing. Re-run strict oracle, then
repeat the complete randomized comparison on the optimized source. Instrumented
runs do not replace low-overhead performance runs; measure observer overhead.

### Profiling evidence and safety

The retained matrix planner supports explicit profiling via `--profiling-binary`,
`--nsys` and `--profile-request-seconds` (180 or 600); both paths must identify
existing owner-pinned executables. Select one short `p128-o1-c1` cell first,
with one fresh process, one warmup and one measured request. This diagnostic
profile does not establish an o32 throughput baseline or TPOT distribution.

1. **Compiled instrumentation and flush:** IPC instrumentation is compile-time
   opt-in; a runtime sink cannot enable an ordinary binary. Matching parent/child
   builds and valid nonzero records are required. Thread-exit sinks are lifetime
   observations, not request deltas; missing/partial records are not zero cost.
   Aggregate v2 copy duration is nullable; detailed clocks perturb measurement.
2. **Custody before progression:** keep the foreground capture wrapper and
   verified outer subreaper alive through cooperative server shutdown and export.
   Do not put a shorter timeout around the cleanup owner or signal foreign PIDs.
   Timeout, failed export or unknown custody stops progression, without automatic
   retry. ECHILD proves process drain, not GPU quiescence; check devices separately.
3. **Coverage before attribution:** retain raw/partial reports and failed requests.
   Correlate root and every expert PID/slot with expected physical GPU identities;
   require positive-duration CUDA events in the aligned HTTP request window. A
   nonempty report, startup kernels or artifact smoke alone is not EP8 coverage.
   Align trace/HTTP clocks and check profiler loss/injection warnings explicitly.
4. **Non-additive timing:** IPC copy is nested in encode, pipe wait in frame I/O,
   and thread/device intervals overlap. Never divide lifetime sums by request
   latency or add API/kernel/copy durations into a latency percentage. Use interval
   unions per device in the aligned window; compare performance using ordinary
   uninstrumented binaries and independent fresh-process samples.

The [older p2048 capture](../target/validation/qwen35-35b-current/profile-handoff/REPORT.md)
had a warmup timeout and missing profiling artifacts. It remains censored/missing
evidence, not a measured bottleneck or completed performance result.

### Fault model and budget conservation

Use existing protocol fixtures first, then bounded faults on the selected real
serving path where supported. Fault instrumentation not yet connected is
**pending**, not a reason to claim hardware coverage. Minimal fault cases:

- Delayed completion plus request cancellation during read/upload/install.
- Duplicate/stale generation completion after cancellation and attempted reuse.
- Primary execution/admission failure plus cleanup failure with retry.
- Publish-pending cancellation and a missing completion/timeout causing Unknown.

Record request/transaction/operation/generation/resource IDs and phase timestamps;
check no duplicate physical dispatch on retry, no stale logical publication,
no provisional output, exactly one caller terminal, no premature credit return
or slot reuse, and remaining service capacity. An irreversible physical Publish
may finish after accepted cancellation; cancellation must not rewrite its intent.
Child-loss evidence belongs only to a separately gated process profile.

For every covered resource domain r (bytes, pages or slots), map actual ledger
states to **disjoint** categories and verify at every event:

```text
capacity_r = free_r + held_r + submitted_r + resident_or_live_r + quarantined_r
charged_r = held_r + submitted_r + resident_or_live_r + quarantined_r <= capacity_r
closing_charge_r = opening_charge_r + acquired_r - returned_unsubmitted_r - physically_retired_r
```

If quarantine is represented as a subset of submitted/live in the implementation,
subtract that subset there before forming this presentation; never double count
it. Unsubmitted held grants may return after safe admission rollback; submitted
claims require matching physical completion/retirement evidence. Submitted-to-
resident is a custody transfer, not a new acquisition. Execution reserve is a
policy partition of capacity, not extra capacity. Map KV COW/shadow,
shared physical pages and last-consumer pins explicitly; logical entry charge
must not be mistaken for unique physical bytes. State ledger exclusions (weights,
allocator reserves, OS cache, metadata etc.) and measure them separately before
claiming a total-memory bound.

Show capacity/high-water/rejections, logical terminal time, physical drain time,
quarantined bytes/slots, outstanding operations and post-fault capacity. Known
completion should release transient claims only after owner/fence evidence;
Unknown must keep custody charged and may prevent full drain indefinitely.
Owner death, process exit, thread join, trace completion or an unrelated root
CUDA event is not physical completion proof. CPU failpoints prove protocol
semantics only; CQE/CUDA claims need real matching completion evidence. No GPU
reset/hang recovery, persistent restart/replay, or general liveness guarantee.

## 8. Artifact schema and reproducibility

Proposed archive schema **`ferrule.paper_eval.v1`**, not an existing benchmark
schema. Use UTF-8 JSON/JSONL, seconds in derived metrics, monotonic nanoseconds in
raw timing, integer bytes/tokens, UUIDs for runs/requests. Numeric `null` always
has a reason (`pending`, `unsupported`, `not_observable`, `not_applicable`,
`startup_failed`, `censored`); missing is never zero. Raw failures are mandatory.

| File/table | Required fields |
| --- | --- |
| `manifest.json` | schema/protocol ID and hash, experiment/cell/pair/run IDs, engine, source/model/binary/environment fingerprints, precision class/contract, capability/oracle status, topology, budgets, resolved commands/env/config, cache policy, corpus/template hashes, seeds, deadlines and instrumentation mode |
| `schedule.json` | preregistered cells, eligible/blocked reason, five block orders, request order, warmup/first-request membership, pilot/tuning choices and amendment history |
| `requests.jsonl` | run/request IDs, scheduled/send/first-token/last-token/terminal times, admission time if available, rendered input IDs/hash/count, target/actual output IDs/hash/count, token/SSE event artifact reference, HTTP/finish status, one terminal class plus cause/phase, partial committed count, quality status and timeout/censoring |
| `events.jsonl` | clock domain, monotonic timestamp, owner/rank/transaction/operation/generation IDs, resource kind and phase, logical/physical action, bytes/slots, completion/fence identity, custody and error/cleanup state |
| `resources.jsonl` | per-owner timestamped capacity/held/submitted/live/quarantine/free, allocation/reserve/RSS/pinned high-water, cache snapshots, physical read/H2D/D2H/install/communication deltas and unavailable-counter reasons |
| `oracle.json` | reference schema/source/model/payload hashes, binary/profile linkage, cases/shapes/elements/tolerances, logits/routes/state/greedy comparisons, nonfinite/violations, coverage and pass/fail/blocked |
| `summary.json` | scheduled/offered/terminal counts, startup/warmup/serving/drain windows, committed/successful tokens, exact denominators, per-run p50/p95 and counts, bootstrap method/seed/CI, comparability and excluded-claim reasons |

**Source fingerprint:** full Git SHA plus dirty status, binary-safe tracked diff
hash AND source archive, all untracked build inputs by path/SHA256, lockfiles,
submodules/vendor revisions, build flags/features/toolchains and build log.
Record the source tree at build and execution; a dirty-count or HEAD alone cannot
identify this worktree. **Binary fingerprint:** executable and rank-worker hashes,
linked CUDA/native libraries or container digest, not just a filename or mtime.

**Model fingerprint:** immutable checkpoint revision plus full shard/config/index/
scale/tokenizer/template hashes, with model-license-compatible acquisition steps.
Hash outside timed windows and declare that hashing can warm storage caches.
The existing reference certifies read payloads and headers, not all unused
experts or entire shards; retain that scope and add a full checkpoint manifest
before claiming complete model identity. Record all server/client/oracle runtime
versions separately; container tags without digests are insufficient.

Store raw HTTP/SSE/errors, exit codes, failed startup logs and shutdown evidence
with checksums. A future schema validator and table generator must reproduce
counts, identities, metrics and all paper tables solely from the archive; they
are **pending**, not delivered by this documentation change. Redact credentials
and private paths without altering token identity; token IDs and traces may
encode sensitive prompts. Publish synthetic corpus and model acquisition hashes,
not private session exports. Ignored `target/` reports are not a public artifact.

## 9. Acceptance gates

Freeze the actual connected profile and independent quality evidence before
benchmarking; then establish a compatible baseline, profile separate diagnostic
runs, validate one evidence-led change, and repeat preregistered process-level
comparisons. Source-pinned status belongs in [current acceptance](CURRENT_ACCEPTANCE.md),
not a second progress checklist here.

W8A16 strict FAIL, the incomplete historical 8k stage, and precision/topology
mismatches remain exclusions. No missing mechanism, failed cell, operator timing
or old measurement may be promoted to current throughput, statistical
significance, novelty or general readiness.
