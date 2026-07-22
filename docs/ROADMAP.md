# Ferrule single-DGX Spark release roadmap

> Canonical engineering and release plan for exact DeepSeek-V4-Flash-DSpark inference on one NVIDIA DGX Spark / GB10.
>
> Updated: 2026-07-22.

## 0. Current decision

The project remains **GO as an engineering program**, but complete-cycle viability is still below the release gate and no SOTA claim is valid. The production runtime now executes a true multi-sequence DSpark transaction: scalar proposals retain sequence isolation, ragged proposal blocks are packed into one exact target verification cohort, and every sequence independently promotes or rolls back its exact prefix. Runtime and backend KV prefixes commit as one cohort transaction, and the production path does not replay accepted target rows.

This transaction is model-neutral and arbitrary-width at runtime; it does not encode a `Q <= 6` scheduling special case. Per-sequence provisional checkpoints map packed rows back to their owning sequence, and the generic scheduler can use bounded prefill deferral to form decode cohorts without sacrificing forced progress. Official vLLM serving runs at multiple concurrency levels observed real production decode batches of two, three, and four requests with no failed requests.

Packed cohorts are a necessary execution foundation, but they are not sufficient. Concurrency improves only modestly because every exact expert miss still follows the inline path `route → materialize_selected_experts → io_uring submit_and_wait → upload wait → MoE`, blocking the model worker. Linux expert streaming now defaults to `io_uring`, and the obsolete mmap backend, one-row probe/replay production path, predictive prefetch experiments, and resident/miss dual-MoE experiments are removed.

Selective official-Python comparison still leaves a fifth-position near-tie divergence in the FP8 backbone, while CPU reconstruction continues to localize the mismatch before the proposal head. That correctness lane remains required, but the immediate systems priority is:

1. make target-layer execution resumable after exact routing while retaining owned arena, KV, branch, and checkpoint state;
2. add a global completion reactor spanning `io_uring` CQEs, pinned payloads, asynchronous H2D, CUDA events, generation validation, residency publication, and waiter wakeup;
3. schedule other resident-ready cohorts while misses are in flight under hard SQE, slab, upload, frame, lease, and GPU-ready-work credits;
4. preserve the all-resident kernel path as the compute floor and continue only measured, parity-safe kernel work after overlap is observable;
5. rerun the frozen production serving suite, then the same-contract competitor suite.

## 1. Release contract

### 1.1 North star

Ferrule releases the single-DGX Spark milestone only when one GB10 sustains:

```text
>= 16 accepted output tokens/s
```

under a frozen warm workload using:

- the exact `models/DeepSeek-V4-Flash-DSpark` checkpoint;
- exact target verification and real router/expert execution;
- the production checkpoint-native DSpark proposal source;
- complete draft, verify, commit/rollback, and uncovered-I/O timing;
- bounded memory with no swap or page-cache collapse;
- zero headline request failures;
- reproducible source, binary, model, environment, and benchmark manifests.

Verified rows/s, target passes/s, kernel TFLOP/s, and isolated microbenchmarks are evidence, not the final metric.

### 1.2 Release equation

The production DSpark contract uses separate quantities:

```text
gamma                    = draft slots produced by the checkpoint attachment
Q                        = target rows executed by verification
accepted_draft_tokens    = draft prefix accepted by the target
correction_tokens        = target correction or trailing bonus tokens
C                        = externally committed output tokens
```

The checkpoint declares `gamma=5`. The DeepSeek reference model, official DeepSpec code, vLLM, and SGLang agree on `Q=gamma+1=6`: target verification consumes the carried anchor plus five draft tokens and externally commits `accepted_draft_tokens + 1`, where the extra target token is a correction after the first rejection or a trailing bonus after full acceptance. Any width override remains experimental and requires independent proposal/logit/confidence and acceptance evidence.

For a frozen operating point `Q`, mean externally committed tokens `C(Q)`, and complete cycle latency `T_cycle(Q)`:

```text
output tok/s = C(Q) / T_cycle(Q)
```

`T_cycle(Q)` includes:

```text
draft proposal
+ exact packed target verification
+ accepted-prefix computation
+ commit or rollback
+ uncovered expert I/O
```

The release gate is:

```text
C(Q) / T_cycle(Q) >= 16
```

Accepted draft prefix length is explanatory telemetry, not the release numerator.

### 1.3 I/O budget

At release throughput, long-run uncovered reads must remain inside the acceptance-aware budget derived from measured storage bandwidth. Raw bandwidth and per-run byte figures belong in benchmark artifacts rather than this roadmap.

This requires a cache-heavy, acceptance-aware regime. Resident target compute cannot compensate for excessive cold expert traffic.

## 2. Platform and implementation invariants

### 2.1 Target platform

| Component | Contract |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1 |
| build target | `sm_121a` |
| CUDA | 13.0 |
| CUTLASS/CuTe | 4.6.1 at `e05f953a5b3d38adc240df2ff928e0421c2abba3` |
| memory | 128 GB nominal coherent memory |
| model | DeepSeek-V4-Flash-DSpark, 43 target layers, 48 shards |

Other hardware is unsupported by this release path and must fail explicitly.

### 2.2 Runtime ownership

- Rust uniquely owns contexts, streams, allocations, plans, KV, residency, I/O, scheduling, and cancellation.
- Native FFI descriptors are versioned POD values with Ferrule-owned pointers.
- The provider allocates nothing and performs no host synchronization.
- No CPU/reference production fallback exists.
- Missing provider capability, plan, shape, or target is an error.
- New hardware receives an independent optimized provider rather than a compatibility branch in the GB10 path.

### 2.3 Kernel-plan contract

Each layer has one semantic `LayerKernelPlan`. M scheduling is private to the provider; the model plan does not contain M=1/2/4/8 variants.

The CUTLASS manifest currently publishes semantic superkernels for:

- FP8 QueryA and KV;
- BF16 compression;
- HC production;
- shared FFN;
- stable-frame routed MXFP4 MoE;
- MLA output and latent packing;
- DSpark target-tap FP8 main projection, BF16 boundary, and RMSNorm;
- DSpark committed-paged-context plus ephemeral full-block hybrid MLA attention;
- DSpark HC head, final norm, base LM head, sequential Markov selection, and confidence projection.

Routing, sparse attention, paged control, and recurrent metadata may remain dedicated CUDA operations where fusion has not demonstrated an end-to-end win.

## 3. Verified evidence

### 3.1 Gate F1 resident verification

Completed:

- packed target verification covers diagnostic widths and the checkpoint-native width with parity;
- the resident path performs no selected-expert I/O and remains allocation-stable after prepare;
- sequential-prefill and packed-row arenas are separated so their compressed buffers cannot alias;
- persistent pinned mirrors remove repeated packed-attention metadata staging;
- compact router control transfer overlaps the already-fused shared FFN;
- target-only execution remains feasibility evidence, not a complete-cycle or output-throughput result;
- experimental width overrides remain excluded from release evidence until proposal and acceptance parity are established.

### 3.2 Operator optimization checkpoint

The current GB10 path uses semantic CUTLASS fusion rather than host-side composition of small GEMMs. Completed operator work includes:

- stable-frame routed MXFP4 MoE;
- optimized generic FP8 matrix paths and a bitwise-equivalent four-warp/N64 BF16 rows kernel;
- fused FP8 QueryA and KV production;
- fused BF16 compressor work;
- fused shared FFN and MLA output stages, with a same-stream three-kernel MLA specialization for one-row verification;
- a bitwise-equivalent 256-column HC-function tile for the one-row producer while wider inputs retain the original 128-column arithmetic path;
- DSpark target-tap projection and normalization;
- DSpark hybrid paged/block attention;
- DSpark proposal head, Markov selection, and confidence projection.

These operators are connected to the production server path. Their isolated wins remain supporting evidence only; complete-cycle profiling is authoritative.

### 3.3 Correctness checkpoint

Verified commands include:

- kernel-plan unit tests;
- CUTLASS provider ABI/semantic tests;
- dynamic-M coverage including the large prefill shape;
- MXFP4 MoE and FP8/BF16 MMA smoke tests;
- bitwise single-row MLA versus cooperative, N64 BF16 versus N16, and HC tile-256 versus tile-128 differential tests;
- full-model packed CUDA versus token-loop CUDA parity.

The latest full-model parity run matched every layer and the selected intermediate cut points exactly.

### 3.4 DSpark execution and transaction checkpoint

Implemented in the prepared GB10 execution path:

- target hidden taps are captured and packed directly on device without host materialization;
- CUTLASS performs the checkpoint's target-tap projection, numerical boundary, and normalization;
- DSpark execution stages maintain independent committed context-KV state from the projected target context;
- packed prompt prefill and target verification publish DSpark context updates inside the same provisional paged transaction as target KV;
- hybrid attention combines committed paged context with ephemeral proposal-block context without publishing proposal KV;
- the proposal head, final normalization, base LM head, sequential Markov dependency, greedy selection, and confidence projection execute on device;
- resident embedding gather builds the checkpoint-native anchor/noise proposal input from the prepared embedding table;
- one compact final transfer returns proposal status, token IDs, and confidence logits.

Implemented in the production runtime/server path:

- a model-neutral `DsparkProposalRunner` executes against authoritative per-session state while preserving scalar proposal isolation;
- ready proposals with ragged widths are collected into one multi-sequence packed target verification;
- per-sequence provisional checkpoints map global packed rows back to sequence-local recurrent, compressed, DSpark-context, and paged-KV state;
- each sequence retains any exact prefix from zero through its full executed width without replaying accepted target rows;
- `KvReservationCommit` and `commit_prefix_batch_with_freed` validate the whole cohort before publishing runtime pages, while the prepared backend retains and commits the same logical prefixes;
- acceptance, correction/bonus staging, external token emission, limits, EOS, stop strings, cancellation, and failure cleanup remain independent per sequence;
- the OpenAI server no longer uses ordinary one-token target decode for DeepSeek-V4 headline serving;
- runtime transaction tests cover ragged full, partial, and zero acceptance, mixed commit/rollback, stale-generation rejection, and arbitrary-width prefix retention.

Correctness status:

- the real proposal is deterministic within one numerical target path and does not mutate committed DSpark context during proposal;
- packed prompt and target execution publish DSpark context and target KV through the same provisional transaction;
- production CUDA smoke requests reconcile externally committed tokens with runtime callbacks after atomic cohort commit;
- the true production cohort path does not replay an accepted prefix and does not contain a checkpoint-width scheduler special case;
- CPU checkpoint reconstruction proves the CUDA proposal head for all native positions, while selective official Python agrees on the first four greedy tokens; the fifth near-tie remains a pre-head FP8 numerical parity gap;
- full official proposal/logit/confidence parity is therefore **not yet established**; immutable proposal identity/hash and a frozen official near-tie fixture remain required.

### 3.5 I/O checkpoint

Expert streaming uses `O_DIRECT + io_uring` reads into registered pinned slabs with stable slot/generation/lease semantics. The expert mmap backend has been removed; Linux production defaults to `io_uring`, with positioned reads retained only as an explicit fallback. GGUF and safetensors ingest mappings are separate and are not expert-streaming backends. GB10 direct NVMe-to-GPU GDS is not claimed because the platform does not provide the required GPU/P2PDMA path.

The current implementation is asynchronous only at the device primitives, not at the execution transaction. After each exact layer route, the submitting model-worker thread still calls `submit_and_wait`, waits for pinned payload completion and upload readiness, then resumes MoE inline. There is no global CQ reactor, durable load ticket, waiter cohort, asynchronous residency publication, or scheduler-visible continuation, so exact misses still serialize otherwise independent cohorts.

Packed serving confirms that this is now the dominant overlap boundary: larger cohorts amortize target work but still create large exact expert unions and block together on misses. The next I/O milestone is therefore resumable exact-route execution with single-flight physical loads and resource-derived backpressure, not predictive prefetch, mmap, a larger blind hotset, or a second resident/miss MoE pass.

### 3.6 Production OpenAI/vLLM serving checkpoint

Completed:

- the existing OpenAI-compatible server and official vLLM serving benchmark exercise the production DSpark path;
- no synthetic proposal source, target-only driver, CPU fallback, second server, or parallel benchmark driver is used for serving results;
- production decode gathers ready sessions into one ragged packed target cohort while proposal state remains isolated per session;
- accepted prefixes are retained independently, then runtime and backend KV state commit through one cohort transaction without production replay;
- bounded decode-cohort formation may advance prefill to collect ready work, but a deferral limit forces decode progress and prevents starvation;
- concurrency validation observed real decode batches of two, three, and four requests with zero request failures;
- owned arena checkout/checkin is active in the DeepSeek packed runner, establishing the resource lifetime needed by future suspended continuations;
- proposal, verification, transaction, expert residency/I/O, CUDA/runtime, and externally emitted-token telemetry come from the real cycle.

Packed cohorts improve production concurrency, but the gain is limited because exact expert loading still blocks the model worker. This evidence closes “form a real cohort” as the immediate scheduler blocker and promotes resumable expert I/O to the next architecture gate. The resident target path remains an independent lower bound: even with no expert reads, current small-M MoE, MLA, projection, compression, and launch costs leave substantial kernel work before the release target is credible.

Current cohort artifacts are under `target/bench/r2-cohort`; prior acceptance and resident-kernel evidence remains under the existing `target/bench/dspark-acceptance-sweep-*` and `target/bench/r4-all-hit-1` directories.

## 4. Remaining execution plan

### Immediate next pass — resumable layer execution and completion reactor

The next optimization pass stays on the existing `just dsv4-serve` → `just dsv4-vllm-bench` production path. It must not introduce a second server, model driver, transaction implementation, speculative route predictor, or headline benchmark path.

A cohort becomes an explicit state machine:

```text
CohortReady
→ CohortRouting
→ CohortWaitingExperts
→ CohortReadyMoe
→ CohortRunning
→ CohortFinished
```

At each routed target layer, execution runs through pre-MoE and exact routing exactly once. All resident dependencies continue immediately. Any missing dependencies produce durable operation IDs and a `CohortWaitingExperts` continuation that owns its arena checkout, provisional model branches, KV reservations/bindings, packed-row mapping, layer position, and cancellation identity. The model worker then runs another resident-ready cohort instead of waiting inline.

Required completion chain:

```text
io_uring CQE
→ pinned host payload ready
→ asynchronous H2D submission
→ CUDA event completion
→ slot generation and source-identity validation
→ logical residency publication
→ wake waiter cohorts
```

Required implementation:

- move submission/completion ownership from the calling transaction to one global reactor with stable load tickets;
- single-flight identical physical loads and attach all dependent cohorts as waiters;
- resume only after every exact dependency for the suspended layer is resident and generation-valid;
- keep proposal state, provisional target/DSpark context, runtime/backend KV, and external commit invisible until the original cohort transaction completes;
- make cancellation remove a waiter and release owned continuation resources without invalidating a physical load still required by another cohort;
- derive backpressure from available SQEs, registered pinned slabs, upload slots, expert frames, leases, and ready GPU cohorts rather than estimated request count;
- preserve immutable request/session/cycle/cohort/load identities and non-synchronizing host/CUDA timing across suspension and resume;
- forbid busy polling, device-wide timing synchronization, mmap expert reads, predictive expert prefetch, and resident/miss duplicate MoE execution.

Exit gate:

1. an exact miss suspends its cohort without blocking the model worker;
2. another resident-ready cohort demonstrably executes while read or upload work is in flight;
3. CQE, H2D, CUDA-event, generation validation, publication, wakeup, resume, cancellation, and failure paths are covered by deterministic tests;
4. physical read/upload bytes reconcile with cache hits, inflight joins, alignment, sharing, cancellation, and eviction;
5. internal proposal/accept/commit counters still reconcile exactly with externally returned vLLM tokens and EOS/stop behavior;
6. cold, warm, and forced all-hit production runs report wall critical path and CPU/GPU/NVMe busy time without double-counting overlap.

### Phase R1 — Production DSpark proposal parity

Purpose: prove official checkpoint parity for the already-landed CUDA/CUTLASS DSpark attachment before treating its tokens, confidence, or acceptance as valid.

Work:

- use the official DeepSpec repository at commit `005e03b81cec38b7da6399833d609ee89a2587f2`, the checkpoint reference model, vLLM, and SGLang as protocol references;
- load the three DSpark stages and prediction heads into the single-owner prepared resources with non-aliasing execution identities 43–45;
- preserve the landed device-side target hidden-tap 40/41/42 capture and fused stage-zero `main_proj`/`main_norm` path, then prove official Python fixture parity and measure its complete latency;
- preserve the landed dedicated sliding-window context-KV state per DSpark stage. Prefill/update it from projected target context; never publish ephemeral proposal-block KV as committed context;
- construct the five-row draft input `[anchor, noise × 4]`, share the target embedding, and run the entire block through all three stages with non-causal intra-block visibility using the landed hybrid-attention semantic bundle;
- compile DSpark weights, block workspaces, expert catalogs, and kernel plans into the GB10 execution image;
- apply the final HC head, final norm, and shared target LM head to produce five base-logit rows;
- generate five proposals sequentially: each row's base logits receive the low-rank Markov bias from the previous sampled token;
- compute per-position confidence from `[final hidden, previous-token Markov embedding]` and expose the native block plus confidence/survival telemetry to the scheduler;
- expose an immutable proposal-source identity/hash and fail if any DSpark plan, tensor, expert frame, context state, or target capability is missing.

Exit gate:

- the checkpoint-native `gamma=5` proposal block, `[anchor, noise × 4]` backbone input, non-causal block attention, six-row target layout, and correction/bonus behavior are explicitly documented and tested;
- any shorter adaptive verification window is a confidence-selected prefix of the already computed native block and does not rerun the backbone with a different block shape;
- any width above the checkpoint declaration, including the historical V=8 probe, is experimental until independent proposal/logit/confidence parity and acceptance evidence pass;
- proposal execution is CUDA-only and allocation-free after prepare;
- proposal parity and state-transition tests pass;
- no target-model or synthetic proposal source is used for headline data.

### Phase R2 — Gate F2 real acceptance

Purpose: first establish the exact checkpoint-native proposal/accept/bonus transaction, then determine whether the optimized device path converts it into a viable complete cycle.

Work:

- instrument the existing production server and `dsv4-vllm-bench` path rather than adding a parallel F2 driver;
- measure draft, verify, accepted-prefix, commit, rollback, and uncovered-I/O components under the authoritative full-cycle profile;
- record `gamma`, target rows, accepted-draft histograms, correction/bonus tokens, and externally committed `C(Q)` for every supported operating point;
- reconcile internal committed tokens with externally returned tokens;
- preserve exact per-layer route dependencies, load-operation joins, wait/resume, generation, and KV transaction traces;
- run full-, partial-, zero-accept, EOS, cancellation, and failure cases.

Correctness exit gate R2a, before graph optimization:

- proposal, target verification, accepted draft prefix, correction/bonus, KV state, and externally returned tokens reconcile exactly;
- component timing and acceptance artifacts exist for the checkpoint-native operating point;
- full-, partial-, zero-accept, EOS, cancellation, and failure cases pass.

Current status: non-zero real acceptance, full/partial/zero transaction shapes, per-cycle accounting, and external SSE reconciliation are verified. The wider long-output/cancellation/EOS/near-tie release corpus remains before R2a is declared complete.

Performance exit gate R2b, after the R4 device-path work:

```text
C(Q) / T_cycle(Q) >= 16
```

for at least one production-validated operating point. The checkpoint-native target-only path leaves a narrow complete-cycle budget, so R2b remains unmeasured and highly constrained. Experimental width overrides cannot satisfy R2b without independent proposal parity and acceptance evidence. If every production-valid point is below the release target with no credible optimization budget, the current release target is rejected.

### Phase R3 — Gate F3 acceptance-aware expert I/O

Purpose: overlap exact cold-expert movement with independent CPU/GPU work while keeping traffic inside the storage and capacity envelope.

Work:

- compute the authoritative dependency set only after the packed layer reaches its exact router; do not predict or prefetch experts before routing;
- represent each miss with stable operation IDs and retain the originating cohort as an owned continuation at the pre-MoE boundary;
- single-flight loads using a key containing at least model instance, layer, expert, source identity, and destination generation;
- run one global `io_uring` submission/completion reactor, pinned-payload lifecycle, asynchronous H2D stage, CUDA-event completion stage, and waiter wakeup path;
- publish logical residency only after source identity and destination generation validate at completion;
- reconcile exact logical dependencies with physical loads, inflight joins, cache hits, alignment, cancellation, failures, and eviction;
- schedule resident-ready cohorts while miss-blocked cohorts wait, with bounded fairness and no busy polling;
- enforce backpressure from real SQEs, pinned slabs, upload slots, expert frames, leases, and ready GPU cohort capacity;
- compare the resulting residency policy with route-trace oracle/Belady bounds after overlap is correct;
- run long-output, cancellation, failure-injection, and memory-pressure tests.

Exit gate:

- uncovered reads remain within the storage budget at the release operating point;
- model execution continues on independent ready work during exact read and upload waits;
- no swap, page-cache collapse, unbounded waiter growth, or resource-credit oversubscription occurs;
- expert memory, continuation arenas, provisional KV, and temporary workspaces remain bounded;
- every miss, shared load, cancellation, stale generation, and failed read/upload remains exact and recoverable.

### Phase R4 — Device all-hit path and stable graphs

Purpose: reduce the unavoidable all-resident compute floor after the execution engine can overlap exact misses with other ready work.

The older ordinary-decode all-hit replay measured about `288.97 ms` with 1,737 launches, but production verification uses the packed semantic target path. Its authoritative pre-optimization one-row lower bound was `239.122 ms` (`4.182 rows/s`), with 1,806 physical kernels, 45 D2H copies, no selected-expert I/O, no measured allocations, and exact capture/replay parity.

The current production-equivalent roofline is `225.999 ms` at V=1 (`4.425 rows/s`, `-5.49%`) and `325.017 ms` at the checkpoint-reference V=6 (`18.461 rows/s`). V=1 now executes 1,892 physical kernels because each of 43 MLA bundles uses three ordered kernels instead of one cooperative kernel; the extra launches are outweighed by lower device time. All V=1/2/4/6/8 top-1 logit bits match the prior baseline, and the path remains resident, allocation-stable, and parity-clean.

Measured operator changes:

- the one-row MLA split reduced MLA time from about `37.14 ms` to `35.58 ms` per pass and improved full V=1 wall by about `0.8%`;
- the four-warp/N64 BF16 rows kernel reduced the 96 BF16 projection calls from about `33.17 ms` to `24.94 ms` (`-24.8%`) while preserving every output bit against the N16 kernel;
- the one-row HC tile-256 specialization reduced the 86 HC producer calls from about `24.46 ms` to `22.81 ms` (`-6.7%`) with bitwise equality for hidden, normalized, packed/scales, and split outputs;
- FP4 MoE padding-elision and direct-warp HC streaming prototypes were rejected after measurements showed respectively negligible benefit and a `60.6%` V=1 regression; neither remains in the production path;
- a device-route experimental path, fused decode/prefill indexer probes, and full-vocabulary LM-head launch reduction preserved parity but did not materially improve target latency.

The latest V=1 profile is still dominated by stable FP4 MoE and MLA output at roughly `35 ms` each, followed by BF16 rows, HC pre, and BF16 compressor work in the `19–26 ms` range. Even with zero expert I/O, `225.999 ms` remains about `3.6×` above the `62.5 ms/token` complete-cycle budget, so resident small-M kernel efficiency remains an unavoidable release blocker. The implementation order is nevertheless to land resumable I/O first: further all-hit wins cannot overlap a cold miss while the model worker remains blocked.

Work:

- execute all-hit routed layers without D2H synchronization or host materialization;
- prototype artifact-preserving resident FP4 weight layouts or grouped small-M kernels that raise stable-MoE bandwidth without weakening generation/route transaction checks;
- continue reducing MLA and BF16-compressor small-M time only with bitwise differential tests and full-pass wins;
- freeze graph-stable descriptors and ping-pong workspaces only after continuation resume points are stable;
- capture graph buckets by forward mode and padded capacity, not by model-plan M variants;
- do not restore the rejected MLA output-B split, wider-grid experiment, device-route fast path, predictive prefetch, or resident/miss dual-MoE path;
- reprofile the final semantic path after graph capture;
- rerun the complete R2b cycle gate rather than reporting launch-count reduction alone.

Exit gate:

- resident layers wake no host service;
- graph replay preserves numerical and transaction parity;
- graph capture improves complete-cycle latency rather than only launch count.

### Phase R5 — Ingest, serving, and release dossier

Purpose: turn the viable cycle into a reproducible public result.

Work:

- complete static safetensors extent/scatter ingest without duplicate full-weight layouts;
- run warm/cold TTFT, ITL, throughput, and long-output serving matrices;
- freeze model, tokenizer, prompt, sampling, server, client, source, and binary manifests;
- identify eligible competing runtimes and run the same contract;
- publish raw JSON, profiler reports, logs, hashes, and aggregation scripts.

Release gate:

- lower 95% confidence bound of warm accepted throughput is at least 16 tok/s;
- zero headline request failures;
- parity and near-tie corpus pass;
- memory remains bounded;
- Ferrule exceeds the strongest eligible competitor beyond measurement noise.

## 5. Benchmark commands

Implemented:

```bash
just cutlass-setup
just build-cuda sm_121a
just test-cutlass-provider sm_121a
just test-runtime
just test-server
just test-cli
just dsv4-prefill-parity
just dsv4-verify-width-sweep
just dsv4-serve
just dsv4-vllm-bench
```

Planned and not yet valid release commands:

```text
just dsv4-route-cache-oracle
just dsv4-static-ingest-ab
just dsv4-release-manifest
just dsv4-release-suite
```

A phase is complete only when its correctness and measured exit gates pass. Compiling code, reducing API calls, or winning a microbenchmark is not phase completion.

## 6. Required counters

### Target and DSpark

- checkpoint `gamma`, requested proposal slots, target verification rows, and graph tier;
- proposal count and immutable source identity/hash;
- proposed tokens, verified rows, accepted draft tokens, correction/bonus tokens, externally committed tokens, and rolled-back rows;
- accepted-draft-prefix histogram and mean externally committed `C(Q)`;
- draft/verify/prefix/commit/rollback latency;
- complete cycle latency and externally committed output tok/s;
- internal/external committed-token reconciliation.

### GPU

- semantic operation counts/time and physical CUDA kernel launches/time;
- graph launches and replays;
- allocation count and workspace bytes;
- H2D/D2H bytes and synchronization count;
- provider manifest, ABI, CUTLASS commit, and SM target.

### Experts and I/O

- selected, resident-hit, inflight-hit, and cold-miss experts;
- unique expert union per cycle;
- exact logical dependencies, physical load operation IDs, inflight joins, and waiter cohorts;
- requested/read/uploaded/cancelled/failed bytes and aligned-read amplification;
- SQE, pinned-slab, upload-slot, expert-frame, lease, and ready-cohort credit high watermarks;
- read, upload, publish, wait, resume, and uncovered critical-path time per externally committed output token;
- slot generations, evictions, cancellation, wakeups, and stale-binding failures.

### Memory and serving

- host, pinned, device, page-cache, and swap peaks;
- TTFT, ITL/TPOT, request/output throughput, and failure count;
- queue, admission, cancellation, and deadline counters.

## 7. Final checklist

### Feasibility

- [x] F1 target-only packed roofline/parity evidence at Q=1/2/4/6/8.
- [x] Checkpoint-native proposal, target-row, and correction/bonus protocol contract frozen; target-only path measured.
- [x] F1 checkpoint-native target-only path establishes resident-compute feasibility.
- [x] Production OpenAI/vLLM path executes real CUDA DSpark proposal, exact target transaction, and external token streaming.
- [x] First production checkpoint-native trace captured; it fails F2 throughput and F3 I/O budget.
- [x] Non-zero real DSpark acceptance and exact externally committed-token reconciliation.
- [x] True multi-sequence ragged target verification executes as one production cohort.
- [x] Bounded decode-cohort formation produces packed serving work while preserving forced progress.
- [ ] F2 complete-cycle viability at `>=16 tok/s`.
- [ ] F3 resumable exact expert I/O, overlap, and bounded capacity.

### Exactness

- [x] 43-layer packed CUDA parity.
- [x] Exact target branch/commit/rollback transaction tests.
- [x] Arbitrary-width per-sequence prefix retention and all-or-nothing cohort page validation/commit.
- [x] Packed prompt/verification updates provisional DSpark context state, and the production cohort retains accepted prefixes without replay.
- [x] Production server uses the single real DSpark proposal/verify/commit path rather than ordinary one-token decode.
- [ ] Official Python fixture parity for target taps, `main_x`, stage outputs, proposal logits/tokens, and confidence.
- [ ] Full official DSpark block/logit/confidence proposal parity; CUDA proposal-head parity is proven, first four selective-official greedy tokens match, and the fifth FP8 near-tie remains.
- [ ] Long-output, cancellation, EOS, and near-tie release corpus.

### Performance

- [x] Checkpoint-native target path establishes the resident target-only feasibility checkpoint.
- [x] Persistent pinned metadata mirrors remove repeated staging and synchronization from packed verification.
- [x] Router-control transfer overlaps the fused shared FFN.
- [x] Existing server/vLLM recipes execute the CUTLASS semantic bundles in the production DSpark cycle.
- [x] Production serving forms real multi-request decode cohorts under bounded scheduler deferral.
- [x] Linux expert streaming uses `O_DIRECT + io_uring` registered pinned slabs with no expert mmap backend.
- [x] Owned arena checkout/checkin provides the lifetime base for suspended execution.
- [ ] Layer execution can suspend at an exact pre-MoE dependency boundary and resume without replay.
- [ ] A global CQE/H2D/CUDA-event reactor publishes validated residency and wakes waiter cohorts.
- [ ] Production profiling proves CPU, GPU, memory, and NVMe overlap and reconciles the complete critical path.
- [ ] Device all-hit, graph, and further DSpark-specific CUTLASS work recover enough complete-cycle budget to meet the release gate.
- [ ] Complete externally committed throughput lower 95% confidence bound reaches 16 tok/s.
- [ ] Device all-hit and graph paths win end to end.
- [ ] Memory remains bounded with zero headline failures.

### Reproducibility

- [x] CUTLASS version and commit pinned with automatic setup.
- [ ] Immutable model/tokenizer/workload/source/binary manifest.
- [ ] Raw F2/F3/serving/competitor artifacts published.
- [ ] Exact release commands and aggregation scripts published.

### SOTA claim

- [ ] Strongest eligible competitors identified.
- [ ] Same-contract runs preserved.
- [ ] Ferrule wins beyond measurement noise.
