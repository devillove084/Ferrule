# Ferrule single-DGX Spark release roadmap

> Canonical engineering and release plan for exact DeepSeek-V4-Flash-DSpark inference on one NVIDIA DGX Spark / GB10.
>
> Updated: 2026-07-20.

## 0. Current decision

The project remains **GO as an engineering program**, but complete-cycle viability is still far below the release gate and no SOTA claim is valid. Real checkpoint-native acceptance is now non-zero and transactionally reconciled, so the previous zero-acceptance blocker is closed as a correctness-localization problem. The remaining gap is dominated by target compute, expert movement, and transaction structure rather than proposal execution.

The checkpoint-native DSpark protocol is frozen from the DeepSeek reference model, official DeepSpec implementation, and serving integrations. The prepared GB10 path implements the real proposal backbone and heads, exact packed target verification, acceptance-aware branch commit/rollback/replay, and correction/bonus staging using the production CUDA/CUTLASS execution image.

Selective official-Python comparison loads the three DSpark stages and shared LM head on GB10. Ferrule matches the official greedy proposal for the first four positions on the frozen fixture; the fifth position remains a near-tie numerical divergence in the FP8 activation/GEMM path. Independently, a CPU reconstruction from checkpoint host payloads matches all five CUDA proposal-head tokens and confidence logits within `4.05e-6`, localizing the remaining parity work before the proposal head rather than in Markov selection.

The critical path is now:

1. close the remaining fifth-position FP8 backbone parity gap and freeze the selective official near-tie fixture;
2. carry one evolving expert-I/O plan through proposal, one-row probe, possible full verification, and replay, reconciling scheduler estimates with physical loads;
3. reduce the resident target path itself: one zero-I/O 43-layer target pass with LM head is currently about `289 ms`, so I/O elimination alone cannot meet the `62.5 ms/token` release budget;
4. remove all-hit router host barriers, improve small-M attention/MoE efficiency, eliminate hot-path allocation, and capture stable graph buckets only after exact restart boundaries exist;
5. rerun the frozen warm vLLM serving suite, then the same-contract competitor suite.

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
- optimized generic FP8 and BF16 matrix paths;
- fused FP8 QueryA and KV production;
- fused BF16 compressor work;
- fused shared FFN and MLA output stages;
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

- a model-neutral `DsparkProposalRunner` capability executes against authoritative per-session state;
- native-width verification and shorter tails use the same semantic packed target path;
- full acceptance promotes the provisional branch, while rejection rolls it back and replays the accepted frontier;
- scheduler metadata, externally emitted tokens, incremental text, limits, EOS, stop strings, correction/bonus staging, cancellation, and failure cleanup reconcile in the production driver;
- the OpenAI server no longer uses ordinary one-token target decode for DeepSeek-V4 headline serving;
- runtime transaction tests cover full, partial, and zero acceptance, including the single-row rejection replay.

Correctness status:

- the real proposal is deterministic within one numerical target path and does not mutate committed DSpark context during proposal;
- server integration exposed missing packed-context updates and missing single-row semantic replay support; both defects were corrected;
- a frozen 12-prompt greedy production sweep observed 16 accepted draft tokens across 68 cycles, including accepted prefixes of 1, 2, 3, and 4 tokens; all 84 externally committed tokens reconcile with runtime callbacks and SSE usage;
- first-miss cycles now verify and commit only the anchor row, while first-hit cycles roll the probe back before the existing atomic packed transaction; unit tests cover both physical transaction shapes;
- CPU checkpoint reconstruction proves the CUDA proposal head for all five positions, while selective official Python agrees on the first four greedy tokens; the fifth near-tie remains a pre-head FP8 numerical parity gap;
- full official proposal/logit/confidence parity is therefore **not yet established**; immutable proposal identity/hash and a frozen official near-tie fixture remain required.

### 3.5 I/O checkpoint

The platform has `O_DIRECT + io_uring` reads into registered pinned slabs and stable expert slot/generation/lease semantics. GB10 direct NVMe-to-GPU GDS is not claimed: platform checks reported the GPU model and P2PDMA path unsupported, so compatibility mode is only a baseline.

The acceptance-aware one-row probe is a measured improvement, but Gate F3 still fails. Against the same 12-prompt contract it preserved every proposal, accepted prefix, correction, output token, and normalized SSE payload while reducing target verification rows from 283 to 114, rolled-back rows from 264 to 47, expert movement from `321.36 GB` to `152.20 GB`, transaction time from `66.20 s` to `36.08 s`, and aggregate wall time from `92.00 s` to `59.32 s`. Output throughput increased from `0.913` to `1.416 tok/s`.

A 96-expert per-layer hotset further reduced expert movement to `92.96 GB` and raised throughput to `1.618 tok/s`; 128 experts reduced movement again to `72.86 GB` but reached only `1.649 tok/s`, so capacity alone has sharply diminishing end-to-end returns. Production serving now defaults to the measured 96-expert point. Scheduler decode admission still sees an anchor-token estimate rather than the realized proposal, probe, target, and replay union, so prediction, admission, physical accounting, and residency must still converge on one shared cycle plan.

### 3.6 Production OpenAI/vLLM serving checkpoint

Completed:

- the existing OpenAI-compatible server and official vLLM serving benchmark exercise the production DSpark path;
- no synthetic proposal source, target-only driver, or CPU fallback is used for serving results;
- proposal, verification, transaction, expert residency/I/O, CUDA/runtime, and externally emitted-token telemetry are collected from the real cycle;
- packed prompt/verification now updates DSpark context state inside the provisional transaction;
- rejection replay uses the same semantic packed CUDA path, including the single-row tail;
- production smoke requests complete through OpenAI streaming without introducing a second server or benchmark driver;
- runtime, server, CLI, CUDA build, and CUTLASS provider validation pass through the project `justfile`.

The production sweep now proves non-zero acceptance and exact external reconciliation, but end-to-end viability remains open. The accepted-prefix histogram is `0:58, 1:7, 2:1, 3:1, 4:1`; first-token agreement is `14.71%`. After acceptance-aware verification and the 96-expert hotset, measured throughput is still only `1.618 tok/s`.

Forced zero-I/O evidence also rules out an I/O-only explanation: a resident 43-layer target pass with LM head takes about `288.97 ms` (`3.46 pass/s`) and launches 1,737 kernels. Packed target-only V=2/4/6/8 reaches `7.45/13.17/17.73/21.71 rows/s`, but checkpoint V=6 costs about `338.34 ms`; these are computed rows, not externally committed throughput. Complete-cycle R2b therefore requires R4 small-M/device-path work in addition to R3 I/O repair. Raw request, cycle, physical-counter, SSE, and comparison artifacts remain under `target/bench/dspark-acceptance-sweep-*`.

## 4. Remaining execution plan

### Immediate next pass — authoritative full-cycle profile

The next optimization pass must instrument the existing `just dsv4-serve` → `just dsv4-vllm-bench` production path. It must not introduce a second server, model driver, transaction implementation, or headline benchmark path.

Every request cycle receives one immutable request/session/cycle-attempt identity that follows it through HTTP handling, scheduler decisions, logical expert dependencies, physical loads, CUDA submissions, transaction state, emitted tokens, and the vLLM result artifact. Host spans use monotonic timestamps; device spans use CUDA events on the owning streams. Profiling must not add a device-wide synchronization to obtain timings, and overlapping spans must report both wall-clock critical path and per-service busy time rather than being summed into a false serial total.

Required decomposition:

- **Request and server:** HTTP parse, chat-template/tokenizer input work, request admission wait, scheduler queue wait, incremental detokenization, token callback/SSE enqueue and flush, cancellation, and final response completion.
- **Scheduler:** candidate classification, route prediction, predicted incremental expert union/bytes, hard-credit decision, resident-ready versus miss-blocked classification, selected plan identity, wake reason, and actual-versus-predicted reconciliation.
- **Prefill:** embedding; each target layer's HC, attention, router, fused shared FFN, and routed MoE; output head; target tap capture; fused `main_proj/main_norm`; and DSpark context-KV publication.
- **Proposal:** resident embedding gather; for each execution stage 43–45, HC pre, hybrid attention, router, fused shared FFN, routed MoE, and HC post; final HC head; final norm/base LM head; sequential Markov embedding/bias and token selection; confidence projection; compact result transfer; and proposal expert I/O.
- **Verification:** provisional branch fork, KV page reserve/bind, packed Q-row target layers, per-layer exact target expert union, output head, accepted-prefix comparison, and the effective Q width.
- **Transaction:** full promotion, rollback, correction replay, runtime page commit, backend page commit, DSpark context commit, externally committed frontier, and state/resource release.
- **Physical I/O:** scheduler-admitted bytes; logical requested, aligned physical read, uploaded, shared/joined, cancelled, and rejected-prefetch bytes; fixed-file/slab queue wait; `O_DIRECT` read time; pinned-slab wait; upload wait; residency publication; eviction; and bytes charged per externally emitted token.
- **GPU/runtime:** kernel and graph launch/replay counts; per-stream CUDA-event durations; H2D/D2H operation counts and bytes; stream/event waits; stream-wide synchronizations; allocation count; arena/workspace high watermarks; graph-bucket identity; and compute/transfer interference.

The profile is complete only when:

1. cold-start, warmed-cache, and forced all-hit runs are reported separately under the same frozen request contract;
2. proposal, target, replay, and physical expert operations are charged to one evolving cycle plan, with predicted and exact per-segment unions distinguished rather than pretending dynamically routed experts are known early;
3. scheduler-admitted bytes reconcile with physical loads through an explicit ledger for cache hits, inflight joins, alignment, sharing, cancellation, eviction, and rejected prefetch;
4. internal proposal/accept/commit counters reconcile exactly with externally returned vLLM tokens and EOS/stop behavior;
5. vLLM E2E wall time reconciles with server queueing plus production cycle critical paths, with no unexplained remainder and no double-counting of overlap;
6. raw machine-readable spans, counters, source/model/config identities, server log, and vLLM result JSON are preserved together in one artifact directory.

The first use of this profile is correctness localization for zero acceptance, followed by I/O/scheduler residency repair, and only then all-hit graph/kernel optimization.

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
- preserve route union, rejected-prefetch, and KV transaction traces;
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

Purpose: keep cold expert traffic inside the storage and capacity envelope.

Work:

- preserve the landed acceptance-aware restart boundary: one target row on first miss, with full packed verification only after a first-token hit;
- carry one authoritative, evolving cycle plan across proposal, target Q verification, and possible replay, separating predicted route unions from exact per-segment unions at the legal restart boundaries;
- connect route/cache prediction to admitted marginal unique expert objects and bytes;
- reconcile scheduler-admitted bytes with actual shared physical loads, inflight joins, cache hits, alignment, cancellation, and evictions;
- charge rejected-prefetch bytes to the cycle that caused them;
- separate resident-ready work from miss-blocked work;
- enforce fixed-file, slab, upload, and device-frame credits;
- compare current policy with route-trace oracle/Belady bounds;
- run long-output and memory-pressure tests.

Exit gate:

- uncovered reads remain within the acceptance-aware storage budget at the release operating point;
- no swap or page-cache collapse;
- expert memory and temporary workspace remain bounded;
- every miss, stale generation, and failed upload remains exact and recoverable.

### Phase R4 — Device all-hit path and stable graphs

Purpose: remove residual host work from resident cycles.

Measured starting point: the forced all-hit 43-layer target plus LM head is about `288.97 ms` with 1,737 launches. A device-route experimental path and fused decode/prefill indexer probes preserved parity but did not materially improve complete target latency. Full-vocabulary LM-head chunking removes 93 launches and gives only a small width-dependent gain, confirming that attention/MoE small-M efficiency is the primary R4 target.

Work:

- execute all-hit routed layers without D2H synchronization or host materialization;
- keep miss side effects behind an explicit restart/rollback boundary;
- freeze graph-stable descriptors and ping-pong workspaces;
- capture graph buckets by forward mode and padded capacity, not by model-plan M variants;
- reprofile the final superkernel path after graph capture;
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

- semantic-kernel launches and time;
- graph launches and replays;
- allocation count and workspace bytes;
- H2D/D2H bytes and synchronization count;
- provider manifest, ABI, CUTLASS commit, and SM target.

### Experts and I/O

- selected, resident-hit, inflight-hit, and cold-miss experts;
- unique expert union per cycle;
- requested/read/uploaded/rejected-prefetch bytes;
- uncovered I/O time and bytes per externally committed output token;
- slot generations, evictions, waits, and stale-binding failures.

### Memory and serving

- host, pinned, device, page-cache, and swap peaks;
- TTFT, ITL/TPOT, request/output throughput, and failure count;
- queue, admission, cancellation, and deadline counters.

## 7. Final checklist

### Feasibility

- [x] F1 target-only packed roofline/parity evidence at Q=2/4/6/8.
- [x] Checkpoint-native proposal, target-row, and correction/bonus protocol contract frozen; target-only path measured.
- [x] F1 checkpoint-native target-only path establishes resident-compute feasibility.
- [x] Production OpenAI/vLLM path executes real CUDA DSpark proposal, exact target transaction, and external token streaming.
- [x] First production checkpoint-native trace captured; it fails F2 throughput and F3 I/O budget.
- [x] Non-zero real DSpark acceptance and exact externally committed-token reconciliation.
- [ ] F2 complete-cycle viability at `>=16 tok/s`.
- [ ] F3 acceptance-aware I/O and capacity.

### Exactness

- [x] 43-layer packed CUDA parity.
- [x] Exact target branch/commit/rollback transaction tests.
- [x] Packed prompt/verification updates provisional DSpark context state, and rejection replay supports Q=1 on the semantic CUDA path.
- [x] Production server uses the single real DSpark proposal/verify/commit path rather than ordinary one-token decode.
- [ ] Official Python fixture parity for target taps, `main_x`, stage outputs, proposal logits/tokens, and confidence.
- [ ] Full official DSpark block/logit/confidence proposal parity; CUDA proposal-head parity is proven, first four selective-official greedy tokens match, and the fifth FP8 near-tie remains.
- [ ] Long-output, cancellation, EOS, and near-tie release corpus.

### Performance

- [x] Checkpoint-native target path establishes the resident target-only feasibility checkpoint.
- [x] Persistent pinned metadata mirrors remove repeated staging and synchronization from packed verification.
- [x] Router-control transfer overlaps the fused shared FFN.
- [x] Existing server/vLLM recipes execute the CUTLASS semantic bundles in the production DSpark cycle.
- [ ] Authoritative full-cycle profiling fully explains production latency across scheduler, I/O, proposal, verification, transaction, and serving; production cycle/physical/SSE reconciliation and the first before/after decomposition are complete.
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
