# Ferrule single-DGX Spark release roadmap

> Canonical engineering and release plan for exact DeepSeek-V4-Flash-DSpark inference on one NVIDIA DGX Spark / GB10.
>
> Updated: 2026-07-16.

## 0. Current decision

The project remains **GO**, but no SOTA claim is currently valid.

The resident target-compute feasibility question has changed materially: the GB10 semantic-superkernel path now verifies V=8 at 16.1958 rows/s p50, with parity, zero selected expert I/O, and zero steady-state allocation. Gate F1 therefore no longer rejects the 16 tok/s headline on resident compute alone.

The critical path is now:

1. connect the real MTP proposal source to the prepared GB10 execution image;
2. run the complete exact DSpark transaction and measure real acceptance;
3. enforce the acceptance-aware expert-I/O budget;
4. remove resident host barriers and capture stable graphs;
5. run the frozen serving and competitor release suite.

## 1. Release contract

### 1.1 North star

Ferrule releases the single-DGX Spark milestone only when one GB10 sustains:

```text
>= 16 accepted output tokens/s
```

under a frozen warm workload using:

- the exact `models/DeepSeek-V4-Flash-DSpark` checkpoint;
- exact target verification and real router/expert execution;
- the production MTP proposal source;
- complete draft, verify, commit/rollback, and uncovered-I/O timing;
- bounded memory with no swap or page-cache collapse;
- zero headline request failures;
- reproducible source, binary, model, environment, and benchmark manifests.

Verified rows/s, target passes/s, kernel TFLOP/s, and isolated microbenchmarks are evidence, not the final metric.

### 1.2 Release equation

For verification width `V`, mean committed tokens `A(V)`, and complete cycle latency `T_cycle(V)`:

```text
accepted tok/s = A(V) / T_cycle(V)
```

`T_cycle(V)` includes:

```text
draft proposal
+ exact packed target verification
+ accepted-prefix computation
+ commit or rollback
+ uncovered expert I/O
```

The release gate is:

```text
A(V) / T_cycle(V) >= 16
```

### 1.3 I/O budget

The measured storage baseline is approximately 10.53 GiB/s. At 16 accepted tok/s, long-run uncovered reads must satisfy:

```text
bytes per accepted token <= 10.53 / 16 = 0.658 GiB
```

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

The CUTLASS manifest currently publishes six semantic superkernels:

1. FP8 QueryA + KV;
2. BF16 compressor;
3. HC producer;
4. shared FFN;
5. stable-frame routed MXFP4 MoE;
6. MLA OutputA → latent pack → OutputB.

Routing, sparse attention, paged control, and recurrent metadata may remain dedicated CUDA operations where fusion has not demonstrated an end-to-end win.

## 3. Verified evidence

### 3.1 Gate F1 resident verification

Artifact:

```text
target/bench/gate-f1/verify-width-sweep-superkernels-3iter.json
```

| V | p50 `T_verify` | p95 `T_verify` | p50 verified rows/s |
|---:|---:|---:|---:|
| 2 | 0.433132 s | 0.448342 s | 4.6175 |
| 4 | 0.418125 s | 0.419454 s | 9.5665 |
| 8 | 0.493955 s | 0.497114 s | **16.1958** |

All measured widths reported parity, resident/no-I/O execution, and zero steady-state allocation.

Interpretation:

- **passed:** F1 headline viability through V=8;
- **failed:** V4/A4 250 ms operating point;
- **failed:** provisional 200 ms verification allowance for a 50 ms non-verification budget;
- **not measured by F1:** real acceptance and complete accepted tok/s.

### 3.2 Operator optimization checkpoint

The current GB10 path uses semantic fusion rather than host-side composition of small GEMMs. Selected improvements include:

| Kernel work | Before | Current checkpoint |
|---|---:|---:|
| routed FP4 MoE | 40.86 ms/layer | approximately 2.83 ms/layer |
| generic FP8 MMA | 0.844 ms | 0.290 ms |
| generic BF16 | 1.085 ms | 0.340 ms |
| FP8 QueryA + KV | 0.980 ms | 0.163 ms |
| BF16 compressor pair | 0.669 ms | 0.339 ms |

See `docs/GB10_SUPERKERNEL_JOURNEY.md` for the engineering narrative and rejected experiments.

### 3.3 Correctness checkpoint

Verified commands include:

- kernel-plan unit tests;
- CUTLASS provider ABI/semantic tests;
- dynamic-M coverage including 4,097 rows;
- MXFP4 MoE and FP8/BF16 MMA smoke tests;
- 43-layer packed CUDA versus token-loop CUDA parity.

The latest 43-layer parity run reported `max_abs_diff=0` at every layer and matching cut points 1/5/23/43.

### 3.4 DSpark transaction checkpoint

Implemented in the generic runtime:

- packed `1 sequence × V rows` exact verification;
- causal accepted-prefix computation;
- physical/logical KV branch handling;
- full-accept promotion;
- partial-accept replay;
- zero-accept rollback;
- internal transaction tests for full, partial, and zero acceptance.

Not yet implemented in production:

- MTP proposal execution in the prepared GB10 image;
- prediction-head execution and proposal identity/hash;
- end-to-end accepted-prefix histogram and `A(V)`;
- complete draft/verify/commit/rollback timing from a real proposal source.

### 3.5 I/O checkpoint

The platform has `O_DIRECT + io_uring` reads into registered pinned slabs and stable expert slot/generation/lease semantics. GB10 direct NVMe-to-GPU GDS is not claimed: platform checks reported the GPU model and P2PDMA path unsupported, so compatibility mode is only a baseline.

Gate F3 still needs acceptance-aware long-run measurements against the 0.658 GiB/accepted-token budget.

## 4. Remaining execution plan

### Phase R1 — Production MTP proposal path

Purpose: make the real DSpark attachment executable on GB10.

Work:

- load MTP layers and prediction heads into the single-owner prepared resources;
- assign non-aliasing execution-layer identities after the 43 target layers;
- compile MTP weights, workspaces, expert catalogs, and kernel plans into the GB10 execution image;
- implement the stage-zero main projection/norm contract;
- execute MTP transformer stages with explicit sequence/KV state;
- implement HC head, final norm, Markov head, confidence head, and proposal token output;
- expose an immutable proposal-source identity and hash;
- fail if any MTP plan, tensor, expert frame, or target capability is missing.

Exit gate:

- real checkpoint proposes deterministic V=2/4/8 token blocks;
- proposal execution is CUDA-only and allocation-free after prepare;
- proposal parity and state-transition tests pass;
- no target-model or synthetic proposal source is used for headline data.

### Phase R2 — Gate F2 real acceptance

Purpose: determine whether real acceptance converts the resident V=8 compute result into a viable complete cycle.

Work:

- add a production F2 benchmark entry;
- measure draft, verify, accepted-prefix, commit, rollback, and uncovered-I/O components;
- record accepted-prefix histograms and `A(V)` for V=2/4/8;
- reconcile internal committed tokens with externally returned tokens;
- preserve route union, rejected-prefetch, and KV transaction traces;
- run full-, partial-, zero-accept, EOS, cancellation, and failure cases.

Exit gate:

```text
A(V) / T_cycle(V) >= 16
```

for at least one viable width, or a measured optimization budget identifies a credible path. If every viable width is below 16 with no credible budget, the current release target is rejected.

### Phase R3 — Gate F3 acceptance-aware expert I/O

Purpose: keep cold expert traffic inside the storage and capacity envelope.

Work:

- connect route/cache prediction to admitted incremental expert bytes;
- charge rejected-prefetch bytes to the cycle that caused them;
- separate resident-ready work from miss-blocked work;
- enforce fixed-file, slab, upload, and device-frame credits;
- compare current policy with route-trace oracle/Belady bounds;
- run long-output and memory-pressure tests.

Exit gate:

- uncovered reads remain at or below 0.658 GiB per accepted token at the release operating point;
- no swap or page-cache collapse;
- expert memory and temporary workspace remain bounded;
- every miss, stale generation, and failed upload remains exact and recoverable.

### Phase R4 — Device all-hit path and stable graphs

Purpose: remove residual host work from resident cycles.

Work:

- execute all-hit routed layers without D2H synchronization or host materialization;
- keep miss side effects behind an explicit restart/rollback boundary;
- freeze graph-stable descriptors and ping-pong workspaces;
- capture graph buckets by forward mode and padded capacity, not by model-plan M variants;
- reprofile the final superkernel path after graph capture.

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
just build-cuda
just test-cutlass-provider
just dsv4-prefill-parity
just dsv4-verify-width-sweep
```

Planned and not yet valid release commands:

```text
dsv4-dspark-cycle-sweep
dsv4-route-cache-oracle
dsv4-static-ingest-ab
dsv4-release-manifest
dsv4-release-suite
```

A phase is complete only when its correctness and measured exit gates pass. Compiling code, reducing API calls, or winning a microbenchmark is not phase completion.

## 6. Required counters

### Target and DSpark

- verification width and rows/cycle;
- proposal count and immutable source identity;
- accepted-prefix histogram and `A(V)`;
- draft/verify/prefix/commit/rollback latency;
- complete cycle latency and accepted tok/s;
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
- uncovered I/O time and bytes per accepted token;
- slot generations, evictions, waits, and stale-binding failures.

### Memory and serving

- host, pinned, device, page-cache, and swap peaks;
- TTFT, ITL/TPOT, request/output throughput, and failure count;
- queue, admission, cancellation, and deadline counters.

## 7. Final checklist

### Feasibility

- [x] F1 headline viability via resident V=8 target verification.
- [ ] F2 real MTP acceptance.
- [ ] F3 acceptance-aware I/O and capacity.

### Exactness

- [x] 43-layer packed CUDA parity.
- [x] Exact target branch/commit/rollback transaction tests.
- [ ] Real MTP proposal parity.
- [ ] Long-output, cancellation, EOS, and near-tie release corpus.

### Performance

- [x] Resident V=8 p50 and p95 verified-row rates exceed 16 rows/s.
- [ ] Complete accepted throughput lower 95% confidence bound reaches 16 tok/s.
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
