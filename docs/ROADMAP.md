# Ferrule Roadmap

_Last updated: 2026-07-08_

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. The current
engineering center is **interactive, resident, CUDA-backed DeepSeek-V4 Flash /
DSpark execution** on GB10-class unified-memory machines, while **OLMoE** remains
the correctness golden model.

The working target is no longer only an offline `decode_tok/s` number. Ferrule
must optimize the user-visible path:

1. `./target/release/ferrule chat ...` reaches the prompt quickly;
2. every user turn appends new prompt tokens without redoing old prefix work;
3. the first response token and the full response arrive with minimal latency;
4. benchmark JSON still reports load / prefill / decode counters for regression
   tracking.

DSV4 drives missing runtime capabilities but must not become a model-specific
runtime architecture. Model-family names and raw HF tensor names stay in
`ferrule-model`; `ferrule-runtime` sees semantic roles, graph ops, artifact
groups, expert registries, KV state, residency policy, and backend object stores.

---

## Table of Contents

1. [Current capabilities](#current-capabilities)
2. [Architecture principles](#architecture-principles)
3. [Composable engine architecture](#composable-engine-architecture)
4. [Priority roadmap](#priority-roadmap)
5. [Implementation order](#implementation-order)
6. [What not to do](#what-not-to-do)
7. [Gap against mainstream engines](#gap-against-mainstream-engines)
8. [Model bring-up contract](#model-bring-up-contract)
9. [DSV4 artifact facts](#dsv4-artifact-facts)
10. [DSV4 synthetic fixture benchmark](#dsv4-synthetic-fixture-benchmark)
11. [GB10 hardware analysis and measured state](#gb10-hardware-analysis-and-measured-state)
12. [Review checklist](#review-checklist)

---

## Current capabilities

### Working now

- **OLMoE correctness fixture**: safetensors loading, GPU Q4_0 inference through
  cuda-oxide kernels, and router correctness around `norm_topk_prob`. CPU
  reference components remain the anchor for artifact decoders, router math, HC
  math, expert execution, and CUDA tests.
- **DeepSeek V4 / DSpark metadata and local artifacts**:
  - local model: `models/DeepSeek-V4-Flash-DSpark`;
  - `hidden_size=4096`, `moe_intermediate_size=2048`, `num_hidden_layers=43`;
  - `n_routed_experts=256`, `num_experts_per_tok=6`, routed expert dtype `fp4`;
  - DSpark metadata is parsed: `dspark_block_size=5`, noise token, target layers
    `[40, 41, 42]`, Markov rank `256`.
- **Full 43-layer DSV4 CUDA greedy path**:
  - `deepseek-v4-generate --backend cuda --json` reports load / prefill / decode
    timing and CUDA counters;
  - `ferrule chat ... -q cuda --chat-template deepseek-v4 --temp 0` runs the
    DSV4 greedy top-k fast path;
  - `lm_head` top-k can stay device-side for greedy decode.
- **Interactive DSV4 chat improvements (2026-07-07)**:
  - chat REPL becomes ready immediately while DSV4 artifacts load on a background
    thread;
  - CUDA context/operator cache is still created on the main thread;
  - prompt append uses `prefill_tokens_topk_interactive`: all prompt tokens except
    the last update KV/MoE/session state without materializing final hidden/logits;
    the last prompt token materializes top-k for generation;
  - `feed_token` now advances session state without unnecessary `hc_head`, output
    norm, or `lm_head` work.
- **DSV4 CUDA execution state**:
  - device-resident decode chain exists for the hot single-token path;
  - attention/KV append, norm weight cache, GPU output-head/top-k, batched routed
    MoE workspace, managed-memory FP4 expert handles, and mmap shard reads are in
    the working path;
  - `FERRULE_CUDA_MOE_TC=1` (default) enables the FP4 `mxf4` Tensor Core path;
    `FERRULE_CUDA_MOE_TC=0` forces scalar FP4 fallback for A/B.
- **Blackwell FP4 Tensor Core smoke**:
  - local `fp4_mxf4_smoke` proves
    `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale...ue8m0` on
    `sm_121a` with correct full-tile scale selection;
  - the integrated MoE path uses packed FP4 activations + E8M0 scales and has a
    fused `SwiGLU -> packed FP4` stage, eliminating the previous hidden-pack pass.
- **Expert residency / storage**:
  - `ExpertStreamingPlanner` tracks per-layer hot expert frequency and can
    prefetch routing-aware hot experts instead of naive low expert IDs;
  - bounded hotset residency (`--moe-hotset-experts`) exists for constrained
    machines but is not the fastest short-run default;
  - `HostStagedExpertCache`, mmap shard cache, and managed expert allocations are
    wired into CUDA DSV4.
- **Runtime graph foundation**: `ferrule-runtime::graph` opaque IR,
  `GraphProgram`, `ExecutionBatch`, `ExecutionOutput`, `ExternalBindingPlan`,
  `BackendObjectStore`, `ReferenceGraphBackend`, and shape validation registry.
- **Resident scheduling spine**: `ResidentScheduler`, `SequenceState`,
  `SchedulerAction`, `ResidentActionExecutor`, `ResidentTopKDriver`, and
  `PagedSequenceKvCache` now form the generic request → prefill/decode action →
  execution output → token streaming → finish/free-KV loop for tests and future
  serving.
- **DSV4 runtime-driver proof (2026-07-08)**: local full 43-layer DSV4 Flash / DSpark
  runs through `bench-interactive --runtime-driver` with `runtime_path =
  "resident_topk_driver"`. The runner remains in `ferrule-model`; runtime sees only
  `TopKModelRunner` plus scheduler/KV/action vocabulary. `PagedSequenceKvCache` is
  metadata/lifecycle only for this path; physical CUDA KV still lives in the DSV4 runner.
- **Model-family boundary**: `ferrule-model/src/families/` owns per-family HF
  tensor parsing. DSV4 raw tensor names are isolated in `families/deepseek_v4.rs`.
- **Storage/residency vocabulary** (`ferrule-runtime::storage`):
  `StorageObjectId`, `ObjectLocator`, `Placement`, `ObjectReplica`,
  `ReplicaHandleId`, `StorageCatalog`, `TransferEngine`, and
  frequency-aware scoring.
- **Quantization / artifacts**: Q4_0/Q8_0 full path, mixed precision policy,
  FP4 E2M1 + E8M0 artifact decoders, FP8 E4M3FN decoders, KV quantization, and
  WeightPack reader/writer for OLMoE.
- **CLI surface**: `info`, `cuda`, `chat`, `bench-interactive`,
  `inspect-weightpack`, `expert-stream-smoke`, `deepseek-v4-probe`, and
  `deepseek-v4-generate`. Serving remains a runtime design target, not an
  active CLI command in the current workspace.

### Main gaps

Current measurements are on GB10 / `sm_121a` with the local DSV4 Flash DSpark
artifact unless noted. The most important gap is **end-to-end interactive
latency caused by model execution shape**, not the generic runtime scheduler.

| Area | Current known state | Gap |
|---|---|---|
| Chat startup | REPL prints immediately; artifact load observed at ~0.82s in a two-turn pipe run | CUDA/operator warmup still happens on first real prompt |
| Runtime-driver full path | `bench-interactive --runtime-driver`, `Hello`, `-n 1`, `--max-layers 43`: **ttft 15.03s**, prefill **13.82s**, decode **0.825 tok/s** | runtime action count is tiny (`actions=2`, `prefill_tokens=5`, `decode_steps=1`); the bottleneck is inside DSV4 model execution |
| 1-layer vs 43-layer | 1-layer smoke: **ttft 1.45s**, prefill **1.20s**, decode **3.95 tok/s**; 43-layer profile: **ttft 15.03s** | full-depth scaling is dominated by per-layer host/device/model work, not scheduler overhead |
| Prompt/prefill | segment/chunk vocabulary exists and non-final chunks avoid top-k/logits | prefill still crosses host `Vec<f32>` / hidden / HC boundaries and uses token loops in key places |
| MoE prefill | 43-layer profile: MoE outer **6.94s**, CUDA-timed MoE kernels only **0.097s** | prefill MoE is token-level `routed_moe_step`, with expert read/upload/submit dominating instead of layer-level batched columns |
| Expert residency | one 43-layer `Hello` profile loaded **919** experts / **12.286 GB** | needs stronger per-layer resident hotset, prefetch, and reuse across warmup/measured turns |
| Attention output projection | attention total **5.72s**; `output_a` / WO-A is **2.30s**, while sparse attention kernel is **0.18s** | optimize grouped output projection/execution shape before chasing sparse-attention micro-kernels |
| Output head | warm path is no longer the 43-layer bottleneck: 32 cache hits, `lm_head_topk` **0.063s** | cold path still pays read/upload, but it is not the current full-depth limiter |
| Serving spine | `ResidentScheduler` / `ResidentActionExecutor` / `ResidentTopKDriver` run real DSV4 without runtime owning concrete DSV4 types | CUDA multi-session backend, continuous batching, physical paged KV, and prefix cache integration remain |
| DSpark | metadata parsed | no real speculative block decode, verification loop, or rollback-safe KV/residency state |

Important measurement corrections:

- Short `run:` wall-clock timings are misleading; use JSON summary for decode
  throughput and explicit chat/`bench-interactive` stats for interactive latency.
- `FERRULE_DSV4_PROFILE_SYNC=1` and `FERRULE_CUDA_MOE_TIMING=1` insert stream
  synchronization and are for attribution, not headline throughput.
- The current 43-layer bottleneck is not runtime scheduling: the runtime-driver
  path did only two actions for the measured `Hello` request.
- CUDA graph replay cannot pay off while host routing, expert streaming,
  compressor/top-k, and prefill `Vec<f32>` boundaries remain in the hot path.
- Bounded hotset residency can reduce resident handles, but on short runs it can
  increase churn and slow down throughput.
- Duplicating the same hidden vector to fake `dspark_block_size=5` would be wrong;
  real GEMM utilization needs real per-column hidden states and route semantics.

---

## Architecture principles

1. **Keep CPU reference components alive** — artifact decoders, router math,
   HC math, and expert executors are correctness anchors. The legacy full-model
   CPU forward path has been removed; reference is component-level, not a runner.
2. **Model families are policies, not runners** — plug into layout/attention/
   router/quant/residency policies; never fork a runner per family.
3. **No privileged target-model path** — generic crates expose semantic roles
   and policy traits, not DeepSeek-named fields.
4. **Measure before optimizing** — add benchmarks and counters before kernel
   rewrites.
5. **Treat WeightPack as a runtime artifact** — Ferrule's local/edge deployment
   unit with manifest, chunks, checksums, and future placement metadata.
6. **Preserve MoE semantics** — router logits, top-k, expert weights, and
   normalization are correctness-critical and must stay explicit.
7. **Compose behavior through typed policies** — `EnginePlan`/config surface,
   not `if model_name` branches.
8. **Make unsupported execution explicit** — `info` may inspect metadata;
   executable entrypoints such as `chat` or future serving must fail with a named
   missing-policy error.
9. **Async I/O is allowed for expert loading** — the correctness path uses
   synchronous bounded reader; expert weight loading uses tokio + `pread` for
   concurrent disk reads. Planning semantics remain synchronous and unchanged.
10. **PK against real engines with reproducible commands** — every perf claim
    needs prompt, context, quant, batch, hardware, and quality note.

---

## Composable engine architecture

A new model bring-up provides descriptor parsing, tensor layout, and policy
implementations; the runtime composes them into an `EnginePlan`:

```
ModelDescriptor → ModelLayout → ModelSupportContract → EnginePlan {
  ModelFamilyPolicy, AttentionPolicy, RouterPolicy, ExpertPolicy,
  QuantPolicy, KvPolicy, SchedulerPolicy, ParallelismPlan,
  ResidencyPolicy, SpeculationPolicy
}
```

| Policy | Responsibility | Near-term default | Future switches |
|---|---|---|---|
| `ModelFamilyPolicy` | map metadata/tensor names → semantic layer components | OLMoE fixture, DSV4 descriptor | family plugins without runner forks |
| `AttentionPolicy` | MHA/GQA/DeepSeek attention, RoPE, cache layout | GQA RoPE | CSA/HCA/MLA, Flash prefill |
| `RouterPolicy` | logits, bias, hash routing, top-k normalization | dense top-k | bias/hash routing, EP routing |
| `ExpertPolicy` | dense MLP, routed/shared experts, activation | routed experts | shared experts, expert batching, EP |
| `QuantPolicy` | tensor quant class, dequant/matvec, conversion | Q4_0/Q8_0 WeightPack | GGUF K/IQ, FP4/FP8, mixed |
| `KvPolicy` | contiguous/paged/quantized KV ownership | contiguous per-session | paged KV, prefix/radix reuse |
| `SchedulerPolicy` | prefill/decode admission, chunking, batching | single request | continuous batching, preemption |
| `ParallelismPlan` | DP/TP/EP/SP/CP/PP placement | single process/GPU | all combinations |
| `ResidencyPolicy` (model-level) | all-GPU vs streaming allowed | all resident if it fits | expert/layer streaming |
| `StorageResidencyPolicy` (runtime-level) | per-tier budgets, eviction, prefetch | — | frequency-aware, host-staged |
| `SpeculationPolicy` | none/MTP/draft-model proposal + acceptance | none | MTP, DSpark, Eagle |

**Two residency policy levels** (see `docs/storage-residency-architecture.md`):

- `ferrule-model`'s `ResidencyPolicy` (`streaming_allowed`,
  `all_resident_required`) — model-level: does this model fit in GPU memory?
- `ferrule-runtime::storage`'s `StorageResidencyPolicy` (`Budgets`,
  `EvictionWeights`, `prefetch_window`) — runtime-level: how much to keep on
  each tier, what to evict. Active only when `streaming_allowed = true`.

---

## Priority roadmap

The priority order is now based on the user-visible interactive path:

```text
startup → prompt append/prefill → first token → steady decode → serving/multi-session
```

### P0 — Correctness and observability gates

- [x] JSON benchmark summary for `deepseek-v4-generate --json` with decode-only
      counters after prefill/warmup baseline subtraction.
- [x] MoE timing counters behind `FERRULE_CUDA_MOE_TIMING=1`.
- [x] DSV4 chat `stats>` line reports per-turn prefill/decode wall time.
- [x] Add a dedicated interactive benchmark command (`bench-interactive`) that
      feeds multiple turns and reports time-to-REPL, artifact load,
      first-token latency, per-turn prefill, per-turn decode, generated tok/s,
      and resident/cache counters.
- [x] Add golden interactive traces (`configs/golden/`) with token IDs and stop
      behavior for a fixed two-turn prompt under greedy decoding;
      `bench-interactive --golden <path>` compares live output against golden.
- [x] Add DSV4 model-internal attribution to `bench-interactive --json`: operator
      counters, per-layer bind/state/prefill/decode/attention/MoE timing,
      attention-stage breakdown, output-head detail, and profiling sync controls
      (`FERRULE_DSV4_PROFILE_SYNC=1`, `FERRULE_CUDA_MOE_TIMING=1`).

### P1 — Resident interactive engine

Goal: match SGLang-style execution at the architecture level before chasing more
micro-kernels: a long-lived engine owns model artifacts, CUDA context, KV/cache,
residency state, and request/session state.

- [x] DSV4 chat REPL starts immediately while artifacts load in the background.
- [x] CUDA context stays on the main thread; background loader only materializes
      CPU-side model artifacts.
- [x] `feed_token` appends session state without materializing final hidden/logits.
- [x] `TopKModelRunner::prefill_tokens` gives runtime a no-logits prompt append
      capability, so non-final prefill chunks no longer have to materialize top-k.
- [x] `prefill_tokens_topk_interactive` now routes through the DSV4 batched/segment
      prefill core and materializes top-k only for the final prompt chunk/token.
- [x] Replace the CLI-local lazy runner loop with a reusable runtime
      `EngineWorker`/`LazyEngineWorker` abstraction over `TopKModelRunner`;
      DSV4 chat and `bench-interactive` now execute turns through it.
- [x] Split worker execution from single-turn `generate_turn` into explicit
      `append_prompt` + `decode_next` phases with a concrete `TopKDecodeState`.
- [x] Add structured session finish reasons (`SequenceFinishReason`, re-exported
      as `TopKFinishReason` for top-k turns) and minimal `cancel_decode` on top
      of the phased API; rollback/KV rewind remains future work.
- [x] Expose structured worker stats (`EngineWorkerStats`) for position, tracked
      tokens, turns, prompt/generated totals, and bound layer count.
- [x] Connect session/scheduler/graph vocabulary: `SequenceState` now tracks
      request id, request sampling/stop/budget, KV handle, logical position, and
      prefill cursor; `SchedulerAction` materializes prefill/decode work into
      `ExecutionBatch`, paired with `ExecutionOutput` for backend results.
- [x] Add `ResidentScheduler` + `ResidentActionExecutor` + `ResidentTopKDriver`
      bridge so the runtime can drive `GenerateRequest -> SchedulerAction ->
      ExecutionBatch -> TopKModelRunner -> ExecutionOutput -> token event ->
      SequenceState/KV finish` without owning a concrete model.
- [x] Prove the bridge with real local full 43-layer DSV4 via
      `bench-interactive --runtime-driver` / `just dsv4-runtime-driver-bench`.
- [x] Add `PagedSequenceKvCache` so paged block tables implement `SequenceKvCache`
      admission/free lifecycle. CUDA paged-attention execution remains future work.
- [x] Avoid unnecessary next-logits work on the final max-token decode step in the
      scheduler/action executor path via `DecodeAction::require_logits`.
- [x] Report real prompt-to-first-token latency (`ttft`), prefill tok/s, decode
      tok/s, final position, finish reason, optional warmup timing/counter
      baselines, `runtime_path`, runtime-driver action stats, and DSV4 prefill path
      stats in chat/interactive benchmark output.
- [ ] Persistent warmup policy: after artifact load, optionally warm common CUDA
      kernels/linears without blocking the REPL.

### P2 — True device-resident CUDA prefill

Goal: eliminate host `Vec<f32>` boundaries and token-level full-layer loops during
prompt append. This is the current largest interactive latency gap and the first
feature to land before kernel tuning.

- [x] First correctness-first DSV4 segment prefill vertical slice:
      `tokens_per_chunk > 1`, existing session prefix, per-layer processing through
      the runner-owned DSV4 state.
- [x] Attention segment append updates window KV, compressed KV, and indexer state
      without resetting compressor state for appended chunks; prompt-start chunks
      still use the start-prefill compressor path.
- [x] Only materialize top-k/logits for final prefill chunks in the runtime driver;
      non-final chunks call `TopKModelRunner::prefill_tokens`.
- [x] Add chunk-size and max-layer controls to `bench-interactive`:
      `--prefill-chunk-size` and `--max-layers`, with per-turn runtime/DSV4 prefill
      stats in JSON.
- [ ] Add a DSV4 device prefill path alongside the current correctness host path:
      token ids → batched initial HC/hidden device buffer → per-layer device chain →
      final top-k only when requested.
- [ ] Keep prompt chunk hidden/HC states device-resident across layers; remove the
      prefill `Vec<f32>` handoff as the main execution contract.
- [ ] Make layer prefill compose device HC pre, norm, attention, MoE, and HC post
      without downloading intermediate hidden rows except for explicit parity/debug.
- [ ] Add automated chunk-size sweep benchmark for chat prompts and full 43-layer
      `Hello` profiles, tracking launches, H2D/D2H copies, expert loads, and stage
      attribution.

### P3 — Device-resident compressor/indexer path

Goal: remove remaining host boundaries that block efficient chunked prefill and
CUDA graph capture.

- [x] Device-resident decode chain exists for the hot single-token path.
- [x] Norm weights, rope tables, output-head top-k, and MoE workspace are cached.
- [ ] Move compressor softmax/state update to CUDA.
- [ ] Move indexer top-k/selection to CUDA.
- [ ] Keep combined window/compressed values device-resident across attention.
- [ ] Reconcile CPU reference and CUDA path with parity tests at layer boundaries.

### P4 — Expert residency and batched FP4 MoE execution

Goal: reduce MoE cost by changing execution shape before micro-optimizing kernels.
Do not fake DSpark batching; use real hidden columns and real per-column routes.

- [x] Managed-memory FP4 expert handles default-on for GB10 unified memory.
- [x] Host mmap cache + `HostStagedExpertCache` are wired.
- [x] Routing-aware hot expert prediction replaces naive low-ID prefetch.
- [x] Batched selected-expert MoE workspace avoids per-call scratch allocation.
- [x] FP4 `mxf4` Tensor Core path is integrated and default-on, with scalar
      fallback via `FERRULE_CUDA_MOE_TC=0`.
- [x] Fused Tensor-Core SwiGLU output pack removes the separate hidden-pack stage.
- [ ] Add layer-level prefill MoE batching, e.g. `routed_moe_prefill_batch`, to
      replace the current per-token `routed_moe_step` loop.
- [ ] Batch expert selection by layer, pre-load/upload selected experts once, and
      submit multi-token / multi-column work instead of per-token expert work.
- [ ] Keep a stronger per-layer resident hotset across warmup and measured turns;
      prefetch selected/predicted experts and avoid eviction/reload churn.
- [ ] Generalize MoE kernels for real `batch_cols > 1` with per-column routing.
- [ ] Convert DSpark/speculative blocks into real hidden columns before using GEMM
      as the main speed lever.
- [ ] Add correctness A/B tests against scalar MoE for integrated gate/up/down and
      batched prefill outputs.

### P5 — CUDA graph and launch/fusion work

Graph capture is not the first lever while host routing, expert streaming,
compressor/top-k, and prefill `Vec<f32>` boundaries are still in the hot path. It
becomes important after P2/P3/P4 make the execution shape device-resident and
launch overhead becomes a meaningful fraction of latency.

- [x] cuda-oxide fork exposes graph APIs and FP4 support; Ferrule also has raw
      graph handle scaffolding.
- [ ] Pre-allocate all decode/prompt buffers needed by stable buckets.
- [ ] Capture stable single-token decode buckets after compressor/indexer is
      device-resident.
- [ ] Capture prompt chunk buckets after true device-resident prefill exists.
- [ ] Fuse low-rank attention projection + norm + rotary where profiler confirms
      launch overhead matters.
- [ ] Do not prioritize graph replay for the current 43-layer `Hello` bottleneck;
      the measured limit is host-bound model execution, not generic graph replay.

### P6 — Benchmarks and competitive parity

- [x] ROADMAP-compatible JSON decode summaries exist.
- [x] Current two-turn chat pipe measurement is recorded below.
- [x] Add `bench-interactive` with reproducible prompts, runtime-driver mode,
      first-token latency, per-turn timing, warmup baselines, and DSV4 attribution.
- [ ] PK matrix by scenario: local CLI vs llama.cpp, CUDA optimized vs TRT-LLM,
      serving vs vLLM/SGLang, out-of-core vs FlexGen.
- [ ] Report model artifact, precision, hardware, arch, prompt, context, batch,
      generated tokens, graph summary, and quality/parity note.

### P7 — Serving runtime

- [ ] Add an OpenAI-compatible serving entrypoint after the resident worker boundary
      is shared and stable.
- [ ] Move DSV4 chat and future serving onto the same resident `EngineWorker`.
- [x] Request/session/sequence lifecycle API over `GenerateRequest`,
      `SequenceState`, `ResidentScheduler`, and `SchedulerAction`.
- [x] Paged KV sequence allocator/block-table lifecycle behind `SequenceKvCache`.
- [ ] Bind paged KV tables into graph/backend `KvState` execution.
- [ ] Prefix/radix KV reuse integrated with sessions.
- [ ] Continuous batching scheduler for decode on a multi-session backend.
- [ ] True chunked prefill scheduler for prompt bursts on CUDA/backend execution.
- [ ] Streaming responses, cancellation, metrics, structured output masks.

### P8 — DSpark / speculation

- [x] DSpark metadata is parsed from config.
- [ ] Represent DSpark/MTP artifacts as semantic `Speculation` bindings.
- [ ] Proposal model interface.
- [ ] Target verification interface.
- [ ] Acceptance/rejection/rollback state.
- [ ] Metrics: proposed, accepted, rejected, rollback, effective tok/s.
- [ ] Scheduler integration after resident engine state exists.

Rule: do not fake `dspark_block_size=5` by duplicating hidden vectors. Enable
DSpark only when the runtime has real speculative hidden states, per-column route
semantics, and rollback-safe KV/residency state.

---

## Implementation order

1. **Observability and runtime spine** — done for the current cut:
   `bench-interactive --runtime-driver` proves full 43-layer DSV4 through the
   generic resident driver, and JSON now attributes layer/attention/MoE/output-head
   time well enough to avoid optimizing the wrong subsystem.
2. **True device-resident prefill skeleton** — add the DSV4 CUDA prefill path that
   keeps token hidden/HC state on device across layers, with the existing host path
   retained as a correctness fallback.
3. **Batched multi-token MoE prefill** — replace the hot per-token
   `routed_moe_step` loop with layer-level batched columns: route/plan once, load
   selected experts once, submit multi-column work, and commit all token rows.
4. **Expert residency / prefetch / hotset** — use `expert_loads`,
   `expert_load_bytes`, `moe_expert_read_s`, and `moe_expert_upload_s` as gates;
   keep per-layer hot experts resident across warmup/measured turns and prefetch
   predicted experts.
5. **Attention WO-A / output projection** — after prefill/MoE shape improves,
   optimize grouped `output_a` because it is the largest attention stage; do not
   start with sparse attention, which is much smaller in the current profile.
6. **Device compressor/indexer** — remove remaining D2H/CPU boundaries needed by
   chunked prefill and graph capture, including compressor/indexer/top-k state.
7. **Correctness parity gates** — keep tokenizer/template, first-token,
   first-N-token, and layer-boundary checks alongside CUDA changes.
8. **CUDA graph replay** — capture stable decode/prompt buckets only after P2/P3/P4
   remove host work and make kernel-launch overhead meaningful.
9. **Paged KV + prefix cache serving** — integrate radix/paged KV with request
   lifecycle and continuous batching.
10. **Competitive PK** — compare against llama.cpp/TRT-LLM/SGLang/vLLM only with
    reproducible prompts and quality notes.

---

## What not to do

- Do not implement continuous batching before request/session state and KV
  abstraction exist.
- Do not implement radix cache before prefix cache and shareable KV pages exist.
- Do not delete CPU FP32 inference — it is the correctness anchor.
- Do not add broad model-family support before loader/runtime boundaries are
  cleaner.
- Do not rewrite all CUDA kernels before `bench-interactive` / JSON diagnostics
  and profiling exist.
- Do not jump to sparse-attention or CUDA-graph micro-optimization before true
  device-resident prefill, batched MoE prefill, and expert residency are fixed.
- Do not optimize `ResidentScheduler` for the current 43-layer `Hello` latency;
  runtime-driver stats show model execution, not scheduling, is the bottleneck.
- Do not add a new quantization format without a logits/parity, perplexity-style,
  or golden-token harness.
- Do not make WeightPack format changes without manifest/version checks.
- Do not add DeepSeek-specific fields to generic scheduler/sampler/session/
  executor — add a policy/layout role instead.
- Do not claim DSV4 execution until attention, router, experts, auxiliary
  tensors, and quant classes are all accounted for.
- Do not claim DSpark support until generic target/draft speculation with
  acceptance/rejection/rollback/metrics is implemented.
- Do not compare Ferrule against llama.cpp/vLLM/SGLang without recording model,
  quant, prompt, context, batch, hardware, and command.
- Do not block on async I/O — keep the synchronous bounded reader as the
  correctness path.
- Do not enable DSpark before base DSV4 correctness and device-resident decode
  are stable.
- Do not treat `PagedSequenceKvCache` as physical DSV4 CUDA KV; in the current
  runtime-driver path it is metadata/lifecycle while physical KV remains runner-owned.

---

## Gap against mainstream engines

| Area | Mainstream engines | Ferrule now | Ferrule gap |
|---|---|---|---|
| Model format | GGUF / engine plans / mature packaging | HF safetensors, WeightPack, semantic graph bindings | WeightPack-only startup, manifests, GGUF compatibility |
| Model coverage | many dense/MoE families | OLMoE executable; DSV4 full 43-layer CUDA/runtime-driver proof exists | DSV4 parity/perf maturity, serving entrypoint, broader adapters |
| Attention kernels | FlashAttention/FlashMLA, paged KV | DSV4 CUDA attention pieces and profiling exist; `output_a` is currently the largest attention stage | WO-A/grouped output projection, device-resident compressor/indexer, physical paged KV |
| MoE execution | batched/fused expert kernels, EP | expert streaming planner, FP4 TC path, artifact bundles, and profiling exist; prefill still has token-level MoE shape | layer-level batched prefill MoE, stronger residency/prefetch/hotset, real `batch_cols > 1` |
| Storage/residency | ad-hoc per-engine | `ferrule-runtime::storage` vocabulary + `ExpertResidencyBackend` trait + host-staged cache | wire policy deeper into DSV4 execution and keep per-layer hotsets resident |
| Quantization | GPTQ/AWQ/FP8/INT4/K/IQ, calibration | Q4/Q8 + FP4/FP8 artifact decoders, mixed policy | calibration, GGUF K/IQ execution, quality gates |
| Scheduler | continuous batching, chunked prefill, preemption | resident scheduler/actions/executor and paged sequence KV exist; CUDA multi-session backend not wired | graph/CUDA batch lowering, preemption, prefix reuse, physical paged KV |
| CUDA performance | fused kernels, CUDA graphs, memory planners, tensor core | 43-layer profile: 14,537 launches, 7,540 H2D, 5,962 D2H; MoE kernels 0.097s but MoE outer 6.94s; attention `output_a` 2.30s | device-resident prefill, batched MoE/residency, WO-A optimization, then CUDA graph |
| Correctness | evals, perplexity, golden suites, parity | unit/local smokes, compare tools | DSV4 official parity, long regressions |
| Serving UX | OpenAI API, streaming, metrics | design target; CLI diagnostics only | serving entrypoint after resident worker stabilizes |
| Distributed | TP/PP/EP, multi-node | none | design should not block, not near-term |

Ferrule's intended differentiation: **explicit runtime state + artifact-aware
MoE residency + Rust-native graph/runtime contracts**.

---

## Model bring-up contract

Every new model family satisfies the same contract. DSV4 is the first hard
target; OLMoE is the correctness fixture.

| Contract piece | Model family provides | Engine provides |
|---|---|---|
| `ModelDescriptor` | architecture, dimensions, tokenizer metadata, tensor inventory | format readers, metadata validation |
| `ModelLayout` | layer graph, required/optional tensor classes, residual/norm ordering | layer iteration, state ownership |
| `TensorBinding` | artifact tensor names → semantic tensor roles | loading, mmap/streaming, placement |
| `AttentionPolicy` | attention kind, projection roles, position encoding, KV shape | attention step scheduling, KV ownership |
| `MlpPolicy` / `ExpertPolicy` | dense MLP or routed/shared experts, activation semantics | expert batching, residency, execution |
| `QuantPolicy` | supported quant classes per tensor role, fallback rules | validators, conversion, kernels |
| `TokenizerPolicy` | tokenizer files, chat templates, special tokens | prompt/session lifecycle, sampler |
| `ValidationPolicy` | reference commands, golden tokens/logits, tolerances | compare tools, regression storage |
| `SpeculationPolicy` | optional draft/MTP attachment + acceptance rules | proposal/verify loop, rollback, metrics |

Source layout rule:

```
crates/ferrule-model/src/families/
  common.rs       # conservative dense Transformer name policy
  deepseek_v4.rs  # DeepSeek V4 / Flash / DSpark HF+GGUF names
```

When adding a family, create `families/<family>.rs` that maps source artifact
names into generic `TensorClass` / `TensorRole` values. Do not add artifact
tensor names to generic `tensor_policy`, runtime runners, KV state, or CUDA
kernels.

---

## DSV4 artifact facts

From local shard headers of `deepseek-ai/DeepSeek-V4-Flash-DSpark`:

| Group | Bytes | GiB | Implication |
|---|---:|---:|---|
| Routed expert gate | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert up | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert down | 52,479,131,648 | 48.875 | dominates memory |
| Attention tensors | 5,721,447,936 | 5.329 | preserve artifact dtype |
| Shared experts | 1,157,698,560 | 1.078 | conservative Q8 |
| Embedding | 1,059,061,760 | 0.986 | conservative Q8 |
| Output head | 1,059,061,760 | 0.986 | conservative Q8 |
| DSpark/MTP | 198,786,204 | 0.185 | keep input dtype until verifier |

Quantization decision: **quality-first default** — preserve official artifact
FP4 for routed experts, implement streaming/residency. Do not re-quantize to
4-bit as the single-node fit strategy. Design streaming before extra
quantization.

---

## DSV4 synthetic fixture benchmark

This section is retained as historical context only. The current optimization
priority is based on the real local model and interactive chat measurements in
the GB10 section below.

### Fixture purpose

`scripts/dsv4_synthetic_fixture.py` creates structurally valid DSV4-like
safetensors without requiring the full 166 GB real model. It is useful for
loader, inventory, graph binding, expert streaming, and residency development.

The fixture preserves:

- DSV4-style top-level config and tokenizer boundary;
- layer counts and semantic tensor roles;
- FP8 block-aligned attention artifacts;
- FP4 routed expert packing and E8M0 scale shapes;
- shard/index structure compatible with the real loader.

### What the old fixture measurements proved

Historical synthetic/fixture work helped validate these architecture changes:

1. DSV4 artifact inventory and typed binders can materialize complete layer
   payloads.
2. Expert streaming can load bounded safetensors byte ranges and install resident
   handles.
3. Batched selected-expert MoE and device-resident attention pieces reduce obvious
   host/device boundaries.
4. JSON counters are required; raw wall-clock command timing was too noisy.

### What is now stale

The old fixture-era bottleneck attribution, next-step list, and git-state notes
were removed because they conflict with the real-model measurements. In
particular:

- CUDA graph capture is no longer the first priority while host routing, expert
  streaming, compressor/top-k, and prefill `Vec<f32>` boundaries remain.
- The current highest-priority user-visible bottleneck is the host-heavy DSV4
  prefill/model execution shape; this needs true device-resident prefill and
  batched layer-level MoE prefill.
- MoE Tensor Core work remains important, but the current `batch_cols=1` path is
  not enough for the 5 tok/s target or for honest DSpark utilization.

---

## GB10 hardware analysis and measured state

### Hardware and model facts

| Item | Value |
|---|---|
| GPU | NVIDIA GB10 / Blackwell, `sm_121a` build target |
| Memory | 128 GB unified LPDDR5X class system memory |
| Model | `models/DeepSeek-V4-Flash-DSpark` |
| DSV4 shape | 43 layers, hidden 4096, MoE intermediate 2048 |
| Experts | 256 routed experts/layer, top-6 experts/token |
| Expert dtype | FP4 E2M1 packed weights + E8M0 block scales |
| DSpark metadata | block size 5, target layers `[40, 41, 42]`, Markov rank 256 |

The theoretical ceiling remains orders of magnitude above current performance;
current work is dominated by runtime architecture and under-utilized kernels, not
raw memory bandwidth alone.

### Measurement methodology

Use JSON decode summaries for throughput and chat `stats>` lines for interactive
latency. Avoid comparing raw shell `run:` time across code paths because it mixes
artifact load, prefill, decode, cold CUDA setup, expert residency, and stdout.

Canonical decode command:

```bash
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark \
  --prompt "Hello" --backend cuda --max-tokens 8 \
  --warmup-tokens 4 --output-head-chunk-rows 4096 \
  --json --chat
```

Profiling-only attribution envs:

```bash
FERRULE_DSV4_PROFILE_SYNC=1 FERRULE_CUDA_MOE_TIMING=1 \
  ./target/release/ferrule bench-interactive models/DeepSeek-V4-Flash-DSpark \
    -p Hello -n 1 \
    --runtime-driver \
    --warmup-tokens 1 \
    --prefill-chunk-size 4096 \
    --max-layers 43 \
    --json > target/dsv4-profile-l43-warm1-fine-sync-moe.json
```

Build CUDA release binaries with cuda-oxide, not plain Cargo:

```bash
cargo oxide build --features cuda --arch sm_121a -- --release -p ferrule-cli
```

`FERRULE_DSV4_PROFILE_SYNC=1` and `FERRULE_CUDA_MOE_TIMING=1` insert stream
synchronizations and should not be used as headline throughput results.

### Latest runtime-driver full-layer profile (2026-07-08)

Profile file:
`target/dsv4-profile-l43-warm1-fine-sync-moe.json`.

| Metric | Value | Interpretation |
|---|---:|---|
| `ttft` | **15.033s** | first-token user-visible latency for `Hello`, `-n 1` |
| prefill wall time | **13.821s** | dominant end-to-end component |
| aggregate decode | **0.825 tok/s** | still far below interactive target |
| runtime actions | **2** | one prefill chunk + one decode step |
| prefill tokens | **5** | scheduler/runtime work is small |
| emitted tokens | **1** | profile isolates first-token path |

Runtime-driver stats were:

```json
{
  "actions": 2,
  "prefill_chunks": 1,
  "prefill_tokens": 5,
  "decode_steps": 1,
  "emitted_tokens": 1,
  "finished_sequences": 1
}
```

Conclusion: for the current full 43-layer `Hello` path, the generic resident
runtime/scheduler is not the bottleneck. The bottleneck is DSV4 model execution:
prefill path shape, MoE staging/submit/residency, attention output projection,
and host/device boundaries.

Layer-level attribution:

| Stage | Time |
|---|---:|
| layer total | **13.475s** |
| prefill total | **12.267s** |
| decode total | **1.208s** |
| attention | **5.725s** |
| MoE | **6.937s** |
| state init | **1.489s** |

Attention fine profile:

| Attention stage | Time | Note |
|---|---:|---|
| `output_a` / WO-A | **2.301s** | largest attention block |
| `main_compress` | **0.805s** | compressor still material in prefill |
| `output_b` | **0.583s** | second output projection stage |
| `indexer_compress` | **0.414s** | indexer/compressor path still visible |
| `q_a` | **0.333s** | projection |
| `kv_proj` | **0.323s** | projection |
| `q_b` | **0.231s** | projection |
| `topk_build` | **0.230s** | attention top-k build |
| `sparse_attention` | **0.177s** | not the dominant attention cost |

MoE attribution:

| MoE stage/counter | Value | Note |
|---|---:|---|
| layer MoE total | **6.937s** | layer-profile MoE time |
| MoE outer sum | **6.927s** | wrapper/staging/submit path |
| CUDA-timed MoE kernels | **0.097s** | kernel body is not the 7s bottleneck |
| router | **0.364s** | CPU/device planning work |
| expert read | **0.973s** | host/disk/mmap staging cost |
| expert upload | **2.504s** | dominant residency transfer cost |
| shared expert | **0.634s** | shared path still material |
| workspace | **0.116s** | setup/allocation/prep cost |
| compute submit | **2.326s** | per-token/per-expert submit shape is too fragmented |
| expert loads | **919** | too much churn for one `Hello` |
| expert load bytes | **12.286 GB** | impossible to hit good UX at this churn |

Operator/copy pressure:

| Counter | Value |
|---|---:|
| kernel launches | **14,537** |
| H2D copies | **7,540** |
| H2D bytes | **342.9 MB** |
| D2H copies | **5,962** |
| D2H bytes | **153.9 MB** |

Output head warm path is no longer the full-depth bottleneck:

| Output-head metric | Value |
|---|---:|
| `lm_head_topk` | **0.063s** |
| chunks | **32** |
| cache hits | **32** |
| cache misses | **0** |

1-layer smoke profile for comparison:
`target/dsv4-profile-l1-fine-nosync.json` reported **ttft 1.449s**, prefill
**1.195s**, and decode **3.948 tok/s**. The full 43-layer slowdown therefore
comes from full-depth model execution shape: host-heavy prefill/MoE boundaries,
expert churn, attention WO-A projection, copies, and launch fragmentation.

### Earlier JSON decode measurements (2026-07-07)

All rows use `--prompt Hello --chat --warmup-tokens 4 --output-head-chunk-rows
4096` unless noted. `TC` means the integrated FP4 `mxf4` path. `Scalar` means
`FERRULE_CUDA_MOE_TC=0`.

| Output file | Tokens | MoE path | Prefetch | Hotset budget | decode tok/s | decode s | Expert loads | Evictions | Resident experts | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `target/dsv4-tc-default-8w4.json` | 8 | TC | 0 | 0 | **0.910** | 8.789 | 365 | 0 | — | best current short JSON run |
| `target/dsv4-hotset-default-8w4.json` | 8 | Scalar | 0 | 0 | 0.865 | 9.250 | 388 | 0 | 1628 | managed planner default, no eviction |
| `target/dsv4-hotprefetch8-8w4.json` | 8 | Scalar | 8 | 0 | 0.897 | 8.918 | 389 | 44 | 1585 | routing-aware prefetch |
| `target/dsv4-tc-hotprefetch8-8w4.json` | 8 | TC | 8 | 0 | 0.902 | 8.870 | 375 | 40 | — | TC + hot prefetch |
| `target/dsv4-hotset16-prefetch8-8w4.json` | 8 | Scalar | 8 | 16 | 0.827 | 9.672 | 605 | 605 | 688 | bounded hotset slowed short run |
| `target/dsv4-hotset-default-16w4.json` | 16 | Scalar | 0 | 0 | 0.740 | 13.515 | ~48.4/tok | 0 | 1728 | noisy longer scalar baseline |
| `target/dsv4-hotprefetch8-16w4.json` | 16 | Scalar | 8 | 0 | 0.817 | 12.233 | ~48.6/tok | 80 | 1642 | hot prefetch helps longer run |
| `target/dsv4-tc-hotprefetch8-16w4.json` | 16 | TC | 8 | 0 | 0.864 | 11.571 | ~47.0/tok | 77 | — | TC + prefetch best measured 16-token run |
| `target/dsv4-fused-scalar-8w4.json` | 8 | Scalar | 0 | legacy | 0.835 | 9.578 | 388 | 43 | 1578 | before managed planner no-evict alignment |
| `target/dsv4-fused-tc-8w4.json` | 8 | TC | 0 | legacy | 0.725 | 11.033 | 364 | 40 | 1554 | old TC path before residency/default fixes |

Interpretation:

- TC is now default because current A/B is no longer a clear regression and is
  required for future real GEMM/DSpark utilization.
- The integrated TC path at `batch_cols=1` is not enough for 5 tok/s. It needs
  real multi-column hidden states, not duplicated data.
- Bounded hotset is a memory-pressure tool, not a short-prompt speed default.
- Routing-aware prefetch can help 8/16-token decode but does not change the main
  latency class because prompt/decode still run full-layer token steps.

### MoE profiling measurements

With `FERRULE_CUDA_MOE_TIMING=1` on a 2-token run, timing syncs make absolute
`decode_tok/s` meaningless, but stage attribution is useful:

| Path | decode tok/s under timing | MoE total | Hidden pack | Notes |
|---|---:|---:|---:|---|
| TC fused | 0.219 | 0.110s | 0.000s | `SwiGLU -> packed FP4` fused; hidden pack removed |
| Scalar | 0.216 | 0.155s | n/a | scalar FP4 fallback |

Conclusion: the fused TC MoE substage improves MoE time, but end-to-end latency
is still dominated by prompt/decode architecture, attention/compressor/indexer
host work, and under-utilized single-column MMA.

### Interactive chat measurements (2026-07-07)

Command:

```bash
printf 'Hello\nHi\n/exit\n' | \
  ./target/release/ferrule chat models/DeepSeek-V4-Flash-DSpark \
  -q cuda -n 1 --chat-template deepseek-v4 --temp 0
```

| State | First turn prefill | First turn decode/feed | Second turn prefill | Second turn decode/feed | Notes |
|---|---:|---:|---:|---:|---|
| Before interactive append | 29.285s | 9.341s | 10.487s | 1.322s | non-final prompt/generated tokens materialized hidden/logits |
| After interactive append | **23.648s** | **1.278s** | **5.878s** | **1.089s** | REPL ready immediately; artifact load ~0.82s background |

This is the most important current user-facing result. The remaining multi-second
per-turn prefill shows that Ferrule still lacks true CUDA chunked prefill over an
existing prefix. SGLang/vLLM-style serving engines win here because prompt chunks,
KV pages, scheduler state, and decode workers are resident and batched.

### Updated bottleneck attribution (2026-07-08)

| Rank | Bottleneck | Current impact | Target fix |
|---|---|---|---|
| 1 | Host-heavy prefill path | 43-layer profile spends **13.82s** before first token; hidden/HC/MoE still cross host `Vec<f32>` boundaries | true device-resident prefill path with old host path kept as fallback |
| 2 | Token-level MoE prefill loop | MoE outer **6.94s** vs CUDA MoE kernels **0.097s**; `compute_submit=2.326s` | layer-level multi-token / multi-column `routed_moe_prefill_batch` |
| 3 | Expert residency churn | **919** expert loads and **12.286 GB** transferred for one `Hello` | per-layer resident hotset, selected/predicted expert prefetch, reuse across warmup/measured turns |
| 4 | Attention WO-A / `output_a` | **2.30s**, much larger than `sparse_attention=0.18s` | grouped output projection execution-shape/kernel work after prefill/MoE shape improves |
| 5 | Compressor/indexer/top-k host boundaries | `main_compress=0.805s`, `indexer_compress=0.414s`, many copies | move compressor/indexer/top-k state updates device-side |
| 6 | Launch/copy fragmentation | **14,537** launches, **7,540** H2D, **5,962** D2H in the profiled request | reduce submit/copy shape first, then CUDA graph stable buckets |
| 7 | Startup/warmup | REPL ready fixed; first prompt still pays CUDA/operator warmup | optional persistent warmup after resident engine load |

### Performance projection

| Phase | Expected user-visible effect | Status |
|---|---|---|
| Background artifact load | faster time-to-REPL | done |
| Runtime-driver spine | clean request → schedule → execute → token lifecycle without concrete model in runtime | done |
| Model-internal observability | enough attribution to avoid scheduler/kernel guesswork | done |
| True device-resident prefill | largest remaining prompt latency drop | next priority |
| Batched MoE prefill columns | attacks `moe_compute_submit_s`, expert upload/read churn, launches/copies | next priority |
| Expert hotset/prefetch reuse | reduce 919 loads / 12GB transfer class of failures | next priority |
| Attention WO-A output projection | reduce largest attention stage after prefill/MoE shape improves | next |
| Device compressor/indexer | enables clean prefill and graph capture | next |
| CUDA graph replay | reduces launch overhead after host boundaries are gone | later |
| Paged KV + continuous batching | serving parity with SGLang/vLLM class engines | runtime skeleton done; CUDA/backend integration later |

### Next-step priority list (2026-07-08)

| Priority | Task | Why now |
|---|---|---|
| **P1** | True device-resident DSV4 prefill skeleton | Removes the biggest host `Vec<f32>` / hidden / HC boundary before kernel tuning |
| **P2** | Batched multi-token / multi-column MoE prefill | Directly targets `compute_submit=2.326s`, expert upload/read time, copies, and launches |
| **P3** | Expert residency / prefetch / hotset reuse | 919 expert loads / 12.286 GB for one `Hello` cannot produce good UX |
| **P4** | Attention WO-A / `output_a` optimization | `output_a=2.30s` is the largest attention stage; sparse attention itself is not first |
| **P5** | Device compressor/indexer/top-k state | Required by clean prefill and later graph capture |
| **P6** | CUDA graph buckets | Only after host routing/streaming/prefill boundaries are gone |
| **P7** | DSpark proposal/verify/rollback state | Needed before using `dspark_block_size=5` honestly |
| **P8** | CUDA paged KV + continuous batching | Runtime skeleton exists; backend/serving stage next |

---

## Review checklist

For every patch:

- **Behavior**: does it change inference output? Is the change tested?
- **Correctness**: do CPU reference components still pass? Do logits/parity or
  golden-token diagnostics regress?
- **Generality**: does it add model-specific names to generic crates?
- **Performance**: does it add host sync to a device path? Does it add kernel
  launches?
- **State ownership**: does it leak state into the wrong crate?
- **Compatibility**: does it break existing CLI commands or WeightPack files?
- **Safety**: does it add `unsafe` without justification?
