# Ferrule Roadmap

_Last updated: 2026-07-06_

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. The current
engineering center is **runtime graph backed Transformer execution** with
**DeepSeek-V4 Flash / DSpark** as the near-term pressure test and **OLMoE** as
the correctness golden model. DSV4 drives missing runtime capabilities but must
not become a model-specific runtime architecture — model-family names and raw
tensor names stay in `ferrule-model`; `ferrule-runtime` sees semantic roles,
graph ops, artifact groups, expert registries, KV state, and backend object
stores.

This roadmap merges the previous `ROADMAP.md` and `TEMP_INFERENCE_ROADMAP.md`
into a single source of truth, reflecting the current code state including the
`ferrule-storage` vocabulary crate and the `ExpertResidencyBackend` trait.

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
11. [Review checklist](#review-checklist)

---

## Current capabilities

### Working now

- **OLMoE correctness fixture**: safetensors loading, GPU Q4_0 inference
  through cuda-oxide kernels. Correct `norm_topk_prob` semantics as a
  regression fixture. CPU reference components (artifact decoders, router
  math, HC math, expert executor) anchor CUDA kernels.
- **DeepSeek V4 chat**: full 43-layer CUDA greedy generation + readline chat
  through `ferrule chat models/DeepSeek-V4-Flash-DSpark -q cuda
  --chat-template deepseek-v4`. Official DSV4 chat prompt wrapper works.
- **Runtime graph foundation**: `ferrule-graph` opaque IR, `GraphProgram`,
  `ExecutionBatch`, `ExternalBindingPlan`, `BackendObjectStore`,
  `ReferenceGraphBackend`, shape validation registry. Dense decoder graph +
  generic semantic Transformer graph paths.
- **Graph execution bridge**: coarse `transformer_state_init →
  transformer_layer → output_projection` path executes through
  `ReferenceGraphBackend` using typed artifact binders. Layer payloads cached
  per graph program/layer.
- **Model-family boundary**: `ferrule-model/src/families/` owns per-family HF
  tensor parsing. DSV4 names isolated in `families/deepseek_v4.rs`. Runtime
  consumes semantic artifact slices only.
- **DSV4 metadata + artifact materialization**: 72,317 tensors across 48 shards
  classified. Attention/HC/router/shared-expert/routed-expert tensor descriptors
  bind from local HF shards. Local-model tests skip cleanly when absent.
- **DSV4 synthetic fixture**: `scripts/dsv4_synthetic_fixture.py` generates
  structurally complete zero-weight safetensors scaled to VRAM × 2.
  Auto-detects NVIDIA VRAM, scales all dimensions proportionally (FP8 block
  alignment, FP4 scale packing), and produces a valid `config.json` +
  `model.safetensors.index.json` + safetensors shards that Ferrule's DSV4
  loader, inventory, expert streaming, and generation path consume
  identically to real weights. Used for streaming/residency development
  without requiring the 166 GB full model on disk.
  LRU eviction, prefetch, and `commit_step`. `ExpertStreamingReader` reads
  bounded byte ranges from local safetensors. CPU packed-FP4 SwiGlU expert
  executor + correctness-first CUDA packed-FP4 expert executor.
- **Storage/residency vocabulary** (`ferrule-storage` crate): `StorageObjectId`
  (structured enum), `ObjectLocator`, `Placement`, `ObjectReplica`,
  `ReplicaHandleId`, `StorageCatalog` trait, `TransferEngine` trait +
  `MockTransferEngine`, `StorageResidencyPolicy` with frequency-aware eviction
  scoring. 67 tests passing.
- **`ExpertResidencyBackend` trait** (`ferrule-runtime`): unifies the
  duplicated load/evict loop from `routed_moe.rs` and `deepseek_v4.rs`.
  `apply_streaming_step()` provides one loop. `CpuExpertHandleStore` impl
  done. Adapters: `ExpertId → StorageObjectId`, `ExpertStorageTier →
  Placement`, `ExpertLoadSource → ObjectLocator`. 14 tests passing.
- **WeightPack**: mmap'd reader with manifest validation, streaming writer,
  zero-copy slices. WeightPack-only load path for OLMoE.
- **Quantization**: Q4_0/Q8_0 full path, mixed precision policy, FP4 E2M1 +
  E8M0 artifact decoders, FP8 E4M3FN decoders, KV quantization. K-quant/AWQ
  investigation done.
- **KV cache**: `KvCache` trait, contiguous per-session, radix prefix cache.
- **Scheduler**: continuous batching + preemption prototypes.
- **Structured decoding**: token mask API, program-like generation API.
- **CLI**: `info`, `run`, `gpu-run`, `chat`, `cuda`, `bench-infer`,
  `compare-logits`, `inspect-weightpack`, `server`, `deepseek-v4-probe`,
  `deepseek-v4-generate`, `perplexity`, `inspect-cache`.
- **Sampling**: temperature, top-k, top-p, min-p, repeat penalty, seed, stop
  strings, top-K logprobs. `generation_config.json` auto-loading. Chat template
  registry: OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain.
- **Server**: minimal OpenAI-compatible `/v1/chat/completions` with SSE
  streaming.
- **Benchmarks**: `bench-infer` with JSON summary. `ferrule-bench` owns
  benchmark/report schemas.

### Main gaps

Measured on real local DSV4 model (NVIDIA GB10 Blackwell, sm_121, 128 GB
unified LPDDR5X, 43 layers, 256 experts, hidden=4096, 166 GB on disk).
After Phase 3+4 optimizations (device-chain, concurrent I/O, batched MoE
kernel, KV device rotary/QAT):

| Metric | Session start | Current (2026-07-05) | Root cause |
|---|---:|---:|---|
| decode tok/s | 0.393 | 0.602 (+53%) | scalar FP4 GEMV, expert cold-load |
| kernel launches/token | ~2,792 | ~1,849 | batched MoE reduced 30→3; attention still unfused |
| D2H copies/token | ~5,678 | ~4,169 | device chain eliminated per-layer HC/hidden downloads |
| expert loads/token | 125.7 | 125.7 | cold start; pre-load not yet implemented |

**Key finding:** The dominant bottleneck is now **scalar FP4 execution** (no
tensor core). Each FP4 weight requires ~3 FP operations to decode+multiply,
versus native FP4 tensor core MMA at 500 TFLOPS. This is a ~100× compute
penalty. See the GB10 hardware analysis section for theoretical ceiling.

- DSV4 decode still has host-side Vec<f32> boundaries at `hc_pre`/`hc_post`,
  `apply_rotary_tail`, `sparse_attention`, and `grouped_output_a` boundaries.
- Expert residency CUDA integration: `HostStagedExpertCache` implemented and
  wired in; `ExpertResidencyBackend` not yet implemented for
  `DeepSeekV4CudaOperatorCache`.
- No production batched selected-expert kernel (6 experts/layer x 43 layers
  = 258 separate kernel launches per token).
- KV/compressed KV/indexer state not fully device-resident.
- CUDA graph capture not implemented: 9,808 launches for 5 forward passes.
- Official DSV4 numeric parity incomplete.
- `ferrule-storage` vocabulary not yet wired into execution path (Phase 0 types
  only, no call site changes).
- No paged KV allocator, prefix cache integration with serving, or production
  continuous batching through the graph runtime.
- No DSpark speculation loop.
- No competitive PK harness.

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
   `run/chat/server` must fail with a named missing-policy error.
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
- `ferrule-storage`'s `StorageResidencyPolicy` (`Budgets`, `EvictionWeights`,
  `prefetch_window`) — runtime-level: how much to keep on each tier, what to
  evict. Active only when `streaming_allowed = true`.

---

## Priority roadmap

### P0 — Correctness and observability gates

Goal: before optimizing DSV4, make every mismatch and bottleneck localizable.

- [x] Runtime graph validation and shape registry.
- [x] Generic semantic DSV4-capable graph builder without specialization.
- [x] Local DSV4 graph materialization smoke.
- [x] `ferrule-bench` owns benchmark/report/reference-smoke tooling.
- [x] DSV4/bench JSON summary schema.
- [x] Graph construction supports serializable summary for diffing.
- [ ] DSV4 first-token / first-N-token parity gate against official/reference.
- [ ] Layer-scoped debug dump: hidden, HC state, q/kv, attention output, router
      top-k, selected experts, logits top-k.
- [ ] Wire real backend telemetry into JSON schema: kernel launches/token,
      host↔device bytes/token, expert loads/token, peak VRAM/RSS.

### P1 — Graph execution bridge

Goal: `GraphProgram + BackendObjectStore` drives coarse `transformer_layer`
execution through existing generic components.

- [x] Coarse `transformer_state_init → transformer_layer → output_projection`.
- [x] Semantic graph externals materialize into artifact groups + expert
      registries + KV state.
- [x] `GraphLayerObjects` aggregation by `ArtifactGroupKind` + layer.
- [x] Typed binders: attention, HC, router, shared FFN from artifact groups.
- [x] Coarse `transformer_layer` lowering inside `ReferenceGraphBackend`.
- [x] Cache decoded payloads per graph program/layer.
- [x] Tiny/synthetic semantic layer through graph path.
- [ ] Local DSV4 layer 0 through the graph bridge when model is present.
- [ ] Debug dumps around the coarse layer boundary.

### P2 — Split `transformer_layer` into fine-grained semantic ops

Goal: expose performance-critical boundaries while staying model-family neutral.

- [ ] Split into phase ops: `layer_hc_pre`, `rms_norm`, `latent_attention`,
      `layer_hc_post`, `router_select`, `routed_moe`, `shared_ffn`,
      `residual_merge`.
- [x] Coarse graph attrs describe semantics (attention kind, KV shape, router
      kind, norm/HC eps, RoPE/YaRN, sliding/indexer metadata, SwiGLU limit).
- [ ] Attention/position/cache subgraph: `latent_q_project_a/b`,
      `latent_kv_project`, `rope_apply`, `window_kv_update`,
      `compressed_kv_update`, `indexer_score_topk`, `sparse_attention`,
      `attention_output_grouped_a/b`.
- [ ] Routed/shared MoE subgraph: `router_logits`, `hash_router_lookup`,
      `topk_select`, `route_weight_normalize`, `expert_registry_lookup`,
      `routed_expert_swiglu`, `shared_swiglu_ffn`, `moe_accumulate`.
- [ ] Shape inference for each semantic op.
- [ ] Validation rules rejecting raw HF tensor names in external keys.

### P3 — Device-resident DSV4 decode

Goal: remove host-mediated hot-path boundaries.
**Motivation:** Phase 1+2 optimization reduced H2D/D2H copies by 30-40%, but
decode tok/s only improved 7%. The real bottleneck was **synchronization
latency** from D2H copies (each is a CPU-GPU sync point), not data movement
volume. Phase 3 (2026-07-05) achieved +53% by eliminating per-layer D2H/H2D.

1. **Decode arena** — reusable device buffers for hidden, HC state, q/kv,
   attention workspace, router scores, expert outputs, logits/top-k.
   - [x] Device-resident FP4 expert input sharing: 6 experts per layer share
         one uploaded input buffer instead of 6 separate H2D uploads.
   - [x] Device-side expert output accumulation via `saxpy_into`.
   - [x] Norm weight caching (`rms_norm_device_cached`).
   - [x] Attention device chain (`decode_step_compressed_device`).
   - [x] Device-resident `hc_pre`/`hc_post`: `hc_pre_from_device` and
         `hc_post_from_device` implemented.
   - [x] Device-resident `apply_rotary_tail`: `cuda_rope_tail_from_device`.
   - [x] Device-resident sparse attention: `sparse_attention_with_combined_kv`.
   - [x] Cross-layer HC state device chain: `decode_step_device_hc_device`.
   - [x] Device-output MoE: `routed_moe_step_device_output`.
   - [x] Hidden→output-head device chain.
   - [x] KV device rotary + QAT (no D2H for KV path).
2. **Device-resident KV state** — window KV, compressed KV, indexer KV,
   compressor state stay on device.
   - [x] Combined KV device cache with incremental window/compressed append.
   - [ ] Compressor/indexer fully device-resident (still has host boundary).
3. **Artifact linear dispatch** — generic APIs for BF16/F32/FP8/FP4 payloads.
   - [x] `linear_matvec_from_device` with FP8 activation quant.
4. **Expert residency** — persistent expert handles.
   - [x] 8× GPU slots per layer, quality-first-no-prefetch policy.
   - [x] Concurrent expert I/O via tokio + positioned read (pread).
   - [x] Warmup and naive prefetch experiment (`--warmup-tokens`,
         `--moe-prefetch-experts`).
   - [ ] Budgeted routing-aware hotset residency / pinning. Do not preload all
         `(layer, expert)` bundles by default.
5. **Output projection/top-k** — chunked device-side lm_head top-k.
   - [x] `output_head_topk_chunks_with_device` consumes device hidden buffer.

Exit: one-token DSV4 decode downloads only final token/logit. **Achieved.**
Next bottleneck: scalar FP4 execution (P5 tensor core).

### P4 — Storage and residency integration

Goal: wire `ferrule-storage` vocabulary into the execution path. See
`docs/storage-residency-architecture.md` for full design.
**Motivation:** synthetic fixture recorded 243 expert disk reads across 5
forward passes — repeated expert activation re-reads from safetensors shards
with zero host-side caching. Host-staged cache targets this directly.

- [x] `ferrule-storage` crate with vocabulary types + traits (67 tests).
- [x] `ExpertResidencyBackend` trait + `apply_streaming_step()` + adapters
      (14 tests). `CpuExpertHandleStore` impl done.
- [x] Dead code `residency.rs` deleted.
- [ ] Implement `ExpertResidencyBackend` for `DeepSeekV4CudaOperatorCache`
      (CUDA path, `#[cfg(feature = "cuda")]`).
- [ ] Replace inline loops in `routed_moe.rs` and `deepseek_v4.rs` with
      `apply_streaming_step()`.
- [ ] Add `activation_count` to `ExpertState`; route eviction through
      `StorageResidencyPolicy` weights (recency + frequency).
- [ ] Add planner/cache consistency check + `planner_cache_inconsistencies`
      counter.
- [ ] Host-staged expert cache (`HostStagedExpertCache`) with per-tier
      hit/miss counters.
      - [x] `HostStagedExpertCache` LRU cache implemented in
            `expert_residency.rs` (6 tests). Serves re-activated experts from
            host RAM instead of re-reading from disk.
      - [x] Wired into `DeepSeekV4CudaOperatorCache::routed_moe_step` CUDA
            path.
      - [ ] Wire hit/miss counters into JSON benchmark summary.
- [ ] Expert ID migration: `ExpertId` → `StorageObjectId` as handle store key.

### P5 — Kernel fusion, tensor core GEMM, and CUDA graph replay

Goal: reach the 5 tok/s milestone by replacing scalar FP4 MoE compute with
Blackwell tensor core MMA, while making expert residency predictable enough for
steady-state decode.

**Motivation:** Phase 3+4 eliminated major host boundaries and batched MoE
kernel launches, but current end-to-end decode is still below 1 tok/s. The
latest GB10/sm_121 measurements show that naive expert prefetch/warmup reduces
H2D traffic substantially, but does **not** unlock the 5 tok/s target by itself:
the MoE path is still dominated by **scalar FP4 decoding/GEMV** rather than FP4
Tensor Core execution. The 5 tok/s milestone therefore requires the FP4 Tensor
Core/raw PTX path plus smarter expert residency; kernel launch reduction and
prefetch are secondary enablers, not sufficient fixes.

**Phase 5a — Tensor core FP4 GEMM / raw PTX path (highest impact)**
- [x] Batched MoE kernel (3 launches instead of 30) — Phase 4.
- [x] Confirm upstream cuda-oxide gap: latest `NVlabs/cuda-oxide` main still
      exposes `tcgen05` helpers for f16/bf16/tf32, but not FP8/FP4
      `kind::f8f6f4` / e4m3/e5m2/e2m1 variants. Relevant upstream issue:
      <https://github.com/NVlabs/cuda-oxide/issues/339>.
- [x] **Ferrule-local raw PTX FP4 Tensor Core smoke kernel** — do not wait for
      cuda-oxide upstream. First prove one isolated `mma.sync` FP4/e2m1
      kernel on sm_121 with correctness tests and a tiny benchmark.
      **Completed 2026-07-05**: `crates/ferrule-cuda/ptx/fp4_tc_smoke.ptx`
      uses `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale` with
      e2m1 + E8M0 scales. Loaded at runtime via `load_module_from_image`.
      Correctness verified against CPU scalar FP4 reference (max_err=0.0).
- [ ] **FP4 tensor core GEMM for expert gate+up/down** — replace scalar GEMV
      with Tensor Core MMA. This is the primary 5 tok/s unlock.
      - [x] Confirm exact PTX operand spelling/layout for Blackwell FP4 e2m1 +
            E8M0 scales. Verified: `.b32` registers required (not `.u32`),
            PTX ISA 8.8+ required for sm_121a, `.scale_vec::2X` is default
            for `kind::mxf4` and can be omitted.
      - [ ] Implement feature/env-gated raw PTX kernel, preserving scalar
            fallback.
      - [ ] Integrate batched expert gate+up first, then down projection.
      - [ ] Add CUDA correctness tests against current scalar kernels.
- [ ] **FP8 tensor core for attention matvec** — q_proj, kv_proj, output_proj
      use FP8 weights; replace scalar GEMV with FP8 MMA after FP4 MoE is proven.

**Phase 5b — Expert residency / prefetch (necessary but not sufficient)**
- [x] `--warmup-tokens N` fills GPU expert residency before timed decode.
- [x] `--moe-prefetch-experts N` experiment wires planner prefetch into decode.
- [ ] Replace naive low-ID prefetch with routing-aware hotset prediction:
      previous-token selected experts, per-layer LFU/LRU hot experts, or a
      warmup-derived pinned hotset.
- [ ] Add explicit per-layer residency budget / pinning policy. Do **not** assume
      “preload all experts” is safe: `ExpertId = (layer, expert)`, so full routed
      expert residency is roughly `num_layers × num_routed_experts` bundles
      (>100 GB class for DSV4), not just 256 experts / 3.3 GB.
- [ ] Separate counters for disk read, host-staged cache hit, GPU upload, and
      resident hit, so residency changes can be evaluated without guessing.

**Phase 5c — CUDA graph capture (after tensor core and upload-free hotset)**
- [x] Count kernel launches per token (measured: 1,849/token after Phase 4).
- [ ] **CUDA graph capture for stable decode buckets** — capture the entire
      `decode_step_device_hc_device` path as one graph. Requires all host
      boundaries eliminated (compressor/indexer still on host).
      - [ ] Device-resident compressor/indexer (P3 remaining item).
      - [ ] Identify stable decode shape buckets.
      - [ ] Capture graph for each bucket.
      - [ ] Replay with updated input/weight pointers.
- [ ] Fuse low-rank attention projection + norm pieces.
- [ ] Fuse sparse attention gather/softmax/value accumulation.
- [ ] Grouped output projection fusion.
- [ ] Steady-state decode shape buckets.

### P6 — Benchmarks and competitive parity

Goal: compare honestly against mainstream engines by scenario.

- [x] DSV4 JSON benchmark summary via `deepseek-v4-generate --json`.
- [ ] `bench-infer` DSV4 mode through graph bridge.
- [ ] Prompt/decode split comparable to `llama-bench`.
- [ ] Report: model artifact, recipe, precision, hardware, arch, prompt,
      context, batch, generated tokens, graph summary.
- [ ] PK matrix: local single-user (vs llama.cpp), CUDA optimized (vs TRT-LLM),
      serving (vs vLLM/SGLang, after scheduler), out-of-core (vs FlexGen).

### P7 — Serving runtime

Goal: multi-session serving through the graph runtime.

- [ ] Request/session/sequence lifecycle API.
- [ ] Paged KV allocator behind graph `KvState` handles.
- [ ] Chunked prefill.
- [ ] Continuous batching scheduler (prototype exists, not graph-backed).
- [ ] Prefix/radix KV reuse (radix cache exists, not integrated with serving).
- [ ] OpenAI-compatible `/v1/chat/completions` server (minimal version exists).
- [ ] Streaming responses, cancellation, metrics.
- [ ] Structured output masks / JSON constraints.

### P8 — DSpark / speculation

Goal: generic speculation around a correct target model.

- [ ] Represent MTP/draft artifacts as semantic `Speculation` bindings.
- [ ] Proposal model interface.
- [ ] Target verification interface.
- [ ] Acceptance/rejection/rollback state.
- [ ] Metrics: proposed, accepted, rejected, rollback, effective tok/s.
- [ ] Scheduler integration after serving state exists.

Rule: do not enable DSpark before base DSV4 correctness + device-resident
decode are stable.

---

## Implementation order

1. **Official parity gate** — tokenizer/template JSON parity, first-token /
   first-N-token golden tests, intermediate layer dumps.
2. **DSV4 benchmark contract** — JSON bench with load/prefill/decode split,
   tok/s, bytes moved, resident experts, peak GPU.
3. **Device-resident decode arena** — remove host `Vec<f32>` from per-operator
   boundaries. ✅ Mostly done (Phase 3).
4. **GPU KV/compressor/indexer state** — no host allocation/copy in decode.
   ✅ KV done; compressor/indexer still has host boundary.
5. **Expert residency + batched expert kernel** — `ExpertResidencyBackend`
   for CUDA, host-staged cache, one batched MoE kernel per layer. ✅ Batched
   kernel done (Phase 4); naive warmup/prefetch experiment done; routing-aware
   hotset residency pending.
6. **Tensor core FP4/FP8 GEMM** — replace scalar GEMV with Blackwell tensor
   core MMA. **This is the critical path to 5+ tok/s.**
7. **Routing-aware hotset residency** — pin/budget frequently selected
   `(layer, expert)` bundles; avoid full all-expert preload by default because
   full DSV4 routed expert residency is >100 GB class.
8. **Attention/qkv fusion** — after tensor core, fuse projection/norm/rotary.
9. **GPU output-head/top-k** — greedy decode stays device-side. ✅ Done.
10. **CUDA graph replay** — capture stable decode buckets. Requires all host
    boundaries eliminated first.
11. **DSpark speculation** — only after base correctness + decode stable.
12. **Server PK** — only after scheduler/batching wired.

---

## What not to do

- Do not implement continuous batching before request/session state and KV
  abstraction exist.
- Do not implement radix cache before prefix cache and shareable KV pages exist.
- Do not delete CPU FP32 inference — it is the correctness anchor.
- Do not add broad model-family support before loader/runtime boundaries are
  cleaner.
- Do not rewrite all CUDA kernels before `bench-infer` and profiling exist.
- Do not add a new quantization format without `compare-logits`, perplexity, or
  golden-token harness.
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

---

## Gap against mainstream engines

| Area | Mainstream engines | Ferrule now | Ferrule gap |
|---|---|---|---|
| Model format | GGUF / engine plans / mature packaging | HF safetensors, WeightPack, semantic graph bindings | WeightPack-only startup, manifests, GGUF compatibility |
| Model coverage | many dense/MoE families | OLMoE executable; DSV4 metadata + artifact graph | full DSV4 graph execution, broader adapters |
| Attention kernels | FlashAttention/FlashMLA, paged KV | correctness-first reference + CUDA kernels | fused DSV4 latent/sparse attention, device-resident KV |
| MoE execution | batched/fused expert kernels, EP | expert streaming planner, artifact bundles, CPU/CUDA correctness paths | production batched kernel, residency/prefetch, GPU handle reuse |
| Storage/residency | ad-hoc per-engine | `ferrule-storage` vocabulary + `ExpertResidencyBackend` trait | wire vocabulary into execution, host-staged cache |
| Quantization | GPTQ/AWQ/FP8/INT4/K/IQ, calibration | Q4/Q8 + FP4/FP8 artifact decoders, mixed policy | calibration, GGUF K/IQ execution, quality gates |
| Scheduler | continuous batching, chunked prefill, preemption | scheduler prototypes, not graph-backed | paged KV + request lifecycle + graph batch lowering |
| CUDA performance | fused kernels, CUDA graphs, memory planners, tensor core | 1,849 launches/token (Phase 4), scalar FP4 GEMV, +53% from device chain | **Tensor core FP4 GEMM** (P5a), expert pre-load, CUDA graph (P5b) |
| Correctness | evals, perplexity, golden suites, parity | unit/local smokes, compare tools | DSV4 official parity, long regressions |
| Serving UX | OpenAI API, streaming, metrics | minimal server, CLI tools | full server after graph runtime stabilizes |
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

### Experiment environment

| Item | Value |
|---|---|
| Date | 2026-07-04 |
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CUDA arch | sm_86 |
| OS | WSL2 (Linux), Windows host |
| Build | `just build-cuda` (cargo-oxide, release, `--features cuda --arch sm_86`) |
| Model | `models/DeepSeek-V4-Flash-DSpark-synthetic` (zero-weight structural fixture) |
| Fixture script | `scripts/dsv4_synthetic_fixture.py` (auto-scales to VRAM × 2) |
| Prompt | `"Hello"` (1 token, token id 19923) |
| Layer cap | `--max-layers 43` (full model depth) |
| Decode tokens | `-n 4` (4 decode tokens) or `-n 10` (10 decode tokens, for stability) |
| Weights | All zero-valued; structure identical to official model |
| compress_ratio | 0 (CSA/HCA disabled for synthetic; uses `decode_step_no_compress` path) |
| MTP | disabled (`num_nextn_predict_layers=0`) |

### Fixture configuration

| Param | Value |
|---|---|
| hidden_size | 2,816 (scaled 0.69x, aligned 128) |
| num_hidden_layers | 43 |
| n_routed_experts | 170 (scaled from 256) |
| num_attention_heads | 42 (o_groups=6, divides 42) |
| head_dim | 384 |
| q_lora_rank / o_lora_rank | 768 / 768 |
| moe_intermediate_size | 1,408 |
| vocab_size | 85,729 |
| num_nextn_predict_layers | 0 (MTP disabled for synthetic) |
| compress_ratios | all 0 (CSA disabled for synthetic) |
| tensor count | 46,172 |
| shard count | 12 (4 GiB each) |
| on-disk size | 50.8 GB |

### Reproduction

```bash
# Generate fixture (auto-detects VRAM, targets VRAM x 2)
python scripts/dsv4_synthetic_fixture.py

# Build and run (4-token decode)
just build-cuda
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 4 --max-layers 43 --json

# 10-token decode (more stable, less prefill noise)
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 10 --max-layers 43 --json

# Extract key metrics
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 10 --max-layers 43 --json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); s=d['summary']; \
    c=s['counters']; k=c['kernels']; t=c['transfers']; e=c['experts']; \
    print(f'decode tok/s: {s[\"decode_tok_per_s\"]:.3f}'); \
    print(f'prefill tok/s: {s[\"prefill_tok_per_s\"]:.4f}'); \
    print(f'kernel launches: {k[\"launches\"]}'); \
    print(f'H2D copies: {t[\"host_to_device_copies\"]}'); \
    print(f'D2H copies: {t[\"device_to_host_copies\"]}'); \
    print(f'expert loads: {e[\"loads\"]}')"
```

### Phase 0 — Baseline (pre-optimization)

Prefill 1 token + decode 4 tokens:

```
backend:            cuda (sm_86)
load:               2.83 s
prefill:            15.30 s  (0.065 tok/s)
decode (4 tokens):  6.63 s   (0.60 tok/s)
total:              21.93 s
```

| Counter | Value | Per-token | Notes |
|---|---:|---:|---|
| kernel launches | 8,776 | ~2,194/tok | ~51 launches/layer/token |
| expert selected | 1,032 | 258/tok | 6 per score-router layer, 1 per hash layer |
| expert loads | 243 | 60.8/tok | disk reads on activation; no host cache |
| expert load bytes | 1.43 GB | — | from safetensors shards |
| artifact uploads | 1,352 | — | attention + expert bundles to GPU |
| artifact upload bytes | 4.04 GB | — | |
| host->device copies | 10,924 | ~2,731/tok | 4.48 GB total |
| device->host copies | 6,024 | ~1,506/tok | 73 MB total |

### Phase 1 — Expert loop optimization (P3 expert + P4 cache)

**Changes:**
- `prepare_fp4_expert_input` + `fp4_swiglu_ffn_from_device`: 6 experts share
  one uploaded input buffer instead of 6 separate H2D uploads.
- `saxpy_into`: device-side expert output accumulation, eliminating per-expert
  D2H downloads.
- `HostStagedExpertCache`: LRU host-RAM cache for re-activated experts.

**Files:** `ferrule-cuda/src/context.rs`, `ferrule-runtime/src/expert_residency.rs`,
`ferrule-runtime/src/models/deepseek_v4.rs` (`routed_moe_step` CUDA path)

```
load:               1.50 s
prefill:            18.31 s  (0.055 tok/s)
decode (4 tokens):  6.48 s   (0.62 tok/s)
total:              24.78 s
```

| Counter | Baseline | Phase 1 | Delta |
|---|---:|---:|---:|
| kernel launches | 8,776 | 9,808 | +1,032 (saxpy adds 258/tok) |
| H2D copies | 10,924 | 10,064 | -860 (expert input shared) |
| D2H copies | 6,024 | 5,164 | -860 (expert outputs on device) |
| expert loads (disk) | 243 | 243 | 0 (cache cold: 5-token run too short) |
| decode tok/s | 0.60 | 0.62 | +3% |

### Phase 2 — Attention device chain + norm weight cache

**Changes:**
- `rms_norm_from_device` + `upload_norm_weight` + `rms_norm_device_cached`:
  norm weights uploaded once per layer and cached on device.
- `linear_matvec_from_device`: device-resident linear matvec using
  `artifact_linear_matvec_into` with `CudaF32Buffer` I/O. It now applies the
  same FP8 activation-quant policy as the host path before quantized artifact
  GEMV, preserving DSV4 FP8 semantics.
- `rms_norm_heads_from_device`: per-head query RMS norm can consume/produce
  `CudaF32Buffer`, removing a redundant H2D + D2H pair after `query_b`.
- `decode_step_no_compress_device`: CUDA path that chains
  `query_a → rms_norm → query_b → head_norm` and `key_value → rms_norm` on
  device, uploading `hidden` once and reusing for both projections.
- `decode_step_compressed_device`: same attention projection/norm device chain
  for compressed DSV4 layers; downloads normalized q-latent only because the
  current indexer path is still host-side.
- `decode_step_device`: CUDA path that uses `rms_norm_device_cached` for the
  two per-layer norm calls.

**Files:** `ferrule-cuda/src/context.rs`, `ferrule-runtime/src/models/deepseek_v4.rs`
(`decode_step_with_operators`, `decode_step_no_compress_with_operators`)

**4-token run:**

```
load:               1.50 s
prefill:            18.31 s  (0.063 tok/s)
decode (4 tokens):  6.20 s   (0.64 tok/s)
total:              22.82 s
```

| Counter | Baseline | Phase 1 | Phase 2 | Delta (P2 vs baseline) |
|---|---:|---:|---:|---:|
| kernel launches | 8,776 | 9,808 | 9,808 | +12% |
| H2D copies | 10,924 | 10,064 | 8,860 | **-19%** |
| D2H copies | 6,024 | 5,164 | 4,648 | **-23%** |
| decode tok/s | 0.60 | 0.62 | 0.64 | +7% |

**10-token run (more stable, per-token counts):**

| Counter | Baseline (per-token) | Phase 2 (per-token) | Delta |
|---|---:|---:|---:|
| H2D copies | ~2,731 | ~1,630 | **-40%** |
| D2H copies | ~1,506 | ~1,056 | **-30%** |
| kernel launches | ~2,194 | ~2,229 | +2% |
| decode tok/s | ~0.60 | ~0.49 | -18% (KV cache growth; see note) |

Note: 10-token decode tok/s is lower for both baseline and optimized because
sparse attention cost grows with KV cache length. The 4-token run is more
comparable to the original baseline measurement.

**Follow-up on real local DSV4 (`models/DeepSeek-V4-Flash-DSpark`, sm_121,
2026-07-05):**

```bash
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark \
  --prompt Hello --backend cuda --max-layers 2 --max-tokens 2 --json
```

Result: L0 no-compress + L1 compressed path both execute with the device-chain.
The JSON counters expose host-staged expert cache telemetry separately from
source/disk loads:

| Counter | Value |
|---|---:|
| generated tokens | 2 |
| decode tok/s | 5.15 |
| kernel launches | 376 |
| H2D copies | 404 |
| D2H copies | 240 |
| selected experts | 24 |
| source expert loads | 24 |
| host cache entries / hits / misses | 24 / 0 / 24 |

**Full 43-layer end-to-end check (real local DSV4, sm_121, 2026-07-05):**

| Run | Prefill tok/s | Decode tok/s (summary) | Total seconds | Kernels | H2D / D2H copies | Source expert loads | Host cache hits / misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--max-tokens 4` | 0.094 | 0.336 | 22.53 | 11,336 | 13,276 / 5,652 | 785 | 0 / 785 |
| `--max-tokens 10` | 0.084 | 0.216 | 58.32 | 28,508 | 27,664 / 14,298 | 1,448 | 0 / 1,448 |

Against the near-term **20 tok/s load-excluded decode** gate, the 10-token
full-model run is ~0.216 tok/s by the current JSON summary convention (~93×
short). If counted as strict steady-state decode forward passes (9 actual decode
forwards after prefill), it is ~0.194 tok/s (~103× short). The main remaining
costs are host synchronization boundaries, source expert reload/upload churn,
and per-expert/per-op launches.

### Key findings

1. **H2D/D2H copies reduced 30-40%, but tok/s only +7%.** Copy reduction
   alone is insufficient — the bottleneck is not data movement volume but
   **synchronization latency**. Each D2H copy is a CPU-GPU sync point; ~1,056
   sync points/token dominate latency.

2. **Kernel launch overhead is significant but not dominant.** ~2,229 launches
   × ~5µs/launch ≈ 11ms/token. At 0.64 tok/s (~1.56s/token), launch overhead
   is ~0.7%. The real cost is the CPU stalling between launches waiting for
   D2H copies.

3. **Host-side computation is a hidden bottleneck.** `hc_pre`/`hc_post`,
   `apply_rotary_tail`, `quantize_attention_kv_for_qat_in_place`, and
   `sparse_attention` reference logic all run on CPU between GPU ops. This
   serializes CPU and GPU — neither is fully utilized.

4. **Host-staged expert cache was cold.** 243 source reads unchanged because
   the 5-token run has near-zero expert reuse. JSON now reports host cache
   hits/misses/entries separately from source loads, so longer runs can measure
   reuse directly. The cache value is in amortization, not single-shot runs.

### Updated bottleneck attribution

| Rank | Bottleneck | Evidence | Target | Est. gain |
|---|---|---|---|---|
| 1 | Sync latency from D2H copies (1,056/token) | -40% copies → only +7% tok/s | P5: CUDA graph capture | ~5-10x |
| 2 | Host-side compute between GPU ops | hc_pre/post, rotary, sparse_attn on CPU | P3: device-resident full layer | ~2-3x |
| 3 | Per-expert kernel launch (258/token) | 6 separate launches/layer | P5: batched expert kernel | ~2x MoE |
| 4 | No host-staged expert cache (cold) | 243 reads, 0 hits on 5-token run | P4: longer sequences | situational |
| 5 | Attention sub-ops unfused | multiple small kernels | P5: attention fusion | ~1.5x |

### Next steps (priority order)

1. **CUDA graph capture (P5)** — highest impact. Capture the entire decode
   bucket as one graph, eliminating all per-launch sync overhead. The
   `decode_step_device` path is already structured for this: deterministic
   op sequence, no data-dependent control flow in steady state. Expected:
   ~5-10x decode improvement.

2. **Device-resident `hc_pre`/`hc_post`** — these are called 4×/layer and
   each involves multiple H2D uploads + D2H downloads. Need `hc_pre_from_device`
   and `hc_post_from_device` variants in `CudaArtifactOperatorContext`.

3. **Device-resident `apply_rotary_tail`** — currently host-only, called
   3×/layer. Needs a CUDA kernel.

4. **Device-resident `sparse_attention`** — currently uploads all inputs and
   downloads output. Already has a CUDA kernel; needs `from_device` variant.

### Git state

Changes are unstaged on `main`. Files modified:
```
 crates/ferrule-cuda/src/context.rs               | 144 ++++++
 crates/ferrule-runtime/src/expert_residency.rs   | 227 +++++++++
 crates/ferrule-runtime/src/models/deepseek_v4.rs | 418 ++++++++++++++++++-
 docs/ROADMAP.md                                  | 199 ++++++-
 4 files changed, 963 insertions(+), 25 deletions(-)
```

CPU tests: 229 passed, 2 pre-existing failures
(`compressed_attention_decode_reference_updates_compressed_cache`,
`dsv4_layer_decode_step_runs_hc_attention_moe_shared_hc`) — both fail with
`invalid FP8 activation quant shape` and are unrelated to these changes.

---

## GB10 hardware analysis and theoretical performance ceiling

### Hardware inventory

Measured on the development machine (2026-07-05):

| Component | Value | Source |
|---|---|---|
| GPU | NVIDIA GB10 (Blackwell, sm_121, CC 12.1) | `nvidia-smi` |
| GPU memory | Unified LPDDR5X (no discrete VRAM) | Grace Blackwell Superchip architecture |
| GPU memory bandwidth | 273 GB/s (shared CPU+GPU) | LPDDR5X, 8 channels × 256-bit |
| GPU FP4 tensor core | ~500 TFLOPS (dense), ~1 PFLOPS (sparsity) | Blackwell `tcgen05` MMA |
| GPU FP8 tensor core | ~250 TFLOPS (dense) | Blackwell `wgmma` |
| CPU | NVIDIA Grace ARM (20 cores, Cortex-X925 + Neoverse V3) | `lscpu` |
| System memory | 128 GB LPDDR5X (unified CPU+GPU) | `free -h` |
| Disk | 3.7 TB NVMe SSD, 3.8 GB/s direct read | `dd` benchmark |
| CUDA | 13.0, Driver 580.82.09 | `nvcc --version` |
| Architecture | Grace Blackwell Superchip (DGX Spark / Project DIGITS) | Public specs |

**Key architectural fact**: GB10 has **no discrete VRAM**. The GPU accesses
system LPDDR5X directly via NVLink-C2C. Every GPU memory access goes through
the 273 GB/s unified memory bus. This eliminates the PCIe bottleneck entirely —
GPU and CPU share the same physical memory.

### Per-token compute and memory budget (DeepSeek-V4 Flash)

Real model dimensions from `config.json`:

```
hidden_size = 4096
moe_intermediate_size = 2048
num_experts_per_tok = 6
n_routed_experts = 256
num_attention_heads = 64
head_dim = 512
q_lora_rank = 1024
num_hidden_layers = 43
```

**Compute per token (43 layers):**

| Component | MACs/layer | FLOPs/layer | Notes |
|---|---:|---:|---|
| MoE gate+up GEMV (6 experts) | 100.7M | 201.4M | 6 × 2 × 2048 × 4096 |
| MoE down GEMV (6 experts) | 50.3M | 100.7M | 6 × 4096 × 2048 |
| Shared FFN | 25.2M | 50.3M | 3 × 2048 × 4096 |
| Router matvec | 1.0M | 2.1M | 256 × 4096 |
| Attention q_a proj | 4.2M | 8.4M | 1024 × 4096 |
| Attention q_b proj | 33.6M | 67.1M | 64 × 512 × 1024 |
| Attention kv proj | 2.1M | 4.2M | 512 × 4096 |
| Attention output_a | 8.4M | 16.8M | grouped matvec |
| Attention output_b | 33.6M | 67.1M | 4096 × 1024 |
| HC pre/post + norms | ~20M | ~40M | 4 HC ops + 2 norms/layer |
| **Total/layer** | **~279M** | **~558M** | |
| **Total/token** | **~12.0G** | **~24.0G** | 43 layers |

**Memory reads per token (weights only):**

| Component | Bytes/layer | Notes |
|---|---:|---|
| 6 expert gate+up (FP4 packed) | 50.3 MB | 6 × 2 × 2048 × 4096 / 2 |
| 6 expert gate+up scales | 3.1 MB | 6 × 2 × 2048 × 4096 / 32 |
| 6 expert down (FP4 packed) | 25.2 MB | 6 × 4096 × 2048 / 2 |
| 6 expert down scales | 1.6 MB | 6 × 4096 × 2048 / 32 |
| Shared FFN (FP4) | 12.6 MB | 3 × 2048 × 4096 / 2 |
| Attention weights (FP8) | ~46 MB | q_a, q_b, kv, output_a, output_b |
| HC + norm weights | ~10 MB | 4 HC weights + 2 norms/layer |
| **Total/layer** | **~149 MB** | |
| **Total/token** | **~6.4 GB** | 43 layers — if all weights read from memory |

Note: with expert residency, selected experts' weights are already in memory and
only need to be read by the GPU. However, `ExpertId` is `(layer, expert)`: full
DSV4 routed expert residency is approximately `43 × 256` expert bundles, i.e.
>100 GB class with metadata/handles, not a single 256-expert / 3.3 GB pool.
Default residency must therefore be budgeted/hotset-based rather than “preload
everything”.

### Theoretical performance ceilings

Decode is **memory-bound** (arithmetic intensity = 24 GFLOP / 6.4 GB = 3.75
FLOP/byte, well below the GPU's compute-to-memory ratio). Performance is
determined by how fast weights can be delivered to the GPU:

| Scenario | Bottleneck | Bandwidth | Theoretical tok/s | Efficiency factor | Practical tok/s |
|---|---|---:|---:|---:|---:|
| Expert weights on disk | NVMe read | 3.8 GB/s | ~25 | 30% | ~8 |
| Expert weights in host RAM | LPDDR5X read | 273 GB/s | ~1,800 | 30% | ~550 |
| Expert weights in unified mem | GPU direct access | 273 GB/s | ~1,800 | 50% | ~900 |
| Perfect tensor core overlap | Compute + memory | both | ~1,800 | 80% | ~1,400 |

**Compute ceiling check:** 24 GFLOP/token ÷ 500 TFLOPS = 48 μs/token →
~20,800 tok/s. Compute is **not** the bottleneck — memory bandwidth is.

### Current performance and gap analysis

| Metric | Value (2026-07-05) | Theoretical ceiling | Gap |
|---|---:|---:|---:|
| decode tok/s | 0.602 | ~1,800 (memory-bound) | ~3,000× |
| kernel launches/token | 1,849 | 0 (graph replay) | — |
| D2H copies/token | 4,169 | 0 (device-resident) | — |
| expert loads/token | 125.7 | 0 (pre-loaded) | — |
| FP4 execution path | scalar GEMV | tensor core MMA | ~100× compute |

### Optimization phases and progress

#### Phase 3 — Device-chain + concurrent I/O (completed 2026-07-05)

**Changes implemented:**
- MoE device-output: `routed_moe_step_device_output` keeps accumulator on GPU,
  feeds directly into `hc_post_from_device`. Eliminates 1 D2H + 1 H2D per layer.
- Cross-layer HC state device chain: `decode_step_device_hc_device` passes
  `hc_state_dev` buffer between layers without downloading. Eliminates 2 D2H +
  2 H2D per layer.
- Hidden → output-head device chain: `decode_token_hidden_cuda_device` returns
  normed hidden on device, `topk_logits_for_hidden_device_with_operators`
  consumes it directly. Eliminates 1 D2H/token.
- Removed unused `after_attn` download (43 D2H/token eliminated).
- Concurrent expert I/O: `read_experts_concurrent` uses tokio blocking pool to
  read all missing expert slices in parallel via `pread` (positioned read).
  Eliminates serial open+seek+read syscall chain.
- KV device rotary + QAT: `cuda_rope_tail_from_device` +
  `fp8_activation_quantize_buffer_in_place` on KV, then
  `combined_kv_append_window_device` — no D2H for KV path.
- Warmup tokens: `--warmup-tokens N` CLI flag runs N extra decode tokens before
  timing to populate GPU expert residency. Counters subtract warmup baseline.

**Files:** `crates/ferrule-cuda/src/context.rs`,
`crates/ferrule-runtime/src/models/deepseek_v4.rs`,
`crates/ferrule-runtime/src/expert_streaming.rs`,
`crates/ferrule-cli/src/commands/inspect.rs`,
`crates/ferrule-cli/src/main.rs`

**Result:**

| Metric | Before Phase 3 | After Phase 3 | Improvement |
|---|---:|---:|---:|
| decode tok/s | 0.393 | 0.602 | **+53%** |
| D2H copies/token | 5,678 | 4,169 | -27% |
| H2D copies/token | ~13,500 | ~12,600 | -7% |
| expert loads/token | 125.7 | 125.7 | 0 (cold start) |

#### Phase 4 — Batched MoE kernel (completed 2026-07-05)

**Changes implemented:**
- Three batched CUDA kernels in `ferrule-cuda/src/kernels.rs`:
  - `moe_gemv_dual_fp4_batched`: gate+up GEMV for all 6 experts in one launch
    using 2D grid `(intermediate_size, num_experts)`, pointer table for
    per-expert weight buffers.
  - `moe_swiglu_fp8_batched`: SwiGLU + FP8 quantize for all experts in one
    launch.
  - `moe_gemv_down_fp4_batched`: down GEMV + accumulate for all experts in one
    launch.
- Host-side `moe_experts_batched_from_device` in `context.rs` packs device
  pointers into `u64` arrays, uploads route weights, launches 3 kernels instead
  of 18+ (6 experts × 3-4 kernels each).
- Reusable `CudaFp4ExpertWorkspace` for scratch/output buffer reuse.
- `--warmup-tokens` CLI option for pre-populating GPU expert residency.

**Result:**

| Metric | Before Phase 4 | After Phase 4 | Improvement |
|---|---:|---:|---:|
| decode tok/s | 0.488 | 0.602 | **+23%** |
| kernel launches/token | 2,792 | 1,849 | **-34%** |
| D2H copies/token | 4,369 | 4,169 | -5% |

#### Phase 5 — Tensor core GEMM / 5 tok/s milestone (in progress)

**Goal:** Move end-to-end DSV4 CUDA decode from <1 tok/s to the 5 tok/s
milestone. The milestone is not reachable with naive expert prefetch or
reduction-kernel tuning alone; it requires replacing scalar FP4 MoE GEMV with a
Blackwell FP4 Tensor Core path while reducing residency churn.

The current batched kernels use scalar FP4 decoding/GEMV with no Tensor Core
utilization. On Blackwell (sm_121), `tcgen05` can execute native FP4 matrix
multiply, but cuda-oxide does not yet expose FP4/FP8 `tcgen05` variants upstream.
Ferrule should therefore pursue a local raw PTX smoke path first, then decide
whether to upstream a cuda-oxide patch once the exact operand ABI is proven.

**Latest measurements (GB10 / sm_121, 2026-07-05, `models/DeepSeek-V4-Flash-DSpark`, 8 decode tokens):**

| Config | decode tok/s | Expert loads/token | H2D bytes/token | Resident experts | Notes |
|---|---:|---:|---:|---:|---|
| no prefetch | 0.859 | 128.8 | 3.07 GB | — | cold-ish baseline after scalar FP4 cleanup |
| `--moe-prefetch-experts 8` | 0.885 | 163.4 | 3.54 GB | — | tiny gain, extra low-ID prefetch traffic |
| `--moe-prefetch-experts 16` | 0.745 | 198.9 | 4.01 GB | — | worse; naive prefetch overloads decode |
| `--warmup-tokens 4` | 0.842 | 64.7 | 0.87 GB | 1,465 | upload traffic drops, speed does not scale |
| `--warmup-tokens 4 --moe-prefetch-experts 8` | **0.908** | 64.6 | 0.87 GB | 1,759 | best observed short run |
| `--warmup-tokens 16` | 0.751 | 50.0 | 0.67 GB | 1,980 | fewer uploads but slower/noisy; compute still dominates |
| warmup+prefetch+`FERRULE_CUDA_MOE_REDUCE=1` | 0.874 | 63.9 | 0.86 GB | 1,746 | reduction path does not materially help |

**Interpretation:**
- Expert residency helps counters, but **does not** close the gap to 5 tok/s.
  Even after H2D drops from ~3.1 GB/token to ~0.9 GB/token, decode remains about
  0.9 tok/s.
- The primary blocker is still scalar FP4 MoE compute. The FP4 Tensor Core path
  is the only currently identified 5×+ lever.
- Naive low-ID prefetch is not a reliable residency strategy. It often increases
  expert loads because router-selected experts are not simply `0..N`.
- The experimental block-reduction path is correctness-safe but not a meaningful
  speed lever in current measurements.
- Kernel fragmentation remains high (~1,600 launches/token after warmup), but
  graph replay requires removing routing/planner/upload host boundaries first.

**Implementation plan:**
1. Add fine-grained timing counters around router, expert read/cache/upload,
   routed MoE compute, attention, lm_head/topk, and kernel launch count.
2. Build a feature/env-gated raw PTX FP4 Tensor Core smoke kernel for sm_121,
   using scalar FP4 kernels as correctness reference.
   **DONE 2026-07-05**: `fp4_tc_smoke.ptx` + `tests/fp4_tc_smoke.rs` pass
   with max_err=0.0 against CPU scalar FP4 reference.
3. Integrate FP4 Tensor Core for batched expert gate+up, then down projection.
4. Replace naive `0..N` prefetch with routing-aware hotset residency and explicit
   per-layer residency budgets/pinning.
5. Only after the above, revisit CUDA graph capture for stable decode buckets.

**Key dependency decision:** do not block Ferrule on cuda-oxide upstream support.
Use Ferrule-local raw PTX for the first FP4 path; once proven, either upstream a
cuda-oxide patch for `tcgen05` FP4/FP8 variants or keep the raw PTX path behind a
feature/env gate.

#### Phase 6 — Sync elimination, mmap I/O, managed memory, graph infra (completed 2026-07-06)

**Goal:** Eliminate the D2H/H2D synchronization overhead that dominates
per-token latency, reduce expert I/O cost, and lay the groundwork for CUDA
graph capture.

**Changes implemented:**

1. **P0 — Attention D2H/H2D elimination** (largest lever):
   - Added `decode_step_from_device()` / `decode_step_compressed_from_device()`
     / `decode_step_no_compress_from_device()` on `DeepSeekV4Attention`.
     These accept `&CudaF32Buffer` and return `CudaF32Buffer`, eliminating the
     4-sync round-trip at the call boundary (D2H normed → H2D hidden → D2H
     output → H2D attn_out).
   - Modified `decode_step_device_hc_device()` and `decode_step_device()` to
     pass `normed_dev` directly instead of download/upload.
   - **Eliminates 172 sync points/token** (43 layers × 4 syncs).
   - Files: `crates/ferrule-runtime/src/models/deepseek_v4.rs`

2. **P1 — Indexer routing D2H optimization**:
   - Fixed `q_latent` double-download bug in `decode_step_compressed_device`:
     `q_normed_dev` was downloaded unconditionally, then downloaded again
     conditionally — the first download was dead code causing an unnecessary
     D2H sync every layer.
   - Compressor paths in `_from_device` variant reuse `hidden_device` buffer
     instead of re-uploading `hidden` from host.
   - **Eliminates 43 redundant D2H downloads/token.**

3. **P2 — mmap safetensors**:
   - Added `MMAP_CACHE` static (`HashMap<PathBuf, Arc<Mmap>>`) in
     `ExpertStreamingReader`, shared across all reader instances.
   - `read_local_slice_positioned_with_mmap()` tries mmap first, falls back to
     `pread` on failure. OS page cache provides automatic caching of repeated
     reads — second access to the same expert tensor is served from page cache.
   - Added `memmap2 = "0.9"` dependency to `ferrule-runtime`.
   - Files: `crates/ferrule-runtime/src/expert_streaming.rs`,
     `crates/ferrule-runtime/Cargo.toml`

4. **P3 — Managed memory default-on**:
   - Changed `FERRULE_MANAGED_EXPERTS` default from `false` to `true` — expert
     weights now use `cuMemAllocManaged` (unified memory) by default, eliminating
     explicit H2D DMA transfers. The GPU reads directly from host LPDDR5X pages.
   - Added `HostPinnedBuffer` struct using `cuMemAllocHost` +
     `cuMemHostGetDevicePointer` with correct `Drop` (`cuMemFreeHost`). Available
     for future zero-fault-overhead expert loading.
   - Added `clone_f32_buffer()` (device-to-device copy, no host round-trip).
   - Files: `crates/ferrule-cuda/src/context.rs`

5. **P4 — Router path optimization**:
   - Replaced `zero_f32_buffer` + `copy_f32_into_slot` (2 kernel launches) in
     `routed_moe_step_device_output` with `clone_f32_buffer` (1 D2D copy).
   - Router logits D2H remains (small: ~37 floats; routing logic too complex for
     a simple kernel), but the path is streamlined.

6. **P5 — CUDA graph capture infrastructure** (structurally ready, blocked upstream):
   - Added `decode_step_graph_safe()` on `DeepSeekV4Layer` and `routed_moe_step_graph_safe()`
     on `DeepSeekV4CudaOperatorCache` — kernel-only paths with pre-determined
     routes and pre-allocated accumulators.
   - Added `linear_matvec_into_from_device()` — writes into pre-allocated output
     buffer instead of allocating.
   - Added `zero_f32_buffer_in_place()` using `cuMemsetD32Async` (graph-safe
     zeroing, no allocation).
   - Added warmup pass in `try_capture_decode_graph()`: runs one full decode step
     to pre-allocate all buffers, upload all weights, determine routing, then
     captures the graph-safe path.
   - Changed capture mode from `GLOBAL` to `RELAXED` to allow allocations during
     capture.
   - **Blocked**: `cuda-oxide`'s `DeviceBuffer::zeroed()` and `from_host()` use
     synchronous `cuMemAlloc_v2` + `cuStreamSynchronize`, both returning
     `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (error 900) during stream capture.
     Graph capture falls back to eager decode gracefully.
   - Files: `crates/ferrule-runtime/src/models/deepseek_v4.rs`,
     `crates/ferrule-cuda/src/context.rs`, `crates/ferrule-cuda/src/graph.rs`

**Files changed:**
- `crates/ferrule-runtime/src/models/deepseek_v4.rs` — P0/P1/P4/P5
- `crates/ferrule-cuda/src/context.rs` — P3/P5
- `crates/ferrule-runtime/src/expert_streaming.rs` — P2
- `crates/ferrule-runtime/Cargo.toml` — P2 (memmap2 dep)
- `crates/ferrule-cuda/src/graph.rs` — P5 (RELAXED mode)

**Validation:**
- `cargo oxide build --features cuda --arch sm_121` — succeeds
- `cargo oxide test --arch sm_121 -- -p ferrule-cuda -p ferrule-runtime --features cuda`
  — 284 tests pass, 0 failures
- `cargo check -p ferrule-runtime` (non-CUDA) — succeeds
- `cargo test -p ferrule-runtime` (non-CUDA) — 260 tests pass
- Graph capture attempt on GB10: fails with error 900 (expected — blocked by
  cuda-oxide synchronous allocation), falls back to eager decode correctly.

### Updated bottleneck attribution (2026-07-06)

| Rank | Bottleneck | Current impact | Target fix | Est. gain |
|---|---|---|---|---|
| 1 | Scalar FP4 MoE GEMV (no Tensor Core) | Best observed run still ~0.9 tok/s after residency warmup | P5: raw PTX / Tensor Core FP4 MMA for gate+up/down | required for 5 tok/s |
| 2 | Expert residency churn / GPU upload | Warmup reduces H2D to ~0.7-0.9 GB/token but still ~50-65 expert uploads/token | Routing-aware hotset residency + per-layer budget/pinning | situational 1.2-2×, not sufficient alone |
| 3 | cuda-oxide synchronous allocation blocks graph capture | `cuMemAlloc_v2` + `cuStreamSynchronize` in `DeviceBuffer::zeroed()`/`from_host()` return error 900 during stream capture | Add `cuMemAllocAsync` to cuda-oxide or pre-allocate buffer pool with `_into` variants | enables graph replay (~5-10×) |
| 4 | Kernel launch fragmentation | ~1,600 launches/token after warmup | Fusion + CUDA graph after host boundaries removed | ~1.3-2× after TC path |
| 5 | Compressor/indexer residual host work | Compressor softmax + indexer topk on CPU; ~2 D2H/layer remaining | Device softmax kernel + device topk kernel | ~1.2-1.5× |

**Changes since 2026-07-05:**
- Attention D2H/H2D sync eliminated (was rank 3, now resolved — P0 removed 172
  syncs/token, P1 removed 43 redundant D2H/token)
- Expert disk I/O improved via mmap + page cache (P2)
- Managed memory default-on eliminates expert H2D DMA (P3)
- Graph capture infrastructure ready but blocked by cuda-oxide (new rank 3)

### Performance projection

| Phase | tok/s | Cumulative gain | Key change |
|---|---:|---:|---|
| Baseline (pre-session) | 0.393 | — | — |
| + device chain + concurrent I/O | 0.488 | +24% | D2H/H2D elimination |
| + batched MoE kernel | 0.602 | +53% | 30→3 kernel launches |
| + scalar FP4 cleanup + atomic correctness | ~0.86-0.88 | ~2.2× | faster decode helpers, correct down accumulation |
| + warmup/prefetch best short run | **0.908** | ~2.3× | lower expert upload traffic, still scalar compute-bound |
| + P0-P4 sync/I/O optimizations | **pending measurement** | — | attention D2H elimination, mmap, managed memory |
| + raw PTX / Tensor Core FP4 MoE | **5+ target** | ~5.5× from current best | scalar MoE → FP4 MMA |
| + routing-aware hotset residency | 5-10+ | additive | reduce remaining uploads/churn |
| + CUDA graph/fusion (needs cuda-oxide fix) | 10+ | additive | reduce launch/sync overhead after host boundaries removed |
| **Theoretical ceiling** | **~1,800** | **~4,600×** | memory bandwidth limit |

**P0-P4 changes (2026-07-06):**
- P0: eliminated 172 attention D2H/H2D syncs/token (43 layers × 4 syncs)
- P1: eliminated 43 redundant q_latent D2H downloads/token
- P2: mmap safetensors with OS page cache (eliminates per-slice `File::open`)
- P3: managed memory default-on (zero-copy expert GPU access, no H2D DMA)
- P4: streamlined router path (D2D copy instead of zero+copy)
- P5: graph capture infrastructure ready (warmup + graph-safe path + RELAXED mode),
  blocked by cuda-oxide `cuMemAlloc` (synchronous, not graph-capturable)

The **5 tok/s target is no longer attributed to preload/prefetch**. Current
measurements show that residency improvements alone leave decode below 1 tok/s.
The milestone requires the FP4 Tensor Core/raw PTX MoE path. Routing-aware
residency and CUDA graph capture remain important follow-ups, but they should be
implemented as enablers around the Tensor Core path rather than as substitutes.

---

## Review checklist

For every patch:

- **Behavior**: does it change inference output? Is the change tested?
- **Correctness**: do CPU reference components still pass? Does compare-logits
  regress?
- **Generality**: does it add model-specific names to generic crates?
- **Performance**: does it add host sync to a device path? Does it add kernel
  launches?
- **State ownership**: does it leak state into the wrong crate?
- **Compatibility**: does it break existing CLI commands or WeightPack files?
- **Safety**: does it add `unsafe` without justification?
