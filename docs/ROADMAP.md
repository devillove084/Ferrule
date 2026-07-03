# Ferrule Roadmap

_Last updated: 2026-07-03_

Ferrule's current engineering center is **runtime graph backed Transformer execution**, with **DeepSeek-V4 Flash / DSpark** as the near-term pressure test. DeepSeek-V4 should drive missing runtime capabilities, but it must not become a model-specific runtime architecture. Model-family names and raw tensor names stay in `ferrule-model`; `ferrule-runtime` should see semantic roles, graph ops, artifact groups, expert registries, KV state, execution batches, and backend object stores. Benchmark/report/reference-smoke tooling lives outside runtime in `ferrule-bench`.

The immediate strategy is:

1. make the runtime graph contract executable and inspectable;
2. connect `GraphProgram + BackendObjectStore` to existing layer components through a generic execution bridge;
3. after the coarse graph path executes, split `transformer_layer` into finer semantic ops;
4. keep correctness gates ahead of kernel/fusion work;
5. make DSV4 faster by removing host-mediated per-token boundaries;
6. then add batching, paged KV, serving, and speculation in the same graph/runtime framework.

---

## Current Capabilities

### Working now

- `ferrule-graph` opaque IR with `ComputeGraph`, `OpKey`, `ExternalKey`, graph templates, validation, and shape registry support.
- Runtime graph program bundle:
  - `GraphProgram`
  - `ExecutionBatch`
  - `ExternalBindingPlan`
  - `BackendObjectStore`
  - `ReferenceGraphBackend`
- Dense decoder graph translation path for standard MHA/GQA + dense MLP plans.
- Generic semantic Transformer graph path for non-dense capabilities:
  - `transformer_state_init`
  - `transformer_layer`
  - `output_projection`
  - semantic `ArtifactGroupKind`
  - `ExpertRegistry`
  - `KvState`
- Descriptor/runtime-plan graph builder that dispatches by capabilities, not by model family name.
- `ferrule-model` owns model-family tensor parsing and policy extraction.
- DSV4 local metadata path:
  - HF safetensors index/header inventory
  - attention / hyper-connection / router / shared expert / routed expert tensor descriptors
  - local-model tests that skip cleanly when the model is absent
- DSV4 artifact materialization through generic graph bindings:
  - attention artifact groups
  - layer/stage norm artifact groups
  - hyper-connection artifact groups
  - router artifact groups
  - shared expert groups
  - routed expert registries
- Generic DSV4-relevant semantics are extracted from config into runtime policy/attrs, including norm eps, HC eps/sinkhorn iters, RoPE/YaRN metadata, sliding/indexer attention metadata, grouped output metadata, SwiGLU limit, route scale, and hash-router layer count.
- Artifact terminology is now used for checkpoint/runtime payloads; old source-prefixed runtime naming has been removed.
- OLMoE remains a useful executable correctness fixture.
- CUDA crate remains the backend/kernel crate; runtime graph does not embed CUDA buffers, KV pages, or `WeightPack` objects in graph nodes.

### Main gaps

- The semantic graph is still coarse for DSV4: one `transformer_layer` node represents many hot-path operations.
- Graph backend lowering is not yet the production DSV4 execution path.
- DSV4 decode still has too many host-side `Vec<f32>` and synchronous artifact/operator boundaries.
- KV/compressed KV/indexer state is not fully device-resident and graph-addressable.
- Routed expert registry exists, but production expert residency, prefetch, batched selected-expert execution, and GPU handle reuse still need to be wired into graph execution.
- DSV4 attention semantics are represented generically at the graph/policy boundary, but kernels still need better fused device-side execution: low-rank q path, RoPE/YaRN, sparse top-k attention, attention sink, grouped output projection, compressor/indexer support.
- Output projection/top-k should avoid materializing full vocab logits on host for greedy/common decode paths.
- CUDA graph capture/persistent decode schedules should come after stable device-resident state and shape buckets.
- Full correctness parity against official/reference DSV4 outputs is not complete.
- `ferrule-bench` now owns benchmark/report schemas and JSON summaries; remaining counter gaps are real CUDA kernel launch counts, host↔device bytes, expert load counts, and peak memory telemetry from backends.
- No production OpenAI-compatible server, continuous batching, paged KV, prefix/radix KV reuse, or DSpark speculation loop yet.

---

## Priority Roadmap

## P0 — Correctness and Observability Gates

Goal: before optimizing DSV4, make every mismatch and every performance bottleneck localizable.

- [x] Runtime graph validation and shape registry.
- [x] Generic semantic DSV4-capable graph builder without `deepseek_v4_graph` specialization.
- [x] Local DSV4 graph materialization smoke through `build_graph_program_from_descriptor` + `materialize_graph_hf_externals`.
- [x] `just fmt`, `just clippy-all`, `just test`, `just check` pass with CUDA through `cargo oxide`.
- [ ] DSV4 first-token / first-N-token parity gate against an official/reference path.
- [ ] Layer-scoped debug dump format for hidden, HC state, q/kv, attention output, router top-k, selected experts, and logits top-k.
- [x] Move benchmark/report/reference-smoke/perplexity tooling out of `ferrule-runtime` into `ferrule-bench`.
- [x] DSV4/bench JSON summary schema for load/prefill/decode time, tok/s, resident expert count/bytes, and graph/object summaries.
- [ ] Wire real backend telemetry into that schema:
  - kernel launches/token
  - host↔device bytes/token
  - expert loads/token
  - peak VRAM / host RSS
- [x] Make graph construction support a stable, serializable summary for diffing operator graph changes.

Exit criteria:

- a DSV4 run can fail with a named unsupported policy or a localized numeric mismatch;
- a DSV4 run can report where time and bytes are spent before we rewrite kernels.

---

## P1 — Graph Execution Bridge for Semantic Transformer Layers

Goal: make `GraphProgram + BackendObjectStore` drive coarse `transformer_layer` execution through existing generic components before splitting the op graph.

### P1.1 Layer object bundle

- [x] Coarse `transformer_state_init` → `transformer_layer` → `output_projection` path.
- [x] Semantic graph externals materialize into artifact groups, expert registries, and KV state.
- [x] Aggregate materialized externals into `GraphLayerObjects` by `ArtifactGroupKind` + layer.
- [x] Add typed slice-based binders from artifact groups:
  - `bind_attention_from_artifact_group`
  - `bind_hyper_connection_from_artifact_group`
  - `bind_router_from_artifact_group`
  - `bind_shared_swiglu_ffn_from_artifact_group`
- [x] Preserve the current family-specific tensor parsing boundary in `ferrule-model`; runtime binders consume semantic artifact slices only.

### P1.2 Coarse `transformer_layer` lowering

- [x] Lower `transformer_layer` inside `ReferenceGraphBackend` as a whole-graph execution step, not as public per-op model APIs.
- [x] Convert `GraphLayerObjects + ArtifactTensorReader + policies` into existing layer payloads / execution state.
- [x] Reuse existing generic components:
  - semantic artifact binding;
  - hyper-connection reference functions;
  - latent/sparse attention reference anchors;
  - router and routed MoE execution;
  - expert streaming / residency planners;
  - KV state handles.
- [x] Cache decoded payloads per graph program/layer so decode does not rebind artifacts per token in the reference graph backend.

### P1.3 Execution proof

- [x] Run a tiny/synthetic semantic layer through the graph path.
- [ ] Run local DSV4 layer 0 through the graph bridge when the model is present.
- [ ] Add debug dumps around the coarse layer boundary: hidden/HC state, attention output, router top-k, selected experts, logits top-k.

Acceptance:

- no runtime graph op or external key contains `deepseek` or raw HF tensor names;
- graph execution can reach existing DSV4-capable components through generic layer objects;
- unsupported policies fail with named semantic reasons, not missing-family branches.

---

## P2 — Split `transformer_layer` into Fine-Grained Semantic Ops

Goal: after the coarse graph path executes, replace `transformer_layer` with a layered graph that exposes performance-critical boundaries while staying model-family neutral.

### P2.1 Semantic layer skeleton

- [ ] Split `transformer_layer` into generic phase ops:
  - `layer_hc_pre`
  - `rms_norm`
  - `latent_attention`
  - `layer_hc_post`
  - `router_select`
  - `routed_moe`
  - `shared_ffn`
  - `residual_merge`
- [x] Add coarse `transformer_layer` graph attrs that describe semantics, not family names:
  - attention kind: `multi_latent_attention`, `grouped_query`, etc.
  - KV shape: `full_kv`, `latent_or_compressed`
  - router kind: `hash_assisted_topk`, `dense_topk`
  - norm/HC eps and HC sinkhorn iters
  - RoPE/YaRN metadata
  - sliding/indexer/compression metadata
  - SwiGLU limit and route scale
- [ ] Carry the same attrs through the future fine-grained semantic ops, including residency mode and precision/artifact dtype classes.
- [ ] Add shape inference for each semantic op.
- [ ] Add validation rules that reject raw HF tensor names in external keys.

### P2.2 Attention / position / cache subgraph

Expose these as generic attention/position/cache ops, not `deepseek_*` ops:

- [ ] `latent_q_project_a`
- [ ] `latent_q_norm`
- [ ] `latent_q_project_b`
- [ ] `latent_kv_project`
- [ ] `latent_kv_norm`
- [ ] `rope_apply` / position policy attrs
- [ ] `window_kv_update`
- [ ] `compressed_kv_update`
- [ ] `indexer_score_topk`
- [ ] `sparse_attention`
- [ ] `attention_output_grouped_a`
- [ ] `attention_output_b`

Acceptance:

- tiny/synthetic graph can validate these ops;
- local DSV4 layer 0/layer 2 can bind all externals through artifact groups;
- no op name or external key contains `deepseek` or raw HF names.

### P2.3 Routed/shared MoE subgraph

Expose MoE as generic routed/shared expert operations:

- [ ] `router_logits`
- [ ] `hash_router_lookup`
- [ ] `topk_select`
- [ ] `route_weight_normalize`
- [ ] `expert_registry_lookup`
- [ ] `routed_expert_swiglu`
- [ ] `shared_swiglu_ffn`
- [ ] `moe_accumulate`

Acceptance:

- hash-assisted first layers and score/bias later layers are represented by attrs/policies;
- `ExpertRegistry` remains the graph external for routed experts;
- graph does not explode into thousands of per-expert external tensors by default.

---

## P3 — Device-Resident DSV4 Decode

Goal: make DSV4 faster by removing host-mediated hot-path boundaries before doing broader serving work.

Priority order:

1. **Decode arena**
   - reusable device buffers for hidden, HC state, q/kv, attention workspace, router scores, route ids/weights, expert outputs, and logits/top-k;
   - graph/backend memory plan owns scratch lifetimes.
2. **Device-resident KV state**
   - window KV, compressed KV, indexer KV, and compressor state stay on device;
   - graph uses `KvState` handles rather than copying assembled KV vectors through host.
3. **Artifact linear dispatch**
   - generic artifact linear APIs for BF16/F32/FP8/FP4 payloads;
   - backend chooses CPU reference, CUDA dequant+matvec, or fused kernels.
4. **Expert residency**
   - persistent expert handles keyed by `ExpertRegistry` + layer + expert id;
   - selected experts reuse device handles when resident;
   - prefetch/evict policy is visible in metrics.
5. **Output projection/top-k**
   - chunked or fused device-side lm-head top-k for greedy/common decode;
   - copy back only token id and selected logit when possible.

Exit criteria:

- one-token DSV4 decode downloads only debug-gated tensors and final token/logit in normal mode;
- expert loads are visible and amortizable;
- decode speed bottleneck moves from host orchestration to measured kernels.

---

## P4 — Kernel Fusion and CUDA Graph Replay

Goal: optimize only after P0/P3 make correctness and data movement visible.

- [ ] Count kernel launches per token and per layer.
- [ ] Fuse low-rank attention projection + norm pieces where shape-stable.
- [ ] Fuse sparse attention gather/softmax/value accumulation with attention sink semantics preserved.
- [ ] Add batched selected-expert FP4/FP8 MoE kernel per layer.
- [ ] Add grouped output projection fusion for DSV4 attention output.
- [ ] Add steady-state decode shape buckets.
- [ ] Capture CUDA graph replay for stable decode buckets with debug fallback.

Exit criteria:

- launch count/token decreases measurably;
- CUDA graph replay improves latency without changing parity results;
- debug mode can still run uncaptured kernels for mismatch localization.

---

## P5 — Benchmarks and Competitive Parity Track

Goal: compare honestly against mainstream engines by scenario.

### Ferrule benchmark targets

- [x] DSV4 local generate path has JSON benchmark summary via `deepseek-v4-generate --json`.
- [ ] Dedicated `bench-infer` DSV4 mode that uses the graph bridge rather than the legacy DSV4 diagnostic command.
- [ ] prompt/decode split comparable to `llama-bench`.
- [ ] report model artifact, recipe, precision policy, hardware, CUDA arch, prompt, context length, batch size, and generated token count.
- [ ] graph summary + backend lowering summary in benchmark report.
- [ ] optional correctness note: first-token match, top-k overlap, or known unsupported policy.

### Initial PK matrix

| Scenario | Reference | Ferrule target |
|---|---|---|
| Local single-user DSV4 | llama.cpp / known working DeepSeek runner | load, TTFT, decode tok/s, memory, first-token parity |
| Quantized local MoE | llama.cpp / ExLlama-like local baselines where applicable | selected-expert latency, output quality smoke, VRAM |
| CUDA optimized decode | TensorRT-LLM / LMDeploy / TurboMind principles | kernel count/token, launch overhead, fused attention/MoE speed |
| Serving throughput | vLLM / SGLang | only after paged KV + scheduler exist |
| Out-of-core MoE | FlexGen / PowerInfer / LLM-in-a-Flash style systems | resident bytes, transfer bytes/token, tokens/s under memory budget |

---

## P6 — Serving Runtime

Goal: move from single-user graph execution to multi-session serving without invalidating P0-P5.

- [ ] Request/session/sequence lifecycle API.
- [ ] Paged KV allocator behind graph `KvState` handles.
- [ ] Chunked prefill.
- [ ] Continuous batching scheduler.
- [ ] Prefix/radix KV reuse.
- [ ] Minimal OpenAI-compatible `/v1/chat/completions` server.
- [ ] Streaming responses, cancellation, metrics.
- [ ] Structured output masks / JSON constraints.

Exit criteria:

- multiple sessions share the runtime safely;
- KV memory is bounded and observable;
- scheduler decisions are reproducible in tests.

---

## P7 — DSpark / Speculation

Goal: treat DSpark as generic speculation around a correct target model, not as a separate base runtime path.

- [ ] Represent MTP/draft artifacts as semantic `Speculation` bindings/attachments.
- [ ] Proposal model interface.
- [ ] Target verification interface.
- [ ] Acceptance/rejection/rollback state.
- [ ] Metrics:
  - proposed tokens
  - accepted tokens
  - rejected tokens
  - rollback count
  - effective tok/s
- [ ] Scheduler integration after serving state exists.

Rule: do not enable DSpark acceleration before base DSV4 correctness and device-resident decode are stable.

---

## Gap Against Mainstream Inference Engines

| Area | Mainstream engines have | Ferrule now | Ferrule gap |
|---|---|---|---|
| Model format/workflow | GGUF / engine plans / mature packaging | HF safetensors inventory, WeightPack direction, semantic graph bindings | WeightPack-only startup, manifests/checksums, conversion reports, GGUF compatibility story |
| Model coverage | many dense/MoE families | OLMoE executable fixture; DSV4 metadata/artifact graph materialization | full DSV4 graph-backed execution, broader dense/MoE adapters |
| Attention kernels | FlashAttention/FlashMLA-style fused kernels, paged KV | correctness-first attention/reference pieces and CUDA kernels | fused DSV4 latent/sparse attention, device-resident compressed/indexer KV |
| MoE execution | batched/fused expert kernels, expert parallelism, residency | expert streaming planner, artifact expert bundles, CPU/CUDA correctness paths | production selected-expert batching, residency/prefetch, GPU handle reuse |
| Quantization | GPTQ/AWQ/FP8/INT4/K/IQ formats, calibration | Q4/Q8 plus FP4/FP8 artifact decoders | calibration, mixed policy, conversion validation, quality gates |
| Scheduler | continuous batching, chunked prefill, preemption | scheduler prototypes but not graph-backed production serving | paged KV + request lifecycle + graph batch lowering |
| Prefix reuse | radix/prefix cache with KV sharing | radix/prefix modules exist but not production graph serving | integrate with paged KV/session scheduler |
| CUDA performance | fused kernels, CUDA graphs, memory planners | many small kernels and correctness-first paths | decode arena, lowering memory plan, fusion, capture |
| Correctness infrastructure | evals, perplexity, golden suites, reference parity | unit/local smokes, some compare tools | DSV4 official parity gates, long multi-turn regressions, benchmark correctness metadata |
| Serving UX | OpenAI API, streaming, metrics | CLI/local tools | server integration after graph runtime stabilizes |
| Distributed | TP/PP/EP, multi-node serving | none | design should not block, but not near-term |

Ferrule's intended differentiation is **explicit runtime state + artifact-aware MoE residency + Rust-native graph/runtime contracts**, not immediate parity with every backend and serving feature.

---

## Immediate Next Steps

1. Add the local DSV4 layer-0 graph bridge execution smoke once the generic reference layer path carries the remaining DSV4-specific semantics as model policies rather than graph specialization.
2. Add graph/layer summary output and DSV4 benchmark JSON with load/prefill/decode split and data-movement counters.
3. Start moving hot-path state to device-resident arenas: hidden/HC state, KV, router scores, selected expert outputs, logits/top-k.
4. Only after the coarse bridge is stable, split `transformer_layer` into fine-grained semantic ops and add shape inference for those ops.
6. Move decode state toward a device-resident arena before adding CUDA graph capture.
