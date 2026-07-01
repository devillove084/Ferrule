# Temporary Inference Engine Roadmap

> Status: working implementation roadmap, not the canonical product roadmap.
>
> Purpose: make Ferrule's gap against current inference frameworks explicit so
> implementation can proceed in small, reviewable patches.

Ferrule's current direction is **Ferrule Runtime**: a Rust-native, typed,
state-aware, hardware-aware LLM runtime. The near-term engineering target is a
clean execution architecture where model families provide metadata, tensor
layout, and policy objects, while the engine composes attention, routing,
experts, KV, quantization, scheduling, residency, parallelism, and speculation.
DeepSeek V4 is the first mainstream pressure-test for this model bring-up
contract, not a new architecture center. OLMoE remains the current executable
correctness fixture rather than the architecture center.

Current baseline:

- Executable correctness fixture: OLMoE / OLMoE-Instruct safetensors loading.
- CPU FP32 OLMoE reference inference for regression.
- GPU Q4_0 OLMoE inference through cuda-oxide kernels.
- llama.cpp-compatible Q4_0 block layout.
- explicit router → top-k experts → expert loop.
- correct OLMoE `norm_topk_prob=false` semantics as a regression fixture.
- single-session GPU KV cache behind explicit KV state objects.
- `WeightPack` for quantized layer weights.
- `ferrule-runtime` abstraction for CPU/GPU runners, session reset, sampling,
  and chat generation loops.
- llama.cpp-style sampling controls: temperature, top-k, top-p, min-p,
  repeat penalty, seed, stop strings, top-K logprobs.
- `generation_config.json` auto-loading with model defaults.
- Chat template registry: OLMoE-Instruct, ChatML, Llama3, Qwen, Plain.
- CLI commands: `info`, `run`, `gpu-run`, `chat`, `cuda`, `bench-infer`,
  `compare-logits`, `inspect-weightpack`, `server`.
- Model-family descriptor boundary exists: OLMoE fixture is executable;
  DeepSeek V4 has metadata inspection, complete tensor classification, HF shard
  inventory, expert source streaming, and a tiny CPU expert reference executor,
  but no full Transformer decode path yet.
- `ModelSupportContract` + `EnginePlan` skeleton exists and can describe dense,
  MoE, OLMoE, and DeepSeek-style layouts without making DeepSeek the generic
  runtime shape.
- `SourceArtifact` + `ConversionPlan` + `QuantizationRecipe` skeleton exists:
  official HF safetensors are source-of-truth, WeightPack is Ferrule's primary
  execution artifact, and GGUF is an optional compatibility/PK target.
- CUDA Transformer hot path is split into executor + attention/MoE/logits/KV
  steps; OLMoE is now one concrete weight/state container, not the generic
  runtime abstraction.
- Model-family tensor policy is now split under `ferrule-model/src/families/`:
  generic `tensor_policy` exposes semantic classes, while per-family files own
  concrete HF/GGUF names.
- DeepSeek V4 HF index inspection is wired for the local official model:
  72,317 tensors, 48 shards, 166.9GB source size, all referenced shards present,
  DSpark metadata detected, and no tensor remains `unknown` after classifying
  attention sinks plus DSpark/MTP projection/Markov/confidence-head tensors.
- DeepSeek V4 local smoke coverage exists in `ferrule-model` tests: it validates
  the DGX Spark model directory when present, but skips cleanly on machines that
  do not have the 166GB checkpoint.
- HF safetensors shard-header inventory exists: it reads dtype/shape/byte ranges
  without touching tensor payloads and feeds dtype counts into `ferrule info`.
- Expert streaming now registers DeepSeek routed expert tensor sets from HF
  safetensors, reads bounded byte ranges for selected experts, builds generic
  `ExpertComputeBundle`s, and has CPU reference tests for packed FP4 + E8M0
  SwiGLU expert math. This proves the source-preserving expert path, not full
  model inference.

---

## 1. Framework Gap Snapshot

### 1.1 llama.cpp

llama.cpp is the closest local-inference reference. It is strong because it has a
complete local workflow, not just fast kernels.

| Area | llama.cpp has | Ferrule has | Gap |
|---|---|---|---|
| Model format | GGUF single-file model, rich metadata, tokenizer/chat template info | safetensors loader, partial GGUF reader, WeightPack sidecar | WeightPack manifest and WeightPack-only startup; GGUF import/export story unclear |
| Model coverage | broad dense/MoE model families | one executable OLMoE fixture plus metadata descriptors for DeepSeek V4 | turn descriptors into model-family policies and executable layouts |
| Quantization | many mature formats: Q4_0/Q8_0, K-quants, IQ quants, mixed CPU/GPU support | Q4_0/Q8_0/Q2S/T1S primitives; Q4_0 path validated best | Q8 full-path validation, mixed precision, K-quant/AWQ-like quality track |
| CLI UX | prompt/chat, many sampling flags, templates, context controls | prompt/chat, initial sampling flags | generation config loading, template registry, context/window controls |
| Correctness tools | perplexity, eval examples, broad community regression coverage | OLMoE fixture smoke tests and CPU reference | logits diff, golden-token tests, perplexity command, DeepSeek reference harness |
| Benchmarks | `llama-bench`, prompt/decode split | CUDA GEMV probe, runtime stats internally | `bench-infer` with pp/tg split and reproducible output |
| Server | OpenAI-compatible server, streaming, embeddings | none | minimal `/v1/chat/completions` server |
| Grammar | grammar / JSON-constrained decoding | none | stop strings only; grammar FSM later |
| KV | mature local KV management, context shifting, some quant/offload paths | contiguous CPU/GPU KV state objects; still single-session on hot path | session object, KV stats, paged KV, DeepSeek-specific KV layouts |
| Portability | CPU, CUDA, Metal, Vulkan, SYCL, etc. | CUDA-focused plus CPU reference | keep CUDA first; design backend trait cleanly |

Ferrule should align with llama.cpp first in **workflow**:

1. load/cache reliably,
2. chat comfortably,
3. benchmark clearly,
4. compare quality automatically,
5. serve locally.

### 1.2 vLLM

vLLM is the main reference for high-throughput serving.

| Area | vLLM has | Ferrule has | Gap |
|---|---|---|---|
| Request model | request/sequence groups, lifecycle state | CLI-only single session | `Request`, `Session`, `SequenceState` abstractions |
| KV memory | PagedAttention block allocator | contiguous CPU/GPU KV state objects; still not paged | session-owned KV cache trait, then paged KV allocator |
| Scheduling | continuous batching, preemption, chunked prefill | one request at a time | scheduler loop after server exists |
| Prefix reuse | prefix cache support | none | token-prefix cache after paged KV |
| Serving | OpenAI API, streaming, metrics | none | minimal server, then scheduler integration |
| LoRA | dynamic adapter loading/serving | none | adapter state object later |
| Quantization | multiple quant backends | local Q4/Q8 kernels | per-backend quant policy and manifest |
| Distributed | tensor/pipeline parallel serving | none | not near-term; design should not block it |

Ferrule should not jump directly to full vLLM parity. The correct path is:

1. engine abstraction,
2. local server,
3. explicit sessions,
4. KV cache trait,
5. contiguous multi-session KV,
6. paged KV,
7. scheduler,
8. continuous batching.

### 1.3 SGLang

SGLang is the main reference for state reuse and structured generation.

| Area | SGLang has | Ferrule has | Gap |
|---|---|---|---|
| Programming model | high-level generation programs | CLI-only generation | maybe later: Rust generation DSL or API layer |
| Prefix reuse | RadixAttention / radix cache | none | prefix-keyed KV cache after paged KV |
| Structured output | JSON/schema/regex constraints | stop strings only | token mask / grammar FSM |
| Batching | serving scheduler integrated with cache reuse | none | depends on request/session scheduler |
| Speculative decoding | multiple speculative paths in serving stack | none | later after stable scheduler |
| Multi-turn state | server-managed conversation state | chat prompt fragments in one session | structured chat history + session state |

Ferrule's SGLang-alignment should start only after local server + KV abstraction.
The useful subset is:

1. prefix cache key,
2. radix tree over token prefixes,
3. KV page sharing,
4. structured decoding masks,
5. program-like generation API.

### 1.4 TensorRT-LLM / LMDeploy / TurboMind

These are kernel/compiler/performance references.

| Area | They have | Ferrule has | Gap |
|---|---|---|---|
| Kernel maturity | fused attention, fused MLP, GEMM/GEMV variants, CUDA graphs | many small custom kernels, per-token GEMV path | reduce launch count, fuse hot paths, CUDA graph replay |
| Quantization | INT8/INT4/FP8/AWQ/SmoothQuant/KV quant | Q4_0/Q8_0 primitives | accuracy-aware quant policy and calibration path |
| Batching | in-flight batching | none | scheduler + batchable kernels |
| Parallelism | tensor/pipeline/expert parallel | none | later; MoE expert-parallel design track |
| Deployment | production server engine | CLI only | server + metrics + config |

Ferrule should borrow principles, not immediately implement a compiler stack:

- measure kernel launch overhead first,
- fuse repeated small kernels,
- add CUDA graph capture for stable decode shapes,
- then consider batched GEMV/GEMM paths.

### 1.5 MLC-LLM / ExecuTorch / edge runtimes

These are portability references.

| Area | They have | Ferrule has | Gap |
|---|---|---|---|
| Backend portability | Vulkan/Metal/WebGPU/mobile/NPU paths | CUDA + CPU reference | backend trait and IR boundary before new backend |
| Compilation | model lowering and generated kernels | hand-written cuda-oxide kernels | keep explicit kernels now; define runtime object model cleanly |
| Mobile packaging | deployment artifacts | none | later WeightPack package format |
| Hardware abstraction | target-specific schedules | implicit CUDA policy | hardware policy layer later |

Ferrule's near-term edge advantage is not broad backend coverage. It is explicit
runtime state plus quantized/cacheable artifacts. Portability should be designed
into interfaces but implemented after CUDA path is stable.

### 1.6 ExLlamaV2 / high-speed single-user engines

These are references for optimized local quantized decode.

| Area | They have | Ferrule has | Gap |
|---|---|---|---|
| Single-user speed | highly optimized GPTQ/AWQ kernels | baseline Q4_0 GEMV kernels | faster quant matvec and fewer launches |
| Quant quality | GPTQ/AWQ formats | naive Q4_0/Q8_0 | activation-aware/offline quantization |
| UX | simple local generation | initial CLI | benchmark and config improvements |

This track matters if Ferrule wants strong single-user edge latency.

### 1.7 Offload systems: FlexGen / PowerInfer / LLM in a Flash

These are important for Ferrule's hardware-aware direction.

| Area | They have | Ferrule has | Gap |
|---|---|---|---|
| Offload policy | CPU/GPU/NVMe placement | none | expert-aware residency policy |
| Locality | hot/cold layer or neuron/expert locality | explicit MoE experts | expert activation counters and prefetch |
| Memory planning | explicit memory budget | fixed GPU allocation | memory planner + WeightPack chunk metadata |
| Edge deployment | out-of-core inference | none | WeightPack chunks + async transfer |

Ferrule has a natural advantage here because MoE experts are explicit runtime
objects. This should become a differentiating track after WeightPack-only startup and
basic benchmarking.

### 1.8 DeepSeek V4 / DSpark pressure-test targets

DeepSeek V4 is the near-term visibility target because it exercises the exact
runtime pieces Ferrule wants to own: DeepSeek-family hybrid attention
(CSA/HCA/MLA-compatible GGUF tensor families), sparse expert routing,
GGUF-side quantized weights, auxiliary routing/compression tensors, and
out-of-core/streaming pressure. It must be treated as a demanding implementation
of the generic model bring-up contract, not as a reason to add DeepSeek-shaped
state to the generic engine. DSpark should be modeled as speculative decoding /
draft-model support attached to a target model, not as a separate base
Transformer architecture unless actual metadata proves otherwise.

| Area | DeepSeek V4 / DSpark needs | Ferrule has | Gap |
|---|---|---|---|
| Model detection | `deepseek4` architecture, optional DSpark attachment metadata, GGUF metadata | `ModelDescriptor`, `ModelFamily::DeepSeekV4`, per-family HF/GGUF tensor policy, DSpark/MTP tensor roles | executable family policy without forking base execution path |
| Official source | `deepseek-ai/DeepSeek-V4-Flash-DSpark` HF safetensors, encoding scripts, inference scripts, 48 shards | `SourceArtifact::deepseek_v4_flash_dspark_official()` metadata fixture; local index summary confirms 72,317 tensors and all 48 shards present; shard-header inventory has dtype/shape/file offsets without loading payloads; typed HF descriptors now exist for attention, HC, router, shared experts, and routed experts | assemble executable per-layer `LayerSourceBinding`s and reference checks |
| Attention | DeepSeek hybrid attention / MLA-compatible projection layout and decode/prefill kernels | MHA/GQA RoPE attention step + `AttentionKind` boundary; `AttentionSourcePayload` binds real DSV4 layer-0/layer-2 HF source tensors; sparse attention reference + correctness-first CUDA sparse-attention ABI exist | q low-rank execution, rotary/inverse-rotary, compressed/sliding KV state, compressor/indexer execution, FP8 CUDA linears, and tiled sparse-flash kernels |
| Routing | hash-assisted first layers, score/bias top-k later layers, `sqrtsoftplus`, normalized weights, route scale | DeepSeek router policy has CPU reference semantics; real layer-0 hash router and layer-3 score/bias router bind from local HF shards | connect router output to executable layer runner and CUDA selected-expert path |
| Experts | routed experts, shared experts, source FP4 expert quant layout | generic expert streaming planner/reader; `ExpertComputeBundle`; CPU packed-FP4 SwiGLU executor; resident expert handle store; correctness-first CUDA packed-FP4 expert executor; real local shared/router bindings pass | production expert batching/residency policy in full decode loop, FP8 shared/attention CUDA linears, and performance tuning |
| Quantization | official FP8/I8/BF16/F32 source; reproducible Ferrule quant recipes; optional GGUF Q_K/IQ export | `QuantizationRecipe` skeleton for DeepSeek Flash WeightPack mixed policy; source-preserving FP4/FP8 decoders exist for correctness | real conversion kernels, calibration/eval, and GGUF export/import validators |
| Artifacts | source checkpoint, Ferrule execution package, compatibility package | `ConversionPlan`: HF safetensors -> WeightPack primary, GGUF optional | streaming converter, manifest checksums, per-role conversion reports |
| Auxiliary tensors | indexer, compressor, HC/output-HC blocks, attention sinks | metadata classification + named missing-policy errors; HC reference math and real HC source binding pass; compressor/indexer source slices are preserved as attention auxiliaries | wire HC into executable layer state and implement compressor/indexer semantics from reference code |
| Scale | 166GB official HF source; 80–150+ GiB quantized/community artifacts; million-token context pressure | single-process local runner; source-preserving expert streaming reader for local HF shards; bounded source readers validate real attention/HC/router/expert payloads without full-model load | residency manager wired into decode, WeightPack chunks, optional remote source policy, async/prioritized I/O later |
| DSpark | draft model, target model, acceptance/rejection/rollback, speculation metrics | source metadata marks DSpark attachment; MTP projection/Markov/confidence tensors map to speculative semantic roles and set `SpeculationPolicy::MultiTokenPrediction` | verifier/rollback loop, acceptance policy, scheduler integration |
| Correctness | compare against official inference and/or reference engine outputs | OLMoE CPU reference fixture + compare-logits; DSV4 local metadata/source-binding smokes for expert, router, shared FFN, attention, HC, tokenizer | reference DeepSeek engine harness and first-token/golden-token tests |

Near-term rule: DeepSeek V4 execution must enter through generic Transformer
executor policies; DSpark must enter through a generic speculative decoding
policy around target/draft models. Do not copy OLMoE forward and rename fields,
and do not add generic fields that only make sense for one DeepSeek artifact.

### 1.9 Competitive PK targets

Ferrule should compare itself against engines by scenario, not marketing claims.
The first PK matrix should use one fixed machine and publish exact commands.

| Scenario | Reference engines | Ferrule target metric |
|---|---|---|
| Local GGUF single-user | llama.cpp | load time, TTFT, decode tok/s, memory, correctness smoke |
| High-throughput serving | vLLM, SGLang | requests/s, p50/p90 latency, continuous batching behavior |
| CUDA optimized MoE | TensorRT-LLM, LMDeploy/TurboMind where applicable | kernel count, decode tok/s, VRAM, expert batching |
| Edge/out-of-core | llama.cpp offload, FlexGen-like baselines | runnable context under memory budget, tokens/J or tok/s/W if available |
| DeepSeek-family visibility | llama.cpp DeepSeek/DSpark branch or known working reference | first-token match/golden text, then speed/memory |

The first publishable PK should be honest:

1. `info` metadata parity for DeepSeek V4 GGUF,
2. first executable DeepSeek V4 smoke on a tiny/sliced fixture,
3. full antirez DeepSeek V4 GGUF load/streaming feasibility report,
4. decode benchmark against llama.cpp on the same quant and prompt,
5. server benchmark against vLLM/SGLang only after batching is actually wired.

### 1.10 Generic model bring-up contract

Every new model family should satisfy the same contract. DeepSeek V4 is the
first hard implementation target, but the API should be equally usable for
Llama/Qwen/Mixtral-like dense or MoE models.

| Contract piece | A model family provides | The engine provides |
|---|---|---|
| `ModelDescriptor` | architecture, dimensions, tokenizer/encoding metadata, tensor inventory | format readers, metadata validation, unsupported-policy errors |
| `ModelLayout` | layer graph, required/optional tensor classes, residual/norm ordering | layer iteration, state ownership, execution orchestration |
| `TensorBinding` | mapping from source tensor names to semantic tensor roles | loading, mmap/streaming, placement, dtype/quant dispatch |
| `AttentionPolicy` | attention kind, projection roles, position encoding, KV shape | attention step scheduling, KV ownership, backend kernel dispatch |
| `MlpPolicy` / `ExpertPolicy` | dense MLP or routed/shared experts, activation and routing semantics | expert batching, residency, dequant/matvec execution |
| `QuantPolicy` | supported quant classes per tensor role and fallback rules | validators, conversion or native kernels, benchmark counters |
| `TokenizerPolicy` | tokenizer files, chat/reasoning templates, special tokens | prompt/session lifecycle, stop handling, sampler integration |
| `ValidationPolicy` | reference commands, golden tokens/logits, tolerances | compare tools, reproducible reports, regression storage |
| `SpeculationPolicy` | optional draft/MTP attachment metadata and acceptance rules | proposal/verify loop, rollback, metrics, scheduler integration |

Design implication: generic runtime structs should not grow fields named after a
specific model's tensors. Model-specific names belong in per-family adapter
modules; the executor sees semantic roles and policy objects.

Current source layout rule:

```text
crates/ferrule-model/src/families/
  common.rs       # conservative dense Transformer name policy
  deepseek_v4.rs  # DeepSeek V4 / Flash / DSpark HF+GGUF names and notes
```

When adding a family, create a new `families/<family>.rs` file that maps source
artifact names into generic `TensorClass` / `TensorRole` values and adds only
family-local metadata refinement. Do not add source tensor names to generic
`tensor_policy`, runtime runners, KV state, or CUDA kernels.

### 1.11 Artifact and conversion strategy

Ferrule should distinguish source artifacts, execution artifacts, and
compatibility artifacts.

```text
Official HF safetensors / tokenizer / encoding / inference scripts
  -> SourceArtifact
  -> ModelDescriptor + ModelSupportContract
  -> ConversionPlan + QuantizationRecipe
  -> WeightPack  primary Ferrule execution artifact
  -> GGUF        optional compatibility / PK artifact
```

Rules:

- Official `deepseek-ai/DeepSeek-V4-Flash-DSpark` is the source-of-truth for
  DeepSeek Flash + DSpark metadata, encoding behavior, and reference inference.
- WeightPack is Ferrule's native execution package because it can carry semantic
  roles, quant recipes, residency hints, streaming chunks, checksums, and future
  DP/TP/EP/SP/CP/PP placement metadata.
- GGUF remains important for llama.cpp compatibility, distribution, and PK, but
  Ferrule should not make GGUF the only internal execution contract.
- Conversion should be reproducible: source repo/revision, recipe name,
  per-role dtype policy, calibration set, checksums, and reference validation
  must be recorded.
- Quantized artifacts are not correctness references. Official source inference
  and tiny high-precision fixtures remain the correctness anchors.

Current DeepSeek V4 Flash DSpark source byte facts from local shard headers:

| Group | Bytes | GiB | Implication |
|---|---:|---:|---|
| Routed expert gate | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert up | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert down | 52,479,131,648 | 48.875 | dominates memory |
| Attention tensors | 5,721,447,936 | 5.329 | small enough to preserve source initially |
| Shared experts | 1,157,698,560 | 1.078 | keep conservative/Q8 initially |
| Embedding | 1,059,061,760 | 0.986 | Q8 conservative |
| Output head | 1,059,061,760 | 0.986 | Q8 conservative |
| DSpark/MTP | 198,786,204 | 0.185 | keep source until speculation verifier exists |

By dtype:

| Dtype | Tensors | Bytes | GiB |
|---|---:|---:|---:|
| `I8` | 35,328 | 148,176,371,712 | 138.000 |
| `F8_E8M0` | 35,718 | 9,261,408,000 | 8.625 |
| `F8_E4M3` | 390 | 6,304,038,912 | 5.871 |
| `BF16` | 445 | 2,967,134,976 | 2.763 |
| `F32` | 433 | 150,966,520 | 0.141 |
| `I64` | 3 | 18,616,320 | 0.017 |

Quantization decision for one DGX Spark:

1. **Quality-first default:** preserve official source quantization for routed
   experts (`I8` containers for official FP4 payloads) and implement expert
   streaming/residency. This avoids extra loss and matches Ferrule's original
   goal: run even when VRAM, host RAM, or local storage cannot hold the full
   model resident at once.
2. **First all-resident smoke:** keep the explicit
   `deepseek-v4-flash-dgxspark-resident-iq2-v1` recipe as a later optional layer.
   It pushes routed experts to an IQ2/Q2-class format, keeping attention/source-
   sensitive tensors conservative. This is for “runs end-to-end on one DGX
   Spark”, not a final quality profile.
3. **Do not use 4-bit re-quant as the single-node fit strategy.** The official
   routed experts are already effectively FP4-sized; another 4-bit format does
   not solve the memory problem and adds conversion error.
4. **Design streaming before extra quantization.** Quantization and streaming can
   be stacked later, but the first executable path should prove source-preserving
   expert streaming from local shards / WeightPack chunks / optional LAN remote
   chunks.

### 1.12 DeepSeek V4 / DSpark bring-up test ladder

The current test hardware is one DGX Spark. A second same-LAN machine may become
available later, but the bring-up plan must be single-node complete first. Every
step below needs an explicit test or smoke command before the next execution
layer is marked supported.

| Stage | Scope | Must pass on normal dev/CI | Must pass on DGX Spark local model | Why it matters |
|---|---|---|---|---|
| T0 metadata fixture | source artifact, family detection, synthetic tensor names | unit tests for `SourceArtifact`, `ModelFamily`, per-family classifiers | `cargo test -p ferrule-model` | prevents generic policy pollution and source metadata drift |
| T1 local HF index | official downloaded HF directory, index count, shard presence | skips if model absent | `cargo test -p ferrule-model local_deepseek_v4_flash_dspark_descriptor_smoke_if_present` | confirms the 166GB checkpoint is complete without loading payloads |
| T2 shard header inventory | dtype/shape/byte-size from safetensors headers only | synthetic header fixtures | full 48-shard header scan with dtype/class/role counts | required before quantization or streaming conversion |
| T3 semantic layer binding | per-layer required/optional roles, MTP attachment, attention sinks | tiny synthetic DeepSeek V4 layer fixture | validate all local tensors bind to a known role or explicit unsupported auxiliary role | prevents conversion plans from silently dropping tensors |
| T4 conversion planning | HF source -> WeightPack recipe; optional GGUF compatibility plan | deterministic recipe/manifest unit tests | local conversion dry-run report: roles, bytes, target chunks, skipped/unsupported=0 unless named | makes quantization reproducible before writing huge artifacts |
| T5 streaming converter | bounded-memory HF -> WeightPack writer | small safetensors fixture conversion | one-shard then full-model conversion with peak RSS/throughput report | DGX Spark memory pressure requires streaming, not all-in-RAM load |
| T6 source-format kernels | per-format CPU decoder + CUDA dequant/matvec | FP4 E2M1 + E8M0 source-format tests and tiny CPU expert executor tests | FP8/FP4 CUDA kernels compared against CPU decoder and official reference snippets | source-preserving kernels precede speed; avoid secondary quantization as the first fit strategy |
| T7 tiny executable fixture | one or few DeepSeek-shaped layers | synthetic logits/golden-token tests through generic Transformer policies | CUDA one-token decode on tiny/sliced fixture | verifies attention/router/HC/MoE policies without needing the 166GB model resident |
| T8 full single-node smoke | quantized WeightPack load + prompt/decode | not required | fixed prompt first-token/golden text; memory, TTFT, tok/s captured | first honest “runs on one DGX Spark” milestone |
| T9 DSpark speculation | target/draft proposal + verifier + rollback | synthetic accept/reject tests | short prompt with speculation metrics: proposed, accepted, rejected, rollback count | DSpark must be validated as generic speculation, not a DeepSeek-only shortcut |
| T10 PK harness | comparable reference engine runs | command serialization/unit tests | llama.cpp/reference command + Ferrule command on same prompt/quant | publishable claims need reproducible apples-to-apples numbers |
| T11 two-node LAN | optional distributed/offload prototype | config validation only | explicit network config, bandwidth/latency probe, failure/retry smoke | only after single-node correctness; do not block T0-T10 |

Testing rules for this track:

- Default `cargo test` must stay useful without the 166GB model; local-model tests
  must skip when the model directory is absent unless an environment variable asks
  for a hard check.
- DGX Spark local tests should default to `models/DeepSeek-V4-Flash-DSpark` and
  also accept `FERRULE_DEEPSEEK_V4_DIR` for alternate mounts.
- Every unsupported tensor class, dtype, kernel, or policy must fail with a named
  missing-policy/conversion error; never silently drop tensors to make a test pass.
- Quantized WeightPack is an execution artifact, not the source of correctness.
  Compare conversion and inference against official HF/reference outputs or tiny
  high-precision fixtures.
- Record memory, peak resident bytes, host↔device transfer bytes, and elapsed time
  for every full-model conversion/load/decode smoke on DGX Spark.
- When a second LAN machine is available, add it behind an explicit placement /
  transport policy test. Do not let two-node assumptions leak into the single-node
  execution path.

Current concrete commands:

```bash
# Always safe; on this DGX Spark it also validates the local DeepSeek directory.
cargo test -p ferrule-model

# Hard-check a non-default mount.
FERRULE_DEEPSEEK_V4_DIR=/path/to/DeepSeek-V4-Flash-DSpark cargo test -p ferrule-model local_deepseek_v4_flash_dspark_descriptor_smoke_if_present

# Human-readable local metadata inspection.
cargo run -p ferrule-cli -- info models/DeepSeek-V4-Flash-DSpark

# Read one selected expert's six source slices from real local safetensors shards.
cargo run -p ferrule-cli -- expert-stream-smoke models/DeepSeek-V4-Flash-DSpark --layer 0 --expert 0 --max-slice-mb 64

# Runtime unit/integration coverage for expert streaming + source-format reference math.
cargo test -p ferrule-runtime

# Focused local DSV4 source-binding smoke: real attention + HC payloads from HF shards.
cargo test -p ferrule-runtime local_deepseek_v4_attention_and_hc_bind_real_sources_if_present
```

### 1.13 Current distance to runnable DeepSeek V4

Short version: Ferrule is past metadata discovery and has started the memory
pressure path, but DeepSeek V4 is **not close to “just wire weights into the old
OLMoE runner.”** The remaining work is a real DeepSeek-shaped Transformer bring-
up through generic policies. The shortest honest path is base-model one-token
decode first, then full prompt/decode, then DSpark speculation.

Already in place:

- complete local HF inventory: 72,317 tensors across 48 shards, 166.9GB source,
  zero unknown tensor classes;
- semantic family adapter boundary under `ferrule-model/src/families/`;
- source byte offsets for bounded safetensors streaming;
- typed HF semantic descriptors for DSV4 attention, HC, router, shared experts,
  and routed experts; concrete source names stay inside the family parser;
- routed expert tensor-set grouping for 43 layers × 256 experts;
- local expert-streaming smoke that reads one expert's six source tensors;
- `ExpertComputeBundle` format inference for official FP4 experts;
- resident expert handle abstraction/store for CPU/reference MoE and future GPU
  handles;
- `source_format` CPU helpers for FP4 E2M1 + E8M0 scales and FP8 E4M3FN + E8M0 block scales;
- tiny `CpuReferenceExpertExecutor` for source-preserving SwiGLU expert math;
- correctness-first CUDA packed-FP4 expert executor path for official source FP4
  experts;
- `ExpertRouterPolicy` CPU reference semantics for score-top-k and hash routing,
  including `sqrtsoftplus`, bias-for-selection, selected-weight normalization,
  and route scaling;
- generic `SourceTensorReader` / `SourceLinearPayload` handles for bounded HF
  safetensors byte ranges and F32/BF16/FP8/FP4 source linear formats;
- `AttentionSourcePayload` binds real local DSV4 layer-0 core attention tensors,
  validates FP8 block-scale shapes, decodes BF16 norms/F32 sinks, and preserves
  layer-2 compressor/indexer slices as auxiliary semantic source tensors;
- `HyperConnectionWeights` / `HyperConnectionHeadWeights` bind real layer HC and
  global HC head tensors from local HF shards and validate official DSV4 shapes;
- HC reference math exists for `hc_pre`, `hc_post`, `hc_head`, and Sinkhorn split;
- sparse attention reference + correctness-first CUDA sparse attention ABI exists
  for top-k + attention-sink semantics;
- `SwiGluFfnPayload` and `execute_routed_moe_reference` tiny reference path that
  wires router → expert streaming → packed-FP4 routed experts → optional shared
  FFN aggregation;
- generic `TokenizerHandle::load` now loads `tokenizer.json` directly for
  non-OLMoE model families, and the local DSV4 tokenizer smoke passes;
- DeepSeek HF family parser now exposes semantic shared-expert and router tensor
  refs, including correct `gate.bias` and `gate.tid2eid` classification;
- real local DSV4 layer-0 shared expert tensors bind into `SwiGluFfnPayload`, and
  real layer-0/layer-3 routers bind into `RouterSourcePayload` with hash-table /
  bias distinction;
- tiny DSV4 MoE layer fixture now runs source router → hash routing → expert
  streaming → packed-FP4 routed experts → shared FFN aggregation.

Not yet in place for first real DSV4 token:

- executable kernels behind `TransformerRuntimePlan`: the generic layer graph now exists and plans DSV4 as MLA + streamed routed/shared experts + MTP attachment, but it is not yet connected to full weight handles or CUDA execution;
- tokenizer/encoding reference harness against `encoding_dsv4.py` / official generation semantics;
- CUDA linear handles for FP8 E4M3 + E8M0 attention/shared linears; CPU reference decode exists and FP4 expert CUDA correctness path exists, but FP8 production execution is still missing;
- full `LayerSourceBinding` / executable layer assembly: attention, HC, router,
  shared FFN, routed experts, embeddings, output norm, and logits need one typed
  per-layer binding consumed by the generic executor;
- DeepSeek router/MoE connection to model-layer execution: hash `tid2eid`,
  score/bias tensors, selected expert ids, route weights, shared FFN, resident
  handles, and CUDA FP4 expert executor exist in pieces but are not yet one full
  DSV4 layer step;
- HC integration into executable layer state and CUDA kernels; reference math and
  real source binding exist, but `hc_pre/post/head` are not yet in the forward path;
- DeepSeek attention execution: q low-rank path, rotary/inverse rotary, sliding
  window KV, compression/indexer logic, sparse gather, attention sinks, and
  compressed KV append/view semantics;
- logits/head path and first-token/golden-token reference;
- DSpark/MTP proposal, verifier, rollback, confidence policy, and metrics.

Run-critical priority queue:

| Priority | Milestone | Why first | Main implementation target | Required test/smoke |
|---|---|---|---|---|
| P0 | Generic execution skeleton | Avoid another model-specific runner | `TransformerRuntimePlan` now lowers semantic contracts into embed/layer-attn/layer-ffn/logits plus MTP attachment steps; next add executable trait dispatch | DSV4 plan fixture passes: MLA + hash routing + streamed routed/shared experts |
| P1 | DSV4 tokenizer/reference harness | Without identical token ids, logits comparisons are noisy | generic `TokenizerHandle` loads official `tokenizer.json`; `encoding_dsv4.py` chat wrapper is absent locally/needs reference parity when available | local tokenizer smoke passes; next fixed chat-format golden ids |
| P2 | Source linear formats | Every DSV4 block depends on BF16/FP8/FP4 linears | generic source tensor + linear handles exist for F32/BF16/FP8/FP4; real local attention FP8 pairs bind for full core layer payloads; FP4 expert CUDA correctness path exists | CPU golden blocks and local FP8 source-linear/source-binding smokes pass; next FP8 CUDA tiny matvec/GEMM |
| P3 | DSV4 MoE step | Experts are the memory bottleneck and current differentiator | CPU reference MoE step wires source router + source-streamed routed experts + optional shared FFN; real DSV4 shared/router bindings pass; resident handles and correctness-first CUDA FP4 expert executor exist | synthetic hash/non-hash router, tiny DSV4 MoE layer fixture, real shared expert binding, and real hash/score router binding pass; next full layer integration |
| P4 | Hyper-Connection block state | DSV4 residual path is not vanilla Transformer | `hc_pre`, `hc_post`, `hc_head`, Sinkhorn reference exist; real layer/global HC source binding validates local DSV4 shapes | CPU tiny HC fixture and local HC binding smoke pass; next wire HC into executable layer state |
| P5 | DSV4 attention/KV | This is the largest correctness blocker | attention source binding, sparse attention reference, and correctness-first CUDA sparse ABI exist | next q LoRA + rotary + sliding/compressed KV + compressor/indexer execution; then tiny prefill/decode attention parity |
| P6 | Tiny DSV4 executable fixture | Proves all policies compose before huge model | 1–2 layers, tiny dims, source-format weights, generic executor | deterministic logits/golden token on CPU and CUDA where available |
| P7 | Full single-node base-model smoke | First “runs on DGX Spark” milestone | stream official HF/WeightPack tensors, source-preserving expert residency, prompt/decode loop | fixed prompt first token/golden text, peak memory, TTFT, tok/s |
| P8 | DSpark as speculation layer | DSpark is optional acceleration, not base-model correctness | generic draft/propose/verify/rollback loop using MTP tensors | synthetic accept/reject tests, then short DSV4 speculative smoke |
| P9 | PK harness | Claims need comparable commands | llama.cpp/reference/Ferrule command capture, metric schema | same prompt/quant artifact, reproducible report |

Execution rule for the next patches:

1. Do **not** wait for GGUF conversion to start execution. The official HF source
   already has enough metadata and byte offsets for source-preserving streaming.
2. Do **not** make DSV4 a special runner. Put DSV4 names in `ferrule-model`
   family adapters; runtime takes semantic policies and tensor handles.
3. Do **not** block on async I/O. Keep the synchronous bounded reader as the
   correctness path; later replace the backend with priority queued async /
   `io_uring` without changing expert planning semantics.
4. Do **not** enable DSpark before base-model decode is correct. DSpark should
   wrap a working target-model path.

---

## 2. Implementation Principles

Use these rules when choosing the next patch:

1. **Keep CPU FP32 reference alive.** It is the correctness anchor.
2. **Do not hide state in the CLI.** Session, KV, sampler, and future scheduler
   state belong in `ferrule-runtime` or backend crates.
3. **Make every feature reviewable.** Prefer small PRs/tickets with a command
   that proves the behavior.
4. **Measure before optimizing.** Add benchmarks and counters before large
   kernel rewrites.
5. **Avoid premature vLLM complexity.** Paged KV and continuous batching require
   request/session abstractions first.
6. **Treat WeightPack as a runtime artifact.** It should become Ferrule's local and
   edge deployment unit.
7. **Preserve MoE semantics.** Router logits, top-k selection, expert weights,
   and normalization policy are correctness-critical.
8. **Treat model families as policies, not runners.** DeepSeek V4, OLMoE,
   Qwen-MoE, Mixtral, and dense Llama-like models should plug into Transformer
   layout, attention, router, quantization, and residency policies. Do not fork a
   full inference loop per family.
9. **No privileged target-model path.** DeepSeek V4 can drive requirements, but
   generic crates should expose semantic roles and policy traits, not
   DeepSeek-named fields or special branches.
10. **Treat DSpark as speculation, not a base model family.** DSpark/DFlash/Eagle
    style features should plug into target/draft model orchestration with explicit
    acceptance, rejection, rollback, and metrics.
11. **Compose behavior through typed policies.** Engine behavior should be
    selected by an `EnginePlan`/config surface, not by ad-hoc `if model_name`
    branches.
12. **Make unsupported execution explicit.** `info` may inspect metadata for a new
    family, but `run/chat/server` must fail clearly until kernels and layout
    semantics are implemented.
13. **PK against real engines with reproducible commands.** Every performance
    claim needs a comparable prompt, context length, quantization, batch size,
    hardware description, and quality/correctness note.

### 2.1 Composable engine architecture target

Ferrule should look more like a typed Rust execution graph than one runner per
model family. A new model bring-up should provide descriptor parsing, tensor
layout, and policy implementations; the runtime should compose them into an
`EnginePlan`.

```text
ModelDescriptor
  -> ModelLayout
  -> ModelSupportContract
  -> EnginePlan {
       ModelFamilyPolicy,
       AttentionPolicy,
       RouterPolicy,
       ExpertPolicy,
       QuantPolicy,
       KvPolicy,
       SchedulerPolicy,
       ParallelismPlan,
       ResidencyPolicy,
       SpeculationPolicy,
     }
```

| Policy | Responsibility | Near-term default | Future switches |
|---|---|---|---|
| `ModelFamilyPolicy` | map metadata/tensor names into semantic layer components | OLMoE fixture, DeepSeek V4 descriptor | model-family plugins without runner forks |
| `AttentionPolicy` | MHA/GQA/DeepSeek-family attention semantics, RoPE/position encoding, cache layout | GQA RoPE | DeepSeek CSA/HCA/MLA-compatible path, Flash-style prefill, future dense variants |
| `RouterPolicy` | logits, bias, hash routing, top-k normalization | dense top-k fixture | DeepSeek bias/hash routing, expert-parallel routing, other MoE routers |
| `ExpertPolicy` | dense MLP, routed experts, shared experts, activation semantics | routed experts | shared experts, expert batching, EP |
| `QuantPolicy` | tensor quant class, dequant/matvec path, conversion policy | Q4_0/Q8_0 WeightPack | GGUF K/IQ, FP4/FP8, mixed policy |
| `KvPolicy` | contiguous/paged/quantized KV ownership and views | contiguous per-session KV | paged KV, prefix/radix reuse, context parallel KV, model-specific KV shapes |
| `SchedulerPolicy` | prefill/decode admission, chunking, batching | single request / prototype scheduler | continuous batching, preemption, priority |
| `ParallelismPlan` | DP/TP/EP/SP/CP/PP placement and collective boundaries | single process / single GPU | DP, TP, EP, SP, CP, PP combinations |
| `ResidencyPolicy` | all-GPU vs CPU/GPU/NVMe streaming and prefetch | all resident if it fits | expert/layer streaming, hardware-aware memory budget |
| `SpeculationPolicy` | none/MTP/draft-model proposal and acceptance logic | none | MTP, DSpark, DFlash, Eagle-style draft models |

Configuration should eventually select these policies explicitly, for example:

- `engine`: local, server, benchmark,
- `attention`: auto, GQA, model-family policy,
- `kv`: contiguous, paged, quantized,
- `parallel`: DP/TP/EP/SP/CP/PP plan,
- `residency`: all-GPU, GPU+CPU, streaming,
- `speculation`: none, MTP, draft model.

These switches should map to typed policy objects. They should not leak
model-specific assumptions into the scheduler, sampler, or CLI. If supporting a
new model requires editing the generic scheduler or sampler, the abstraction is
probably wrong.

---

## 3. P0 — Stabilize the Runtime Layer

Goal: make the new `ferrule-runtime` layer the only generation path.

Current state:

- `ferrule-runtime` exists.
- CPU and GPU runner wrappers exist.
- `run`, `gpu-run`, and `chat` share the generation loop.
- basic sampling controls exist.

### P0.1 Review runtime API boundaries

Tasks:

- Ensure `ModelRunner` contains only backend-independent methods:
  - `encode`
  - `decode`
  - `prefill`
  - `decode_token`
  - `reset_session`
  - `eos_token_id`
  - `model_info`
- Keep OLMoE-specific details out of generic generation code.
- Add doc comments explaining that `prefill` advances session state.

Acceptance:

- `cargo check -p ferrule-cli`
- `cargo check -p ferrule-cli --features cuda`
- no CLI-specific generation loop duplicates remain except UI printing.

Review focus:

- no hidden CPU/GPU behavior divergence,
- no accidental session reset between chat turns,
- no extra full model reload inside generation.

### P0.2 Add smoke-test scripts  ✅ DONE or examples

Tasks:

- Add a tiny shell script or documented commands:

```bash
printf 'hi\n/exit\n' | ./target/release/ferrule chat models/OLMoE-Instruct -q cpu -n 32
printf 'hi\n/exit\n' | ./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 32
```

- Add expected output notes, not exact long text.

Acceptance:

- CPU output starts with a normal assistant greeting.
- GPU Q4 output does not degenerate into debug ids or markup loops.

Review focus:

- tests should not require committing model weights,
- commands should be copy-pasteable.

---

## 4. P1 — llama.cpp Local UX Parity

Goal: make Ferrule pleasant and debuggable for one local user.

### P1.1 Generation config loading

Gap:

- llama.cpp and HF workflows respect model-side generation defaults.
- Ferrule currently only uses CLI flags.

Tasks:

- Load `generation_config.json` if present.
- Support at least:
  - `eos_token_id`
  - `pad_token_id`
  - `temperature`
  - `top_k`
  - `top_p`
  - `repetition_penalty`
  - `max_new_tokens` only as default, not override if CLI passes `-n`
- Define clear precedence:
  1. CLI flags,
  2. `generation_config.json`,
  3. Ferrule defaults.

Acceptance:

```bash
./target/release/ferrule chat models/OLMoE-Instruct --help
./target/release/ferrule run models/OLMoE-Instruct -p 'hi' -n 8
```

Review focus:

- no panic on missing fields,
- no hard dependency on HF-specific schema beyond supported fields.

### P1.2 Chat template registry

Gap:

- llama.cpp supports many chat templates.
- Ferrule currently has OLMoE-Instruct shortcut + plain fallback.

Tasks:

- Replace ad-hoc detection with registry:

```rust
enum ChatTemplate {
    Olmoe,
    ChatML,
    Llama3,
    Qwen,
    Plain,
}
```

- Detect from `tokenizer_config.json` using simple pattern matching first.
- Add `--chat-template <name>` override.
- Add `--no-template` or `--template plain` option.

Acceptance:

- OLMoE-Instruct still uses:

```text
<|endoftext|>
<|user|>
...
<|assistant|>
```

- Plain mode still works.

Review focus:

- no full Jinja engine yet unless truly needed,
- template formatting must preserve multi-turn KV state assumptions.

### P1.3 Context and session controls  ✅ DONE

Gap:

- llama.cpp exposes context size, prompt cache, context shifting.
- Ferrule has fixed GPU `max_seq` and no user-facing state controls.

Tasks:

- Add CLI:
  - `--ctx-size <N>`
  - `/reset` already exists or should remain supported
  - `/stats` to print prompt/decode tokens and current session length
- Thread context size into GPU KV allocation.
- Return a clear error if sequence exceeds capacity.

Acceptance:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q q4 --ctx-size 2048
```

Review focus:

- avoid silent memory overwrite,
- context size must affect both CPU and GPU session limits consistently.

### P1.4 Token/logprob debugging

Gap:

- llama.cpp can print tokens, logprobs, probabilities.
- Ferrule only has gated top-k debug ids inside GPU forward.

Tasks:

- Add `--verbose-tokens` for generated token ids and decoded pieces.
- Add optional `--logprobs <K>` for top-K tokens from logits.
- Move top-K display to runtime/CLI, not backend debug env only.

Acceptance:

```bash
./target/release/ferrule run models/OLMoE-Instruct -p 'hi' -n 4 --logprobs 5
```

Review focus:

- do not print debug spam by default,
- token display should work for CPU and GPU.

---

## 5. P2 — Correctness and Regression Tools

Goal: make quality failures reproducible and reviewable.

### P2.1 `compare-logits`

Gap:

- Ferrule needs an automatic CPU-vs-GPU correctness tool.

Command target:

```bash
ferrule compare-logits models/OLMoE-Instruct -q q4 -p 'hi' -n 16
```

Metrics:

- per-step CPU top-1 token,
- per-step GPU top-1 token,
- first divergence step,
- max absolute logit error,
- mean absolute logit error,
- top-5/top-10 overlap,
- decoded CPU/GPU strings.

Implementation tasks:

- Add command in CLI.
- Use two runners:
  - `CpuOlmoeRunner`
  - `GpuOlmoeRunner`
- Feed identical prompt tokens.
- At each decode step:
  - compare logits before sampling,
  - pick token according to a policy.

Two modes:

1. `--teacher cpu`: GPU follows CPU token so per-step error is comparable.
2. `--free-run`: both sample/argmax independently to find divergence.

Acceptance:

- command exits 0 with report,
- no model output needed in tests,
- works for `-q q4`; `-q q8` can be marked experimental.

Review focus:

- teacher-forcing mode is essential,
- avoid using decoded text as correctness metric only.

### P2.2 Golden-token smoke tests  ✅ DONE

Gap:

- Manual chat checks are fragile.

Tasks:

- Add ignored integration tests or scripts that require local model path.
- Store short expected token prefixes for CPU FP32.
- Store relaxed expectations for GPU Q4:
  - first token match,
  - or top-k overlap threshold,
  - or no degeneration markers.

Acceptance:

```bash
FERRULE_MODEL=models/OLMoE-Instruct cargo test -p ferrule-cli --test olmoe_smoke -- --ignored
```

Review focus:

- tests must be opt-in because model files are large,
- do not make Q4 deterministic quality stricter than current quantization allows.

### P2.3 Perplexity command

Gap:

- llama.cpp has perplexity tools.

Command target:

```bash
ferrule perplexity models/OLMoE-Instruct --file data/sample.txt -q cpu
ferrule perplexity models/OLMoE-Instruct --file data/sample.txt -q q4
```

Tasks:

- Tokenize input.
- Teacher-force each token.
- Compute negative log-likelihood from logits.
- Report tokens/sec and perplexity.

Acceptance:

- CPU path works first.
- GPU path optional after logits path is stable.

Review focus:

- numerically stable log-softmax,
- no sampling involved.

---

## 6. P3 — Benchmarking and Profiling

Goal: know where Ferrule is slow before rewriting kernels.

### P3.1 `bench-infer`

Gap:

- Ferrule lacks a `llama-bench` equivalent.

Command target:

```bash
ferrule bench-infer models/OLMoE-Instruct -q q4 -p 'Hello' -n 128
```

Report:

- model path,
- backend,
- quant type,
- prompt tokens,
- generated tokens,
- prompt processing seconds,
- prompt processing tok/s,
- decode seconds,
- decode tok/s,
- total tok/s,
- GPU memory free/total if CUDA.

Tasks:

- Reuse `GenerateStats`.
- Add `--json` output for tracking.
- Add `--warmup <N>`.
- Add `--repeat <N>` with median/p50/p90.

Acceptance:

```bash
ferrule bench-infer models/OLMoE-Instruct -q cpu -p 'hi' -n 8
ferrule bench-infer models/OLMoE-Instruct -q q4 -p 'hi' -n 32 --json
```

Review focus:

- separate prefill and decode,
- do not include model loading unless `--include-load` is passed.

### P3.2 Kernel-level counters

Gap:

- We know throughput but not launch count or per-layer cost.

Tasks:

- Add optional `FERRULE_PROFILE=1` or `--profile`.
- Count kernel launches per token.
- Time major regions:
  - embedding,
  - attention projections,
  - attention score/combine,
  - router,
  - expert loop,
  - lm_head.
- Start with coarse host timers; later CUDA events.

Acceptance:

- profile output only when enabled,
- no measurable overhead when disabled.

Review focus:

- no spam in normal chat,
- use stable labels so results can be compared.

### P3.3 Competitive PK harness

Goal: make Ferrule-vs-engine comparisons reproducible enough to trust.

Tasks:

- Add a benchmark manifest format, e.g. `benchmarks/pk/deepseek-v4.toml`, with:
  - model artifact path / Hugging Face repo,
  - tokenizer/template,
  - quantization and context length,
  - prompt set,
  - max new tokens,
  - batch/concurrency setting,
  - hardware notes,
  - reference engine command.
- Extend `bench-infer --json` with fields needed for comparison:
  - load seconds when requested,
  - TTFT,
  - prompt tok/s,
  - decode tok/s,
  - total tok/s,
  - peak RSS if available,
  - GPU memory before/after,
  - generated token count,
  - stop reason.
- Add script or command wrapper for:
  - Ferrule,
  - llama.cpp,
  - vLLM,
  - SGLang,
  - optional TensorRT-LLM/LMDeploy when install is available.
- Store raw JSON output; generate markdown summary table separately.

Acceptance:

- One command can run Ferrule local PK on OLMoE.
- DeepSeek V4 PK manifest exists even before Ferrule execution; it records the
  reference command and expected blockers.
- Results distinguish load time, prefill, decode, and serving throughput.

Review focus:

- do not compare different quantizations as if they are equivalent,
- include prompt and context length in every result,
- never hide unsupported Ferrule paths as zeros or failures in benchmark charts.

---

## 7. P4 — WeightPack-Only Startup

Goal: avoid full FP32 model load on WeightPack hit.

This is the biggest local usability and memory milestone.

Current gap:

- GPU WeightPack hit still loads the full FP32 model before using cache.
- Q8_0 is blocked by RAM because loading FP32 + quantized weights exceeds
  available memory.

### P4.1 WeightPack manifest

Tasks:

- Add WeightPack header/manifest with:
  - magic bytes,
  - format version,
  - model architecture,
  - model config hash,
  - tokenizer hash,
  - quant type,
  - quant layout version,
  - tensor names,
  - tensor shapes,
  - dtype policy,
  - chunk offsets,
  - checksums.

Acceptance:

- old cache mismatch fails with a clear error,
- `ferrule info` or new `inspect-weightpack` can show manifest.

Review focus:

- never silently load incompatible WeightPack,
- distinguish Q4_0 old layout from `q4_0_llama` layout.

### P4.2 lightweight model load path

Tasks:

- Use `OlmoeModel::load_lightweight` or refine it.
- Load only:
  - config,
  - tokenizer,
  - embeddings,
  - lm_head,
  - final norm,
  - layer norms,
  - router weights,
  - q/k head norms.
- Load quantized attention/expert weights directly from WeightPack.

Acceptance:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 16
```

On WeightPack hit:

- should not print full shard tensor loading count,
- startup should be close to lightweight load + GPU upload time,
- peak RAM should be much lower than FP32 full load.

Review focus:

- verify all tensors required by GPU path are present,
- no fallback to empty weights unless explicitly validated.

### P4.3 streaming WeightPack writer  ✅ DONE

Tasks:

- Quantize one layer at a time.
- Write WeightPack chunks incrementally.
- Drop FP32 layer tensors after quantization.

Acceptance:

- first WeightPack creation stays under a defined RAM budget,
- Q8_0 cache creation becomes possible on 32 GB RAM.

Review focus:

- file must be atomic: write temp file, then rename,
- partial cache should not be considered valid.

---

## 8. P5 — Minimal Local Server

Goal: create the server surface before implementing vLLM-like scheduling.

### P5.1 Single-request OpenAI-compatible server

Command target:

```bash
ferrule server models/OLMoE-Instruct -q q4 --host 127.0.0.1 --port 8080
```

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Initial behavior:

- one request at a time,
- no batching,
- stream optional but nice to have,
- model loaded once at startup.

Tasks:

- Add HTTP dependency only if acceptable.
- Prefer small dependency footprint.
- Convert messages to Ferrule chat template.
- Use `InferenceEngine` internally.

Acceptance:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"olmoe","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

Review focus:

- no reload per request,
- clear error responses,
- generation cancellation can be deferred.

### P5.2 Streaming responses  ✅ DONE

Tasks:

- Support `stream: true` with SSE chunks.
- Emit token text as generated.
- Emit final usage stats.

Acceptance:

- compatible enough for common OpenAI clients.

Review focus:

- flush after chunks,
- handle client disconnect gracefully if possible.

---

## 9. P6 — Runtime Request/Session Model

Goal: prepare for vLLM/SGLang features without implementing them all at once.

### P6.1 Add explicit request/session types

Suggested types:

```rust
struct RequestId(u64);
struct SessionId(u64);

struct GenerateRequest {
    id: RequestId,
    session_id: Option<SessionId>,
    prompt_tokens: Vec<u32>,
    sampling: SamplingConfig,
    max_new_tokens: usize,
    stop: Vec<String>,
}

struct SequenceState {
    session_id: SessionId,
    tokens: Vec<u32>,
    generated: usize,
    status: SequenceStatus,
}
```

Tasks:

- Keep single-request execution initially.
- Make chat/server use `SessionId` instead of implicit local variables.
- Track token history in `SequenceState`.

Acceptance:

- CLI behavior unchanged,
- server can maintain multiple named sessions later.

Review focus:

- do not overbuild scheduler yet,
- types should make KV ownership explicit.

### P6.2 Split tokenizer/template from model runner  ✅ DONE

Gap:

- Tokenization currently lives behind `ModelRunner`.
- Serving wants tokenizer/template access without necessarily advancing model.

Tasks:

- Consider `TokenizerRuntime` or `ModelAssets` abstraction.
- Keep backend runner focused on numerical forward/session state.

Acceptance:

- prompt formatting can be unit-tested without GPU.

Review focus:

- avoid circular ownership between runner, tokenizer, and session.

---

## 10. P7 — KV Cache Road to vLLM

Goal: move from one contiguous KV cache to schedulable KV memory.

### P7.1 Extract KV cache interface  ✅ DONE

Current state / remaining gap:

- GPU hot path owns contiguous KV through `CudaContiguousKvCache`, which groups:
  - `k_cache`
  - `v_cache`
  - `scores_buf`
  - `cur_seq`
  - fixed `max_seq`
- CPU runner owns explicit contiguous KV state through `CpuContiguousKvState`.
- Ownership is now explicit, but the hot path is still contiguous and effectively
  single-session.

Tasks:

- Define a backend-facing KV abstraction:

```rust
trait KvCache {
    type Handle;
    fn append(&mut self, layer: usize, pos: usize, k: ..., v: ...) -> Result<()>;
    fn view(&self, layer: usize, sequence: Self::Handle) -> ...;
    fn reset(&mut self, handle: Self::Handle) -> Result<()>;
}
```

- First implementation wraps the existing contiguous cache.

Acceptance:

- no behavior change,
- code makes KV ownership and sequence length explicit.

Review focus:

- do not force paged layout into first refactor,
- kernel call sites should remain understandable.

### P7.2 Multi-session contiguous KV

Tasks:

- Allocate KV as `[session][layer][seq][kv_dim]` or equivalent.
- Add session handles.
- Allow server to keep multiple sessions but decode one at a time.

Acceptance:

- two chat sessions can alternate without losing context.

Review focus:

- memory accounting,
- max sessions and max sequence must be explicit.

### P7.3 Paged KV allocator

Tasks:

- Define page/block size.
- Add free list.
- Map logical token positions to KV pages.
- Update attention kernels to read block table.

Acceptance:

- single-session output matches contiguous KV,
- multi-session memory can be allocated/freed without copying full KV.

Review focus:

- first correctness, then speed,
- block-table indexing must be heavily tested.

### P7.4 Prefix cache

Tasks:

- Hash token prefixes.
- Store prefix → KV page references.
- Reuse pages when a request shares a prefix.
- Add refcounts.

Acceptance:

- repeated prompt has lower prefill work,
- cache can evict safely.

Review focus:

- no mutation of shared prefix pages,
- tokenizer/template must be deterministic.

### P7.5 Radix cache  ✅ DONE

Tasks:

- Replace simple prefix hash with radix tree.
- Support longest-prefix match.
- Split nodes when partial prefix overlaps occur.

Acceptance:

- SGLang-style prefix reuse works for overlapping prompts.

Review focus:

- keep it behind an interface,
- add unit tests for prefix insertion/split/eviction.

---

## 11. P8 — Scheduler and Continuous Batching

Goal: align with vLLM serving throughput after KV pages exist.

### P8.1 Basic scheduler loop

Tasks:

- Add queues:
  - waiting prefill,
  - running decode,
  - finished,
  - cancelled.
- Define scheduling budget:
  - max batch tokens,
  - max active sequences,
  - max KV pages.
- Initially execute serially but through scheduler APIs.

Acceptance:

- server behavior unchanged,
- internal state visible through metrics.

Review focus:

- no premature async complexity,
- request cancellation state should exist even if basic.

### P8.2 Chunked prefill

Tasks:

- Split long prompts into chunks.
- Interleave decode requests with prefill chunks.
- Track partial prefill state.

Acceptance:

- long prompt does not fully block short decode requests.

Review focus:

- logits only valid after final prompt token,
- KV positions must remain correct.

### P8.3 Continuous batching  ✅ DONE

Tasks:

- Batch decode tokens across active sequences.
- Add batchable kernels or loop over sequences first.
- Use scheduler to admit new requests while others decode.

Acceptance:

- multi-request throughput improves over serial baseline.

Review focus:

- do not sacrifice single-user correctness,
- report latency and throughput separately.

### P8.4 Preemption  ✅ DONE

Tasks:

- If KV pages exhausted, preempt low-priority sequence.
- Free or swap KV pages.
- Resume later.

Acceptance:

- no crash under memory pressure,
- clear metrics for preemption count.

Review focus:

- correctness first; swapping can be deferred.

---

## 12. P9 — Structured Decoding and SGLang-like Control

Goal: support reliable JSON/tool output and generation programs.

### P9.1 Token mask API

Tasks:

- Add optional sampler mask:

```rust
trait TokenConstraint {
    fn allow(&mut self, token: u32, piece: &str) -> bool;
}
```

- Apply mask before top-k/top-p.

Acceptance:

- unit test with tiny fake vocab.

Review focus:

- masking must happen before sampling,
- avoid decoding every vocab token each step unless needed.

### P9.2 JSON / regex constraints

Tasks:

- Add simple JSON object grammar first.
- Then regex/FSM constraints.

Acceptance:

- `--json` mode produces syntactically valid JSON for easy prompts.

Review focus:

- tokenizer boundary handling,
- invalid-state fallback behavior.

### P9.3 Program-like generation API  ✅ DONE

Tasks:

- Provide Rust API for:
  - append text,
  - generate field,
  - constrain field,
  - branch or call tool.

Acceptance:

- no CLI requirement first; library example is enough.

Review focus:

- keep this separate from low-level scheduler.

---

## 13. P10 — Kernel and Performance Track

Goal: close the speed gap after measurements exist.

Likely current bottlenecks:

- many small kernel launches per token,
- per-expert loop launches,
- host/device sync for expert top-k indices/weights,
- FP32 `lm_head` GEMV over full vocab,
- naive attention kernels for longer contexts,
- no CUDA graph replay,
- no batched decode kernels.

### P10.1 Remove host sync from expert loop  ✅ DONE

Current gap:

- GPU router top-k runs on GPU, but selected indices/weights are downloaded to
  host before expert loop.

Tasks:

- Keep top-k expert ids/weights on GPU.
- Launch expert kernels using device-side offsets if possible.
- Or batch selected experts into one kernel.

Acceptance:

- same tokens as old path for fixed prompt,
- reduced per-token latency in profile.

Review focus:

- avoid complicated dynamic parallelism unless necessary,
- correctness of expert offsets.

### P10.2 Fuse expert gate/up/down where possible  ✅ DONE

Tasks:

- Current Q4 gate/up fused path exists.
- Explore fusing:
  - gate/up GEMV,
  - SiLU/mul,
  - down projection,
  - weighted accumulation.

Acceptance:

- profile shows fewer launches per selected expert.

Review focus:

- memory bandwidth vs occupancy tradeoff,
- keep Q8 path correct too.

### P10.3 CUDA graph replay for decode  ✅ DONE

Tasks:

- Identify stable-shape decode region.
- Capture CUDA graph for one-token decode when sequence length changes are
  manageable.
- Or capture per sequence-length bucket.

Acceptance:

- measurable launch overhead reduction.

Review focus:

- graph capture should not make debugging impossible,
- fallback path must remain.

### P10.4 Attention improvements  ✅ DONE

Tasks:

- Implement better prefill attention first if multi-token prefill exists.
- For decode, optimize score/combine path for current seq length.
- Consider FlashAttention-style exact kernels later.

Acceptance:

- correctness vs CPU attention on small prompts.

Review focus:

- long-context numerical stability,
- GQA correctness.

### P10.5 Vocab projection optimization  ✅ DONE

Tasks:

- Investigate quantizing or partitioning `lm_head`.
- Add top-k directly on GPU without copying full logits if sampling allows.
- For logprobs, copy only requested top-K.

Acceptance:

- normal generation does not always require full logits download.

Review focus:

- sampler currently runs on CPU; moving it GPU-side changes architecture.
- keep debug mode that can download full logits.

### P10.6 DeepSeek attention kernels

Goal: add the first mainstream non-fixture attention implementation through the
Transformer executor policy boundary.

Tasks:

- Define `AttentionPolicy`/layout objects for:
  - existing dense MHA/GQA RoPE,
  - DeepSeek-family hybrid attention and MLA-compatible GGUF projection tensors,
  - future dense model variants.
- Map GGUF tensors:
  - `attn_q_a`, `attn_q_b`,
  - `attn_kv`,
  - `attn_output_a`, `attn_output_b`,
  - `attn_compressor_*` if required by the selected DeepSeek V4 artifact.
- Define DeepSeek KV state shape separately from GQA KV state; do not force it
  into `kv_dim = num_kv_heads * head_dim` if that is semantically wrong.
- Confirm CSA/HCA/mHC semantics from paper/reference code before committing the
  final kernel contract.
- Implement a correctness-first decode kernel before optimizing prefill.
- Add tiny fixture tests using synthetic DeepSeek attention tensor shapes before
  full model load.

Acceptance:

- `ferrule info` reports DeepSeek attention layout from GGUF metadata/tensor
  names.
- A tiny DeepSeek-style fixture can run one attention step without OLMoE fields.
- Unsupported auxiliary attention tensors fail with a named missing-policy error,
  not a panic or silent ignore.

Review focus:

- keep the OLMoE fixture GQA path untouched,
- no fake `DeepSeekV4` execution until full layer semantics are wired,
- match reference shapes before chasing speed.

---

## 14. P11 — Quantization Quality Track

Goal: improve quality beyond naive Q4_0 without losing edge usability.

### P11.1 Q8_0 full-path validation  ✅ DONE

Tasks:

- Use WeightPack-only or streaming loading to avoid RAM explosion.
- Run CPU-vs-GPU compare-logits.
- Run short chat smoke.

Acceptance:

- Q8_0 divergence is much lower than Q4_0,
- Q8 path is selectable by CLI.

Review focus:

- Q8 cache file format must be distinct from Q4,
- memory usage must be reported.

### P11.2 Mixed precision policy  ✅ DONE

Tasks:

- Define tensor classes:
  - embeddings,
  - lm_head,
  - norms,
  - router,
  - attention projections,
  - expert gate/up/down,
  - KV cache.
- Allow per-class dtype policy:
  - FP32,
  - F16/BF16 later,
  - Q8_0,
  - Q4_0.

Acceptance:

- manifest records policy,
- CLI can select policy name, e.g. `--quant q4-mixed`.

Review focus:

- quality-sensitive tensors should stay high precision by default.

### P11.3 K-quant / AWQ investigation  ✅ DONE

Tasks:

- Read llama.cpp K-quant layouts.
- Decide whether to implement compatible K-quants or a Ferrule-specific AWQ path.
- Build offline calibration flow if AWQ-like.

Acceptance:

- design doc first,
- one projection type prototype second.

Review focus:

- don't add a format without eval tooling,
- WeightPack manifest must support layout versioning.

### P11.4 KV quantization  ✅ DONE

Tasks:

- Wait until KV abstraction exists.
- Add FP16 first if useful.
- Then KIVI/KVQuant-style low-bit exploration.

Acceptance:

- long-context memory reduction with measured quality impact.

Review focus:

- do not combine with paged KV refactor in same patch.

### P11.5 GGUF K-quant / IQ execution for DeepSeek V4

Goal: execute the quant formats used by DeepSeek V4 GGUF instead of converting
blindly into Ferrule's older Q4_0/Q8_0 assumptions.

Tasks:

- Inventory exact quant types in the target artifacts:
  - `IQ2_XXS`,
  - `Q2_K`,
  - `Q4_K`,
  - `Q8_0`,
  - `F16`,
  - `F32`.
- Decide per tensor class whether to:
  - implement native CUDA dequant/matvec,
  - convert once into WeightPack layout,
  - keep high precision and upload directly.
- Start with one matvec path per family:
  - Q_K routed expert down/up/gate,
  - IQ routed expert gate/up if required,
  - Q8 attention/output/shared expert path.
- Add byte-size and shape validators before loading huge tensors.
- Add `compare-reference` path against llama.cpp logits or golden tokens where CPU
  Ferrule cannot provide a DeepSeek reference.

Acceptance:

- `ferrule inspect-gguf` or `info` reports unsupported quant classes clearly.
- First implemented GGUF quant kernel has unit tests against a CPU decoder.
- DeepSeek V4 execution remains guarded until every tensor class in one layer has
  an execution policy.

Review focus:

- no silent F32 fallback for 80–150 GiB models,
- quant layout names must match GGUF/ggml semantics,
- benchmark each kernel before claiming model-level speed.

---

## 15. P12 — MoE-Specific Differentiation

Goal: make Ferrule better than dense-first runtimes for sparse expert systems.

### P12.1 Expert activation telemetry

Tasks:

- Count selected experts per layer.
- Track average top-k weights.
- Print `/experts` in chat or profile report.

Acceptance:

- profile shows hot/cold experts for a prompt.

Review focus:

- telemetry should be optional and low overhead.

### P12.2 Expert residency and streaming policy  ✅ PARTIAL

Goal: make routed experts streamable without changing their source precision.
This is the primary DeepSeek V4 quality-first path and the architectural answer
for models that do not fit in VRAM, host RAM, or local storage all at once.

Current status:

- `ferrule-runtime::expert_streaming` defines generic, model-family-agnostic
  streaming primitives:
  - `ExpertId`, `ExpertTensorKey`, `ExpertMatrixKind`,
  - `ExpertSource` for GPU, CPU, host mmap, local shard, local tensor sets,
    WeightPack chunk, and remote/LAN source,
  - `ExpertStreamingPolicy` with source-preserving quality-first defaults,
  - `ExpertStreamingPlanner` for selected expert loads, prefetch, eviction, and
    remote-source gating,
  - `ExpertStreamingReader` for bounded local byte-range reads.
- DeepSeek-like MoE `EnginePlan` policies now report `residency=streaming_allowed`.
- DeepSeek V4 HF routed expert tensors are bound from shard headers into target
  model source sets: 43 layers × 256 experts × 3 matrices × weight/scale =
  66,048 streamable tensor slices. The local DGX Spark smoke reads one selected
  expert's six slices from real safetensors shards without materializing the
  full model; `ferrule expert-stream-smoke` exposes the same path manually.

Tasks:

- Map DeepSeek V4 HF/WeightPack expert tensors into `ExpertSource` entries:
  - gate/up/down per `(layer, expert)`,
  - source shard path, byte offset, byte length, dtype/source format. ✅ HF local-shard mapping done for target model experts.
- Add a concrete streaming backend:
  - header/index lookup, ✅ HF header inventory done,
  - bounded read buffer, ✅ local byte-range reader done,
  - optional mmap path,
  - async/local prefetch hooks, TODO later; do not block the correctness path,
  - CUDA upload handle once kernels exist.
- Add execution integration points:
  - router selects experts,
  - planner loads selected experts,
  - MoE step consumes loaded expert handles,
  - prefetch predicted experts,
  - evict LRU/non-target experts.
- Support source tiers in priority order:
  - GPU resident,
  - host mmap / local shard,
  - WeightPack chunk,
  - optional remote/LAN source.
- Keep the older activation telemetry/top-N manager as a policy input, not the
  streaming source of truth.

Acceptance:

- selected experts are planned as mandatory loads before MoE execution,
- prefetch never evicts selected experts,
- remote sources fail unless explicitly enabled,
- tests cover local-shard, remote-source, eviction, and insufficient-slot cases,
- DeepSeek V4 keeps source FP4/FP8 payloads until a separate quantization profile
  is explicitly selected.

Deferred streaming I/O optimization TODO:

- Keep the first executable path synchronous and bounded: source tensor slice →
  byte-range read → source-format decode/dequant → expert compute handle.
- Add async expert streaming only after correctness works end-to-end.
- Introduce an I/O backend trait before adding platform-specific APIs, so runtime
  is not tied to `io_uring`.
- Linux/NVMe backend later:
  - `io_uring` submission/completion queues,
  - priority queue: selected experts > next-layer prefetch > speculative prefetch,
  - read coalescing for adjacent slices in the same shard,
  - bounded in-flight bytes,
  - cancellation for stale prefetch,
  - metrics: queue latency, read latency, bytes/s, cache hit rate.
- Keep portable fallbacks:
  - sync `pread`/`std::fs` byte-range reader,
  - optional mmap reader.

Review focus:

- streaming policy must remain generic; no DeepSeek-named runtime fields,
- no all-in-RAM expert materialization,
- source-preserving path first; IQ2/Q2 resident path later as an optional stack,
- executor integration must keep unsupported kernels as named missing-policy
  errors until fully implemented,
- do not let async I/O optimization delay source-format decode and expert compute
  integration.
- keep policy separate from kernel implementation.

### P12.3 Expert prefetch  ✅ DONE

Tasks:

- Use recent router history to prefetch likely experts.
- For each layer, prefetch next-token likely hot experts.

Acceptance:

- trace shows prefetch decisions,
- no correctness dependence on prediction.

Review focus:

- prefetch misses should not break generation,
- async transfer design needs clear ownership.

### P12.4 Expert batching  ✅ DONE

Tasks:

- Across sequences, group same expert execution.
- Across top-k within a token, batch expert GEMVs where possible.

Acceptance:

- after scheduler exists, MoE decode throughput improves with batches.

Review focus:

- needs sequence batching first,
- preserve per-token weighted accumulation order or numerical tolerance.

---

## 16. P13 — Edge, Cloud, and Hardware Co-Design

Goal: turn Ferrule artifacts into portable runtime state.

### P13.1 WeightPack package

Tasks:

- Treat WeightPack as deployable artifact:
  - manifest,
  - chunks,
  - checksum,
  - source model identity,
  - runtime compatibility.
- Add cloud-built WeightPack flow:
  - build on large machine,
  - deploy to edge machine,
  - run without source safetensors.

Acceptance:

- edge runtime can start from WeightPack + tokenizer/config only.

Review focus:

- reproducibility and compatibility.

### P13.2 Hardware policy layer

Tasks:

- Define abstract hardware resources:
  - GPU memory,
  - CPU RAM,
  - NVMe bandwidth,
  - future NPU/RISC-V accelerator memory.
- Runtime policy maps state objects to resources.

Acceptance:

- no new backend required,
- policy can explain current allocation decisions.

Review focus:

- keep it descriptive until offload exists.

### P13.3 RISC-V path

Possible future hooks:

- tokenizer/control-plane offload,
- WeightPack verification/checksum engine,
- small expert/router acceleration,
- KV/page management controller,
- edge security/isolation for model artifacts.

Near-term task:

- document interface points only.

Acceptance:

- architecture doc has a realistic RISC-V integration section,
- no speculative code yet.

---

## 17. P14 — Generic Model Bring-up: DeepSeek V4 Vertical Slice + DSpark Speculation

Goal: prove the generic model bring-up contract on a difficult mainstream target.
DeepSeek V4 should become a first-class supported model, but the resulting
interfaces must also make Llama/Qwen/Mixtral-style bring-up easier. DSpark
belongs to the speculative decoding layer around a target model, not to the base
DeepSeek Transformer forward path.

Non-goal: do not immediately support every DeepSeek derivative or every GGUF
quant format, and do not introduce DeepSeek-only shortcuts into generic runtime
state. The first milestone is a correctness-first single-request path for one
target DeepSeek V4 artifact; DSpark comes through the generic speculation
interface after the target path can produce correct tokens.

Current state:

- DeepSeek V4 architecture can be detected from GGUF metadata.
- DeepSeek V4 tensor classes can be classified.
- `ModelSupportContract` and `EnginePlan` can describe generic dense, MoE,
  OLMoE, and DeepSeek-style layouts.
- `SourceArtifact`, `ConversionPlan`, and `QuantizationRecipe` skeletons exist,
  including official `deepseek-ai/DeepSeek-V4-Flash-DSpark` source metadata.
- Typed HF source binding now covers DSV4 attention, HC, router, shared experts,
  and routed experts without putting concrete tensor names in runtime code.
- Real local DSV4 source-binding smokes pass for attention core tensors,
  compressor/indexer auxiliary slices, layer/global HC tensors, routers, shared
  experts, tokenizer, and one selected routed expert.
- Execution is guarded with an explicit engine-plan missing-policy error until the
  generic layer runner is wired.
- The CUDA hot path is split into executor + attention/MoE/logits/KV steps.

### P14.1 Model bring-up contract + DeepSeek V4 and DSpark metadata  ✅ PARTIAL

Tasks:

- Define the minimal model-support contract in code or docs:
  - descriptor,
  - layout,
  - tensor bindings,
  - attention policy,
  - MLP/expert policy,
  - quant policy,
  - tokenizer/encoding policy,
  - validation policy.
- Keep base execution keyed on `ModelFamily::DeepSeekV4` unless DSpark metadata
  proves it is a different Transformer architecture.
- Record DSpark as an optional speculation/draft attachment when metadata or a
  sidecar manifest proves it exists.
- Add fixture metadata tests for:
  - `deepseek4`,
  - `DeepSeek-V4`,
  - DSpark artifact or attachment strings,
  - target Hugging Face GGUF tensor names.
- Extend `GgufTensorPolicy` only if DSpark artifacts attach new tensors; do not
  fork the base DeepSeek execution path.
- Print tensor-class counts in `ferrule info`.

Acceptance:

- `ferrule info <deepseek-v4.gguf>` shows family, architecture, DeepSeek
  attention tensors, experts, quantization counts, and tensor-class counts.
- The same model-support contract can describe at least one dense Llama/Qwen-like
  layout at descriptor level, even if execution is not implemented yet.
- DSpark artifacts fail as `metadata-known/speculation-unsupported` or
  `metadata-known/execution-unsupported`, not `unknown`.

### P14.1b Official source artifact and conversion plan  ✅ DONE

Tasks:

- Treat official `deepseek-ai/DeepSeek-V4-Flash-DSpark` HF safetensors as the
  source-of-truth artifact.
- Represent source metadata separately from execution artifacts:
  - repo id,
  - revision,
  - architecture/model type,
  - shard count,
  - declared source dtypes,
  - tokenizer/encoding files,
  - official inference files,
  - DSpark attachment marker.
- Add `ConversionPlan` and `QuantizationRecipe` skeletons.
- Make WeightPack the primary Ferrule target and GGUF an optional compatibility / PK target.

Acceptance:

- official source artifact metadata fixture exists.
- DeepSeek Flash conversion plan can target WeightPack and records reference-validation needs.
- GGUF conversion plan explicitly reports compatibility/PK intent.

### P14.2 Generic layer-layout IR + DeepSeek binding  ✅ PARTIAL

Tasks:

- Define a backend-neutral layer layout description using semantic roles rather
  than model-specific field names:
  - norms,
  - attention projections and auxiliary attention tensors,
  - dense MLP or router/router bias/hash routing,
  - routed expert gate/up/down,
  - shared experts,
  - residual/hyper-connection style auxiliary tensors,
  - output head/norm.
- Implement DeepSeek V4 as one binding from HF/GGUF tensor descriptors into those
  semantic roles:
  - DeepSeek attention projections and compressor/indexer tensors,
  - router bias,
  - hash routing tables,
  - routed and shared experts,
  - HC/output-HC or mHC-related auxiliary tensors.
- Build layout from source tensor names/headers without loading full tensor
  payloads; then bind only bounded required payloads for the layer being executed.
- Validate required tensor presence per layer.
- Current partial completion:
  - HF parser emits typed semantic descriptors for attention and HC tensors;
  - `HfSafetensorsInventory::{attention_tensors,hyper_connection_tensors}` exposes
    those descriptors;
  - runtime `bind_attention_from_hf`, `bind_hyper_connection_from_hf`, and
    `bind_hyper_connection_head_from_hf` read real local source payloads behind
    generic roles and validate source formats/shapes;
  - DSV4 names remain isolated in `ferrule-model::families::deepseek_v4`.
- Allow optional DSpark-specific draft/speculation tensors only via explicit
  `SpeculationPolicy` metadata.

Acceptance:

- `ModelDescriptor` can produce a DeepSeek layer-layout summary.
- The same IR can represent a dense attention + dense MLP layer without DeepSeek
  fields.
- Local DSV4 source-binding smoke validates core attention count `43 * 13`,
  compressor/indexer auxiliary slices, `43 * 6 + 3` HC tensors, and real layer-0
  attention/HC payload shapes.
- Missing required tensors produce a precise error naming layer and tensor class.

### P14.3 Reference correctness harness

Tasks:

- Choose a reference engine for DeepSeek V4:
  - llama.cpp branch/build that runs the target GGUF,
  - official DeepSeek inference code if available for the selected artifact,
  - or Python reference for tiny synthetic fixtures.
- Use official DeepSpec/DSpark code only for draft/speculation acceptance tests,
  not as proof that base DeepSeek execution is correct.
- Add golden prompts and store:
  - tokenized prompt ids,
  - first generated token,
  - first N tokens if deterministic,
  - reference logits/top-k if available.
- Add `compare-reference` design if CPU Ferrule cannot execute the model.

Acceptance:

- Ferrule can verify tokenization/template and first-token behavior against a
  reference before optimizing kernels.

### P14.4 DeepSeek attention decode fixture  ✅ PARTIAL

Tasks:

- Implement DeepSeek attention tensor upload/load for one layer fixture.
  - DONE for bounded HF source binding: layer-0 core attention payloads bind into
    semantic `AttentionSourcePayload`; layer-2 compressor/indexer tensors are
    preserved as auxiliary source slices.
  - NEXT: lower `AttentionSourcePayload` into executable CPU/CUDA buffers and step
    state.
- Define DeepSeek KV/cache state and append/view semantics.
- Implement correctness-first decode attention:
  - q low-rank path: `wq_a -> q_norm -> wq_b`, per-head normalization, rotary;
  - `wkv -> kv_norm`, KV rotary, source quant/dequant policy;
  - sparse top-k gather + attention sink denominator;
  - inverse rotary + `wo_a` + `wo_b`.
- Keep GQA path and DeepSeek path selected by `AttentionPolicy`, not model name.
- Keep DeepSeek tensor names inside the binding/layout layer; executor inputs are
  semantic roles and buffers.

Acceptance:

- Local DSV4 source-binding smoke passes for attention payloads and auxiliary
  compressor/indexer slices.
- Tiny DeepSeek-like fixture runs attention decode step.
- OLMoE fixture GQA tests still pass.

### P14.5 DeepSeek router and experts  ✅ PARTIAL

Tasks:

- Implement router bias handling (`exp_probs_b` / HF `gate.bias`). ✅ CPU/reference
- Implement hash-assisted routing tables (`ffn_gate_tid2eid` / HF `gate.tid2eid`) if required by the
  first target layers. ✅ CPU/reference
- Add shared expert execution path. ✅ CPU/reference source binding + tiny MoE fixture
- Add resident expert handle path. ✅ CPU/reference store; CUDA handle wiring still pending
- Add correctness-first CUDA packed-FP4 expert executor. ✅ initial source-preserving path
- Add routed expert WeightPack/GGUF upload policy and production expert batching/residency integration.

Acceptance:

- A tiny layer fixture runs router + selected experts and matches reference within
  tolerance.
- Real local DSV4 router/shared/routed expert source-binding smokes pass.
- Full DSV4 layer execution must still wire these pieces into the generic forward
  path.

### P14.6 Full single-request DeepSeek V4 smoke

Tasks:

- Wire tokenizer/encoding, prefill, decode, sampler, and stop handling.
- Start with one quantization profile known to fit the available hardware.
- If the full model cannot fit, implement layer/expert streaming or residency
  before claiming support.

Acceptance:

- `ferrule run <deepseek-v4.gguf> -p '...' -n 1` produces the reference first
  token or a documented tolerance/golden-token match.
- `ferrule chat` works only after `run` smoke is stable.

### P14.7 DSpark speculative decoding interface

Tasks:

- Define `TargetModel`, `DraftModel`, and `SpeculationPolicy` traits/types.
- Model DSpark as a draft/speculative module attached to a DeepSeek V4 target.
- Track proposed tokens, accepted tokens, rejected tokens, rollback count,
  acceptance rate, and speedup/slowdown.
- Keep MTP, DSpark, DFlash, and Eagle-style approaches behind the same policy
  boundary.

Acceptance:

- A tiny fake target/draft fixture can exercise accept/reject/rollback behavior.
- No DSpark-specific branches exist in the base DeepSeek V4 Transformer forward
  path.

Review focus for P14:

- no OLMoE-field assumptions in DeepSeek path,
- no DSpark-specific hacks in the base DeepSeek path,
- no fake support by skipping auxiliary tensors,
- no full-model memory blow-up if streaming/residency is required,
- every unsupported tensor class has an explicit error.

---

## 18. P15 — Competitive PK Suite

Goal: make Ferrule's claims falsifiable against the engines users already know.

### P15.1 Local single-request PK

Reference engines:

- llama.cpp for GGUF local inference,
- DSpark/DeepSeek reference runtime if separate,
- Ferrule CPU/GPU where supported.

Metrics:

- load time,
- TTFT,
- prompt tok/s,
- decode tok/s,
- peak RSS,
- GPU memory,
- first-token/golden-token correctness,
- output text for fixed seed.

Acceptance:

- one markdown table generated from raw JSON results,
- commands are copy-pasteable,
- model/quant/context/hardware are recorded.

### P15.2 Server throughput PK

Reference engines:

- vLLM,
- SGLang,
- llama.cpp server where relevant.

Metrics:

- requests/s,
- tokens/s,
- p50/p90/p99 latency,
- TTFT distribution,
- memory under concurrency,
- cancellation/streaming behavior.

Acceptance:

- only run after Ferrule scheduler/batching is integrated into server,
- single-request server results are not presented as high-throughput serving
  parity.

### P15.3 MoE-specific PK

Metrics:

- expert activation distribution,
- expert residency hit/miss rate,
- expert batching efficiency,
- bytes moved from CPU/NVMe per token,
- speed under constrained VRAM.

Acceptance:

- Ferrule can show a MoE-specific advantage or identify the missing bottleneck.

---

## 19. Suggested Implementation Order

If implementing alone from the current refactor state, prefer this order:

1. Lock the `ModelSupportContract` + `EnginePlan`/policy boundary in code
   comments and small traits/types. ✅ DONE
2. Add descriptor/layout smoke fixtures for one dense Llama/Qwen-like model and
   one MoE fixture so the contract is not DeepSeek-shaped. ✅ DONE
3. Add official HF `SourceArtifact`, `ConversionPlan`, and `QuantizationRecipe`
   skeletons for DeepSeek-V4-Flash-DSpark. ✅ DONE
4. Add DeepSeek V4 metadata fixtures and DSpark attachment/speculation metadata.
5. Build generic layer-layout IR, then implement the DeepSeek GGUF/HF binding and
   required tensor validation on top of it.
6. Parse local `model.safetensors.index.json` to inventory official source tensor
   names, shards, and dtypes without loading payloads.
7. Extend `info`/inspection output for tensor classes, quant classes, required
   policies, conversion plans, and unsupported execution reasons.
8. Add reference correctness harness against official DeepSeek inference code for
   encoding, tokenization, and first token.
9. Add DeepSeek attention decode fixture with tiny synthetic tensors before
   full-model tensors.
10. Add GGUF Q_K/IQ and/or official FP8/I8 decoder tests for one tensor type used
    by the selected artifact.
11. Add the first CUDA dequant/matvec kernel or explicit conversion policy for the
    most blocking DeepSeek quant class.
12. Add DeepSeek router bias/hash routing fixture.
13. Add shared expert execution fixture.
14. Run a one-layer DeepSeek Transformer fixture through `CudaTransformerExecutor`.
15. Run full single-request DeepSeek V4 smoke (`run -n 1`) if memory fits.
16. Add streaming/residency path if the target artifact cannot fit in VRAM/RAM.
17. Add DSpark `SpeculationPolicy` fake target/draft fixture after base target
    first-token correctness exists.
18. Add `bench-infer` PK manifest for Ferrule vs llama.cpp.
19. Optimize DeepSeek attention/expert kernels only after first-token correctness.
20. Run server PK against vLLM/SGLang only after scheduler/batching is wired.

Already completed prerequisites remain important:

- `compare-logits`,
- `bench-infer`,
- WeightPack manifest validation,
- WeightPack-only startup,
- generation config loading,
- chat template registry,
- minimal server,
- request/session/KV abstractions.

Rationale:

- DeepSeek V4 is the priority vertical slice, not a privileged design center;
  DSpark is the priority speculation track after the target can run.
- A dense descriptor/layout smoke fixture protects the generic contract from
  becoming DeepSeek-shaped.
- Correct tensor layout and quant semantics must come before kernels.
- Reference correctness must exist because Ferrule has no CPU DeepSeek path yet.
- DeepSeek attention KV state may not be the same abstraction as GQA KV state,
  but the generic KV policy must be able to express both.
- Streaming/residency is not optional if the chosen artifact does not fit.
- PK benchmark results must be generated from reproducible manifests, not manual screenshots.

---

## 20. Review Checklist for Each Patch

For every implementation patch, include:

### Behavior

- What command demonstrates the feature?
- Does CPU behavior still work?
- Does GPU Q4 behavior still work if CUDA is enabled?
- Does chat still preserve session state across turns?

### Correctness

- Is there a CPU reference path?
- Are EOS and stop strings handled explicitly?
- Are tokenizer/template changes deterministic?
- For MoE: are router softmax/top-k semantics unchanged?

### Generality

- Did the patch add model-specific names to generic runtime structs?
- Can the same abstraction describe at least one non-target model family?
- Are model-specific tensor names isolated to descriptor/layout/binding modules?
- Would a future Llama/Qwen/Mixtral contributor know where to plug in without
  copying a whole runner?

### Performance

- Does the patch add overhead to normal decode?
- If it claims speedup, what benchmark proves it?
- Are model load time and decode time reported separately?

### State ownership

- Where does session state live?
- Where does KV state live?
- Who owns tokenizer/chat template state?
- Is reset behavior explicit?

### Compatibility

- Does WeightPack compatibility have version checks?
- Are old caches rejected clearly if format changes?
- Are CLI defaults backward-compatible?

### Safety

- Any new unsafe code?
- Any new file writes? Are they atomic?
- Any network server? Are bind address and errors explicit?

---

## 21. What Not To Do Yet

Avoid these until prerequisites exist:

- Do not implement full continuous batching before request/session state and KV
  abstraction exist.
- Do not implement radix cache before a prefix cache and shareable KV pages exist.
- Do not delete CPU FP32 inference; it is the correctness anchor.
- Do not add broad model-family support before loader/runtime boundaries are
  cleaner.
- Do not rewrite all CUDA kernels before `bench-infer` and profiling exist.
- Do not add a new quantization format without `compare-logits`, perplexity, or
  a reference-engine golden-token harness.
- Do not make WeightPack format changes without manifest/version checks.
- Do not add DeepSeek-specific fields or branches to generic scheduler, sampler,
  session, or executor state; add a policy/layout role instead.
- Do not claim DeepSeek V4 execution until DeepSeek attention, router, experts,
  auxiliary tensors, and quant classes used by the selected artifact are all
  accounted for.
- Do not claim DSpark support until generic target/draft speculative decoding,
  acceptance/rejection, rollback, and metrics are implemented.
- Do not compare Ferrule against llama.cpp/vLLM/SGLang without recording model,
  quantization, prompt, context length, batch/concurrency, hardware, and command.

---

## 22. Near-Term Tickets Ready for Implementation

### Ticket A: `compare-logits`  ✅ DONE

Deliverables:

- CLI command.
- CPU/GPU teacher-forced comparison.
- first-divergence report.
- max/mean abs error.
- top-k overlap.

Validation:

```bash
cargo check -p ferrule-cli --features cuda
ferrule compare-logits models/OLMoE-Instruct -q q4 -p 'hi' -n 8
```

### Ticket B: `bench-infer`  ✅ DONE

Deliverables:

- CLI command.
- load-excluded benchmark.
- prompt/decode split.
- optional JSON output.

Validation:

```bash
cargo check -p ferrule-cli
ferrule bench-infer models/OLMoE-Instruct -q cpu -p 'hi' -n 8
```

### Ticket C: WeightPack manifest reader  ✅ DONE

Deliverables:

- manifest struct.
- version and quant layout checks.
- cache inspection command or `info --cache`.

Validation:

```bash
ferrule inspect-weightpack models/OLMoE-Instruct/model.q4_0_llama.weightpack
```

### Ticket D: WeightPack-only load path  ✅ DONE

Deliverables:

- lightweight model load on cache hit.
- direct quantized weight upload from WeightPack.
- no full FP32 layer load on cache hit.

Validation:

```bash
/usr/bin/time -v ./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 8
```

Expected:

- much lower max RSS than full load,
- cached startup remains fast.

### Ticket E: minimal server  ✅ DONE

Deliverables:

- `ferrule server` command.
- `/health`.
- `/v1/chat/completions` non-streaming.
- model loaded once.

Validation:

```bash
curl http://127.0.0.1:8080/health
```

### Ticket F: `ModelSupportContract` / `EnginePlan` policy skeleton  ✅ DONE

Deliverables:

- Add small typed placeholders or design notes for:
  - `ModelSupportContract`,
  - `ModelFamilyPolicy`,
  - `AttentionPolicy`,
  - `RouterPolicy`,
  - `ExpertPolicy`,
  - `QuantPolicy`,
  - `KvPolicy`,
  - `ParallelismPlan`,
  - `ResidencyPolicy`,
  - `SpeculationPolicy`.
- Add at least one non-DeepSeek descriptor/layout smoke fixture so the contract is
  not shaped around a single target model.
- Keep current OLMoE fixture path working through existing executor boundaries.
- Ensure unsupported policy combinations produce explicit errors.

Validation:

```bash
cargo check -p ferrule-cli --features cuda
cargo test -p ferrule-model model_support_contract
```

### Ticket G: DeepSeek V4 / DSpark metadata fixtures  ✅ DONE

Deliverables:

- Add DSpark attachment/speculation aliases if confirmed by artifact metadata.
- Add metadata-only fixtures/tests for DeepSeek V4 and DSpark artifact names.
- Ensure unsupported execution/speculation errors remain explicit.

Validation:

```bash
cargo test -p ferrule-model deepseek
ferrule info /path/to/deepseek-v4.gguf
```

### Ticket H: DeepSeek semantic layer-layout/source binding IR  ✅ PARTIAL

Deliverables:

- `DeepSeekLayerLayout` or generic layer-layout IR.
- Typed HF semantic descriptors for attention, HC, router, shared experts, and
  routed experts. ✅
- Runtime source-binding payloads for real DSV4 attention and HC tensors. ✅
- Required/optional tensor validation per layer.
- Tensor-class count, quant-class count, and missing-policy report.
- Remaining: assemble a single executable `LayerSourceBinding` consumed by the
  generic Transformer executor.

Validation:

```bash
cargo test -p ferrule-model deepseek
cargo test -p ferrule-runtime local_deepseek_v4_attention_and_hc_bind_real_sources_if_present
ferrule info /path/to/deepseek-v4.gguf
```

Expected:

- DeepSeek attention, routed expert, shared expert, router, hash-router, HC, and
  auxiliary tensor classes are counted/bound by semantic descriptor.
- Missing required tensors name the layer and class.

### Ticket I: Reference-engine correctness harness

Deliverables:

- Reference command manifest for llama.cpp/official DeepSeek runtime.
- Optional DeepSpec/DSpark reference manifest for speculation acceptance tests.
- Golden prompt set for DeepSeek V4.
- First-token / first-N-token expected outputs when deterministic.

Validation:

```bash
cargo test -p ferrule-runtime reference_manifest
```

Expected:

- Ferrule can store and compare reference outputs even before full DeepSeek CPU
  execution exists.

### Ticket J: DeepSeek attention execution tiny fixture

Deliverables:

- Synthetic DeepSeek-style attention tensor fixture.
- Real-source attention binding smoke for layer-0/layer-2. ✅
- DeepSeek KV/cache state shape definition.
- One-step decode attention path through generic `AttentionPolicy` dispatch and
  `CudaTransformerExecutor` policy.
- FP8 attention linear execution path or explicit CPU/reference fallback for the
  fixture.

Validation:

```bash
cargo test -p ferrule-cuda deepseek_attention_fixture --features cuda
```

Expected:

- OLMoE fixture GQA path remains unchanged.
- DeepSeek fixture does not use OLMoE field assumptions.

### Ticket K: GGUF Q_K/IQ quant prototype

Deliverables:

- CPU decoder for one GGUF quant class used by DeepSeek V4.
- CUDA matvec/dequant prototype or explicit conversion policy.
- Unit tests against known packed bytes.

Validation:

```bash
cargo test -p ferrule-quant gguf_quant
cargo test -p ferrule-cuda gguf_quant --features cuda
```

Expected:

- Quant layout names match GGUF/ggml semantics.
- Unsupported quant classes remain explicit.

### Ticket L: DeepSeek router/expert tiny fixture  ✅ PARTIAL

Deliverables:

- Router bias support. ✅ CPU/reference
- Hash routing table policy if required by target layers. ✅ CPU/reference
- Shared expert execution path. ✅ CPU/reference
- Routed expert fixture through existing MoE step boundary. ✅ CPU/reference
- Resident expert handle store. ✅ CPU/reference
- Correctness-first CUDA packed-FP4 expert executor. ✅ initial
- Remaining: full DSV4 layer integration and production expert batching/residency.

Validation:

```bash
cargo test -p ferrule-runtime routed_moe
cargo test -p ferrule-runtime local_deepseek_v4_router_binds_hash_and_score_layers_if_present
cargo test -p ferrule-runtime local_deepseek_v4_expert_streaming_reads_one_selected_expert_if_present
cargo oxide test -- -p ferrule-cuda
```

Expected:

- Router/expert fixture matches reference within tolerance.
- Shared experts are not silently skipped.
- Real local DSV4 router/shared/routed expert source smokes pass when model is present.

### Ticket M: DeepSeek V4 first-token smoke

Deliverables:

- Load one selected DeepSeek V4 artifact or sliced fixture.
- Run `ferrule run ... -n 1` through generic Transformer executor.
- Compare first token against reference.

Validation:

```bash
ferrule run /path/to/deepseek-v4.gguf -p 'hello' -n 1
```

Expected:

- first token matches the reference harness, or mismatch is explained by a known
  unsupported policy.
- unsupported full-model memory paths fail with an explicit streaming/residency
  error.

### Ticket N: DSpark speculative decoding plan

Deliverables:

- Treat DSpark as a speculative decoding / draft-model feature, not a separate
  base model unless metadata proves otherwise.
- Add `DraftModel`, `TargetModel`, and `SpeculationPolicy` design notes/types.
- Define acceptance/rejection counters and rollback behavior.
- Add benchmark manifest fields for draft model, target model, block size, and
  acceptance rate.

Validation:

```bash
cargo test -p ferrule-runtime speculation_policy
```

Expected:

- no DSpark-specific hacks in the base DeepSeek V4 execution path.
- DSpark can later plug into the generic speculative decoding interface.

### Ticket O: Competitive PK manifest

Deliverables:

- Add benchmark manifest schema for Ferrule vs llama.cpp/vLLM/SGLang.
- Include model, quant, prompt set, context length, hardware, and commands.
- Add raw JSON result storage and markdown summary generation plan.

Validation:

```bash
cargo test -p ferrule-runtime pk_manifest
```

### Ticket P: DSV4 executable layer vertical slice

Deliverables:

- Add a typed `LayerSourceBinding` or equivalent generic per-layer source bundle
  that composes:
  - attention source payload,
  - HC attention/FFN weights,
  - router payload,
  - shared FFN payload,
  - routed expert source/resident handles,
  - layer-local KV/cache step state.
- Implement a tiny executable DSV4-shaped layer fixture through generic executor
  traits, not a `DeepSeekForward` copy of OLMoE.
- Start with CPU/reference execution for correctness, then attach CUDA backends for
  FP8 linears, sparse attention, HC, and FP4 experts behind policies.
- Keep source names out of runtime; runtime consumes semantic descriptors and
  source payloads only.

Validation:

```bash
cargo test -p ferrule-runtime dsv4_layer_vertical_slice
cargo test -p ferrule-runtime local_deepseek_v4_attention_and_hc_bind_real_sources_if_present
cargo check -p ferrule-cli --features cuda
```

Expected:

- One synthetic DSV4-shaped layer can run HC → attention → HC → MoE/shared FFN → HC.
- Real local DSV4 layer-0 source payloads can be bound into the same generic bundle.
- The fixture exposes any remaining attention/KV/FP8 execution gaps before the full
  43-layer first-token smoke.

---

This temporary roadmap should be deleted or merged into the canonical roadmap once
the runtime architecture and DeepSeek-family bring-up plan are stable.
