# Temporary Inference Engine Roadmap

> Status: working implementation roadmap, not the canonical product roadmap.
>
> Purpose: make Ferrule's gap against current inference frameworks explicit so
> implementation can proceed in small, reviewable patches.

Ferrule's current direction is **Ferrule Runtime**: a Rust-native, state-aware,
hardware-aware LLM runtime. The near-term engineering target is not to copy every
feature from every framework. The target is to reach a strong local inference
baseline first, then add modern serving and state-management capabilities in the
same order they become useful.

Current baseline:

- OLMoE / OLMoE-Instruct safetensors loading.
- CPU FP32 reference inference.
- GPU Q4_0 quantized inference through cuda-oxide kernels.
- llama.cpp-compatible Q4_0 block layout.
- explicit router → top-k experts → expert loop.
- correct OLMoE `norm_topk_prob=false` semantics.
- single-session GPU KV cache.
- `qcache` for quantized layer weights.
- `ferrule-runtime` abstraction for CPU/GPU runners, session reset, sampling,
  and chat generation loops.
- llama.cpp-style sampling controls: temperature, top-k, top-p, min-p,
  repeat penalty, seed, stop strings, top-K logprobs.
- `generation_config.json` auto-loading with model defaults.
- Chat template registry: OLMoE-Instruct, ChatML, Llama3, Qwen, Plain.
- CLI commands: `info`, `run`, `gpu-run`, `chat`, `cuda`, `bench-infer`,
  `compare-logits`, `inspect-cache`, `server`.

---

## 1. Framework Gap Snapshot

### 1.1 llama.cpp

llama.cpp is the closest local-inference reference. It is strong because it has a
complete local workflow, not just fast kernels.

| Area | llama.cpp has | Ferrule has | Gap |
|---|---|---|---|
| Model format | GGUF single-file model, rich metadata, tokenizer/chat template info | safetensors loader, partial GGUF reader, qcache sidecar | qcache manifest and qcache-only startup; GGUF import/export story unclear |
| Model coverage | broad dense/MoE model families | OLMoE path only | add model-family abstraction before more loaders |
| Quantization | many mature formats: Q4_0/Q8_0, K-quants, IQ quants, mixed CPU/GPU support | Q4_0/Q8_0/Q2S/T1S primitives; Q4_0 path validated best | Q8 full-path validation, mixed precision, K-quant/AWQ-like quality track |
| CLI UX | prompt/chat, many sampling flags, templates, context controls | prompt/chat, initial sampling flags | generation config loading, template registry, context/window controls |
| Correctness tools | perplexity, eval examples, broad community regression coverage | manual smoke tests and CPU reference | logits diff, golden-token tests, perplexity command |
| Benchmarks | `llama-bench`, prompt/decode split | CUDA GEMV probe, runtime stats internally | `bench-infer` with pp/tg split and reproducible output |
| Server | OpenAI-compatible server, streaming, embeddings | none | minimal `/v1/chat/completions` server |
| Grammar | grammar / JSON-constrained decoding | none | stop strings only; grammar FSM later |
| KV | mature local KV management, context shifting, some quant/offload paths | single contiguous GPU KV cache | session object, KV stats, later paged KV |
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
| KV memory | PagedAttention block allocator | contiguous single-session KV inside `GpuOlmoeModel` | separate KV cache trait, then paged KV allocator |
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
| Mobile packaging | deployment artifacts | none | later qcache package format |
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
| Memory planning | explicit memory budget | fixed GPU allocation | memory planner + qcache chunk metadata |
| Edge deployment | out-of-core inference | none | qcache chunks + async transfer |

Ferrule has a natural advantage here because MoE experts are explicit runtime
objects. This should become a differentiating track after qcache-only startup and
basic benchmarking.

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
6. **Treat qcache as a runtime artifact.** It should become Ferrule's local and
   edge deployment unit.
7. **Preserve MoE semantics.** Router logits, top-k selection, expert weights,
   and normalization policy are correctness-critical.

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

---

## 7. P4 — qcache-Only Startup

Goal: avoid full FP32 model load on qcache hit.

This is the biggest local usability and memory milestone.

Current gap:

- GPU qcache hit still loads the full FP32 model before using cache.
- Q8_0 is blocked by RAM because loading FP32 + quantized weights exceeds
  available memory.

### P4.1 qcache manifest

Tasks:

- Add qcache header/manifest with:
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
- `ferrule info` or new `inspect-cache` can show manifest.

Review focus:

- never silently load incompatible qcache,
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
- Load quantized attention/expert weights directly from qcache.

Acceptance:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 16
```

On qcache hit:

- should not print full shard tensor loading count,
- startup should be close to lightweight load + GPU upload time,
- peak RAM should be much lower than FP32 full load.

Review focus:

- verify all tensors required by GPU path are present,
- no fallback to empty weights unless explicitly validated.

### P4.3 streaming qcache writer  ✅ DONE

Tasks:

- Quantize one layer at a time.
- Write qcache chunks incrementally.
- Drop FP32 layer tensors after quantization.

Acceptance:

- first qcache creation stays under a defined RAM budget,
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

### P7.1 Extract KV cache interface

Current gap:

- `GpuOlmoeModel` owns:
  - `k_cache: Vec<DeviceBuffer<f32>>`
  - `v_cache: Vec<DeviceBuffer<f32>>`
  - `cur_seq`
  - fixed `max_seq`

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

- First implementation can wrap existing contiguous cache.

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

---

## 14. P11 — Quantization Quality Track

Goal: improve quality beyond naive Q4_0 without losing edge usability.

### P11.1 Q8_0 full-path validation  ✅ DONE

Tasks:

- Use qcache-only or streaming loading to avoid RAM explosion.
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
- qcache manifest must support layout versioning.

### P11.4 KV quantization  ✅ DONE

Tasks:

- Wait until KV abstraction exists.
- Add FP16 first if useful.
- Then KIVI/KVQuant-style low-bit exploration.

Acceptance:

- long-context memory reduction with measured quality impact.

Review focus:

- do not combine with paged KV refactor in same patch.

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

### P12.2 Expert residency policy

Tasks:

- Define expert state:
  - resident on GPU,
  - resident on CPU,
  - resident in qcache/NVMe,
  - loading,
  - evictable.
- Start with static policy: keep all experts on GPU.
- Then allow selected layers/experts offloaded.

Acceptance:

- no behavior change with default policy,
- policy object exists and can be unit-tested.

Review focus:

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

### P13.1 qcache package

Tasks:

- Treat qcache as deployable artifact:
  - manifest,
  - chunks,
  - checksum,
  - source model identity,
  - runtime compatibility.
- Add cloud-built qcache flow:
  - build on large machine,
  - deploy to edge machine,
  - run without source safetensors.

Acceptance:

- edge runtime can start from qcache + tokenizer/config only.

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
- qcache verification/checksum engine,
- small expert/router acceleration,
- KV/page management controller,
- edge security/isolation for model artifacts.

Near-term task:

- document interface points only.

Acceptance:

- architecture doc has a realistic RISC-V integration section,
- no speculative code yet.

---

## 17. Suggested Implementation Order

If implementing alone, prefer this order:

1. `compare-logits`
2. `bench-infer`
3. qcache manifest validation
4. qcache-only startup
5. generation config loading
6. chat template registry
7. minimal server
8. explicit request/session types
9. KV cache interface
10. multi-session contiguous KV
11. paged KV
12. prefix cache
13. scheduler and continuous batching
14. structured decoding
15. kernel fusion / CUDA graph replay
16. expert telemetry and offload policy

Rationale:

- correctness tools must come before performance rewrites,
- benchmark tools must come before optimization,
- qcache-only startup unlocks Q8 and realistic edge deployment,
- server must exist before vLLM-style scheduling matters,
- paged KV needs explicit sessions first,
- SGLang-style prefix reuse needs paged/shareable KV first.

---

## 18. Review Checklist for Each Patch

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

- Does qcache compatibility have version checks?
- Are old caches rejected clearly if format changes?
- Are CLI defaults backward-compatible?

### Safety

- Any new unsafe code?
- Any new file writes? Are they atomic?
- Any network server? Are bind address and errors explicit?

---

## 19. What Not To Do Yet

Avoid these until prerequisites exist:

- Do not implement full continuous batching before request/session state and KV
  abstraction exist.
- Do not implement radix cache before a prefix cache and shareable KV pages exist.
- Do not delete CPU FP32 inference; it is the correctness anchor.
- Do not add broad model-family support before loader/runtime boundaries are
  cleaner.
- Do not rewrite all CUDA kernels before `bench-infer` and profiling exist.
- Do not add a new quantization format without `compare-logits` or perplexity.
- Do not make qcache format changes without manifest/version checks.

---

## 20. Near-Term Tickets Ready for Implementation

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

### Ticket C: qcache manifest reader  ✅ DONE

Deliverables:

- manifest struct.
- version and quant layout checks.
- cache inspection command or `info --cache`.

Validation:

```bash
ferrule inspect-cache models/OLMoE-Instruct/model.q4_0_llama.qcache
```

### Ticket D: qcache-only load path  ✅ DONE

Deliverables:

- lightweight model load on cache hit.
- direct quantized weight upload from qcache.
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

---

This temporary roadmap should be deleted or merged into the canonical roadmap once
P1/P2 are implemented and the runtime architecture stabilizes.
