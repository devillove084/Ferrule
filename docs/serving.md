# Ferrule Serving Architecture

_Status: OpenAI-compatible greedy serving implemented; official benchmark validation pending._

## Framework choice

Ferrule uses **Axum 0.8 on Hyper 1 and Tokio**. TechEmpower plaintext rankings are a
useful HTTP-stack sanity check, but an LLM server is dominated by long-lived SSE
responses, cancellation, backpressure, scheduler admission, and GPU execution rather
than tiny-response request parsing. Axum/Hyper provides a competitive low-level HTTP
stack while retaining mature Tower/Tokio interoperability and a direct path to custom
Hyper bodies if profiling later justifies it.

`ntex`, `may-minihttp`, custom epoll loops, alternative allocators, and concurrent maps
are not selected merely for plaintext leaderboard position. Ferrule's request table is
owned by one model thread, so a standard `HashMap` has no lock or contention. The only
cross-thread structures are Tokio's bounded `mpsc`/`oneshot` channels and atomic
cancellation flags.

## Ownership model

```text
Tokio/Axum HTTP tasks
  -> bounded model-command channel
  -> one dedicated OS thread per loaded model
       -> tokenizer
       -> ResidentTopKDriver
       -> ResidentScheduler
       -> NativeMultiSessionExecutor
       -> model/CUDA runner
  -> bounded per-request token channel
  -> HTTP task converts events to SSE
```

The model, CUDA initialization, prepared resources, driver, scheduler, logical KV page
manager, and backend physical pages are all created and used on the same owner thread.
Handlers never lock or execute the driver. The worker processes a bounded number of
new commands between model steps so admission traffic cannot starve decode.

Every request has its own bounded token channel. A full or closed channel sets that
request's cancellation flag; it never blocks the model worker and never returns an
error through the driver's token callback. The worker observes cancellation between
model steps, calls `ResidentTopKDriver::cancel_request`, and releases scheduler, model,
and paged-KV state. Cancellation is therefore cooperative at a model-step boundary;
an already-running CUDA batch is not interrupted.

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Both completion endpoints support streaming and non-streaming responses. Streaming
uses `text/event-stream`, content-bearing token events, a final finish-reason event,
an optional usage-only event with `choices: []`, and `data: [DONE]`. Chat does not emit
an initial role-only event because benchmark clients can incorrectly count it as TTFT.
Proxy buffering is disabled with `X-Accel-Buffering: no` and `Cache-Control: no-cache,
no-transform`.

Malformed JSON and rejected options use an OpenAI-shaped error envelope. The request
queue is bounded; overload returns HTTP 429 rather than consuming unbounded memory.

## Current truthful request surface

Implemented generation semantics:

- `temperature = 0`
- `n = 1`
- `top_p = 1`
- `top_k = 1`
- `min_p = 0`
- `repetition_penalty = 1`
- `max_tokens` / `max_completion_tokens`
- `stop` string or array
- per-request `ignore_eos`
- `stream_options.include_usage`
- deterministic `seed` accepted for greedy requests

Unsupported sampling, logprobs, tools, response formats, echo, `best_of > 1`, and batch
prompts are rejected explicitly instead of being silently ignored.

## Start the server

```bash
just dsv4-serve
```

Equivalent explicit command:

```bash
cargo oxide build --features cuda --arch sm_121a -- --release -p ferrule-cli
./target/release/ferrule serve models/DeepSeek-V4-Flash-DSpark \
  --backend cuda \
  --served-model-name deepseek-v4 \
  --host 127.0.0.1 \
  --port 8000 \
  --ctx-size 4096 \
  --max-active-sequences 64 \
  --max-batch-tokens 4096
```

Important tuning controls:

- `--max-active-sequences`: resident scheduler and KV capacity.
- `--max-batch-tokens`: total prefill plus decode tokens in one action.
- `--prefill-chunk-size`: maximum prompt chunk per sequence.
- `--request-queue-capacity`: HTTP-to-model admission bound.
- `--event-queue-capacity`: independent SSE backpressure bound per request.

## Compatibility smoke

Chat:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"deepseek-v4",
    "messages":[{"role":"user","content":"Hello"}],
    "max_completion_tokens":8,
    "temperature":0,
    "stream":true,
    "stream_options":{"include_usage":true},
    "ignore_eos":true
  }'
```

Text completion:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"deepseek-v4",
    "prompt":"Hello",
    "max_tokens":8,
    "temperature":0,
    "stream":true,
    "stream_options":{"include_usage":true},
    "ignore_eos":true
  }'
```

## Official benchmark targets

Primary neutral harness:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-v4 \
  --tokenizer models/DeepSeek-V4-Flash-DSpark \
  --dataset-name custom \
  --dataset-path prompts.jsonl \
  --custom-output-len 8 \
  --num-prompts 10 \
  --max-concurrency 2 \
  --temperature 0 \
  --save-result \
  --save-detailed
```

SGLang cross-check:

```bash
python -m sglang.benchmark.serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-v4 \
  --tokenizer models/DeepSeek-V4-Flash-DSpark \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 8 \
  --num-prompts 10 \
  --max-concurrency 2 \
  --temperature 0 \
  --output-file ferrule-smoke.jsonl \
  --output-details
```

For direct engine comparisons, use the same client, prompt file, tokenizer, seed,
input/output lengths, request-arrival process, warmup, and concurrency for every
server. SGLang recommends at least `num_prompts >= 5 * max_concurrency` for steady
state.

## Remaining work

- Run and archive official vLLM/SGLang smoke and concurrency sweeps.
- Add server metrics and profiling endpoints; client-side latency metrics do not depend
  on them.
- Implement device sampling before accepting non-greedy API settings.
- Connect automatic radix-prefix lookup/admission for independent API requests.
- Add E7 graph buckets and E8 profiler-driven fusion.
- Add optional auth, TLS/proxy deployment guidance, graceful request-drain deadlines,
  and production observability.
