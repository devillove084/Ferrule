# Read-only Ferrule diagnostic analysis

This directory is independent of the experiment runner. Nothing here launches a
server, imports CUDA/Torch/NVML, changes a protocol/configuration, or writes an
analysis result automatically. The CLIs print JSON to stdout. Tests create and
remove header-only fixtures **inside this directory**. Use the base interpreter
with `-S -B` to avoid site hooks and bytecode files.

## API and tests

`__init__.py` exports:

- `analyze_log(path)` / `analyze_lines(lines, source=...)`: analyze **one server
  process lifetime**. Supply a separate call/source for each restart.
- `analyze_http(report)`: extract maintained client measurements separately.
- `expert_headers(model)`: validate the complete routed-expert directory using
  config/index plus bounded safetensors JSON headers; no payload reads.
- `cache_capacity(...)` and `counterfactuals(metadata, baseline_required_device_bytes)`:
  return capacity bounds and proposed flag diffs, never launch authorization.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  /opt/conda310/bin/python3 -S -B -m unittest discover \
  -s papers/arxiv/experiments/analysis -p 'test_*.py' -v

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  /opt/conda310/bin/python3 -S -B \
  papers/arxiv/experiments/analysis/performance.py \
  papers/arxiv/experiments/runs/m7-feasibility-exclusive-002/ferrule/server.log \
  --http-report papers/arxiv/experiments/runs/m7-feasibility-exclusive-002/ferrule/http.json

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  /opt/conda310/bin/python3 -S -B \
  papers/arxiv/experiments/analysis/cache_plan.py \
  /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
  --baseline-required-device-bytes 6846743296
```

For a later diagnostic, replace only the input log/report paths. The baseline
owner must provide those paths; there is no automatic discovery of active runs.

## Interpretation contract

The parser accepts the maintained JSON logger's exact target/event pairs:
`ferrule_performance` / `mixed_decoder_invocation`, and `ferrule_request` /
`request_admitted`, `request_http_admitted`, `request_worker_terminal`,
`request_http_terminal`. Plain stderr/model-plan lines are not performance events.

- Invocation counters are already deltas: **do not difference adjacent events**.
  They belong to the shared batch transaction, not each participating request.
  No transaction-to-request cost join is invented.
- Prefill, decode, mixed and unknown groups are mutually exclusive. Mixed costs
  are not split proportionally into supposedly measured prefill/decode costs.
  `first_observed_prefill` means first pure-prefill event in this input file, not
  necessarily cold startup, first request, or first token of each request.
- Per-row aggregates divide known total cost by known total rows, not the mean
  of differently sized invocation ratios. Failed/quarantined observations remain
  visible; completed non-quarantined invocations have a separate summary.
- Missing, malformed, negative, nonfinite and reset counters remain `null`.
  Incomplete aggregate `total` is `null`; `observed_sum` is explicitly partial.
  A genuinely observed zero remains zero. A reset invalidates the cumulative
  delta family conservatively; independent host proof timings/gauges remain usable.
- Exact repeated JSON records are deduplicated. Conflicting transaction/device
  identities block aggregate totals rather than guessing which event to retain.
- Expert live/scratch/pending/quarantine bytes and owner-lifetime peak are gauges:
  summarize max/last, **never sum them**. Partial maxima are labeled `observed_max`.
- Proof times are host elapsed time for the three instrumented call groups. They
  do not measure native event-query counts or GPU kernel time. Facade kernel and
  sync counters do not include all cuBLAS/internal activity. Checkpoint read bytes
  are bytes requested through the reader, not measured disk traffic.
- Allocation attempts/bytes are available. Allocation duration is **unmeasured**
  and explicitly `null`, not derived from whole-invocation latency.
- Worker first/last times are elapsed from worker enqueue, with intervals between
  successful channel emissions (including empty-text emissions). These are not
  client SSE arrival times. Zero/one emission has no mean interval. Disabled or
  absent timing stays unknown. Cancellation does not become a successful TPOT.
- Client `mean_tpot_usage_proxy_seconds` stays a proxy; no token-exact TPOT is
  reconstructed from content chunks. Client observations are not silently joined
  to worker identities.

## Retained M7 evidence — not a diagnostic run

Read-only inspection of `../runs/m7-feasibility-exclusive-002/ferrule/` found:

- `launch.json`: `FERRULE_LOG=info`, JSON format; public `serve`, CUDA,
  `qwen35-mixed-bf16`, resident KV, lazy experts, full 40 layers.
- `server.log`: requested/effective context 128, batch/chunk 8, sequences 2;
  expert entries 8, cache 335544320 bytes, scratch 8 MiB, workspace 2048 MiB,
  total admission ceiling 7168 MiB. Logged required total: **6,846,743,296 bytes**.
- **No** `ferrule_performance` invocation or request-worker terminal observations.
  Parser returns `instrumentation="not_observed"`; causal counters are unknown.
- `http.json`: one `Hello` raw-completion request, 1 input token, **2 output
  tokens**, client TTFT 0.6655100006610155 s, response E2E 1.8603699430823326 s,
  usage-normalized TPOT proxy 0.6120898853987455 s, exact token TPOT `null`.

This is Hello2 feasibility, **not** an eight-generation-token validation, a
repeatability result, or an exact vLLM speedup comparison. Historical full40/fresh
quality evidence does not approve a new release/configuration automatically.
No G10 active-run artifacts or GPU state were inspected.

## Cache counterfactuals for the baseline owner

### Source/header evidence

- `crates/ferrule-model/src/transformer/standard/expert_cache.rs`:
  `ExpertCacheKey` includes layer/expert; the cache is image-wide. Its entry cap
  includes resident/pending entries; eviction uses completed entries only.
- `crates/ferrule-model/src/transformer/standard/cuda/residency.rs::metadata_plan`:
  Mixed charges all gate/up/down numeric payloads plus their scales per expert.
- Actual checkpoint headers: each expert has three 1,048,576-byte E4M3 weight
  matrices and three 128-byte BF16 scale grids: **3,146,112 bytes per expert**.
  The header-only API verifies all **10,240** layer-qualified expert identities.
- Forty layers, top-8: one full single-row traversal visits **320 distinct
  layer-qualified experts**. Two rows can span at most 640 unique identities;
  this is not measured routing reuse. The catalog contains 40 × 256 = 10,240.
- `cuda/bf16_hybrid.rs::bf16_routed` already groups rows by expert. Slot changes
  do not implement a new M>1 grouping strategy.

### Fixed-budget entry-cap ablation

Keep every other CLI/workload setting unchanged. Compare:

| Case | Expert bytes | Entry cap | Payload-only resident upper bound |
|---|---:|---:|---:|
| A | 335544320 (320 MiB) | 8 | 8 |
| B | 335544320 (320 MiB) | 256 | 103 |

The only flag diff is `--cuda-expert-device-entries: 8 -> 256`. Both retain
`--cuda-expert-scratch-mb 8`, `--cuda-workspace-mb 2048`, and the same total device
ceiling. It is a valid **entry-cap-setting** ablation, not evidence that 256
experts fit: floor((320 MiB - 8 MiB) / 3146112) = 103, before additional operation
scratch/pending consumers. The 320 MiB is a ceiling, not actual residency.

A useful negative-control hypothesis: for uninterrupted serial M=1 full-depth
traversals, the other 39 layers alone touch 312 distinct expert identities between
revisits to a layer. Neither 8 nor at most 103 slots can hold that reuse distance.
Do not promise cross-token cache hits or speedup from B. Actual grouping, call
counts, cache statistics and latency from the diagnostic must decide.

### Larger-budget candidate (not same-budget acceleration)

Proposed M7-relative flag diff:

```json
{
  "--cuda-expert-device-entries": "512",
  "--cuda-expert-device-bytes": "2147483648",
  "--cuda-device-budget-mb": "10240"
}
```

At 2 GiB, the payload-only byte bound is 679 experts; the 512-entry cap then
limits it to at most 512. This can hold a 320-identity single-row working set in
principle, but not the full catalog and not necessarily a two-row working set.

Source-derived conditional total:

```
6846743296 + (2147483648 - 335544320) = 8658682624 bytes
```

This is **8.064 GiB**, below a 10 GiB (10240 MiB) admission ceiling by
2,078,735,616 bytes. The source basis is
`decoder/hybrid_memory_plan.rs::numeric_bf16_resources` (cache-minus-scratch added
to weight bytes), and `model_factory/bf16_serving.rs::open_hf_with_precision`
(fixed granted workspace/state/KV/margin in the total). Weight-dependent workspace
*demand* also increases; its extra weight-slack term is at most 181,193,933 bytes.

**This Python arithmetic is not a fresh invocation of the Rust factory planner,
public CLI admission, physical-memory measurement, or quality approval.**
`factory_metadata_admission="not_run"` is intentional. Do not launch solely from
this bound. The public `info` command does not validate a serving capacity plan,
and `serve` would proceed to loading/CUDA, so neither was used to fake a
metadata-only serving admission.

### Owner's gated sequence (no protocol changes made here)

1. Finish the currently owned actual G10 baseline; keep its uninstrumented
   timings as baseline data. Do not modify that runner/configuration in place.
2. Create a separate diagnostic with identical public Mixed CUDA profile,
   full-depth model, template/raw endpoint, greedy/EOS policy, context 128,
   batch/chunk 8, sequences 2, KV 16 MiB, scratch/workspace and budget. Enable
   `FERRULE_LOG=info,ferrule_request=debug,ferrule_performance=debug` and JSON.
3. Require actual successful **8 generated tokens**, reconciled worker events,
   and fresh same-budget quality evidence for the pinned release. Natural EOS
   can shorten output: report it, do not silently change EOS policy to force 8.
   Keep client TPOT proxy distinct from worker interval means.
4. Run A/B entry-cap cases only after the owner approves new prepared identities.
   On a G10 comparison, use `--cuda-device-budget-mb 10240` in **both** A and B;
   this does not increase their 320 MiB expert-cache cap.
5. Validate the 2 GiB candidate's Rust metadata/workspace plan and live admission
   separately. Under G10 use a paired 320 MiB control with the same 10 GiB ceiling;
   do not describe the increased cache allowance as a same-cache-budget win.
6. Preserve lazy loading, stable page/storage custody, exact consumer-event proof,
   unknown-completion handling, and numerical contracts. No CPU fallback or
   alternative numeric profile. Treat OS/page/JIT cache state as uncontrolled
   unless the owner actually controls it.
7. Compare measured proof call counts/host time, reads/uploads/evictions and
   per-row costs. Only then choose a system optimization. This change implements
   **no** proof elimination, decoded-weight cache, or kernel optimization.
