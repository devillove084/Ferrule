# vLLM Qwen3.5 validation (opt-in)

Independent Python harness; no Ferrule import/build, dependency installation,
version changes, CPU inference fallback, environment repair or server launch.
Uses the interpreter you invoke (default shebang `/opt/conda310/bin/python`).
Core preflight/reporting/HTTP work with the standard library. `packaging` is
optional for starting preflight, but required to **evaluate** installed Python
and dependency constraints; absent/broken evaluation blocks generation.

```sh
# Default: check only. No tensor computation or LLM construction.
/opt/conda310/bin/python -B scripts/test_vllm_qwen35.py

# Metadata/model/read-only driver checks without importing torch/vLLM or running pip:
/opt/conda310/bin/python -B scripts/test_vllm_qwen35.py check --skip-import-probes --skip-pip-check

# Explicitly executes a small CUDA tensor operation and synchronize:
/opt/conda310/bin/python -B scripts/test_vllm_qwen35.py gpu --timeout 30

# Explicitly loads the actual local checkpoint with vLLM; may require multiple GPUs:
/opt/conda310/bin/python -B scripts/test_vllm_qwen35.py generate \
  --tp 1 --ctx 2048 --max-seqs 1 --memory 0.8 --max-tokens 32 --timeout 900

# An already-running OpenAI-compatible server; this script does NOT start one:
/opt/conda310/bin/python -B scripts/test_vllm_qwen35.py http \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --served-model Qwen3.5-35B-A3B-FP8 --warmup 1 --repeat 3 --max-tokens 32
```

Default local model: `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8`; override with `--model`.
`--tp N --ep` enables vLLM expert parallelism within its parallel group, not N
independent replicas or Ferrule thread EP. PP/DP are explicitly 1, prefix caching
and speculation are off, CPU offload is zero. No unverified Torch version is
hardcoded. This harness was written after reading installed vLLM **0.28.0**
metadata and API source, not importing or executing it. Its `LLM` forwards some
options to `EngineArgs`; both installed signatures and `SamplingParams` are
checked before construction. Unsupported options fail rather than being dropped.
`generate` uses a raw text prompt (no chat template), greedy temperature 0,
fixed seed, equal min/max tokens and ignore-EOS; short/non-length completion
is recorded as failure. It archives actual prompt/output token IDs, text,
finish reason, init/load and generation walltimes separately. Successful loading
or greedy generation is **not** numerical oracle acceptance.

## Diagnostics and artifacts

- `check`: installed `Requires-Dist` / `Requires-Python`, recursive active
  dependencies including requested extras, local config/tokenizer existence and
  hashes, read-only `nvidia-smi`, bounded `pip check` and separate subprocess
  imports of torch/vLLM. Import stderr preserves missing-module and ABI evidence.
  Nothing infers pip package names from import names. No install commands are
  generated from unevaluated markers. With `--no-deps` installations, satisfy the
  recorded active requirements yourself and rerun; preflight never relaxes them.
- All subprocesses use the existing `run_process` interface via an isolated,
  single-thread Linux subreaper helper. Before application launch it sets/verifies
  `PR_SET_CHILD_SUBREAPER`, resets SIGCHLD to a non-reaping handler, and completes
  a private parent handshake. Only kernel-verified unreaped direct children are
  signalled (`waitid` WNOWAIT plus birth/PPID checks); setsid/doublefork descendants
  are adopted. Cleanup stops fork sources, terminates/reaps iteratively and only
  kernel ECHILD proves process drain. No pidfd/cgroup privilege or group/name
  signals are used. Helper failure/timeout leaves custody unknown and blocks the
  next matrix restart. This is not a malicious-code sandbox or GPU-quiescence
  proof; external launch services are outside the contract. No inference CLI or
  vendor package is changed. HTTP keeps whole-worker/request deadlines: check/gpu
  30 s, generation 900 s, HTTP 300 s (plus helper handshake/cleanup margins).
- After saving JSON, every mode prints a bounded, sanitized summary to stdout:
  status/path, failure causes and manual next steps first, mode/model, recorded
  Python/vLLM/torch/CUDA runtime/driver information, passed/failed/skipped phases,
  missing dependencies/import errors, model checks, and measurements. Unknown or
  unexecuted values are `N/A`; summaries never run additional imports or probes.
  HTTP uses the served model ID and actual measured tokens, not requested limits.
  Only a limited number of errors/requests are displayed; the complete existing
  report is still saved. `--quiet` restores the old status/path-only output.
  Suggested `pip check`/`pip show` commands are manual diagnostics, not automatic
  installs; dependency names and installation commands are never inferred from
  unevaluated markers or import-module names. Exit codes and JSON schema are unchanged.
- Fresh private directories are created under
  `target/validation/vllm-qwen35-check/`; `--output PATH` must not already exist
  and is reserved **before** any test. `report.json` embeds sanitized subprocess
  log excerpts (last 64 KiB per stream). Existing artifacts are not overwritten.
  Schema: `ferrule.vllm-qwen35-check.v1`, with `phases`, `status`, `errors`,
  `missingdeps`, `versions`, `config`, `sourceidentity`, and `measurements`.
  Unexecuted measurements/oracle/quiescence remain `null`; phases say `notrun`.
  Exit 0 means the selected scope passed; 1 means diagnostic/test failure;
  2 means invalid CLI/output path. Read skipped phases before interpreting pass.
- Never dumps the environment: records presence/hash of a small relevant CUDA/
  loader/offline variable allowlist without changing their values. Known secret
  environment values, credential assignments, URL userinfo and private-key
  blocks are filtered from reports. Use only **non-sensitive prompts/models**:
  generated text and third-party diagnostics are archived; generic redaction is
  not a guarantee against secrets embedded arbitrarily in generated text.
- Identity hashes cover the script and local small metadata/tokenizer files,
  not all weight shards, runtime kernels, source commits or a remote server.
  Missing identities remain explicitly unknown. `http` does not inspect/import
  local torch/vLLM and cannot establish which implementation the server runs.

## HTTP accounting and comparison limits

Uses stdlib `urllib`, one sequential chat request at a time, with
`temperature=0`, seed, `max_tokens` and `stream_options.include_usage`.
No vLLM-specific `ignore_eos` is sent; HTTP output may end early at EOS.
Unsupported request fields fail visibly rather than being silently retried.
No auth CLI
or credentials in URLs; redirects are refused. Prompts go to the supplied server
(and inherited proxy configuration); ensure it is trusted. Warmup is separately
labelled and the first failure stops the run without replacement or retry.

Strictly requires a single choice, successful terminal finish reason, valid
final usage and `[DONE]`; rejects errors, missing/duplicate terminals, malformed
counts and no-content responses. TTFT is **client send to receipt of the first
non-empty `delta.content` SSE event**, not headers, role-only, empty, keepalive
or reasoning-only events. SSE events are not tokens: content event offsets/gaps
are reported, but TPOT and token ITL are **null**. Rates use actual usage token
counts and client walltime through `[DONE]`, never requested max tokens. No
aggregate throughput is claimed; a failed request retains elapsed worker time
and failure logs, while unavailable usage/partial token counts remain null.
HTTP does not assert fixed output length or that seed was honored;
compare actual usage and freeze the server contract before performance work.

Follow [CPU_REFERENCE](../docs/CPU_REFERENCE.md) and
[PERFORMANCE_EVALUATION](../docs/PERFORMANCE_EVALUATION.md): offline total is not
HTTP TTFT, no CPU oracle latency is a serving baseline, FP8 storage does not
imply matched arithmetic/native FP8 on SM86, and TP8 is not Ferrule EP8 equivalence.
Hardware/topology, shard/runtime identities, effective arithmetic, server
configuration, workload and independent quality gates must be frozen separately.
This is a smoke/diagnostic tool, **not** the full repeated paper benchmark.

## CPU-only regression tests

Run only this dedicated directory, not other GPU/reference script tests:

```sh
/opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p test_vllm_validation.py -v
```

Tests use standard-library mocks for metadata, subprocess preflight, API
signatures and SSE/HTTP; no torch/vLLM imports, GPU, model, server or network.
A bounded stdlib sleep-child test exercises real process-group cleanup, and
`--help` is checked in a stdlib subprocess. No packages are needed for tests.
