# Matched 4 GiB cache-off/on A/B — CPU preparation only

## Current state

No GPU query, UUID lock, HTTP server, model execution or profiler was run in this
preparation. The arena model/release and a fresh GPU authorization are pending.
There is no prepared executable A/B run or qualified arena admission yet.
Historical smoke/pilot/profile receipts were not changed.

Old artifact: `target/validation/tp2-mixed-optimized-release-20261009/ferrule-optimized`.
SHA256: `4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173`.
Future OFF execution reuses the already staged, hash-identical historical binary.
The candidate is deliberately **null**, not the prior dadf release relabelled
as an arena candidate.

## What the actual CPU probes prove

The old binary's actual `--help` / `serve --help` return 0 and advertise
`--cuda-workspace-mb` and `--cuda-device-budget-mb`.

The full strict TP2 arguments, including workspace **4096 MiB** and total device
cap **22528 MiB**, were then passed to that same binary under an explicit hidden
GPU environment. The model path was an intentionally empty, existing directory.
The process accepted parsing/pre-model argument validation and exited with:

```text
model directory ... has neither config.json nor a .gguf file
```

**This does not prove full model resource admission or GPU serving at 4096.**
Those must succeed in the later authorized OFF lifecycle. No real-model `serve`
was attempted: the old CLI has no demonstrated metadata-only dry-run switch.
The first probe's raw output is retained; its CPU wrapper incorrectly expected
OS-level `No such file` wording. The second, separately named receipt records
the actual model-directory boundary and nonzero exit; it is not a GPU retry.

## Reused entry points and exact schedule

`paired.py` now accepts `--purpose cache_ab`. It owns the same prepare/run/worker,
sealed authentication, existing supervisor/subreaper, UUID locks, observer,
resource limits, deadlines, stop-on-failure and drain/release checks. The new
`cache_ab.py` is an admission/report hook only; it has no GPU/server launcher.

The short screen has **two total fresh lifecycles**, OFF then ON, one per binary.
No V slots and no profiler. Each retains the existing Hello2 + three frozen
canary checks before QA8. QA8 is exactly the existing **excluded first + 2 warmups
= 3 unmeasured requests**, followed by **5 measured** requests. A warm request is
not relabelled as a measured request, and the first QA request is not claimed to
be the first request after server readiness. Both variants use identical recipes.
This is one fixed order, not a counterbalanced statistical experiment.

Both variants must match every inference flag, environment and runtime-library
pin except executable path: PP1 TP2 EP1, thread ranks, BF16 activation/KV/conv,
F32 SSM, resident FP8 shards, no CPU offload, ctx128/batch8/seq2/concurrency1,
workspace4096 and device22528. The measured environment is `FERRULE_LOG=info`,
with no one-sided debug, profiler injection or RUST_LOG override.

CPU plan from repository root:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/paired.py plan --purpose cache_ab
```

After the final arena release has its actual reviewed build/quality/source audit,
use the existing bootstrap smoke admission as `cache_ab.py admission` input.
That command requires `--candidate-admission` and a fresh `--output` inside an
existing paper admission directory. Then use the normal `paired.py prepare`
with `--purpose cache_ab`, a fresh ID and that derived admission. The existing
`paired.py run` still requires a fresh explicit sole-owner grant. None of these
candidate prepare/run steps has been performed here.

## Narrow historical binary exception

The only frozen exception is the exact existing 4d40 SHA and hash-pinned old
admission. The adapter verifies its existing build/test-quality/runtime evidence
and historical HTTP smoke, but never substitutes current hashes for historical
source pins or claims the current tree was tested. All nonsource evidence pins
are still checked. No arbitrary source-directory exclusion or new-binary
exception is provided. The old Nsight-only frozen policy is not widened.

Derived OFF metadata changes only workspace1024 to4096. The old 1024-MiB smoke
is moved to an explicitly historical field and is **not** accepted as a 4096-MiB
smoke certificate. A/B requires a distinct cache-on candidate and its normal
current-release numerical/build/source audit; frozen old evidence cannot satisfy
that gate. Both variants' actual build-source maps remain separately recorded.

The fresh A/B Hello/canary gates validate this short public path; this does not
claim the separate S1/S2 cancellation/reuse smoke. A/B output cannot be used as a
substitute for that smoke receipt. Normal full F/V promotion remains downstream
and is not automatically triggered by this screen.

## Capacity and supplementary evidence

ON must retain actual ready/shutdown cache snapshots and pass the existing
physical component/current/peak checks at <=3 GiB cache and <=22 GiB/card total.
Preserve capacity, fixed pins, active routed backing, reusable pool, pending
reserves and slack separately. Effective arena/routable capacity must come from
the final admitted layout/snapshot schema; do not invent it from a nominal cap
or call slack physically wasted memory. If final arena fields/schema change,
review their exact release binding before preparing the candidate.

OFF has no new cache snapshot API: unavailable remains null, not fabricated
zero counters. Both measured variants keep rankperf off, so profiled hit/miss
activity is null. Future separately qualified model/diagnostic hit/miss evidence
may support interpretation, but must bind the actual arena binary/source/scope,
not be borrowed from the previous dadf profile. It must not introduce one-sided
logging overhead into timed A/B or be combined with benchmark timings.

## Claims and next boundary

Matching workspace removes the prior 1-vs-4-GiB ceiling confound, but old/new
binaries still include public policy, snapshot and source differences. Therefore
this is a **matched-deployment descriptive A/B**, not a final isolated cache-only
causal effect. `fairmatch=false`, `exact_speedup=null` remain explicit. Fresh
processes do not prove cold filesystem/JIT/page cache or cache steady state.

The existing same-machine vLLM TP2 Marlin results remain descriptive historical
context only. Do not rerun the complete F/V matrix per arena iteration; after a
candidate wins this short screen, qualify and execute the full paired F/V stage
under separate authorization.

## Validation / receipts

- 13 cache A/B-specific CPU tests PASS.
- Full current paper suite: **228 tests PASS**, GPU hidden.
- Actual binary help/preflight, frozen-only source semantics, matched flags,
  current candidate qualification, OFF/ON routing, 3+5 population, existing
  supervisor reuse and stop-before-ON on OFF failure are covered.
- `plan.json`: non-executable plan, candidate null.
- `cacheoff4g-metadata.json`: derived baseline metadata only.
- `frozen-cacheoff-audit.json`: exact historical binary/source evidence scope.
- `cpu-tests.json` / `cpu-tests.log`: full CPU regression result.
- `workspace4096-preflight-02.json` and raw output: actual safe CPU boundary.
