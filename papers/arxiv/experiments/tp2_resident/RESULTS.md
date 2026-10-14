# TP2 resident results — current crossblocks and historical pilots

## Current routebatch crossblocks — descriptive only (2026-10-12)

[Immutable actual report](runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.md) · [Raw metrics/log hashes](runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.json) · [Sealed artifact/source/API hashes](runs/route-batch-vllm-crossblocks-20261012-01/sealed-identities.json) · [Historical postflight](runs/route-batch-vllm-crossblocks-20261012-01/postflight.json).

| Slot | Output token/s | Mean TTFT ms | Mean E2E ms | Startup s | Sampled GPU peak GiB/card | Host PSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 14.317 | 176.248 | 523.399 | 349.585 | 21.04 | 61.05 |
| V1 | 10.376 | 132.676 | 694.228 | 66.239 | 19.28 | 7.71 |
| V2 | 10.330 | 134.651 | 699.213 | 62.814 | 19.28 | 7.32 |
| F2 | 14.804 | 171.333 | 515.106 | 366.133 | 21.04 | 60.14 |

F1→V1 / V2→F2: 3 unmeasured QA8 requests +5 measured per fresh slot, each actual 5-in/8-out, 40 output tokens per measured window. F/V rate ratios **1.379803 / 1.433173** are descriptive only. F repeated faster fixed-work completion; V had lower TTFT, startup and host memory. GPU/PSS peaks are sampled, not exact or per-owner GPU attribution.

Same TP2 resident ctx128/batch8/seq2/concurrency1; F info/rankperf off, workspace4 GiB/arena3 GiB/device22 GiB; V original utilization0.8 unchanged, not identical allocation flags. Same pinned checkpoint/tokenizer/URL, original engine-specific API aliases retained. HF-eager GemmEx/routing and BF16/FP8 Marlin are different implementations: no arithmetic bitexact/parity claim; `fairmatch=false`, `exact_speedup=null`. No official profile/activity/kernel-count claim.

**E2 is not PASS:** only 2 independent blocks, not the preregistered ≥5; long-context/concurrency and statistical acceptance remain outstanding. No statistical/causal or general performance-superiority claim. Raw logs, identity/source hashes and earlier receipts are unchanged.

For user execution, use the actual release [`target/validation/route-batch-release-20261012-01/ferrule-route-batch`](../../../../target/validation/route-batch-release-20261012-01/ferrule-route-batch), SHA256 `e674eda31c0b1379e19afbd1446662278d98047c27b77c0f1233670e601c87f7`; scope and linked actual receipts above apply, not universal model/context support.

## Historical initial resident pilot — original evidence retained


## Outcome

**TP2 resident vLLM (no EP, no CPU offload) reached readiness and generated
Hello2 twice. No independent canary request completed or was issued. QA8 was
not run, because its quality gate did not pass.** These are readiness/resource
observations, not a quality-qualified performance baseline or Ferrule speedup.

The run IDs are `tp2-resident-20261008-001` and
`tp2-resident-20261008-002`. They are separate append-only attempts. The second
attempt changes the observer, not the vLLM flags, reference, resource ceilings,
or transport environment. Neither run auto-retries a failed stage.

## Hardware and declared budget

- GPU0: `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`
- GPU1: `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`
- Both live-verified RTX 3090, 24 GiB; TP2/PP1/DP1; expert parallel false.
- vLLM planning 19.2 GiB/card (`gpu_memory_utilization=0.8`).
- Observational ceiling 22 GiB/card, server-tree PSS ceiling 128 GiB.
- These are sampled ceilings, not physical hard caps. PSS excludes unmapped page
  cache and does not establish CUDA pinned-memory usage.

## Observed readiness and Hello2

| Run | Startup to ready | GPU0 peak | GPU1 peak | Tree PSS peak | Hello2 TTFT | Hello2 E2E | HTTP wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001 | 128.3967 s | 19.0518 GiB | 19.0518 GiB | 7.5003 GiB | 1.2450 s | 1.3594 s | 1.3597 s |
| 002 | 123.6694 s | 19.0518 GiB | 19.0518 GiB | 7.2642 GiB | 1.0492 s | 1.1292 s | 1.1295 s |

Both Hello2 requests used **1 input / 2 output tokens**, emitted `, I`, and
finished with `length`. These are first-request measurements (including possible
JIT effects), not the requested five-measurement warmed QA baseline. Each case
used a fresh server process. Startup excludes CPU pin validation before launch.

| Run/stage | Authenticated launch ID | Local server PID | Result |
|---|---|---:|---|
| 001/hello2 | `b60349304a4c4d17af34c7c028eb3dca` | 69856 | Ready + Hello2 pass |
| 001/canary8 | `fea4f714c4fa48939df7f98981fb7b29` | 76127 | Observer abort at 25.8901 s; no HTTP |
| 002/hello2 | `57deffedb25a41ee8b2d543c4f99fa67` | 137751 | Ready + Hello2 pass |
| 002/canary8 | `2186502ffbd8448493e2a63d3924db44` | 140656 | Nonselected-device guard abort at 25.4921 s; no HTTP |

Local PIDs are **not** claimed to map to host NVML PIDs. Physical-device memory
is whole-device observation; owned-process GPU attribution remains unknown.

## Why canaries were blocked

1. **001:** the original shared PSS helper raised `ProcessLookupError` (`ESRCH`)
   while scanning `/proc`. A CPU-only process-churn reproduction located this
   at `telemetry.py:93` (`/proc/<pid>/stat` read). The v2 observer retains the
   maintained helper but allows at most five ESRCH resamples with unchanged
   live server-root kernel identity. Other errors, missing live root, and
   missing memory evidence remain failures; no missing sample becomes zero.
2. **002:** the v2 all-device monitor observed **1 MiB on GPUs 2–5**, each at
   utilization 0, with an empty NVML compute-app list, during canary startup.
   Its stricter during-case nonselected-device guard stopped the owned server.
   The provenance is **unknown**: it is not proven foreign interference and is
   not silently classified as harmless driver initialization. The observation
   and failure are preserved in `canary8/samples.jsonl` and `case.json`.

The original reference manifest remains
`102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e`.
All three first-eight-token expected texts were verified before launch,
including the arithmetic prefix ` ?\n\n7 + 8 = ` (not forced to `15`).
No canary output is available; this is **untested**, not numerical failure or PASS.
The HF-derived reference's seed/EOS/tie settings differ from this raw greedy,
seed1, natural-EOS canary; even a future text match is only a limited canary.

## Effective runtime evidence

- vLLM 0.28.0+cu129 / Torch 2.13.0+cu129, allowlisted CUDA12.9 JIT environment;
  no inherited Ferrule CUDA12.4 library path.
- Logged `MarlinFP8ScaledMMLinearKernel` and `MARLIN Fp8 MoE`.
- Logged model loading memory **17.04 GiB**; no CPU-offload flags except explicit
  zero. Installed-source defaults select `NoopOffloader`. Its direct log is
  debug-only and **was not observed**; no allocator-level tensor-residency audit
  was performed. This is not a claim that every tensor's residency was inspected.
- GDN prefill: Triton/FLA. GDN decode: Triton fallback because
  `fused_gdn_decode_post_conv_mtp` is not built. Arithmetic equality is not claimed.
- NCCL 2.29.7, declared and logged `NCCL_P2P_DISABLE=1`; live topology P2P `CNS`.
  Both directions log local channels `via SHM/direct/direct`.
- The Socket network plugin initializes after no IB device is found, but that
  does **not** mean local inter-GPU payload used Socket: channel logs show SHM.
- vLLM itself disables custom all-reduce after the P2P capability check. No
  transport override, vendor modification, installation, or fallback rerun occurred.

## Cleanup, preservation, validation

All four attempted cases have authenticated subreaper `proven_drained`/ECHILD
receipts and all-eight-GPU idle observations after cleanup. A final independent
recheck reacquired both UUID locks, observed all eight GPUs at **0 MiB / 0%**
with **no compute apps**, then released both locks. No foreign PID was signaled.

- Existing M7/G10 files and both prepared runs' source/runtime/reference pins
  were rehashed successfully after execution.
- Original suite: **42 passed**. TP2/observer suite: **15 passed**.
- CPU tests include sorted two-UUID acquisition, occupied-second-lock rollback,
  duplicate rejection, no-offload/no-EP contracts, unchanged runtime flags,
  canary mismatch retention, bounded ESRCH handling, and live-root failure checks.
- `fairmatch=false`, `exact_speedup=null`; Ferrule TP2 Mixed was not executed.
- Checkpoint identity remains partial (metadata hashes + shard stats). Dynamic
  runtime/JIT/driver closure and exact instantaneous resource peaks remain unknown.

## Artifacts

- [`RESULTS.json`](RESULTS.json): machine-readable summary, full metrics, final artifact hashes.
- [`experiment.py`](experiment.py): frozen first observer revision.
- [`experiment_v2.py`](experiment_v2.py), [`sampling_v2.py`](sampling_v2.py): explicit observer fix.
- [`001 execution`](../runs/tp2-resident-20261008-001/execution.json)
- [`002 execution`](../runs/tp2-resident-20261008-002/execution.json)
- [`001 Hello server log`](../runs/tp2-resident-20261008-001/hello2/server.log)
- [`002 Hello server log`](../runs/tp2-resident-20261008-002/hello2/server.log)
- [`002 canary server log`](../runs/tp2-resident-20261008-002/canary8/server.log)
- [`002 nonselected-device samples`](../runs/tp2-resident-20261008-002/canary8/samples.jsonl)
- [`Final idle recheck`](final-owner-recheck.json), [`final release`](final-owner-release.json)

Worker-local log hashes are snapshots before the custodian's final descendant
cleanup; final log hashes in `RESULTS.json` include any late descendant output.
The next step is to identify the nonselected 1 MiB startup observation under
custody before authorizing another run—not relax the guard or alter expected text.
