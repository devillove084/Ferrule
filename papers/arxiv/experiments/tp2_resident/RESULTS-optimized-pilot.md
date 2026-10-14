# Optimized release: actual smoke/pair and collector trace evidence

Executed release SHA `4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173`, compared with preserved release `e017a3a8c8ec4019a7f291a818643d9693e32b2dfee0a9cc503fcf84926c1b0b`. Production was not changed. Paper supervision was unchanged during smoke/pair; later profiling used the explicitly documented CPU-isolation, bounded B1-cleanup and software-trace fixes.

## Completed scope

- Same-release S1/S2 HTTP smoke PASS: Hello2, original three max8 canaries, cross-request reuse, cancellation/post-cancel reuse, normal shutdown and fresh CPU owner.
- Fresh no-trace F1,V1 / V2,F2 PASS: 48/48 normal HTTP requests; 20/20 measured QA8 requests, 160 actual measured output tokens. Each server used excluded first QA8 +2 warmups +5 measured.
- Same FP8 model source; BF16 activation/KV/conv, F32 SSM; resident/no CPU offload; context128/batch8/seq2; selected physical GPU0/1; unchanged engine-specific argv/environment, seed1, raw natural EOS and frozen HF manifest. No HF regeneration.
- Existing bounded supervisor: startup900/request120/all-requests440/teardown120/supervisor1600/overall1700 per lifecycle. Source/runtime/admission pins revalidated after execution.
- CPU collective vs NCCL, Mixed/HF-eager reference vs Marlin, and Ferrule effective decode batch1 remain explicit differences. `fairmatch=false`, `exact_speedup=null`, E2 incomplete.

## Observed old/new values (not a causal or statistical result)

| Slot | Startup old → new (s) | TTFT old → new (ms) | E2E old → new (ms) | tok/s old → new |
|---|---:|---:|---:|---:|
| F1 | 385.979 → 243.480 | 516.128 → 512.099 | 1441.267 → 1435.682 | 5.5213 → 5.5483 |
| V1 | 63.794 → 57.582 | 134.081 → 131.576 | 709.799 → 703.128 | 10.3966 → 10.4834 |
| V2 | 60.095 → 58.300 | 130.161 → 129.583 | 695.507 → 688.657 | 10.5621 → 10.6852 |
| F2 | 367.234 → 241.209 | 511.283 → 511.890 | 1437.886 → 1433.096 | 5.5414 → 5.5586 |

Ferrule F1 startup changed by **−142.499 s (−36.92%)**, F2 by **−126.024 s (−34.32%)**. F1/F2 tok/s changed only **+0.49% / +0.31%**; TTFT changed **−4.029 / +0.607 ms**. Unchanged vLLM also moved +0.83%/+1.17% in throughput. Five same-server requests per slot, uncontrolled caches and no trace do not support attributing the small request differences to any one optimization. Startup improvement is observed; its decomposition is not measured.

## GPU/PSS and after-first smaps

Ferrule sampled GPU peak stayed **18.042 GiB/card**; vLLM stayed **19.284 GiB/card**. Ferrule lifetime tree PSS peaks changed from old **49.237/49.237 GiB** to **49.246/49.577 GiB**. There is no demonstrated lifetime PSS reduction.

New fields below are GiB, independent per-component maxima. All after-first rows are observations, not a steady-state proof:

| Slot | Startup PSS | After-first PSS | After-first Anonymous | After-first Private_Dirty | Complete after-first samples |
|---|---:|---:|---:|---:|---:|
| F1 | 49.246 | 49.227 | 49.000 | 49.040 | 6 |
| V1 | 7.862 | 6.682 | 5.758 | 5.818 | 4 |
| V2 | 7.559 | 6.658 | 5.729 | 5.789 | 4 |
| F2 | 49.577 | 49.235 | 49.008 | 49.048 | 6 |

`Pss_Anon` and `Pss_File` are unavailable in observed rollups and remain **null**, not zero. Historical observer had no Anonymous/Private_Dirty component fields or sample-end phase timestamps; no valid old/new after-first component delta is asserted. Anonymous is resident anonymous memory, not proportional PSS. GPU/PSS ceilings are sampled observational limits, not hard partitions.

## Historical first collector probe: import-time isolation failure

Run `tp2-optimized-collector-profile-20261009-001`, prepared SHA `ebb24a2624a1000a47373079b8635689993cbf704330e9aa42c33af71a053b3c`, launch ID `ec99752b7b6f4219b22b651a3e883af8`. The run-bound eight-UUID grant, successful eight-lock acquisition and true all-eight B0 are recorded. Model visibility was declared only for GPU0/1.

At **5.338 s** from collector launch, target failed with `ValueError: Target must retain exactly selected model visibility`; outer observer then reported collector root exited. The direct cause is reproducible without CUDA: `instrumentation_capture → instrumentation_live → pair_contract → experiment_current → experiment_v4 → runner`, whose import-time `runner.py:15` replaces `CUDA_VISIBLE_DEVICES` with empty and `NVIDIA_VISIBLE_DEVICES` with `none`. The fresh interpreter reproduction retained in the failure analysis shows selected UUIDs before import and those hidden values afterwards.

**No B1, model-release gate, model process, HTTP requests or `.nsys-rep` was produced.** This is not the earlier nonselected-memory-delta failure and is not evidence of a production CUDA/numerical or Nsight allocation defect. No retry, tolerance change or visibility fallback was performed. CUDA sync counts/durations, decode/GEMM/pack kernel time, copy bytes and GPU gaps remain unavailable—not zero; three quantified bottleneck candidates cannot be supplied from this attempt.

Minimal future paper fix: prevent CPU orchestration imports from overwriting the gated target’s authenticated inherited model visibility (or retain/validate/restore only those two keys before handoff while preserving Nsight injection). Add a fresh-interpreter import regression test; the prior CPU mocks missed this import-time side effect. Do not change production or loosen the UUID gate. A new attempt requires fresh pins/run/grant, not resuming this failed run.

## Intermediate attempts retained

The import-time bug was fixed by keeping imports environment-neutral and moving
GPU hiding to explicit `contract.require_cpu_python()` CPU entry points; CPU
subprocesses retain the existing hidden allowlist. Exact selected UUID visibility
is required at the collector target. Fresh-interpreter tests reject broad,
ordinal, reordered and foreign visibility, not just empty visibility.

Run `tp2-optimized-collector-profile-20260910-002` then reached requests but
failed cleanup when selected utilization still read 8%/5% after model reap.
The bounded cleanup now waits for two zero-utilization exact B1 samples without
relaxing PID, missing data, nonselected memory/utilization or identity checks.
Run `tp2-optimized-collector-profile-20261009-003` completed lifecycle cleanup
but had no kernel table; run002 had many unknown kernels and importer failures.
Neither is relabeled as a clean trace. Old scripts and receipts remain retained.

## Fresh collector-first profile run004: lifecycle pass, CUDA trace rejected

Run `tp2-optimized-collector-profile-20261009-004` used the same optimized release
(`4d40c347…b3173`), the fresh CPU-tested paper pins, and the explicitly supported
Nsight trace `cuda-sw,nvtx,osrt`. The run-bound grant acquired all eight sorted
UUID locks for instrumentation custody; the model remained visible only on the
physical GPU0/1 UUIDs. Startup was **256.384 s** (outside the request window).
Hello2 and three frozen max8 canaries passed; both diagnostic QA8 requests produced
8 actual tokens: warmup TTFT/E2E **1.244692/2.724168 s**, measured
**1.569021/3.021099 s**. These are profiling observations, not benchmark data.

Collector/model custody passed: B1 was 4 MiB on each selected card and 1 MiB on
each nonselected card; the maximum observed nonselected delta relative to B1 was
**0 bytes** across 124 valid GPU samples. After model reap, two exact B1 samples
were recorded, then collector drain returned to original B0. Authenticated
subreaper drain passed, and an independent final check found all eight cards at
0 MiB/0% with no compute PID and all locks available/released. See
[`final-owner-release.json`](runs/tp2-optimized-collector-profile-20261009-004/final-owner-release.json).

The trace is **not complete enough for the requested bottleneck ranking**. The
SQLite report contains API/runtime and memcpy tables for both owner threads and
GPU0/1, but no `CUPTI_ACTIVITY_KIND_KERNEL` table, no NVTX events, and importer
diagnostics include `GetNumTpcs(): NumTpcs was not initialized`, **428**
`StringStorage ... Cannot find bucket` errors, and CUDA/NVTX/OSRT incomplete
warnings. The explicit software trace therefore did not establish kernel time,
GPU idle gaps, Marlin/GDN/weight-decode attribution, or a clean full request
trace. Partial evidence (not a causal attribution) is:

- `cuEventSynchronize`: **153 calls / 641,769 ns total**; `cuStreamSynchronize`:
  **475 / 410,326 ns**. Synchronization includes GPU work and these sums overlap.
- CUDA memcpy rows: device0 **21 / 1,081,704 bytes**, device1 **62 /
  2,254,916 bytes** across the capture; this is not a complete model/request
  transfer inventory. The report separates H2D/D2H/D2D in
  `cuda_gpu_mem_size_sum.txt` and `cuda_gpu_mem_time_sum.txt`.
- No kernel rows are available, so kernel duration and any Marlin/GDN/pack/GEMM
  ranking are **missing, not zero**. Startup log milestones are recorded only as
  observations: build-entry→source admission **64.750 s**, source admission→owner
  image ready **117.537/117.539 s**.

The observed blocker is in the installed Nsight event-processing path, not an
observed production numerical error; its underlying cause remains unresolved: the software trace still loses CUDA activity/kernel
metadata. The offline audit also verifies both requests inside the 6.429226301-second
report window (warmup offsets 0.276330188–3.000511342 s; measured
3.263626196–6.284737208 s), with monotonic/unix crosschecks. Imported API rows
name `dp-owner-0` (90,401 calls) and `dp-owner-1` (88,703); these demonstrate
partial owner coverage, not completeness. `TARGET_INFO_GPU` matches the actual
physical UUIDs, but the single `TARGET_INFO_CUDA_DEVICE` row contains a different
UUID, another reason not to claim a clean identity mapping. The full offline
audit, raw SQL queries, immutable report and exact
failure counts are in [`trace-audit.json`](runs/tp2-optimized-collector-profile-20261009-004/trace-audit.json)
and [`trace-queries.json`](runs/tp2-optimized-collector-profile-20261009-004/trace-queries.json).
No second profile, transport fallback, precision/workload change, foreign signal,
production change, or oracle change was made.

## Historical first-execution drain, release and preserved receipts

Final independent check found **all eight GPUs at 0 bytes, utilization0, no compute PID**. All eight UUID locks were reacquired and released; all seven supervised trees are proven drained and their recorded kernel identities absent. Six model servers completed (two smoke + four pair); the seventh root was a failed collector, not a model server. No foreign PID was signaled.

- [Optimized smoke](runs/tp2-optimized-smoke-20261009-001/serving-smoke.json)
- [Old/new descriptive raw projection](runs/tp2-optimized-fv-qa8-20261009-001/old-vs-optimized.json)
- [Observed results, deltas, runtime and artifact pins](runs/tp2-optimized-fv-qa8-20261009-001/optimized-results.json)
- [Profile failure and CPU-only root-cause reproduction](runs/tp2-optimized-collector-profile-20261009-001/profiling-failure-analysis.json)
- [Profile target raw log](runs/tp2-optimized-collector-profile-20261009-001/runs/P1/session/server.log)
- [Final eight-card idle/lock/drain receipt](runs/tp2-optimized-collector-profile-20261009-001/final-owner-release.json)

Old M7/G10, baseline/canary failures, old smoke/pair and earlier failed profile receipts remain preserved. A mistyped initial smoke ID was rejected before lock/server launch and is noted separately; it was not a failed GPU attempt.
