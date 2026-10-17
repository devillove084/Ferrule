# Public HTTP expert-transfer acceptance and failed pilot attempt 01

This is a bounded engineering curation, pending main review, not a pilot performance result. It separates four completed public-CLI acceptance cells, a permanently stopped prelaunch pilot, and one subsequently authorized recovery-only safety observation. No manuscript body is changed.

## Actual outcomes

| Acceptance cell | Requests passed | Measured QA | Measured output tokens |
|---|---:|---:|---:|
| Blocking / synchronous | 12/12 | 5/5 | 40 |
| Blocking / grouped-synchronous | 12/12 | 5/5 | 40 |
| Lookahead / synchronous | 12/12 | 5/5 | 40 |
| Lookahead / grouped-synchronous | 12/12 | 5/5 | 40 |

Acceptance totals: four actual fresh servers, 48/48 requests, 20 measured QA8 requests and 160 measured tokens; all 48 requests produced 360 tokens including excluded/canary/warmup work. Fresh server does not mean cold filesystem. The four exact output records and four payloads are deduplicated losslessly across the shared 12-slot workload and all four cells. Actual worker-enqueue token IDs, HTTP completion-ID joins, text, usage, finish reason and EOS-enabled payloads matched the reviewed references. IDs are not inferred by retokenization. Server-local request IDs restart in each server.

Pilot attempt 01 stopped with:

    ProtocolError: insufficient absolute reserve: supervisor and finite cleanup

The check at run_actual_v2.py:248 precedes planned.supervise; newversionedprotocol/run_pilot.py:119-120 enforces it. One launch.json journal is not a server start. Actual workers/servers/requests: zero. All 96 scheduled requests and 40 measured QA slots remain unissued. Eight arm metric records and four pair ratios remain null/unqualified. No traceback was emitted/preserved by the caught-error path. No retry, replacement, remaining next call, or finalize rerun occurred.

## Saved setup timing: measured boundaries, unknown internal costs

The outer budget was 650 s and required remaining reserve 610 s, leaving 40 s for setup. Admission to recorded STOP was 63.480144 s; remaining time was 586.519848 s, a 23.480152 s shortfall. Strict pre-idle reported 2.693733 s. Admission to the first pre-idle sample was 0.026427 s; the last pre-idle sample finish to recorded STOP was 60.760061 s.

Only admission, the three telemetry-query sample boundaries, permanent STOP, and coordinator-failure recording have explicit saved stamps. Lock acquisition, candidate validation, execution validation, baseline query, payload write, exact reserve-check instant and context release have no individual start/end stamps. Those durations are null, not inferred from file mtimes. The 60.760061 s interval includes validation and other setup/error bookkeeping; expensive revalidation is a plausible contributor, not an isolated measured cause.

## Recovery is separate, not retrospective pilot qualification

MAIN authorized exactly one recovery-only safety closeout after the zero-spawn failure. The recovery process authenticated its own PID/birth and checked historical owner absence and relevant owned jobs before actions. Using the same eight UUIDs and existing lock-file inodes, it acquired eight local locks, ran one strict three-sample all-eight idle window, exited the contexts with all FDs closed, performed one nonblocking all-eight reprobe, and exited those contexts with all FDs closed. Recovery passed in 2.754247 s. No foreign signal, enforced stop, model/server/worker, compute, rebuild, retry or spurious subreaper was used. GPU safety queries then stopped.

The old pilot still lacks its own post-idle/reprobe/supervisor receipts. Recovery does not manufacture old-worker ECHILD, physical resource release, or a late-healed successful arm. Both old-worker and recovery-helper authenticated ECHILD remain unobserved; no helper was needed. Recovery proves only the recorded local-lock and strict-idle observations at its later timestamps.

## Source, exact invocation and workload

Compiled production identity: d8c92d80dc2ec95c44ba7513198232d36f290a34. Archive/control identity: 37b2f40c2dfc02825a373e417b0d5bb428e1622e. The full declared Cargo/Rust scope is byte-equal between these committed trees; this is not the old EB3 executable. Exactly one fresh release build used CUDA 12.4.99, SM86 and features cuda,nvtx-profile; NVTX was runtime-disabled in every cell. CLI SHA256:

    c3db43b8c556a210a77f7fac29dc23c4bc9399d5e0374e6dc9fecd016625ccaa

source-pins.json records the immutable archive/descriptor/build/dependency/native-cubin/runtime/helper/external-input receipts, toolchain versions, actual build argv, and exact public CLI base argv plus the two losslessly reconstructed per-cell flag overrides. The flags were --expert-transfer-policy synchronous|grouped-synchronous and --expert-read-schedule blocking|lookahead. The model was /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8, backend cuda, engine resident, explicit profile qwen35-mixed-bf16 (MixedBF16/F32) with bounded reads.

Identical quotas: host read 64 MiB, metadata 8 MiB, worker stack 2 MiB; expert cache 64 entries/320 MiB, scratch 8 MiB; device budget 7 GiB, workspace 2 GiB; KV 16 MiB, context 128, sequences 2, batch/prefill 8, cohort target/max-deferrals 1/0. CUBLAS_WORKSPACE_CONFIG=:0:0 and runtime NVTX=0 were identical. No new GPU references or force-EOS changes were introduced; ignore_eos was omitted and normal EOS stopping enabled. All exact payloads, seeds and output references are represented in acceptance-results.json and bound to the historical HTTP reference document; eight oracle/tokenizer pins and four model-metadata pins are recorded in source-pins.json. Fourteen checkpoint shards were stat-identity-bound, not content-SHA256-certified.

The frozen CPU receipt reports 220 Python tests passing with zero failures/errors. Previous production Rust initial/policy tests were main-reported as 6/7 passing; they were not rerun or independently re-counted here. Curation validation is offline hash/schema/output/counter checking, not a new benchmark or build.

## Observed counters and limits

Every acceptance cell observed the INFO initial expert_read_initial_endpoint from the real owner snapshot, status configured, workers_created=0, before Ready and first invocation. Its all-zero initial counters are actual typed observations, not a synthetic missing-event fallback. First-request deltas and final observed counters are included; common values plus explicit per-cell overrides reconstruct the compact snapshots. Full snapshot/raw-line hashes bind the omitted timing accumulators and original events.

Each acceptance cell reported 40,943 cache uploads and 128,811,927,312 logical H2D bytes. Blocking had zero speculative/taken-prefetch reads; lookahead reported 36,863 of each. Final pending/current/unknown reads were settled, while the retained reader remained alive and logical idle host charge was 10 MiB (8 MiB metadata + 2 MiB stack), not zero. Positive uploads plus frozen-source dispatch binding support policy-path engineering evidence; no direct grouped-backend-call counter, route or logit observation is claimed.

Each acceptance cell bound actual CLI runtime maps at Ready, after requests and pre-stop to the same 12 pinned libraries. Loader resolution alone is not actual mapping evidence; pathname/inode checks do not hash live mapped memory or prove overlay backing identity. Planned stops retained actual child return code 0 and observer ACK-before-signal evidence; authenticated acceptance ECHILD, strict pre/post idle and lock reprobes passed, with no helper-forced descendant signals. These process/idle proofs do not certify universal physical CUDA release. Physical NAS/backend bytes and full resource-contract completeness remain unknown/unverified. Invocation-record owner identity is not invented.

TTFT, E2E, phase W, throughput and goodput are deliberately null in this quality-only curation. Raw acceptance client timings are not pooled into the failed pilot. No statistical significance, vLLM/SOTA, first-GPU-token latency, route/logit equivalence, or grouped-call-count claim is supported.

## Files, hashes and reproduction boundary

- acceptance-results.json: four actual acceptance cells, deduplicated exact output-ID/text/usage/finish records and payloads, observed counter snapshots/hashes, runtime-map and cleanup references.
- pilot-attempt-01.json: zero-spawn failure, preserved denominators/null metrics, saved stamped timeline and explicitly separate recovery evidence.
- source-pins.json: immutable source/build/runtime/input identities and exact historical invocation/test/review/check command vectors.
- manifest.json: one unique file-list entry for each of the other four files; the manifest intentionally does not hash itself. Its own SHA256 is supplied in the handoff.

Paths in artifact references are repository-relative unless an external model/toolchain path or literal executed argv requires an absolute path. Map/log references within acceptance cells are relative to raw_evidence_root. SHA256 encoding for derived JSON snapshots is documented in acceptance-results.json. Raw receipts, proc maps, SSE/logs and build inputs remain local owner-only evidence; no public raw archive or independent public replay without those inputs is promised. No weights, binaries, vendor tree, SQLite or large raw logs were copied here.

From the repository root, this CPU-only check validates the curated file set without running a model:

    CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -c 'import hashlib,json,pathlib; p=pathlib.Path("papers/arxiv/experiments/storage_demand/paired_read_v1/http-transfer-v1"); m=json.loads((p / "manifest.json").read_text()); assert all(hashlib.sha256((p / r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in m["files"]); print("PASS")'

The exact historical build/test/public invocation and maintained CPU review/check argv are data in source-pins.json. Historical GPU start/next/finalize commands are references only, not renewed authorization. Do not replay the stopped pilot directory or its admission failure. Reproduction requiring local private model inputs or new GPU work needs a separately reviewed stage and authorization; none is requested or created here.
