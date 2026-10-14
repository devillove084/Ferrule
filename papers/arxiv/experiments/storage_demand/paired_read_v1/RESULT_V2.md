# Actual paired study — permanent STOP on first planned arm

Actual run: `artifacts-current-20261015-01/`.
Canonical outcome: `artifacts-current-20261015-01/paired-final.json`.
Post-final custody/diagnostic closeout: `artifacts-current-20261015-01/closeout.json`.

**One planned server lifecycle was attempted (`P1-blocking`), none completed,
and no HTTP request was issued. No retry, warm probe, replacement arm or profiling
capture was launched.** The entire registered study remains permanently stopped.
All five paired blocks, ten lifecycle slots, 120 request slots and 50 measured-QA
slots remain in the report. Nine later lifecycles and all 120 requests are
unissued. E2E/TTFT/goodput, actual HTTP output token IDs, positive pipeline reads,
waste/host-credit/wait/IO-time statistics and paired estimates are unobserved/null,
not fabricated zeros. Missing `resource_shutdown_summary` makes the owner/read
resource ledger unknown despite known process drain and observed idle.

## Actual failure

The exact current CLI failed before Ready with:

```text
Error: invalid runtime request: experimental qwen35-mixed-bf16 serving: prepared source profile changed
```

The immutable source explains this deterministic **pre-model-worker** failure:

1. `crates/ferrule-cli/src/commands/serve.rs:38` unconditionally calls
   `prepared.mixed_tp2_cache_requirements()?`, before spawning the model worker.
2. `crates/ferrule-composition/src/model_factory/bf16_serving.rs:541-542,732-780`
   gives an enabled TP1 numeric read plan a `prepared_source_plan` payload.
3. `crates/ferrule-composition/src/model_factory/mixed_tensor_serving.rs:275-280`
   rejects any non-TP2 request with such a payload as `prepared source profile changed`.

Pinned source copies are under the qualification's `frozen-source/` and their
hashes are in `closeout.json`. The actual log contains no `cuInit` error and
never reaches the loading/listening messages. This failure is not evidence that
the previously flaky CUDA initialization root cause was resolved or encountered.
The nine distinct test-binary quality gates still PASS; they do not exercise this
CLI diagnostic accessor. Exact help/build/source/native/runtime review is not a
mathematical proof of CLI serving acceptance.

A separate latent configuration mismatch is visible in the actual model-plan
log: CLI defaults select `decode_cohort_target=4`,
`decode_cohort_max_deferrals=3`, while the qualification configuration freezes
`1/0`. `cases.json` does not explicitly pass those flags. A future newly reviewed
case would need `--decode-cohort-target 1 --decode-cohort-max-deferrals 0` in both
arms. That does **not** repair the pre-worker accessor failure above. Neither the
case nor production nor raw failed run was changed after observing the outcome.
Main owns any production fix/new immutable artifact and subsequent new
preregistration/authorization. **Do not resume or replace this study.**

## Actual custody and coverage

- Server/capture exit 1; worker exit 1; no timeout and no recorded cleanup signals.
- Maintained authenticated subreaper: containment verified, ECHILD,
  `descendant_custody=proven_drained`; recorded PID/birth owners absent.
- Independent final idleV2: three consecutive all-eight zero-memory,
  zero-utilization, no-compute-app samples.
- All eight local UUID locks were nonblockingly reacquired after STOP and released.
  `final-lock-reprobe.json`, `final-all-eight-idle.json`, `final-lock-release.json`
  preserve the actual evidence. These are not global reservations or CUDA release
  certificates.
- Supervisor wall time: 7.603352844715118 s. Registration through first final
  receipt: 42.69262756407261 s, within the absolute 7200 s / 650 s envelope.
- NVTX OFF is in the exact launch environment. Actual Ready/after-request mappings
  and `/proc` environment gate were never reached and remain unobserved.
- Passive NFS raw samples exist for prelaunch, shutdown-pre and shutdown-post.
  No Ready/startup-delta or request windows exist. Shutdown mountstats are a valid
  timestamp-bracketed **shared client delta**; observed zeros are retained as
  actual counter differences, never invented or clamped. Server-attributable
  physical NAS disk/network bytes remain unknown.
- Worker-enqueue/HTTP token joins are empty because no request was issued;
  no output retokenization or generated IDs were fabricated.

## Scope and handoff

New explicit review/binding: `actual-build-review-v2.json`,
`actual-binding-v2.json`, `execution-sealed-v2.json`, `current_binding_v2.py`.
Narrow per-arm maintained-worker coordinator: `run_actual_v2.py`.
Precheck: `admission-20261015-01/`.
Tests: `test_actual_v2.py`, `test_stopped_closeout_v2.py` plus prior CPU regressions.
All artifacts are within `paired_read_v1`; the original CPU seal, baseline,
qualification raw receipts, production, manuscript and history remain unchanged.
No commit or push was made.

There are no legitimate paired performance estimates from this stopped run. The
scientific status is rejected/unobserved, not an incomplete-case speedup. Stage
only the approved new paired receipts/code if needed; never sweep in unrelated
paper modifications, weights or arbitrary ENV secrets. Maintained authenticated
launch receipts contain local ephemeral launch nonces and belong in restricted
owner-local custody, not public upload.
