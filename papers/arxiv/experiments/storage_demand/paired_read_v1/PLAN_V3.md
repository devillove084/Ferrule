# Prospective corrective paired study V3 — unbound, CPU only

**Independent new preregistration**, ID
`storage-middle64-paired-read-corrective-20261015-v3`. No GPU query, server, HTTP,
NAS query, build/help probe, new binary or candidate binding is authorized or
created by this version. **Zero V3 lifecycle attempts.** There is no V3 `run` or
`bind-candidate` command.

## Correct the study case, not production defaults or history

`cases-v3.json` contains the two complete prospective cases. Both append exactly:

```text
--decode-cohort-target 1 --decode-cohort-max-deferrals 0
```

to the original case argv. The sole **between-arm** difference remains
`--expert-read-schedule blocking` versus `lookahead`. All other argv, model,
topology, host/cache/device limits, request controls, frozen QA input/output IDs
and text, logging and deadlines are unchanged. The public CLI defaults stay
**target4 / max-deferrals3**; no production/default change is requested here.
The target1/deferrals0 policy is explicit in this study only.

`plan_v3.py` models the reviewed CLI argument lowering and compares **all nine**
`ResidentSchedulerConfig` fields, including exact types, missing and unexpected
fields, against qualified test config and both requested/effective factory
metadata:

| Field | Expected |
|---|---:|
| prefill_chunk_size | 8 |
| max_active_sequences | 2 |
| max_decode_batch | 2 |
| decode_cohort_target | 1 |
| decode_cohort_max_deferrals | 0 |
| max_batch_tokens | 8 |
| prefill_reserve_tokens | 1 |
| allow_mixed_batches | true |
| prefix_cache_capacity_pages | 0 |

This declared CPU configuration check is **not** proof that an arbitrary newly
compiled CLI lowers those flags identically. The future build reviewer must bind
actual help/source/factory evidence and verify all these fields for the new
artifact. Comparing an old quality report with the prospective case can establish
configuration equality only, never V3 source/GPU/CLI qualification. Even a
V3-shaped input with matching protocol/case hashes gets **no candidate PASS** from
the namespace guard.

Both arms retain **64 MiB host** including **8 MiB metadata + 2 MiB requested
stack**; **64 CUDA cache entries, 320 MiB including 8 MiB scratch; 7 GiB device**.
No 8k/context or PP/TP expansion. NVTX-capable build requirements are unchanged;
ordinary timing explicitly keeps NVTX OFF. No profiling capture is planned in V3.

## Independent design and prior failure denominators

Five new paired blocks, ten planned fresh lifecycles, 120 scheduled requests,
50 measured QA, same request phases and exact generation8 oracles. Seed
`20261015` and order **B/L, B/L, L/B, L/B, B/L** are explicitly reused from the
original declaration, not selected/adapted from results. No measured QA
E2E/TTFT/goodput observations were obtained in the stopped study; its observed
startup/custody timings are not used for allocation or estimator selection.
The original paired median/log-ratio estimands, exploratory n=5 interval method,
failure/no-retry rules and claim limits are preserved.

Absolute prospective envelope remains **7200 s**, **650 s per lifecycle**,
**100 s final reserve**, with a fresh registration only after future admission is
reviewed and separately authorized. No registration/deadline starts in this CPU
plan.

`artifacts-current-20261015-01/` stays permanently stopped and byte-for-byte
unchanged: one attempted lifecycle, zero completed, all 120 requests unissued,
all 50 measured-QA slots retained and no estimates. V3 is **not a resume,
replacement of failed pairs, or repair of that study's denominator**. The
original `protocol.json`, `cases.json`, order/source freeze, consumed V2 execution
seal, raw failure, closeout and manifests are untouched. `protocol-v3.json` pins
all 68 preexisting files, including the stopped run, to enforce preservation.
Baseline `newversionedprotocol` is also unchanged.

V2 integrity and V3 compatibility are separate. `v2-configuration-rejection-v3.json`
is a new append-only CPU finding: old V2 receipts remain valid historical evidence,
but old implicit CLI cohort4/3 does not match the qualified1/0 configuration and
cannot silently pass V3. No old acceptance/result is rewritten or reclassified.

## Future candidate prerequisites — all remain pending

1. Main's separate production agent fixes the TP1 prepared-read / TP2 cache-accessor
   clash. Explicit scheduler flags do **not** fix that root cause.
2. A new immutable CLI is compiled from that corrected source cut, with a truthful
   versioned build/archive/compiler/dependency/native/runtime reviewer. The old
   CLI `de515995…` and old adapter declarations do not qualify it.
3. Actual same-cut constructor and **both modes** full40, retained fresh, ordinary
   no-observer equality and cold-owner GPU gates PASS with source/reference,
   actual fingerprints, budget, identity and authenticated custody bindings.
   Old033519c PASS reports are configuration/oracle references only.
4. New V3 protocol/case-bound candidate and separately reviewed execution receipts
   must validate actual CLI help/lowering for every scheduler field and unchanged
   budgets. The future first planned arm must then prove actual Ready/runtime maps,
   NVTX OFF, worker-enqueue plus HTTP text/IDs/usage and final owner/resource ledger.
5. Renewed sole-GPU-owner authorization and maintained local UUID locks, all-eight
   idle, capture/observer/subreaper and finite deadlines. No warm probe, retry,
   foreign signal, cache flush, model write or old-study continuation.

Candidate binary, source cut, quality reports, CLI Ready and bindings are **null**
in this preregistration. No current source digest or old PASS is presented as the
future corrected artifact.

## CPU-only versioned commands

From repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/plan_v3.py plan
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/plan_v3.py check
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1 -p 'test_plan_v3.py' -v
```

The exclusive creation commands, already used for the new outputs, are:
`plan_v3.py seal-cpu --output papers/arxiv/experiments/storage_demand/paired_read_v1/cpu-sealed-v3.json`
and `plan_v3.py audit-v2-config --output papers/arxiv/experiments/storage_demand/paired_read_v1/v2-configuration-rejection-v3.json`.
Do not repeat them over existing receipts. `compare-report --arm blocking|lookahead
--report LOCAL_REPORT` is read-only semantic comparison, **not a GPU producer or
candidate acceptance**. There are no GPU execution commands in this version.

## Safe main-agent stage paths

Stage these new text/source/JSON files only, not the whole prior actual manifest
or any target artifacts/binaries:

- `plan_v3.py`
- `protocol-v3.json`
- `cases-v3.json`
- `PLAN_V3.md`
- `test_plan_v3.py`
- `cpu-sealed-v3.json`
- `v2-configuration-rejection-v3.json`
- `cpu-validation-v3.json`

All paths are under `papers/arxiv/experiments/storage_demand/paired_read_v1/`.
No protocol temp/source clone/dependency copy, weights archive, secret ENV dump,
GPU experiment, commit or push is added. Main separately coordinates the failure
stage and this prospective corrective stage.
