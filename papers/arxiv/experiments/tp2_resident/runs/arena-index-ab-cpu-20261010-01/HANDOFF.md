# Arena index A/B: CPU implementation handoff

Status: branch implemented; AFTER release SHA, reviewed release adapter, fresh numerical qualification, same-binary S1/S2 and builder source delta are pending. No executable AFTER prepared artifact is fabricated. No GPU query/run occurred.

## Existing CLI/API

Use `/home/admin/wxw403883/anaconda3/bin/python3.12 -S`; working directory is the repository root. Paths/hashes in angle brackets below are required future builder inputs, not generated receipts.

```sh
# Available now, CPU-only
/home/admin/wxw403883/anaconda3/bin/python3.12 -S papers/arxiv/experiments/tp2_resident/paired.py plan --purpose arena_index_ab

# Once reviewed AFTER release + fresh 23 gates + actual SAME-binary S1/S2 are available:
/home/admin/wxw403883/anaconda3/bin/python3.12 -S papers/arxiv/experiments/tp2_resident/arena_index_ab.py admission --cycle 1 --after-admission <promoted-AFTER-admission.json> --after-admission-sha256 <final-hash> --source-delta <builder-production-delta.json> --source-delta-sha256 <delta-hash> --output <fresh-cycle1-admission-path-under-paper-admission>
# Repeat admission with --cycle 2 and a distinct output; use identical AFTER/delta inputs.

/home/admin/wxw403883/anaconda3/bin/python3.12 -S papers/arxiv/experiments/tp2_resident/paired.py prepare --id arena-index-c1-20261010-01 --purpose arena_index_ab --admission <cycle1-admission.json>
/home/admin/wxw403883/anaconda3/bin/python3.12 -S papers/arxiv/experiments/tp2_resident/paired.py prepare --id arena-index-c2-20261010-01 --purpose arena_index_ab --admission <cycle2-admission.json>
```

APIs: `arena_index_ab.admission(output, after_path, after_sha, delta_path, delta_sha, cycle)` and `paired.prepare(identity, admission_path, purpose="arena_index_ab")`. Actual future execution remains `paired.py run --id ... --purpose arena_index_ab` and requires a renewed sole-owner grant/note; no new launcher.

Cycle 1 BEFORE→AFTER; cycle 2 AFTER→BEFORE. Each cycle has two fresh owners, same selected roster and bounded authenticated supervisor/UUID locks/pre-post idle/ECHILD/drain. No vLLM, profiler or rankperf. QA8 concurrency1, excluded first +2 warmups (3 unmeasured), then 5 measured, actual input5/output8. Both variants have enabled 3-GiB arena, workspace4096 MiB, device22528 MiB/card, TP2 ctx128 batch8 seq2 resident/no offload.

## Qualification/source gate

BEFORE is exactly ae3e with its existing same-binary smoke and ON A/B receipts. Original reference-case metadata/source pins are retained, not rebound to the current tree. `paired.py` executed bytes were archived before editing. The two qualification source files concurrently changed by the model agent retain their original closed historical hashes, without claiming current byte equality; binaries/tool/runtime/raw receipts still hash. See baseline-audit.json.

AFTER must pass the reviewed current-source release adapter, 23 named fresh selectors (37 calls/40293 exact, no reused baseline launch IDs), actual same-binary S1/S2, and explicit admission hash. Unknown/new builder schemas fail closed until their adapter is reviewed; this branch is NOT a new build-receipt bypass.

The dedicated builder delta JSON must exactly match the v1 schema in builder-contract.json and be pinned in AFTER build evidence. Compare full `pre_serving.release_sources(builder source_pins_before)` inventories, including native CPU .cc inputs, with no added/deleted production paths. The only two changed paths must be standard/expert_cache.rs and standard/mixed_decoded.rs. Tests may differ separately; they do not enlarge the production allowlist. Before/after build pins must be stable. Build environment/profile and cargo/rustc/nvcc/cxx/CUTLASS revision must agree, as must runtime pins and serving flags.

## Observation and interpretation

Observer is deliberately unchanged to avoid timing/provenance drift. Model CPU affinity and thread count are explicit null/uncollected, not orchestration-process measurements masquerading as model data. No CPU oversubscription, thermal/clock or absence-of-foreign-interference claim.

Snapshot v2 is checked for both arenas. Whole-lifecycle reuse/eviction counters do not locate measured hot activity; null activity is not zero decode. Sampling counts include only kind=sample, excluding monitor_terminal. Original throughput retains the runner-observed fixed-work window including terminal checkpoint/executor exit.

Even with an exact two-file source delta, two crossed cycles are descriptive, not statistical significance or automatic attribution of every difference to indexing. Compare only matching comparison_identity values; retain each cycle/order/startup/resource record.

## Validation

15 new CPU tests + 13 unchanged cache_ab tests + 17 paired tests pass (45 total). Tests cover real frozen baseline closure, named fresh gates, same-binary smoke prerequisite, exact source delta and builder binding, scope/debug rejection, opposite cycle ordering, fail-stop, prepared/slot wiring, same supervisor/observer, null CPU metadata and terminal-row exclusion. Synthetic prepare/run tests use mocks and are not GPU receipts.

No production, observer, cache_ab.py, or historical result receipt was edited. AFTER remains pending; release-specific prepare/run requires the actual new reviewed receipt.
