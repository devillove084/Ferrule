# Operator retirement V3 — CPU adapter handoff

**CPU preparation only. No GPU query, GPU lock, CUDA execution, serving launch,
release build or profile collection. No execution grant.** The AFTER builder
belongs to another agent. AFTER binary/hash and its reviewed release schema are
still pending; do not infer admission from the completed numerical handoff.

## Implemented scope

- New independent purpose: `paired.py --purpose operator_retirement_ab`.
  Admission, prepare/run/worker binding, existing finite authenticated supervisor,
  request gates, resources, drain and reporting are reused. No second harness.
- `operator_retirement_ab.py` implements fixed BEFORE evidence, exact source
  relation, registered-review validation, crossed order and descriptive reports.
  `REVIEWED_BUILD = None` intentionally rejects prepare/admission before any run
  directory or launch is created. CLI paths/hashes cannot register a new policy.
- `cache_ab.py`, `arena_index_ab.py`, their old manifests, historical results,
  qualification gates, release auditors and profiler policies are unchanged.
  The previous `paired.py` bytes were copied to the new history directory before
  wiring changes; old executed source evidence is not relabelled as new code.
- The new CPU tests exercise both real fixed evidence and mocked launch wiring.
  Mocked positive admission tests are not AFTER release qualification.

## Fixed identities and source accounting

BEFORE is the same independently frozen indexed-LRU release:

- Binary: `target/validation/tp2-mixed-arena-indexed-lru-release-20261010-01/ferrule-indexed-lru`
- SHA256: `9ef9841598b4694fb611250bf8876c4314d252400136ddb23f606a979cadb656`
- Promoted admission SHA256:
  `bb804493bac2d1f17a3200dd015bf453610ac99ae2997631c2c2a82ee3cd0939`
- The adapter checks its immutable build, quality, actual same-binary S1/S2 and
  crossed-cycle artifacts, without substituting current production source bytes.
  Two closed historical qualification-workflow identities are disclosed exactly,
  not implemented as a path wildcard or a source-drift allowlist.

Fresh allocator eligibility:

- `runs/decoded-allocator-v3-20261011-03/handoff.json`
- SHA256: `faa71361682e8b454b2e85feffd456c06598bcc4ff4572a34bfaa54367e83fd4`
- Schema: `ferrule.paper.allocator-v3-handoff.v1`
- 23 named fresh GPU gates, 37 full-model calls, 40293 exact metrics. The separate
  18 small GPU checks do not substitute for specific selectors. CPU preparation
  checks pinned document identities and the named inventory; full authenticated
  per-gate receipt/comparator audit remains a requirement of the release adapter.

Source differences are not an unrestricted backend exception:

1. Production behavior: exactly `crates/ferrule-backend/src/cuda/allocator.rs`
   and `crates/ferrule-backend/src/cuda/runtime.rs`, at the hashes in the fixed
   qualification source-relation receipt.
2. Test-only bytes in a production source file:
   `crates/ferrule-backend/src/cuda/providers/cutlass.rs` changes from
   `cf567b42135279ea1de45fb61a7ba935ab9be03497713067d4d0c7d2d393a3b5` to
   `389ee1596b3af387fcb4782bceed8ce54d6870de56a93f4979f8df97209dbce4`.
   Removing the one expected `NumericBf16M1Gemv` entry after `#[cfg(test)]`
   must reconstruct the entire old file hash. This is a manifest-contract array
   patch, **not** permission to change Cargo manifests or other providers.
3. Crate test-tree additions/changes/removals are separately enumerated, not
   hidden under the production category. Full raw inventory digests additionally
   bind all builder inventory entries.
4. The historical release inventory has 584 production/build-source entries,
   including four native `.cc` files omitted by the qualification inventory.
   That pre-existing qualification coverage gap stays disclosed. The AFTER
   builder must include them and compile-time documentation/dependencies; they
   must not be dropped from the comparison. An incomplete inventory cannot be
   justified by the numerical test-source map.

`source_delta(before_inventory, after_inventory, after_sha)` gives the exact
new delta schema. Input maps are authenticated full builder inventories, not
inventories synthesized from the two changed paths. Stable build-before/after
maps, current source checks and independently reviewed dependency coverage are
all required. Compare build environment, release profile, toolchain and runtime
libraries; unplanned differences block admission.

## Builder/main-agent rendezvous — still blocking

1. Supply actual AFTER release hash/handoff and stable full source inventories.
   Add its actual typed release auditor without relaxing any old auditor.
   The new auditor must authenticate qualification receipts, source transfer,
   CPU-hidden release build, runtime libraries and binary-not-test identity.
2. Review and register **one exact manifest path/SHA** in `REVIEWED_BUILD`.
   The fixed schema and required fields are in `plan.json`. Bind both binary
   hashes, the exact eligibility handoff, release handoff, recomputed source
   delta, rationale and safety invariants. Pin the review/delta in AFTER admission
   build evidence. Avoid circular hashing: the review can bind an already-final
   release handoff and be included by the auditor in admission evidence; do not
   retrofit its hash into that already-final handoff.
3. Only after a fresh explicit main-agent GPU grant: actual same-AFTER-binary
   S1 and S2 through the existing serving-smoke workflow. Then promote. No prior
   BEFORE smoke or test executable may satisfy AFTER smoke.
4. After promotion and authorization: two fresh crossed cycles,
   BEFORE→AFTER / AFTER→BEFORE, one owner at a time. Each slot uses Hello/canaries,
   then excluded first + 2 warmups + 5 measured QA8; actual input 5 / output 8,
   concurrency 1. Both stay arena 3 GiB, cap 22 GiB, workspace 4096 MiB,
   ctx128/batch8/seq2, TP2 thread, resident/no-offload, FERRULE_LOG=info,
   rankperf off and no profiler. Two cycles are descriptive, not significance
   or proof of isolated retirement causality.

## Subsequent official short profiles — plan only

Reuse official Nsight Systems 2025.3.1 from `target/vendor`, with the exact
CLI/importer/CUPTI pins in `plan.json` and `OFFICIAL-NVIDIA-TOOLS.md`. Do not
reinstall, probe GPUs, claim idle, or collect now.

- AFTER profile follows promotion and a separately authorized run directory.
  Reuse collector-first lifecycle: eight UUID locks, stable B0, selected 0/1 B1,
  authenticated owner/ECHILD/drain back to B0. No new framework or wrapper.
- Keep rankperf off. Request DEBUG only is enough for request/clock identity;
  cache-activity DEBUG is not required for retirement API counts. **Current
  cache profiling automatically adds performance DEBUG**. Before profiling,
  implement/review a separate typed rankperf-off policy; this plan does not
  silently change historical profiler policies or admit that logging override.
- Existing short workload: warm1 + measured1 QA8, actual input5/output8. Use the
  actual measured HTTP request window, not process lifetime, startup/shutdown,
  smoke windows or no-trace throughput window (which includes executor exit).
- Use `profile_trace_audit_v2` bidirectional full-export API/activity correlation
  by owner PID and correlation ID. Record create/record/query/destroy API counts
  for runtime and driver layers separately, plus waits/synchronization, names,
  owner/thread/phase, durations and return status. Disclose boundary-crossing
  calls and missing tables/coverage; missing evidence is unknown, not zero.
- Preserve raw report/SQLite, clock anchors, device identity, diagnostic/loss
  evidence. Exported correlation closure alone does not prove complete tracing.
- **No official 9ef profile exists.** A BEFORE/AFTER trace comparison requires
  a separate short profile of that exact 9ef artifact under matching tools and
  window/logging policy. The old 4d40 trace cannot establish a pure retirement
  effect. A frozen-9ef profile policy still needs explicit review; do not reuse
  the frozen-4d40 exception or pretend the current tree built 9ef.
- Retirement-ledger created-fence counts describe complete successfully created
  event sets, not total CUDA event calls. Partial-failure driver calls are not
  counted there. Runtime/driver counts are not additive unique-event counts;
  query/poll counts are not completed-retirement counts.
- Unmeasured `unknown_pressure` and `pending_peak` remain null. No production
  instrumentation, CPU stack build, profiler symbols search or new counters were
  added during this CPU phase.
