# Indexed arena status and next operator-batch checklist

**CPU-only documentation closeout. No GPU/proc/lock query, profiling, symbol search, build, new harness or workflow change.** Original GPU reports/receipts remain immutable. This is not an admission or execution grant.

## Current result: what is actually established

- [x] Historical `9ef` quality + actual same-binary S1/S2 + promotion and two crossed indexed/unindexed arena cycles passed.
- [x] Fixed-geometry index comparison controls the prior negative unindexed-arena regression: BEFORE 2.422126 / 2.485096 tok/s; AFTER 5.205591 / 5.247737 tok/s, descriptive ratios 2.149183 / 2.111683. This is a scoped recovery, not proof of total optimization completion.
- [ ] **No false parity:** these cycles did not run V. They do not demonstrate current F/V parity or that the remaining gap is closed. Historical V runs are not a substitute for a contemporaneous qualified comparison.
- [ ] **Fit-working-set zero eviction is not real prefill-union fit.** The small fit fixture only proves its bounded working-set case. Real multi-request/rank/prefill routed-key union, size-class occupancy and measured-phase churn were not collected. Whole-lifecycle 54,621 evictions per rank do not locate the responsible phase.
- [ ] **Startup has not improved:** AFTER 358.708880 vs BEFORE 355.329678 s in cycle 1; AFTER 356.955392 vs BEFORE 348.405517 s in cycle 2. Both observed differences are longer, without an n=2 significance/general regression claim.
- [ ] **Host PSS has no demonstrated improvement:** cycle 1 61.221485 vs 61.771711 GiB; cycle 2 61.090178 vs 61.084245 GiB. Mixed sampled deltas are not a memory optimization result.
- [x] Historical roots/physical peaks held 3 GiB per rank; GPU sampled peaks held 21.041992 GiB/card under 22 GiB.
- [ ] Rankperf off/activity null does not prove zero hot decode, DMA or synchronization. CPU affinity/thread counts and GPU temperature/clocks/power were not sampled.
- [ ] NVML host/local PID attribution remains unavailable: authenticated local drain is proven in historical receipts, absence of foreign interference is not. No current GPU idle claim is made this turn.

## Next operator-batch optimization: plan only

BEFORE is the independently frozen, serving-qualified **9ef indexed binary**. AFTER is pending. Backend bounded batch-retirement is a different optimization class; **do not insert it into `arena_index_ab` or enlarge that exact-two-file exception.** The 4d40 cache-off protocol and ae3e/indexed protocol remain unchanged.

Proposed quality class: `operator_batch_retirement_v1` (draft identifier, not an implemented/accepted schema). Retain the existing numerical oracle, 37-call boundary/lifecycle recipes and physical limits; publish the actual new required named CPU/GPU retirement selectors before qualification. Requirements must cover bounded batching/queue high-water, compulsory flush/drain, completion ordering and delayed completion, borrowed/live root leases, cancellation/errors/Unknown quarantine, no premature reuse/refund/double retirement and unchanged exact outputs. The old 23-gate PASS cannot automatically certify a changed backend lifetime mechanism. Exact test names and receipt schemas remain pending the implementation handoff.

### Specific source-bound builder manifest

- Pin BEFORE/AFTER binaries, source snapshots/inventories, actual builds, tools/libraries and quality identities. Keep the closed 9ef baseline evidence; do not compare its historical source identity to a mutated current tree.
- Enumerate the complete production/dependency tree independently, including native .cc and build/config inputs; compare stable build-before/after inventories.
- A **registered optimization class plus one specifically reviewed builder manifest SHA** defines exact modified/added/removed paths and before/after hashes, rationale and safety-invariant mapping. Parameterization is data bound to this reviewed build, not a free user `--skip`/`--allow-any-source` option. Every unexpected delta rejects.
- Pin that manifest in AFTER build evidence; recompute actual delta from full maps. Report test-source changes separately; tests are not permission to widen production scope. An intentional ABI/owner/interface change must be explicitly reviewed, not treated as an arbitrary backend wildcard.
- Preserve matching runtime/build/tool settings and geometry; reject or separately review unplanned flag/library/source changes rather than attributing them to batching.

### Admission and execution requirements, not yet implemented

1. Finish/freeze the scoped operator implementation and specific manifest; coordinate minimal paper changes **before** the new GPU workflow/source pins are sealed.
2. Authenticate fresh source-bound numerical and retirement quality, then actual AFTER release.
3. Actual SAME-AFTER-binary S1/S2 must PASS before promote; old 9ef smoke does not transfer.
4. Later add a narrow registered typed optimization variant to the existing `paired` admission/prepare/run/worker/report hooks. Reuse request gates, selected roster/UUID locks, finite authenticated supervisor, ECHILD/drain and pre/post idle. **No second harness.**
5. If separately authorized, use two independent crossed-order cycles on the same machine: BEFORE→AFTER / AFTER→BEFORE; concurrency1, 3 unmeasured +5 measured QA8, actual input5/output8. Both remain TP2/ctx128/batch8/seq2/resident/no-offload, arena3GiB/workspace4096MiB/device22528MiB. Rankperf off/activity null in no-trace benchmark. No extra models or paper experiment expansion.

## Next official Nsight checklist

- [ ] Reuse the pinned official NVIDIA 2025.3.1 CLI/importer/CUPTI and existing collector-first lifecycle. Preserve historical traces and warnings; their acceptance status is not upgraded. No capture/retry is authorized by this note.
- [ ] Collect **CUDA API event counts in the same actual request window** as kernel/copy/sync evidence: runtime `cudaEvent*`, driver `cuEvent*`, stream-wait-event and synchronization families, with owner/thread/request/phase correlation, API durations and return status where available. Separate runtime/driver layers; do not double count them as unique events or equate polling API calls with completed retirements.
- [ ] Establish explicit warmup/measured/startup/shutdown windows and clock-domain/correlation boundaries. Counts from a process-lifetime or different request window cannot explain the measured request. Keep the distinction from the no-trace throughput run window, which includes terminal checkpoint/executor exit.
- [ ] Preserve raw `.nsys-rep`/SQLite, exact tool/library pins, selected-UUID identity, bidirectional API/activity correlation and loss/coverage diagnostics. Missing records stay unknown, not zero. Historical partial/full-trace rejection remains unchanged.
- [ ] **CPU stacks require a separate unstripped optimized profile build.** Pin binary SHA/build ID, source snapshot, compiler/profile flags, debug symbols/sidecar and unwind metadata. Review any frame-pointer/debug-option difference and its relation to serving 9ef; the profile build requires its own reviewed admission/smoke, not substitution under the 9ef hash. Profile overhead/timings are diagnostic, not a new benchmark rate.
- [ ] Do not repeat symbol searches against stripped 9ef/ae3e or identical stripped Cargo outputs. Their build-input receipts already say `strip=symbols`; the earlier CPU preflight documented the missing-symbol blocker. This turn performs no symbol scan.
- [ ] Review supported CPU sampling/call-stack options and permissions before any capture; do not guess flags, broaden profiler device scope, or infer that sampling is enabled by an admission file. CPU affinity/thread-count capture is currently absent and needs a separate minimal-overhead review.

## Preserved actual paths

- 9ef serving/profile admission: `admission/indexed-lru-20261010-01/staged/ferrule`, `admission/indexed-lru-20261010-01/profile-admission.json`.
- ae3e serving/profile admission: `admission/arena-public-20261010-01/staged/ferrule`, `admission/arena-public-20261010-01/profile-admission.json`. These admission files are not new profiling results.
- Original index closeout: `runs/arena-index-crossed-closeout-20261010-01/{RESULTS.md,owner-closeout.json,hashes.json}`.
- Historical official model profiles remain `runs/tp2-official-full-profile-20261010-001` (4d40) and `runs/tp2-decoded-cache-official-profile-20261010-001` (dadf); they must not be relabelled as ae3e/9ef traces.
- CLI: `target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys`.
- Importer/CUPTI and all linked receipt hashes are recorded in `status-and-next-plan.json`; see `OFFICIAL-NVIDIA-TOOLS.md` for the preserved official installation/capture policy.

All run-relative paths above are under `papers/arxiv/experiments/tp2_resident`; vendor paths are repository-relative. No new executable plan, candidate SHA or qualification is invented.
