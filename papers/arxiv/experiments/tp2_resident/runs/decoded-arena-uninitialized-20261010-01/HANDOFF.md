# Dedicated uninitialized arena — fresh qualification PASS

## Release CPU builder entry

Use the hashed `handoff.json`, `acceptance-summary.json` and `prepared.json` in
`decoded-arena-uninitialized-20261010-01`. This is new source/binary qualification,
not an inherited result from the previous zero-initializing arena03 candidate.

**23 fresh GPU gates passed:** five backend gates (including both new allocation-only
microtests), all eleven current model gates, full40 and six lifecycle recipes.
No GPU PASS was reused: backend sources changed, so even the former three gates
were executed again against the newly staged binaries.

**37 calls / 40293 exact metrics / zero failures** against the original oracles.
No numerical tolerance, allocation count, quota, memory cap or observer threshold
was relaxed. Each full-model comparison passed before the next GPU gate started.

## Exact numerical inventory

| Gate | Calls | Exact metrics |
| --- | ---: | ---: |
| full40 | 4 | 4356 |
| greedy4 | 10 | 10890 |
| fork2 | 6 | 6534 |
| canary2 | 5 | 5445 |
| recreate1 | 4 | 4356 |
| cold-owner1-01 | 4 | 4356 |
| cold-owner1-02 | 4 | 4356 |
| **Total** | **37** | **40293** |

Full40 hello and capital each require 2178 exact fields. Each lifecycle call
requires all 1089 fields. Greedy identity, max-abs=0 and RMSE=0 are retained.

## Allocation, custody and initialization ordering

The qualified path is **allocate dedicated uninitialized root → immediately
install retained custody and views → full slot decode → readiness/consumer**.
There is no whole-arena zeroing. This describes buffer initialization, not a
claim that an already-created CUDA operator lacks a driver context.

- Two model CPU guards verify that successful allocation-to-install contains no
  allocator observation, zeroing or sync, and that readiness follows complete
  decode plus completion classification.
- Two backend CPU guards verify exact rounded grant/overflow/budget semantics and
  an allocation-only source call chain, including absence of uncounted memset.
- Backend GPU allocation/error microtests passed exact initialized-range decode,
  retained alias lifetime, and budget/capture/failpoint/shutdown failure behavior.
- Model test counters bracket only root allocation plus **all view installation**,
  after GemmEx preparation: exactly one allocation request, one huge allocation,
  one driver allocation, exact admitted grant and driver-byte growth, no reuse,
  and zero kernel/H2D/D2H/stream-sync increments. The offset/window was not widened.
- New view-fault tests cover first/partial view error, Unknown and panic, retaining
  the entire root and reservation before any projection publication.
- First-fixed ordinary failure stays unready and retained until existing known
  completion permits healthy rollback. Unknown/panic stays terminal and retained.
- Uninitialized single/pair slots reject consumers without launching; complete
  same-stream decode then permits ready/hot reuse. Existing fixed/routed exact,
  true class-full LRU/fairness, native fallback, alias and Unknown gates all passed.
- Hot fitting-set decode/eviction/fallback deltas remain zero; allocation counters
  do not grow after preparation at the decoded-cache boundary. These are exact
  **source-pinned GPU assertions**, not raw per-step telemetry, and do not claim
  that all model activation paths allocate nothing or that full40 never churns.

## Same configuration, physical quota and snapshots

The core custody and single quota planner source hashes match arena03 exactly;
`handoff.json` records these pins separately from the six changed/new inputs.
Checkpoint/oracle identity, TP2 owner geometry, descriptor inventory, physical KV
pages, row/session limits and margins remain bound to the same full configuration.
The cache binding retains 3 GiB capacity, 1340 MiB fixed pins, 768 MiB conservative
routed working set and 20480 routed descriptors per owner.

All **32 actual ready/shutdown snapshots** report, per owner:
- `arena_backing_grants == capacity_grants == physical_current_bytes ==`
  `physical_peak_bytes == driver_allocation_growth_bytes == 3221225472`.
- Logical pins/active/pool/free/tail/pending partition the root exactly once;
  padding and root backing are not additional charges, and legacy slack is zero.
- Ready pins are 1340 MiB, free routed slots 1732 MiB; full40 shutdown active is
  1732 MiB. Idle/no-quarantine/no-borrow/no-pending/unknown=0 and owner-generation
  continuity/recreate checks all passed.
- Rankperf remains disabled: `profiled_activity=null`, not synthetic zero.

These are cache-only host ledgers, not global GPU memory or a new completion
authority. Dedicated driver growth equals the full grant and is recorded as an
observation, **not another component charge**.

Whole-device sampled peak across the full-model gates was **21.1044921875 GiB/card**
on both selected UUIDs, below the unchanged **22 GiB/card** limit. Sampling is
not a continuous peak proof. The full cache grant remains **3 GiB/owner**.

## CPU/build and artifact pins

- CUDA12.4/sm_86, hidden GPUs during preparation, 600-second bounds per build.
- Seven fresh staged test executables, all required selectors listed exactly once.
- **33 exact CPU selectors passed**, including the four new source/custody guards.
- **246 paper CPU tests passed**, with raw stdout/stderr and receipt in this run.
- Full source/tool/build/environment pins are in `prepared.json`; the resolved
  composition loader closure under the sealed test environment is separately
  recorded in `runtime-library-pins.json` (not a claim of per-process loader tracing).

Acceptance summary SHA256:
`2068f9f200ec99a8e94054dbfd2e585b2abb3fca1264e140cad5bb8641b1434c`

Machine-readable handoff SHA256:
`d6b15ddc6590f5f3b445ac5f02564bad45345365401044247e53ec01ecd5783e`

| Test executable | SHA256 |
| --- | --- |
| backend | `a6f5fb2bade4f085fcfc9c91c2745a5adaa4393fddd9fb76e8e64b219e503b0b` |
| backend-slice | `f84751452f20b9ee83493a2afffadf6fd9921ac1cd7fa9e3b6dff26840df5b3c` |
| backend-arena | `7f9223912fd20ead8bce37bd0f73b92ca13e9e52d3d61e9f68bd529974841da1` |
| model | `705b9d16b81e16a016180ed6c479a5bd722c311f3df8547842ebbb05bfb5a050` |
| model-handoff | `c4ba0b5c8b04e85bbd8d20d3ff3178fb72ed9faf5d9cd3b8c891f800461ed426` |
| runtime | `da1c0afc077a6269e10d1d7aa120bb593efb93100330ee5262042ab0842d0948` |
| composition | `6d818ba4a6079283beafb180827477c585e19c09ace029345022073415ae54e7` |

The `/tmp` model-only binary named in the developer handoff is not substituted
for these full-gate build/staging receipts. Old arena03, failed rounds, probe
evidence, frozen 4d40 OFF and dadf cache-on serving binaries remain intact.

## Final custody and next boundary

All 23 executed gates have authenticated supervisor launch IDs, ECHILD/drain,
after-idle and owner release receipts. Final locks were independently reacquired,
all eight cards verified idle, then locks released. See `final-idle.json` and
`final-locks-*.json`. No reset, modprobe or foreign-process signal was used.

The release CPU builder must verify this handoff and all linked evidence, then
build/review a **fresh serving binary** against these exact qualified production
sources. Any source, dependency or configuration drift needs explicit review;
old dadf/arena03 PASS cannot be relabelled. The existing historical release adapter
still needs explicit current-schema/source-review integration, not weaker drift
checks. Keep workspace4096, device22528 MiB, full 3 GiB root once, actual v2
ready/shutdown fields and null benchmark activity.

**No serving release, public smoke, A/B, throughput result or profiler acceptance
was produced here.** This handoff is input to the CPU release builder, not GPU
launch authority or public-performance evidence. Official Nsight full-trace
completeness limitations remain unchanged.
