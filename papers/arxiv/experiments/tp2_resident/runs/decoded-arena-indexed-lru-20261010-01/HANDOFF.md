# Indexed completed-group LRU — fresh GPU numerical qualification PASS

Run: `decoded-arena-indexed-lru-20261010-01`. **23 fresh GPU gates, 37 calls,
40293 exact metrics, zero failures.** No old arena/uninitialized release PASS
was inherited. No numerical tolerance, allocation boundary, quota or memory
ceiling was widened.

## Frozen production inputs

Only two production files changed relative to the uninitialized-arena candidate:
- `expert_cache.rs`: `cbb6b8cf01119ca649f3446ecf5a189fcd0e3d0886d449b1aa0a9989428be229`
- `mixed_decoded.rs`: `bfb5faa7f7098b067df7e2e463a281e43547d1c95314d2aa6289aa2c9f28859f`

`handoff.json` records the two accompanying test-source changes. The snapshot
API, arena quota planner and dedicated allocation-to-custody production path
remain hash-identical to the previous candidate. The old model API handoff is
historical context; its old core source hash is not substituted for these pins.

## CPU and build validation

- Correct tao CUDA12.4 environment, NVCC:
  `/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc`; explicit `sm_86`, GPUs hidden.
- `cargo check -p ferrule-model --features cuda --lib --tests`: PASS.
- `cargo clippy -p ferrule-model --features cuda --lib --tests -- -D warnings`: PASS.
- 600-second per-step build/check bounds; seven freshly staged executable pins.
- **38 exact CPU selectors passed**: prior 33 arena/metadata guards, four new
  grouped-cache equivalence/lease/credit tests, and one CPU benchmark test.
- The randomized grouped-cache test reran all **8 seeds × 5000 transitions**
  against the linear reference. Benchmark victim sequence, statistics and final
  ordered entries are asserted equal, not accepted by timing.
- The user-reported prior CPU517 suite is not relabelled as this run's execution.

### Bound CPU microbenchmark — not E2E

`cpu-indexed-lru-benchmark.json` binds the actual current model-test binary,
production/test source hashes, bounded supervisor receipt and two short raw
summary lines. No large victim-vector dump or GPU timing is substituted.

```text
CPU only indexed=false entries=1155 classes=2 descriptors=20000 misses=50000 wall=5.693966263s victim_work=Work { scanned_entries: 57750000, descriptor_lookups: 57750000, selected: 50000 } incoming_descriptor_lookups=50000 frozen_registration_lookups=0 (excludes preparation; includes lease/complete/index maintenance; not E2E)
CPU only indexed=true entries=1155 classes=2 descriptors=20000 misses=50000 wall=41.478824ms victim_work=Work { scanned_entries: 0, descriptor_lookups: 0, selected: 50000 } incoming_descriptor_lookups=50000 frozen_registration_lookups=50000 (excludes preparation; includes lease/complete/index maintenance; not E2E)
```

This single CPU sample uses 1155 entries, 20000 descriptors and 50000 misses.
Preparation is excluded; lease/completion/index maintenance is included.
There is no timing acceptance threshold, statistical speedup claim or E2E
performance conclusion.

## Fresh GPU qualification

Five backend micro/operator gates and all eleven current model gates passed,
including exact M1/M5 numerical checks, real class-LRU/fairness, hot-zero
decode/eviction/fallback, native fallback, alias retention, Unknown/panic,
view faults, first-fixed failure and uninitialized-slot rejection.

| Full-model gate | Calls | Exact metrics |
| --- | ---: | ---: |
| full40 | 4 | 4356 |
| greedy4 | 10 | 10890 |
| fork2 | 6 | 6534 |
| canary2 | 5 | 5445 |
| recreate1 | 4 | 4356 |
| cold-owner1-01 | 4 | 4356 |
| cold-owner1-02 | 4 | 4356 |
| **Total** | **37** | **40293** |

Full40 hello/capital each require 2178 fields; lifecycle calls require all 1089.
Each original-oracle comparison passed before the next gate was launched.

All **32 actual ready/shutdown snapshots** preserve the unchanged v2 partition:
root/current/peak/driver growth each **3 GiB/owner**, counted once. Pins are
1340 MiB and ready free slots 1732 MiB; no root/padding double charge or hidden
Unknown refund. Rankperf activity remains null, not fabricated zero. Whole-device
sampled peak was **21.1044921875 GiB/card** on both selected UUIDs, under the
unchanged 22 GiB ceiling. Sampling is not a continuous peak proof.

## Paper regression caveat — retained, not silently waived

Current gate/snapshot policy tests: **41 passed** (`cpu-policy-targeted.*`).
The full paper suite ran **271 tests, 2 failures + 1 error**, recorded in
`cpu-policy-full.*`. These are historical unindexed-release bindings:
- The old serving handoff audit correctly rejects the changed `expert_cache.rs` hash.
- Two tool/library drift subtests expect their mocked error, but now stop earlier
  at the changed required CPU gate-plan contract.

The old receipt was not rewritten and no source/tool/library check was weakened.
The builder/old-release test adaptation remains separate work. This caveat does
not relabel the full paper suite as green or the historical release as current.

## Pins and cleanup

Acceptance summary SHA256:
`396710eeb5366b6a2aa607168e4cb33de01aea098790cb0e94f0d87e03b65165`

Machine-readable handoff SHA256:
`514b0d8a7f5e05abf958c5e278d13f026d5bda9c4f4191e51e3e9a7e2fc56cf3`

Model test binary SHA256:
`b0a1f243076045b1503ec21f6f004383e6c58686e6a24c9ca85ec76df854a366`

Composition test binary SHA256:
`653c5bbb11a76db0b1ef8ee955554b3b20a34c8c53af8fe2f8eeba3d3d953515`

`prepared.json` and `handoff.json` bind all source/tool/build/runtime-library
and binary evidence. Every executed gate has authenticated ECHILD/drain and
after-idle/release receipts. Final owner locks were independently reacquired,
all eight cards verified idle, then locks released (`final-locks-*.json`).
Old runs and frozen 4d40/dadf serving binaries remain unchanged.

## CPU builder next step

Build a **new indexed-LRU serving release** against the exact qualified sources,
link this current 23-gate qualification and review every build/source difference.
Do not reuse the old unindexed arena or dadf admission as current qualification.
Keep the checkpoint/TP2 geometry, exact quotas, workspace4096/device22528 MiB,
3 GiB dedicated root, actual v2 snapshots and null unprofiled activity.

No serving release, public smoke/A/B, public performance run or new Nsight
acceptance was performed here. CPU microbenchmark timings are not E2E evidence.
