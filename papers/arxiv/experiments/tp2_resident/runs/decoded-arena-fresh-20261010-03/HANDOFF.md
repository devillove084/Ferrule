# Root-arena fresh numerical qualification — PASS

## Evidence and scope

Fresh run: `decoded-arena-fresh-20261010-03`. **37 calls / 40293 exact metrics,
zero failures** against the original frozen full40 and six-recipe oracles.
No threshold, capacity or numerical tolerance was widened.

- **15 fresh GPU gates:** all eight model selectors plus full40 and six recipes.
- **Three explicitly reused backend gates:** exact, preflight and typed-view.
  `backend-reuse.json` proves identical backend source/dependency/tool inventory,
  bit-identical backend test binaries and original authenticated receipts.
  Only the reviewed two model test-scope source inputs changed; no old model
  or dadf qualification was inherited.
- **29 fresh exact CPU selectors passed** under hidden GPUs, including runtime
  snapshots, composition admission/claim/binding and actual single-planner tests.
- **244 paper CPU tests passed**, saved in `cpu-paper-tests.*`.
- Builds use CUDA12.4/sm_86 with 600-second bounds. Existing Cargo outputs were
  reused only through successful no-run/build receipts and fresh staged hashes.

## Independent comparison inventory

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

Full40 hello/capital each require all 2178 fields. Lifecycle calls each require
all 1089 fields. Every recipe was compared successfully before the next launch.

## Arena, fit and Unknown checks

- All **32 actual ready/shutdown snapshots** across the owner epochs report an
  installed root, current reservation and peak reservation of **3221225472 bytes
  (3 GiB) per owner**. This is cache-only host accounting, not a global CUDA
  memory measurement or a substitute for authenticated completion.
- Ready partition: **1340 MiB pins + 1732 MiB free routed slots**. Full40 shutdown
  has **1732 MiB routed active**, with pool/free/tail/pending zero. Root backing
  and slot padding are not additional charges. Legacy segment slack is zero.
- The scoped model test assertions cover root allocation plus all slice creation:
  **one allocation request, one huge allocation, exact admitted root grant**.
  GemmEx preparation is outside that measurement; the expected count was not
  relaxed to two.
- The real fit-working-set GPU selector passed hot compute-launch delta zero,
  zero eviction/fallback and unchanged allocation-request/granted/driver-byte
  counters after preparation. These are **source-pinned exact assertions**, not
  raw per-step telemetry or proof that the full40 working set fits without churn.
- Real class-full LRU/fairness, pre-submit native fallback, Unknown/panic, escaped
  alias retention and fixed-pin Unknown/panic-before-publication all passed.
- Rankperf stayed off; actual `profiled_activity` remains **null**, not fake zero.

Whole-device sampled peak across full-model gates: **21.1044921875 GiB/card** on
both selected UUIDs, below the unchanged **22 GiB/card** observer ceiling.
Sampling is not a continuous device-memory peak proof.

## Artifact pins

- Acceptance summary SHA256:
  `37fa9f13aae0a3436ebf0805522d39d1a95d92117f38440f79b104d8d522d431`
- Machine-readable handoff SHA256:
  `836b47773933fc8e24dd3dc77efe72f362210af31501a9b41d7a54760119b6ec`
- Model test binary SHA256:
  `ca7957146344c694aa41c2f3d697f7b789ff8b15bc19657507f5467c00066897`
- Composition test binary SHA256:
  `1a605fe1bb7705324d09f6a4ee5b8cb7419a981c31400b7ee0f3831b030e7acd`

`prepared.json` records every current source/tool/build/CPU selector pin.
`acceptance-summary.json` binds all authenticated launches, comparisons and
actual snapshots. `handoff.json` adds aggregate memory observations and final
closure. Independent binary copies remain under this paper run.

## Cleanup and next boundary

Every executed gate has ECHILD/drain and after-idle/release receipts. Final
owner locks were independently reacquired, all eight devices were verified
idle, then locks were released; see `final-idle.json` and `final-locks-*.json`.
No GPU reset, module reload, foreign signal or benchmark/profiler was run.

Prior failed rounds and probe receipts remain unchanged. The earlier cuInit
failure has **no established root cause**; a successful official probe does not
prove a driver transient. Original 4d40 OFF and dadf cache-on serving binaries
were rehashed unchanged.

**This is internal numerical qualification, not a serving release or performance
claim.** A release owner must build/pin the new serving binary and review its
current source/quality relationship. The historical dadf admission and the
hard-coded old quality adapter cannot be relabelled as this arena release.
The matched-workspace4096/device22528 OFF→ON A/B and any public smoke/promotion
remain separate downstream work. Official Nsight full-trace completeness is
unchanged and not claimed here.
