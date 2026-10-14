# Fresh arena qualification — stopped at CUDA initialization

Status: **BLOCKED, no GPU numerical PASS**. This run inherits no old dadf PASS.

## Completed

- CUDA12.4 / sm_86, GPUs hidden: five fresh test artifacts built and independently
  staged (backend lib, typed-view micro, model lib, runtime snapshot, composition).
- Build bound: 600 seconds per artifact. Longest observed build was model at
  238.85 seconds; all five completed without a timeout.
- All **29 exact CPU selectors passed**, including single-planner geometry,
  same-config identity, fit-set/Unknown custody, runtime v2 snapshots and
  composition header/claim/binding gates. Source inventory remained stable.
- Paper CPU regression: **243 tests passed**; see `cpu-paper-tests.*`.
- Existing 4d40 OFF and dadf historical cache-on binaries were rehashed and match
  their original pins; neither was changed or used as this candidate.

## GPU stop

The first exact backend test failed in `cuInit` with code 3,
`CUDA_ERROR_NOT_INITIALIZED`, before numerical execution. Its libtest exit was
101; the authenticated supervisor reported failure, not timeout. The observer
had no completed memory samples before exit, so resource values remain null,
not fabricated zeros. `STOP.json` prevents every later gate.

Single-device CUDA/NVIDIA visibility and observer roster agree. Read-only
inspection found libcuda, kernel and UVM version 575.57.08, matching device-node
majors, and no demonstrated loader mismatch. Older kernel Xid messages were
not attributed to this run. The initialization root cause is **not established**.
No driver reset, module reload, device-node modification or GPU retry occurred.

The after-idle receipt verifies all eight devices idle. The supervisor proved
ECHILD/drain, and both owner locks were released. See `failure-closeout.json`,
`backend-exact/supervisor.json`, `after.json` and `release.json`.

## Remaining execution

After CUDA initialization is repaired, use a fresh run ID (never resume/overwrite
this failed run). If every source/tool/workflow pin remains unchanged, the
existing `reseal` entry can stage independent copies and reuse only these CPU
build receipts; it inherits **zero GPU passes**. Source/tool drift requires a
fresh `prepare` with the existing 600-second build bound.

Required order remains backend exact/preflight/typed-view, eight actual model
GPU selectors (including fixed-pin Unknown/panic), then full40 and all six
recipes. Full40 requires 2178 exact fields per case; total numerical closure is
37 calls / 40293 exact metrics. No later gate ran here.

Composition already initializes the existing tracing subscriber. The runner
selects `FERRULE_LOG_FORMAT=json`, keeps rankperf off and requires actual v2
ready/shutdown snapshots for each owner/epoch. Their logical partition charges
the full 3 GiB root once; root/padding are not added twice. Small fit-test zero
launch/eviction/fallback/allocation deltas are source-pinned assertions, not raw
sampled counters. The frozen model source currently allocates via
`zero_bf16_buffer`; the separate backend `uninitialized_dedicated` API is not
misrepresented as that model allocation path.

No full40/six-recipe acceptance, new public release, smoke, A/B or performance
claim was produced. Current official Nsight completeness limitations are unchanged.
