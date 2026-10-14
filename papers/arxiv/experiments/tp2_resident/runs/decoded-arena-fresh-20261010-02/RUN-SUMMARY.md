# Fresh arena round 02 — initialization recovered; model assertion blocked

## Initialization probe

`../decoded-arena-init-probe-20261010-02/` contains the authenticated, bounded
probe: driver `cuInit=0`, `cuDeviceGetCount=0`, one visible GPU, and the official
CUDA12.4 vectorAdd **Test PASSED**. The same failed-gate CUDA/NVIDIA UUID roster,
PCI ordering and LD_LIBRARY_PATH were used. Actual libcuda mappings, driver
hash, library pins, node major/minor and successful O_RDWR open results are
retained. No reset, modprobe, foreign signal or GPU retry loop was used.

Probe -01 stopped at a CPU environment assertion before loading CUDA because
`require_cpu_python` hides visibility. Its history is retained. After this
wrapper-only fix, probe -02 was the first actual driver/vectorAdd probe.
The earlier cuInit failure's cause remains **unproven**, not labelled a proven
driver transient.

## One fresh gate round, no rebuild

Round -02 independently stages the same five exact CPU-built binaries from -01
and repins the reviewed workflow; it inherits **zero GPU PASS**. Production and
Rust test source hashes are unchanged. No repeated CUDA builds were performed.

Fresh authenticated GPU results:

| Gate | Result |
| --- | --- |
| backend-exact | PASS |
| backend-preflight | PASS |
| backend-typed-view | PASS |
| model-cold-hot-fit | FAIL during preparation assertion |

Model binary SHA256:
`278c8a7c47ed3f40ddffec4b0f9113cfbe1316edf65a119ef93187bf85e3807f`.
The failed test reports `huge_allocations` delta **actual 2, expected 1** at
`mixed_decoded_gpu.rs:325`. It exits 101 before the cold/hot output-comparison
path. The resource observer reports `observed_within_targets` with no errors;
this is not a fresh CUDA initialization failure or a proven memory overrun.

## Read-only localization for the model owner

The assertion's baseline is taken before the entire `prepare_mixed_rank_block`.
That path first prepares GemmEx storage (32 MiB workspace plus staging), then
allocates the 128 MiB decoded root. The allocator counts **every request >16 MiB**
as a huge allocation. Thus this counter's scope includes a provider allocation
as well as the root. Delta=2 alone does **not** prove two decoded-cache roots.
See `source-diagnosis.json` for exact source hashes tied to this test binary.

Do not simply change the expected value to 2 or relax the budget. The model owner
needs to isolate the root allocation boundary from provider preparation while
preserving exactly-one-root, partition and physical-budget checks. No model or
backend implementation/test was modified here, and this binary was not retried.

## Closure and remaining work

STOP.json preserves the failed round. All four executed gates have authenticated
ECHILD/drain, verified after-idle and released owner locks; all eight devices
were idle at final release. No foreign process was signalled.

The other seven model selectors, full40 and six recipes were **not executed**.
There are no new full-model calls or exact metrics, no measured hot-fit deltas,
no full-model 3 GiB snapshots and no public/performance acceptance. No missing
value is replaced by zero. Paper CPU regression after the workflow change:
**243 tests passed**, recorded in `cpu-paper-tests.*`.

A model/test correction requires newly frozen source and test-binary pins before
another authorized qualification; do not resume this directory or inherit its
passes into a source-different candidate. Old failed runs and frozen 4d40/dadf
artifacts remain intact.
