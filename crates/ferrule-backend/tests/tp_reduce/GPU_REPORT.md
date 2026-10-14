# Pinned NCCL two-card microgate: arithmetic proved, lifecycle BLOCKED

**Overall gate failed. Do not enable a public/model/runtime path or claim a hard
cap. No GPU replay was performed after the first failure.**

Evidence root:
`target/validation/tp-reduce-gpu-nccl2275-cuda124-20261011-02/`

Primary receipts: `closeout.json`, `actual-inventory.json`,
`preflight-eight-idle.json`, `finite/supervisor.json`, `finite/authentication.json`,
`finite/events.json`, `finite/memory-report.json`, `finite/resources.json`,
`finite/observed-library-sha256.json`, `finite/test-exit.json`, `finite/after.json`.
The earlier `...-01` directory preserves a GPU-hidden compile error, not a GPU run.

## Real hardware and source

User-authorized sole GPU owner, backend microgate only. Selected physical indices
0 and 1 were actually NVIDIA GeForce RTX 3090, compute capability 8.6, driver
**575.57.08**, with UUIDs:

- `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`
- `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`

Both existing UUID flock FDs were validated against the authenticated coordinator.
All eight devices had zero used-memory/utilization and no compute-app roster before
execution. This is local serialization, NOT a global reservation.

Compiler: exact tao NVCC **12.4.99**, sm_86. Runtime loaded tao CUDA **12.4.99** and
cuBLAS/cuBLASLt **12.4.5.8**, not CUDA13 or the Python3.12 NCCL versions.
NCCL source: CUDA12 package `nvidia-nccl-cu12 2.27.5`:

`target/vendor/vllm-baseline/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2`

| Observed dependency | SHA256 |
| --- | --- |
| NCCL2.27.5 library | `1d9c10322efacde695cf640db1f29fc1586c929f1087f0e1202d2d5e70fb7168` |
| Pinned nccl.h | `4b240a1fa79858ca8cdffcf5c2fc40aeaca8b3eaec61728213715f0902aba3a8` |
| tao libcudart.so.12.4.99 | `e8023d1aa80d86175f22986faf55e23c0a9286ff625cede0b560220c6cd8a560` |
| tao libcublas.so.12.4.5.8 | `88397a191cfc25f94c272b638ffa942be58d5594ba5d41ac4b766c2cd3fe54ac` |
| tao libcublasLt.so.12.4.5.8 | `a18a341a9b6de15918e410a2337a74f2a644fd80c0600e61d69ddad69be15687` |
| /usr/lib64/libcuda.so.575.57.08 | `f2470aa637fa72422534edeaeb9de19afe3d0646388baf7ffb5337b2edafc59e` |

The observed SHA receipt also records matching 575.57.08 NVML/PTX JIT libraries.
Exact native `dlopen`/symbol DSO identity/version checks passed. No NCCL or CUDA API
compatibility error was emitted. This does **not** qualify the full lifecycle.

## What actually ran

One exact ignored backend unit selector ran under the existing paper sealed
launch/authentication, owned Linux subreaper, and all-eight observer. It had a
120-second child deadline and a 180-second outer deadline. No model was imported
or executed. No Nsight/performance benchmark was run.

Proved in the real GPU process before its failing cleanup assertion:

- Both existing CUDA owners initialized rank0/rank1 using one unique ID and scoped
  threads. Both scoped threads joined; nonblocking async init reached success.
- Warmup completed at rows8/count16384; real compute events and 4-byte/rank status
  reads proved completion.
- M1/count2048, M5/count10240, rows8/count16384 produced **word-exact** BF16 results
  against the first-member host oracle.
- The gathered words were `[member37, member9]`, not sorted rank-ID order.
- Signed zero (including -0 + -0), BF16/F32 subnormals, cancellation and BF16 RNE
  tie cases were included in all three shape checks.
- Resident producer D2D on a separate stream was ordered by a producer event onto
  the original compute stream. Producer source/stream/event custody was retained;
  there was no host producer-completion fence before reduction enqueue.
- Reduction-site owner counters recorded no payload H2D/D2H or stream-wide sync.
  Actual native status-read instrumentation recorded exactly **8 bytes jointly**
  at each completed operation (4 bytes per rank). Oracle payload downloads were
  subsequent diagnostic validation, not site execution or a benchmark.

**Not run after the first failure:** GPU NaN/Inf/F32-overflow/BF16-overflow
rejection, repeated-two-site sticky-status gate, or the independent post-enqueue
Unknown/abort-retention process. They are compiled and supervised selectors, not
claimed as validated. Their CPU numeric/custody contracts remain covered.

## Measured memory, not attributable bounds

Both selected devices reported identical values at these snapshots. CU free bytes
are device-global, not process allocation counters. Before context creation, only
NVML free MiB was sampled; no context was created to fabricate a CU-byte baseline.
All eight-device free/used/utilization rows are retained for every stage.

| Stage | Global CU free bytes on each selected device | NVML used MiB |
| --- | ---: | ---: |
| Before context | Not queried; NVML free 24125 MiB | 0 |
| After context | 25021054976 | 263 |
| Before NCCL, owned backing allocated | 25018957824 | 265 |
| Async init ready, before warmup | 24909905920 | 369 |
| After rows8/count16384 warmup | 24880545792 | 397 |
| After finite checks / ordinary input allocations | 24813436928 | 461 |
| After comm/library drop, owners still alive | 24830214144 | 445 |
| After both Rust owners/context weak refs dropped | Not queried | **379** |
| After independent process exit | Not queried | **0 on all eight** |

Measured global free-memory changes per selected device:

- Owned backing preparation: **2 MiB**, while the Ferrule allocator recorded one
  **262144-byte logical driver-request grant**. Driver granularity/event overhead
  is not independently attributed; the logical grant is not an observed physical
  device-memory cap.
- NCCL init interval: **104 MiB** additional global occupancy; no additional Ferrule
  allocator driver requests.
- First warmup interval: **28 MiB** additional global occupancy; no additional
  Ferrule allocator driver requests.
- Ordinary fixture/producer inputs: **64 MiB** canonical allocator segment growth,
  separately reflected in the V3 ledger (two driver allocations total per owner,
  67371008 logical reserved bytes).
- Comm/library drop returned **16 MiB** globally. The Ferrule backing/input live
  grant reached zero; idle segments remained in its reserved ledger until owner
  shutdown. Both owners dropping returned a further 66 MiB in NVML, but did not
  make the process GPU-idle.

Known per-rank backing payload: 131076 bytes; logical grant: 262144 bytes. Native
library + two opaque handles minimum: 168 bytes. Default two-site input-view payload:
96 bytes. These exclude allocator metadata and NCCL/driver/runtime private state.

NCCL minCTAs=maxCTAs=2, cgaClusterSize=1, collnetEnable=0, netName=Socket,
blocking=0 were kept. There is no established exact channel/sliding-buffer bound.
**Private device and host bounds remain Unknown**, not 104/132 MiB and not zero.
No free-memory delta, sampled peak or warmup establishes a hard bound.

The existing observer recorded device-global sampled peaks of 461 MiB per selected
GPU, process-tree PSS peak 686304256 bytes and RSS peak 735244288 bytes. It reported
16 complete/1 incomplete host samples and 17 GPU samples. Owned-GPU attribution was
**unknown** under the PID namespace policy; owned-process GPU peak was **null**, not
zero. Exact peaks, hard partition and continuous coverage were false. Soft-target
observation success is not lifecycle or hard-cap qualification.

## Cleanup failure and safe stop

The test reached its normal destructor path: Rust context weak refs became empty,
the pinned NCCL DSO disappeared from maps, and all five post-init additional TIDs
were absent after comm/library drop. Thread names were inherited; TID-set evidence,
not a name filter, is the basis for this observation. Baseline CUDA driver/event
handler threads remained.

Nevertheless, NVML still reported **379 MiB used per selected device** after both
owners dropped, with compute-app entries (370 MiB each). The strict in-process idle
assertion failed. The precise CUDA runtime/driver primary-device retainer is
unresolved; extra runtime retains are a hypothesis, not a proven root cause.
No extra primary-context release, cudaDeviceReset, abort-as-completion recovery,
threshold relaxation, environment change, alternate library or GPU replay was used.

The outer authenticated custodian subsequently proved **kernel ECHILD**,
`proven_drained`, no cleanup errors, and process-tree exit. The observer ended
without errors. The post-exit all-eight idle receipt has zero used bytes,
zero utilization and no compute apps. Both same UUID locks were reacquired and
released in CPU-only closeout, proving the prior owner FDs had closed.

Status: **BLOCKED_NORMAL_OWNER_DROP_RETAINS_DEVICE_STATE**. CPU/default production
route remains disabled. Before any further qualification, a separately authorized
backend-only control/primary-context ownership diagnostic must distinguish existing
CUDA runtime baseline retention from NCCL-added lifetime state without force reset
or unowned release. The unexecuted rejection/Unknown gates must then be run in
independent supervised processes; no public/model integration follows from these
finite arithmetic subchecks alone.

## CPU-only closeout validation

After the GPU stop, no GPU context/communicator was recreated. The current source
passed supervised `cargo check --features cuda --tests`, CUDA-feature library/CPU
clippy with `-D warnings`, and all **8 CPU contracts**. Running the unit-test filter
WITHOUT `--ignored` proved **0 passed / 0 failed / 4 ignored**: default execution
still cannot run these GPU gates. The evidence parser/measurement-policy Python
suite passed **4/4**, with GPU queries mocked or absent.

Receipts: `cpu-closeout-validation/result.json` and its individual supervised
CPU-command receipts. The updated TID-based test assertion was only compiled;
its evidence basis is the recorded five-TID difference from the failed run. It was
not used to replace or relabel that failed GPU result.
