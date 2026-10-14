# Internal two-member TP backend handoff

> Ownership A/B/C control is now complete: [OWNERSHIP_REPORT.md](OWNERSHIP_REPORT.md).
> It establishes baseline process-scoped CUDA residency and a stable measured NCCL
> first-use increment across five cycles, not a hard cap or numerical-gate pass.
> The historical strict in-process-idle numerical gate below remains archived as failed.

> Latest authorized GPU result: **BLOCKED_NORMAL_OWNER_DROP_RETAINS_DEVICE_STATE**.
> Finite M1/M5/rows8 arithmetic is word-exact on the actual two RTX3090/575.57.08
> devices, but normal in-process owner drop did not return the GPUs to idle.
> See [GPU_REPORT.md](GPU_REPORT.md) for measured-only allocation data, dependency
> SHA receipts, safe process-exit cleanup proof, and unexecuted GPU cases.
> The CPU/hidden-device validation section below describes the earlier CPU stage,
> not an assertion that no authorized GPU run has subsequently occurred.


No model/runtime/composition/paper changes. This is not a selected provider or a
public execution route. Routing/control and head collectives stay on their host
fallback. GPU execution is not part of CPU/hidden-device validation.

## Source and build

- CUDA12 package: `target/vendor/vllm-baseline/lib/python3.10/site-packages/`
  `nvidia_nccl_cu12-2.27.5.dist-info/METADATA`, `nvidia/nccl/include/nccl.h`,
  `nvidia/nccl/lib/libnccl.so.2`.
- Header SHA256:
  `4b240a1fa79858ca8cdffcf5c2fc40aeaca8b3eaec61728213715f0902aba3a8`.
- Library SHA256:
  `1d9c10322efacde695cf640db1f29fc1586c929f1087f0e1202d2d5e70fb7168`.
- Build verifies the optional header hash. `FERRULE_NCCL_INCLUDE_DIR` may specify
  another directory containing the exact same header. Without the local header,
  the native API reports unavailable; it never falls back to system NCCL.
- Runtime requires an absolute path, canonicalizes it, verifies the library hash,
  then native `dlopen(RTLD_NOW|RTLD_LOCAL)` verifies version 22705 and symbol DSO
  identity. Pinned artifacts must remain immutable during loading.
- Do not mix vllm028 Python3.12 NCCL2.29.7 or system CUDA13.2 NCCL2.30.4.
- Compiler used for the intended hidden-device check:
  `/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc`, CUDA12.4.99, target sm_86.
  Compilation is not proof of GPU/NCCL topology, execution, or performance.

## Operator API and semantics

`cuda::operators::tp_reduce` is doc-hidden and not in the provider catalog.

1. Construct `TpReducePlan::new([first_sealed_member, second_sealed_member], budget)`;
   rank 0 is the first member and rank 1 is the second. Actual IDs are never sorted.
2. Load `NcclLibrary` with its exact absolute source path and an explicit minimum
   native-handle budget; this is CPU-only, not communicator/GPU initialization.
3. `InternalTpReduceTwo::start([&owner0,&owner1], plan, library)` preflights both
   existing CUDA owners, installs backing custody, then uses one unique ID on two
   scoped threads. CommInitRankConfig is nonblocking (`blocking=0`).
4. Call `advance` from the existing caller's bounded progress/deadline facility.
   It makes one status probe per rank, with no internal polling loop/watchdog.
   Warmup always AllGathers count16384/rows8. Call `check_at_quiescence` to validate
   the warmup events and sticky numeric flags before real enqueue. `start` preallocates custody for two sites; use
   `input_custody_estimate` and `start_with_custody` with an explicit host-view budget
   for ALL sites up to the caller's existing quiescence (e.g. a request). Zero/
   overflow/underbudget capacities are rejected before communicator initialization.
5. `enqueue` accepts two contiguous BF16 `[rows,2048]` inputs, rows1..8, checks
   BOTH exact owners/byte counts/alias ranges before either submission, and retains
   both input allocations. Multiple sites enqueue without a status read or completion
   checkpoint between them, until the preallocated custody limit. Consumers of a
   prior output must be queued on its original stream before the next site reuses
   the workspace. The original compute streams order:
   input producers -> ncclAllGather -> finite validation -> ordered F32 add/BF16
   RNE -> future same-stream kernels.
6. `Pending` means host enqueue is incomplete. Call `advance` from the existing
   progress facility before scheduling downstream kernels. Do not equate an
   ncclInProgress return with completed enqueue. No host payload copy or stream
   synchronization is performed at a reduction site.
7. Provisional output for downstream kernels is available only through unsafe
   `tentative_output`, with explicit no-publication/joint-custody obligations.
   `check_at_quiescence` queries existing events and async status and reads FOUR
   status bytes/rank, never site payloads. Only `output` after a successful check
   permits publication. NaN/Inf input, F32 overflow, and BF16 RNE overflow reject
   the invocation. Device status is sticky and is never reset between sites.
8. Drop/destruction queries completion, including the full compute-stream tail;
   it never waits for a missing peer. Any lost proof leaks the joint native handles,
   library, events, input/backing allocations and context owners rather than freeing
   memory under pending work. `abort_unknown` is cancellation, NOT completion proof.
   Unknown is terminal; even a later successful fence cannot authorize release.

The arithmetic matches `HostCollectiveGroup::compute`: expand BF16 member0,
`__fadd_rn(member0, member1)`, one BF16 RNE. There is no initial `+0`, NCCL SUM,
rank-ID sorting, or pre-rounding. A single quiescence check covers every queued
site via the last re-recorded event on each stream and the persistent sticky flag. Two native kernels ensure a nonfinite status
prevents any output writes for that invocation. Publication remains jointly gated.

## Honest resource accounting

Per rank, one dedicated owned backing contains:

- zeroed warmup input: 32768 bytes;
- rank-ordered gather: 65536 bytes;
- contiguous output: 32768 bytes;
- sticky status: 4 bytes.

Total payload backing: **131076 bytes**. The existing allocator's small-block
power-of-two grant is **262144 bytes**, not 131328. Native opaque handle sizes are
queried with `sizeof`; they exclude Rust/dlopen/NCCL private host metadata and
are a minimum metadata budget, not a host hard cap. The CPU-only native test
measured **168 bytes** for the library plus two communicator handles, and the
default two-site retained input-view array payload measured **96 bytes**. These
exclude allocator overhead, Rust object metadata, dlopen and private NCCL state.

Pinned config sets minCTAs=maxCTAs=2, cgaClusterSize=1, collnetEnable=0,
netName=Socket, blocking=0. This does **not** prove an exact channel allocation
count or cap NCCL protocol/transport allocations. NCCL2.27.5 has no per-comm
sliding/ring-buffer size setter; environment/config-file defaults can still affect
private allocation. Existing NCCL environment overrides are rejected, never
mutated. Nsight/internal workspace data has not been measured and is **Unknown**,
not zero. `TpResourceEstimate::hard_cap_bytes` deliberately returns an error.

A future model/resource-claim adapter can charge the owned-device grant, but MUST
NOT admit a hard-capped public route while NCCL private host/device bounds are
unknown. A credible attributable bound requires separate authorized research of
this exact library/config/transport; a global free-memory delta is not such a bound.

## GPU gate (main authorization only)

Ignored unit tests in `gpu.rs` additionally require
`FERRULE_TP_REDUCE_GPU_AUTH=main-authorized-two-card-proof`.
They cover original-member AllGather mapping, word-exact signed zero/subnormals/
BF16 ties at rows1 and rows8, pre-enqueue owner/byte rejection, nonfinite/F32/BF16
overflow rejection, two-site enqueue without intermediate quiescence/sticky rejection, and deterministic
post-enqueue Unknown custody. Run only after
main sole authorization, on the pinned CUDA12 toolchain and two suitable GPUs.
No model integration or performance claim follows from compilation alone.

## CPU / hidden-device validation completed

All commands used `CUDA_VISIBLE_DEVICES=''`; no GPU gate was run.

- CPU contract/artifact tests: **7 passed** (`cargo test -p ferrule-backend --test tp_reduce`).
- CPU clippy (`cargo clippy -p ferrule-backend --test tp_reduce -- -D warnings`).
- Standalone native TU compiled with the exact pinned header and with the
  unavailable-provider stub, NVCC12.4.99/sm_86, warnings as errors.
- PTX inspection showed `add.rn.f32` in both validation/addition kernels, no FTZ
  addition (compiled with explicit `--ftz=false --fmad=false`).
- `cargo check -p ferrule-backend --features cuda --tests` with the pinned NVCC
  on PATH and `FERRULE_CUDA_ARCH=sm_86` compiled the ignored GPU gate as well.
- CUDA-feature library/CPU-test clippy with `-D warnings`.
- CUDA-feature CPU-only tests: **8 passed**, including pinned `dlopen`/source/SHA/
  budget validation, linked against
  tao CUDA12.4.99 using temporary CUDA12 library aliases under
  `target/validation/tp-reduce-cpu/cuda12-lib`; `ldd` confirmed libcudart.so.12.
- Targeted rustfmt and backend diff whitespace checks.

The broader `native_layout` suite has two unrelated existing failures: audited
native-file lists omit other pre-existing CUDA files, and private-extern lists
omit other pre-existing modules. The two TP files/private FFI module were added
without widening those unrelated allowlists. No claim that the full suite passes.

## Authenticated backend gate runner

`gate.py` and `gate_support.py` reuse the existing paper UUID flock, sealed
launch/authentication, owned subreaper and observer without changing paper code.
Preparation is GPU-hidden and freezes the exact staged binary/source/tool hashes.
Execution requires the sole-owner flag/note, both UUID locks, all-eight idle, the
sealed physical 0/1 roster, and the worker authentication receipt. Each exact
selector is a fresh process with 120s/180s child/outer deadlines. The first failure
stops all later cases; never remove the normal-owner-drop assertion to pass a gate.

Example CPU-only preparation (requires a NEW identity):

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  crates/ferrule-backend/tests/tp_reduce/gate.py prepare --id NEW_ID
```

Do not replay `nccl2275-cuda124-20261011-02`: its failed GPU evidence is immutable.
A fresh explicit main decision is required for a separate backend ownership
control diagnostic before any further GPU qualification. Overflow/Unknown GPU
cases have been compiled but were NOT executed following this failure.
