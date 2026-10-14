# Chunk continuation, reset and optional operator timing

This additive suite owns only `chunk_continuation*` files. It does not modify
production, the shared chunk/recurrent suites, model profiles or old evidence.
`chunk_continuation.cu` includes the existing `chunk.cu` guarded Buffer and
contracts with its main renamed; observation is disabled. Preparation uses the
same fixture formats, CUDA12.4 compiler helpers and authenticated supervisor.

## Coverage added

- Changed dense inputs across consecutive **5 -> 65 -> 129** row calls with
  the same F32 SSM and scratch allocation. Every input is uploaded before the
  first launch. Calls and state snapshots are queued on one nonblocking stream,
  **without host fences, state resets, scratch clearing or allocation between
  calls**. Snapshot copies are stream-ordered; outputs remain independently held.
- Dense129 uses three chunks, including a one-row tail, grouped heads and actual
  full-width Hv32/Dk128/Dv128 geometry. Dense64 uses the same geometry.
- All new synthetic inputs use finite nonzero, non-underflowing negative decay,
  dense Q/K/V, and non-BF16 F32 initial state. Alternating key-axis signs with
  near-cancelling magnitudes stress cancellation; these are not identity or
  zero-state fixtures. CPU checks characterize inputs, not GPU arithmetic parity.
- Dk/Dv **511/512**, **512/511**, **512/512** exercise near-limit and exact-limit
  admissible geometries with changed inputs and carried state.
- After the carried sequence, a *different* independent F32 reset seed runs
  with the dirty shared scratch. A second run uses the same reset seed/inputs
  and separately poisoned fresh scratch. Output/state must be bit-exact to each
  other and to separately generated independent HF reset expectations.
- Every call output and F32 state is checked bitwise against HF, not just the
  last state. Both-end canaries and input/seed immutability are required.

The actual-five-row case copies the unchanged `chunk.sha256`-pinned fixture and
its already-independent expected output/state. No original files are rewritten.
New synthetic/carry/reset expectations **do not exist after CPU preparation**.
The GPU worker calls the pinned unmodified native HF chunk function sequentially,
carrying only its own HF state. It never reads native outputs as expected values.
It uses Torch2.13 CUDA12.9 / Transformers5.5.3 in a separate supervised process;
the native executable is strictly CUDA12.4/sm86. This is the existing CHUNK
oracle/runtime separation, not a claim that HF runs on CUDA12.4.

## Ready CPU evidence

`target/validation/chunk-continuation-cpu-20261006/prepared/`

- `prepared.json`: case inventory and source/input/binary pins;
  status `CPU_PREPARED_GPU_NOT_RUN`.
- `build.json`: compiler/resource output.
- `contracts.json`: host-only native ABI and near-limit sizing checks.
- New numerical gates and all timings remain **pending GPU execution**.

CPU-only reproduction, using a **fresh** preparation directory:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_continuation.py prepare target/validation/chunk-continuation-next-prepared
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_continuation_test.py --prepared target/validation/chunk-continuation-next-prepared -v
```

## Next GPU owner commands

**Do not run until sole ownership is granted and all GPUs are idle.** Existing
prepared source/artifact pins must still verify; source changes require a fresh
CPU preparation, not updating old pins in place. Evidence directories must not
exist. No model/full40 execution is involved.

Correctness only:

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_continuation.py run target/validation/chunk-continuation-cpu-20261006/prepared --evidence target/validation/chunk-continuation-gpu-20261006
```

Correctness plus explicit opt-in timing, as an alternative single ownership cycle:

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_continuation.py run target/validation/chunk-continuation-cpu-20261006/prepared --evidence target/validation/chunk-continuation-performance-gpu-20261006 --performance
```

All children use the existing authenticated subreaper; HF child <=120s, each
native correctness/performance child <=120s. Matching helper/launch identities,
`proven_drained`/ECHILD, immutable pins, bounded cleanup and three all-eight-GPU
idle samples are required. Receipts and final release are preserved on failure.
Exact failures stop before timing, with no tolerance or reference substitution.

## Performance interpretation

Only actual5, dense64 and dense129 are timed. The two variants use identical
input/seed addresses and the same stream. Twenty AB/BA pairs follow warmup;
CUDA events cover one complete operator call. State reset, allocation, transfers,
validation and output/state checks are outside events. Chunk scratch is preallocated
and reused; scan ignores it. Every timed output/state must match that variant's
own untimed result, and timed chunk results must also match independent HF.
Raw times, medians and **separate scan/chunk errors against the HF chunk reference**
are retained. The current multirow chronological scan is not asserted equivalent
to chunk: **different numerical profiles, not an equal-quality speedup claim**.
No clocks are locked, throughput extrapolated or performance result claimed by
CPU preparation. CUDA lifetime success is also not inferred from host contracts.
