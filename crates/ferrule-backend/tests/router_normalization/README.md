# Explicit full-softmax router normalization

`RouterNormalization::SelectedSoftmax` is the unchanged default. Existing APIs
still select it. Mixed alone selects
`FullSoftmaxThenSelectedRenormalization` using
`CudaOperators::router_softmax_topk_with_normalization_into` and the existing
`SelectedSoftmaxTopKLayout`. No new model precision profile or layer rule.

Both policies select descending **raw logits**, skipping NaNs, with lower IDs
only for exactly equal raw values. Unequal raw logits whose probabilities round
to the same F32 must NOT be canonicalized as ties. Synthetic `raw-collision`
asserts this explicitly. This contract is not a claim of matching arbitrary
Torch `topk` probability-tie choices.

The new policy performs full F32 softmax, gathers selected probabilities, sums
those probabilities in F32, then divides and multiplies `output_scale`.
Source derivation from pinned Torch 2.13.0+cu129:

- `ATen/native/cuda/PersistentSoftmax.cuh`: next-power-of-two width, strided
  per-lane local max, descending XOR max, `std::exp`, serial per-lane sum,
  descending XOR sum, direct division.
- `ATen/native/cuda/Reduce.cuh`: contiguous selected sum, `last_pow2(k)` lanes,
  zero-initialized accumulators then descending shuffle-down. For k<=32 each
  lane has at most two inputs (remaining thread accumulators are zero).

These are source-grounded arithmetic choices, **not GPU parity evidence**.
CUDA12.4 versus pinned Torch12.9 exp/div and actual dispatch require the gates
below. No fast-math flags or host-computed expected floating-point arithmetic.

## Capability, resources, and nonfinite behavior

- Checked rows in 1..=i32::MAX, experts in 1..=1024, top-k in 1..=min(experts,32).
- Existing descriptor gains kind6; size/layout unchanged. Native preflight
  validates null/alignment/address overflow, packed-span overlap (including
  i32/F32 outputs), flags, bounds and finite output scale. Rust also checks
  context ownership, exact extents and F32 aliasing before dispatch.
- One 32-thread block per row, one existing tracked submission. No launch-time
  allocation, H2D, host arithmetic, synchronization, new handle or global scratch.
- Shared memory = `4*next_power_of_two(experts)+256` bytes/block: **1280 B** at
  256 experts, **4352 B** maximum. Test-trace sm86 compilation reports no stack
  or spills. Existing model workspace/cap/provider reservation is unchanged.
- NaN logits are ignored **only for IDs**, not sanitized in softmax. NaN,
  positive-infinity and all-negative-infinity rows give NaN weights for valid
  IDs. Missing IDs remain -1/zero. Finite rows with some -infinities use normal
  IEEE softmax. No underflow rescue or uniform-mass fixup. Model route validation
  still rejects invalid IDs/weights; default selected-only special cases remain.
- Work ownership, leases, completion/Unknown handling and lazy expert execution
  are unchanged. Existing buffers flow through existing submission tracking.

## Frozen fixtures and independent expected values

`prepare.py OLD NEW` is CPU-only, refuses an existing destination, verifies
corrected `.direct.bf16` equals pinned `.hf.bf16`, expands BF16 exactly, and copies
original HF IDs/weights without modification. Four actual cases: hello0/hello3
M1, capital0/capital1 M5 (3072 logits, 96 selected weights). It excludes old
hello3 intermediates because those used a differing raw logit.

`oracle.py NEW --execute-cuda` is **GPU-only and grant-required**. It uses the
existing pinned HF initializer in a separate Torch process, verifies immutable
source fixture hashes, reproduces all original actual weights bitwise, and
writes independent probabilities/selected/sum/normalized stages. It then adds
56 dense M1/M5 cases (expert counts1/7/32/128/256/257/1024, k1/3/8/13/32 where
admissible), plus exact ties, unequal-raw probability collision, underflow and
wide finite cases: **64 cases total**. Original weight files are never rewritten.
New synthetic expected arithmetic is generated only on the future GPU run.

`probe.cu` instantiates a trace-only specialization of the same kernel. It saves
raw roundtrip, IDs, max, exponentials, full denominator, full probabilities,
selected probabilities, selected denominator and output. `compare.py` compares
all independently observable stages bitwise and writes every case's counts,
max error and first differing stage before returning failure. Max/exp/full sum
are retained diagnostically, not compared with a different Torch reduction.
`--require-production` also requires traced output equal the public Rust API.

## CPU validation / build

Always hide GPUs **after** sourcing the CUDA12.4 environment. The standalone
probe needs `--cudart=shared --cudadevrt=none` and the maintained toolchain link
path; `--preflight` performs no CUDA API calls. Rust nonignored test is CPU-only:

```sh
cargo test -p ferrule-backend --features cuda --test router_normalization
cargo test -p ferrule-model --features cuda --lib router_normalization_is_mixed_only
cargo clippy -p ferrule-backend --features cuda --lib --test router_normalization -- -D warnings
cargo clippy -p ferrule-model --features cuda --lib --tests -- -D warnings
```

## Next sole-GPU-owner gate order

Prepared CPU-only evidence: `target/validation/router-normalization-20261007/`.
`PREPARED.json` pins sources, binaries, CUDA libraries, original expected values,
and Torch headers. It defines literal environment and <=120s per child. Rebuild
and repin if **any** source/binary pin drifts; do not blindly update hashes of an
old binary. No whole-model capture, tiny or full40 in these plans.

After an explicit ownership grant, run each label sequentially, stopping on
failure, with the existing authenticated subreaper/quiescence supervisor:

```sh
/opt/conda310/bin/python -B crates/ferrule-backend/tests/router_normalization/run_prepared.py target/validation/router-normalization-20261007 safety --execute-cuda --sole-owner-permission
```

Repeat that command for labels in this exact order:
`safety`, `nonfinite`, `oracle`, `trace`, `stages`, `weights`,
`production-stages`, `model`.

Production children start with CUDA12.4 libraries and
`CUBLAS_WORKSPACE_CONFIG=:0:0`; only the pinned Torch oracle uses its own CUDA12.9
libraries and `:4096:8`. Supervisor authenticates drain/ECHILD and all-GPU idle.
Fresh receipts are mandatory; failed arithmetic is not a retry permission.
The model gate reuses actual resident BF16 projection dispatch and now **asserts
zero weight differences**, rather than merely logging the prior48/96 mismatch.
