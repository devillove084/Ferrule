# Bounded segmented WMMA reduction regression

This changes only the M>8 W8A16 kernel's accumulation. M<=8 vector arithmetic,
F32-default kernels, BF16-RNE weights/activations, original scale dtypes and
128x128 origins, F32 output, one launch and zero **global** numeric workspace
are unchanged. This is original Ferrule code, not a Marlin port.

## Reproduced cause and change

The old M>8 path kept one F32 WMMA accumulator alive across all K tiles.
Rounding in that long Tensor Core chain lost low-order contributions even
though the inputs and decoded BF16 operands were correct. This is a reduction
accuracy failure, not an output-dtype or quantization-format mismatch.

On the pinned capital QKV row (K2048, N8192), M9 failed 81/73728 outputs and
M32 failed 288/262144 under the UNCHANGED `1e-5 + 2e-6 * sum(abs(A*W))` bound;
max error was 0.0002833883627317846, fraction 2.2175527212181514. The vector
M5/M8 path passed. Baseline evidence remains immutable at
`target/validation/marlin-audit/capital.captured-triangle.json`.

The fix resets WMMA after at most 128 K elements (eight k16 MMA operations).
A binary carry merges equal-sized adjacent partials with explicit `__fadd_rn`.
Only initialized/occupied levels are read; trailing partial subtrees fold
low-to-high to retain logarithmic depth without adding artificial zero nodes.
It shortens the hardware accumulation chain independently of K and avoids
repeatedly adding small products into the final large sum. It does NOT rely
on guessing the Tensor Core's internal summation order, and is not a proof
that all finite/overflowing/subnormal inputs obey a universal error bound.

For S=ceil(K/128), the tree needs floor(log2(S))+1 levels. K<=2048 selects 5
levels, K<=16384 selects 8, larger legal K selects 25 (K<=INT_MAX-32). Each
level holds a 16x32 tile, split into coalesced **thread-private** WMMA fragment
slots; no warp reads another warp's slots. The first 22 levels fit in shared
memory, and any remaining three use registers. All CTA staging/epilogue
barriers remain intact. Total shared storage is 13312/19456/48128 bytes,
reported by `NumericFp8LinearPlan::fused_shared_bytes_per_cta`. No dynamic
allocation, split-K global partial matrix, atomics, repacking or vector-all-M
fallback. CUDA12.4/SM86 ptxas: zero stack and spill bytes for every variant.
The register-only candidate was rejected due to spills and ~3x large-shape
regression; its evidence is retained, not the production implementation.

## Permanent regressions

- `wmma.rs`: hermetic xorshift32 seed `0x574d4d41`, positive low-mantissa
  products and cancellation probes. M8/9/32, K2048/2051/16384/16385/32769,
  N33/7, origins 127/125, both scale dtypes, both activation APIs, output
  canaries, input immutability, no execution allocations, independent scalar
  CUDA/F64 comparison. 60 cases, 17052 outputs. All production outputs satisfy
  the original bound. Independent serial F32 itself loses accuracy on some
  larger positive sums: those failures are reported, never used as the truth;
  its bits still match the independent host serial calculation exactly.
- The M9 N33 K2051 positive F32-scale case was also run against the frozen old
  native kernel: **288/297 failures**, max fraction **1.50457259297**. This
  ensures the seeded fixture actually catches the original defect.
- `capital.rs`: permanent named real-case regression with hardcoded manifest
  SHA256 `b351ece2ac055fb4cf63bd1696190a90de0bbfb3f45c76724c908103e8418325`.
  It reuses the existing captured entry; manifest and every payload are
  verified BEFORE CUDA initialization. No caller-supplied alternative hash.
  External data is not copied into the repository; missing data is not a pass.
- Existing 320-case triangle, ragged/nonaligned dispatch tests and precision
  tests remain unchanged. `metadata.rs` includes the shared-memory thresholds.

## Actual CUDA12.4 / SM86 results

Evidence directory: `target/validation/w8a16-wmma-fix/`.

- Capital M9/M32: **0 violations**; max error 8.730159606784582e-6,
  max fraction **0.07702892920629677**. Original M5 and repeated M1/M8 match
  archived vector outputs bitwise. Independent scalar: zero bound violations.
- Hello shared gate captured test: pass, including M9/M32.
- Existing triangle: **320 cases / 92496 outputs / zero production or scalar
  violations**. Both scale dtypes, ragged K and four origin pairs retained.
- Non-external numeric W8A16 suite: **16/16 pass** (including new seeded test).
- Compute Sanitizer 2024.1.0.0, seeded cases: memcheck **0 errors**;
  racecheck **0 errors / 0 warnings**. Bounds 1500s per subprocess; actual
  about 61s / 151s. No full40/model execution. Old W8 quality FAIL unchanged.
- RTX3090 SM86; driver 575.57.08. Commands use the existing CUDA12.4 toolchain
  environment and verified `_owned_subreaper` custody; all children drained.

Warm CUDA graph microbenchmark (100 calls, 3 warmups, median of 7), BF16 scale
and input, K2048. Clocks were not locked; values are not serving throughput.

| M | N | Before us | After us | Change |
|---|---|---:|---:|---:|
| 1 | 512 | 13.7830 | 13.7830 | 0.00% |
| 8 | 512 | 18.3603 | 18.2989 | -0.33% |
| 9 | 512 | 77.3837 | 79.5238 | +2.77% |
| 32 | 512 | 78.1619 | 80.4762 | +2.96% |
| 1 | 8192 | 62.1363 | 60.9485 | -1.91% |
| 8 | 8192 | 169.9431 | 170.0762 | +0.08% |
| 9 | 8192 | 118.4461 | 122.5318 | +3.45% |
| 32 | 8192 | 229.4374 | 239.5955 | +4.43% |

All 182 benchmark cases and individual ratios are retained. Across the measured
WMMA cases the regression is +1.81% to +7.26%; the worst is M32 N4096 K2048
with BF16 scales/F32 input (121.897 -> 130.744 us). Do not generalize the
representative table to all geometries.

## Reproduction (sole GPU owner, run sequentially)

Source `target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`, then set
`FERRULE_CUDA_TEST_ORACLE=1`. Use bounded `run_process` from
`scripts/test_vllm_qwen35.py` for CUDA subprocesses, as in this evidence's
`owned_run.py`, never run simultaneous owners. Build:

```sh
cargo test --offline --locked -p ferrule-backend --features cuda --test numeric_w8a16 --no-run
```

Run the compiled test executable with `--ignored --nocapture --test-threads=1`
and `--exact` for each of:

- `scalar::reduction::seeded_segment_carry_and_dispatch_boundaries`
- `scalar::capital::capital_m9_m32_fixed_operator_bound`, setting
  `FERRULE_W8A16_CAPITAL_CASE_DIR` to the original `capital-l1-qkv` fixture
  directory and `FERRULE_W8A16_CAPITAL_REPORT` to a **new** JSON path.
- `scalar::independent_scalar_cuda_host_production_triangle`, optionally
  setting `FERRULE_W8A16_TRIANGLE_REPORT` to a new JSON path.

For the full non-external suite use `--include-ignored --skip scalar::captured
--skip scalar::capital --test-threads=1`. Invoke Compute Sanitizer with
`--tool memcheck` / `--tool racecheck --error-exitcode 99` on the exact seeded
regression. The actual captured test also checks original M5 and repeat-row
M1/8/9/32 without changing model code or full-model acceptance.

Limits: tested hardware is SM86 only. Largest allocation-tested K is 32769;
higher carry levels are compile/resource bounded but not exercised by a huge
checkpoint allocation. Overflow/subnormal behavior is not newly guaranteed.
