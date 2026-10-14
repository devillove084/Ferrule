# Cooperative F32 paged GQA validation

## Contract

The existing `PagedF32Gqa` API and core provider ABI/layout are unchanged.
One block owns a query head/row. Each key's dot product retains serial dimension
order; thread 0 computes the original **online** softmax weight/rescale events
and denominator in token order. Value dimensions replay those events in the
original order. This is not a two-pass max/sum or parallel-reduction softmax.

Shared scratch is bounded to 2048-token tiles (about 10 KiB per block).
Longer visible rows loop over tiles, preserving the running maximum and
normalizer. Caller-owned output stores unnormalized accumulators between tiles;
normalization happens only after the final tile. This also supports head
dimensions larger than the block size. Scalar fallback is retained only for
query-head grids exceeding `INT32_MAX`, not for long contexts. No global score
matrix, extra allocation, CPU attention, precision change, additional CUTLASS
family, or production test-selector ABI is introduced.

`gqa_legacy.cuh` freezes the pre-change GPU kernel independently of the new
score helper. `gqa_cooperative.cu` compares output **bits**, then applies the
same deterministic GPU projection and compares its logits **bits**. These are
small synthetic projected logits, not a claim about full-model 35B logits.

Coverage includes:

- Context 1/16/64/1024/2048/2049/4096/4106/8192, including mixed one/two-tile rows
  and new maxima in successive tiles.
- D=3/31/33/128/129/256/257/513, Q/KV=6/2, 5/1, 9/3, 3/3, 16/2.
- Interleaved sequences, causal history, permuted physical pages, shared
  read-only prefixes with private post-COW pages, nonzero layer index, distinct
  padded K/V strides, output padding, score ties, repeated maxima, underflow,
  zero visible tokens, and invalid metadata/key/value addresses in first/later tiles.
- NaN/Inf Q/K/V on both sides of the 2048 boundary, equal infinite scores,
  and poisoned future tiles masked by either causal position or KV length.
- The production launch helper is inspected through CUDA graph capture, including
  the oversized-grid scalar fallback (captured but never executed). Shared-memory
  size and output canaries are checked.
- The public API tests append across page boundaries, check exact cache writes
  and untouched storage, retain the existing CPU oracle tolerance `3e-6`, and
  reject all six foreign-owner operands before any cache mutation.

## Historical sm86 kernel timing (before tiled recovery)

The following measurements and results are retained as historical evidence;
they do not validate the current tiled implementation.

Measured 2026-09-26, GPU 0 NVIDIA GeForce RTX 3090, sm86, driver 575.57.08,
NVCC 12.9.86, `-O3`, default floating-point/FMA flags. One row, Q16/KV2/D256,
page size 16, layer index 1 of 3, fixed inputs and physical page table.
CUDA events measure attention only: no uploads, allocation, append, status
readback, projection, model/runtime or CPU work. Five batches of 30 launches,
five warmup launches per batch, alternate old/new order, median batch mean.
The final measurement ran separately from other GPU tests; clocks were not
locked. Do not extrapolate these ratios directly to end-to-end 35B latency.

| Context | Frozen old kernel ms | Cooperative kernel ms | Speedup | Output/logit comparison |
| ---: | ---: | ---: | ---: | --- |
| 23 | 6.972928 | 0.305118 | 22.85x | Bitwise pass |
| 55 | 16.714411 | 0.352802 | 47.38x | Bitwise pass |
| 128 | 39.149639 | 0.464418 | 84.30x | Bitwise pass |
| 1024 | 314.205963 | 2.733500 | 114.95x | Bitwise pass |

## Reproduction (repository root)

Compile-only validation (no CUDA device initialization):

```sh
NVCC=/usr/local/cuda/bin/nvcc cargo test -p ferrule-backend --features cuda --test gqa_cooperative gqa_cooperative_compile_only
```

The build script also needs `nvcc` on `PATH`; set `FERRULE_CUDA_ARCH` explicitly
on hosts without a GPU. `cargo test --no-run` only compiles the Rust harness;
the named compile-only test additionally compiles its CUDA fixture.

Run GPU tests/benchmarks sequentially, not concurrently:

```sh
cargo test -p ferrule-backend --features cuda --test cuda_standard -- --include-ignored --test-threads=1
cargo test -p ferrule-backend --features cuda --test gqa_cooperative -- --include-ignored --test-threads=1
cargo test -p ferrule-backend --features cuda --test cuda_semantic_operators paged_ -- --include-ignored --test-threads=1
cargo test -p ferrule-backend --features cuda --test native_layout --test cuda_build_paths
```

Direct sm86 benchmark and bounded sanitizer fixtures:

```sh
nvcc -std=c++17 -O3 -arch=sm_86 -I crates/ferrule-backend/native/cuda -Xcompiler=-Wall,-Wextra --Werror=all-warnings crates/ferrule-backend/tests/support/gqa_cooperative.cu -o target/gqa-cooperative
target/gqa-cooperative
target/gqa-cooperative --bench
compute-sanitizer --tool memcheck --error-exitcode 1 target/gqa-cooperative --sanitize
compute-sanitizer --tool racecheck --error-exitcode 1 target/gqa-cooperative --sanitize
```

Historical pre-recovery results: 41 direct fixed-input fixtures passed bitwise (plus zero-visible and
invalid-device-metadata subcases); 6 public standard tests passed; 2 paged BF16
regression tests passed; 8 native layout and 5 build/provider ABI tests passed.
Memcheck reported 0 errors; racecheck reported 0 hazards/errors/warnings.
`cargo fmt --all -- --check` and `git diff --check` passed.

No 35B model was loaded. Full-model logits/oracles and end-to-end Nsight timing
remain for the coordinating integration run; no oracle threshold was relaxed.
