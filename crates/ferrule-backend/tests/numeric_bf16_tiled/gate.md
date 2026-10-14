# Frozen Hello L0 gate: native reduction compatibility

## Result (2026-10-05)

The original two-word mismatch is a **native reduction-structure difference**,
not a decoded-operand, output-tile addressing, or BF16 conversion defect.
Authenticated, bounded CUDA12.4/sm86 runs are recorded under
`target/validation/bf16-gate-native-20261005/`. No full40 run, reference update,
tolerance change, bespoke-index rounding, or model-default change was made.

The new **explicit backend-only** `NumericFp8Precision::Bf16SlicedK2F32Accumulate`
profile matches all 4096 pinned gate BF16 words **and all 4096 native F32
accumulators** at output tile sizes 64/257/1024/2047. Both typed BF16 input and
F32-input staging passed. Existing `Bf16RneF32Accumulate` remains unchanged and
matches all 8192 QKV BF16 words and native F32 accumulators. Do not replace this M1 QKV
profile with sliced-K2 or infer a universal HF kernel-selection policy. The
separate Capital M5 QKV gate requires sliced-K2; see `m5.md`.

## Provenance and observed words

`extract_gate.py` checks identical 2048-word BF16 input against the frozen
native HF and prior fused/tiled captures. Weight/scale payload SHA256 values:

- `828b21475685542ee11d4655872ec25a5d4cf9ded3f2922949108401f1d7c4ee`
- `4ad6470feea638ff238a7691cdb7545b9bb62224e4362da344988bd97cfac474`

Gate fixture SHA256SUMS pin:
`c6a90487a76ab30d5babb4e0154886c44024969d372a60833a3ff9c9a10190c5`.

| Index | HF BF16 | Fused BF16 | Original tiled BF16 | Sliced-K2 BF16 |
|---:|---|---|---|---|
| 753 | bfab | bfac | bfab | bfab |
| 1497 | bfa0 | bfa1 | bfa0 | bfa0 |
| 2650 | b9f3 | b9f2 | b9f3 | b9f3 |
| 3471 | bb50 | bb50 | bb4f | bb50 |
| 4065 | bc66 | bc66 | bc65 | bc66 |

| Index | Original tiled F32 accumulator | HF / sliced-K2 F32 accumulator |
|---:|---|---|
| 3471 | `0xbb4f7d50` | `0xbb4f8100` |
| 4065 | `0xbc657ba8` | `0xbc658030` |

All 4096 original tiled accumulators are identical across the four tile sizes.
The original max BF16 difference is `6.103515625e-05`; the candidate is exact.

## Actual native HF kernels

`profile_hf_gate.py` uses the original pinned producer initialization:
Torch 2.13.0+cu129, CUDA12.9, transformers 5.5.3, pinned source hashes,
TF32 off, BF16 reduced-precision reduction off. This deliberately does **not**
replace pinned Torch with CUDA12.4; Ferrule's separate native build uses 12.4.
Only frozen `F.linear` / `mm(out_dtype=float32)` operands are executed.

Gate BF16 kernel:
`ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`.
F32 output uses the corresponding `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x5_tn`.

QKV BF16 kernel:
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`.
F32 output uses its corresponding F32 epilogue variant.

For both shapes, native F32 output rounded to BF16 equals the native BF16 output
and pinned fixture exactly. Tiling HF `F.linear` itself changes its selected
reduction and reproduces the fused differences; that is not an oracle change.

## Implementation and ownership

`cuda_plan.rs`, `cutlass_bf16.rs`, private `ffi/cutlass.rs`, and
`native/cuda/cutlass/bf16_linear.cuh` expose a distinct sliced-K2 plan/kernel.
It was initially bounded to M=1; the validated multirow extension in `m5.md`
retains K divisible by 64 and all shared dimension/span limits. It transposes
the GEMM view (weight rows as M), uses CTA64x64x64, warp64x32x32, MMA16x8x16, five stages,
two on-chip K32 partitions per K64 tile, and F32 epilogue reduction. No global
split-K workspace, cuBLAS handle, persistent allocation, or new owner is added.

Decode remains `BF16_RNE(F32_RN(FP8 * BF16_scale))`, with global-row 128x128
scale addressing. Input pack, scratch planning, output strides, final BF16 RNE,
submission stream, completion, alias checks and quarantine remain unchanged.
On-chip geometry differs; **global scratch and launch counts do not**.

| Tile rows | Gate tiles | Final rows | Scratch bytes | Staging + GEMM launches |
|---:|---:|---:|---:|---:|
| 64 | 64 | 64 | 266240 | 129 |
| 257 | 16 | 241 | 1056768 | 33 |
| 1024 | 4 | 1024 | 4198400 | 9 |
| 2047 | 3 | 2 | 8388608 | 7 |

Each owner replay adds one BF16 writeback launch. The original model reservation
is 8 MiB, hence production tiles `[0,2047)`, `[2047,4094)`, `[4094,4096)`.
The two mismatches are internal to the second tile, not boundary words.

## Executable gates

All GPU execution must be under the authenticated supervisor with an exact
filter, explicit ownership, bounded timeout and post-run quiescence checks.
Fresh plan/report paths are required; existing evidence is never overwritten.
The evidence `run.py` uses the repository's authenticated subreaper supervisor.

Owner tests in `gpu.rs`:

- `gpu::fixed_input_gate_native_tiled_reduction_profile`: unchanged baseline.
- `gpu::fixed_input_gate_sliced_k2_exact_hf`: strict zero-difference gate,
  wrong-profile rejection, allocator/launch checks, both input APIs.
- `gpu::fixed_input_qkv_native_tiled_compatibility`: strict original QKV gate.
- `gpu::typed_tiled_padding_strides_and_graph_replay`: includes sliced-K2 with
  nonzero scale origins, padded strides, partial tile, graph replay and guards.

Gate tests take `FERRULE_BF16_TILED_GATE_DIR` and a fresh
`FERRULE_BF16_TILED_GATE_REPORT`. The exact test additionally takes
`FERRULE_BF16_GATE_NATIVE_F32`, pinned SHA256
`0a4c26c497f737540a287029015531091296d5a45e4a677bba0d7f6d474c1243`.
This is an additional diagnostic accumulator capture, not a replacement for
`hf.bf16`. QKV uses the existing `FERRULE_BF16_TILED_QKV_DIR/REPORT` variables.

`profile_hf_gate.py --output FRESH_DIR --execute-cuda --sole-owner-permission`
produces the profile/decoded operands/raw accumulator diagnostic files under
supervision without loading a model or changing any fixture.

`native_compat.cu` is a standalone operand/epilogue regression. Arguments:
`FIXTURE N DECODED_BF16 NATIVE_F32 PROFILE`, where 0=existing tiled and 1=sliced-K2.
It verifies every packed input/decoded weight word against HF (including both
disputed rows), compares raw and BF16 results, and tests direct BF16 versus F32
intermediate epilogues. Build with CUDA12.4 NVCC, `-arch=sm_86`,
`-DFERRULE_CUDA_TARGET_SM=86`, `-std=c++17`, `--expt-relaxed-constexpr`,
`--expt-extended-lambda`, native CUDA/CUTLASS include paths, `--cudart shared`,
`--cudadevrt none` and the actual CUDA12.4 library path. Gate/profile1 and
QKV/profile0 are the exact regression pair; unsupported profile choices fail.

## Scope limit

Both frozen operator gates pass; model integration remains explicitly absent.
An eventual model policy must select compatible reductions deliberately across
shapes; this result does not validate all layers or establish downstream full40
causality. Do not silently promote sliced-K2 to a universal model default.
