# Frozen real-operand reduction experiment

No model run, precision fallback, router or output-rounding change. The existing
independent scalar CUDA entry (`reference.cu`) is reused without modification.
The production native kernel is also unchanged.

`prepare_captured_triangle.py` runs on CPU only. Select an existing numeric
`linear.output` by its exact sequence number; it locates the corresponding
captured input and reads only that projection's checkpoint slices. The fixture
freezes BF16 input words, original FP8 weights, original numeric BF16 scales,
production F32 output, F64 dot products and sum-absolute-products. All files are
hashed. It compares serial F32, current packed-vector order, two-bank F32 and
balanced pairwise F32. It requires current-vector replay to match captured output
bit-for-bit; replay is a diagnostic, never independent acceptance.

Example (existing layer-0 hello QKV):

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B crates/ferrule-backend/tests/numeric_w8a16/prepare_captured_triangle.py --trace target/validation/w8a16-first-divergence-20261002/gpu-trace/hello --sequence 4 --output target/validation/w8a16-captured-triangle-new/hello-l0-qkv
```

CPU tests:

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B crates/ferrule-backend/tests/numeric_w8a16/test_captured_reductions.py
CUDA_VISIBLE_DEVICES='' cargo test -p ferrule-backend --test numeric_w8a16 --offline
```

Compile-only (no GPU initialization):

```sh
CUDA_VISIBLE_DEVICES='' PATH=/usr/local/cuda/bin:/root/.cargo/bin:/usr/bin:/bin FERRULE_CUDA_ARCH=sm_86 FERRULE_CUDA_TEST_ORACLE=1 cargo test -p ferrule-backend --features cuda --test numeric_w8a16 --offline --no-run
```

After sole-owner GPU handoff, run only
`scalar::captured::captured_same_bf16_operands_vector_mma_scalar_f64` with:

- `FERRULE_W8A16_CAPTURED_CASE_DIR`: frozen fixture directory
- `FERRULE_W8A16_CAPTURED_MANIFEST_SHA256`: literal SHA256 of its `SHA256SUMS`
- `FERRULE_W8A16_CAPTURED_REPORT`: fresh JSON path
- `FERRULE_CUDA_TEST_ORACLE=1` at build time

The ignored test runs all observed rows once, then repeats **the exact first
row** at M=1,8,9,32 to isolate vector/WMMA dispatch from upstream/model changes.
Each call compares production F32-input/BF16-input APIs, independent scalar CUDA,
and F64. It checks canaries and input immutability. Saved-production mismatches
are recorded, not treated as acceptance failures for a changed implementation.
The only numerical acceptance remains `1e-5 + 2e-6*sum_abs_products`; pairwise
bits and BF16-bin changes are diagnostics. It is not a full-model quality gate.

This test is not a microbenchmark: the scalar host-array wrapper allocates and
copies. Do not interpret wall-clock runtime as kernel latency. Any future kernel
candidate needs separate CUDA-event timing using `performance/kernel_bench.cu`,
with register/spill reports and unchanged numerical acceptance.

The 2026-10-03 no-GPU investigation, six pinned real fixtures, exact handoff
command and source hashes are in:

- `target/validation/w8a16-captured-triangle-20261003/REPORT.md`
- `target/validation/w8a16-captured-triangle-20261003/SUMMARY.json`

Conclusion: M1 serial is numerically worse on every tested real projection;
pairwise/two-bank changes have mixed results. No production change is justified.
All captured full40 cases have M<=5, so WMMA cannot cause their observed drift.
