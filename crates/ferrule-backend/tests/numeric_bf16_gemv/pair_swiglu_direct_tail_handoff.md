# Pair-SwiGLU direct-tail gate — additive test freeze

**GPU numeric gate NOT RUN. This is a native formula regression oracle, not a
Hugging Face oracle and not an extension of the finite HF contract to nonfinite
inputs.** Independent HF8 fixtures will come from the separate oracle work;
none were sourced or claimed here.

This handoff supplements, and does not overwrite, `pair_swiglu_handoff.md` or
`pair_swiglu.sha256`. The prior receipt and every one of its eight file hashes
still verify. In particular the production arithmetic, split/new kernels,
Rust provider, API and previous tests are byte-identical to that receipt.

## Why direct tail, not another GEMM fixture

GEMM can normalize signed zero, flush subnormals, and turn `0 * Inf` into NaN
before the tail receives an operand. The new test uploads packed integer
BF16 `[M,2N]` directly and bypasses GEMM entirely. It runs M1/M2/M5, N256;
K is deliberately absent because no GEMM/decode is run.

The one opt-in FFI test entry runs, on the caller's stream:

1. Original production `pair_tail<false>` / `split_pair` into exact gate/up
   buffers. Rust asserts both outputs equal **every uploaded word**, including
   signed zeros and signaling-NaN payloads.
2. Test-only `old_literal` on the split buffers. Its frozen expression is
   `silu = v / (1.0f + expf(-v)); activated = BF16_RNE(silu);`
   `result = BF16_RNE(activated * up)`. It uses local bit expansion and an
   independent residue/tie-even integer conversion, **not any production BF16
   arithmetic/conversion helper**. Both raw F32 stages are returned as `u32`.
3. Actual production `pair_tail<true>` / fused tail on the same packed input.
4. Rust also invokes current `swiglu_bf16` on the split buffers and compares it
   with the independently frozen literal. It is a subject, not the oracle.

No production arithmetic was copied into a new production path. The frozen
literal lives only in `pair_swiglu_tail_oracle.cuh` under the tests directory.
Do not update this reference in lockstep with production.

## Explicit word coverage and assertions

Each row covers all 249 cases (24 fixed-word cases + a 15x15 Cartesian product),
with a per-row cyclic permutation; remaining columns repeat cases.

- Gate and up axes include `+0`, `-0`, signed subnormals, the normal transition,
  positive/negative finite values, `+Inf`, `-Inf`, signed qNaN `7fc1/ffc1`, and
  signed sNaN `7f81/ff81`.
- 24 independently fixed output words pin signed-zero sign propagation,
  subnormal SiLU halves, normal/subnormal transitions, positive/negative
  product midpoint rounding, negative-gate saturation and defined Inf cases.
- **Both `0x0000` and `0x8000` must actually occur in fused GPU output** for
  every M. Input presence alone is not accepted as proof.
- Raw SiLU midpoint words are pinned to `00008000/80008000` and
  `00018000/80018000` (tie down/up, both signs).
- Raw product midpoint words are pinned to `43418000/c3418000` and
  `43448000/c3448000` (129/131 times +/-1.5). Expected BF16 words are
  `4342/c342` and `4344/c344` respectively.
- All three outputs (frozen literal, original boundary API, new tail) are
  compared as `u16`, **not float equality, tolerances, or NaN-equals-NaN**.
  Invalid/NaN outputs use the frozen CUDA expression's exact word as expected;
  additional assertions require a quiet NaN where mathematically applicable.
  These payload/sign expectations are CUDA/libdevice regression behavior,
  not a framework-independent or HF nonfinite contract.
- Input words remain unchanged. Raw product words also undergo independent
  host integer RNE checks against reference output.

The CPU test checks fixture construction, the Cartesian coverage, fixed
integer converter tie/quiet-bit vectors, and packed row layout. **CPU success
or compile/link success does not prove GPU `-0` presence or numeric equality.**
Only the ignored GPU gate below executes those assertions.

## Test-only wiring and source delta

Only three existing files changed for this follow-up:

| File | Additive change |
| --- | --- |
| `build.rs` | One `rerun-if-changed` line for the test oracle header |
| `native/cuda/core/provider.cu` | Guarded include after existing provider bodies |
| `tests/numeric_bf16_gemv.rs` | Include the dedicated test module |

Two test implementation files were added:
`tests/numeric_bf16_gemv/pair_swiglu_direct_tail.rs` and
`tests/numeric_bf16_gemv/pair_swiglu_tail_oracle.cuh`.
The native header is included in **core** provider's TU, where the existing
GemmEx tail is compiled, not a duplicate TU or new production export.

The hook `ferrule_test_pair_swiglu_tails` exists only with the existing opt-in
`FERRULE_CUDA_TEST_ORACLE=1`; Rust FFI/GPU test additionally requires
`feature="cuda"` and `ferrule_cuda_test_oracle`. It is not in production ABI
headers or public Rust APIs. There is one new opt-in C hook, no handle,
allocation, GEMM, event or synchronization inside it. Test allocations and
completion proofs are owned by the harness.

Before/after SHA256 of the existing files (repository-relative paths prefixed
with `crates/ferrule-backend/`):

```text
build.rs
  before 8bc5a6aef61893e38526f540ceccaf884b5a54e30ec2b255006b4d53116251d4
  after  7c820c1925afbf32ee394883db1a1dd2567e715f6c66e3dfafcc63fcf34d9ed5
native/cuda/core/provider.cu
  before c5aba90508b6782eb98fff88795e16aca0a36845fff8e6ddeb1a407024ce41ac
  after  49ee85e75601cba95dfc111efd37e16f266f481bf0593bae6414bba35fb0cb6c
 tests/numeric_bf16_gemv.rs
  before 901902aef711e63fe8fbd3e7771c24abb37a64aa66b742b3470d7c80dcc717bd
  after  a12b41388707ace8ea6c9abdeeaecbc02531fa35792cb18872562e8404257fd8
```

Unchanged prior receipt SHA256:
`5e16d782e1781e86e06e0aa2743a7392f6538aab893250af9accf19b5bb8ae02`.
New, separate receipt: `pair_swiglu_direct_tail.sha256`, covering the original
freeze plus this additive test/hook/handoff. Neither receipt freezes unrelated
whole-repository work. No model/runtime/composition/paper edits.

## GPU-hidden validation (2026-10-11)

- CPU `cargo test -p ferrule-backend --test numeric_bf16_gemv`: **2 passed**.
- CUDA12.4.99/sm86, oracle opt-in: native compile + Rust link passed;
  same test target: **2 host tests passed, 1 GPU test ignored**.
- Ordinary CUDA build (`FERRULE_CUDA_TEST_ORACLE=0`): compiled and linked;
  **2 host tests passed**; direct-tail GPU test intentionally absent.
- CPU and oracle-enabled CUDA clippy `--lib --tests -- -D warnings`: passed.
- `nm --defined-only` on compiled core archives: **one** direct-tail test
  hook with opt-in, **zero** without opt-in.
- Compiled oracle-enabled binary `--list` confirms the exact selector below.
- Scoped rustfmt, whitespace, original receipt verification: passed.
- **GPU direct-tail execution, GPU word equality and HF fixture comparison:
  NOT RUN.** No GPU was used, including for capability checks.

Build/host validation used `CUDA_VISIBLE_DEVICES=''`,
`NVIDIA_VISIBLE_DEVICES=void`, `FERRULE_CUDA_ARCH=sm_86`,
`CUBLAS_WORKSPACE_CONFIG=:0:0` and these pinned toolchain settings:

```sh
export CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao
export NVCC=/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc
export PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64
export RUSTFLAGS='-L native=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-35b-current/toolchain/link -L native=/home/admin/wxw403883/anaconda3/envs/tao/lib'
export CARGO_TARGET_DIR=target/pair-swiglu-direct-tail-cuda124
export FERRULE_CUDA_ARCH=sm_86
export CUBLAS_WORKSPACE_CONFIG=:0:0
export FERRULE_CUDA_TEST_ORACLE=1
cargo test -p ferrule-backend --features cuda --test numeric_bf16_gemv -- --test-threads=1
cargo clippy -p ferrule-backend --features cuda --lib --tests -- -D warnings
```

## GPU validator: new minimal gate

Obtain sole-owner authorization and set authorized GPU visibility using the
existing custody procedure (not the GPU-hidden visibility used above).
Use the pinned environment above **including `FERRULE_CUDA_TEST_ORACLE=1`**.
Verify both receipts from repository root:

```sh
sha256sum -c crates/ferrule-backend/tests/numeric_bf16_gemv/pair_swiglu.sha256
sha256sum -c crates/ferrule-backend/tests/numeric_bf16_gemv/pair_swiglu_direct_tail.sha256
cargo test -p ferrule-backend --features cuda --test numeric_bf16_gemv -- --list
```

Require the following exact selector in `--list`. A zero-test run with the
oracle flag omitted is **not** a pass. Execute only this one GPU gate:

```sh
FERRULE_CUDA_TEST_ORACLE=1 cargo test -p ferrule-backend --features cuda --test numeric_bf16_gemv pair_swiglu_direct_tail::gpu::direct_tail_words_old_literal_vs_fused_m1_m2_m5 -- --exact --ignored --test-threads=1 --nocapture
```

This gate is independent of the earlier ignored GEMM tests. Passing it would
establish direct-tail equality to this frozen native reference and the fixed
words; it would not establish HF equivalence, distributed TP2 correctness,
GEMM equivalence, or performance.
