# Bounded pair-SwiGLU — backend handoff

Implementation complete; **GPU numeric equivalence and performance are not yet
validated**. No model/runtime/composition/paper integration is included.

## Public API

Types are exported from `ferrule_backend::cuda::operators::linear`.

```rust
impl CudaBf16GemmEx {
    pub fn execute_pair_swiglu_view_into(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weights: [&CudaNumericFp8Artifact; 2], // gate, up
        input: &CudaBf16Buffer,
        activation: &mut CudaBf16Buffer,
    ) -> ferrule_common::Result<()>;

    pub fn execute_bf16_pair_swiglu_view_into(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        activation: &mut CudaBf16Buffer,
        layout: Bf16GemmLayout,
    ) -> ferrule_common::Result<()>;
}
```

- Pass the **existing prepared full `2N` plan**. This is not a half-width GEMM.
  Same supported numeric geometry: M1–32, N and K multiples of 128, zero
  origins, BF16 scales; no new admission policy. Tests target M1/M2/M5,
  rank-local **N256/K2048**, i.e. plan width 512.
- `input` is exactly `M*K` BF16 words; `activation` exactly `M*N` words.
  Resident `weight` is exactly `2*N*K`, ordered gate rows then up rows;
  `layout = Bf16GemmLayout::contiguous(M, 2*N, K)` describes the GEMM.
  Short/oversized tensors, strides, foreign contexts/stream owners, unprepared
  views, output/read overlap and provider-storage overlap are rejected.
- Resident input can be the caller's contiguous pair cache, populated by
  `decode_numeric_bf16_pair_into`. Cold entry takes both original immutable
  FP8 artifacts; it keeps the original pack and two native decodes. Cache
  selection/fallback remains caller policy, not a new automatic backend path.
- Both entries retain the same BF16 packed `[M,2N]` result in the original
  provider workspace. Only the tail changes: one kernel loads gate/up from
  that result and writes activation. No intermediate output views required.
- Exact arithmetic is shared with `bf16_boundary` kind 4 in
  `core/device.cuh::bf16_swiglu`:
  `v = bf16_value(gate); activated = bf16_round(v / (1.0f + expf(-v)));`
  `out = bf16_rne(activated * bf16_value(up));`
  This preserves both BF16-RNE boundaries, libdevice `expf`, division and NaN
  handling. No fast-exp/sigmoid substitute, single-rounding fusion, W8 arithmetic
  or clipping is introduced. “Saturation” tests exercise SiLU/exp behavior, not
  a new saturating arithmetic mode.
- Native pair launch/preflight and tail dispatch are shared by old split and
  new fused entries. `cublasGemmEx` geometry, flags, compute type, algorithm,
  decoded weight arithmetic and M1/M5 packing are unchanged.
- No new handle, allocation, workspace binding, synchronization or event in
  execution. Existing provider storage/admission/grants remain unchanged.
  Counter units: cold fused 5, resident fused 2, versus original pair plus
  standalone SwiGLU 6/3. Each replaces one boundary kernel and avoids two
  caller gate/up buffers; this is not a measured latency or speedup claim.
- Eager-only. Caller retains weights, input, activation and owner through the
  last consumer. Submission failure uses the original sticky owner poison +
  context Unknown path; caller must retain/quarantine its operands on Unknown.
  No new capture API or recovery semantics. Activation/cache allocation remains
  caller-accounted and outside the provider grant.

Example (buffers and full-2N view are already admitted):

```rust
owner.execute_bf16_pair_swiglu_view_into(
    op, full_2n_plan, resident_gate_up, input, activation,
    Bf16GemmLayout::contiguous(rows, 2 * n, k),
)?;
// Or select the original encoded sources when no resident pair is available:
owner.execute_pair_swiglu_view_into(
    op, full_2n_plan, [gate_artifact, up_artifact], input, activation,
)?;
// Do not call swiglu_bf16 again: activation is already the two-rounding result.
```

## Tests and CPU-only validation

`pair_swiglu.rs` is a private provider test module, so host validation and GPU
selectors use `--lib`, not `--test numeric_bf16_gemv`.

- Host-only native preflight: M1/M2/M5, null/alignment/overflow/geometry,
  output/source/workspace aliases, workspace underbudget, output ABI agreement;
  original split still rejects a shared gate/up destination.
- Host-only fixture admission checks both rank payloads and BF16 scale encoding.
- GPU rank-pair oracle: two distinct rank-local pairs, M1/M2/M5 N256/K2048;
  original cold pair + original SwiGLU vs cold fused, resident unfused and
  resident fused, every BF16 word. Repeat resident after encoded source
  retirement. Check no execution allocation/copy/fence and launch counters.
- GPU edge oracle: one-hot GEMM inputs expose BF16 saturation, signed-zero
  inputs, subnormal/normal transitions, adjacent BF16 values and product
  overflow; compare original resident pair + original SwiGLU word-for-word.
  GEMM may normalize signed zeros/flush subnormals; this tests equality of the
  actual GemmEx path, not preservation of every source bit through GEMM.
- GPU rejection/Unknown: exact extents, layouts, owner/context, alias, grants,
  capture, preflight without poison, poisoned owner and the common native-error
  transition. Unknown retains caller custody; these are not GPU-passed claims.

Validation environment: GPU hidden (`CUDA_VISIBLE_DEVICES=''`,
`NVIDIA_VISIBLE_DEVICES=void`), CUDA 12.4.99/sm86, existing `tao` installation.
Final GPU-hidden results (2026-10-11):

| Check | Result |
| --- | --- |
| CPU `bf16_gemmex`, `numeric_bf16_gemv`, `numeric_fp8_payload`, `numeric_w8a16` | 9 passed |
| CUDA12.4.99/sm86 backend library native compile + Rust link | Passed |
| `--lib pair_swiglu_tests -- --test-threads=1` | 3 host tests passed, 3 GPU tests ignored |
| Existing `mixed_decode_reuse_tests --test-threads=1` | 3 host tests passed, 2 GPU tests ignored |
| CPU and CUDA clippy, `--lib --tests -- -D warnings` | Passed |
| Scoped rustfmt / whitespace checks | Passed |
| GPU numeric equivalence, GPU rejection/Unknown, timing | **Not run** |

CPU checks used:

```sh
cargo test -p ferrule-backend --test bf16_gemmex --test numeric_bf16_gemv --test numeric_fp8_payload --test numeric_w8a16
cargo clippy -p ferrule-backend --lib --tests -- -D warnings
```

CUDA build/host-test environment (paths verified locally):

```sh
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=void
export CUBLAS_WORKSPACE_CONFIG=:0:0
export FERRULE_CUDA_ARCH=sm_86
export CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao
export NVCC=/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc
export PATH=/home/admin/wxw403883/anaconda3/envs/tao/bin:/root/.cargo/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64
export RUSTFLAGS='-L native=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-35b-current/toolchain/link -L native=/home/admin/wxw403883/anaconda3/envs/tao/lib'
export CARGO_TARGET_DIR=target/pair-swiglu-cuda124
cargo test -p ferrule-backend --features cuda --lib pair_swiglu_tests -- --test-threads=1
cargo clippy -p ferrule-backend --features cuda --lib --tests -- -D warnings
```

The compiled binary's `--list` confirmed all exact selectors below. Source and
handoff freeze manifest: [pair_swiglu.sha256](pair_swiglu.sha256); verify from
repository root with `sha256sum -c
crates/ferrule-backend/tests/numeric_bf16_gemv/pair_swiglu.sha256` before GPU work.
The manifest covers this backend change, not a frozen whole-repository snapshot.

## Exact GPU selectors — do not run without sole-owner authorization

Build/run with the existing pinned environment, **not the GPU-hidden validation
visibility settings**. Select the authorized device using the established GPU
custody procedure. Keep `CUBLAS_WORKSPACE_CONFIG=:0:0` from process startup,
CUDA12.4/sm86 and cuBLAS 120405. Do not run these in parallel.

```sh
cargo test -p ferrule-backend --features cuda --lib cuda::providers::bf16_gemmex::pair_swiglu_tests::pair_swiglu_rankpair_m1_m2_m5_exact -- --exact --ignored --test-threads=1 --nocapture
cargo test -p ferrule-backend --features cuda --lib cuda::providers::bf16_gemmex::pair_swiglu_tests::pair_swiglu_bf16_edges_m1_m2_m5_exact -- --exact --ignored --test-threads=1 --nocapture
cargo test -p ferrule-backend --features cuda --lib cuda::providers::bf16_gemmex::pair_swiglu_tests::pair_swiglu_validation_and_poison -- --exact --ignored --test-threads=1 --nocapture
```

The two rank fixtures are backend-local, **not a distributed TP2/model test**.
No official QA8 replay, numeric GPU equivalence, kernel trace or performance
measurement was run as part of this GPU-hidden implementation.
