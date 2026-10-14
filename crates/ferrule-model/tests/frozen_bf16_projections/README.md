# Frozen BF16 numeric projection matrix

Local arithmetic compatibility diagnostic, **not model acceptance**. It reuses
`CudaOperators`, `NumericFp8LinearPlan`, the existing BF16 conversion/resource
counters, and the backend QKV extractor's CPU safetensors utilities. No new CUDA
kernel, production dispatch, native HF execution or oracle synthesis.

## Frozen coverage

All rows are M=1; both captured layers are GDN, not GQA.

| Role | N x K | Trace count | Independent HF pairs | 8 MiB tile rows |
|---|---:|---:|---:|---:|
| LinearAttentionQkv | 8192 x 2048 | 2 | 2 | 2047 |
| LinearAttentionZ | 4096 x 2048 | 2 | 2 | 2047 |
| AttentionOutput (GDN out_proj) | 2048 x 4096 | 2 | 2 | 1023 |
| SharedExpertGate / Up | 512 x 2048 | 4 | 4 | 512 |
| SharedExpertDown | 2048 x 512 | 2 | 2 | 2048 |
| RoutedExpertGate / Up | 512 x 2048 | 32 | 0 | 512 |
| RoutedExpertDown | 2048 x 512 | 16 | 0 | 2048 |

The 48 routed projections have production descriptors but no independent
per-expert intermediate outputs. Whole `mlp.experts.output` is **not** such an
oracle; routed down also lacks an independent per-expert activation. Gate/up may
be packed in the native call, so shared projection shape does not prove expert
arithmetic compatibility. GQA, M>1 and native BF16 A/B/router/scalar gate/head
are outside this matrix. The router HF hook output is probabilities, not logits.

Only layer0 QKV/Z/out_proj production inputs match original HF inputs. All 12
local cases instead feed **original HF inputs**, including layer1 and shared
projections. No production output is imported into the fixture.

The existing sliced trace's first differing same-input boundary is layer0
out_proj: 5/2048 unequal BF16 words, max abs 0.0001220703125, zero outside the
existing tiny bound. Valid rounding differences alone do not prove a defect or
explain full40 drift. Native kernel identities are captured for QKV and Z only;
out/shared/routed native selection remains unknown.

## Five choices per pair

1. Fused `W8A16Bf16F32Accumulate`, zero scratch (control).
2. Tiled `Bf16RneF32Accumulate`, tile257 scratch.
3. `Bf16SlicedK2F32Accumulate`, tile257 scratch.
4. Tiled, actual 8 MiB model budget.
5. Sliced-K2, same 8 MiB budget.

These are 60 comparisons across 12 independent pairs. All five observed shape
families admit tiled/sliced plans; admission is **not** arithmetic equivalence.
Scratch is `(1 + tile_rows) * padded_K * 2`; only one workspace is live, fenced
before retirement. Grants, launches, zero submission allocations and zero
submission stream-wide synchronizations are checked. Each submission/comparison
is durably written before the final assertion; numerical differences do not
short-circuit remaining rows. The test asserts exactness of the **currently
selected model schedule** for the 12 pairs, not of the fused control or all
alternatives. Alternative exact matches are reported, not installed into dispatch.
The unchanged tiny bound (`atol=0.00025`, `rtol=0.012`) is diagnostic only.

## CPU extraction and validation

Run from the repository root. Extraction uses fresh output only, with pinned
capture/profile/provenance SHAs and independently pinned checkpoint payloads.
It does not import Torch or dequantize weights. Existing fixture need not be
regenerated; expected tensors remain original HF BF16 words.

```sh
/opt/conda310/bin/python -B -m unittest discover \
  -s crates/ferrule-model/tests/frozen_bf16_projections -p test_prepare.py -v
# Only if the output directory does not already exist:
/opt/conda310/bin/python -B crates/ferrule-model/tests/frozen_bf16_projections/prepare.py \
  --output target/validation/bf16-frozen-projections-20261005/fixture
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES=''
export FERRULE_FROZEN_BF16_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/validation/bf16-frozen-projections-20261005/fixture
cargo test -p ferrule-model --features cuda --test frozen_bf16_projections
cargo test -p ferrule-model --features cuda --test frozen_bf16_projections \
  -- --ignored --exact frozen_fixture_integrity
cargo clippy -p ferrule-model --features cuda --test frozen_bf16_projections --no-deps -- -D warnings
```

Fixture SHA256SUMS pin:
`59684d1793711f31dd168c5a9154f93a1cddebfeb292ba11c6dcdead62b5bff8`.
Pinned Torch `2.13.0+cu129`, Transformers `5.5.3`; TF32 and BF16 reduced-precision
reduction disabled in the independent reference. The separate profile manifest
pin remains `351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9`.

## Next GPU-owner action (not executed during preparation)

Run only the local matrix, **not full40**, after ownership is explicitly granted.
The prepared `ready.json` records the exact compiled binary, hashes, environment
and test arguments. This command reuses the existing authenticated supervisor;
it does not add another runner. Refuse changed sources/binary and existing logs.

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES=0
/opt/conda310/bin/python -B - <<'PY'
import hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
from test_vllm_qwen35 import run_process
base = Path('target/validation/bf16-frozen-projections-20261005')
ready = json.loads((base / 'ready.json').read_text())
for filename, expected in ready['sha256'].items():
    h = hashlib.sha256()
    with Path(filename).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    assert h.hexdigest() == expected, filename
os.environ.update(ready['env'])
report = Path(ready['env']['FERRULE_FROZEN_BF16_REPORT'])
assert not report.exists(), report
with (base / 'operator-supervisor.json').open('x') as evidence:
    result = run_process(ready['argv'], ready['timeout_seconds'])
    json.dump(result, evidence, indent=2)
    evidence.flush()
    os.fsync(evidence.fileno())
assert result.get('handshake', {}).get('containment_verified') is True
assert result.get('custody', {}).get('status') == 'proven_drained'
assert not result.get('cleanup_errors') and not result.get('error')
raise SystemExit(0 if result['returncode'] == 0 and not result['timed_out'] else 1)
PY
```

The owner must verify GPU0–7 quiescence and release ownership afterward. If
out_proj/shared have no exact candidate, next evidence is a **bounded native HF
F.linear kernel/layout capture** on these frozen inputs/weights. Expert coverage
requires independently observing the actual native packed gate/up/down calls;
never substitute production outputs, regenerate full references, or infer a
universal dispatch rule from shared shapes. No full40 until coverage improves.

## Previously added sliced model API (unchanged in this phase)

- New enum variant: `NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate`.
- CPU-visible `NumericFp8Precision::linear_plan(&TensorRole, NumericFp8Layout,
  rows, scratch_bytes) -> Result<NumericFp8LinearPlan>`.
- New `NumericFp8Bf16HybridCudaProfile::SLICED_K2_NAME`:
  `cuda-hybrid-numeric-fp8-model-bf16-sliced-k2-f32-ssm`.
- Standard identity:
  `cuda-standard-numeric-fp8-model-bf16-sliced-k2-f32-accumulate`.

`LinearAttentionZ && M == 1 && K % 64 == 0` selects backend sliced-K2. Every
other numeric projection, M>1 or non-K64 Z uses tiled. Invalid layouts/budgets
still reject. No layer-index rule; native BF16 parameters bypass numeric dispatch.
Existing explicit hooks accept this enum: `open_hf_with_precision`,
`prepare_numeric_fp8_bf16_serving_with_precision`,
`with_hybrid_cuda_numeric_fp8_bf16_precision`, `new_numeric_fp8_with_precision`.
They were not newly introduced by sliced integration. Factory use requires
explicit BF16 model precision and `E4M3FnBf16Scales`; legacy F32/W8 and expert-wire
model-BF16 serialization remain rejected. Fused BF16 factory defaults, enum
default, old tiled/W8 profiles and reference artifacts are unchanged.

Implementation files from that phase:
- `crates/ferrule-model/src/transformer/standard/numeric_precision.rs`
- `crates/ferrule-model/src/transformer/standard/bf16_hybrid_profile.rs`
- `crates/ferrule-model/src/transformer/standard/cuda/binding.rs`
- `crates/ferrule-model/src/transformer/standard/cuda/numeric_binding.rs`
- `crates/ferrule-model/src/decoder/hybrid_memory_plan.rs`
- `crates/ferrule-runtime/src/parallel/expert.rs`

`NumericBinding` retains the prepared semantic role. Workspace replacement also
checks the precision tag, fences retirement, and retains one bounded reservation.
The existing factory hook lives in
`crates/ferrule-composition/src/model_factory/bf16_serving.rs`; the existing
runner option lives in `crates/ferrule-model/src/decoder/runner.rs`.
