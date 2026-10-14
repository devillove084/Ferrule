# Capital L0 frozen numeric projection matrix (2026-10-06)

## Bounded result — full layer schedule NOT established

Evidence: `target/validation/capital-projections-20261006/FINAL.json`.
This task ran no full-model forward, GDN, chunk backend, or full40. No production
arithmetic, backend admission, defaults, M1 policy, or Mixed dispatch changed.
The only repository edits are the dedicated Capital fixture/matrix/probe/tests
and their inclusion in the existing frozen-projection test target.

114 independent pairs cover every numeric-FP8 projection in the original
Capital layer0 trace: six dense M5 pairs plus108 routed projections from36
selected experts (32 M1 groups and4 M2 groups). Native BF16 A/B, router and
shared scalar gate are not numeric-FP8 projections and are outside this matrix.

The current Mixed selection is exact for96/114 cases. The supported fused,
tiled and sliced-K2 alternatives provide at least one exact result for107/114.
All570 comparisons preserved launch counts, allocation grants, owner workspace,
finite outputs and separate BF16 RNE writeback. Tile257 and production8MiB
budgets were compared. GEMV is correctly rejected by host admission for the
M5 and routed shapes; it was not silently forced through the production API.

## Provenance and independent supplemental capture

The unchanged original HF capture is
`bf16-layer01-supported-retry-20261005/hf-layer01.safetensors`, SHA256
`6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07`.
Its manifest's30 independent replay boundary checks remain exact. The six
existing dense inputs/outputs are extracted directly from those HF hooks, not
from divergent producer inputs in the current handoff.

Individual routed pairs were absent. `capital_prepare.py` constructs only the
original pinned HF `Qwen3_5MoeExperts` module with the original lazy checkpoint
operand binding, passes the retained HF expert input/IDs/routing weights, and
observes `F.linear` without substituting its result. The aggregate must equal
the retained `mlp.experts.output` bit-for-bit. It does. Gate/up are captured as
halves of the native concatenated gate-up projection, preserving native kernel
selection; down uses the actual native intermediate. Every FP8 and BF16 scale
payload is checked against original HF provenance hashes.

Supplemental fixture: `fixture-v2/`, SHA256SUMS pin
`0381178a03f7c848a73b153ddf3cec3284c316814ef3007f3f7a8043f96e984f`.
It is additive evidence, not a reference rewrite. All114 CUDA pack/decode
checks match HF: **248320 input words** and **149946368 decoded weight words**.

## Measured dense schedule (not installed)

| Role | M,N,K | Exact existing profile | Current tiled BF16 differences |
|---|---|---|---:|
| QKV | 5,8192,2048 | Sliced-K2 | 26 (Mixed already selects sliced) |
| Z | 5,4096,2048 | Tiled | 0 |
| Attention output | 5,2048,4096 | Sliced-K2 | 25 |
| Shared gate | 5,512,2048 | Fused W8A16 | 3 |
| Shared up | 5,512,2048 | Fused W8A16 | 1 |
| Shared down | 5,2048,512 | Tiled | 0 |

For M2 routed groups, fused gate/up and tiled down are exact across all4 experts.
This is evidence for potential explicit role/shape guards, NOT a universal
multirow dispatch rule. A partial Mixed change was deliberately not installed
because the requested entire-layer exact schedule is blocked by M1 routed cases
and existing M1/default behavior must be preserved.

## HF kernel and epilogue findings

Pinned HF remains Torch2.13.0+cu129/CUDA12.9, transformers5.5.3, source-verified,
TF32/reduced-precision BF16 reductions off. Ferrule runs CUDA12.4/sm86.

- M5 QKV: `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`.
- M5 Z: `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`.
- M5 attention output: `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`.
- M5 shared gate/up and M2 routed gate-up: TensorOp64x64 K32,6 stages plus
  cuBLASLt split-K reduction, complete names in the supplemental manifest.
- M5 shared down and M2 routed down: WMMA32x32 K128x2.
- Routed M1 gate-up/down: native `internal::gemvx::kernel` with BF16 operands
  and output, F32 accumulation, template configuration7; full name in manifest.

Native `mm(out_dtype=float32)` is diagnostic, **not an interchangeable oracle**.
For M5 attention output its BF16-rounded result differs in17 words from actual
HF BF16 `F.linear`. Sliced-K2 still matches all10240 original BF16 output words.
The exporter initially stopped on this unsupported epilogue-equivalence
assumption. The failed receipt is retained; the successful fresh export records
the difference instead of rewriting expected bytes or relaxing comparison.

## Concrete routed blockers

No supported alternative is exact for these7 M1 pairs:
`expert19_down`, `expert57_down`, `expert146_down`, `expert151_up`,
`expert226_down`, `expert231_up`, `expert238_down`.
All exact positions/raw and BF16 patterns are in `matrix.jsonl` and
`schedule-evidence.json`.

A bounded CUDA12.4.5 cuBLAS diagnostic compares TSS (F32 output) and TST (BF16
output), at tile257 and full output width, using decoded operands already
verified equal to HF. These are test-only unsupported-shape diagnostics, not
production admission changes. TSS and TST have identical BF16 mismatch counts.
Full-width GEMV fixes most earlier misses but is still not generally exact:

- `expert146_down`, flat index1620: existing fused BF16 `32e0`, HF `32e1`;
  neither GEMV geometry is exact (one word each).
- `expert238_down`: tiled GEMV one word, full GEMV three words differ.
- Full-width GEMV also introduces one difference for `expert200_down`, which
  existing fused already gets right. Choosing geometry per expert would be an
  invalid index-dependent correction, not a general arithmetic implementation.

Thus no safe general GEMV-admission widening or arithmetic fix is justified by
this bounded cycle. The full-layer exact goal remains unresolved and is not
reported as passed.

## Validation and next executable gate

Passed:3 Python CPU provenance/coverage/epilogue tests,2 Rust CPU admission and
coverage tests, CUDA12.4 build, Clippy with warnings denied, the570-comparison
owner matrix, full operand decode verification, and existing typed tiled graph
replay and owner/profile/budget rejection GPU regressions. Every successful GPU
run has fresh authenticated subreaper custody and idle-after evidence; failures
are retained separately. No full40 or lifecycle/unknown-completion semantics
changed.

Reusable files:
- `capital_prepare.py`: supplemental independent capture (explicit CUDA flag).
- `capital.rs`: frozen owner matrix, host coverage/admission checks.
- `capital_operands.cu`: decode equality and general TSS/TST GEMV diagnostic.
- `test_capital_prepare.py`: CPU integrity tests.

Exact Rust filter: `capital::frozen_capital_projection_matrix` in
`frozen_bf16_projections`. Variables: `FERRULE_CAPITAL_PROJECTION_DIR`,
`FERRULE_CAPITAL_PROJECTION_REPORT` (fresh path). Set
`FERRULE_CAPITAL_REQUIRE_SELECTED_EXACT=1` to enforce all114 selected cases;
the unchanged current schedule intentionally fails that stronger gate.

Plans/binaries are in `target/validation/capital-projections-20261006/`.
Use fresh labels/reports with its authenticated `run.py`; never reuse receipts.
`matrix-plan.json` and `operands-final-plan.json` are concrete bounded replay
commands. Do not run them without explicit GPU ownership.

The next bounded arithmetic gate, if authorized separately, is the retained
M1 routed bank (especially146/238/200 down) against the actual pinned native
GEMV reduction: distinguish CUDA12.9-vs12.4 dispatch/accumulator ordering on
identical operands before any provider policy or backend change. Do not
substitute a BF16 epilogue, choose per-expert tile sizes, or run full40 to find
the same defect again. Original supplemental fixtures and114-case acceptance
remain the required regression matrix for a general fix.
