# Historical W8A16 causal diagnosis (workflow retired)

This records the 2026-10-03 investigation, not an executable current runbook.
The capture/cause-gate orchestration and exporter module-tracing/replay modes
are retired. Existing references, plans, captures and failure receipts under
`target/validation` remain immutable. The retained full-model exporter is
documented in [the reference contract](../../../../scripts/qwen35_fp8_cuda_reference.md);
it does not reproduce the retired internal-module capture workflow.

This is additive diagnosis on the existing `numeric::w8a16_profile` full40 tests.
It does not change production arithmetic, precision, fallback, or acceptance.
Historical CPU failure and the independent CUDA failure remain failures at
`atol=rtol=2e-4`. A successful diagnostic export is not a successful model test.

## Evidence obtained without GPU execution (2026-10-03)

Reference: `target/validation/qwen35-35b-current/native-cuda-reference-owner/full40-attempt1`.
Manifest SHA256: `0af0d913acc4f1c36dad8b4a7f8f253d44fa1f8a6ba4bcc52edae7cd4f569841`.
Native source SHA256: `b3001ab72dbddce268207a5771365e09961505e5e9fd092244fa17d362f57da5`.
It contains 654 tensors per case, but no internal module input/output tensors.
Its cuBLAS reductions are NOT assumed to equal the production vector reduction.

Inputs: existing `w8a16-first-divergence-20261002/gpu-trace/{hello,capital}`.
Outputs: `target/validation/w8a16-causal-device-20261003/{hello,capital}-final.json` (historical first-four diagnostic).
These identify the old captured production execution, not a new current binary.

- 216 hello and 624 capital same-input checks: zero strict failed checks.
- Includes numeric/nonquantized linear, RMS/head norm, residual, SwiGLU, shared
  gate, routing, routed combine, native GDN conv/chunk/norm/gate, and layer-3
  RoPE/causal GQA/sigmoid gate. Checkpoint weights/scales are payload-hash checked.
- Both cases: routed combine is bit-exact under production route-slot order.
  HF expert-ID order is different. Changing order on these inputs is at most
  1.12e-8 (hello), 7.45e-9 (capital); capital layer 1 crosses one BF16 bin.
  This is a real order difference, not evidence of wrong routing or double weighting.
- Layer-3 attention same-input max error: hello 1.49e-8, capital 1.062e-7.
  Physical KV values are NOT exposed by `CudaDiagnosticEvent`. Reconstructed
  RoPE K/V are explicitly labelled and are not certified physical cache contents.
- Layer-1 QKV amplification relative to CUDA cache (raw pre-convolution QKV):

| Case | Changed BF16 inputs | Original max error / outside | After subtracting operand delta |
|---|---:|---:|---:|
| hello | 69 | 0.00242424 / 2169 | 3.41714e-6 / 0 |
| capital | 155 | 0.00134850 / 558 | 3.77186e-6 / 0 |

The operand delta uses F64 `W*(RNE(x_production)-RNE(x_native_reconstructed))`.
Native input RMS is reconstructed on CPU from the *captured CUDA layer hidden*,
not claimed to be the unrecorded native CUDA norm output. Capital compares the
last four QKV rows retained by the convolution cache, not all five rows. This
strongly isolates operand-rounding amplification, but does not identify which
first-layer primitive supplies the first native-CUDA ULP difference.

The regression `test_operator_bound_does_not_imply_end_to_end_tolerance` gives a
constructive counterexample: identical BF16 inputs, upstream outputs 1.00390625
and its next F32 value both satisfy the existing operator bound; BF16-RNE then a
weight 2 yields outputs 2 and 2.015625, exceeding 0.0006. Thus operator bounds
cannot guarantee the end-to-end threshold across arbitrary legal reduction
orders. This does NOT prove this checkpoint cannot meet the fixed threshold.
No threshold is relaxed, nor is operator PASS or argmax promoted to quality PASS.

## Scope of the archived diagnosis

The historical capture covered operator events, router IDs/weights, final norm,
logits and GDN snapshots. Trace completion was not numerical acceptance. Native
module comparisons distinguished identical rounded operands from propagated-input
differences; production-order replay was diagnostic, not an independent oracle.

The separate model-test analyzer can still consume archived native module traces.
Its full40 trace requirement refers to those historical inputs; the retained
exporter no longer creates them. This cleanup does not change that analyzer.

Physical Ferrule KV was not exported through the diagnostic seam. Reconstructed
RoPE K/V and decode replay using reconstructed prefill state remain conditional,
not certified physical-cache observations. Neither a complete event inventory
nor a successful replay closes that evidence gap.

The prepared plan remains at
`target/validation/w8a16-next-cause-gate-20261003-v3/plan.json`, with its original
source/binary/reference pins. It is historical evidence, not a launch entry point.
No archived plan should be edited to match current source or relabeled as current
validation. Old CPU/CUDA failures remain failures at their original thresholds.

The retained reference's runtime RoPE audit requires F32 `inv_freq` and
`original_inv_freq` exactly equal to fresh native F32 construction. Its tests
reject both BF16 frequencies and BF16-then-F32 precision loss. This invariant is
independent of the retired tracing workflow; no model arithmetic was changed.
