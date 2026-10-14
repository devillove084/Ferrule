//! Fused W8A16 uses the independent Torch BF16-RNE operand/F32-boundary oracle,
//! not a BF16-output model or a Marlin bitwise reference. The shared full-runner
//! harness keeps the existing 2e-5 absolute / 3e-4 relative logit tolerances.
use super::*;

#[test]
fn w8a16_tiny_oracle_and_zero_workspace_admission_are_explicit() {
    let precision = NumericFp8Precision::W8A16Bf16F32Accumulate;
    let f = fixture::Fixture::with_precision(precision);
    let reference: Value =
        serde_json::from_str(include_str!("../fixtures/numeric_standard_oracle.json")).unwrap();
    let f32_reference: Value =
        serde_json::from_str(include_str!("../fixtures/numeric_standard_f32_oracle.json")).unwrap();
    assert_eq!(f.oracle, reference);
    assert_ne!(f.oracle["logits"], f32_reference["logits"]);
    let logits = f.oracle["logits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(floats)
        .collect::<Vec<_>>();
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(
        logits
            .iter()
            .any(|&v| half::bf16::from_f32(v).to_f32() != v),
        "the reference must retain F32 outputs, not BF16 output rounding"
    );
    let r = resources(&f);
    let opts = options(&r)
        .with_hybrid_cuda_numeric_fp8_precision(
            ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 8192,
            },
            0,
            precision,
        )
        .unwrap();
    assert_eq!(opts.precision(), ExecutionPrecisionPolicy::f32());
    assert_eq!(opts.hybrid_cuda_numeric_fp8_precision(), Some(precision));
    assert_eq!(opts.hybrid_cuda_numeric_fp8().unwrap().1, 0);
    HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
}

#[test]
#[ignore = "requires CUDA GPU; fused W8A16 against BF16-RNE operands/F32-boundary Torch oracle"]
fn hybrid_numeric_w8a16_tiny_moe_top2_cache1_full_runner_matches_torch() {
    // Three-row prefill, M1 decode, abort/retry, shared/routed experts, cache
    // eviction and sequence fork/release all run through the same harness.
    tiny_numeric_runner(NumericFp8Precision::W8A16Bf16F32Accumulate);
}

// Keep full40 strict acceptance independent of the tiny fixture and of F32.
// The CPU oracle rounds numeric operands only; CUDA reduction order is not emulated.
#[test]
#[ignore = "requires single CUDA GPU + independent BF16-RNE full40 oracle; strict all rows"]
fn qwen35_35b_numeric_w8a16_full40_hello_prefill_decode_matches_reference() {
    f32_profile::full40_case("hello", NumericFp8Precision::W8A16Bf16F32Accumulate);
}

#[test]
#[ignore = "requires single CUDA GPU + independent BF16-RNE full40 oracle; strict all rows"]
fn qwen35_35b_numeric_w8a16_full40_capital_prefill_decode_matches_reference() {
    f32_profile::full40_case("capital", NumericFp8Precision::W8A16Bf16F32Accumulate);
}
