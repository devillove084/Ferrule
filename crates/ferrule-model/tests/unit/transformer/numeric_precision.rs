use super::*;
use crate::decoder::GenericDecoderOptions;
use crate::execution::ExecutionPrecisionPolicy;
use crate::spec::{ModelFamily, WeightSource};
use crate::transformer::ExpertCacheLimits;
use ferrule_common::execution::{ExecutionCapabilities, KvBindingMode, LogitsRowPolicy};
use std::num::NonZeroU32;

fn options(precision: ExecutionPrecisionPolicy) -> GenericDecoderOptions {
    let width = NonZeroU32::new(11);
    GenericDecoderOptions::new(
        ModelFamily::Unknown("numeric-profile-test".into()),
        WeightSource::Safetensors,
        ExecutionCapabilities {
            max_batch_tokens: 8,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: 8,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: width,
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: width,
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::Any,
        },
        2,
        32,
        1 << 20,
        precision,
    )
}

#[test]
fn numeric_precision_default_is_bf16_and_f32_is_explicit_cpu_metadata() {
    let limits = ExpertCacheLimits {
        max_experts: 64,
        max_bytes: 320 << 20,
    };
    let old = options(ExecutionPrecisionPolicy::f32());
    assert_eq!(old.hybrid_cuda_numeric_fp8_precision(), None);
    assert_eq!(
        NumericFp8Precision::default(),
        NumericFp8Precision::Bf16RneF32Accumulate
    );
    let bf16 = old
        .clone()
        .with_hybrid_cuda_numeric_fp8(limits, 64 << 20)
        .unwrap();
    let f32 = old
        .with_hybrid_cuda_numeric_fp8_precision(limits, 64 << 20, NumericFp8Precision::F32Tf32x3)
        .unwrap();
    assert_eq!(
        bf16.hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::Bf16RneF32Accumulate)
    );
    assert_eq!(
        f32.hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::F32Tf32x3)
    );
    assert_eq!(
        bf16.hybrid_cuda_numeric_fp8(),
        f32.hybrid_cuda_numeric_fp8()
    );
    assert_eq!(f32.hybrid_cuda_numeric_fp8(), Some((limits, 64 << 20)));
    assert_ne!(
        bf16.hybrid_cuda_numeric_fp8_precision()
            .unwrap()
            .hybrid_backend_name(),
        f32.hybrid_cuda_numeric_fp8_precision()
            .unwrap()
            .hybrid_backend_name()
    );
}

#[test]
fn numeric_precision_rejects_invalid_limits_and_execution_policy() {
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        for (count, bytes, scratch) in [(0, 8192, 4096), (1, 4096, 4096), (1, 8192, 0)] {
            assert!(
                options(ExecutionPrecisionPolicy::f32())
                    .with_hybrid_cuda_numeric_fp8_precision(
                        ExpertCacheLimits {
                            max_experts: count,
                            max_bytes: bytes
                        },
                        scratch,
                        precision,
                    )
                    .is_err()
            );
        }
        assert!(
            options(ExecutionPrecisionPolicy::bf16_compatibility())
                .with_hybrid_cuda_numeric_fp8_precision(
                    ExpertCacheLimits {
                        max_experts: 1,
                        max_bytes: 8192
                    },
                    4096,
                    precision,
                )
                .is_err()
        );
    }
}

#[test]
fn w8a16_selection_is_explicit_with_zero_or_reserved_workspace() {
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 8192,
    };
    let w8a16 = NumericFp8Precision::W8A16Bf16F32Accumulate;
    for scratch in [0, 4096] {
        let selected = options(ExecutionPrecisionPolicy::f32())
            .with_hybrid_cuda_numeric_fp8_precision(limits, scratch, w8a16)
            .unwrap();
        assert_eq!(selected.precision(), ExecutionPrecisionPolicy::f32());
        assert_eq!(selected.hybrid_cuda_numeric_fp8_precision(), Some(w8a16));
        assert_eq!(selected.hybrid_cuda_numeric_fp8(), Some((limits, scratch)));
        assert!(
            options(ExecutionPrecisionPolicy::bf16_compatibility())
                .with_hybrid_cuda_numeric_fp8_precision(limits, scratch, w8a16)
                .is_err()
        );
    }
    for (count, bytes, scratch) in [(0, 8192, 0), (1, 0, 0), (1, 4096, 4096)] {
        assert!(
            options(ExecutionPrecisionPolicy::f32())
                .with_hybrid_cuda_numeric_fp8_precision(
                    ExpertCacheLimits {
                        max_experts: count,
                        max_bytes: bytes
                    },
                    scratch,
                    w8a16
                )
                .is_err()
        );
    }
    assert_eq!(
        w8a16.standard_backend_name(),
        "cuda-standard-numeric-fp8-w8a16-bf16-f32-accumulate"
    );
    assert_eq!(
        w8a16.hybrid_backend_name(),
        "cuda-hybrid-numeric-fp8-w8a16-bf16-f32-accumulate"
    );
    for legacy in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        assert_ne!(
            w8a16.standard_backend_name(),
            legacy.standard_backend_name()
        );
        assert_ne!(w8a16.hybrid_backend_name(), legacy.hybrid_backend_name());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_model_precision_selects_fused_zero_scratch_backend_for_every_bucket() {
    use ferrule_backend::cuda::operators::linear::{
        NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType, NumericFp8SelectedKernel,
    };
    for rows in [1, 3, 8, 9, 16, 33, 128] {
        for scale_type in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
            let layout = NumericFp8Layout {
                n: 129,
                k: 257,
                row_origin: 7,
                column_origin: 11,
                scale_type,
            };
            let plan = NumericFp8LinearPlan::new(
                layout,
                rows,
                0,
                NumericFp8Precision::W8A16Bf16F32Accumulate.backend(),
            )
            .unwrap();
            assert_eq!(
                plan.selected_kernel(),
                if rows <= 8 {
                    NumericFp8SelectedKernel::W8A16FusedBf16Vector
                } else {
                    NumericFp8SelectedKernel::W8A16FusedBf16Mma
                }
            );
            assert_eq!(plan.workspace_requirements().bytes, 0);
            assert_eq!(plan.workspace_allocation_grant_bytes().unwrap(), 0);
            assert_eq!(plan.activation_bytes(), 0);
            assert_eq!(plan.weight_tile_bytes(), 0);
            assert_eq!(plan.kernel_launches(), 1);
            assert_eq!(plan.layout(), layout);
        }
    }
}

#[test]
fn model_bf16_numeric_requires_separate_explicit_options() {
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 8192,
    };
    let p = NumericFp8Precision::Bf16ModelF32Accumulate;
    for policy in [
        ExecutionPrecisionPolicy::f32(),
        ExecutionPrecisionPolicy::bf16_compatibility(),
    ] {
        assert!(
            options(policy)
                .with_hybrid_cuda_numeric_fp8_precision(limits, 0, p)
                .is_err()
        );
    }
    assert!(
        options(ExecutionPrecisionPolicy::f32())
            .with_hybrid_cuda_numeric_fp8_bf16(limits, 0)
            .is_err()
    );
    let image = options(ExecutionPrecisionPolicy::bf16_compatibility())
        .with_hybrid_cuda_numeric_fp8_bf16(limits, 0)
        .unwrap();
    assert_eq!(image.hybrid_cuda_numeric_fp8_precision(), Some(p));
    assert_ne!(
        p.hybrid_backend_name(),
        NumericFp8Precision::W8A16Bf16F32Accumulate.hybrid_backend_name()
    );
    assert_eq!(
        p.backend(),
        ferrule_backend::cuda_plan::NumericFp8Precision::W8A16Bf16F32Accumulate
    );
}

#[path = "numeric_tiled_precision.rs"]
mod tiled;

#[path = "numeric_sliced_precision.rs"]
mod sliced;

#[path = "numeric_mixed_precision.rs"]
mod mixed;

#[test]
#[cfg(feature = "cuda")]
fn router_normalization_is_mixed_only() {
    use NumericFp8Precision::*;
    use ferrule_backend::cuda::operators::moe::RouterNormalization;
    assert_eq!(
        Bf16ModelMixedF32Accumulate.router_normalization(),
        RouterNormalization::FullSoftmaxThenSelectedRenormalization
    );
    for profile in [
        Bf16RneF32Accumulate,
        F32Tf32x3,
        W8A16Bf16F32Accumulate,
        Bf16ModelF32Accumulate,
        Bf16ModelTiledF32Accumulate,
        Bf16ModelSlicedK2F32Accumulate,
    ] {
        assert_eq!(
            profile.router_normalization(),
            RouterNormalization::SelectedSoftmax
        );
    }
}
