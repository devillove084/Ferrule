//! Numeric FP8 is compressed storage; arithmetic is an explicit image policy.

/// Immutable arithmetic selection for a prepared numeric FP8 CUDA image.
/// Available without CUDA so factories can describe and validate configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumericFp8Precision {
    /// Round activation and decoded weights to BF16, accumulate/output F32.
    #[default]
    Bf16RneF32Accumulate,
    /// Keep activation and decoded weights F32; use CUTLASS TF32x3.
    F32Tf32x3,
}

impl NumericFp8Precision {
    pub const fn standard_backend_name(self) -> &'static str {
        match self {
            Self::Bf16RneF32Accumulate => "cuda-standard-numeric-fp8-bf16-rne-f32-accumulate",
            Self::F32Tf32x3 => "cuda-standard-numeric-fp8-f32-tf32x3",
        }
    }

    pub const fn hybrid_backend_name(self) -> &'static str {
        match self {
            Self::Bf16RneF32Accumulate => "cuda-hybrid-numeric-fp8-bf16-rne-f32-accumulate",
            Self::F32Tf32x3 => "cuda-hybrid-numeric-fp8-f32-tf32x3",
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn backend(
        self,
    ) -> ferrule_backend::cuda::operators::linear::NumericFp8Precision {
        use ferrule_backend::cuda::operators::linear::NumericFp8Precision as Backend;
        match self {
            Self::Bf16RneF32Accumulate => Backend::Bf16RneF32Accumulate,
            Self::F32Tf32x3 => Backend::F32Tf32x3,
        }
    }
}

#[cfg(test)]
mod tests {
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
            .with_hybrid_cuda_numeric_fp8_precision(
                limits,
                64 << 20,
                NumericFp8Precision::F32Tf32x3,
            )
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
}
