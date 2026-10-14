//! Numeric FP8 is compressed storage; arithmetic is an explicit image policy.

/// Immutable arithmetic selection for a prepared numeric FP8 CUDA image.
/// Available without CUDA so factories can describe and validate configuration.
#[doc = include_str!("../../../tests/compile/numeric_w8a16.md")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumericFp8Precision {
    /// Round activation and decoded weights to BF16, accumulate/output F32.
    #[default]
    Bf16RneF32Accumulate,
    /// Keep activation and decoded weights F32; use CUTLASS TF32x3.
    F32Tf32x3,
    /// Fused weight-only E4M3FN with BF16-RNE operands and F32 accumulation/output.
    /// Decode times the original BF16/F32 scale is separately rounded to F32
    /// before BF16 conversion. F32 activations are rounded inside operand tiles;
    /// model activation, KV and state boundaries remain F32. No global decoded
    /// weight/activation workspace, repacking, or bitwise Marlin equivalence.
    W8A16Bf16F32Accumulate,
    /// Numeric E4M3FN/BF16-scale storage with actual BF16 model/activation/KV
    /// boundaries, F32 accumulation and SSM. Not an expert-wire W8 profile.
    Bf16ModelF32Accumulate,
    /// BF16 model boundaries with bounded FP8-to-BF16 staging and native tiled
    /// TensorOp GEMM, F32 accumulation/SSM. No HF equivalence claim.
    Bf16ModelTiledF32Accumulate,
    /// Explicit heterogeneous model schedule: sliced-K2 only for the GDN Z
    /// gate at M=1/K64, existing tiled BF16 elsewhere and for M>1.
    Bf16ModelSlicedK2F32Accumulate,
    /// Explicit heterogeneous BF16-boundary schedule. Supported numeric
    /// attention/shared/dense projections and routed down use native BF16-output
    /// GemmEx; routed gate/up use one concatenated native-width GEMM. One
    /// image-owned prepared shape pool charges full decoded weights, 32 MiB
    /// workspace and 16 MiB external provider reserve. Resident BF16 routers and A/B projections
    /// reuse that pool directly. Output head and unsupported layouts
    /// keep the prior role/shape fallback schedule.
    /// Router normalization is full F32 softmax then selected tree renormalization.
    /// This is a numerical profile, not universal HF equivalence.
    /// Fresh GDN history and multi-row cohorts use bounded chunk arithmetic;
    /// one-row cohorts with previous history retain cached recurrence.
    Bf16ModelMixedF32Accumulate,
}

impl NumericFp8Precision {
    /// Explicit router arithmetic; all other profiles retain selected-only softmax.
    #[cfg(feature = "cuda")]
    pub(crate) const fn router_normalization(
        self,
    ) -> ferrule_backend::cuda::operators::moe::RouterNormalization {
        use ferrule_backend::cuda::operators::moe::RouterNormalization;
        if matches!(self, Self::Bf16ModelMixedF32Accumulate) {
            RouterNormalization::FullSoftmaxThenSelectedRenormalization
        } else {
            RouterNormalization::SelectedSoftmax
        }
    }

    /// Legacy expert boots default to F32, not the model's historical BF16 default.
    pub const fn expert_wire_default() -> Self {
        Self::F32Tf32x3
    }

    /// The expert wire deliberately exposes only the two supported profiles.
    pub fn serialize_expert_wire<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self {
            Self::F32Tf32x3 => serializer.serialize_str("f32"),
            Self::W8A16Bf16F32Accumulate => serializer.serialize_str("w8a16-bf16"),
            Self::Bf16RneF32Accumulate
            | Self::Bf16ModelF32Accumulate
            | Self::Bf16ModelTiledF32Accumulate
            | Self::Bf16ModelSlicedK2F32Accumulate
            | Self::Bf16ModelMixedF32Accumulate => Err(serde::ser::Error::custom(
                "Legacy BF16 arithmetic is not a numeric expert wire profile",
            )),
        }
    }

    pub fn deserialize_expert_wire<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let value = <String as serde::Deserialize>::deserialize(deserializer)?;
        match value.as_str() {
            "f32" => Ok(Self::F32Tf32x3),
            "w8a16-bf16" => Ok(Self::W8A16Bf16F32Accumulate),
            _ => Err(serde::de::Error::unknown_variant(
                &value,
                &["f32", "w8a16-bf16"],
            )),
        }
    }

    pub const fn standard_backend_name(self) -> &'static str {
        match self {
            Self::Bf16ModelTiledF32Accumulate => {
                "cuda-standard-numeric-fp8-model-bf16-tiled-f32-accumulate"
            }
            Self::Bf16ModelSlicedK2F32Accumulate => {
                "cuda-standard-numeric-fp8-model-bf16-sliced-k2-f32-accumulate"
            }
            Self::Bf16ModelMixedF32Accumulate => {
                "cuda-standard-numeric-fp8-model-bf16-mixed-f32-accumulate"
            }
            Self::Bf16ModelF32Accumulate => "cuda-standard-numeric-fp8-model-bf16-f32-accumulate",
            Self::Bf16RneF32Accumulate => "cuda-standard-numeric-fp8-bf16-rne-f32-accumulate",
            Self::F32Tf32x3 => "cuda-standard-numeric-fp8-f32-tf32x3",
            Self::W8A16Bf16F32Accumulate => "cuda-standard-numeric-fp8-w8a16-bf16-f32-accumulate",
        }
    }

    pub const fn hybrid_backend_name(self) -> &'static str {
        match self {
            Self::Bf16ModelTiledF32Accumulate => super::NumericFp8Bf16HybridCudaProfile::TILED_NAME,
            Self::Bf16ModelSlicedK2F32Accumulate => {
                super::NumericFp8Bf16HybridCudaProfile::SLICED_K2_NAME
            }
            Self::Bf16ModelMixedF32Accumulate => super::NumericFp8Bf16HybridCudaProfile::MIXED_NAME,
            Self::Bf16ModelF32Accumulate => "cuda-hybrid-numeric-fp8-model-bf16-f32-ssm",
            Self::Bf16RneF32Accumulate => "cuda-hybrid-numeric-fp8-bf16-rne-f32-accumulate",
            Self::F32Tf32x3 => "cuda-hybrid-numeric-fp8-f32-tf32x3",
            Self::W8A16Bf16F32Accumulate => "cuda-hybrid-numeric-fp8-w8a16-bf16-f32-accumulate",
        }
    }

    /// The fused W8A16 profile needs no global numeric workspace for any row
    /// count. F32 profiles retain their established scratch requirements.
    pub(crate) const fn allows_zero_scratch(self) -> bool {
        matches!(
            self,
            Self::W8A16Bf16F32Accumulate | Self::Bf16ModelF32Accumulate
        )
    }

    /// Model-boundary identity, independent of compressed checkpoint storage.
    pub const fn has_bf16_model_boundaries(self) -> bool {
        matches!(
            self,
            Self::Bf16ModelF32Accumulate
                | Self::Bf16ModelTiledF32Accumulate
                | Self::Bf16ModelSlicedK2F32Accumulate
                | Self::Bf16ModelMixedF32Accumulate
        )
    }

    /// F32-output fallback plan. BF16 Mixed model execution first checks
    /// `gemmex_plan` (or `gemmex_gate_up_plan` for routed pairs); this method
    /// retains the previous schedule for unsupported/native-F32 call sites.
    /// Fallback arithmetic is not an HF-equivalence assertion.
    /// CPU-visible semantic projection plan. The sliced-K2 model schedule is
    /// deliberately role/shape based: GDN Z at M1/K64 uses sliced-K2; all other
    /// roles, M>1, and non-K64 gates retain tiled BF16. Invalid layouts/budgets
    /// still fail admission; this is not a universal HF kernel-selection rule.
    pub fn linear_plan(
        self,
        role: &crate::support::TensorRole,
        layout: ferrule_backend::cuda_plan::NumericFp8Layout,
        rows: usize,
        scratch_bytes: usize,
    ) -> ferrule_common::Result<ferrule_backend::cuda_plan::NumericFp8LinearPlan> {
        ferrule_backend::cuda_plan::NumericFp8LinearPlan::new(
            layout,
            rows,
            scratch_bytes,
            self.backend_for_linear(role, rows, layout),
        )
    }

    /// Explicit native BF16-output path for the Mixed image. Unsupported
    /// geometry retains the previous schedule; admitted geometry must satisfy
    /// the separate full-matrix/provider budgets, never silently fall back.
    pub fn gemmex_plan(
        self,
        role: &crate::support::TensorRole,
        layout: ferrule_backend::cuda_plan::NumericFp8Layout,
        rows: usize,
        storage_bytes: usize,
        provider_bytes: usize,
    ) -> ferrule_common::Result<Option<ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan>>
    {
        use crate::support::TensorRole::*;
        let role_supported = matches!(
            role,
            AttentionQuery
                | AttentionKey
                | AttentionValue
                | AttentionOutput
                | LinearAttentionQkv
                | LinearAttentionZ
                | LinearAttentionBeta
                | LinearAttentionA
                | SharedExpertGate
                | SharedExpertUp
                | SharedExpertDown
                | SharedExpertOutputGate
                | DenseMlpGate
                | DenseMlpUp
                | DenseMlpDown
                | RoutedExpertDown
        );
        if !role_supported {
            return Ok(None);
        }
        self.gemmex_geometry(layout, rows, storage_bytes, provider_bytes)
    }

    /// Resident BF16 routing/control projections share the prepared image pool.
    /// FP8 scale-block geometry is irrelevant to already resident BF16 weights.
    pub fn resident_bf16_gemmex_plan(
        self,
        role: &crate::support::TensorRole,
        n: usize,
        k: usize,
        rows: usize,
        storage_bytes: usize,
        provider_bytes: usize,
    ) -> ferrule_common::Result<Option<ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan>>
    {
        use crate::support::TensorRole::{
            LinearAttentionA, LinearAttentionBeta, RouterLogits, SharedExpertOutputGate,
        };
        if self != Self::Bf16ModelMixedF32Accumulate
            || !matches!(
                role,
                RouterLogits | LinearAttentionA | LinearAttentionBeta | SharedExpertOutputGate
            )
            || !(1..=32).contains(&rows)
            || n == 0
            || k == 0
        {
            return Ok(None);
        }
        ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan::new_resident(
            n,
            k,
            rows,
            storage_bytes,
            provider_bytes,
        )
        .map(Some)
    }

    /// Routed gate/up are one native-width projection. Metadata binding checks
    /// establish their common expert/source identities before using this plan.
    pub fn gemmex_gate_up_plan(
        self,
        gate: ferrule_backend::cuda_plan::NumericFp8Layout,
        up: ferrule_backend::cuda_plan::NumericFp8Layout,
        rows: usize,
        storage_bytes: usize,
        provider_bytes: usize,
    ) -> ferrule_common::Result<Option<ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan>>
    {
        if gate != up || gate.n == 0 || !gate.n.is_multiple_of(128) {
            return Ok(None);
        }
        let n = gate
            .n
            .checked_mul(2)
            .ok_or_else(|| ferrule_common::Error::Internal {
                message: "Mixed GemmEx concatenated rows overflow".into(),
            })?;
        self.gemmex_geometry(
            ferrule_backend::cuda_plan::NumericFp8Layout { n, ..gate },
            rows,
            storage_bytes,
            provider_bytes,
        )
    }

    fn gemmex_geometry(
        self,
        layout: ferrule_backend::cuda_plan::NumericFp8Layout,
        rows: usize,
        storage_bytes: usize,
        provider_bytes: usize,
    ) -> ferrule_common::Result<Option<ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan>>
    {
        use ferrule_backend::cuda_plan::{NumericFp8ScaleType, bf16_gemmex::Bf16GemmExPlan};
        if self != Self::Bf16ModelMixedF32Accumulate
            || !(1..=32).contains(&rows)
            || layout.n == 0
            || layout.k == 0
            || !layout.n.is_multiple_of(128)
            || !layout.k.is_multiple_of(128)
            || layout.row_origin != 0
            || layout.column_origin != 0
            || layout.scale_type != NumericFp8ScaleType::Bf16
        {
            return Ok(None);
        }
        Bf16GemmExPlan::new(layout, rows, storage_bytes, provider_bytes).map(Some)
    }

    /// Numerical branch for a sequence cohort, not the packed batch or its
    /// scheduler phase. The caller must derive history from the validated,
    /// journal-owned GDN frontier, never allocation presence or state values.
    pub(crate) const fn gdn_uses_chunk(self, rows: usize, has_previous_state: bool) -> bool {
        matches!(self, Self::Bf16ModelMixedF32Accumulate)
            && rows != 0
            && !(has_previous_state && rows == 1)
    }

    /// Only Mixed opts fresh-history and multi-row cohorts into chunk arithmetic.
    /// `has_previous_state` must come from the validated journal-owned frontier.
    pub fn gdn_chunk_plan(
        self,
        layout: ferrule_backend::cuda_plan::GatedDeltaNetLayout,
        has_previous_state: bool,
        scratch_bytes: usize,
    ) -> ferrule_common::Result<Option<ferrule_backend::cuda_plan::Bf16ChunkPlan>> {
        if self.gdn_uses_chunk(layout.rows, has_previous_state) {
            ferrule_backend::cuda_plan::Bf16ChunkPlan::new_prefill(layout, scratch_bytes).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(crate) const fn backend_for_linear(
        self,
        role: &crate::support::TensorRole,
        rows: usize,
        layout: ferrule_backend::cuda_plan::NumericFp8Layout,
    ) -> ferrule_backend::cuda_plan::NumericFp8Precision {
        use crate::support::TensorRole as R;
        use ferrule_backend::cuda_plan::NumericFp8Precision as Backend;
        if matches!(self, Self::Bf16ModelMixedF32Accumulate)
            && layout.row_origin == 0
            && layout.column_origin == 0
            && matches!(
                layout.scale_type,
                ferrule_backend::cuda_plan::NumericFp8ScaleType::Bf16
            )
        {
            if rows > 1 {
                return if matches!(role, R::LinearAttentionQkv) && layout.k == 2048 {
                    Backend::Bf16SlicedK2F32Accumulate
                } else {
                    Backend::Bf16RneF32Accumulate
                };
            }
            return match (role, layout.n, layout.k) {
                (R::LinearAttentionQkv, 8192, 2048) => Backend::Bf16RneF32Accumulate,
                (R::LinearAttentionZ, 4096, 2048) => Backend::Bf16SlicedK2F32Accumulate,
                (R::AttentionOutput, 2048, 4096) => Backend::Bf16M1GemvF32Accumulate,
                // Frozen K/V pairs validate this reduction, not universal HF equivalence.
                (R::AttentionKey | R::AttentionValue, 512, 2048) => Backend::W8A16Bf16F32Accumulate,
                (R::SharedExpertGate | R::SharedExpertUp, 512, 2048)
                | (R::SharedExpertDown, 2048, 512)
                // Baseline only: no independent per-projection routed HF oracle.
                | (R::RoutedExpertGate | R::RoutedExpertUp | R::RoutedExpertDown, _, _) => Backend::W8A16Bf16F32Accumulate,
                _ => Backend::Bf16RneF32Accumulate,
            };
        }
        if matches!(self, Self::Bf16ModelSlicedK2F32Accumulate)
            && matches!(role, crate::support::TensorRole::LinearAttentionZ)
            && rows == 1
            && layout.k.is_multiple_of(64)
        {
            Backend::Bf16SlicedK2F32Accumulate
        } else {
            self.backend()
        }
    }

    pub(crate) const fn backend(self) -> ferrule_backend::cuda_plan::NumericFp8Precision {
        use ferrule_backend::cuda_plan::NumericFp8Precision as Backend;
        match self {
            Self::Bf16RneF32Accumulate => Backend::Bf16RneF32Accumulate,
            Self::Bf16ModelTiledF32Accumulate
            | Self::Bf16ModelSlicedK2F32Accumulate
            | Self::Bf16ModelMixedF32Accumulate => Backend::Bf16RneF32Accumulate,
            Self::F32Tf32x3 => Backend::F32Tf32x3,
            Self::W8A16Bf16F32Accumulate => Backend::W8A16Bf16F32Accumulate,
            Self::Bf16ModelF32Accumulate => Backend::W8A16Bf16F32Accumulate,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/transformer/numeric_precision.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/transformer/bf16_chunk/plan.rs"]
mod chunk_tests;

#[cfg(test)]
#[path = "../../../tests/unit/transformer/bf16_gemmex/plan.rs"]
mod gemmex_tests;

#[cfg(test)]
#[path = "../../../tests/unit/transformer/bf16_chunk/history.rs"]
mod chunk_history_tests;
