//! Native and numeric-storage BF16 hybrid admission and declared precision policies.
use super::*;
use crate::checkpoint::CheckpointDType;
use crate::transformer::{RouterScoreFunction, RouterSelection};

/// Single-owner resident native-BF16 GQA/GDN + dense/shared/routed SwiGLU.
/// Hidden, residual, KV, conv history and device logits are BF16; SSM is F32.
/// Router parameter weights and logits are BF16; normalized selected-softmax
/// probabilities and route weights are F32 (ties use lower expert index).
/// This algebraically equivalent router reduction
/// is not bitwise HF full-softmax. Expert combine is HF eager expert-index order.
/// Numeric FP8, tensor/expert parallelism and HostKV are not admitted here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bf16HybridCudaProfile;
impl Bf16HybridCudaProfile {
    pub const NAME: &'static str = "cuda-hybrid-native-bf16-f32-ssm";
    pub const WEIGHT_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const SCALE_DTYPE: Option<CheckpointDType> = None;
    pub const ROW_DTYPE: crate::transformer::RowsDType = crate::transformer::RowsDType::Bf16;
    pub const KV_DTYPE: ferrule_common::execution::KvElementType =
        ferrule_common::execution::KvElementType::Bf16;
    pub const CONV_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const SSM_DTYPE: CheckpointDType = CheckpointDType::F32;
    /// Router gate parameter matrix, not the selected route weights.
    pub const ROUTER_WEIGHT_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROUTER_LOGITS_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROUTER_PROBABILITY_DTYPE: CheckpointDType = CheckpointDType::F32;
    pub const ROUTE_WEIGHT_DTYPE: CheckpointDType = CheckpointDType::F32;

    pub fn validate(resources: &BoundDecoderResources) -> Result<()> {
        Self::validate_spec(resources.spec())?;
        for p in resources.state_dict().parameters() {
            if p.weight().slice().dtype != CheckpointDType::Bf16
                || p.scale().is_some()
                || p.numeric_fp8_encoding().is_some()
            {
                return Err(model_error(
                    "native BF16 hybrid requires unscaled BF16 parameters",
                ));
            }
        }
        Ok(())
    }
    pub fn validate_spec(spec: &DecoderModelSpec) -> Result<()> {
        if spec.output().has_bias()
            || spec.layers().iter().any(|l| {
                !matches!(l.attention_residual(), Residual::Add)
                    || !matches!(l.feed_forward_residual(), Residual::Add)
            })
        {
            return Err(model_error(
                "BF16 hybrid requires additive residuals and bias-free LM head",
            ));
        }
        if !spec.attachments().is_empty() || spec.output_hyper_connection().is_some() {
            return Err(model_error(
                "BF16 hybrid attachments/hyperconnection unsupported",
            ));
        }
        let linear = |l: &Linear| -> Result<()> {
            if l.has_bias() || !l.in_features().is_multiple_of(8) {
                Err(model_error("BF16 hybrid requires bias-free K8 projections"))
            } else {
                Ok(())
            }
        };
        let swiglu = |f: &SwiGlu| -> Result<()> {
            if f.activation_limit().is_some() {
                return Err(model_error("BF16 hybrid rejects clamped SwiGLU"));
            }
            for l in [f.gate(), f.up(), f.down()] {
                linear(l)?;
            }
            Ok(())
        };
        linear(spec.output())?;
        let mut full_attention = false;
        for layer in spec.layers() {
            match layer.attention() {
                Attention::Gqa(g) => {
                    full_attention = true;
                    for l in [g.query(), g.key(), g.value(), g.output()] {
                        linear(l)?;
                    }
                    use crate::transformer::{RotaryPairing, RotaryRegion, RotaryScaling};
                    if (g.rotary().pairing() != RotaryPairing::SplitHalf
                        && g.rotary().region().dimensions() != 2)
                        || !matches!(g.rotary().region(), RotaryRegion::Prefix { .. })
                        || !matches!(g.rotary().scaling(), RotaryScaling::None)
                    {
                        return Err(model_error(
                            "BF16 hybrid requires split-half prefix unscaled RoPE",
                        ));
                    }
                    if let Some(reason) = standard_rope_unsupported(g.rotary()) {
                        return Err(model_error(format!("BF16 hybrid RoPE: {reason}")));
                    }
                }
                Attention::GatedDeltaNet(g) => {
                    for l in [g.qkv(), g.z(), g.beta(), g.a(), g.output()] {
                        linear(l)?;
                    }
                    if g.norm().one_plus_weight() {
                        return Err(model_error("GDN gated norm must be ordinary affine"));
                    }
                }
                _ => return Err(model_error("BF16 hybrid supports GQA/GDN only")),
            }
            match layer.feed_forward() {
                FeedForward::SwiGlu(f) => swiglu(f)?,
                FeedForward::Moe(m) => {
                    swiglu(m.expert())?;
                    if let Some(f) = m.shared_expert() {
                        swiglu(f)?;
                    }
                    if let Some(l) = m.shared_expert_gate() {
                        linear(l)?;
                    }
                    linear(m.router())?;
                    let r = m.router_spec();
                    if r.score_function() != RouterScoreFunction::Softmax
                        || r.selection() != &RouterSelection::TopK
                        || !r.normalize_selected()
                        || r.selection_bias()
                        || r.route_scale() != 1.0
                    {
                        return Err(model_error(
                            "BF16 hybrid requires normalized unbiased softmax top-k",
                        ));
                    }
                }
            }
        }
        if !full_attention {
            return Err(model_error(
                "BF16 paged hybrid requires a full-attention layer",
            ));
        }
        Ok(())
    }
}

/// Raw E4M3FN with paired BF16 block scales; native resident parameters and
/// model boundaries are BF16. GDN A_log/norm may have F32 checkpoint sources,
/// cast once to BF16 on load; their operator operands are BF16, SSM state F32.
/// RoPE inverse frequencies/trigonometry are F32, resident cos/sin and output BF16.
/// These constants declare device policy, not proof of GPU storage or quality.
/// This does not rename the model-F32 W8 path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NumericFp8Bf16HybridCudaProfile;
impl NumericFp8Bf16HybridCudaProfile {
    /// Historical fused execution identity; unchanged.
    pub const NAME: &'static str = "cuda-hybrid-numeric-fp8-model-bf16-f32-ssm";
    /// Same storage/boundary contract, distinct bounded tiled execution identity.
    pub const TILED_NAME: &'static str = "cuda-hybrid-numeric-fp8-model-bf16-tiled-f32-ssm";
    pub const SLICED_K2_NAME: &'static str = "cuda-hybrid-numeric-fp8-model-bf16-sliced-k2-f32-ssm";
    pub const MIXED_NAME: &'static str = "cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm";
    pub const WEIGHT_DTYPE: CheckpointDType = CheckpointDType::F8E4M3;
    pub const SCALE_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    /// Optional F32 checkpoint-source exception for GDN A_log/norm only;
    /// BF16 sources are also admitted. Other GDN parameters remain native BF16.
    pub const GDN_ALOG_NORM_SOURCE_DTYPE: CheckpointDType = CheckpointDType::F32;
    /// Compatibility name for the A_log/norm source exception, not resident dtype.
    pub const GDN_SOURCE_DTYPE: CheckpointDType = Self::GDN_ALOG_NORM_SOURCE_DTYPE;
    pub const GDN_ALOG_NORM_RESIDENT_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    /// A_log/norm operator operands, not internal accumulation or recurrent state.
    pub const GDN_ALOG_NORM_OPERATOR_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const NATIVE_WEIGHT_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROW_DTYPE: crate::transformer::RowsDType = crate::transformer::RowsDType::Bf16;
    pub const KV_DTYPE: ferrule_common::execution::KvElementType =
        ferrule_common::execution::KvElementType::Bf16;
    pub const CONV_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const SSM_DTYPE: CheckpointDType = CheckpointDType::F32;
    pub const ACCUMULATOR_DTYPE: CheckpointDType = CheckpointDType::F32;
    /// Router gate parameter matrix, not the selected route weights.
    pub const ROUTER_WEIGHT_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROUTER_LOGITS_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROUTER_PROBABILITY_DTYPE: CheckpointDType = CheckpointDType::F32;
    pub const ROUTE_WEIGHT_DTYPE: CheckpointDType = CheckpointDType::F32;
    pub const ROPE_INV_FREQ_DTYPE: CheckpointDType = CheckpointDType::F32;
    /// Compatibility name for inverse-frequency precision, not table storage.
    pub const ROPE_FREQUENCY_DTYPE: CheckpointDType = Self::ROPE_INV_FREQ_DTYPE;
    pub const ROPE_TRIG_DTYPE: CheckpointDType = CheckpointDType::F32;
    /// Resident cos/sin tables after the F32 trigonometric calculation.
    pub const ROPE_COS_SIN_DTYPE: CheckpointDType = CheckpointDType::Bf16;
    pub const ROPE_OUTPUT_DTYPE: CheckpointDType = CheckpointDType::Bf16;

    pub fn validate(resources: &BoundDecoderResources) -> Result<()> {
        Bf16HybridCudaProfile::validate_spec(resources.spec())?;
        let mut numeric = false;
        for p in resources.state_dict().parameters() {
            if p.is_alias() {
                return Err(model_error(
                    "numeric BF16 hybrid requires unaliased image parameters",
                ));
            }
            if let Some(encoding) = p.numeric_fp8_encoding() {
                if encoding != crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128Bf16
                    || !crate::decoder::hybrid_memory_plan::numeric_linear_role(p.role())
                {
                    return Err(model_error(
                        "numeric BF16 hybrid requires individual unaliased E4M3FN/BF16-scale projections",
                    ));
                }
                let source = p.numeric_fp8_source(encoding)?;
                source.validate_source_identity()?;
                if source.expert_count().is_some() {
                    return Err(model_error("numeric BF16 hybrid rejects packed experts"));
                }
                numeric = true;
            } else if p.scale().is_some()
                || !(p.weight().slice().dtype == CheckpointDType::Bf16
                    || (p.weight().slice().dtype == CheckpointDType::F32
                        && matches!(
                            p.role(),
                            crate::support::TensorRole::LinearAttentionALog
                                | crate::support::TensorRole::LinearAttentionNorm
                        )))
            {
                return Err(model_error(
                    "numeric BF16 hybrid requires native BF16 words or declared F32 GDN A_log/norm loading",
                ));
            }
        }
        if !numeric {
            return Err(model_error(
                "numeric BF16 hybrid requires numeric checkpoint storage",
            ));
        }
        Ok(())
    }
}
