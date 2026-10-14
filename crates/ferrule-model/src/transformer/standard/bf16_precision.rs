//! Admission contract for the first real BF16 standard CUDA image.
use crate::checkpoint::CheckpointDType;
use crate::transformer::{
    Attention, BoundDecoderResources, DecoderModelSpec, FeedForward, RotaryPairing, RotaryRegion,
    RotaryScaling,
};
use ferrule_common::{Error, Result};

/// Dense, ungated GQA + ordinary RMSNorm + unclamped SwiGLU, one CUDA owner.
///
/// Checkpoint parameters, activation/residual boundaries and both KV planes are
/// native BF16 (2 bytes/element). No F32 resident mirror or fallback is allowed.
/// GEMM reductions and attention Q/output scratch are F32; Q is an exact expansion
/// of BF16 and output is rounded before the next model operator. Logits are BF16
/// on device and exactly expanded for the existing F32 host logits contract.
/// Norm normalization, RoPE products, SiLU/product, and eager attention
/// dot/scale/probability have distinct HF tensor rounding boundaries. Arithmetic
/// inside reductions/exp/SiLU remains F32. This is not a Qwen3.5 GDN/MoE profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bf16DenseCudaProfile;
impl Bf16DenseCudaProfile {
    pub fn validate_spec(spec: &DecoderModelSpec) -> Result<()> {
        let reject = || {
            error(
                "requires dense ungated GQA, ordinary RMSNorm, unclamped SwiGLU and split-half prefix RoPE",
            )
        };
        if spec.final_norm().one_plus_weight()
            || spec.output_hyper_connection().is_some()
            || !spec.attachments().is_empty()
        {
            return Err(reject());
        }
        let check_linear = |l: &crate::transformer::Linear| {
            if l.has_bias() || !l.in_features().is_multiple_of(8) {
                Err(error("requires bias-free K8 projections"))
            } else {
                Ok(())
            }
        };
        check_linear(spec.output())?;
        for layer in spec.layers() {
            let Attention::Gqa(g) = layer.attention() else {
                return Err(reject());
            };
            let FeedForward::SwiGlu(f) = layer.feed_forward() else {
                return Err(reject());
            };
            for l in [
                g.query(),
                g.key(),
                g.value(),
                g.output(),
                f.gate(),
                f.up(),
                f.down(),
            ] {
                check_linear(l)?;
            }
            if g.gated_query()
                || layer.input_norm().one_plus_weight()
                || layer.post_attention_norm().one_plus_weight()
                || g.query_norm().is_some_and(|n| n.one_plus_weight())
                || g.key_norm().is_some_and(|n| n.one_plus_weight())
                || f.activation_limit().is_some()
                || (g.rotary().pairing() != RotaryPairing::SplitHalf
                    && g.rotary().region().dimensions() != 2)
                || !matches!(g.rotary().region(), RotaryRegion::Prefix { .. })
                || !matches!(g.rotary().scaling(), RotaryScaling::None)
            {
                return Err(reject());
            }
        }
        Ok(())
    }
    pub fn validate(resources: &BoundDecoderResources) -> Result<()> {
        Self::validate_spec(resources.spec())?;
        // Reject before materialization, upload or KV mutation. Bias and K padding
        // are deliberately unadmitted until their separate boundaries are tested.
        for p in resources.state_dict().parameters() {
            if p.weight().slice().dtype != CheckpointDType::Bf16
                || p.numeric_fp8_encoding().is_some()
                || p.spec().scale().tensor().is_some()
            {
                return Err(error("requires native unscaled BF16 checkpoint parameters"));
            }
            let shape = p.spec().shape();
            if shape.len() == 2 && !shape[1].is_multiple_of(8) {
                return Err(error("native BF16 GEMM requires K divisible by 8"));
            }
            if p.path().as_str().ends_with(".bias") {
                return Err(error("linear bias is not admitted"));
            }
        }
        Ok(())
    }
}
fn error(message: &str) -> Error {
    Error::Internal {
        message: format!("BF16 dense CUDA profile: {message}"),
    }
}
