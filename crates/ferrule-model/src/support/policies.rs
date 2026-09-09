use crate::spec::{
    AttentionKind, ModelFamily, QuantFormatCount, RouterKind, TransformerSemantics,
    TransformerSpec, WeightSource,
};

use super::layout::{AttentionLayout, FeedForwardLayout};
use super::roles::{FeedForwardKind, KvCacheShape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPolicy {
    pub kind: AttentionKind,
    pub kv_shape: KvCacheShape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterPolicy {
    pub kind: RouterKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPolicy {
    pub kind: FeedForwardKind,
    pub has_shared_experts: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantPolicy {
    pub weight_source: WeightSource,
    pub formats: Vec<QuantFormatCount>,
}

impl QuantPolicy {
    pub fn has_gguf_quantized_tensors(&self) -> bool {
        matches!(self.weight_source, WeightSource::Gguf)
            && self.formats.iter().any(|item| {
                let format = item.format.as_str();
                !matches!(format, "F32" | "F16" | "Bf16" | "BF16")
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPolicy {
    pub shape: KvCacheShape,
    pub paged: bool,
    pub quantized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPolicy {
    pub streaming_allowed: bool,
    pub all_resident_required: bool,
}

pub use ferrule_common::{ParallelismPlan, ParallelismPlanError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculationMode {
    None,
    DraftModel,
    MultiTokenPrediction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculationPolicy {
    pub mode: SpeculationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerPolicy {
    pub requires_external_encoding: bool,
    pub has_chat_template: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationPolicy {
    pub requires_reference_engine: bool,
    pub supports_cpu_reference: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicySet {
    pub attention: AttentionPolicy,
    pub router: RouterPolicy,
    pub expert: ExpertPolicy,
    pub quant: QuantPolicy,
    pub kv: KvPolicy,
    pub residency: ResidencyPolicy,
    pub parallelism: ParallelismPlan,
    pub speculation: SpeculationPolicy,
    pub tokenizer: TokenizerPolicy,
    pub validation: ValidationPolicy,
    pub semantics: TransformerSemantics,
}

#[cfg(test)]
mod tests {
    use super::{ParallelismPlan, ParallelismPlanError};

    #[test]
    fn public_parallelism_types_are_common_types() {
        let plan: ferrule_common::ParallelismPlan = crate::ParallelismPlan::default();
        let policy_plan: ParallelismPlan = plan;
        let _: crate::ParallelismPlan = policy_plan;
        let error: ferrule_common::ParallelismPlanError =
            crate::ParallelismPlan::validated(1, 0, 1, 1, 1, 1).unwrap_err();
        let _: crate::ParallelismPlanError = error;
        let topology = ferrule_common::ValidatedParallelTopology::new(
            ferrule_common::ParallelTopologyId::new(1),
            1,
            ferrule_common::ParallelRankId::new(0),
            policy_plan,
        )
        .unwrap();
        assert_eq!(topology.plan(), plan);
    }

    #[test]
    fn default_parallelism_plan_is_valid() {
        assert!(ParallelismPlan::default().validate().is_ok());
    }

    #[test]
    fn validated_parallelism_plan_rejects_zero_dimension() {
        let error = ParallelismPlan::validated(1, 0, 1, 1, 1, 1).unwrap_err();
        assert_eq!(
            error,
            ParallelismPlanError::ZeroDimension {
                dimension: "tensor_parallel"
            }
        );
        assert!(error.to_string().contains("tensor_parallel"));
    }
}

impl PolicySet {
    pub fn from_spec(spec: &TransformerSpec) -> Self {
        let attention_layout = AttentionLayout::from_spec(spec);
        let feed_forward = FeedForwardLayout::from_spec(spec);
        let deepseek_like = matches!(
            spec.family,
            ModelFamily::DeepSeekV4 | ModelFamily::DeepSeekV3 | ModelFamily::DeepSeekV2
        );
        Self {
            attention: AttentionPolicy {
                kind: spec.attention.clone(),
                kv_shape: attention_layout.kv_shape.clone(),
            },
            router: RouterPolicy {
                kind: spec.moe.router.clone(),
            },
            expert: ExpertPolicy {
                kind: feed_forward.kind,
                has_shared_experts: spec.moe.has_shared_experts,
            },
            quant: QuantPolicy {
                weight_source: spec.weight_source,
                formats: spec.quantization.clone(),
            },
            kv: KvPolicy {
                shape: attention_layout.kv_shape,
                paged: false,
                quantized: false,
            },
            residency: ResidencyPolicy {
                streaming_allowed: deepseek_like && spec.moe.is_moe(),
                all_resident_required: !(deepseek_like && spec.moe.is_moe()),
            },
            parallelism: ParallelismPlan::default(),
            speculation: SpeculationPolicy {
                mode: SpeculationMode::None,
            },
            tokenizer: TokenizerPolicy {
                requires_external_encoding: deepseek_like,
                has_chat_template: !deepseek_like,
            },
            validation: ValidationPolicy {
                requires_reference_engine: true,
                supports_cpu_reference: false,
            },
            semantics: spec.semantics.clone(),
        }
    }
}
