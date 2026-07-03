//! Runtime-side Transformer execution plan.
//!
//! `ferrule-model` owns model-family detection, tensor-name classification, and
//! semantic layout. This module turns that semantic contract into the ordered
//! runtime steps an executor must implement. It intentionally contains no
//! DeepSeek tensor names and no OLMoE fields: family adapters produce roles and
//! policies; runtime executors consume this plan.

use ferrule_model::{
    AttentionKind, FeedForwardKind, KvCacheShape, ModelFamily, ModelSupportContract, PolicySet,
    RouterKind, SpeculationMode, TensorRole,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerRuntimePlan {
    pub family: ModelFamily,
    pub architecture: Option<String>,
    pub hidden_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub prologue: RuntimePrologue,
    pub layers: Vec<TransformerLayerPlan>,
    pub epilogue: RuntimeEpilogue,
    pub policies: PolicySet,
    pub attachments: Vec<RuntimeAttachment>,
}

impl TransformerRuntimePlan {
    pub fn from_contract(contract: &ModelSupportContract) -> Self {
        let layout = &contract.layout;
        let spec = &contract.spec;
        let prologue = RuntimePrologue {
            token_embedding: layout.token_embedding.clone(),
        };
        let semantics = &contract.policies.semantics;
        let norm_epsilon = semantics.norm_epsilon.unwrap_or(1e-6);
        let hyper_connection_epsilon = semantics.hyper_connection_epsilon.unwrap_or(norm_epsilon);
        let hyper_connection_sinkhorn_iters =
            semantics.hyper_connection_sinkhorn_iters.unwrap_or(4);
        let layers = layout
            .layers
            .iter()
            .map(|layer| {
                let router = layer_router_kind(
                    layer.index,
                    &layer.feed_forward.router,
                    semantics.num_hash_layers,
                );
                TransformerLayerPlan {
                    index: layer.index,
                    pre_norm_roles: layer.norms.clone(),
                    attention: AttentionStepPlan {
                        kind: layer.attention.kind.clone(),
                        kv_shape: layer.attention.kv_shape.clone(),
                        num_heads: spec.num_heads,
                        num_kv_heads: spec.num_kv_heads,
                        head_dim: spec.head_dim,
                        rope_theta: semantics.rope_theta,
                        rope_head_dim: semantics.rope_head_dim,
                        rope_factor: semantics.rope_factor,
                        rope_original_max_position_embeddings: semantics
                            .rope_original_max_position_embeddings,
                        rope_beta_fast: semantics.rope_beta_fast,
                        rope_beta_slow: semantics.rope_beta_slow,
                        compress_rope_theta: semantics.compress_rope_theta,
                        window_size: semantics.attention_window_size,
                        index_topk: semantics.attention_index_topk,
                        index_num_heads: semantics.attention_index_num_heads,
                        index_head_dim: semantics.attention_index_head_dim,
                        compress_ratio: semantics
                            .attention_compress_ratios
                            .get(layer.index)
                            .copied(),
                        required_roles: layer.attention.required_roles.clone(),
                        optional_roles: layer.attention.optional_roles.clone(),
                        needs_sparse_indices: matches!(
                            layer.attention.kv_shape,
                            KvCacheShape::LatentOrCompressed
                        ),
                        needs_attention_sink: layer
                            .attention
                            .required_roles
                            .iter()
                            .chain(layer.attention.optional_roles.iter())
                            .any(|role| matches!(role, TensorRole::AttentionSink))
                            || matches!(layer.attention.kind, AttentionKind::MultiLatentAttention),
                    },
                    feed_forward: FeedForwardStepPlan {
                        kind: layer.feed_forward.kind.clone(),
                        router,
                        num_experts: spec.moe.num_experts,
                        num_experts_per_tok: spec.moe.num_experts_per_tok,
                        required_roles: layer.feed_forward.required_roles.clone(),
                        optional_roles: layer.feed_forward.optional_roles.clone(),
                        swiglu_limit: semantics.swiglu_limit,
                        route_scale: semantics.route_scale,
                        expert_residency: if matches!(
                            layer.feed_forward.kind,
                            FeedForwardKind::RoutedExperts
                                | FeedForwardKind::RoutedAndSharedExperts
                        ) && contract.policies.residency.streaming_allowed
                        {
                            ExpertResidencyMode::Streamable
                        } else {
                            ExpertResidencyMode::AllResident
                        },
                        has_shared_experts: matches!(
                            layer.feed_forward.kind,
                            FeedForwardKind::RoutedAndSharedExperts
                        ),
                    },
                    auxiliary_roles: layer.auxiliary_roles.clone(),
                    norm_epsilon,
                    hyper_connection_epsilon,
                    hyper_connection_sinkhorn_iters,
                }
            })
            .collect();
        let epilogue = RuntimeEpilogue {
            output_norm: layout
                .output_roles
                .iter()
                .find(|role| matches!(role, TensorRole::OutputNorm))
                .cloned(),
            output_head: layout
                .output_roles
                .iter()
                .find(|role| matches!(role, TensorRole::OutputHead))
                .cloned(),
        };
        let attachments = runtime_attachments(contract);
        Self {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            hidden_size: spec.hidden_size,
            vocab_size: spec.vocab_size,
            num_heads: spec.num_heads,
            num_kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            prologue,
            layers,
            epilogue,
            policies: contract.policies.clone(),
            attachments,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn step_count(&self) -> usize {
        // embedding + per-layer attention/ffn + optional output norm + output head + attachments.
        1 + self.layers.len() * 2
            + usize::from(self.epilogue.output_norm.is_some())
            + usize::from(self.epilogue.output_head.is_some())
            + self.attachments.len()
    }

    pub fn requires_streaming_experts(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| layer.feed_forward.expert_residency == ExpertResidencyMode::Streamable)
    }

    pub fn uses_hash_routing(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer.feed_forward.router, RouterKind::HashAssistedTopK))
    }

    pub fn uses_latent_or_compressed_attention(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer.attention.kv_shape, KvCacheShape::LatentOrCompressed))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePrologue {
    pub token_embedding: TensorRole,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerLayerPlan {
    pub index: usize,
    pub pre_norm_roles: Vec<TensorRole>,
    pub attention: AttentionStepPlan,
    pub feed_forward: FeedForwardStepPlan,
    pub auxiliary_roles: Vec<TensorRole>,
    pub norm_epsilon: f32,
    pub hyper_connection_epsilon: f32,
    pub hyper_connection_sinkhorn_iters: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionStepPlan {
    pub kind: AttentionKind,
    pub kv_shape: KvCacheShape,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rope_head_dim: Option<usize>,
    pub rope_factor: Option<f32>,
    pub rope_original_max_position_embeddings: Option<usize>,
    pub rope_beta_fast: Option<usize>,
    pub rope_beta_slow: Option<usize>,
    pub compress_rope_theta: Option<f32>,
    pub window_size: Option<usize>,
    pub index_topk: Option<usize>,
    pub index_num_heads: Option<usize>,
    pub index_head_dim: Option<usize>,
    pub compress_ratio: Option<usize>,
    pub required_roles: Vec<TensorRole>,
    pub optional_roles: Vec<TensorRole>,
    pub needs_sparse_indices: bool,
    pub needs_attention_sink: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardStepPlan {
    pub kind: FeedForwardKind,
    pub router: RouterKind,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub required_roles: Vec<TensorRole>,
    pub optional_roles: Vec<TensorRole>,
    pub swiglu_limit: Option<f32>,
    pub route_scale: Option<f32>,
    pub expert_residency: ExpertResidencyMode,
    pub has_shared_experts: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertResidencyMode {
    AllResident,
    Streamable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEpilogue {
    pub output_norm: Option<TensorRole>,
    pub output_head: Option<TensorRole>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAttachment {
    MultiTokenPrediction { roles: Vec<TensorRole> },
    DraftModel,
}

fn layer_router_kind(
    layer: usize,
    base: &RouterKind,
    num_hash_layers: Option<usize>,
) -> RouterKind {
    if matches!(base, RouterKind::HashAssistedTopK)
        && num_hash_layers.is_some_and(|hash_layers| layer >= hash_layers)
    {
        RouterKind::DenseTopK
    } else {
        base.clone()
    }
}

fn runtime_attachments(contract: &ModelSupportContract) -> Vec<RuntimeAttachment> {
    match contract.policies.speculation.mode {
        SpeculationMode::None => Vec::new(),
        SpeculationMode::DraftModel => vec![RuntimeAttachment::DraftModel],
        SpeculationMode::MultiTokenPrediction => {
            let mut roles = contract
                .bound_roles()
                .into_iter()
                .filter(TensorRole::is_speculative)
                .collect::<Vec<_>>();
            roles.sort();
            vec![RuntimeAttachment::MultiTokenPrediction { roles }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_model::{
        AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TensorClass,
        TensorClassCount, TransformerSpec, WeightSource,
    };

    #[test]
    fn olmoe_contract_builds_generic_routed_plan_without_streaming() {
        let contract = ModelSupportContract::from_spec(
            &TransformerSpec {
                family: ModelFamily::Olmoe,
                architecture: Some("olmoe".into()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(16),
                num_layers: Some(2),
                vocab_size: Some(32),
                num_heads: Some(4),
                num_kv_heads: Some(2),
                head_dim: Some(4),
                attention: AttentionKind::GroupedQuery,
                moe: MoeSpec {
                    num_experts: Some(4),
                    num_experts_per_tok: Some(2),
                    has_shared_experts: false,
                    router: RouterKind::DenseTopK,
                },
                semantics: Default::default(),
                tensor_count: None,
                quantization: Vec::new(),
                notes: Vec::new(),
            },
            &[],
        );
        let plan = TransformerRuntimePlan::from_contract(&contract);
        assert_eq!(plan.layer_count(), 2);
        assert_eq!(plan.step_count(), 7);
        assert_eq!(plan.layers[0].attention.kind, AttentionKind::GroupedQuery);
        assert_eq!(
            plan.layers[0].attention.kv_shape,
            KvCacheShape::GroupedKeysValues
        );
        assert_eq!(
            plan.layers[0].feed_forward.kind,
            FeedForwardKind::RoutedExperts
        );
        assert_eq!(
            plan.layers[0].feed_forward.expert_residency,
            ExpertResidencyMode::AllResident
        );
        assert!(!plan.requires_streaming_experts());
    }

    #[test]
    fn deepseek_v4_contract_builds_mla_streaming_hash_routed_plan() {
        let contract = ModelSupportContract::from_spec(
            &TransformerSpec {
                family: ModelFamily::DeepSeekV4,
                architecture: Some("deepseek_v4".into()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(4096),
                num_layers: Some(43),
                vocab_size: Some(129280),
                num_heads: Some(64),
                num_kv_heads: Some(1),
                head_dim: Some(512),
                attention: AttentionKind::MultiLatentAttention,
                moe: MoeSpec {
                    num_experts: Some(256),
                    num_experts_per_tok: Some(6),
                    has_shared_experts: true,
                    router: RouterKind::HashAssistedTopK,
                },
                semantics: Default::default(),
                tensor_count: Some(72_317),
                quantization: vec![QuantFormatCount {
                    format: "I8".into(),
                    tensors: 35_328,
                }],
                notes: Vec::new(),
            },
            &[],
        );
        let plan = TransformerRuntimePlan::from_contract(&contract);
        assert_eq!(plan.family, ModelFamily::DeepSeekV4);
        assert_eq!(plan.layer_count(), 43);
        assert!(plan.requires_streaming_experts());
        assert!(plan.uses_hash_routing());
        assert!(plan.uses_latent_or_compressed_attention());
        assert_eq!(
            plan.layers[0].attention.kind,
            AttentionKind::MultiLatentAttention
        );
        assert_eq!(
            plan.layers[0].attention.kv_shape,
            KvCacheShape::LatentOrCompressed
        );
        assert!(plan.layers[0].attention.needs_sparse_indices);
        assert!(plan.layers[0].attention.needs_attention_sink);
        assert_eq!(
            plan.layers[0].feed_forward.kind,
            FeedForwardKind::RoutedAndSharedExperts
        );
        assert_eq!(
            plan.layers[0].feed_forward.expert_residency,
            ExpertResidencyMode::Streamable
        );
        assert!(plan.layers[0].feed_forward.has_shared_experts);
    }

    #[test]
    fn dspark_mtp_attachment_is_planned_as_runtime_attachment() {
        let contract = ModelSupportContract::from_spec(
            &TransformerSpec {
                family: ModelFamily::DeepSeekV4,
                architecture: Some("deepseek_v4".into()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(16),
                num_layers: Some(1),
                vocab_size: Some(32),
                num_heads: Some(4),
                num_kv_heads: Some(1),
                head_dim: Some(4),
                attention: AttentionKind::MultiLatentAttention,
                moe: MoeSpec {
                    num_experts: Some(8),
                    num_experts_per_tok: Some(2),
                    has_shared_experts: true,
                    router: RouterKind::HashAssistedTopK,
                },
                semantics: Default::default(),
                tensor_count: None,
                quantization: Vec::new(),
                notes: Vec::new(),
            },
            &[
                TensorClassCount {
                    class: TensorClass::SpeculativeProjection,
                    tensors: 2,
                },
                TensorClassCount {
                    class: TensorClass::SpeculativeMarkovHead,
                    tensors: 2,
                },
                TensorClassCount {
                    class: TensorClass::SpeculativeConfidenceHead,
                    tensors: 1,
                },
            ],
        );
        let plan = TransformerRuntimePlan::from_contract(&contract);
        assert_eq!(plan.attachments.len(), 1);
        let RuntimeAttachment::MultiTokenPrediction { roles } = &plan.attachments[0] else {
            panic!("expected MTP attachment");
        };
        assert!(roles.contains(&TensorRole::SpeculativeProjection));
        assert!(roles.contains(&TensorRole::SpeculativeMarkovHead));
        assert!(roles.contains(&TensorRole::SpeculativeConfidenceHead));
    }
}
