use crate::spec::{AttentionKind, ModelFamily, RouterKind, TransformerSpec};

use super::roles::{FeedForwardKind, KvCacheShape, TensorRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionLayout {
    pub kind: AttentionKind,
    pub kv_shape: KvCacheShape,
    pub required_roles: Vec<TensorRole>,
    pub optional_roles: Vec<TensorRole>,
}

impl AttentionLayout {
    pub fn from_spec(spec: &TransformerSpec) -> Self {
        match spec.attention {
            AttentionKind::DenseMha => Self {
                kind: spec.attention.clone(),
                kv_shape: KvCacheShape::FullKeysValues,
                required_roles: vec![
                    TensorRole::AttentionQuery,
                    TensorRole::AttentionKey,
                    TensorRole::AttentionValue,
                    TensorRole::AttentionOutput,
                ],
                optional_roles: Vec::new(),
            },
            AttentionKind::GroupedQuery => Self {
                kind: spec.attention.clone(),
                kv_shape: KvCacheShape::GroupedKeysValues,
                required_roles: vec![
                    TensorRole::AttentionQuery,
                    TensorRole::AttentionKey,
                    TensorRole::AttentionValue,
                    TensorRole::AttentionOutput,
                ],
                optional_roles: Vec::new(),
            },
            AttentionKind::MultiLatentAttention => Self {
                kind: spec.attention.clone(),
                kv_shape: KvCacheShape::LatentOrCompressed,
                required_roles: vec![
                    TensorRole::AttentionLatentQueryA,
                    TensorRole::AttentionLatentQueryB,
                    TensorRole::AttentionQueryNorm,
                    TensorRole::AttentionLatentKv,
                    TensorRole::AttentionKeyValueNorm,
                    TensorRole::AttentionLatentOutputA,
                    TensorRole::AttentionLatentOutputB,
                    TensorRole::AttentionSink,
                ],
                optional_roles: vec![
                    TensorRole::AttentionCompressor,
                    TensorRole::AuxIndexer,
                    TensorRole::AuxHiddenCompressor,
                ],
            },
            AttentionKind::Unknown(_) => Self {
                kind: spec.attention.clone(),
                kv_shape: KvCacheShape::Unknown,
                required_roles: Vec::new(),
                optional_roles: Vec::new(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedForwardLayout {
    pub kind: FeedForwardKind,
    pub router: RouterKind,
    pub required_roles: Vec<TensorRole>,
    pub optional_roles: Vec<TensorRole>,
}

impl FeedForwardLayout {
    pub fn from_spec(spec: &TransformerSpec) -> Self {
        if spec.moe.is_moe() {
            let mut required_roles = vec![
                TensorRole::RouterLogits,
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ];
            let mut optional_roles = Vec::new();
            if matches!(spec.moe.router, RouterKind::HashAssistedTopK) {
                required_roles.push(TensorRole::HashRouterTable);
            }
            if spec.moe.has_shared_experts {
                required_roles.extend([
                    TensorRole::SharedExpertGate,
                    TensorRole::SharedExpertUp,
                    TensorRole::SharedExpertDown,
                ]);
            } else {
                optional_roles.extend([
                    TensorRole::SharedExpertGate,
                    TensorRole::SharedExpertUp,
                    TensorRole::SharedExpertDown,
                ]);
            }
            Self {
                kind: if spec.moe.has_shared_experts {
                    FeedForwardKind::RoutedAndSharedExperts
                } else {
                    FeedForwardKind::RoutedExperts
                },
                router: spec.moe.router.clone(),
                required_roles,
                optional_roles,
            }
        } else {
            Self {
                kind: FeedForwardKind::DenseMlp,
                router: RouterKind::None,
                required_roles: vec![
                    TensorRole::DenseMlpGate,
                    TensorRole::DenseMlpUp,
                    TensorRole::DenseMlpDown,
                ],
                optional_roles: Vec::new(),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    pub index: usize,
    pub norms: Vec<TensorRole>,
    pub attention: AttentionLayout,
    pub feed_forward: FeedForwardLayout,
    pub auxiliary_roles: Vec<TensorRole>,
}

impl LayerLayout {
    pub fn from_spec(index: usize, spec: &TransformerSpec) -> Self {
        let attention = AttentionLayout::from_spec(spec);
        let mut auxiliary_roles = Vec::new();
        if matches!(attention.kv_shape, KvCacheShape::LatentOrCompressed) {
            auxiliary_roles.extend([
                TensorRole::AuxIndexer,
                TensorRole::AuxHiddenCompressor,
                TensorRole::AuxOutputHiddenCompressor,
            ]);
        }
        Self {
            index,
            norms: layer_norm_roles_for_spec(spec),
            attention,
            feed_forward: FeedForwardLayout::from_spec(spec),
            auxiliary_roles,
        }
    }

    pub fn required_roles(&self) -> Vec<TensorRole> {
        let mut roles = self.norms.clone();
        roles.extend(self.attention.required_roles.iter().cloned());
        roles.extend(self.feed_forward.required_roles.iter().cloned());
        roles
    }
}

fn layer_norm_roles_for_spec(spec: &TransformerSpec) -> Vec<TensorRole> {
    if matches!(spec.family, ModelFamily::DeepSeekV4) {
        vec![TensorRole::AttentionNorm, TensorRole::FeedForwardNorm]
    } else {
        vec![TensorRole::LayerNorm]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLayout {
    pub token_embedding: TensorRole,
    pub output_roles: Vec<TensorRole>,
    pub layers: Vec<LayerLayout>,
}

impl ModelLayout {
    pub fn from_spec(spec: &TransformerSpec) -> Self {
        let layers = (0..spec.num_layers.unwrap_or(0))
            .map(|index| LayerLayout::from_spec(index, spec))
            .collect();
        Self {
            token_embedding: TensorRole::TokenEmbedding,
            output_roles: vec![TensorRole::OutputNorm, TensorRole::OutputHead],
            layers,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::spec::{MoeSpec, WeightSource};

    use super::*;

    #[test]
    fn deepseek_v4_layout_requires_stage_norms_without_generic_layer_norm() {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("deepseek_v4".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(4096),
            num_layers: Some(1),
            vocab_size: Some(129_280),
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
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        let layout = ModelLayout::from_spec(&spec);
        assert_eq!(
            layout.layers[0].norms,
            vec![TensorRole::AttentionNorm, TensorRole::FeedForwardNorm]
        );
        let required = layout.layers[0].required_roles();
        assert!(!required.contains(&TensorRole::LayerNorm));
        assert!(required.contains(&TensorRole::AttentionNorm));
        assert!(required.contains(&TensorRole::FeedForwardNorm));
        assert!(required.contains(&TensorRole::AttentionQueryNorm));
        assert!(required.contains(&TensorRole::AttentionKeyValueNorm));
    }

    #[test]
    fn dense_layout_keeps_generic_layer_norm_requirement() {
        let spec = TransformerSpec {
            family: ModelFamily::Llama,
            architecture: Some("llama".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(16),
            num_layers: Some(1),
            vocab_size: Some(32),
            num_heads: Some(4),
            num_kv_heads: Some(4),
            head_dim: Some(4),
            attention: AttentionKind::DenseMha,
            moe: MoeSpec::none(),
            semantics: Default::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        let layout = ModelLayout::from_spec(&spec);
        assert_eq!(layout.layers[0].norms, vec![TensorRole::LayerNorm]);
    }
}
