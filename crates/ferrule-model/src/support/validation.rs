use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{ModelLayout, TensorBinding, TensorRole};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RoleScope {
    Model,
    Layer { index: usize },
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingRequiredRole {
    pub scope: RoleScope,
    pub role: TensorRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionalRoleStatus {
    pub scope: RoleScope,
    pub role: TensorRole,
    pub present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutValidationReport {
    pub missing_required: Vec<MissingRequiredRole>,
    pub optional_roles: Vec<OptionalRoleStatus>,
    pub bound_role_counts: Vec<BoundRoleCount>,
}

impl LayoutValidationReport {
    pub fn is_complete(&self) -> bool {
        self.missing_required.is_empty()
    }

    pub fn missing_roles(&self) -> BTreeSet<TensorRole> {
        self.missing_required
            .iter()
            .map(|missing| missing.role.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BoundRoleCount {
    pub role: TensorRole,
    pub tensors: usize,
}

pub fn validate_model_layout_bindings(
    layout: &ModelLayout,
    bindings: &[TensorBinding],
) -> LayoutValidationReport {
    let counts = binding_role_counts(bindings);
    let mut missing_required = Vec::new();
    let mut optional_roles = Vec::new();

    require_role(
        &counts,
        RoleScope::Model,
        layout.token_embedding.clone(),
        &mut missing_required,
    );
    for role in &layout.output_roles {
        require_role(
            &counts,
            RoleScope::Output,
            role.clone(),
            &mut missing_required,
        );
    }

    for layer in &layout.layers {
        let scope = RoleScope::Layer { index: layer.index };
        for role in layer.required_roles() {
            require_role(&counts, scope.clone(), role, &mut missing_required);
        }
        for role in layer
            .attention
            .optional_roles
            .iter()
            .chain(layer.feed_forward.optional_roles.iter())
            .chain(layer.auxiliary_roles.iter())
        {
            optional_roles.push(OptionalRoleStatus {
                scope: scope.clone(),
                role: role.clone(),
                present: counts.get(role).copied().unwrap_or(0) > 0,
            });
        }
    }

    LayoutValidationReport {
        missing_required,
        optional_roles,
        bound_role_counts: counts
            .into_iter()
            .map(|(role, tensors)| BoundRoleCount { role, tensors })
            .collect(),
    }
}

fn binding_role_counts(bindings: &[TensorBinding]) -> BTreeMap<TensorRole, usize> {
    let mut counts = BTreeMap::<TensorRole, usize>::new();
    for binding in bindings {
        *counts.entry(binding.role.clone()).or_default() += binding.tensors;
    }
    counts
}

fn require_role(
    counts: &BTreeMap<TensorRole, usize>,
    scope: RoleScope,
    role: TensorRole,
    missing_required: &mut Vec<MissingRequiredRole>,
) {
    if counts.get(&role).copied().unwrap_or(0) == 0 {
        missing_required.push(MissingRequiredRole { scope, role });
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AttentionKind, ModelLayout, MoeSpec, RouterKind, TensorBinding, TensorClass,
        TransformerSpec, WeightSource,
    };

    use super::*;

    #[test]
    fn validates_dense_layout_without_deepseek_roles() {
        let spec = TransformerSpec {
            family: crate::ModelFamily::Llama,
            architecture: Some("llama-fixture".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(8),
            num_layers: Some(1),
            vocab_size: Some(16),
            num_heads: Some(2),
            num_kv_heads: Some(2),
            head_dim: Some(4),
            attention: AttentionKind::DenseMha,
            moe: MoeSpec::none(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        let layout = ModelLayout::from_spec(&spec);
        let bindings = vec![
            binding(TensorClass::TokenEmbedding, TensorRole::TokenEmbedding, 1),
            binding(TensorClass::OutputNorm, TensorRole::OutputNorm, 1),
            binding(TensorClass::OutputHead, TensorRole::OutputHead, 1),
            binding(TensorClass::LayerNorm, TensorRole::LayerNorm, 1),
            binding(TensorClass::AttentionQuery, TensorRole::AttentionQuery, 1),
            binding(TensorClass::AttentionKey, TensorRole::AttentionKey, 1),
            binding(TensorClass::AttentionValue, TensorRole::AttentionValue, 1),
            binding(TensorClass::AttentionOutput, TensorRole::AttentionOutput, 1),
            binding(TensorClass::Auxiliary, TensorRole::DenseMlpGate, 1),
            binding(TensorClass::Auxiliary, TensorRole::DenseMlpUp, 1),
            binding(TensorClass::Auxiliary, TensorRole::DenseMlpDown, 1),
        ];
        let report = validate_model_layout_bindings(&layout, &bindings);
        assert!(
            report.is_complete(),
            "missing: {:?}",
            report.missing_required
        );
    }

    #[test]
    fn reports_deepseek_layer_and_role_for_missing_attention_sink() {
        let spec = TransformerSpec {
            family: crate::ModelFamily::DeepSeekV4,
            architecture: Some("deepseek_v4".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(4096),
            num_layers: Some(1),
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
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        let layout = ModelLayout::from_spec(&spec);
        let bindings = vec![
            binding(TensorClass::TokenEmbedding, TensorRole::TokenEmbedding, 1),
            binding(TensorClass::OutputNorm, TensorRole::OutputNorm, 1),
            binding(TensorClass::OutputHead, TensorRole::OutputHead, 1),
            binding(TensorClass::LayerNorm, TensorRole::LayerNorm, 1),
            binding(TensorClass::MlaQueryA, TensorRole::AttentionLatentQueryA, 2),
            binding(TensorClass::MlaQueryB, TensorRole::AttentionLatentQueryB, 2),
            binding(TensorClass::MlaKv, TensorRole::AttentionLatentKv, 2),
            binding(
                TensorClass::MlaOutputA,
                TensorRole::AttentionLatentOutputA,
                2,
            ),
            binding(
                TensorClass::MlaOutputB,
                TensorRole::AttentionLatentOutputB,
                2,
            ),
            binding(TensorClass::Router, TensorRole::RouterLogits, 1),
            binding(TensorClass::HashRouterTable, TensorRole::HashRouterTable, 1),
            binding(
                TensorClass::RoutedExpertGate,
                TensorRole::RoutedExpertGate,
                256,
            ),
            binding(TensorClass::RoutedExpertUp, TensorRole::RoutedExpertUp, 256),
            binding(
                TensorClass::RoutedExpertDown,
                TensorRole::RoutedExpertDown,
                256,
            ),
            binding(
                TensorClass::SharedExpertGate,
                TensorRole::SharedExpertGate,
                1,
            ),
            binding(TensorClass::SharedExpertUp, TensorRole::SharedExpertUp, 1),
            binding(
                TensorClass::SharedExpertDown,
                TensorRole::SharedExpertDown,
                1,
            ),
        ];
        let report = validate_model_layout_bindings(&layout, &bindings);
        assert!(report
            .missing_required
            .iter()
            .any(|missing| missing.scope == RoleScope::Layer { index: 0 }
                && missing.role == TensorRole::AttentionSink));
    }

    fn binding(source_class: TensorClass, role: TensorRole, tensors: usize) -> TensorBinding {
        TensorBinding {
            source_class,
            role,
            tensors,
        }
    }
}
