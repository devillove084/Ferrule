use super::*;
use crate::spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSpec,
    WeightSource,
};
use crate::support::SpeculationMode;
use crate::tensor_policy::{TensorClass, TensorClassCount};

fn dense_llama_spec() -> TransformerSpec {
    TransformerSpec {
        family: ModelFamily::Llama,
        architecture: Some("llama".into()),
        weight_source: WeightSource::Safetensors,
        hidden_size: Some(4096),
        num_layers: Some(2),
        vocab_size: Some(32000),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        head_dim: Some(128),
        attention: AttentionKind::GroupedQuery,
        moe: MoeSpec::none(),
        semantics: Default::default(),
        tensor_count: None,
        quantization: Vec::new(),
        notes: Vec::new(),
    }
}

fn qwen_moe_spec(shared: bool) -> TransformerSpec {
    TransformerSpec {
        family: ModelFamily::QwenMoe,
        architecture: Some("qwen2_moe".into()),
        weight_source: WeightSource::Safetensors,
        hidden_size: Some(2048),
        num_layers: Some(2),
        vocab_size: Some(151936),
        num_heads: Some(16),
        num_kv_heads: Some(16),
        head_dim: Some(128),
        attention: AttentionKind::DenseMha,
        moe: MoeSpec {
            num_experts: Some(64),
            num_experts_per_tok: Some(4),
            has_shared_experts: shared,
            router: RouterKind::DenseTopK,
        },
        semantics: Default::default(),
        tensor_count: None,
        quantization: Vec::new(),
        notes: Vec::new(),
    }
}

fn deepseek_spec() -> TransformerSpec {
    TransformerSpec {
        family: ModelFamily::DeepSeekV4,
        architecture: Some("deepseek4".into()),
        weight_source: WeightSource::Gguf,
        hidden_size: Some(7168),
        num_layers: Some(3),
        vocab_size: Some(129280),
        num_heads: Some(128),
        num_kv_heads: None,
        head_dim: None,
        attention: AttentionKind::MultiLatentAttention,
        moe: MoeSpec {
            num_experts: Some(256),
            num_experts_per_tok: Some(8),
            has_shared_experts: true,
            router: RouterKind::HashAssistedTopK,
        },
        semantics: Default::default(),
        tensor_count: Some(12),
        quantization: vec![
            QuantFormatCount {
                format: "IQ2_XXS".into(),
                tensors: 4,
            },
            QuantFormatCount {
                format: "Q4_K".into(),
                tensors: 8,
            },
        ],
        notes: Vec::new(),
    }
}

#[test]
fn model_support_contract_dense_layout_is_not_deepseek_shaped() {
    let spec = dense_llama_spec();
    let contract = ModelSupportContract::from_spec(&spec, &[]);
    assert_eq!(contract.layout.layer_count(), 2);
    let layer = &contract.layout.layers[0];
    assert_eq!(layer.attention.kind, AttentionKind::GroupedQuery);
    assert_eq!(layer.attention.kv_shape, KvCacheShape::GroupedKeysValues);
    assert!(
        layer
            .attention
            .required_roles
            .contains(&TensorRole::AttentionQuery)
    );
    assert!(
        layer
            .attention
            .required_roles
            .contains(&TensorRole::AttentionOutput)
    );
    assert!(
        !layer
            .attention
            .required_roles
            .iter()
            .any(TensorRole::is_attention_latent)
    );
    assert_eq!(layer.feed_forward.kind, FeedForwardKind::DenseMlp);
    assert!(
        layer
            .feed_forward
            .required_roles
            .contains(&TensorRole::DenseMlpGate)
    );

    let plan = contract.engine_plan();
    assert_eq!(plan.status, EnginePlanStatus::MetadataOnly);
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::ModelFamily)
    );
}

#[test]
fn model_support_contract_moe_layout_expresses_shared_experts_generically() {
    let spec = qwen_moe_spec(true);
    let contract = ModelSupportContract::from_spec(&spec, &[]);
    let layer = &contract.layout.layers[0];
    assert_eq!(
        layer.feed_forward.kind,
        FeedForwardKind::RoutedAndSharedExperts
    );
    assert!(
        layer
            .feed_forward
            .required_roles
            .contains(&TensorRole::RouterLogits)
    );
    assert!(
        layer
            .feed_forward
            .required_roles
            .contains(&TensorRole::RoutedExpertGate)
    );
    assert!(
        layer
            .feed_forward
            .required_roles
            .contains(&TensorRole::SharedExpertDown)
    );
    assert!(
        !layer
            .feed_forward
            .required_roles
            .contains(&TensorRole::HashRouterTable)
    );
}

#[test]
fn model_support_contract_deepseek_binding_stays_semantic() {
    let spec = deepseek_spec();
    let classes = vec![
        TensorClassCount {
            class: TensorClass::MlaQueryA,
            tensors: 3,
        },
        TensorClassCount {
            class: TensorClass::MlaQueryNorm,
            tensors: 3,
        },
        TensorClassCount {
            class: TensorClass::MlaKv,
            tensors: 3,
        },
        TensorClassCount {
            class: TensorClass::MlaKvNorm,
            tensors: 3,
        },
        TensorClassCount {
            class: TensorClass::HashRouterTable,
            tensors: 1,
        },
        TensorClassCount {
            class: TensorClass::OutputHiddenCompressor,
            tensors: 1,
        },
    ];
    let contract = ModelSupportContract::from_spec(&spec, &classes);
    let roles = contract.bound_roles();
    assert!(roles.contains(&TensorRole::AttentionLatentQueryA));
    assert!(roles.contains(&TensorRole::AttentionQueryNorm));
    assert!(roles.contains(&TensorRole::AttentionLatentKv));
    assert!(roles.contains(&TensorRole::AttentionKeyValueNorm));
    assert!(roles.contains(&TensorRole::HashRouterTable));
    assert!(roles.contains(&TensorRole::AuxOutputHiddenCompressor));
    assert!(roles.iter().all(|role| !role.as_str().contains("deepseek")));
    assert!(contract.policies.residency.streaming_allowed);
    assert!(!contract.policies.residency.all_resident_required);

    let plan = contract.engine_plan();
    assert_eq!(plan.status, EnginePlanStatus::MetadataOnly);
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::Attention)
    );
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::Quantization)
    );
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::Router)
    );
}

#[test]
fn model_support_contract_dspark_tensors_select_mtp_speculation_policy() {
    let spec = deepseek_spec();
    let classes = vec![
        TensorClassCount {
            class: TensorClass::AttentionSink,
            tensors: 46,
        },
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
    ];
    let contract = ModelSupportContract::from_spec(&spec, &classes);
    let roles = contract.bound_roles();
    assert!(roles.contains(&TensorRole::AttentionSink));
    assert!(roles.contains(&TensorRole::SpeculativeProjection));
    assert!(roles.contains(&TensorRole::SpeculativeMarkovHead));
    assert!(roles.contains(&TensorRole::SpeculativeConfidenceHead));
    assert_eq!(
        contract.policies.speculation.mode,
        SpeculationMode::MultiTokenPrediction
    );

    let plan = contract.engine_plan();
    assert_eq!(plan.status, EnginePlanStatus::MetadataOnly);
    assert!(
        plan.missing.iter().any(
            |item| item.area == PolicyArea::Attention && item.reason.contains("attention sink")
        )
    );
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::Speculation)
    );
}

#[test]
fn model_support_contract_deepseek_v4_safetensors_is_executable() {
    let spec = TransformerSpec {
        family: ModelFamily::DeepSeekV4,
        architecture: Some("deepseekv4".into()),
        weight_source: WeightSource::Safetensors,
        hidden_size: Some(4096),
        num_layers: Some(43),
        vocab_size: Some(129280),
        num_heads: Some(32),
        num_kv_heads: Some(32),
        head_dim: Some(128),
        attention: AttentionKind::MultiLatentAttention,
        moe: MoeSpec {
            num_experts: Some(256),
            num_experts_per_tok: Some(6),
            has_shared_experts: false,
            router: RouterKind::DenseTopK,
        },
        semantics: Default::default(),
        tensor_count: None,
        quantization: Vec::new(),
        notes: Vec::new(),
    };
    let contract = ModelSupportContract::from_spec(&spec, &[]);
    let plan = contract.engine_plan();
    // DSV4 is recognized but MLA attention is flagged as missing policy
    assert!(!plan.missing.is_empty());
}

#[test]
fn model_support_contract_unknown_family_is_unsupported() {
    let mut spec = dense_llama_spec();
    spec.family = ModelFamily::Unknown("mystery".into());
    let contract = ModelSupportContract::from_spec(&spec, &[]);
    let plan = contract.engine_plan();
    assert_eq!(plan.status, EnginePlanStatus::Unsupported);
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == PolicyArea::ModelFamily)
    );
}
