use super::*;
use crate::artifact::SourceArtifact;
use crate::spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSpec,
    WeightSource,
};
use crate::support::{ModelSupportContract, TensorRole};

fn deepseek_flash_contract() -> ModelSupportContract {
    let spec = TransformerSpec {
        family: ModelFamily::DeepSeekV4,
        architecture: Some("DeepseekV4ForCausalLM".into()),
        weight_source: WeightSource::Safetensors,
        hidden_size: Some(7168),
        num_layers: Some(2),
        vocab_size: Some(129280),
        num_heads: Some(128),
        num_kv_heads: None,
        head_dim: None,
        attention: AttentionKind::MultiLatentAttention,
        moe: MoeSpec {
            num_experts: Some(256),
            num_experts_per_tok: Some(6),
            has_shared_experts: true,
            router: RouterKind::HashAssistedTopK,
        },
        tensor_count: None,
        quantization: vec![
            QuantFormatCount {
                format: "F8_E4M3".into(),
                tensors: 12,
            },
            QuantFormatCount {
                format: "I8".into(),
                tensors: 24,
            },
        ],
        notes: Vec::new(),
    };
    ModelSupportContract::from_spec(&spec, &[])
}

#[test]
fn deepseek_weightpack_recipe_keeps_weightpack_as_primary_target() {
    let recipe = QuantizationRecipe::deepseek_v4_flash_weightpack_mixed_v1();
    assert_eq!(
        recipe.name,
        "deepseek-v4-flash-weightpack-source-fp4-streaming-v1"
    );
    assert_eq!(
        recipe
            .policy_for_role(&TensorRole::RoutedExpertGate)
            .map(|policy| &policy.format),
        Some(&QuantizationFormat::PreserveSource)
    );
    assert_eq!(
        recipe
            .policy_for_role(&TensorRole::RouterBias)
            .map(|policy| &policy.format),
        Some(&QuantizationFormat::F32)
    );
    assert!(recipe
        .notes
        .iter()
        .any(|note| note.contains("WeightPack is the primary")));
    assert!(recipe
        .notes
        .iter()
        .any(|note| note.contains("expert streaming/residency")));
}

#[test]
fn deepseek_dgxspark_resident_recipe_uses_two_bit_class_experts() {
    let recipe = QuantizationRecipe::deepseek_v4_flash_dgxspark_resident_iq2_v1();
    assert_eq!(recipe.name, "deepseek-v4-flash-dgxspark-resident-iq2-v1");
    for role in [
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::RoutedExpertDown,
    ] {
        assert_eq!(
            recipe.policy_for_role(&role).map(|policy| &policy.format),
            Some(&QuantizationFormat::Iq2Xxs)
        );
    }
    assert_eq!(
        recipe.calibration.as_ref().map(|set| set.prompt_count),
        Some(64)
    );
    assert!(recipe
        .notes
        .iter()
        .any(|note| note.contains("first end-to-end smoke")));
}

#[test]
fn conversion_plan_links_official_hf_source_to_weightpack_target() {
    let contract = deepseek_flash_contract();
    let source = SourceArtifact::deepseek_v4_flash_dspark_official();
    let plan = ConversionPlan::new(
        &source,
        &contract,
        ArtifactTarget::WeightPack,
        QuantizationRecipe::deepseek_v4_flash_weightpack_mixed_v1(),
    );
    assert_eq!(plan.source.name, "deepseek-ai/DeepSeek-V4-Flash-DSpark");
    assert_eq!(plan.family, ModelFamily::DeepSeekV4);
    assert_eq!(plan.target, ArtifactTarget::WeightPack);
    assert!(plan.requires_reference_validation);
    assert!(plan.output_name.ends_with(".weightpack"));
    assert!(plan
        .notes
        .iter()
        .any(|note| note.contains("source of truth")));
}

#[test]
fn gguf_conversion_plan_is_explicitly_compatibility_target() {
    let contract = deepseek_flash_contract();
    let source = SourceArtifact::deepseek_v4_flash_dspark_official();
    let plan = ConversionPlan::new(
        &source,
        &contract,
        ArtifactTarget::Gguf,
        QuantizationRecipe::deepseek_v4_flash_weightpack_mixed_v1(),
    );
    assert_eq!(plan.target, ArtifactTarget::Gguf);
    assert!(plan.output_name.ends_with(".gguf"));
    assert!(plan
        .notes
        .iter()
        .any(|note| note.contains("compatibility/PK")));
}
