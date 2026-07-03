use super::*;
use crate::spec::ModelFamily;

#[test]
fn deepseek_flash_dspark_official_artifact_records_input_metadata() {
    let input_artifact = InputArtifact::deepseek_v4_flash_dspark_official();
    let InputArtifact::HfSafetensors(artifact) = input_artifact else {
        panic!("expected HF safetensors artifact");
    };

    assert_eq!(artifact.repo_id, "deepseek-ai/DeepSeek-V4-Flash-DSpark");
    assert_eq!(artifact.family, ModelFamily::DeepSeekV4);
    assert_eq!(artifact.model_type.as_deref(), Some("deepseek_v4"));
    assert_eq!(
        artifact.architecture.as_deref(),
        Some("DeepseekV4ForCausalLM")
    );
    assert_eq!(artifact.shard_count, Some(48));
    assert!(artifact.has_dspark_attachment);
    assert!(artifact.has_file("model.safetensors.index.json"));
    assert!(artifact.has_file("encoding/encoding_dsv4.py"));
    assert!(artifact.has_file("inference/generate.py"));
    assert!(artifact.declared_precision.contains(&"F8_E4M3".to_string()));
    assert!(artifact.declared_precision.contains(&"I8".to_string()));
}

#[test]
fn artifact_identity_keeps_format_separate_from_family() {
    let input_artifact = InputArtifact::deepseek_v4_flash_dspark_official();
    let identity = input_artifact.identity();
    assert_eq!(identity.format, ArtifactFormat::HfSafetensors);
    assert_eq!(identity.family, ModelFamily::DeepSeekV4);
    assert_eq!(identity.name, "deepseek-ai/DeepSeek-V4-Flash-DSpark");
}
