use crate::artifact::{ArtifactFormat, ArtifactIdentity, SourceArtifact};
use crate::spec::ModelFamily;
use crate::support::{EnginePlanStatus, ModelSupportContract};

use super::recipe::{ArtifactTarget, QuantizationRecipe};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPlan {
    pub source: ArtifactIdentity,
    pub family: ModelFamily,
    pub target: ArtifactTarget,
    pub recipe: QuantizationRecipe,
    pub output_name: String,
    pub requires_reference_validation: bool,
    pub notes: Vec<String>,
}

impl ConversionPlan {
    pub fn new(
        source: &SourceArtifact,
        contract: &ModelSupportContract,
        target: ArtifactTarget,
        recipe: QuantizationRecipe,
    ) -> Self {
        let source_identity = source.identity();
        let engine_plan = contract.engine_plan();
        let output_name = format!(
            "{}-{}.{}",
            sanitize_name(&source_identity.name),
            recipe.name,
            target.as_str()
        );
        let mut notes = Vec::new();
        if matches!(source_identity.format, ArtifactFormat::HfSafetensors)
            && matches!(target, ArtifactTarget::WeightPack)
        {
            notes.push("HF safetensors are the source of truth; WeightPack is Ferrule's execution artifact".into());
        }
        if matches!(target, ArtifactTarget::Gguf) {
            notes.push("GGUF target is for compatibility/PK; do not make it the only internal execution format".into());
        }
        if !matches!(engine_plan.status, EnginePlanStatus::Executable) {
            notes.push(format!(
                "source model is currently {}; conversion can produce artifacts before execution is supported",
                engine_plan.status
            ));
        }
        Self {
            source: source_identity,
            family: contract.spec.family.clone(),
            target,
            recipe,
            output_name,
            requires_reference_validation: contract.policies.validation.requires_reference_engine,
            notes,
        }
    }

    pub fn deepseek_v4_flash_dspark_weightpack_plan(contract: &ModelSupportContract) -> Self {
        let source = SourceArtifact::deepseek_v4_flash_dspark_official();
        Self::new(
            &source,
            contract,
            ArtifactTarget::WeightPack,
            QuantizationRecipe::deepseek_v4_flash_weightpack_mixed_v1(),
        )
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod local_tests {
    use super::*;

    #[test]
    fn sanitize_name_removes_path_delimiters() {
        assert_eq!(
            sanitize_name("deepseek-ai/DeepSeek-V4-Flash-DSpark"),
            "deepseek-ai-deepseek-v4-flash-dspark"
        );
    }
}
