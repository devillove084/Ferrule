use std::path::PathBuf;

use crate::spec::ModelFamily;

use super::hf::HfSafetensorsArtifact;
use super::identity::{ArtifactFormat, ArtifactIdentity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputArtifact {
    HfSafetensors(HfSafetensorsArtifact),
    Gguf { path: PathBuf, family: ModelFamily },
    WeightPack { path: PathBuf, family: ModelFamily },
}

impl InputArtifact {
    pub fn deepseek_v4_flash_dspark_official() -> Self {
        Self::HfSafetensors(HfSafetensorsArtifact::deepseek_v4_flash_dspark_official())
    }

    pub fn identity(&self) -> ArtifactIdentity {
        match self {
            Self::HfSafetensors(artifact) => artifact.identity(),
            Self::Gguf { path, family } => ArtifactIdentity::new(
                ArtifactFormat::Gguf,
                family.clone(),
                path.display().to_string(),
                None,
            ),
            Self::WeightPack { path, family } => ArtifactIdentity::new(
                ArtifactFormat::WeightPack,
                family.clone(),
                path.display().to_string(),
                None,
            ),
        }
    }

    pub fn family(&self) -> ModelFamily {
        self.identity().family
    }
}
