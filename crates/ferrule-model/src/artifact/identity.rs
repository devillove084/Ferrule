use std::fmt;

use crate::spec::ModelFamily;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactFormat {
    HfSafetensors,
    Gguf,
    WeightPack,
}

impl ArtifactFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HfSafetensors => "hf_safetensors",
            Self::Gguf => "gguf",
            Self::WeightPack => "weightpack",
        }
    }
}

impl fmt::Display for ArtifactFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIdentity {
    pub format: ArtifactFormat,
    pub family: ModelFamily,
    pub name: String,
    pub revision: Option<String>,
}

impl ArtifactIdentity {
    pub fn new(
        format: ArtifactFormat,
        family: ModelFamily,
        name: impl Into<String>,
        revision: Option<String>,
    ) -> Self {
        Self {
            format,
            family,
            name: name.into(),
            revision,
        }
    }
}
