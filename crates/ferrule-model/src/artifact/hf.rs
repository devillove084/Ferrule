use std::path::PathBuf;

use crate::spec::ModelFamily;

use super::identity::{ArtifactFormat, ArtifactIdentity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRepoFile {
    pub path: String,
    pub required_for: HfFilePurpose,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfFilePurpose {
    Descriptor,
    TensorIndex,
    TensorShard,
    Tokenizer,
    Encoding,
    ReferenceInference,
    License,
    Documentation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSafetensorsArtifact {
    pub repo_id: String,
    pub revision: Option<String>,
    pub family: ModelFamily,
    pub model_type: Option<String>,
    pub architecture: Option<String>,
    pub local_dir: Option<PathBuf>,
    pub shard_count: Option<usize>,
    pub declared_precision: Vec<String>,
    pub files: Vec<HfRepoFile>,
    pub has_dspark_attachment: bool,
}

impl HfSafetensorsArtifact {
    pub fn identity(&self) -> ArtifactIdentity {
        ArtifactIdentity::new(
            ArtifactFormat::HfSafetensors,
            self.family.clone(),
            self.repo_id.clone(),
            self.revision.clone(),
        )
    }

    pub fn has_file(&self, path: &str) -> bool {
        self.files.iter().any(|file| file.path == path)
    }

    pub fn required_files_for(&self, purpose: HfFilePurpose) -> Vec<&HfRepoFile> {
        self.files
            .iter()
            .filter(|file| file.required_for == purpose)
            .collect()
    }

    /// Official checkpoint artifact metadata for `deepseek-ai/DeepSeek-V4-Flash-DSpark`.
    ///
    /// This is a canonical checkpoint entry, not a Ferrule execution format.
    /// It should be converted into semantic tensor bindings and then into
    /// WeightPack/GGUF through an explicit conversion plan.
    pub fn deepseek_v4_flash_dspark_official() -> Self {
        let repo_id = "deepseek-ai/DeepSeek-V4-Flash-DSpark".to_string();
        let revision = Some("913f0657a874f76844e2e91cbe706dbcaceeb6d7".to_string());
        let mut files = vec![
            file("config.json", HfFilePurpose::Descriptor),
            file("generation_config.json", HfFilePurpose::Descriptor),
            file("model.safetensors.index.json", HfFilePurpose::TensorIndex),
            file("tokenizer.json", HfFilePurpose::Tokenizer),
            file("tokenizer_config.json", HfFilePurpose::Tokenizer),
            file("encoding/README.md", HfFilePurpose::Encoding),
            file("encoding/encoding_dsv4.py", HfFilePurpose::Encoding),
            file("encoding/test_encoding_dsv4.py", HfFilePurpose::Encoding),
            file("inference/README.md", HfFilePurpose::ReferenceInference),
            file("inference/config.json", HfFilePurpose::ReferenceInference),
            file("inference/convert.py", HfFilePurpose::ReferenceInference),
            file("inference/generate.py", HfFilePurpose::ReferenceInference),
            file("inference/kernel.py", HfFilePurpose::ReferenceInference),
            file("inference/model.py", HfFilePurpose::ReferenceInference),
            file("LICENSE", HfFilePurpose::License),
            file("README.md", HfFilePurpose::Documentation),
        ];
        files.extend((1..=48).map(|idx| {
            file(
                format!("model-{idx:05}-of-00048.safetensors"),
                HfFilePurpose::TensorShard,
            )
        }));
        Self {
            repo_id,
            revision,
            family: ModelFamily::DeepSeekV4,
            model_type: Some("deepseek_v4".into()),
            architecture: Some("DeepseekV4ForCausalLM".into()),
            local_dir: None,
            shard_count: Some(48),
            declared_precision: vec![
                "BF16".into(),
                "F32".into(),
                "F8_E8M0".into(),
                "F8_E4M3".into(),
                "I8".into(),
                "I64".into(),
            ],
            files,
            has_dspark_attachment: true,
        }
    }
}

fn file(path: impl Into<String>, required_for: HfFilePurpose) -> HfRepoFile {
    HfRepoFile {
        path: path.into(),
        required_for,
    }
}
