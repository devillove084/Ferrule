#![allow(clippy::needless_range_loop)]
//! Model metadata, OLMoE weights, and tokenizer utilities.
pub mod artifact;
pub mod config;
pub mod conversion;
pub mod cpu_forward;
pub mod descriptor;
pub mod families;
pub mod loader;
pub mod spec;
pub mod support;
pub mod tensor_policy;
pub mod weights;

// Re-exports — keep the same public API surface while adding generic model metadata.
pub use artifact::{
    ArtifactFormat, ArtifactIdentity, DtypeCount, HfAttentionTensorInfo, HfFilePurpose,
    HfHyperConnectionTensorInfo, HfRepoFile, HfRoutedExpertTensorInfo, HfRouterTensorInfo,
    HfSafetensorsArtifact, HfSafetensorsIndex, HfSafetensorsInventory, HfSafetensorsShardSummary,
    HfSafetensorsTensorInfo, HfSharedExpertTensorInfo, SourceArtifact, TensorRoleCount,
};
pub use config::OlmoeConfig;
pub use conversion::{
    ArtifactTarget, CalibrationSet, ConversionPlan, QuantizationFormat, QuantizationRecipe,
    TensorRoleQuantPolicy,
};
pub use cpu_forward::rms_norm;
pub use descriptor::ModelDescriptor;
pub use spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSpec,
    WeightSource,
};
pub use support::{
    validate_model_layout_bindings, AttentionLayout, AttentionPolicy, BoundRoleCount, EnginePlan,
    EnginePlanStatus, ExpertPolicy, FeedForwardKind, FeedForwardLayout, KvCacheShape, KvPolicy,
    LayerLayout, LayoutValidationReport, MissingPolicy, MissingRequiredRole, ModelLayout,
    ModelSupportContract, OptionalRoleStatus, ParallelismPlan, PolicyArea, PolicySet, QuantPolicy,
    ResidencyPolicy, RoleScope, RouterPolicy, SpeculationMode, SpeculationPolicy, TensorBinding,
    TensorRole, TokenizerPolicy, ValidationPolicy,
};
pub use tensor_policy::{GgufTensorPolicy, HfTensorPolicy, TensorClass, TensorClassCount};
pub use weights::{AttnWeights, ExpertWeights, LayerWeights, LinearWeight};

use ferrule_core::Result;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct OlmoeModel {
    pub config: OlmoeConfig,
    pub embed: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub model_dir: PathBuf,
    pub(crate) tokenizer: Tokenizer,
}

impl OlmoeModel {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ferrule_core::Error::Model(format!("encode: {e}")))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| ferrule_core::Error::Model(format!("decode: {e}")))
    }

    pub fn transformer_spec(&self) -> TransformerSpec {
        TransformerSpec::from_olmoe_config(&self.config, WeightSource::Safetensors)
    }

    /// Extract the tokenizer, consuming the model.
    pub fn into_tokenizer(self) -> Tokenizer {
        self.tokenizer
    }
}
