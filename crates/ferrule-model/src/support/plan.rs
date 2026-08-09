use std::fmt;

use crate::execution::ModelExecutionBackend;
use crate::spec::{AttentionKind, ModelFamily, RouterKind, WeightSource};

use super::contract::ModelSupportContract;
use super::policies::{PolicySet, SpeculationMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnginePlanStatus {
    Executable,
    MetadataOnly,
    Unsupported,
}

impl EnginePlanStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Executable => "executable",
            Self::MetadataOnly => "metadata-only",
            Self::Unsupported => "unsupported",
        }
    }
}

impl fmt::Display for EnginePlanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyArea {
    ModelFamily,
    Backend,
    Attention,
    Router,
    Expert,
    Quantization,
    Kv,
    Tokenizer,
    Residency,
    Speculation,
    Validation,
}

impl PolicyArea {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ModelFamily => "model_family",
            Self::Backend => "backend",
            Self::Attention => "attention",
            Self::Router => "router",
            Self::Expert => "expert",
            Self::Quantization => "quantization",
            Self::Kv => "kv",
            Self::Tokenizer => "tokenizer",
            Self::Residency => "residency",
            Self::Speculation => "speculation",
            Self::Validation => "validation",
        }
    }
}

impl fmt::Display for PolicyArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingPolicy {
    pub area: PolicyArea,
    pub reason: String,
}

impl MissingPolicy {
    pub fn new(area: PolicyArea, reason: impl Into<String>) -> Self {
        Self {
            area,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnginePlan {
    pub family: ModelFamily,
    pub architecture: Option<String>,
    pub backend: ModelExecutionBackend,
    /// Truthful implementation profile, when the selected backend is available.
    pub backend_profile: Option<&'static str>,
    pub status: EnginePlanStatus,
    pub policies: PolicySet,
    pub missing: Vec<MissingPolicy>,
}

impl EnginePlan {
    /// Build a plan for the always-available CPU reference backend.
    pub fn from_contract(contract: &ModelSupportContract) -> Self {
        Self::from_contract_for_backend(contract, ModelExecutionBackend::Cpu)
    }

    pub fn from_contract_for_backend(
        contract: &ModelSupportContract,
        backend: ModelExecutionBackend,
    ) -> Self {
        let missing = missing_policies(contract, backend);
        let status = if missing.is_empty() {
            EnginePlanStatus::Executable
        } else if matches!(
            contract.spec.family,
            ModelFamily::Unknown(_) | ModelFamily::Qwen3 | ModelFamily::QwenMoe
        ) {
            EnginePlanStatus::Unsupported
        } else {
            EnginePlanStatus::MetadataOnly
        };
        Self {
            family: contract.spec.family.clone(),
            architecture: contract.spec.architecture.clone(),
            backend,
            backend_profile: backend_profile(&contract.spec.family, backend),
            status,
            policies: contract.policies.clone(),
            missing,
        }
    }

    pub fn is_executable(&self) -> bool {
        matches!(self.status, EnginePlanStatus::Executable)
    }
}

fn backend_profile(family: &ModelFamily, backend: ModelExecutionBackend) -> Option<&'static str> {
    match (family, backend) {
        (ModelFamily::QwenMoe, ModelExecutionBackend::Cpu) => Some("cpu-reference"),
        (ModelFamily::QwenMoe, ModelExecutionBackend::Cuda) if cfg!(feature = "cuda") => {
            Some("cuda-correctness-nonresident")
        }
        (ModelFamily::DeepSeekV4, ModelExecutionBackend::Cuda) if cfg!(feature = "cuda") => {
            Some("cuda")
        }
        _ => None,
    }
}

fn missing_policies(
    contract: &ModelSupportContract,
    backend: ModelExecutionBackend,
) -> Vec<MissingPolicy> {
    let spec = &contract.spec;
    let mut missing = Vec::new();

    if !spec.family.is_supported_runtime_family() {
        let reason = match &spec.family {
            ModelFamily::Qwen3 => {
                "dense Qwen3 has no registered runtime runner; only Qwen3-MoE is supported".into()
            }
            ModelFamily::Unknown(_) => {
                "unknown model family has no descriptor-to-layout binding".into()
            }
            family => format!(
                "{family} descriptor is recognized, but no executable model-family policy is wired yet"
            ),
        };
        missing.push(MissingPolicy::new(PolicyArea::ModelFamily, reason));
    }

    if spec.family.is_supported_runtime_family()
        && !matches!(spec.weight_source, WeightSource::Safetensors)
    {
        missing.push(MissingPolicy::new(
            PolicyArea::Quantization,
            format!(
                "{} runtime loading requires Hugging Face safetensors, got {}",
                spec.family, spec.weight_source
            ),
        ));
    }

    match (&spec.family, backend) {
        (ModelFamily::QwenMoe, ModelExecutionBackend::Cpu) => {}
        (ModelFamily::QwenMoe, ModelExecutionBackend::Cuda) if cfg!(feature = "cuda") => {}
        (ModelFamily::QwenMoe, ModelExecutionBackend::Cuda) => {
            missing.push(MissingPolicy::new(
                PolicyArea::Backend,
                "Qwen3-MoE CUDA correctness-nonresident execution requires the cuda feature",
            ));
        }
        (ModelFamily::DeepSeekV4, ModelExecutionBackend::Cuda) if cfg!(feature = "cuda") => {}
        (ModelFamily::DeepSeekV4, ModelExecutionBackend::Cuda) => {
            missing.push(MissingPolicy::new(
                PolicyArea::Backend,
                "DeepSeek-V4 CUDA execution requires the cuda feature",
            ));
        }
        (ModelFamily::DeepSeekV4, ModelExecutionBackend::Cpu) => {
            missing.push(MissingPolicy::new(
                PolicyArea::Backend,
                "DeepSeek-V4 has no registered CPU resident backend",
            ));
        }
        _ => {}
    }

    if matches!(spec.family, ModelFamily::QwenMoe) {
        if !spec.moe.is_moe() {
            missing.push(MissingPolicy::new(
                PolicyArea::Expert,
                "Qwen3-MoE runtime requires routed experts",
            ));
        }
        if !matches!(spec.moe.router, RouterKind::DenseTopK) {
            missing.push(MissingPolicy::new(
                PolicyArea::Router,
                "Qwen3-MoE runtime requires dense top-k routing",
            ));
        }
    }

    if matches!(spec.attention, AttentionKind::MultiLatentAttention) {
        missing.push(MissingPolicy::new(
            PolicyArea::Attention,
            "latent/compressed attention requires a dedicated attention policy, KV shape, and kernels",
        ));
    } else if matches!(spec.attention, AttentionKind::Unknown(_)) {
        missing.push(MissingPolicy::new(
            PolicyArea::Attention,
            "unknown attention kind has no execution policy",
        ));
    }

    if !matches!(spec.family, ModelFamily::QwenMoe)
        && matches!(spec.moe.router, RouterKind::HashAssistedTopK)
    {
        missing.push(MissingPolicy::new(
            PolicyArea::Router,
            "hash-assisted routing tables require router policy integration",
        ));
    }

    if spec.moe.has_shared_experts && !matches!(spec.family, ModelFamily::DeepSeekV4) {
        missing.push(MissingPolicy::new(
            PolicyArea::Expert,
            "shared experts require an explicit shared-expert execution policy",
        ));
    }

    if contract.policies.quant.has_gguf_quantized_tensors() {
        missing.push(MissingPolicy::new(
            PolicyArea::Quantization,
            "GGUF quantized tensors require native dequant/matvec kernels or an explicit conversion policy",
        ));
    }

    if contract
        .tensor_bindings
        .iter()
        .any(|binding| binding.role.is_attention_auxiliary())
    {
        missing.push(MissingPolicy::new(
            PolicyArea::Attention,
            "attention sink tensors require explicit attention policy semantics before execution",
        ));
    }

    if contract
        .tensor_bindings
        .iter()
        .any(|binding| binding.role.is_auxiliary())
    {
        missing.push(MissingPolicy::new(
            PolicyArea::Attention,
            "auxiliary attention/residual tensors require family policy semantics before execution",
        ));
    }

    if matches!(
        contract.policies.speculation.mode,
        SpeculationMode::MultiTokenPrediction | SpeculationMode::DraftModel
    ) || contract
        .tensor_bindings
        .iter()
        .any(|binding| binding.role.is_speculative())
    {
        missing.push(MissingPolicy::new(
            PolicyArea::Speculation,
            "speculative decoding attachments require scheduler, verifier, and draft-token acceptance policies",
        ));
    }

    if contract.policies.tokenizer.requires_external_encoding {
        missing.push(MissingPolicy::new(
            PolicyArea::Tokenizer,
            "model requires external tokenizer/encoding policy before chat/run support",
        ));
    }

    missing
}
