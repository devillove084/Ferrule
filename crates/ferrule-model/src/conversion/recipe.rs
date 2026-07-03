use std::fmt;

use crate::support::TensorRole;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactTarget {
    WeightPack,
    Gguf,
}

impl ArtifactTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WeightPack => "weightpack",
            Self::Gguf => "gguf",
        }
    }
}

impl fmt::Display for ArtifactTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationFormat {
    PreserveArtifact,
    F32,
    F16,
    Bf16,
    Fp8E4M3,
    Fp8E8M0,
    Q8_0,
    Q4_0,
    Q4K,
    Q2K,
    Iq2Xxs,
    Unsupported(String),
}

impl QuantizationFormat {
    pub fn as_str(&self) -> &str {
        match self {
            Self::PreserveArtifact => "preserve_artifact",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E8M0 => "fp8_e8m0",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4K => "q4_k",
            Self::Q2K => "q2_k",
            Self::Iq2Xxs => "iq2_xxs",
            Self::Unsupported(name) => name.as_str(),
        }
    }
}

impl fmt::Display for QuantizationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRoleQuantPolicy {
    pub role: TensorRole,
    pub format: QuantizationFormat,
    pub reason: String,
}

impl TensorRoleQuantPolicy {
    pub fn new(role: TensorRole, format: QuantizationFormat, reason: impl Into<String>) -> Self {
        Self {
            role,
            format,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationSet {
    pub name: String,
    pub prompt_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationRecipe {
    pub name: String,
    pub per_role: Vec<TensorRoleQuantPolicy>,
    pub calibration: Option<CalibrationSet>,
    pub notes: Vec<String>,
}

impl QuantizationRecipe {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            per_role: Vec::new(),
            calibration: None,
            notes: Vec::new(),
        }
    }

    /// Quality-first planning recipe for official DeepSeek-V4-Flash-DSpark checkpoint
    /// artifacts.
    ///
    /// The downloaded HF checkpoint is already aggressively mixed precision: routed
    /// experts are stored as I8 containers for official FP4 payloads, attention is
    /// mostly FP8 plus FP8 scales, and small correctness-critical tensors are
    /// BF16/F32. Re-quantizing the FP4 experts to another 4-bit format does not
    /// materially reduce memory and can only hurt quality. This recipe therefore
    /// preserves the artifact quantization and expects WeightPack residency/streaming
    /// to make it runnable on one DGX Spark.
    pub fn deepseek_v4_flash_weightpack_mixed_v1() -> Self {
        Self {
            name: "deepseek-v4-flash-weightpack-artifact-fp4-streaming-v1".into(),
            per_role: vec![
                TensorRoleQuantPolicy::new(
                    TensorRole::TokenEmbedding,
                    QuantizationFormat::Q8_0,
                    "embeddings are bandwidth-heavy but quality-sensitive",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentQueryA,
                    QuantizationFormat::PreserveArtifact,
                    "preserve official FP8/BF16 attention artifact until kernels are validated",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentQueryB,
                    QuantizationFormat::PreserveArtifact,
                    "preserve official FP8/BF16 attention artifact until kernels are validated",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionQueryNorm,
                    QuantizationFormat::F32,
                    "low-rank query norm is small and correctness-critical",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentKv,
                    QuantizationFormat::PreserveArtifact,
                    "KV compression semantics must be validated before lossy conversion",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionKeyValueNorm,
                    QuantizationFormat::F32,
                    "compressed KV norm is small and correctness-critical",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RouterLogits,
                    QuantizationFormat::F16,
                    "routing is correctness-critical and cheap relative to experts",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RouterBias,
                    QuantizationFormat::F32,
                    "small bias tensors should remain exact while router semantics are validated",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertGate,
                    QuantizationFormat::PreserveArtifact,
                    "official artifact stores routed experts as FP4 payloads in I8 containers; preserve and stream first",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertUp,
                    QuantizationFormat::PreserveArtifact,
                    "official artifact stores routed experts as FP4 payloads in I8 containers; preserve and stream first",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertDown,
                    QuantizationFormat::PreserveArtifact,
                    "official artifact stores routed experts as FP4 payloads in I8 containers; preserve and stream first",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertGate,
                    QuantizationFormat::Q8_0,
                    "shared experts are always available and should start conservative",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertUp,
                    QuantizationFormat::Q8_0,
                    "shared experts are always available and should start conservative",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertDown,
                    QuantizationFormat::Q8_0,
                    "shared experts are always available and should start conservative",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::OutputHead,
                    QuantizationFormat::Q8_0,
                    "vocab projection quality affects token choice directly",
                ),
            ],
            calibration: None,
            notes: vec![
                "WeightPack is the primary Ferrule execution artifact".into(),
                "default DeepSeek-V4 profile is quality-first: preserve official FP4/FP8 artifact format and rely on expert streaming/residency".into(),
                "GGUF export should be treated as compatibility/PK output".into(),
                "recipe must be validated against official DeepSeek reference outputs".into(),
            ],
        }
    }

    /// Single-DGX-Spark all-resident smoke recipe.
    ///
    /// This is not the default quality profile. It is the practical first target
    /// when we want a full model to fit in roughly 128GB unified memory without a
    /// second LAN node or expert streaming. It compresses the routed expert payload
    /// from artifact FP4-sized storage toward ~2-bit class storage and must be judged
    /// by reference prompts before any quality claim.
    pub fn deepseek_v4_flash_dgxspark_resident_iq2_v1() -> Self {
        Self {
            name: "deepseek-v4-flash-dgxspark-resident-iq2-v1".into(),
            per_role: vec![
                TensorRoleQuantPolicy::new(
                    TensorRole::TokenEmbedding,
                    QuantizationFormat::Q8_0,
                    "keep embeddings conservative while shrinking routed experts",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentQueryA,
                    QuantizationFormat::PreserveArtifact,
                    "attention is a small fraction of bytes; preserve artifact for first smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentQueryB,
                    QuantizationFormat::PreserveArtifact,
                    "attention is a small fraction of bytes; preserve artifact for first smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionQueryNorm,
                    QuantizationFormat::F32,
                    "low-rank query norm is small and correctness-critical",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionLatentKv,
                    QuantizationFormat::PreserveArtifact,
                    "KV compression semantics must be validated before lossy conversion",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::AttentionKeyValueNorm,
                    QuantizationFormat::F32,
                    "compressed KV norm is small and correctness-critical",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RouterLogits,
                    QuantizationFormat::F16,
                    "routing errors change expert selection and are too costly for the first smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RouterBias,
                    QuantizationFormat::F32,
                    "small bias tensors should remain exact",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertGate,
                    QuantizationFormat::Iq2Xxs,
                    "routed experts dominate artifact bytes; ~2-bit class storage is needed for all-resident DGX Spark smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertUp,
                    QuantizationFormat::Iq2Xxs,
                    "routed experts dominate artifact bytes; ~2-bit class storage is needed for all-resident DGX Spark smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::RoutedExpertDown,
                    QuantizationFormat::Iq2Xxs,
                    "routed experts dominate artifact bytes; ~2-bit class storage is needed for all-resident DGX Spark smoke",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertGate,
                    QuantizationFormat::Q8_0,
                    "shared experts are small and always active",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertUp,
                    QuantizationFormat::Q8_0,
                    "shared experts are small and always active",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::SharedExpertDown,
                    QuantizationFormat::Q8_0,
                    "shared experts are small and always active",
                ),
                TensorRoleQuantPolicy::new(
                    TensorRole::OutputHead,
                    QuantizationFormat::Q8_0,
                    "vocab projection quality affects token choice directly",
                ),
            ],
            calibration: Some(CalibrationSet {
                name: "deepseek-v4-flash-dgxspark-smoke-calibration-v1".into(),
                prompt_count: 64,
            }),
            notes: vec![
                "single-node DGX Spark resident profile: intended for first end-to-end smoke, not final quality".into(),
                "expected memory target is roughly artifact non-experts plus ~half-sized routed experts, before KV/workspace".into(),
                "must be validated against official DeepSeek reference prompts before speed claims".into(),
            ],
        }
    }

    pub fn policy_for_role(&self, role: &TensorRole) -> Option<&TensorRoleQuantPolicy> {
        self.per_role.iter().find(|policy| &policy.role == role)
    }
}
