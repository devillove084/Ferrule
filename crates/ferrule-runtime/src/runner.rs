use ferrule_core::{Error, Result};
use ferrule_model::{AttentionKind, EnginePlan, ModelDescriptor, ModelFamily, WeightSource};
use ferrule_quant::QuantType;

use std::path::Path;

// ── ModelInfo ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: ModelFamily,
    pub architecture: Option<String>,
    pub attention: AttentionKind,
    pub weight_source: WeightSource,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub backend: &'static str,
}

impl ModelInfo {
    pub fn from_descriptor(descriptor: &ModelDescriptor, backend: &'static str) -> Self {
        let spec = &descriptor.spec;
        Self {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            attention: spec.attention.clone(),
            weight_source: spec.weight_source,
            hidden_size: spec.hidden_size.unwrap_or(0),
            num_layers: spec.num_layers.unwrap_or(0),
            num_experts: spec.moe.num_experts.unwrap_or(0),
            num_experts_per_tok: spec.moe.num_experts_per_tok.unwrap_or(0),
            vocab_size: spec.vocab_size.unwrap_or(0),
            backend,
        }
    }
}

// ── ModelRunner trait ────────────────────────────────────────────────────

pub trait ModelRunner {
    fn model_info(&self) -> ModelInfo;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>>;
    fn reset_session(&mut self) -> Result<()>;
    fn eos_token_id(&self) -> Option<u32>;
    /// Optional expert activation report (MoE models only).
    fn expert_report(&self) -> Option<String> {
        None
    }
}

// ── Engine-plan helpers ──────────────────────────────────────────────────

pub fn unsupported_runtime_message(plan: &EnginePlan) -> String {
    let mut message = format!(
        "{} metadata is recognized, but the current executable backend cannot run it yet (engine plan: {})",
        plan.family, plan.status
    );
    if !plan.missing.is_empty() {
        message.push_str(". Missing policies:");
        for item in &plan.missing {
            message.push_str(&format!(" {}: {};", item.area, item.reason));
        }
    }
    message
}

// ── RuntimeRunner — model-family dispatch enum ───────────────────────────

/// Concrete runner that dispatches at load time by model family.
/// Used directly by `InferenceEngine<R>` (generic over `R: ModelRunner`).
pub enum RuntimeRunner {
    #[cfg(feature = "cuda")]
    OlmoeGpu(GpuOlmoeRunner),
    /// Placeholder variant when no backend is compiled.
    #[cfg(not(feature = "cuda"))]
    NoBackend,
}

impl RuntimeRunner {
    /// Load the best available runner for a model directory.
    /// For OLMoE: CUDA GPU runner (Q4_0 default).
    pub fn load(_model_dir: &Path) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        return Err(Error::Model(
            "no runtime backend available (build with --features cuda)".into(),
        ));
        #[cfg(feature = "cuda")]
        {
            let descriptor = ModelDescriptor::load(_model_dir)?;
            match descriptor.spec.family {
                ModelFamily::Olmoe => {
                    let runner = GpuOlmoeRunner::load(_model_dir, QuantType::Q4_0)?;
                    Ok(Self::OlmoeGpu(runner))
                }
                ref family => Err(Error::Model(format!(
                    "model family '{family}' is not yet supported by the runtime executor"
                ))),
            }
        }
    }

    /// Load with explicit quantization (GPU runners only).
    #[cfg(feature = "cuda")]
    pub fn load_with_quant(model_dir: &Path, quant: QuantType) -> Result<Self> {
        let descriptor = ModelDescriptor::load(model_dir)?;
        match descriptor.spec.family {
            ModelFamily::Olmoe => {
                let runner = GpuOlmoeRunner::load(model_dir, quant)?;
                Ok(Self::OlmoeGpu(runner))
            }
            ref family => Err(Error::Model(format!(
                "model family '{family}' is not yet supported by the runtime executor"
            ))),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load_with_quant(_model_dir: &Path, _quant: QuantType) -> Result<Self> {
        Err(Error::Model(
            "quantized loading requires CUDA backend (build with --features cuda)".into(),
        ))
    }
}

impl ModelRunner for RuntimeRunner {
    fn model_info(&self) -> ModelInfo {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.model_info(),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => panic!("no runtime backend compiled"),
        }
    }

    fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.encode(_text),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => Err(Error::Model("no runtime backend compiled".into())),
        }
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.decode(_tokens),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => Err(Error::Model("no runtime backend compiled".into())),
        }
    }

    fn prefill(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.prefill(_tokens),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => Err(Error::Model("no runtime backend compiled".into())),
        }
    }

    fn decode_token(&mut self, _token: u32) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.decode_token(_token),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => Err(Error::Model("no runtime backend compiled".into())),
        }
    }

    fn reset_session(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.reset_session(),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => Err(Error::Model("no runtime backend compiled".into())),
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => r.eos_token_id(),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => None,
        }
    }

    fn expert_report(&self) -> Option<String> {
        match self {
            #[cfg(feature = "cuda")]
            Self::OlmoeGpu(r) => Some(r.expert_report()),
            #[cfg(not(feature = "cuda"))]
            Self::NoBackend => None,
        }
    }
}

// ── OLMoE GPU runner ─────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct GpuOlmoeRunner {
    model: ferrule_model::OlmoeModel,
    gpu: ferrule_cuda::forward::GpuOlmoeModel,
    quant: QuantType,
}

#[cfg(feature = "cuda")]
impl GpuOlmoeRunner {
    pub fn load(model_dir: &Path, quant: QuantType) -> Result<Self> {
        let qt_suffix = ferrule_cuda::weightpack::quant_suffix(quant);
        let pack_path = ferrule_cuda::weightpack::weightpack_path(model_dir, qt_suffix);
        if pack_path.exists() {
            match Self::load_weightpack(model_dir, quant) {
                Ok(runner) => return Ok(runner),
                Err(err) => tracing::warn!(
                    "weight-pack load failed ({}); falling back to full model load",
                    err
                ),
            }
        }
        let model = ferrule_model::OlmoeModel::load(model_dir)?;
        Self::from_model(model, quant)
    }

    pub fn load_weightpack(model_dir: &Path, quant: QuantType) -> Result<Self> {
        let qt_suffix = ferrule_cuda::weightpack::quant_suffix(quant);
        let pack_path = ferrule_cuda::weightpack::weightpack_path(model_dir, qt_suffix);
        let weightpack = ferrule_cuda::weightpack::WeightPackReader::open(&pack_path)?;
        let model = ferrule_model::OlmoeModel::load_lightweight(model_dir)?;
        let gpu =
            ferrule_cuda::forward::GpuOlmoeModel::from_lightweight(&model, &weightpack, quant)?;
        Ok(Self { model, gpu, quant })
    }

    pub fn from_model(model: ferrule_model::OlmoeModel, quant: QuantType) -> Result<Self> {
        let gpu = ferrule_cuda::forward::GpuOlmoeModel::from_cpu(&model, quant)?;
        Ok(Self { model, gpu, quant })
    }

    pub fn quant(&self) -> QuantType {
        self.quant
    }

    pub fn model(&self) -> &ferrule_model::OlmoeModel {
        &self.model
    }

    pub fn expert_report(&self) -> String {
        self.gpu.expert_report()
    }
}

#[cfg(feature = "cuda")]
impl ModelRunner for GpuOlmoeRunner {
    fn model_info(&self) -> ModelInfo {
        let c = &self.model.config;
        let spec = self.model.transformer_spec();
        ModelInfo {
            family: spec.family,
            architecture: spec.architecture,
            attention: spec.attention,
            weight_source: spec.weight_source,
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_experts: c.num_experts,
            num_experts_per_tok: c.num_experts_per_tok,
            vocab_size: c.vocab_size,
            backend: "gpu",
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.model.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.model.decode(tokens)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.decode_token(token)?;
        }
        Ok(logits)
    }

    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
        self.gpu.forward(token)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.gpu.reset_session();
        Ok(())
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.model.eos_token_id()
    }

    fn expert_report(&self) -> Option<String> {
        Some(self.gpu.expert_report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_model::{
        AttentionKind, ModelSupportContract, MoeSpec, QuantFormatCount, RouterKind,
        TransformerSpec, WeightSource,
    };

    #[test]
    fn unsupported_runtime_message_reports_engine_plan_gaps() {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("deepseek4".into()),
            weight_source: WeightSource::Gguf,
            hidden_size: Some(7168),
            num_layers: Some(1),
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
            tensor_count: Some(1),
            quantization: vec![QuantFormatCount {
                format: "Q4_K".into(),
                tensors: 1,
            }],
            notes: Vec::new(),
        };
        let plan = ModelSupportContract::from_spec(&spec, &[]).engine_plan();
        let message = unsupported_runtime_message(&plan);
        assert!(message.contains("DeepSeek-V4"));
        assert!(message.contains("engine plan: metadata-only"));
        assert!(message.contains("attention:"));
        assert!(message.contains("quantization:"));
        assert!(message.contains("router:"));
    }
}
