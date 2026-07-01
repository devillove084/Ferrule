use ferrule_core::{Error, Result};
use ferrule_model::{
    AttentionKind, EnginePlan, ModelDescriptor, ModelFamily, OlmoeModel, WeightSource,
};

use crate::cpu_kv::CpuContiguousKvState;
use std::path::Path;

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
    fn from_olmoe(model: &OlmoeModel, backend: &'static str) -> Self {
        let c = &model.config;
        let spec = model.transformer_spec();
        Self {
            family: spec.family,
            architecture: spec.architecture,
            attention: spec.attention,
            weight_source: spec.weight_source,
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_experts: c.num_experts,
            num_experts_per_tok: c.num_experts_per_tok,
            vocab_size: c.vocab_size,
            backend,
        }
    }
}

fn ensure_current_runtime_family(model_dir: &Path) -> Result<()> {
    let descriptor = ModelDescriptor::load(model_dir)?;
    let plan = descriptor.engine_plan();
    if plan.is_executable() {
        return Ok(());
    }
    Err(Error::Model(unsupported_runtime_message(&plan)))
}

fn unsupported_runtime_message(plan: &EnginePlan) -> String {
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

pub struct CpuOlmoeRunner {
    model: OlmoeModel,
    kv: CpuContiguousKvState,
}

/// Current CPU runtime runner. Today it executes OLMoE, but callers should use
/// this generic alias so model-family dispatch can grow without changing CLI code.
pub type CpuModelRunner = CpuOlmoeRunner;

impl CpuOlmoeRunner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        ensure_current_runtime_family(model_dir)?;
        Ok(Self::new(OlmoeModel::load(model_dir)?))
    }

    pub fn new(model: OlmoeModel) -> Self {
        let num_layers = model.config.num_layers;
        Self {
            model,
            kv: CpuContiguousKvState::new(num_layers),
        }
    }

    pub fn model(&self) -> &OlmoeModel {
        &self.model
    }
}

impl ModelRunner for CpuOlmoeRunner {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::from_olmoe(&self.model, "cpu-fp32")
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
        let model = &self.model;
        let (_, logits) = self
            .kv
            .decode_one(|k_cache, v_cache, pos| model.forward(&[token], k_cache, v_cache, pos))?;
        Ok(logits)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.kv.reset(self.model.config.num_layers);
        Ok(())
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.model.eos_token_id()
    }
}

#[cfg(feature = "cuda")]
pub struct GpuOlmoeRunner {
    model: OlmoeModel,
    gpu: ferrule_cuda::forward::GpuOlmoeModel,
    quant: ferrule_quant::QuantType,
}

/// Current CUDA runtime runner. Today it executes OLMoE, but callers should use
/// this generic alias so model-family dispatch can grow without changing CLI code.
#[cfg(feature = "cuda")]
pub type GpuModelRunner = GpuOlmoeRunner;

#[cfg(feature = "cuda")]
impl GpuOlmoeRunner {
    pub fn load(model_dir: &Path, quant: ferrule_quant::QuantType) -> Result<Self> {
        ensure_current_runtime_family(model_dir)?;
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
        let model = OlmoeModel::load(model_dir)?;
        Self::from_model(model, quant)
    }

    /// Load using weight-pack-only path (no full FP32 model load).
    pub fn load_weightpack(model_dir: &Path, quant: ferrule_quant::QuantType) -> Result<Self> {
        ensure_current_runtime_family(model_dir)?;
        let qt_suffix = ferrule_cuda::weightpack::quant_suffix(quant);
        let pack_path = ferrule_cuda::weightpack::weightpack_path(model_dir, qt_suffix);
        let weightpack = ferrule_cuda::weightpack::WeightPackReader::open(&pack_path)?;
        let model = OlmoeModel::load_lightweight(model_dir)?;
        let gpu =
            ferrule_cuda::forward::GpuOlmoeModel::from_lightweight(&model, &weightpack, quant)?;
        Ok(Self { model, gpu, quant })
    }

    pub fn from_model(model: OlmoeModel, quant: ferrule_quant::QuantType) -> Result<Self> {
        let gpu = ferrule_cuda::forward::GpuOlmoeModel::from_cpu(&model, quant)?;
        Ok(Self { model, gpu, quant })
    }

    pub fn quant(&self) -> ferrule_quant::QuantType {
        self.quant
    }

    pub fn model(&self) -> &OlmoeModel {
        &self.model
    }

    pub fn expert_report(&self) -> String {
        self.gpu.expert_report()
    }
}

#[cfg(feature = "cuda")]
impl ModelRunner for GpuOlmoeRunner {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::from_olmoe(&self.model, "gpu")
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
