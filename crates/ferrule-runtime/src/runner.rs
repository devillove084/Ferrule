use ferrule_core::Result;
use ferrule_model::OlmoeModel;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelInfo {
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
        Self {
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_experts: c.num_experts,
            num_experts_per_tok: c.num_experts_per_tok,
            vocab_size: c.vocab_size,
            backend,
        }
    }
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
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pos: usize,
}

impl CpuOlmoeRunner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Ok(Self::new(OlmoeModel::load(model_dir)?))
    }

    pub fn new(model: OlmoeModel) -> Self {
        let num_layers = model.config.num_layers;
        Self {
            model,
            k_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            v_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            pos: 0,
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
        let (_, logits) =
            self.model
                .forward(&[token], &mut self.k_cache, &mut self.v_cache, self.pos)?;
        self.pos += 1;
        Ok(logits)
    }

    fn reset_session(&mut self) -> Result<()> {
        let num_layers = self.model.config.num_layers;
        self.k_cache = (0..num_layers).map(|_| Vec::new()).collect();
        self.v_cache = (0..num_layers).map(|_| Vec::new()).collect();
        self.pos = 0;
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

#[cfg(feature = "cuda")]
impl GpuOlmoeRunner {
    pub fn load(model_dir: &Path, quant: ferrule_quant::QuantType) -> Result<Self> {
        let qt_suffix = ferrule_cuda::qcache::quant_suffix(quant);
        let cache_path = ferrule_cuda::qcache::cache_path(model_dir, qt_suffix);
        if cache_path.exists() {
            match Self::load_qcache(model_dir, quant) {
                Ok(runner) => return Ok(runner),
                Err(err) => tracing::warn!(
                    "qcache load failed ({}); falling back to full model load",
                    err
                ),
            }
        }
        let model = OlmoeModel::load(model_dir)?;
        Self::from_model(model, quant)
    }

    /// Load using qcache-only path (no full FP32 model load).
    pub fn load_qcache(model_dir: &Path, quant: ferrule_quant::QuantType) -> Result<Self> {
        let qt_suffix = ferrule_cuda::qcache::quant_suffix(quant);
        let cache_path = ferrule_cuda::qcache::cache_path(model_dir, qt_suffix);
        let cache = ferrule_cuda::qcache::QCacheReader::open(&cache_path)?;
        let model = OlmoeModel::load_lightweight(model_dir)?;
        let gpu = ferrule_cuda::forward::GpuOlmoeModel::from_lightweight(&model, &cache, quant)?;
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
