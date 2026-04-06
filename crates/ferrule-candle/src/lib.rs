mod device;
mod family;
mod prompt;
mod sampling;
mod source;
mod tokenizer;

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use family::llama::{LlamaBackend, LlamaLoadConfig, LlamaSession};
use ferrule_core::{
    FerruleError, FerruleResult, ModelConfig, ModelOutput, ModelStep, PolicyModel, SamplingParams,
    TokenUsage, async_trait,
};
use sampling::build_logits_processor;
use source::{ResolvedModelPaths, resolve_model_paths};
use std::sync::Arc;
use tokenizer::FerruleTokenizer;
use tracing::info;

#[derive(Clone)]
enum RealBackend {
    Llama(Arc<LlamaBackend>),
}

#[derive(Debug, Clone)]
struct MockSession {
    step_idx: usize,
    _params: SamplingParams,
}

pub enum CandleSession {
    Mock(MockSession),
    Llama(LlamaPolicySession),
}

pub struct LlamaPolicySession {
    backend: Arc<LlamaBackend>,
    inner: LlamaSession,
    processor: LogitsProcessor,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated_text: String,
    params: SamplingParams,
    finished: bool,
    finish_reason: Option<String>,
}

#[derive(Clone)]
pub struct CandlePolicy {
    backend: String,
    family: String,
    model_id: String,
    device_spec: String,
    device: Arc<Device>,
    model_paths: ResolvedModelPaths,
    tokenizer: Arc<FerruleTokenizer>,
    chat_template: String,
    dtype: String,
    use_flash_attn: bool,
    use_kv_cache: bool,
    real_backend: Option<RealBackend>,
}

#[derive(Debug, Clone)]
pub struct DoctorInfo {
    pub backend: String,
    pub family: String,
    pub model_id: String,
    pub device_spec: String,
    pub resolved_device: String,
    pub compiled_backends: String,
    pub root_dir: String,
    pub tokenizer_json: String,
    pub config_json: Option<String>,
    pub weight_files: usize,
    pub vocab_size_hint: usize,
    pub chat_template: String,
    pub dtype: String,
    pub use_flash_attn: bool,
    pub use_kv_cache: bool,
    pub real_backend_loaded: bool,
}

impl CandlePolicy {
    pub fn from_config(cfg: &ModelConfig) -> FerruleResult<Self> {
        let model_paths = resolve_model_paths(cfg)?;
        let tokenizer = FerruleTokenizer::from_file(&model_paths.tokenizer_json)?;
        let device = device::select_device(&cfg.device)?;

        let mut policy = Self {
            backend: cfg.backend.clone(),
            family: cfg.family.clone(),
            model_id: cfg.model_id.clone(),
            device_spec: cfg.device.clone(),
            device: Arc::new(device),
            model_paths,
            tokenizer: Arc::new(tokenizer),
            chat_template: cfg.chat_template.clone(),
            dtype: cfg.dtype.clone(),
            use_flash_attn: cfg.use_flash_attn,
            use_kv_cache: cfg.use_kv_cache,
            real_backend: None,
        };

        if cfg.backend == "real" {
            policy.real_backend = Some(policy.load_real_backend()?);
        }

        Ok(policy)
    }

    fn load_real_backend(&self) -> FerruleResult<RealBackend> {
        match self.family.as_str() {
            "llama" => Ok(RealBackend::Llama(Arc::new(self.build_llama_backend()?))),
            other => Err(FerruleError::Config(format!(
                "real backend not implemented for family: {other}"
            ))),
        }
    }

    fn build_llama_backend(&self) -> FerruleResult<LlamaBackend> {
        let config_path = self.model_paths.config_json.as_ref().ok_or_else(|| {
            FerruleError::Config("config.json is required for llama backend".to_string())
        })?;

        let eos_token_id = self
            .tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|eot_id|>"));

        LlamaBackend::load_local(
            self.device.clone(),
            config_path,
            &self.model_paths.weight_files,
            &LlamaLoadConfig {
                dtype: self.dtype.clone(),
                use_flash_attn: self.use_flash_attn,
                use_kv_cache: self.use_kv_cache,
            },
            eos_token_id,
        )
    }

    pub fn candle_sanity_check(&self) -> FerruleResult<()> {
        let _tensor = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &self.device)
            .map_err(|e| FerruleError::Model(format!("candle sanity check failed: {e}")))?;
        Ok(())
    }

    pub fn doctor_info(&self) -> DoctorInfo {
        DoctorInfo {
            backend: self.backend.clone(),
            family: self.family.clone(),
            model_id: self.model_id.clone(),
            device_spec: self.device_spec.clone(),
            resolved_device: device::device_kind_str(&self.device).to_string(),
            compiled_backends: device::compiled_backends_summary(),
            root_dir: self.model_paths.root_dir.display().to_string(),
            tokenizer_json: self.model_paths.tokenizer_json.display().to_string(),
            config_json: self
                .model_paths
                .config_json
                .as_ref()
                .map(|p| p.display().to_string()),
            weight_files: self.model_paths.weight_files.len(),
            vocab_size_hint: self.tokenizer.vocab_size_hint(),
            chat_template: self.chat_template.clone(),
            dtype: self.dtype.clone(),
            use_flash_attn: self.use_flash_attn,
            use_kv_cache: self.use_kv_cache,
            real_backend_loaded: self.real_backend.is_some(),
        }
    }

    pub fn encode_text(&self, text: &str, add_special_tokens: bool) -> FerruleResult<Vec<u32>> {
        self.tokenizer.encode(text, add_special_tokens)
    }

    pub fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> FerruleResult<String> {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    pub fn llama_backend(&self) -> FerruleResult<Arc<LlamaBackend>> {
        match &self.real_backend {
            Some(RealBackend::Llama(b)) => Ok(b.clone()),
            None => Err(FerruleError::Model(
                "llama backend is not loaded; set backend='real'".to_string(),
            )),
        }
    }

    pub fn llama_prefill_logits(&self, input_ids: &[u32]) -> FerruleResult<Tensor> {
        let backend = self.llama_backend()?;
        let mut session = backend.new_session()?;
        backend.forward_prefill(&mut session, input_ids)
    }

    pub fn generate_text_once(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> FerruleResult<family::llama::GenerateOutput> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                backend.generate_text_once(&self.tokenizer, prompt, params)
            }
            other => Err(FerruleError::Config(format!(
                "generate_text_once not implemented for family: {other}"
            ))),
        }
    }

    fn step_mock_session(&self, session: &mut MockSession) -> FerruleResult<ModelStep> {
        let action = match session.step_idx {
            0 => ModelOutput::Text {
                content: "Thought: I should call a tool.".to_string(),
            },
            1 => ModelOutput::CallTool {
                name: "echo".to_string(),
                arguments_json: r#"{"query":"2+2"}"#.to_string(),
            },
            2 => ModelOutput::Text {
                content: "Final answer: 4".to_string(),
            },
            _ => ModelOutput::Finish {
                reason: "mock_done".to_string(),
                final_text: Some("Thought: I should call a tool. Final answer: 4".to_string()),
            },
        };

        session.step_idx += 1;

        Ok(ModelStep {
            action,
            usage: TokenUsage {
                prompt_tokens: 8,
                completion_tokens: 8,
            },
        })
    }

    fn step_llama_session(&self, session: &mut LlamaPolicySession) -> FerruleResult<ModelStep> {
        if session.finished {
            return Ok(ModelStep {
                action: ModelOutput::Finish {
                    reason: session
                        .finish_reason
                        .clone()
                        .unwrap_or_else(|| "completed".to_string()),
                    final_text: Some(session.generated_text.clone()),
                },
                usage: TokenUsage {
                    prompt_tokens: session.prompt_ids.len(),
                    completion_tokens: session.generated_ids.len(),
                },
            });
        }

        let out = session.backend.generate_next_chunk(
            &self.tokenizer,
            &mut session.inner,
            &mut session.processor,
            &session.prompt_ids,
            &mut session.generated_ids,
            &mut session.generated_text,
            &session.params,
        )?;

        if let Some(reason) = out.finish_reason {
            session.finished = true;
            session.finish_reason = Some(reason);
        }

        if !out.chunk_text.is_empty() {
            return Ok(ModelStep {
                action: ModelOutput::Text {
                    content: out.chunk_text,
                },
                usage: TokenUsage {
                    prompt_tokens: session.prompt_ids.len(),
                    completion_tokens: session.generated_ids.len(),
                },
            });
        }

        Ok(ModelStep {
            action: ModelOutput::Finish {
                reason: session
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "completed".to_string()),
                final_text: Some(session.generated_text.clone()),
            },
            usage: TokenUsage {
                prompt_tokens: session.prompt_ids.len(),
                completion_tokens: session.generated_ids.len(),
            },
        })
    }
}

#[async_trait]
impl PolicyModel for CandlePolicy {
    type Session = CandleSession;

    fn name(&self) -> &str {
        &self.model_id
    }

    async fn new_session(
        &self,
        prompt_token_ids: &[u32],
        params: &SamplingParams,
    ) -> FerruleResult<Self::Session> {
        info!(
            backend = %self.backend,
            family = %self.family,
            model_id = %self.model_id,
            device = %self.device_spec,
            prompt_tokens = prompt_token_ids.len(),
            "creating policy session"
        );

        match self.backend.as_str() {
            "mock" => Ok(CandleSession::Mock(MockSession {
                step_idx: 0,
                _params: params.clone(),
            })),
            "real" => match self.family.as_str() {
                "llama" => {
                    let backend = self.llama_backend()?;
                    let inner = backend.new_session()?;
                    let processor = build_logits_processor(params);

                    Ok(CandleSession::Llama(LlamaPolicySession {
                        backend,
                        inner,
                        processor,
                        prompt_ids: prompt_token_ids.to_vec(),
                        generated_ids: Vec::new(),
                        generated_text: String::new(),
                        params: params.clone(),
                        finished: false,
                        finish_reason: None,
                    }))
                }
                other => Err(FerruleError::Config(format!(
                    "real session creation not implemented for family: {other}"
                ))),
            },
            other => Err(FerruleError::Config(format!(
                "unsupported backend: {other}"
            ))),
        }
    }

    async fn step(&self, session: &mut Self::Session) -> FerruleResult<ModelStep> {
        match session {
            CandleSession::Mock(s) => self.step_mock_session(s),
            CandleSession::Llama(s) => self.step_llama_session(s),
        }
    }
}
