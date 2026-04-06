mod device;
mod diff_scoring;
mod family;
mod policy_bias;
mod prompt;
mod sampling;
mod scoring;
mod source;
mod tokenizer;

use candle_core::{Device, Tensor, Var};
use candle_transformers::generation::LogitsProcessor;
use family::llama::{LlamaBackend, LlamaLoadConfig, LlamaSession};
use ferrule_core::{
    FerruleError, FerruleResult, ModelConfig, ModelOutput, ModelStep, PolicyModel, SamplingParams,
    TokenUsage, async_trait,
};
use policy_bias::PolicyBiasHead;
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
    policy_bias: Option<Arc<PolicyBiasHead>>,
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
            policy_bias: None,
        };

        if cfg.backend == "real" {
            policy.real_backend = Some(policy.load_real_backend()?);
        }

        let vocab_size = policy.tokenizer.vocab_size_hint();
        let bias = PolicyBiasHead::new(vocab_size, policy.bias_init_device())?;
        policy.policy_bias = Some(Arc::new(bias));

        Ok(policy)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn bias_init_device(&self) -> &Device {
        match &self.real_backend {
            Some(RealBackend::Llama(b)) => b.device(),
            None => &self.device,
        }
    }

    pub fn trainable_vars(&self) -> FerruleResult<Vec<Var>> {
        match &self.policy_bias {
            Some(head) => Ok(head.trainable_vars()),
            None => Err(FerruleError::Model(
                "policy bias head is not initialized".to_string(),
            )),
        }
    }

    pub fn apply_policy_bias(&self, logits: &Tensor) -> FerruleResult<Tensor> {
        match &self.policy_bias {
            Some(head) => head.apply_to_logits(logits),
            None => Ok(logits.clone()),
        }
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

    pub fn score_completion(&self, prompt: &str, completion: &str) -> FerruleResult<Vec<f32>> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                backend.score_completion(&self.tokenizer, prompt, completion)
            }
            other => Err(FerruleError::Config(format!(
                "score_completion not implemented for family: {other}"
            ))),
        }
    }

    pub fn score_completion_ids(
        &self,
        prompt_ids: &[u32],
        completion_ids: &[u32],
    ) -> FerruleResult<Vec<f32>> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                backend.score_completion_ids(prompt_ids, completion_ids)
            }
            other => Err(FerruleError::Config(format!(
                "score_completion_ids not implemented for family: {other}"
            ))),
        }
    }

    pub fn score_completion_ids_with_bias(
        &self,
        prompt_ids: &[u32],
        completion_ids: &[u32],
    ) -> FerruleResult<Vec<f32>> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                if prompt_ids.is_empty() {
                    return Err(FerruleError::Runtime(
                        "score_completion_ids_with_bias got empty prompt ids".to_string(),
                    ));
                }
                if completion_ids.is_empty() {
                    return Ok(vec![]);
                }

                let mut session = backend.new_session()?;
                let mut out = Vec::with_capacity(completion_ids.len());

                let logits = backend.forward_prefill(&mut session, prompt_ids)?;
                let logits = self.apply_policy_bias(&logits)?;
                out.push(crate::scoring::token_logprob_from_logits(
                    &logits,
                    completion_ids[0],
                )?);

                for idx in 1..completion_ids.len() {
                    let prev = completion_ids[idx - 1];
                    let logits = backend.forward_decode_one(&mut session, prev)?;
                    let logits = self.apply_policy_bias(&logits)?;
                    out.push(crate::scoring::token_logprob_from_logits(
                        &logits,
                        completion_ids[idx],
                    )?);
                }

                Ok(out)
            }
            other => Err(FerruleError::Config(format!(
                "score_completion_ids_with_bias not implemented for family: {other}"
            ))),
        }
    }

    pub fn generate_text_once_with_bias(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> FerruleResult<family::llama::GenerateOutput> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                let prompt_ids = self.tokenizer.encode(prompt, true)?;
                if prompt_ids.is_empty() {
                    return Err(FerruleError::Runtime(
                        "generate_text_once_with_bias got empty prompt ids".to_string(),
                    ));
                }

                let mut session = backend.new_session()?;
                let mut processor = crate::sampling::build_logits_processor(params);

                let mut generated_ids = Vec::new();
                let mut generated_text = String::new();
                let mut finish_reason = "max_new_tokens".to_string();

                while generated_ids.len() < params.max_new_tokens {
                    let next_token = if generated_ids.is_empty() {
                        let logits = backend.forward_prefill(&mut session, &prompt_ids)?;
                        let logits = self.apply_policy_bias(&logits)?;
                        crate::sampling::sample_next_token(
                            &logits,
                            &mut processor,
                            &prompt_ids,
                            params,
                        )?
                    } else {
                        let mut prior = Vec::with_capacity(prompt_ids.len() + generated_ids.len());
                        prior.extend_from_slice(&prompt_ids);
                        prior.extend_from_slice(&generated_ids);

                        let logits = backend
                            .forward_decode_one(&mut session, *generated_ids.last().unwrap())?;
                        let logits = self.apply_policy_bias(&logits)?;
                        crate::sampling::sample_next_token(&logits, &mut processor, &prior, params)?
                    };

                    generated_ids.push(next_token);

                    let new_text = self.tokenizer.decode(&generated_ids, true)?;
                    generated_text = new_text;

                    if let Some(eos) = backend.eos_token_id {
                        if next_token == eos {
                            finish_reason = "eos".to_string();
                            break;
                        }
                    }

                    if !params.stop_strings.is_empty()
                        && params
                            .stop_strings
                            .iter()
                            .any(|s| generated_text.contains(s))
                    {
                        finish_reason = "stop_string".to_string();
                        break;
                    }
                }

                Ok(family::llama::GenerateOutput {
                    token_ids: generated_ids.clone(),
                    text: generated_text,
                    finish_reason,
                    usage: ferrule_core::TokenUsage {
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: generated_ids.len(),
                    },
                })
            }
            other => Err(FerruleError::Config(format!(
                "generate_text_once_with_bias not implemented for family: {other}"
            ))),
        }
    }

    pub fn compute_agent_loss_tensor(
        &self,
        steps: &[(&[u32], &[u32], f32)], // (prompt_ids, action_ids, advantage)
    ) -> FerruleResult<Tensor> {
        if steps.is_empty() {
            return Err(FerruleError::Runtime(
                "compute_agent_loss_tensor got empty steps".to_string(),
            ));
        }

        let mut losses = Vec::with_capacity(steps.len());

        for (prompt_ids, action_ids, advantage) in steps {
            let lps = self.score_completion_ids_with_bias(prompt_ids, action_ids)?;
            let logprob_sum = lps.iter().copied().sum::<f32>();
            let loss_scalar = -(*advantage) * logprob_sum;
            let t = Tensor::new(&[loss_scalar], self.policy_bias.as_ref().unwrap().device())
                .map_err(|e| {
                    FerruleError::Model(format!("failed to build scalar loss tensor: {e}"))
                })?;
            losses.push(t);
        }

        let stacked = Tensor::cat(&losses, 0)
            .map_err(|e| FerruleError::Model(format!("failed to concat loss tensors: {e}")))?;

        stacked
            .mean(0)
            .map_err(|e| FerruleError::Model(format!("failed to reduce mean loss: {e}")))
    }

    pub fn differentiable_action_logprob_sum(
        &self,
        prompt_ids: &[u32],
        action_ids: &[u32],
    ) -> FerruleResult<Tensor> {
        match self.family.as_str() {
            "llama" => {
                let backend = self.llama_backend()?;
                if prompt_ids.is_empty() {
                    return Err(FerruleError::Runtime(
                        "differentiable_action_logprob_sum got empty prompt ids".to_string(),
                    ));
                }
                if action_ids.is_empty() {
                    return Err(FerruleError::Runtime(
                        "differentiable_action_logprob_sum got empty action ids".to_string(),
                    ));
                }

                let mut session = backend.new_session()?;
                let mut logits_per_step = Vec::with_capacity(action_ids.len());

                let logits = backend.forward_prefill(&mut session, prompt_ids)?;
                let logits = self.apply_policy_bias(&logits)?;
                logits_per_step.push(logits);

                for idx in 1..action_ids.len() {
                    let prev = action_ids[idx - 1];
                    let logits = backend.forward_decode_one(&mut session, prev)?;
                    let logits = self.apply_policy_bias(&logits)?;
                    logits_per_step.push(logits);
                }

                crate::diff_scoring::sequence_logprob_sum_tensor(&logits_per_step, action_ids)
            }
            other => Err(FerruleError::Config(format!(
                "differentiable_action_logprob_sum not implemented for family: {other}"
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
