use crate::sampling::{build_logits_processor, sample_next_token};
use crate::scoring::token_logprob_from_logits;
use crate::tokenizer::FerruleTokenizer;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use ferrule_core::{FerruleError, FerruleResult, SamplingParams, TokenUsage};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaLoadConfig {
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub use_flash_attn: bool,
    #[serde(default = "default_use_kv_cache")]
    pub use_kv_cache: bool,
}

impl Default for LlamaLoadConfig {
    fn default() -> Self {
        Self {
            dtype: default_dtype(),
            use_flash_attn: false,
            use_kv_cache: default_use_kv_cache(),
        }
    }
}

fn default_dtype() -> String {
    "f16".to_string()
}

fn default_use_kv_cache() -> bool {
    true
}

#[derive(Clone)]
pub struct LlamaBackend {
    model: Arc<Llama>,
    config: candle_transformers::models::llama::Config,
    device: Arc<Device>,
    dtype: DType,
    use_kv_cache: bool,
    pub eos_token_id: Option<u32>,
    pub weight_files: Vec<PathBuf>,
    pub config_path: PathBuf,
}

pub struct LlamaSession {
    pub cache: Cache,
    pub index_pos: usize,
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: String,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone)]
pub struct ChunkOutput {
    pub chunk_text: String,
    pub finish_reason: Option<String>,
    pub generated_text: String,
}

impl LlamaBackend {
    pub fn load_local(
        device: Arc<Device>,
        config_path: &Path,
        weight_files: &[PathBuf],
        load_cfg: &LlamaLoadConfig,
        eos_token_id: Option<u32>,
    ) -> FerruleResult<Self> {
        let raw = fs::read(config_path)?;
        let raw_cfg: LlamaConfig = serde_json::from_slice(&raw)
            .map_err(|e| FerruleError::Model(format!("failed to parse llama config.json: {e}")))?;

        let config = raw_cfg.into_config(load_cfg.use_flash_attn);
        let dtype = parse_dtype(&load_cfg.dtype)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_files, dtype, &device)
                .map_err(|e| FerruleError::Model(format!("failed to mmap safetensors: {e}")))?
        };

        let model = Llama::load(vb, &config)
            .map_err(|e| FerruleError::Model(format!("failed to build llama model: {e}")))?;

        info!(
            weight_files = weight_files.len(),
            dtype = ?dtype,
            use_flash_attn = load_cfg.use_flash_attn,
            use_kv_cache = load_cfg.use_kv_cache,
            "loaded local llama backend"
        );

        Ok(Self {
            model: Arc::new(model),
            config,
            device,
            dtype,
            use_kv_cache: load_cfg.use_kv_cache,
            eos_token_id,
            weight_files: weight_files.to_vec(),
            config_path: config_path.to_path_buf(),
        })
    }

    pub fn new_session(&self) -> FerruleResult<LlamaSession> {
        let cache = Cache::new(self.use_kv_cache, self.dtype, &self.config, &self.device)
            .map_err(|e| FerruleError::Model(format!("failed to create llama cache: {e}")))?;

        Ok(LlamaSession {
            cache,
            index_pos: 0,
        })
    }

    pub fn forward_prefill(
        &self,
        session: &mut LlamaSession,
        input_ids: &[u32],
    ) -> FerruleResult<Tensor> {
        if input_ids.is_empty() {
            return Err(FerruleError::Runtime(
                "forward_prefill received empty input_ids".to_string(),
            ));
        }

        let input = Tensor::new(input_ids, &*self.device)
            .map_err(|e| FerruleError::Model(format!("failed to build input tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerruleError::Model(format!("failed to unsqueeze input tensor: {e}")))?;

        let logits = self
            .model
            .forward(&input, session.index_pos, &mut session.cache)
            .map_err(|e| FerruleError::Model(format!("llama prefill forward failed: {e}")))?;

        session.index_pos += input_ids.len();
        Ok(logits)
    }

    pub fn forward_decode_one(
        &self,
        session: &mut LlamaSession,
        next_input_id: u32,
    ) -> FerruleResult<Tensor> {
        let input = Tensor::new(&[next_input_id], &*self.device)
            .map_err(|e| FerruleError::Model(format!("failed to build decode tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerruleError::Model(format!("failed to unsqueeze decode tensor: {e}")))?;

        let logits = self
            .model
            .forward(&input, session.index_pos, &mut session.cache)
            .map_err(|e| FerruleError::Model(format!("llama decode forward failed: {e}")))?;

        session.index_pos += 1;
        Ok(logits)
    }

    pub fn sample_next_from_prefill(
        &self,
        session: &mut LlamaSession,
        prompt_ids: &[u32],
        processor: &mut LogitsProcessor,
        params: &SamplingParams,
    ) -> FerruleResult<u32> {
        let logits = self.forward_prefill(session, prompt_ids)?;
        sample_next_token(&logits, processor, prompt_ids, params)
    }

    pub fn sample_next_from_decode(
        &self,
        session: &mut LlamaSession,
        next_input_id: u32,
        prior_tokens: &[u32],
        processor: &mut LogitsProcessor,
        params: &SamplingParams,
    ) -> FerruleResult<u32> {
        let logits = self.forward_decode_one(session, next_input_id)?;
        sample_next_token(&logits, processor, prior_tokens, params)
    }

    pub fn generate_next_chunk(
        &self,
        tokenizer: &FerruleTokenizer,
        session: &mut LlamaSession,
        processor: &mut LogitsProcessor,
        prompt_ids: &[u32],
        generated_ids: &mut Vec<u32>,
        generated_text: &mut String,
        params: &SamplingParams,
    ) -> FerruleResult<ChunkOutput> {
        let mut chunk = String::new();
        let mut finish_reason = None;

        for _ in 0..8 {
            let next_token = if generated_ids.is_empty() {
                self.sample_next_from_prefill(session, prompt_ids, processor, params)?
            } else {
                let mut prior = Vec::with_capacity(prompt_ids.len() + generated_ids.len());
                prior.extend_from_slice(prompt_ids);
                prior.extend_from_slice(generated_ids);

                self.sample_next_from_decode(
                    session,
                    *generated_ids.last().ok_or_else(|| {
                        FerruleError::Runtime("missing last generated token".to_string())
                    })?,
                    &prior,
                    processor,
                    params,
                )?
            };

            generated_ids.push(next_token);

            let new_text = tokenizer.decode(generated_ids, true)?;
            let delta = compute_delta(generated_text, &new_text);

            if !delta.is_empty() {
                chunk.push_str(&delta);
                *generated_text = new_text;
            }

            if let Some(eos) = self.eos_token_id {
                if next_token == eos {
                    finish_reason = Some("eos".to_string());
                }
            }

            if finish_reason.is_none() && generated_ids.len() >= params.max_new_tokens {
                finish_reason = Some("max_new_tokens".to_string());
            }

            if finish_reason.is_none() && !params.stop_strings.is_empty() {
                if params
                    .stop_strings
                    .iter()
                    .any(|s| generated_text.contains(s))
                {
                    finish_reason = Some("stop_string".to_string());
                }
            }

            if !chunk.is_empty() || finish_reason.is_some() {
                return Ok(ChunkOutput {
                    chunk_text: chunk,
                    finish_reason,
                    generated_text: generated_text.clone(),
                });
            }
        }

        Err(FerruleError::Runtime(
            "llama generation produced no visible chunk within bounded inner loop".to_string(),
        ))
    }

    pub fn generate_text_once(
        &self,
        tokenizer: &FerruleTokenizer,
        prompt: &str,
        params: &SamplingParams,
    ) -> FerruleResult<GenerateOutput> {
        let prompt_ids = tokenizer.encode(prompt, true)?;
        if prompt_ids.is_empty() {
            return Err(FerruleError::Runtime(
                "generate_text_once got empty prompt ids".to_string(),
            ));
        }

        let mut session = self.new_session()?;
        let mut processor = build_logits_processor(params);
        let mut generated_ids = Vec::new();
        let mut generated_text = String::new();
        let mut finish_reason = "max_new_tokens".to_string();

        while generated_ids.len() < params.max_new_tokens {
            let out = self.generate_next_chunk(
                tokenizer,
                &mut session,
                &mut processor,
                &prompt_ids,
                &mut generated_ids,
                &mut generated_text,
                params,
            )?;

            if let Some(reason) = out.finish_reason {
                finish_reason = reason;
                break;
            }
        }

        Ok(GenerateOutput {
            token_ids: generated_ids.clone(),
            text: generated_text.clone(),
            finish_reason,
            usage: TokenUsage {
                prompt_tokens: prompt_ids.len(),
                completion_tokens: generated_ids.len(),
            },
        })
    }

    pub fn score_completion(
        &self,
        tokenizer: &FerruleTokenizer,
        prompt: &str,
        completion: &str,
    ) -> FerruleResult<Vec<f32>> {
        let prompt_ids = tokenizer.encode(prompt, true)?;
        let completion_ids = tokenizer.encode(completion, false)?;
        self.score_completion_ids(&prompt_ids, &completion_ids)
    }

    pub fn score_completion_ids(
        &self,
        prompt_ids: &[u32],
        completion_ids: &[u32],
    ) -> FerruleResult<Vec<f32>> {
        if prompt_ids.is_empty() {
            return Err(FerruleError::Runtime(
                "score_completion_ids got empty prompt ids".to_string(),
            ));
        }

        if completion_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut session = self.new_session()?;
        let mut out = Vec::with_capacity(completion_ids.len());

        // First completion token is predicted from prompt prefill.
        let logits = self.forward_prefill(&mut session, prompt_ids)?;
        out.push(token_logprob_from_logits(&logits, completion_ids[0])?);

        // Then teacher-force the rest of the completion tokens one by one.
        for idx in 1..completion_ids.len() {
            let prev = completion_ids[idx - 1];
            let logits = self.forward_decode_one(&mut session, prev)?;
            out.push(token_logprob_from_logits(&logits, completion_ids[idx])?);
        }

        Ok(out)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn parse_dtype(spec: &str) -> FerruleResult<DType> {
    match spec {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        other => Err(FerruleError::Config(format!(
            "unsupported dtype for llama backend: {other}"
        ))),
    }
}

fn compute_delta(old: &str, new: &str) -> String {
    if new.starts_with(old) {
        new[old.len()..].to_string()
    } else {
        new.to_string()
    }
}
