use crate::sampling::sampler::{Logprobs, Sampler, SamplingConfig};
use crate::stats::GenerateStats;
use ferrule_common::{Error, Result};
use ferrule_model::runner::{ModelRunner, TokenLogit, TopKModelRunner};
use std::time::{Duration, Instant};

pub use crate::scheduling::SequenceFinishReason as TopKFinishReason;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
    /// Stop generation when the backend emits its EOS token.
    pub stop_at_eos: bool,
    /// Feed EOS into the backend session when generation stops on EOS.
    pub append_eos_to_session: bool,
    /// If > 0, collect top-K logprobs for each generated token.
    pub logprobs_k: usize,
    /// Context window size (max tokens in KV cache).
    pub ctx_size: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            stop: Vec::new(),
            stop_at_eos: true,
            append_eos_to_session: true,
            logprobs_k: 0,
            ctx_size: 4096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub index: usize,
    pub token: u32,
    pub text: String,
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub stats: GenerateStats,
    pub stopped_by_eos: bool,
    pub stopped_by_string: Option<String>,
    pub all_logprobs: Vec<Logprobs>,
}

#[derive(Debug, Clone)]
pub struct TopKTokenEvent {
    pub index: usize,
    pub token: u32,
    pub logit: f32,
    pub text: String,
}

/// `TopKTurnResult` keeps legacy boolean helpers for compatibility, but
/// schedulers and serving code should prefer `finish_reason`.
#[derive(Debug, Clone)]
pub struct TopKTurnResult {
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub stopped_by_eos: bool,
    pub stopped_by_string: Option<String>,
    pub stopped_by_context: bool,
    pub finish_reason: TopKFinishReason,
    pub final_position: usize,
}

pub struct InferenceEngine<R: ModelRunner> {
    runner: R,
    sampler: Sampler,
    history: Vec<u32>,
}

impl<R: ModelRunner> InferenceEngine<R> {
    pub fn new(runner: R, sampling: SamplingConfig) -> Self {
        Self {
            runner,
            sampler: Sampler::new(sampling),
            history: Vec::new(),
        }
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn reset_session(&mut self) -> Result<()> {
        self.runner.reset_session()?;
        self.history.clear();
        Ok(())
    }

    pub fn prefill_text(&mut self, prompt: &str) -> Result<PrefillOutput> {
        let tokens = self.runner.encode(prompt)?;
        self.prefill_tokens(tokens)
    }

    pub fn prefill_text_checked(&mut self, prompt: &str, ctx_size: usize) -> Result<PrefillOutput> {
        let tokens = self.runner.encode(prompt)?;
        ensure_context_room(self.history.len(), tokens.len(), ctx_size)?;
        self.prefill_tokens(tokens)
    }

    fn prefill_tokens(&mut self, tokens: Vec<u32>) -> Result<PrefillOutput> {
        let t0 = Instant::now();
        let logits = self.runner.prefill(&tokens)?;
        let prefill_time = t0.elapsed();
        self.history.extend_from_slice(&tokens);
        Ok(PrefillOutput {
            tokens,
            logits,
            prefill_time,
        })
    }

    pub fn generate_text<F>(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(&TokenEvent) -> Result<()>,
    {
        ferrule_common::observability::METRICS.request_started();
        self.reset_session()?;
        let prefill = self.prefill_text_checked(prompt, config.ctx_size)?;
        self.generate_from_logits(
            prefill.logits,
            prefill.tokens.len(),
            prefill.prefill_time,
            config,
            on_token,
        )
    }

    pub fn generate_from_logits<F>(
        &mut self,
        mut logits: Vec<f32>,
        prompt_tokens: usize,
        prefill_time: std::time::Duration,
        config: &GenerationConfig,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(&TokenEvent) -> Result<()>,
    {
        let mut generated = Vec::new();
        let mut text = String::new();
        let mut stopped_by_eos = false;
        let mut stopped_by_string = None;
        let mut all_logprobs = Vec::new();
        let decode_start = Instant::now();

        for index in 0..config.max_new_tokens {
            if logits.is_empty() || self.history.len() >= config.ctx_size {
                break;
            }

            let next = self.sampler.sample(&logits, &self.history);
            self.history.push(next);

            if config.stop_at_eos && self.runner.eos_token_id() == Some(next) {
                stopped_by_eos = true;
                if config.append_eos_to_session {
                    let _ = self.runner.decode_token(next)?;
                }
                break;
            }

            let piece = self.runner.decode(&[next])?;

            let logprobs = if config.logprobs_k > 0 {
                let top = self.sampler.top_logprobs(&logits, config.logprobs_k);
                if top.is_empty() {
                    None
                } else {
                    let lp = Logprobs {
                        token: next,
                        text: piece.clone(),
                        entries: top,
                    };
                    all_logprobs.push(lp.clone());
                    Some(lp)
                }
            } else {
                None
            };

            let event = TokenEvent {
                index,
                token: next,
                text: piece.clone(),
                logprobs,
            };
            on_token(&event)?;

            generated.push(next);
            text.push_str(&piece);

            if let Some(stop) = matched_stop(&text, &config.stop) {
                stopped_by_string = Some(stop);
                let _ = self.runner.decode_token(next)?;
                break;
            }

            logits = self.runner.decode_token(next)?;
        }

        ferrule_common::observability::METRICS
            .prompt_tokens
            .fetch_add(prompt_tokens as u64, std::sync::atomic::Ordering::Relaxed);
        ferrule_common::observability::METRICS.request_finished();
        let stats = GenerateStats {
            prompt_tokens,
            generated_tokens: generated.len(),
            prefill_time,
            decode_time: decode_start.elapsed(),
        };

        Ok(GenerationResult {
            tokens: generated,
            text,
            stats,
            stopped_by_eos,
            stopped_by_string,
            all_logprobs,
        })
    }
}

/// Run the generic top-k decode loop after the caller has already performed
/// prefill and obtained initial candidates.
///
/// This lower-level primitive is useful for diagnostics that need to do
/// model-specific setup around prefill (for example cache warmup or counter
/// baselining) while still keeping EOS/session/context/stop handling in runtime.
pub fn generate_topk_from_candidates<R, F>(
    runner: &mut R,
    initial_topk: Vec<TokenLogit>,
    prompt_token_count: usize,
    prefill_time: Duration,
    config: &GenerationConfig,
    top_k: usize,
    mut on_token: F,
) -> Result<TopKTurnResult>
where
    R: TopKModelRunner,
    F: FnMut(&TopKTokenEvent) -> Result<()>,
{
    if top_k == 0 {
        return Err(Error::Internal("top_k must be greater than zero".into()));
    }
    if config.ctx_size == 0 {
        return Err(Error::Internal("ctx_size must be greater than zero".into()));
    }

    let decode_start = Instant::now();
    let mut top = initial_topk;
    let mut generated = Vec::new();
    let mut text = String::new();
    let mut stopped_by_eos = false;
    let mut stopped_by_string = None;
    let mut stopped_by_context = false;
    let mut finish_reason = TopKFinishReason::MaxTokens;

    for index in 0..config.max_new_tokens {
        if runner.position() >= config.ctx_size {
            stopped_by_context = true;
            finish_reason = TopKFinishReason::Context;
            break;
        }
        let Some(next) = top.first().copied() else {
            finish_reason = TopKFinishReason::NoCandidate;
            break;
        };

        if config.stop_at_eos && runner.eos_token_id() == Some(next.token_id) {
            stopped_by_eos = true;
            finish_reason = TopKFinishReason::Eos;
            if config.append_eos_to_session {
                runner.feed_token(next.token_id)?;
            }
            break;
        }

        let piece = runner.decode(&[next.token_id])?;
        on_token(&TopKTokenEvent {
            index,
            token: next.token_id,
            logit: next.logit,
            text: piece.clone(),
        })?;

        generated.push(next.token_id);
        text.push_str(&piece);

        if let Some(stop) = matched_stop(&text, &config.stop) {
            runner.feed_token(next.token_id)?;
            stopped_by_string = Some(stop);
            finish_reason = TopKFinishReason::StopString;
            break;
        }

        if index + 1 == config.max_new_tokens {
            runner.feed_token(next.token_id)?;
            break;
        }

        top = runner.decode_topk(next.token_id, top_k)?;
    }

    Ok(TopKTurnResult {
        prompt_tokens: prompt_token_count,
        tokens: generated,
        text,
        prefill_time,
        decode_time: decode_start.elapsed(),
        stopped_by_eos,
        stopped_by_string,
        stopped_by_context,
        finish_reason,
        final_position: runner.position(),
    })
}

#[derive(Debug, Clone)]
pub struct PrefillOutput {
    pub tokens: Vec<u32>,
    pub logits: Vec<f32>,
    pub prefill_time: std::time::Duration,
}

pub(crate) fn matched_stop(text: &str, stop: &[String]) -> Option<String> {
    stop.iter()
        .find(|candidate| !candidate.is_empty() && text.ends_with(candidate.as_str()))
        .cloned()
}

pub(crate) fn ensure_context_room(
    current_tokens: usize,
    new_tokens: usize,
    ctx_size: usize,
) -> Result<()> {
    if ctx_size == 0 {
        return Err(Error::Internal("ctx_size must be greater than zero".into()));
    }
    let requested = current_tokens.saturating_add(new_tokens);
    if requested > ctx_size {
        return Err(Error::Internal(format!(
            "context length {requested} exceeds ctx_size {ctx_size}"
        )));
    }
    Ok(())
}
