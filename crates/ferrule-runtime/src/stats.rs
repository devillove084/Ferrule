use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl GenerateStats {
    pub fn total_time(&self) -> Duration {
        self.prefill_time + self.decode_time
    }

    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_time)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.decode_time)
    }

    pub fn total_tokens_per_second(&self) -> f64 {
        rate(
            self.prompt_tokens + self.generated_tokens,
            self.total_time(),
        )
    }
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if tokens == 0 || secs <= 0.0 {
        0.0
    } else {
        tokens as f64 / secs
    }
}

#[derive(Debug, Clone)]
pub struct TokenDebug {
    pub verbose: bool,
    pub token_history: Vec<TokenDebugEntry>,
}

#[derive(Debug, Clone)]
pub struct TokenDebugEntry {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32, String)>,
}

impl TokenDebug {
    pub fn new(verbose: bool) -> Self {
        Self {
            verbose,
            token_history: Vec::new(),
        }
    }
    pub fn record(
        &mut self,
        step: usize,
        token_id: u32,
        text: String,
        logprob: f32,
        top_k: &[(u32, f32)],
        decode_token: impl Fn(u32) -> String,
    ) {
        if self.verbose {
            eprintln!("  step {step}: token={token_id} text='{text}' logprob={logprob:.4}");
        }
        self.token_history.push(TokenDebugEntry {
            step,
            token_id,
            token_text: text,
            logprob,
            top_logprobs: top_k
                .iter()
                .map(|&(t, lp)| (t, lp, decode_token(t)))
                .collect(),
        });
    }
}
