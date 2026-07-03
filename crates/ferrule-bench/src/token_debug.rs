//! Token-level debug recording for CLI/benchmark tooling.

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
                .map(|&(token, logprob)| (token, logprob, decode_token(token)))
                .collect(),
        });
    }
}
