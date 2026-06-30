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
