//! Runtime execution summaries and benchmark counters.

use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimingCounters {
    pub load_us: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    /// MoE calls observed in the CUDA operator path.
    pub moe_calls: u64,
    /// Env-gated GPU-synchronized MoE timings. These stay zero unless
    /// `FERRULE_CUDA_MOE_TIMING=1` is set for the run.
    pub moe_total_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_down_us: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCounters {
    pub launches: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferCounters {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCounters {
    pub peak_host_bytes: Option<u64>,
    pub peak_device_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCounters {
    pub timing: RuntimeTimingCounters,
    pub kernels: KernelCounters,
    pub transfers: TransferCounters,
    pub memory: MemoryCounters,
}

#[allow(dead_code)]
impl RuntimeCounters {
    pub fn record_load(&mut self, elapsed: Duration) {
        self.timing.load_us = self.timing.load_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_prefill(&mut self, elapsed: Duration) {
        self.timing.prefill_us = self.timing.prefill_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_decode(&mut self, elapsed: Duration) {
        self.timing.decode_us = self.timing.decode_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_kernel_launches(&mut self, launches: u64) {
        self.kernels.launches = self.kernels.launches.saturating_add(launches);
    }

    pub fn record_host_to_device(&mut self, bytes: u64) {
        self.transfers.host_to_device_copies =
            self.transfers.host_to_device_copies.saturating_add(1);
        self.transfers.host_to_device_bytes =
            self.transfers.host_to_device_bytes.saturating_add(bytes);
    }

    pub fn record_device_to_host(&mut self, bytes: u64) {
        self.transfers.device_to_host_copies =
            self.transfers.device_to_host_copies.saturating_add(1);
        self.transfers.device_to_host_bytes =
            self.transfers.device_to_host_bytes.saturating_add(bytes);
    }

    pub fn record_artifact_uploads(&mut self, uploads: u64, bytes: u64) {
        self.transfers.artifact_uploads = self.transfers.artifact_uploads.saturating_add(uploads);
        self.transfers.artifact_upload_bytes =
            self.transfers.artifact_upload_bytes.saturating_add(bytes);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeBenchSummary {
    pub counters: RuntimeCounters,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub kernel_launches_per_token: f64,
    pub host_to_device_bytes_per_token: f64,
    pub device_to_host_bytes_per_token: f64,
}

impl RuntimeBenchSummary {
    pub fn new(counters: RuntimeCounters, prompt_tokens: usize, generated_tokens: usize) -> Self {
        let total_tokens = prompt_tokens.saturating_add(generated_tokens).max(1) as f64;
        let prefill_tok_per_s = rate(prompt_tokens, counters.timing.prefill_us);
        let decode_tok_per_s = rate(generated_tokens, counters.timing.decode_us);
        Self {
            kernel_launches_per_token: counters.kernels.launches as f64 / total_tokens,
            host_to_device_bytes_per_token: counters.transfers.host_to_device_bytes as f64
                / total_tokens,
            device_to_host_bytes_per_token: counters.transfers.device_to_host_bytes as f64
                / total_tokens,
            counters,
            prompt_tokens,
            generated_tokens,
            prefill_tok_per_s,
            decode_tok_per_s,
        }
    }
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn rate(tokens: usize, elapsed_us: u64) -> f64 {
    if tokens == 0 || elapsed_us == 0 {
        0.0
    } else {
        tokens as f64 / (elapsed_us as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_counters_compute_per_token_rates() {
        let mut counters = RuntimeCounters::default();
        counters.timing.prefill_us = 2_000_000;
        counters.timing.decode_us = 500_000;
        counters.kernels.launches = 30;
        counters.record_host_to_device(120);
        counters.record_device_to_host(60);

        let summary = RuntimeBenchSummary::new(counters, 4, 6);
        assert_eq!(summary.prefill_tok_per_s, 2.0);
        assert_eq!(summary.decode_tok_per_s, 12.0);
        assert_eq!(summary.kernel_launches_per_token, 3.0);
        assert_eq!(summary.host_to_device_bytes_per_token, 12.0);
        assert_eq!(summary.device_to_host_bytes_per_token, 6.0);
    }
}
