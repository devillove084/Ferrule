#![allow(clippy::unnecessary_sort_by, clippy::needless_range_loop)]
//! Ferrule Runtime — state-aware generation loops over model backends.
//!
//! This crate keeps tokenization, prefill/decode state, sampling, and chat
//! formatting out of the CLI so CPU and GPU backends share the same behavior.

pub mod chat;
pub mod config;
pub mod constraint;
pub mod generation;
pub mod kv;
pub mod paged_kv;
pub mod perplexity;
pub mod precision;
pub mod prefix_cache;
pub mod profiler;
pub mod program;
pub mod radix_cache;
pub mod residency;
pub mod runner;
pub mod sampler;
pub mod scheduler;
pub mod session;
pub mod stats;
pub mod structured;
pub mod tokenizer;

pub use chat::{detect_chat_template, ChatTemplate};
pub use config::ModelGenerationDefaults;
pub use generation::{GenerationConfig, GenerationResult, InferenceEngine, TokenEvent};
pub use kv::{ContiguousKvCache, KvCache, MultiSessionKvCache};
pub use profiler::Profiler;
pub use program::GenerationProgram;
pub use runner::{CpuOlmoeRunner, ModelInfo, ModelRunner};
pub use sampler::{Logprobs, Sampler, SamplingConfig};

/// Argmax: return the index of the maximum logit value.
/// Used by golden-token regression tests.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, logits[0]),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0 as u32
}
pub use scheduler::{BatchedScheduler, PreemptionPolicy, Scheduler};
pub use session::{GenerateRequest, RequestId, SequenceState, SequenceStatus, SessionId};
pub use stats::GenerateStats;

#[cfg(feature = "cuda")]
pub use runner::GpuOlmoeRunner;
