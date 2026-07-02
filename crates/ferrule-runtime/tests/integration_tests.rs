//! Integration tests — composition of runtime modules.
//!
//! Uses mock runners to avoid requiring model files.

use std::path::{Path, PathBuf};

use ferrule_core::Result;
use ferrule_model::{AttentionKind, ModelFamily, WeightSource};
use ferrule_runtime::{
    ContiguousKvCache, GenerationConfig, InferenceEngine, KvCache, ModelInfo, ModelRunner,
    MultiSessionKvCache, SamplingConfig, Scheduler, SequenceState, SessionId,
};

// ── Mock runner ─────────────────────────────────────────────────────────

struct MockRunner {
    vocab_size: usize,
    eos: Option<u32>,
    /// Pre-recorded logits to return on each decode_token call.
    logits_sequence: Vec<Vec<f32>>,
    call_count: usize,
    session_reset_count: usize,
}

impl MockRunner {
    fn new(vocab_size: usize, logits: Vec<Vec<f32>>) -> Self {
        Self {
            vocab_size,
            eos: Some(vocab_size as u32 - 1),
            logits_sequence: logits,
            call_count: 0,
            session_reset_count: 0,
        }
    }

    fn next_logits(&mut self) -> Vec<f32> {
        if self.call_count < self.logits_sequence.len() {
            let l = self.logits_sequence[self.call_count].clone();
            self.call_count += 1;
            l
        } else {
            // Return uniform logits after sequence exhausted
            vec![0.0f32; self.vocab_size]
        }
    }
}

impl ModelRunner for MockRunner {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            family: ModelFamily::Unknown("mock".into()),
            architecture: Some("mock".into()),
            attention: AttentionKind::GroupedQuery,
            weight_source: WeightSource::Unknown,
            hidden_size: 64,
            num_layers: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            vocab_size: self.vocab_size,
            backend: "mock",
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Simple mock: each char → token id
        Ok(text
            .chars()
            .map(|c| c as u32 % self.vocab_size as u32)
            .collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(tokens
            .iter()
            .map(|t| format!("[{t}]"))
            .collect::<Vec<_>>()
            .join(""))
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut last = Vec::new();
        for _ in tokens {
            last = self.next_logits();
        }
        Ok(last)
    }

    fn decode_token(&mut self, _token: u32) -> Result<Vec<f32>> {
        Ok(self.next_logits())
    }

    fn reset_session(&mut self) -> Result<()> {
        self.session_reset_count += 1;
        self.call_count = 0;
        Ok(())
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn engine_generate_text_with_mock() {
    // Logits that make argmax pick tokens 0, 1, 0
    let logits = vec![
        vec![10.0, -1.0, -1.0], // argmax=0
        vec![-1.0, 10.0, -1.0], // argmax=1
        vec![10.0, -1.0, -1.0], // argmax=0 again
    ];
    let runner = MockRunner::new(3, logits.clone());
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    let gen_cfg = GenerationConfig {
        max_new_tokens: 2,
        ..Default::default()
    };

    let result = engine.generate_text("hi", &gen_cfg, |_| Ok(())).unwrap();
    assert_eq!(result.tokens.len(), 2);
    // Token 0 is not EOS (EOS is vocab_size-1 = 2)
    assert!(!result.stopped_by_eos);
    assert_eq!(result.stats.prompt_tokens, 2); // "hi" → 2 chars
}

#[test]
fn engine_stops_on_eos() {
    // EOS = 2, first generated token = 2 (EOS)
    let logits = vec![vec![-1.0, -1.0, 10.0]];
    let runner = MockRunner::new(3, logits);
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    let gen_cfg = GenerationConfig::default();

    let result = engine.generate_text("x", &gen_cfg, |_| Ok(())).unwrap();
    assert!(result.tokens.is_empty());
    assert!(result.stopped_by_eos);
}

#[test]
fn engine_generate_text_with_real_olmoe_if_present() {
    let Some(model_dir) = local_olmoe_dir() else {
        return;
    };

    let runner = ferrule_runtime::RuntimeRunner::load(&model_dir)
        .expect("local OLMoE model should load through RuntimeRunner");
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());
    let gen_cfg = GenerationConfig {
        max_new_tokens: 1,
        append_eos_to_session: false,
        ..Default::default()
    };

    let mut events = 0usize;
    let result = engine
        .generate_text("The capital of France is", &gen_cfg, |_| {
            events += 1;
            Ok(())
        })
        .expect("real OLMoE generation smoke should produce a first-token result");

    assert!(result.stats.prompt_tokens > 0);
    assert!(result.tokens.len() <= 1);
    assert_eq!(events, result.tokens.len());
}

#[test]
fn engine_session_reset() {
    let logits = vec![vec![1.0, 0.0]];
    let runner = MockRunner::new(2, logits);
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    engine
        .generate_text("a", &GenerationConfig::default(), |_| Ok(()))
        .unwrap();
    assert_eq!(engine.runner().session_reset_count, 1); // generate_text resets

    engine.reset_session().unwrap();
    assert_eq!(engine.runner().session_reset_count, 2);
}

#[test]
fn kv_cache_composition() {
    let mut cache = ContiguousKvCache::new(2, 4, 16);
    cache
        .append(0, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
        .unwrap();
    cache
        .append(0, 1, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8])
        .unwrap();

    assert_eq!(cache.seq_len(), 2);
    assert_eq!(cache.k_slice(0).len(), 8);
    assert_eq!(cache.k_slice(0)[0], 1.0);
}

#[test]
fn session_composition_with_engine() {
    let mut session = SequenceState::new(SessionId(1));
    assert_eq!(session.status, ferrule_runtime::SequenceStatus::Pending);

    let logits = vec![vec![1.0, 0.0, -1.0]];
    let runner = MockRunner::new(3, logits);
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    let result = engine
        .generate_text("test", &GenerationConfig::default(), |_| Ok(()))
        .unwrap();

    session.set_prompt(&[1, 2, 3, 4]);
    assert_eq!(session.prompt_len, 4);

    if !result.tokens.is_empty() {
        session.extend_generated(&result.tokens);
        assert!(session.generated > 0);
    }

    session.reset();
    assert_eq!(session.generated, 0);
}

#[test]
fn generation_logprobs_collection() {
    let logits = vec![
        vec![5.0, 2.0, 1.0], // argmax=0
        vec![1.0, 5.0, 2.0], // argmax=1
    ];
    let runner = MockRunner::new(3, logits);
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    let gen_cfg = GenerationConfig {
        max_new_tokens: 2,
        logprobs_k: 2,
        ..Default::default()
    };

    let result = engine.generate_text("x", &gen_cfg, |_| Ok(())).unwrap();

    assert_eq!(result.all_logprobs.len(), 2);
    assert_eq!(result.all_logprobs[0].token, 0);
    assert_eq!(result.all_logprobs[0].entries.len(), 2);
    // Highest probability should be token 0
    assert_eq!(result.all_logprobs[0].entries[0].0, 0);
}

#[test]
fn multiple_sequences_independent() {
    let mut s1 = SequenceState::new(SessionId(100));
    let mut s2 = SequenceState::new(SessionId(200));

    s1.set_prompt(&[1, 2]);
    s2.set_prompt(&[99, 98]);
    s1.extend_generated(&[3]);
    s2.extend_generated(&[97, 96]);

    assert_eq!(s1.tokens, vec![1, 2, 3]);
    assert_eq!(s2.tokens, vec![99, 98, 97, 96]);
    assert_ne!(s1.session_id, s2.session_id);
}

#[test]
fn kv_cache_multilayer_composition() {
    let mut cache = ContiguousKvCache::new(4, 8, 32);
    for layer in 0..4 {
        cache
            .append(layer, 0, &[layer as f32; 8], &[layer as f32; 8])
            .unwrap();
    }
    assert_eq!(cache.k_slice(0)[0], 0.0);
    assert_eq!(cache.k_slice(2)[0], 2.0);

    // Reset should clear all
    cache.reset().unwrap();
    for layer in 0..4 {
        assert_eq!(cache.k_slice(layer).len(), 0);
    }
}

#[test]
fn engine_preserves_history_across_generations() {
    let logits = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    let runner = MockRunner::new(2, logits);
    let mut engine = InferenceEngine::new(runner, SamplingConfig::greedy());

    let _ = engine
        .generate_text("first", &GenerationConfig::default(), |_| Ok(()))
        .unwrap();
    let hist_after_first = engine.history().len();
    assert!(hist_after_first > 0);

    let _ = engine
        .generate_text("second", &GenerationConfig::default(), |_| Ok(()))
        .unwrap();
    let hist_after_second = engine.history().len();
    // After reset (in generate_text), history reflects new session
    assert!(hist_after_second > 0);
}

#[test]
fn scheduler_with_multi_session_kv() {
    let mut s = Scheduler::new(1);
    let mut kv = MultiSessionKvCache::new(2, 4, 8, 4);

    // Session A: allocate KV slot, submit request
    let h_a = kv.alloc().unwrap();
    let req_a = ferrule_runtime::GenerateRequest {
        id: ferrule_runtime::RequestId(1),
        session_id: Some(ferrule_runtime::SessionId(h_a.0 as u64)),
        prompt_tokens: vec![1, 2],
        sampling: SamplingConfig::greedy(),
        max_new_tokens: 4,
        stop: vec![],
    };
    s.submit(req_a);
    s.schedule();
    assert_eq!(s.running_id(), Some(ferrule_runtime::RequestId(1)));

    // Simulate KV writes
    kv.append(h_a, 0, &[1., 2., 3., 4.], &[5., 6., 7., 8.])
        .unwrap();
    kv.append(h_a, 1, &[9., 10., 11., 12.], &[13., 14., 15., 16.])
        .unwrap();
    // One token's KV written across two layers should count as one position.
    assert_eq!(kv.seq_len(h_a), 1);

    s.finish(ferrule_runtime::RequestId(1));
    let done = s.drain_finished();
    assert_eq!(done.len(), 1);

    // Free KV
    kv.free(h_a);
    assert_eq!(kv.active_count(), 0);
}

#[test]
fn scheduler_fifo_ordering() {
    let mut s = Scheduler::new(1);
    s.submit(ferrule_runtime::GenerateRequest {
        id: ferrule_runtime::RequestId(10),
        session_id: None,
        prompt_tokens: vec![],
        sampling: SamplingConfig::greedy(),
        max_new_tokens: 1,
        stop: vec![],
    });
    s.submit(ferrule_runtime::GenerateRequest {
        id: ferrule_runtime::RequestId(20),
        session_id: None,
        prompt_tokens: vec![],
        sampling: SamplingConfig::greedy(),
        max_new_tokens: 1,
        stop: vec![],
    });
    s.schedule();
    assert_eq!(s.running_id(), Some(ferrule_runtime::RequestId(10)));
}

#[test]
fn kv_concurrent_sessions_no_interference() {
    let mut kv = MultiSessionKvCache::new(1, 2, 4, 3);
    let s1 = kv.alloc().unwrap();
    let s2 = kv.alloc().unwrap();

    kv.append(s1, 0, &[1., 1.], &[2., 2.]).unwrap();
    kv.append(s1, 0, &[3., 3.], &[4., 4.]).unwrap();
    kv.append(s2, 0, &[10., 10.], &[20., 20.]).unwrap();

    assert_eq!(kv.k_slice(s1, 0), &[1., 1., 3., 3.]);
    assert_eq!(kv.k_slice(s2, 0), &[10., 10.]);

    kv.free(s1);
    // s2 should be unaffected
    assert_eq!(kv.k_slice(s2, 0), &[10., 10.]);
    assert_eq!(kv.active_count(), 1);
}

fn local_olmoe_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FERRULE_MODEL") {
        let path = PathBuf::from(path);
        assert!(
            path.join("config.json").exists(),
            "FERRULE_MODEL must point at a HF model directory with config.json"
        );
        return Some(path);
    }

    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models")
        .join("OLMoE-Instruct");
    default.join("config.json").exists().then_some(default)
}
