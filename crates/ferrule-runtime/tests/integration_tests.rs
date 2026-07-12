//! Integration tests — composition of runtime modules.
//!
//! Uses mock runners to avoid requiring model files.

use ferrule_common::Result;
use ferrule_model::{AttentionKind, ModelFamily, WeightSource};
use ferrule_model::{ModelInfo, ModelRunner};
use ferrule_runtime::{
    ContiguousKvCache, GenerateRequest, GenerationConfig, InferenceEngine, KvCache,
    MultiSessionKvCache, RequestId, ResidentScheduler, ResidentSchedulerConfig, SamplingConfig,
    SchedulerAction, SequenceFinishReason, SequenceState, SessionId,
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

fn scheduler_request(
    id: u64,
    session_id: Option<SessionId>,
    prompt_tokens: Vec<u32>,
) -> GenerateRequest {
    GenerateRequest {
        id: RequestId(id),
        session_id,
        prompt_tokens,
        sampling: SamplingConfig::greedy(),
        max_new_tokens: 4,
        stop: Vec::new(),
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
fn resident_scheduler_with_multi_session_kv_admits_and_finishes() {
    let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
        prefill_chunk_size: 8,
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    });
    let mut kv = MultiSessionKvCache::new(2, 4, 8, 4);
    scheduler.submit(scheduler_request(1, Some(SessionId(100)), vec![1, 2]));

    let SchedulerAction::PrefillChunk(prefill) = scheduler
        .next_prefill_action(&mut kv)
        .unwrap()
        .expect("resident prefill action")
    else {
        panic!("expected prefill action");
    };
    assert_eq!(prefill.request_id, Some(RequestId(1)));
    assert_eq!(prefill.session_id, SessionId(100));
    assert_eq!(prefill.tokens, vec![1, 2]);
    assert_eq!(scheduler.waiting_len(), 0);
    assert_eq!(scheduler.active_len(), 1);
    assert_eq!(kv.active_count(), 1);

    let handle = prefill.kv_handle.expect("resident KV handle");
    kv.append(handle, 0, &[1., 2., 3., 4.], &[5., 6., 7., 8.])
        .unwrap();
    kv.append(handle, 1, &[9., 10., 11., 12.], &[13., 14., 15., 16.])
        .unwrap();
    assert_eq!(kv.seq_len(handle), 1);
    scheduler.commit_prefill_action(&prefill).unwrap();

    let finish = scheduler
        .finish_sequence(SessionId(100), SequenceFinishReason::MaxTokens, &mut kv)
        .unwrap();
    assert!(matches!(
        finish,
        SchedulerAction::Finish {
            request_id: Some(RequestId(1)),
            session_id: SessionId(100),
            reason: SequenceFinishReason::MaxTokens,
        }
    ));

    let finished = scheduler.drain_finished();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].request_id, Some(RequestId(1)));
    assert_eq!(
        finished[0].finish_reason,
        Some(SequenceFinishReason::MaxTokens)
    );
    assert!(finished[0].kv_handle.is_none());
    assert_eq!(kv.active_count(), 0);
    assert!(scheduler.is_idle());
}

#[test]
fn resident_scheduler_preserves_fifo_across_admission_and_finish() {
    let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
        prefill_chunk_size: 8,
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    });
    let mut kv = MultiSessionKvCache::new(1, 1, 4, 1);
    scheduler.submit(scheduler_request(10, None, vec![10]));
    scheduler.submit(scheduler_request(20, None, vec![20]));

    assert_eq!(scheduler.total_submitted(), 2);
    assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);
    assert_eq!(scheduler.active_len(), 1);
    assert_eq!(scheduler.waiting_len(), 1);
    assert_eq!(kv.active_count(), 1);

    let SchedulerAction::PrefillChunk(first) = scheduler
        .next_prefill_action(&mut kv)
        .unwrap()
        .expect("first FIFO prefill")
    else {
        panic!("expected first prefill action");
    };
    assert_eq!(first.request_id, Some(RequestId(10)));
    scheduler.commit_prefill_action(&first).unwrap();
    let first_finish = scheduler
        .finish_sequence(first.session_id, SequenceFinishReason::MaxTokens, &mut kv)
        .unwrap();
    assert!(matches!(
        first_finish,
        SchedulerAction::Finish {
            request_id: Some(RequestId(10)),
            ..
        }
    ));
    assert_eq!(scheduler.active_len(), 0);
    assert_eq!(kv.active_count(), 0);

    assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);
    assert_eq!(scheduler.waiting_len(), 0);
    let SchedulerAction::PrefillChunk(second) = scheduler
        .next_prefill_action(&mut kv)
        .unwrap()
        .expect("second FIFO prefill")
    else {
        panic!("expected second prefill action");
    };
    assert_eq!(second.request_id, Some(RequestId(20)));
    scheduler.commit_prefill_action(&second).unwrap();
    scheduler
        .finish_sequence(second.session_id, SequenceFinishReason::MaxTokens, &mut kv)
        .unwrap();

    let finished_ids = scheduler
        .drain_finished()
        .into_iter()
        .map(|sequence| sequence.request_id.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(finished_ids, vec![RequestId(10), RequestId(20)]);
    assert_eq!(kv.active_count(), 0);
    assert!(scheduler.is_idle());
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
