//! Resident engine worker over model-runner capabilities.
//!
//! This module is the runtime boundary that moves Ferrule toward SGLang/vLLM-style
//! long-lived workers without letting `ferrule-runtime` own a concrete model.
//! It keeps the first slice deliberately small: a single resident runner, an
//! explicit prompt-append phase, and an incremental decode phase over model
//! capabilities.

mod driver;
mod lazy;
mod topk_compatibility;
mod worker;

pub use driver::{
    ResidentActionKind, ResidentDriverStep, ResidentTokenEvent, ResidentTopKDriver,
    ResidentTopKDriverConfig, ResidentTopKDriverStats,
};
pub use lazy::LazyEngineWorker;
pub use topk_compatibility::{TopKCompatibilityExecutor, TopKCompatibilityExecutorConfig};
pub use worker::{EngineWorker, EngineWorkerStats, TopKDecodeState, TopKDecodeStep};

#[cfg(test)]
mod tests {
    use ferrule_common::Result;
    use ferrule_model::{ModelInfo, ModelRunner, PrefillMode, TokenLogit, TopKModelRunner};

    use crate::{GenerationConfig, SessionId, TopKFinishReason};

    use super::*;

    #[derive(Debug, Default)]
    struct MockTopKRunner {
        position: usize,
        fed: Vec<u32>,
    }

    impl ModelRunner for MockTopKRunner {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                family: ferrule_model::ModelFamily::Unknown("mock".into()),
                architecture: Some("mock".into()),
                attention: ferrule_model::AttentionKind::Unknown("mock".into()),
                weight_source: ferrule_model::WeightSource::Unknown,
                hidden_size: 1,
                num_layers: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                vocab_size: 8,
                backend: "mock",
            }
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens
                .iter()
                .map(|token| char::from_u32(*token).unwrap_or('?'))
                .collect())
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.position += tokens.len();
            Ok(vec![0.0, 1.0])
        }

        fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
            self.fed.push(token);
            self.position += 1;
            Ok(vec![0.0, 1.0])
        }

        fn reset_session(&mut self) -> Result<()> {
            self.position = 0;
            self.fed.clear();
            Ok(())
        }

        fn eos_token_id(&self) -> Option<u32> {
            None
        }

        fn bound_layer_count(&self) -> Option<usize> {
            Some(1)
        }
    }

    impl TopKModelRunner for MockTopKRunner {
        fn position(&self) -> usize {
            self.position
        }

        fn feed_token(&mut self, token_id: u32) -> Result<()> {
            self.fed.push(token_id);
            self.position += 1;
            Ok(())
        }

        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            _top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.position += token_ids.len();
            Ok(vec![TokenLogit {
                token_id: b'a' as u32,
                logit: 1.0,
            }])
        }

        fn decode_topk(&mut self, token_id: u32, _top_k: usize) -> Result<Vec<TokenLogit>> {
            self.feed_token(token_id)?;
            Ok(vec![TokenLogit {
                token_id: b'b' as u32,
                logit: 0.9,
            }])
        }
    }

    #[test]
    fn worker_tracks_turns_and_resets_runner() {
        let mut worker = EngineWorker::with_session(MockTopKRunner::default(), SessionId(7));
        let cfg = GenerationConfig {
            max_new_tokens: 2,
            stop_at_eos: true,
            append_eos_to_session: true,
            logprobs_k: 0,
            ctx_size: 16,
            stop: Vec::new(),
        };

        let mut decode = worker
            .append_prompt(&[1, 2], &cfg, PrefillMode::Interactive, 1)
            .unwrap();
        let turn = loop {
            match worker.decode_next(&mut decode).unwrap() {
                TopKDecodeStep::Token(_) => {}
                TopKDecodeStep::Finished(turn) => break turn,
            }
        };
        assert_eq!(turn.tokens, vec![b'a' as u32, b'b' as u32]);
        assert_eq!(turn.finish_reason, TopKFinishReason::MaxTokens);
        assert_eq!(
            worker.session().finish_reason,
            Some(TopKFinishReason::MaxTokens)
        );

        let stats = worker.stats();
        assert_eq!(stats.session_id, SessionId(7));
        assert_eq!(stats.turns, 1);
        assert_eq!(stats.prompt_tokens, 2);
        assert_eq!(stats.generated_tokens, 2);
        assert_eq!(stats.bound_layers, Some(1));
        assert_eq!(stats.position, 4);
        assert_eq!(worker.session().position, 4);

        worker.reset().unwrap();
        let stats = worker.stats();
        assert_eq!(stats.position, 0);
        assert_eq!(stats.turns, 0);
        assert_eq!(stats.tracked_tokens, 0);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(worker.session().position, 0);
    }

    #[test]
    fn phased_append_and_decode_updates_session_incrementally() {
        let mut worker = EngineWorker::with_session(MockTopKRunner::default(), SessionId(9));
        let cfg = GenerationConfig {
            max_new_tokens: 2,
            stop_at_eos: true,
            append_eos_to_session: true,
            logprobs_k: 0,
            ctx_size: 16,
            stop: Vec::new(),
        };

        let mut decode = worker
            .append_prompt(&[42], &cfg, PrefillMode::Interactive, 1)
            .unwrap();
        assert_eq!(decode.prompt_tokens(), 1);
        assert_eq!(worker.stats().prompt_tokens, 1);
        assert_eq!(worker.stats().generated_tokens, 0);
        assert_eq!(worker.session().position, 1);

        match worker.decode_next(&mut decode).unwrap() {
            TopKDecodeStep::Token(event) => assert_eq!(event.token, b'a' as u32),
            TopKDecodeStep::Finished(_) => panic!("first step should emit a token"),
        }
        assert_eq!(worker.stats().generated_tokens, 1);
        assert_eq!(decode.generated_tokens(), &[b'a' as u32]);
        assert_eq!(worker.session().position, 2);

        match worker.decode_next(&mut decode).unwrap() {
            TopKDecodeStep::Token(event) => assert_eq!(event.token, b'b' as u32),
            TopKDecodeStep::Finished(_) => panic!("second step should emit a token"),
        }
        assert!(decode.is_finished());
        assert_eq!(worker.stats().generated_tokens, 2);
        assert_eq!(worker.session().position, 3);

        let result = match worker.decode_next(&mut decode).unwrap() {
            TopKDecodeStep::Token(_) => panic!("finished state should not emit more tokens"),
            TopKDecodeStep::Finished(result) => result,
        };
        assert_eq!(result.tokens, vec![b'a' as u32, b'b' as u32]);
        assert_eq!(result.finish_reason, TopKFinishReason::MaxTokens);
        assert_eq!(
            worker.session().finish_reason,
            Some(TopKFinishReason::MaxTokens)
        );
        assert_eq!(worker.stats().turns, 1);
    }

    #[test]
    fn cancel_decode_finishes_with_cancelled_reason() {
        let mut worker = EngineWorker::with_session(MockTopKRunner::default(), SessionId(11));
        let cfg = GenerationConfig {
            max_new_tokens: 4,
            stop_at_eos: true,
            append_eos_to_session: true,
            logprobs_k: 0,
            ctx_size: 16,
            stop: Vec::new(),
        };

        let mut decode = worker
            .append_prompt(&[42], &cfg, PrefillMode::Interactive, 1)
            .unwrap();
        match worker.decode_next(&mut decode).unwrap() {
            TopKDecodeStep::Token(event) => assert_eq!(event.token, b'a' as u32),
            TopKDecodeStep::Finished(_) => panic!("first step should emit a token"),
        }

        let result = worker.cancel_decode(&mut decode);
        assert_eq!(result.tokens, vec![b'a' as u32]);
        assert_eq!(result.finish_reason, TopKFinishReason::Cancelled);
        assert!(decode.is_finished());
        assert_eq!(decode.finish_reason(), Some(TopKFinishReason::Cancelled));
        assert_eq!(
            worker.session().finish_reason,
            Some(TopKFinishReason::Cancelled)
        );
        assert_eq!(worker.stats().turns, 1);
    }

    #[test]
    fn lazy_worker_loads_artifact_then_builds_runner_once() {
        let mut lazy = LazyEngineWorker::spawn(
            SessionId(3),
            || Ok(vec![1_u8, 2, 3]),
            |artifact| {
                assert_eq!(artifact, vec![1, 2, 3]);
                Ok(MockTopKRunner::default())
            },
        );
        assert!(!lazy.is_loaded());
        let worker = lazy.ensure_loaded().unwrap();
        assert_eq!(worker.stats().session_id, SessionId(3));
        assert!(lazy.is_loaded());
        assert!(lazy.ensure_loaded().is_ok());
    }
}
