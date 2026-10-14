use super::fixture::*;
use ferrule_composition::{BackendSelection, ResidentModelPlanner};
use ferrule_runtime::{GenerateRequest, InferenceShutdownProgress, RequestId, SessionId};

/// Through the object-safe serving factory, not a second runner or test-owned KV
/// manager. The separate model full-state test covers per-layer KV/COW/rollback.
#[test]
#[ignore = "requires exclusive CUDA SM86/CUDA12.4; native HF BF16 independent retained-turn oracle"]
fn public_factory_bf16_greedy_decode_and_retained_turns_match_native_hf() {
    let f = Fixture::new();
    let oracle: serde_json::Value = serde_json::from_str(include_str!("hf_serving.json")).unwrap();
    assert_eq!(oracle["inverse_frequency_dtype"], "float32");
    let mut comparisons = 0;
    let mut max_error = 0f32;
    for chunk in [1, 3] {
        let mut o = options();
        o.scheduler_config.prefill_chunk_size = chunk;
        let plan = ResidentModelPlanner
            .prepare_serving(&f.config(), BackendSelection::Cuda, None, o, serving())
            .unwrap();
        let budget = plan.dense_bf16_memory_plan().unwrap().unwrap();
        let mut engine = plan.build().unwrap();
        assert_eq!(engine.bound_layer_count(), Some(2));
        let kv = engine.observability_snapshot().kv_cache.unwrap();
        assert_eq!(kv.configured_bytes, Some(budget.kv_bytes));
        let session = SessionId(40);
        engine.retain_session(session).unwrap();
        for (turn, case) in oracle["turns"].as_array().unwrap().iter().enumerate() {
            if case["reset"].as_bool().unwrap() {
                engine.reset_session(session).unwrap();
            }
            let id = RequestId(100 + turn as u64);
            engine
                .try_submit(GenerateRequest {
                    id,
                    session_id: Some(session),
                    prompt_tokens: serde_json::from_value(case["prompt"].clone()).unwrap(),
                    max_new_tokens: 2,
                    stop: vec![],
                    ignore_eos: true,
                })
                .unwrap();
            let mut events = Vec::new();
            let mut completed = false;
            for _ in 0..64 {
                engine
                    .step(&mut |event| {
                        events.push(event.clone());
                        Ok(())
                    })
                    .unwrap();
                assert!(engine.drain_failed().is_empty());
                if engine.take_request_terminal(id).is_some() {
                    completed = true;
                    break;
                }
            }
            assert!(completed, "retained request did not finish");
            assert_eq!(events.len(), 2);
            for (i, event) in events.iter().enumerate() {
                assert_eq!(event.token as u64, case["generated"][i].as_u64().unwrap());
                let expected = case["logits"][i].as_f64().unwrap() as f32;
                let actual = event.logit.unwrap();
                max_error = max_error.max((actual - expected).abs());
                assert_eq!(
                    actual.to_bits() & 0xffff,
                    0,
                    "LM-head exports actual BF16 values"
                );
                assert!(
                    (actual - expected).abs() <= 0.00025 + 0.012 * expected.abs(),
                    "chunk={chunk}, turn={turn}, token={i}: CUDA {actual}, HF {expected}"
                );
                comparisons += 1;
            }
            assert_eq!(
                engine.retained_session_position(session),
                Some(case["retained_position"].as_u64().unwrap() as usize)
            );
        }
        engine.reset_session(session).unwrap();
        assert_eq!(engine.retained_session_position(session), Some(0));
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
    }
    assert_eq!(comparisons, 12);
    eprintln!("PUBLIC_BF16_ORACLE comparisons={comparisons} max_abs_error={max_error}");
}
