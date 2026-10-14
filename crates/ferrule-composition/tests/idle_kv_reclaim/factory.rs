//! Real factory -> boxed engine -> same suspension journal. No mock attention.
#![cfg(feature = "cuda")]
use super::*;
use crate::model_factory::resident_f32_bound;
use ferrule_runtime::{
    GenerateRequest, IdleKvReclaimConfig, InferenceShutdownProgress, KvExecutionPolicy, SessionId,
};
fn config() -> IdleKvReclaimConfig {
    IdleKvReclaimConfig {
        staging_page_limit: 4,
        device_bytes: 1 << 20,
        host_bytes: 1 << 20,
        pinned_bytes: 1 << 20,
    }
}
fn build(reclaim: bool) -> (host_kv_fixture::Fixture, BoxedSessionInferenceEngine) {
    let fixture = host_kv_fixture::Fixture::new();
    let descriptor =
        super::super::super::tests::descriptor(ModelFamily::Qwen35, "host-kv-two-gqa", 2);
    let mut options = super::super::super::tests::options();
    options.driver_config.ctx_size = 16;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_active_sequences = 2;
    options.scheduler_config.max_decode_batch = 2;
    options.scheduler_config.prefill_chunk_size = 2;
    options.scheduler_config.max_batch_tokens = 4;
    options.scheduler_config.prefix_cache_capacity_pages = 0;
    let entry = resolve(&descriptor, BackendSelection::Cuda).unwrap();
    let mut request = configure_request(&descriptor, entry, options).unwrap();
    let policy = if reclaim {
        KvExecutionPolicy::IdleReclaim(config())
    } else {
        KvExecutionPolicy::Resident
    };
    request.execution.kv_execution_policy = policy;
    let mut engine = resident_f32_bound::build_resident_f32_bound(
        request,
        fixture.resources(),
        fixture.tokenizer(),
    )
    .unwrap();
    engine.configure_kv_execution(policy).unwrap();
    (fixture, engine)
}
fn submit(engine: &mut BoxedSessionInferenceEngine, id: u64, session: u64, prompt: Vec<u32>) {
    engine
        .try_submit(GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(session)),
            prompt_tokens: prompt,
            max_new_tokens: 2,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
}
fn finish(engine: &mut BoxedSessionInferenceEngine, count: usize) -> Vec<(u64, u32)> {
    let mut events = Vec::new();
    let mut done = 0;
    for _ in 0..128 {
        engine
            .step(&mut |e| {
                events.push((e.session_id.0, e.token));
                Ok(())
            })
            .unwrap();
        done += engine.drain_finished().len();
        assert!(engine.drain_failed().is_empty());
        if done == count {
            return events;
        }
    }
    panic!("engine failed to finish");
}
#[test]
#[ignore = "CUDA 12.4 idle GPU owner only; compile-only during implementation"]
fn engine_two_sessions_idle_offload_one_restore_and_continue_matches_fixed() {
    let (_fixture, mut engine) = build(true);
    assert!(
        engine
            .configure_kv_execution(KvExecutionPolicy::HostKvDecode(Default::default()))
            .is_err()
    );
    assert!(
        engine
            .configure_kv_execution(KvExecutionPolicy::Resident)
            .is_err()
    );
    for session in [SessionId(1), SessionId(2)] {
        engine.retain_session(session).unwrap();
    }
    submit(&mut engine, 1, 1, vec![1, 3, 2]);
    submit(&mut engine, 2, 2, vec![2, 3, 1]);
    let mut actual = finish(&mut engine, 2);
    let frontier = engine.retained_session_position(SessionId(1)).unwrap();
    let before = engine
        .observability_snapshot()
        .kv_cache
        .unwrap()
        .idle_physical
        .unwrap();
    let logical = engine.observability_snapshot().kv_cache.unwrap().stats;
    engine.suspend_session_kv(SessionId(1)).unwrap();
    assert!(engine.suspend_session_kv(SessionId(1)).is_err());
    let off = engine
        .observability_snapshot()
        .kv_cache
        .unwrap()
        .idle_physical
        .unwrap();
    assert!(off.live_payload_bytes > 0 && off.live_payload_bytes < before.live_payload_bytes);
    assert_eq!(
        before.live_payload_bytes - off.live_payload_bytes,
        off.host_snapshot_bytes
    );
    assert_eq!(
        engine.retained_session_position(SessionId(1)),
        Some(frontier)
    );
    assert_eq!(
        engine
            .observability_snapshot()
            .kv_cache
            .unwrap()
            .stats
            .allocated_pages,
        logical.allocated_pages
    );
    // Other session continues while session 1's authoritative pages stay on host.
    submit(&mut engine, 3, 2, vec![4]);
    actual.extend(finish(&mut engine, 1));
    for _ in 0..100 {
        engine.trim_idle_kv().unwrap();
        if engine
            .observability_snapshot()
            .kv_cache
            .unwrap()
            .idle_physical
            .unwrap()
            .driver_freed_bytes
            > before.driver_freed_bytes
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        engine
            .observability_snapshot()
            .kv_cache
            .unwrap()
            .idle_physical
            .unwrap()
            .driver_freed_bytes
            > before.driver_freed_bytes
    );
    engine.restore_session_kv(SessionId(1)).unwrap();
    assert!(engine.restore_session_kv(SessionId(1)).is_err());
    assert_eq!(
        engine.retained_session_position(SessionId(1)),
        Some(frontier)
    );
    submit(&mut engine, 4, 1, vec![5]);
    actual.extend(finish(&mut engine, 1));
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    let (_reference_fixture, mut reference) = build(false);
    for s in [SessionId(1), SessionId(2)] {
        reference.retain_session(s).unwrap();
    }
    submit(&mut reference, 1, 1, vec![1, 3, 2]);
    submit(&mut reference, 2, 2, vec![2, 3, 1]);
    let mut expected = finish(&mut reference, 2);
    submit(&mut reference, 3, 2, vec![4]);
    expected.extend(finish(&mut reference, 1));
    submit(&mut reference, 4, 1, vec![5]);
    expected.extend(finish(&mut reference, 1));
    let grouped = |events: Vec<(u64, u32)>| {
        let mut sequences = std::collections::BTreeMap::<u64, Vec<u32>>::new();
        for (session, token) in events {
            sequences.entry(session).or_default().push(token);
        }
        sequences
    };
    assert_eq!(grouped(actual), grouped(expected));
    assert_eq!(
        reference.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
}

#[path = "runner.rs"]
mod runner;
