use super::*;

#[test]
fn reported_moe_cpu_forwards_kv_trace_lifecycle() {
    let mut reported = reported_cpu_engine();
    let engine: &mut dyn ferrule_runtime::engine::SessionInferenceEngine = &mut reported;
    assert!(engine.drain_kv_trace().is_none());
    assert!(engine.finalize_kv_trace().is_none());
    let config = ferrule_runtime::RuntimeKvTraceConfig {
        run_scope: "reported-moe-cpu-test".into(),
        ..Default::default()
    };
    engine.enable_kv_trace(config.clone()).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(1),
            session_id: None,
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 0,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    engine.cancel_request(RequestId(1)).unwrap();
    assert_eq!(engine.drain_cancelled().len(), 1);

    let drained = engine
        .drain_kv_trace()
        .expect("reported wrapper must forward trace drain");
    assert_eq!(drained.start_seq, 0);
    assert!(drained.next_seq > 0);
    assert!(!drained.lifecycle_events.is_empty());
    assert!(!drained.finalized);
    assert!(!drained.complete);
    assert_eq!(drained.header.config, config);

    let finalized = engine
        .finalize_kv_trace()
        .expect("reported wrapper must forward trace finalize");
    assert!(finalized.finalized);
    assert!(finalized.complete);
    assert_eq!(finalized.start_seq, drained.next_seq);
    assert_eq!(finalized.start_seq, finalized.next_seq);
    assert!(finalized.lifecycle_events.is_empty());
    ferrule_runtime::engine::project_runtime_kv_trace(&[drained, finalized]).unwrap();
    assert!(engine.drain_kv_trace().is_none());
    assert!(engine.finalize_kv_trace().is_none());
    assert!(engine.enable_kv_trace(config).is_err());
}

#[test]
fn reported_moe_cpu_forwards_kv_trace_errors_unchanged() {
    use ferrule_runtime::engine::InferenceEngine;
    let mut engine = reported_cpu_engine();
    let direct = engine
        .inner_mut()
        .enable_kv_trace(Default::default())
        .unwrap_err();
    let forwarded = engine.enable_kv_trace(Default::default()).unwrap_err();
    match (direct, forwarded) {
        (
            Error::InvalidRequest { message: direct },
            Error::InvalidRequest { message: forwarded },
        ) => assert_eq!(forwarded, direct),
        other => panic!("wrapper changed the trace error: {other:?}"),
    }
    assert!(engine.drain_kv_trace().is_none());
    assert!(engine.finalize_kv_trace().is_none());
    let config = ferrule_runtime::RuntimeKvTraceConfig {
        run_scope: "trace-enable-error".into(),
        ..Default::default()
    };
    engine.enable_kv_trace(config.clone()).unwrap();
    let direct = engine
        .inner_mut()
        .enable_kv_trace(config.clone())
        .unwrap_err();
    let forwarded = engine.enable_kv_trace(config).unwrap_err();
    match (direct, forwarded) {
        (
            Error::InvalidRequest { message: direct },
            Error::InvalidRequest { message: forwarded },
        ) => assert_eq!(forwarded, direct),
        other => panic!("wrapper changed the repeated-enable error: {other:?}"),
    }
}

#[test]
fn reported_moe_cpu_forwards_kv_trace_unsupported_and_overflow_status() {
    for overflow in [false, true] {
        let mut reported = reported_cpu_engine();
        let engine: &mut dyn ferrule_runtime::engine::SessionInferenceEngine = &mut reported;
        engine
            .enable_kv_trace(ferrule_runtime::RuntimeKvTraceConfig {
                run_scope: "trace-incomplete".into(),
                max_events: if overflow { 0 } else { 65_536 },
                ..Default::default()
            })
            .unwrap();
        engine.retain_session(SessionId(7)).unwrap();
        let request = |id| GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(7)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 0,
            stop: vec![],
            ignore_eos: true,
        };
        engine.try_submit(request(1)).unwrap();
        let mut finished = false;
        for _ in 0..100 {
            engine.step(&mut |_| Ok(())).unwrap();
            if engine.take_request_terminal(RequestId(1)).is_some() {
                finished = true;
                break;
            }
        }
        assert!(finished, "CPU prefill must finish within the step budget");
        assert_eq!(engine.retained_session_position(SessionId(7)), Some(3));
        engine.try_submit(request(2)).unwrap();
        engine.cancel_request(RequestId(2)).unwrap();
        assert_eq!(engine.drain_cancelled().len(), 1);
        let snapshot = engine.finalize_kv_trace().unwrap();
        assert!(snapshot.finalized);
        assert!(!snapshot.complete);
        assert!(
            snapshot.status.unsupported,
            "retained ancestry must not become supported in the wrapper"
        );
        assert_eq!(snapshot.status.overflow, overflow);
        assert!(ferrule_runtime::engine::project_runtime_kv_trace(&[snapshot]).is_err());
    }
}
