use super::*;
use crate::cuda::runtime::GraphFault;
fn exercise(op: &CudaOperators, provider: &mut crate::cuda::operators::linear::CudaBf16GemmEx) {
    let p = provider.plan();
    let l = p.layout();
    let w = op
        .upload_numeric_fp8_linear(
            l,
            &vec![0x38; l.n * l.k],
            &[0x80, 0x3f].repeat(l.n * l.k / 16384),
        )
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; p.rows() * l.k]).unwrap();
    let mut y = op.zero_bf16_buffer(p.rows() * l.n).unwrap();
    provider.execute(op, &w, &x, &mut y).unwrap();
    assert_eq!(
        op.download_bf16_buffer(&y).unwrap(),
        vec![l.k as f32; p.rows() * l.n]
    );
}
#[test]
#[ignore = "sole GPU owner; two independent owners on one GPU coexist with separate ledger charges"]
fn bf16_gemmex_independent_owners_same_device_coexist() {
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let mut x = a.prepare_bf16_gemmex(p).unwrap();
    let mut y = b.prepare_bf16_gemmex(p).unwrap();
    let ka = (
        Arc::as_ptr(&a.stream_clone()) as usize,
        ProviderFamily::GemmEx,
    );
    let kb = (
        Arc::as_ptr(&b.stream_clone()) as usize,
        ProviderFamily::GemmEx,
    );
    let charge = Charge {
        storage_grant: p.storage_grant_bytes(),
        external: p.provider_bytes(),
    };
    assert_eq!(ledger().owners[&ka], (0, charge));
    assert_eq!(ledger().owners[&kb], (0, charge));
    exercise(&a, &mut x);
    exercise(&b, &mut y);
    drop(x);
    assert!(!ledger().owners.contains_key(&ka));
    assert!(ledger().owners.contains_key(&kb));
    exercise(&b, &mut y);
    drop(y);
    assert!(!ledger().owners.contains_key(&kb));
}
#[test]
#[ignore = "GPU owner must expose two idle sm86 GPUs; run later, never silently skip"]
fn bf16_gemmex_two_devices_coexist() {
    assert!(
        crate::cuda::runtime::CudaContext::device_count().unwrap() >= 2,
        "expose two GPUs for this explicit gate"
    );
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let p = plan(1, 128, 128);
    let mut x = a.prepare_bf16_gemmex(p).unwrap();
    let mut y = b.prepare_bf16_gemmex(p).unwrap();
    assert_eq!(
        ledger().owners[&(
            Arc::as_ptr(&a.stream_clone()) as usize,
            ProviderFamily::GemmEx
        )]
            .0,
        0
    );
    assert_eq!(
        ledger().owners[&(
            Arc::as_ptr(&b.stream_clone()) as usize,
            ProviderFamily::GemmEx
        )]
            .0,
        1
    );
    exercise(&a, &mut x);
    exercise(&b, &mut y);
    drop(x);
    exercise(&b, &mut y);
    drop(y);
}
fn external_completion_failure(fault: GraphFault, facade: bool) {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let w = op
        .upload_numeric_fp8_linear(p.layout(), &vec![0x38; 128 * 128], &[0x80, 0x3f])
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; 128]).unwrap();
    let mut y = op.zero_bf16_buffer(128).unwrap();
    let mut provider = op.prepare_bf16_gemmex(p).unwrap();
    let graph = provider.capture(&op, &w, &x, &mut y).unwrap();
    graph.launch(&op).unwrap();
    let stream = op.stream_clone();
    let event = stream.record_event(None).unwrap();
    let before = op.allocator_metrics().live_requested_bytes;
    stream.context().arm_graph_fault(fault);
    match fault {
        GraphFault::StreamSynchronize if facade => assert!(op.sync_stream().is_err()),
        GraphFault::StreamSynchronize => assert!(stream.synchronize().is_err()),
        GraphFault::EventSynchronize => assert!(event.synchronize().is_err()),
        GraphFault::EventQuery => assert!(event.query().is_err()),
        _ => panic!("completion fault"),
    }
    assert!(stream.context().graph_completion_unknown());
    assert!(provider.is_poisoned());
    assert!(stream.synchronize().is_err());
    assert!(event.synchronize().is_err());
    assert!(event.query().is_err());
    assert!(graph.launch(&op).is_err());
    assert!(graph.download(&op).is_err());
    assert!(provider.execute(&op, &w, &x, &mut y).is_err());
    assert!(op.prepare_bf16_gemmex(p).is_err());
    drop((graph, provider, w, x, y, event));
    assert!(op.allocator_metrics().live_requested_bytes >= before);
    assert!(
        ledger()
            .owners
            .contains_key(&(Arc::as_ptr(&stream) as usize, ProviderFamily::GemmEx))
    );
    // A healthy independently budgeted owner is not held hostage by Unknown.
    let other = CudaOperators::new_on_device(0).unwrap();
    let mut healthy = other.prepare_bf16_gemmex(p).unwrap();
    exercise(&other, &mut healthy);
    drop(healthy);
}
#[test]
#[ignore = "isolated GPU child; facade sync failure after graph submission"]
fn bf16_gemmex_external_operator_sync_unknown() {
    external_completion_failure(GraphFault::StreamSynchronize, true)
}
#[test]
#[ignore = "isolated GPU child; raw owner-stream sync failure after graph submission"]
fn bf16_gemmex_external_stream_sync_unknown() {
    external_completion_failure(GraphFault::StreamSynchronize, false)
}
#[test]
#[ignore = "isolated GPU child; external event sync failure after graph submission"]
fn bf16_gemmex_external_event_sync_unknown() {
    external_completion_failure(GraphFault::EventSynchronize, false)
}
#[test]
#[ignore = "isolated GPU child; external event query failure after graph submission"]
fn bf16_gemmex_external_event_query_unknown() {
    external_completion_failure(GraphFault::EventQuery, false)
}
#[test]
#[ignore = "GPU owner; NOT_READY and capture preflight don't poison healthy graph"]
fn bf16_gemmex_completion_pending_and_preflight_are_retryable() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let mut provider = op.prepare_bf16_gemmex(p).unwrap();
    let stream = op.stream_clone();
    let event = stream.record_event(None).unwrap();
    stream.context().arm_graph_fault(GraphFault::EventNotReady);
    assert!(!event.query().unwrap());
    assert!(!provider.is_poisoned());
    event.synchronize().unwrap();
    stream.begin_capture().unwrap();
    assert!(stream.synchronize().is_err());
    assert!(event.query().is_err());
    assert!(event.synchronize().is_err());
    drop(stream.end_capture().unwrap());
    assert!(!provider.is_poisoned());
    exercise(&op, &mut provider);
}
fn native_unknown(phase: u8) {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let larger = plan(2, 256, 128);
    let baseline = op.allocator_metrics().live_requested_bytes;
    WARMUP_FAULT.set(phase);
    let error = if phase == 1 {
        op.prepare_bf16_gemmex(p)
    } else {
        op.prepare_bf16_gemmex_views(
            &[p, larger],
            larger.storage_grant_bytes(),
            larger.provider_bytes(),
        )
    }
    .err()
    .expect("injected native Unknown");
    assert!(error.to_string().contains("status 9"), "{error}");
    assert!(op.stream_clone().context().graph_completion_unknown());
    assert!(op.allocator_metrics().live_requested_bytes >= baseline + p.storage_bytes());
    assert!(op.prepare_bf16_gemmex(p).is_err());
    assert!(ledger().owners.contains_key(&(
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::GemmEx
    )));
    let other = CudaOperators::new_on_device(0).unwrap();
    let mut healthy = other.prepare_bf16_gemmex(p).unwrap();
    exercise(&other, &mut healthy);
}
#[test]
#[ignore = "isolated GPU child; native first warmup fence Unknown, second fence succeeds"]
fn bf16_gemmex_initial_warmup_unknown_is_sticky() {
    native_unknown(1)
}
#[test]
#[ignore = "isolated GPU child; shape warmup fence Unknown, second fence succeeds"]
fn bf16_gemmex_shape_warmup_unknown_is_sticky() {
    native_unknown(2)
}

#[test]
#[ignore = "GPU owner; failed setup without a native handle retires backing before readmission"]
fn bf16_gemmex_prehandle_failure_retires_before_readmission() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let baseline = op.allocator_metrics().live_requested_bytes;
    WARMUP_FAULT.set(3);
    let error = op
        .prepare_bf16_gemmex(p)
        .err()
        .expect("injected prehandle failure");
    assert!(
        error
            .to_string()
            .contains("prehandle initialization failure")
    );
    assert!(!op.stream_clone().context().graph_completion_unknown());
    assert!(!ledger().owners.contains_key(&(
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::GemmEx
    )));
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
    let mut provider = op.prepare_bf16_gemmex(p).unwrap();
    exercise(&op, &mut provider);
    drop(provider);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
