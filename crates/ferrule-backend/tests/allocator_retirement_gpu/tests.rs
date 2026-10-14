use crate::driver::*;
use crate::runtime::{CudaContext, CudaStream, DeviceBuffer, GraphFault};
use std::sync::Arc;
use std::sync::atomic::Ordering::SeqCst;

fn context() -> Arc<CudaContext> {
    assert_eq!(CudaContext::device_count().expect("real CUDA required"), 1);
    let ctx = CudaContext::new(0).expect("required assigned CUDA device 0");
    println!("DEVICE {}", ctx.device_name().unwrap());
    ctx
}
fn buffer(stream: &CudaStream, bytes: usize) -> DeviceBuffer<u8> {
    DeviceBuffer::zeroed(stream, bytes).unwrap()
}
fn uninitialized(stream: &CudaStream, bytes: usize) -> DeviceBuffer<u8> {
    unsafe { DeviceBuffer::uninitialized_async(stream, bytes) }.unwrap()
}
fn metrics(ctx: &CudaContext, phase: &str) -> crate::allocator::CudaAllocatorMetrics {
    let m = ctx.allocator_metrics();
    println!("METRICS {phase} {m:?}");
    counts(phase);
    m
}
fn drain(ctx: Arc<CudaContext>, streams: Vec<Arc<CudaStream>>) {
    for s in &streams {
        s.synchronize().unwrap();
    }
    let m = metrics(&ctx, "completed");
    assert_eq!(m.pending_retirement_bytes, 0);
    assert_eq!(m.live_requested_bytes, 0);
    ctx.trim_allocator().unwrap();
    let m = metrics(&ctx, "final_drain");
    assert_eq!(m.reserved_bytes, 0);
    assert_eq!(m.driver_allocations, m.driver_frees);
    drop(streams);
    drop(ctx);
    assert_eq!(RELEASES.load(SeqCst), 1);
    counts("context_dropped");
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_n8_two_streams_batch_and_reuse() {
    let ctx = context();
    let a = ctx.new_stream().unwrap();
    let b = ctx.new_stream().unwrap();
    let buffers: Vec<_> = (0..8)
        .map(|i| buffer(if i % 2 == 0 { &a } else { &b }, 256))
        .collect();
    let ptrs: Vec<_> = buffers.iter().map(DeviceBuffer::cu_deviceptr).collect();
    a.synchronize().unwrap();
    b.synchronize().unwrap();
    let gate = block_stream(&a);
    let syncs = STREAM_SYNCS.load(SeqCst);
    drop(buffers);
    assert_eq!(
        CREATES.load(SeqCst),
        0,
        "last Drop must buffer without per-block events"
    );
    assert_eq!(RECORDS.load(SeqCst), 0);
    counts("n8_last_drop_before_seal");
    let m = metrics(&ctx, "n8_sealed_pending");
    assert_eq!(m.retirement_batches, 1);
    assert_eq!(m.pending_retirement_blocks, 8);
    assert_eq!(m.pending_retirement_bytes, 8 * 256);
    assert_eq!(m.retirement_created_fences, 2);
    assert_eq!(CREATES.load(SeqCst), 2);
    assert_eq!(RECORDS.load(SeqCst), 2);
    let other = uninitialized(&b, 256);
    assert!(
        !ptrs.contains(&other.cu_deviceptr()),
        "pending extent reused"
    );
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    assert_eq!(EVENT_SYNCS.load(SeqCst), 0);
    assert_eq!(STREAM_SYNCS.load(SeqCst), syncs);
    gate.release();
    a.synchronize().unwrap();
    assert!(!gate.timed_out.load(SeqCst));
    assert_eq!(metrics(&ctx, "n8_after_poll").pending_retirement_bytes, 0);
    let reused = uninitialized(&b, 256);
    assert_eq!(reused.cu_deviceptr(), ptrs[0]);
    assert_eq!(metrics(&ctx, "n8_reused").driver_allocations, 1);
    drop(other);
    drop(reused);
    drain(ctx, vec![a, b]);
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_pending_byte_pressure_allows_safe_extent_not_growth() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let seed = buffer(&s, 256);
    let blocks: Vec<_> = (0..4).map(|_| buffer(&s, 64 << 20)).collect();
    s.synchronize().unwrap();
    let gate = block_stream(&s);
    drop(blocks);
    let m = metrics(&ctx, "byte_pressure");
    assert_eq!(m.pending_retirement_bytes, 256 << 20);
    assert_eq!(m.pending_retirement_batches, 4);
    assert_eq!(m.pending_unfenced_bytes, 0);
    let safe = uninitialized(&s, 256);
    assert_eq!(
        metrics(&ctx, "safe_extent_at_byte_pressure").driver_allocations,
        m.driver_allocations
    );
    let err = unsafe { DeviceBuffer::<u8>::uninitialized_async(&s, 80 << 20) }.unwrap_err();
    assert!(err.to_string().contains("growth deferred"), "{err}");
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    assert_eq!(EVENT_SYNCS.load(SeqCst), 0);
    gate.release();
    s.synchronize().unwrap();
    assert!(!gate.timed_out.load(SeqCst));
    drop(seed);
    drop(safe);
    drain(ctx, vec![s]);
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_pending_batch_pressure_allows_safe_extent_not_growth() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let blocks: Vec<_> = (0..(256 * 64)).map(|_| uninitialized(&s, 256)).collect();
    let gate = block_stream(&s);
    drop(blocks);
    let m = metrics(&ctx, "batch_pressure");
    assert_eq!(m.pending_retirement_batches, 256);
    assert_eq!(m.pending_retirement_blocks, 256 * 64);
    assert_eq!(m.retirement_created_fences, 256);
    assert_eq!(m.pending_unfenced_bytes, 0);
    let safe = uninitialized(&s, 256);
    assert_eq!(
        metrics(&ctx, "safe_extent_at_batch_pressure").driver_allocations,
        1
    );
    let err = unsafe { DeviceBuffer::<u8>::uninitialized_async(&s, 80 << 20) }.unwrap_err();
    assert!(err.to_string().contains("growth deferred"), "{err}");
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    assert_eq!(EVENT_SYNCS.load(SeqCst), 0);
    gate.release();
    s.synchronize().unwrap();
    assert!(!gate.timed_out.load(SeqCst));
    drop(safe);
    drain(ctx, vec![s]);
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_weak_gone_pending_sync_new_registration_is_not_quiescent() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let block = buffer(&s, 256);
    let original = block.cu_deviceptr();
    let busy = block_stream(&s);
    drop(block);
    let weak = Arc::downgrade(&s);
    let gate = Arc::new(Gate::default());
    *STREAM_GATE.lock().unwrap() = Some(gate.clone());
    let teardown = std::thread::spawn(move || drop(s));
    gate.entered();
    assert!(weak.upgrade().is_none());
    let other = ctx.new_stream().unwrap();
    for _ in 0..3 {
        let m = metrics(&ctx, "weak_gone_sync_pending");
        assert_eq!(m.pending_retirement_bytes, 256);
        assert_eq!(m.pending_unfenced_bytes, 256);
    }
    let live = uninitialized(&other, 256);
    assert_ne!(live.cu_deviceptr(), original);
    assert_eq!(CREATES.load(SeqCst), 0);
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    // Only release the destructor gate first: real cuStreamSynchronize must still wait for GPU work.
    gate.release();
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!teardown.is_finished());
    assert_eq!(
        metrics(&ctx, "real_stream_sync_blocked").pending_retirement_bytes,
        256
    );
    busy.release();
    teardown.join().unwrap();
    assert!(!busy.timed_out.load(SeqCst));
    assert!(!gate.timed_out.load(SeqCst));
    other.synchronize().unwrap();
    assert_eq!(
        metrics(&ctx, "ticket_quiescent").pending_retirement_bytes,
        0
    );
    assert_eq!(CREATES.load(SeqCst), 1);
    let reused = uninitialized(&other, 256);
    assert_eq!(reused.cu_deviceptr(), original);
    drop(live);
    drop(reused);
    drain(ctx, vec![other]);
}
fn capture_proof(complete: bool) {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let capture = ctx.new_stream().unwrap();
    let block = buffer(&s, 256);
    s.synchronize().unwrap();
    let busy = (!complete).then(|| block_stream(&s));
    drop(block);
    let gate = Arc::new(Gate::default());
    *QUERY_GATE.lock().unwrap() = Some(gate.clone());
    QUERY_GATE_AT.store(if complete { 2 } else { 1 }, SeqCst);
    let worker = ctx.clone();
    let poll = std::thread::spawn(move || worker.allocator_metrics());
    gate.entered();
    capture.begin_capture().unwrap();
    gate.release();
    let m = poll.join().unwrap();
    println!("METRICS capture_publication_barrier complete={complete} {m:?}");
    assert_eq!(m.pending_retirement_bytes, 256);
    assert_eq!(
        m.pending_unfenced_bytes, 0,
        "Recorded/Complete proof was discarded"
    );
    assert_eq!(m.retirement_created_fences, 2);
    let graph = capture.end_capture().unwrap();
    drop(graph);
    if let Some(busy) = busy {
        busy.release();
        s.synchronize().unwrap();
        assert!(!busy.timed_out.load(SeqCst));
    }
    let m = metrics(&ctx, "capture_original_proof_reused");
    assert_eq!(m.pending_retirement_bytes, 0);
    assert_eq!(m.retirement_created_fences, 2);
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    assert_eq!(EVENT_SYNCS.load(SeqCst), 0);
    assert!(!gate.timed_out.load(SeqCst));
    drain(ctx, vec![s, capture]);
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_capture_preserves_recorded_proof() {
    capture_proof(false);
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_capture_preserves_complete_proof() {
    capture_proof(true);
}
#[test]
#[ignore = "real GPU; intentional quarantine; fresh process only"]
fn gpu_retirement_query_and_fallback_unknown_retains_whole_batch() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let blocks: Vec<_> = (0..8).map(|_| buffer(&s, 256)).collect();
    s.synchronize().unwrap();
    drop(blocks);
    FAIL_QUERY.store(true, SeqCst);
    FAIL_EVENT_SYNC.store(true, SeqCst);
    let m = metrics(&ctx, "query_fallback_unknown");
    assert!(ctx.graph_completion_unknown());
    assert_eq!(m.pending_retirement_blocks, 8);
    assert_eq!(m.pending_retirement_bytes, 2048);
    assert_eq!(m.retirement_batches, 1);
    assert_eq!(m.retirement_created_fences, 1);
    assert_eq!(EVENT_SYNCS.load(SeqCst), 1);
    let again = metrics(&ctx, "sticky_unknown");
    assert_eq!(again.pending_retirement_bytes, 2048);
    assert_eq!(again.reserved_bytes, m.reserved_bytes);
    drop(s);
    drop(ctx);
    assert_eq!(FREES.load(SeqCst), 0);
    assert_eq!(RELEASES.load(SeqCst), 0);
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 0);
    counts("unknown_process_exit_required");
}
#[test]
#[ignore = "real GPU; fresh supervised exact process"]
fn gpu_shutdown_known_drains_and_releases() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    drop(buffer(&s, 256));
    s.synchronize().unwrap();
    ctx.shutdown_allocator();
    let m = metrics(&ctx, "shutdown_known");
    assert!(m.shutdown_quiescent);
    assert_eq!(m.pending_retirement_bytes, 0);
    assert_eq!(m.reserved_bytes, 0);
    drop(s);
    drop(ctx);
    assert_eq!(FREES.load(SeqCst), 1);
    assert_eq!(RELEASES.load(SeqCst), 1);
    counts("shutdown_known_final");
}
#[test]
#[ignore = "real GPU; intentional quarantine; fresh process only"]
fn gpu_shutdown_fault_retains_primary_and_segments() {
    let ctx = context();
    let s = ctx.new_stream().unwrap();
    let block = buffer(&s, 256);
    s.synchronize().unwrap();
    drop(block);
    drop(s);
    ctx.arm_graph_fault(GraphFault::AllocatorSynchronize);
    drop(ctx);
    assert_eq!(CONTEXT_SYNCS.load(SeqCst), 1);
    assert_eq!(FREES.load(SeqCst), 0);
    assert_eq!(RELEASES.load(SeqCst), 0);
    counts("shutdown_unknown_process_exit_required");
}
