//! Run only in the actual library test binary, one exact test per fresh process.
//! Agent322 owns the GPU gate. Faults are synthetic result injection after real
//! CUDA operations, not GPU-damaging illegal-memory-access kernels.
use super::*;
use std::sync::atomic::Ordering::SeqCst;

fn retired_context() -> (
    Arc<CudaContext>,
    Arc<CudaDeviceAllocator>,
    Arc<GraphTestHooks>,
) {
    let context = CudaContext::new(0).expect("agent322 gate requires CUDA device 0");
    let stream = context.default_stream();
    let buffer = DeviceBuffer::<u8>::zeroed(&stream, 256).unwrap();
    stream.synchronize().unwrap();
    drop(buffer);
    drop(stream);
    let allocator = context.allocator.clone();
    let hooks = context.graph_test_hooks.clone();
    (context, allocator, hooks)
}

fn assert_quarantined(allocator: &CudaDeviceAllocator, hooks: &GraphTestHooks) {
    let metrics = allocator.metrics();
    assert!(metrics.shutdown_retained);
    assert!(!metrics.shutdown_quiescent);
    assert_eq!(metrics.pending_retirement_blocks, 1);
    assert_eq!(metrics.pending_retirement_bytes, 256);
    assert!(metrics.reserved_bytes >= 256);
    assert_eq!(metrics.driver_frees, 0);
    assert_eq!(
        hooks.calls[GraphFault::PrimaryRelease as usize].load(SeqCst),
        0
    );
}

#[test]
#[ignore = "agent322 sole GPU owner; fresh subprocess; real CUDA positive teardown"]
fn gpu_context_drop_quiescent_releases_primary_once() {
    let (context, allocator, hooks) = retired_context();
    drop(context);
    let metrics = allocator.metrics();
    assert!(metrics.shutdown_quiescent);
    assert!(!metrics.shutdown_retained);
    assert_eq!(metrics.pending_retirement_bytes, 0);
    assert_eq!(metrics.reserved_bytes, 0);
    assert_eq!(metrics.driver_frees, 1);
    assert_eq!(
        hooks.calls[GraphFault::PrimaryRelease as usize].load(SeqCst),
        1
    );
}

#[test]
#[ignore = "agent322 sole GPU owner; injected status after real context sync; intentional quarantine"]
fn gpu_context_drop_shutdown_sync_fault_retains_primary_and_segments() {
    let (context, allocator, hooks) = retired_context();
    context.arm_graph_fault(GraphFault::AllocatorSynchronize);
    drop(context);
    assert_quarantined(&allocator, &hooks);
    assert_eq!(
        hooks.calls[GraphFault::AllocatorSynchronize as usize].load(SeqCst),
        1
    );
}

#[test]
#[ignore = "agent322 sole GPU owner; Unknown before final Drop; intentional quarantine"]
fn gpu_context_drop_unknown_before_does_not_attempt_release() {
    let (context, allocator, hooks) = retired_context();
    context.mark_graph_unknown();
    drop(context);
    assert_quarantined(&allocator, &hooks);
    assert_eq!(
        hooks.calls[GraphFault::AllocatorSynchronize as usize].load(SeqCst),
        0
    );
}

#[test]
#[ignore = "agent322 sole GPU owner; real event query with injected failure; intentional quarantine"]
fn gpu_context_drop_query_fault_cannot_be_washed_by_later_event() {
    let context = CudaContext::new(0).expect("agent322 gate requires CUDA device 0");
    let stream = context.default_stream();
    let buffer = DeviceBuffer::<u8>::zeroed(&stream, 256).unwrap();
    let event = context.new_event(false).unwrap();
    event.record(&stream).unwrap();
    event.synchronize().unwrap();
    drop(buffer);
    drop(stream);
    context.arm_graph_fault(GraphFault::EventQuery);
    assert!(event.query().is_err());
    assert!(context.graph_completion_unknown());
    assert!(
        event.query().is_err(),
        "consuming the one-shot fault must not clear Unknown"
    );
    drop(event);
    let allocator = context.allocator.clone();
    let hooks = context.graph_test_hooks.clone();
    drop(context);
    assert_quarantined(&allocator, &hooks);
}
