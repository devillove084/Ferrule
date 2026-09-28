#![cfg(feature = "cuda")]

use ferrule_backend::cuda::operators::linear::CudaOperators;

#[test]
fn public_allocator_diagnostics_track_reuse_and_trim() {
    let Ok(operators) = CudaOperators::new() else {
        return;
    };

    let first = operators.zero_f32_buffer(1024).unwrap();
    let first_ptr = first.as_device_buffer().cu_deviceptr();
    let live = operators.allocator_metrics();
    assert_eq!(live.live_requested_bytes, 1024 * size_of::<f32>());
    assert!(live.reserved_bytes >= live.live_granted_bytes);
    assert_eq!(live.driver_allocations, 1);
    drop(first);

    operators.sync_stream().unwrap();
    let second = operators.zero_f32_buffer(1024).unwrap();
    assert_eq!(second.as_device_buffer().cu_deviceptr(), first_ptr);
    let reused = operators.allocator_metrics();
    assert!(reused.reuse_allocations >= 1);
    assert_eq!(reused.driver_allocations, 1);
    drop(second);

    operators.sync_stream().unwrap();
    let released = operators.trim_device_allocator().unwrap();
    let trimmed = operators.allocator_metrics();
    assert!(released > 0);
    assert_eq!(trimmed.reserved_bytes, 0);
    assert_eq!(trimmed.live_requested_bytes, 0);
    assert_eq!(trimmed.live_granted_bytes, 0);
    assert_eq!(trimmed.driver_frees, 1);
}

#[test]
fn shutdown_rejects_new_ordinary_device_buffers() {
    let Ok(operators) = CudaOperators::new() else {
        return;
    };
    operators.shutdown_device_allocator();
    let error = match operators.zero_i32_buffer(1) {
        Ok(_) => panic!("allocator growth after shutdown must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("shut down"));
}

#[test]
#[ignore = "requires actual CUDA GPU; PR01 zero-side-effect guard regression"]
fn pr01_capture_guard_precedes_allocator_and_preserves_failpoint() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    let seed = op.zero_f32_buffer(16).unwrap();
    drop(seed);
    op.sync_stream().unwrap();
    let before = op.allocator_metrics();
    op.reset_counters();
    op.failpoints().arm_allocation();
    op.enable_capture_safe();
    assert!(op.zero_f32_buffer(16).is_err());
    let after = op.allocator_metrics();
    assert_eq!(after.allocation_requests, before.allocation_requests);
    assert_eq!(after.reuse_allocations, before.reuse_allocations);
    assert_eq!(after.driver_allocations, before.driver_allocations);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    op.disable_capture_safe();
    assert!(op.zero_f32_buffer(16).is_err(), "guard consumed failpoint");
    assert_eq!(
        op.allocator_metrics().allocation_requests,
        before.allocation_requests
    );
    op.zero_f32_buffer(16).unwrap();
    assert_eq!(op.counters().device_allocation_attempts, 1);
}

#[test]
#[ignore = "requires actual CUDA GPU; PR01 managed, i32 D2H and pinned entries"]
fn pr01_omitted_public_entries_reject_capture_assertion() {
    use ferrule_backend::cuda::context::CudaArtifactLinearShape;
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    let buffer = op.zero_i32_buffer(1).unwrap();
    let shape = CudaArtifactLinearShape::F32 {
        out_features: 1,
        in_features: 1,
    };
    op.reset_counters();
    op.enable_capture_safe();
    assert!(op.download_i32_buffer(&buffer).is_err());
    assert!(op.allocate_artifact_linear_managed(shape, 4, 0).is_err());
    assert!(
        op.upload_artifact_linear_managed(shape, &[0; 4], &[])
            .is_err()
    );
    assert!(op.i32_host_mirror(&[1]).is_err());
    assert!(op.pin_u8_host_buffer(&[1]).is_err());
    assert!(op.dsv4_router_token_ids(&[0], 1).is_err());
    assert!(op.sync_stream().is_err());
    assert!(op.sync_upload_stream().is_err());
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().device_to_host_copies, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
}

#[test]
#[ignore = "actual CUDA GPU; ordinary sync failpoint precedes copy and submitted cleanup bypasses it"]
fn pr01_sync_failpoint_precedes_copy_and_guard_does_not_consume_it() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    let mut buffer = op.upload_i32_buffer(&[7]).unwrap();
    op.reset_counters();
    op.failpoints().arm_stream_sync();
    op.enable_capture_safe();
    assert!(op.overwrite_i32_buffer(&[11], &mut buffer).is_err());
    op.disable_capture_safe();
    assert!(op.overwrite_i32_buffer(&[11], &mut buffer).is_err());
    assert_eq!(op.stream_wide_sync_attempts(), 0);
    assert_eq!(op.counters().host_to_device_copies, 0);
    assert_eq!(op.download_i32_buffer(&buffer).unwrap(), [7]);
    op.overwrite_i32_buffer(&[11], &mut buffer).unwrap();
    assert_eq!(op.stream_wide_sync_attempts(), 1);
    assert_eq!(op.counters().stream_wide_syncs, 1);
    assert_eq!(op.download_i32_buffer(&buffer).unwrap(), [11]);
    op.failpoints().arm_stream_sync();
    op.failpoints().arm_copy_event();
    assert!(op.i32_host_mirror(&[17]).is_err());
    assert_eq!(
        op.counters().stream_wide_syncs,
        2,
        "submitted drain was suppressed"
    );
    assert!(
        op.sync_stream().is_err(),
        "submitted drain consumed ordinary failpoint"
    );
    op.sync_stream().unwrap();
}
