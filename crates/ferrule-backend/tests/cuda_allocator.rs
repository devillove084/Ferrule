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
