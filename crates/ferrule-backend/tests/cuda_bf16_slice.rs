//! Host-only checked-range/custody contracts plus an opt-in GPU lifetime microtest.
#[path = "../src/cuda/operators/typed_buffer_range.rs"]
mod typed_buffer_range;
use typed_buffer_range::validate_slice;

#[test]
fn bf16_slice_element_bounds_and_empty_policy() {
    for offset in 0..=8 {
        for len in 0..=8 - offset {
            validate_slice::<u16>(0x1000, 8, offset, len).unwrap();
        }
    }
    assert!(validate_slice::<u16>(0x1000, 8, 9, 0).is_err());
    assert!(validate_slice::<u16>(0x1000, 8, 8, 1).is_err());
    assert!(validate_slice::<u16>(0x1000, 8, 3, 6).is_err());
    validate_slice::<u16>(0, 0, 0, 0).unwrap();
    assert!(validate_slice::<u16>(0, 0, 1, 0).is_err());
    // A nested view cannot reach bytes that exist only in its parent's tail.
    assert!(validate_slice::<u16>(0x1004, 2, 1, 2).is_err());
}

#[test]
fn bf16_slice_rejects_u16_element_byte_and_address_overflow() {
    assert!(validate_slice::<u16>(0, usize::MAX, usize::MAX, 1).is_err());
    let too_many_words = usize::MAX / size_of::<u16>() + 1;
    assert!(validate_slice::<u16>(0, usize::MAX, too_many_words, 0).is_err());
    assert!(validate_slice::<u16>(0, usize::MAX, 0, too_many_words).is_err());
    assert!(validate_slice::<u16>(u64::MAX - 1, 2, 1, 0).is_err());
    assert!(validate_slice::<u16>(u64::MAX - 1, 2, 0, 1).is_err());
    validate_slice::<u16>(u64::MAX - 2, 1, 0, 1).unwrap();
    validate_slice::<u16>(u64::MAX - 2, 1, 1, 0).unwrap();
}

#[test]
fn bf16_slice_custody_delegates_to_existing_shared_root() {
    // Structural CPU guard, not a simulated CUDA allocation/lifetime test.
    let owner = include_str!("../src/cuda/operator_owner.rs");
    let body = owner
        .split_once("impl CudaTypedBuffer<u16> {")
        .unwrap()
        .1
        .split_once("impl CudaTypedBuffer<f32> {")
        .unwrap()
        .0;
    assert!(body.contains("typed_buffer_range::validate_slice::<u16>"));
    let compact: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(compact.contains("Self::from_device_buffer(cu(self.buffer.slice(offset,len))?)"));
    assert!(!body.contains("unsafe"));
    assert!(!body.contains("synchronize("));
    let runtime = include_str!("../src/cuda/runtime.rs");
    let slice = runtime
        .split_once("pub fn slice(&self, offset: usize, len: usize)")
        .unwrap()
        .1
        .split_once("pub fn cu_deviceptr")
        .unwrap()
        .0;
    assert!(slice.contains("allocation: Arc::clone(&self.allocation)"));
    assert!(runtime.contains("self.allocation.context()"));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires authorized sole GPU owner; BF16 view offset/counters/root-drop lifetime"]
fn bf16_slice_gpu_shared_pointer_no_work_parent_drop() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::sync::Arc;

    let op = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let root = op
        .upload_bf16_words(&[0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0])
        .unwrap();
    let root_ptr = root.as_device_buffer().cu_deviceptr();
    let before = op.allocator_metrics();
    op.reset_counters();
    let middle = root.slice(1, 4).unwrap();
    let child = middle.slice(1, 2).unwrap();
    let empty = root.slice(root.len(), 0).unwrap();
    let overlap = root.slice(2, 2).unwrap();
    assert_eq!(middle.as_device_buffer().cu_deviceptr(), root_ptr + 2);
    assert_eq!(child.as_device_buffer().cu_deviceptr(), root_ptr + 4);
    assert_eq!(
        overlap.as_device_buffer().cu_deviceptr(),
        child.as_device_buffer().cu_deviceptr()
    );
    assert_eq!(empty.as_device_buffer().cu_deviceptr(), root_ptr + 12);
    assert!(empty.is_empty());
    assert_eq!(child.len(), 2);
    assert!(Arc::ptr_eq(
        root.as_device_buffer().context(),
        child.as_device_buffer().context()
    ));
    assert!(root.slice(6, 1).is_err());
    assert!(root.slice(usize::MAX, 1).is_err());
    assert!(root.slice(0, usize::MAX / 2 + 1).is_err());
    assert!(middle.slice(3, 2).is_err());
    drop((root, middle, empty, overlap));
    let after = op.allocator_metrics();
    assert_eq!(after.live_requested_bytes, before.live_requested_bytes);
    assert_eq!(after.allocation_requests, before.allocation_requests);
    assert_eq!(after.driver_allocations, before.driver_allocations);
    let c = op.counters();
    assert_eq!(c.device_allocation_attempts, 0);
    assert_eq!(c.kernel_launches, 0);
    assert_eq!(c.host_to_device_copies, 0);
    assert_eq!(c.device_to_host_copies, 0);
    assert_eq!(c.stream_wide_syncs, 0);
    assert_eq!(op.download_bf16_buffer(&child).unwrap(), [3.0, 4.0]);
    drop(child);
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
