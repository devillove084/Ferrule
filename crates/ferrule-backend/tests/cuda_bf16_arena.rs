//! Allocation-only BF16 arena contract; GPU tests require explicit authorization.
use ferrule_backend::cuda_allocation_grant_bytes;
#[path = "../src/cuda/operators/bf16_arena.rs"]
mod bf16_arena;

#[test]
fn arena_checks_word_bytes_rounded_grant_and_budget() {
    for (words, grant) in [
        (0, 0),
        (1, 256),
        (129, 512),
        (524_289, 2 << 20),
        (3 << 29, 3 << 30),
    ] {
        assert_eq!(bf16_arena::checked_grant(words, grant).unwrap(), grant);
        if grant != 0 {
            assert!(bf16_arena::checked_grant(words, grant - 1).is_err());
        }
    }
    assert!(
        bf16_arena::checked_grant(1, 2).is_err(),
        "budget is rounded grant, not logical bytes"
    );
    assert!(bf16_arena::checked_grant(usize::MAX / 2 + 1, usize::MAX).is_err());
    assert!(
        bf16_arena::checked_grant(usize::MAX / 2, usize::MAX).is_err(),
        "grant rounding overflow"
    );
}

#[test]
fn arena_has_no_initialization_path_before_return_or_on_error() {
    // Structural guard: counters alone cannot detect an uncounted driver memset.
    let owner = include_str!("../src/cuda/operator_owner.rs");
    let body = owner
        .split_once("pub fn allocate_bf16_arena_uninitialized(")
        .unwrap()
        .1
        .split_once("pub fn zero_bf16_buffer(")
        .unwrap()
        .0;
    let compact: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    for token in [
        "check_capture_safe(",
        "check_graph_ready()",
        "bf16_arena::checked_grant(len,grant_budget_bytes)?",
        "record_device_allocation(len,",
        "DeviceBuffer::<u16>::uninitialized_dedicated(&self.stream,len)",
        "CudaTypedBuffer::from_device_buffer(",
    ] {
        assert!(
            compact.contains(token),
            "missing allocation-only boundary: {token}"
        );
    }
    assert!(
        compact.find("checked_grant(").unwrap()
            < compact.find("record_device_allocation(").unwrap()
    );
    let runtime = include_str!("../src/cuda/runtime.rs");
    let dedicated = runtime
        .split_once("pub(crate) fn uninitialized_dedicated(")
        .unwrap()
        .1
        .split_once("///")
        .unwrap()
        .0;
    assert!(dedicated.contains("allocation_bytes::<T>(len)?"));
    assert!(dedicated.contains(".allocate_dedicated(bytes, std::mem::align_of::<T>())?"));
    for source in [body, dedicated] {
        for forbidden in [
            "unsafe",
            "zeroed(",
            "zeroed_device_buffer",
            "zero_bf16_buffer(",
            "Memset",
            "memset",
            "copy_from",
            "synchronize(",
            "launch(",
        ] {
            assert!(
                !source.contains(forbidden),
                "allocation-only path contains {forbidden}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use ferrule_backend::cuda::operators::linear::{Bf16GemmExPlan, CudaOperators};
    use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
    use std::sync::Arc;

    fn no_initialization(op: &CudaOperators) {
        let c = op.counters();
        assert_eq!(c.kernel_launches, 0);
        assert_eq!(c.host_to_device_copies, 0);
        assert_eq!(c.device_to_host_copies, 0);
        assert_eq!(c.stream_wide_syncs, 0);
    }

    #[test]
    #[ignore = "sole GPU owner required; dedicated allocation-only root before decode, no GPU init at allocation"]
    fn arena_allocates_then_retains_root_before_full_decode() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let layout = NumericFp8Layout {
            n: 128,
            k: 128,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let plan = Bf16GemmExPlan::new(layout, 1, 64 << 20, 16 << 20).unwrap();
        let mut decoder = op.prepare_bf16_gemmex(plan).unwrap();
        let weight = op
            .upload_numeric_fp8_linear(layout, &vec![0x38; 128 * 128], &[0x80, 0x3f])
            .unwrap();
        let words = 2 * 128 * 128;
        let grant = cuda_allocation_grant_bytes(words * 2).unwrap();
        let before = op.allocator_metrics();
        op.reset_counters();
        // This Option models retained owner state, installed before initialization.
        let mut retained = Some(op.allocate_bf16_arena_uninitialized(words, grant).unwrap());
        let root = retained.as_ref().unwrap();
        let mut fixed = root.slice(0, words / 2).unwrap();
        let mut slot = root.slice(words / 2, words / 2).unwrap();
        assert!(Arc::ptr_eq(
            root.as_device_buffer().context(),
            fixed.as_device_buffer().context()
        ));
        no_initialization(&op);
        let c = op.counters();
        assert_eq!(c.device_allocation_attempts, 1);
        assert_eq!(c.device_allocations, 1);
        assert_eq!(c.device_allocation_bytes, (words * 2) as u64);
        let after = op.allocator_metrics();
        assert_eq!(after.driver_allocations - before.driver_allocations, 1);
        assert_eq!(after.reserved_bytes - before.reserved_bytes, grant);
        assert_eq!(after.live_granted_bytes - before.live_granted_bytes, grant);
        assert_eq!(
            after.live_requested_bytes - before.live_requested_bytes,
            words * 2
        );
        // Never inspect uninitialized data, including a failure path.
        op.reset_counters();
        let mut short = root.slice(0, words / 2 - 1).unwrap();
        assert!(
            decoder
                .decode_numeric_bf16_into(&op, &weight, &mut short)
                .is_err()
        );
        no_initialization(&op);
        assert!(retained.is_some());
        decoder
            .decode_numeric_bf16_into(&op, &weight, &mut fixed)
            .unwrap();
        decoder
            .decode_numeric_bf16_into(&op, &weight, &mut slot)
            .unwrap();
        // Both exact ranges have writes ordered before this first consumer.
        assert_eq!(
            op.download_bf16_buffer(retained.as_ref().unwrap()).unwrap(),
            vec![1.0; words]
        );
        drop(retained.take());
        assert_eq!(
            op.download_bf16_buffer(&slot).unwrap(),
            vec![1.0; words / 2]
        );
    }

    #[test]
    #[ignore = "sole GPU owner required; arena budget/capture/allocation errors submit no initialization"]
    fn arena_errors_never_submit_initialization() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let before = op.allocator_metrics();
        op.reset_counters();
        assert!(
            op.allocate_bf16_arena_uninitialized(usize::MAX, usize::MAX)
                .is_err()
        );
        assert!(op.allocate_bf16_arena_uninitialized(128, 255).is_err());
        op.failpoints().arm_allocation();
        op.enable_capture_safe();
        assert!(op.allocate_bf16_arena_uninitialized(128, 256).is_err());
        op.disable_capture_safe();
        // Capture preflight must not consume the allocation failpoint.
        assert!(op.allocate_bf16_arena_uninitialized(128, 256).is_err());
        no_initialization(&op);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(
            op.allocator_metrics().driver_allocations,
            before.driver_allocations
        );
        assert_eq!(
            op.allocator_metrics().live_requested_bytes,
            before.live_requested_bytes
        );
        let graph = op
            .capture_decode_graph(|| {
                assert!(op.allocate_bf16_arena_uninitialized(128, 256).is_err());
                Ok(())
            })
            .unwrap();
        drop(graph);
        op.reset_counters();
        let empty = op.allocate_bf16_arena_uninitialized(0, 0).unwrap();
        assert!(empty.is_empty());
        no_initialization(&op);
        op.shutdown_device_allocator();
        assert!(op.allocate_bf16_arena_uninitialized(128, 256).is_err());
        no_initialization(&op);
    }
}
