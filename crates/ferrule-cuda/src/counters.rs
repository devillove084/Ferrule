use std::cell::Cell;

/// Deterministic failure injection for E2 allocation/arena/resource lifecycle.
///
/// When a failpoint is armed, the next matching operation returns an error
/// instead of succeeding. Each failpoint fires at most once before disarming,
/// so tests can arm a point, trigger one operation, and verify the error path
/// without affecting subsequent calls.
#[derive(Debug, Default)]
pub struct CudaFailpoints {
    /// Fail the next device allocation attempt.
    next_allocation: Cell<bool>,
    /// Fail the next arena acquire or grow attempt.
    next_arena_acquire: Cell<bool>,
    /// Fail the next D2D preservation copy during arena growth.
    next_d2d_preserve: Cell<bool>,
    /// Fail the next prepared-expert load/upload.
    next_expert_upload: Cell<bool>,
    /// Fail after the next main-compressor semantic transition, before publication.
    next_main_compressor_transition: Cell<bool>,
    /// Fail after the next indexer-compressor semantic transition, before publication.
    next_indexer_compressor_transition: Cell<bool>,
    /// Fail the next prepared-resource installation.
    next_resource_install: Cell<bool>,
}

impl CudaFailpoints {
    pub fn arm_allocation(&self) {
        self.next_allocation.set(true);
    }

    pub fn arm_arena_acquire(&self) {
        self.next_arena_acquire.set(true);
    }

    pub fn arm_d2d_preserve(&self) {
        self.next_d2d_preserve.set(true);
    }

    pub fn arm_expert_upload(&self) {
        self.next_expert_upload.set(true);
    }

    pub fn arm_main_compressor_transition(&self) {
        self.next_main_compressor_transition.set(true);
    }

    pub fn arm_indexer_compressor_transition(&self) {
        self.next_indexer_compressor_transition.set(true);
    }

    pub fn arm_resource_install(&self) {
        self.next_resource_install.set(true);
    }

    /// Returns `true` (and disarms) if the allocation failpoint was armed.
    pub fn check_allocation(&self) -> bool {
        let armed = self.next_allocation.get();
        if armed {
            self.next_allocation.set(false);
        }
        armed
    }

    /// Returns `true` (and disarms) if the arena acquire failpoint was armed.
    pub fn check_arena_acquire(&self) -> bool {
        let armed = self.next_arena_acquire.get();
        if armed {
            self.next_arena_acquire.set(false);
        }
        armed
    }

    /// Returns `true` (and disarms) if the D2D preserve failpoint was armed.
    pub fn check_d2d_preserve(&self) -> bool {
        let armed = self.next_d2d_preserve.get();
        if armed {
            self.next_d2d_preserve.set(false);
        }
        armed
    }

    /// Returns `true` (and disarms) if the expert upload failpoint was armed.
    pub fn check_expert_upload(&self) -> bool {
        let armed = self.next_expert_upload.get();
        if armed {
            self.next_expert_upload.set(false);
        }
        armed
    }

    pub fn check_main_compressor_transition(&self) -> bool {
        let armed = self.next_main_compressor_transition.get();
        if armed {
            self.next_main_compressor_transition.set(false);
        }
        armed
    }

    pub fn check_indexer_compressor_transition(&self) -> bool {
        let armed = self.next_indexer_compressor_transition.get();
        if armed {
            self.next_indexer_compressor_transition.set(false);
        }
        armed
    }

    /// Returns `true` (and disarms) if the resource install failpoint was armed.
    pub fn check_resource_install(&self) -> bool {
        let armed = self.next_resource_install.get();
        if armed {
            self.next_resource_install.set(false);
        }
        armed
    }

    pub fn disarm_all(&self) {
        self.next_allocation.set(false);
        self.next_arena_acquire.set(false);
        self.next_d2d_preserve.set(false);
        self.next_expert_upload.set(false);
        self.next_main_compressor_transition.set(false);
        self.next_indexer_compressor_transition.set(false);
        self.next_resource_install.set(false);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaOpCounters {
    pub kernel_launches: u64,
    pub host_to_device_copies: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_copies: u64,
    pub device_to_host_bytes: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
    /// Device-buffer allocation requests observed by `CudaArtifactOperatorContext`.
    pub device_allocation_attempts: u64,
    /// Successfully created device buffers. Free completion is intentionally not
    /// inferred because cuda-oxide owns buffer RAII destruction.
    pub device_allocations: u64,
    pub device_allocation_failures: u64,
    pub device_allocation_bytes: u64,
    /// Successful whole-stream synchronizations requested by Ferrule.
    pub stream_wide_syncs: u64,
    pub stream_wide_sync_failures: u64,
    pub moe_calls: u64,
    pub moe_tc_calls: u64,
    pub moe_scalar_calls: u64,
    pub moe_reduce_calls: u64,
    pub moe_total_us: u64,
    pub moe_pointer_upload_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_hidden_pack_us: u64,
    pub moe_down_us: u64,
    /// Arena bucket cache hits: an existing bucket matched the request.
    pub arena_hits: u64,
    /// Arena bucket cache misses: no existing bucket matched, requiring creation.
    pub arena_misses: u64,
    /// Arena bucket growth events: an existing bucket was replaced with a larger one.
    pub arena_grows: u64,
    /// Arena bucket reuse events: a previously created bucket was reused.
    pub arena_reuses: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaMoeExecutionPath {
    TensorCore,
    Scalar,
    Reduce,
}

#[derive(Default)]
pub(crate) struct CudaOpCounterCells {
    kernel_launches: Cell<u64>,
    host_to_device_copies: Cell<u64>,
    host_to_device_bytes: Cell<u64>,
    device_to_host_copies: Cell<u64>,
    device_to_host_bytes: Cell<u64>,
    artifact_uploads: Cell<u64>,
    artifact_upload_bytes: Cell<u64>,
    device_allocation_attempts: Cell<u64>,
    device_allocations: Cell<u64>,
    device_allocation_failures: Cell<u64>,
    device_allocation_bytes: Cell<u64>,
    stream_wide_syncs: Cell<u64>,
    stream_wide_sync_failures: Cell<u64>,
    moe_calls: Cell<u64>,
    moe_tc_calls: Cell<u64>,
    moe_scalar_calls: Cell<u64>,
    moe_reduce_calls: Cell<u64>,
    moe_total_us: Cell<u64>,
    moe_pointer_upload_us: Cell<u64>,
    moe_input_prepare_us: Cell<u64>,
    moe_gate_up_us: Cell<u64>,
    moe_swiglu_us: Cell<u64>,
    moe_hidden_pack_us: Cell<u64>,
    moe_down_us: Cell<u64>,
    arena_hits: Cell<u64>,
    arena_misses: Cell<u64>,
    arena_grows: Cell<u64>,
    arena_reuses: Cell<u64>,
}

impl CudaOpCounterCells {
    pub(crate) fn snapshot(&self) -> CudaOpCounters {
        CudaOpCounters {
            kernel_launches: self.kernel_launches.get(),
            host_to_device_copies: self.host_to_device_copies.get(),
            host_to_device_bytes: self.host_to_device_bytes.get(),
            device_to_host_copies: self.device_to_host_copies.get(),
            device_to_host_bytes: self.device_to_host_bytes.get(),
            artifact_uploads: self.artifact_uploads.get(),
            artifact_upload_bytes: self.artifact_upload_bytes.get(),
            device_allocation_attempts: self.device_allocation_attempts.get(),
            device_allocations: self.device_allocations.get(),
            device_allocation_failures: self.device_allocation_failures.get(),
            device_allocation_bytes: self.device_allocation_bytes.get(),
            stream_wide_syncs: self.stream_wide_syncs.get(),
            stream_wide_sync_failures: self.stream_wide_sync_failures.get(),
            moe_calls: self.moe_calls.get(),
            moe_tc_calls: self.moe_tc_calls.get(),
            moe_scalar_calls: self.moe_scalar_calls.get(),
            moe_reduce_calls: self.moe_reduce_calls.get(),
            moe_total_us: self.moe_total_us.get(),
            moe_pointer_upload_us: self.moe_pointer_upload_us.get(),
            moe_input_prepare_us: self.moe_input_prepare_us.get(),
            moe_gate_up_us: self.moe_gate_up_us.get(),
            moe_swiglu_us: self.moe_swiglu_us.get(),
            moe_hidden_pack_us: self.moe_hidden_pack_us.get(),
            moe_down_us: self.moe_down_us.get(),
            arena_hits: self.arena_hits.get(),
            arena_misses: self.arena_misses.get(),
            arena_grows: self.arena_grows.get(),
            arena_reuses: self.arena_reuses.get(),
        }
    }

    pub(crate) fn reset(&self) {
        self.kernel_launches.set(0);
        self.host_to_device_copies.set(0);
        self.host_to_device_bytes.set(0);
        self.device_to_host_copies.set(0);
        self.device_to_host_bytes.set(0);
        self.artifact_uploads.set(0);
        self.artifact_upload_bytes.set(0);
        self.device_allocation_attempts.set(0);
        self.device_allocations.set(0);
        self.device_allocation_failures.set(0);
        self.device_allocation_bytes.set(0);
        self.stream_wide_syncs.set(0);
        self.stream_wide_sync_failures.set(0);
        self.moe_calls.set(0);
        self.moe_tc_calls.set(0);
        self.moe_scalar_calls.set(0);
        self.moe_reduce_calls.set(0);
        self.moe_total_us.set(0);
        self.moe_pointer_upload_us.set(0);
        self.moe_input_prepare_us.set(0);
        self.moe_gate_up_us.set(0);
        self.moe_swiglu_us.set(0);
        self.moe_hidden_pack_us.set(0);
        self.moe_down_us.set(0);
        self.arena_hits.set(0);
        self.arena_misses.set(0);
        self.arena_grows.set(0);
        self.arena_reuses.set(0);
    }

    pub(crate) fn add_kernel_launch(&self) {
        self.kernel_launches
            .set(self.kernel_launches.get().saturating_add(1));
    }

    pub(crate) fn add_host_to_device(&self, bytes: u64) {
        self.host_to_device_copies
            .set(self.host_to_device_copies.get().saturating_add(1));
        self.host_to_device_bytes
            .set(self.host_to_device_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn add_device_to_host(&self, bytes: u64) {
        self.device_to_host_copies
            .set(self.device_to_host_copies.get().saturating_add(1));
        self.device_to_host_bytes
            .set(self.device_to_host_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn add_artifact_upload(&self, bytes: u64) {
        self.artifact_uploads
            .set(self.artifact_uploads.get().saturating_add(1));
        self.artifact_upload_bytes
            .set(self.artifact_upload_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn begin_device_allocation(&self) {
        self.device_allocation_attempts
            .set(self.device_allocation_attempts.get().saturating_add(1));
    }

    pub(crate) fn complete_device_allocation(&self, bytes: u64) {
        self.device_allocations
            .set(self.device_allocations.get().saturating_add(1));
        self.device_allocation_bytes
            .set(self.device_allocation_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn fail_device_allocation(&self) {
        self.device_allocation_failures
            .set(self.device_allocation_failures.get().saturating_add(1));
    }

    pub(crate) fn complete_stream_wide_sync(&self) {
        self.stream_wide_syncs
            .set(self.stream_wide_syncs.get().saturating_add(1));
    }

    pub(crate) fn fail_stream_wide_sync(&self) {
        self.stream_wide_sync_failures
            .set(self.stream_wide_sync_failures.get().saturating_add(1));
    }

    pub(crate) fn add_moe_call(&self, path: CudaMoeExecutionPath) {
        self.moe_calls.set(self.moe_calls.get().saturating_add(1));
        match path {
            CudaMoeExecutionPath::TensorCore => self
                .moe_tc_calls
                .set(self.moe_tc_calls.get().saturating_add(1)),
            CudaMoeExecutionPath::Scalar => self
                .moe_scalar_calls
                .set(self.moe_scalar_calls.get().saturating_add(1)),
            CudaMoeExecutionPath::Reduce => self
                .moe_reduce_calls
                .set(self.moe_reduce_calls.get().saturating_add(1)),
        }
    }

    pub(crate) fn add_moe_total_us(&self, us: u64) {
        self.moe_total_us
            .set(self.moe_total_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_pointer_upload_us(&self, us: u64) {
        self.moe_pointer_upload_us
            .set(self.moe_pointer_upload_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_input_prepare_us(&self, us: u64) {
        self.moe_input_prepare_us
            .set(self.moe_input_prepare_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_gate_up_us(&self, us: u64) {
        self.moe_gate_up_us
            .set(self.moe_gate_up_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_swiglu_us(&self, us: u64) {
        self.moe_swiglu_us
            .set(self.moe_swiglu_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_down_us(&self, us: u64) {
        self.moe_down_us
            .set(self.moe_down_us.get().saturating_add(us));
    }

    pub(crate) fn add_arena_hit(&self) {
        self.arena_hits.set(self.arena_hits.get().saturating_add(1));
    }

    pub(crate) fn add_arena_miss(&self) {
        self.arena_misses
            .set(self.arena_misses.get().saturating_add(1));
    }

    pub(crate) fn add_arena_grow(&self) {
        self.arena_grows
            .set(self.arena_grows.get().saturating_add(1));
    }

    pub(crate) fn add_arena_reuse(&self) {
        self.arena_reuses
            .set(self.arena_reuses.get().saturating_add(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_and_sync_counters_distinguish_success_and_failure() {
        let counters = CudaOpCounterCells::default();
        counters.begin_device_allocation();
        counters.complete_device_allocation(128);
        counters.begin_device_allocation();
        counters.fail_device_allocation();
        counters.complete_stream_wide_sync();
        counters.fail_stream_wide_sync();

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.device_allocation_attempts, 2);
        assert_eq!(snapshot.device_allocations, 1);
        assert_eq!(snapshot.device_allocation_failures, 1);
        assert_eq!(snapshot.device_allocation_bytes, 128);
        assert_eq!(snapshot.stream_wide_syncs, 1);
        assert_eq!(snapshot.stream_wide_sync_failures, 1);

        counters.reset();
        assert_eq!(counters.snapshot(), CudaOpCounters::default());
    }

    #[test]
    fn arena_counters_track_hit_miss_grow_reuse() {
        let counters = CudaOpCounterCells::default();
        counters.add_arena_hit();
        counters.add_arena_hit();
        counters.add_arena_miss();
        counters.add_arena_grow();
        counters.add_arena_reuse();
        counters.add_arena_reuse();
        counters.add_arena_reuse();

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.arena_hits, 2);
        assert_eq!(snapshot.arena_misses, 1);
        assert_eq!(snapshot.arena_grows, 1);
        assert_eq!(snapshot.arena_reuses, 3);

        counters.reset();
        assert_eq!(counters.snapshot(), CudaOpCounters::default());
    }

    #[test]
    fn failpoints_fire_once_then_disarm() {
        let fp = CudaFailpoints::default();

        // Allocation failpoint fires once.
        fp.arm_allocation();
        assert!(fp.check_allocation());
        assert!(!fp.check_allocation());

        // Arena acquire failpoint fires once.
        fp.arm_arena_acquire();
        assert!(fp.check_arena_acquire());
        assert!(!fp.check_arena_acquire());

        // D2D preserve failpoint fires once.
        fp.arm_d2d_preserve();
        assert!(fp.check_d2d_preserve());
        assert!(!fp.check_d2d_preserve());

        // Expert upload failpoint fires once.
        fp.arm_expert_upload();
        assert!(fp.check_expert_upload());
        assert!(!fp.check_expert_upload());

        fp.arm_main_compressor_transition();
        assert!(fp.check_main_compressor_transition());
        assert!(!fp.check_main_compressor_transition());

        fp.arm_indexer_compressor_transition();
        assert!(fp.check_indexer_compressor_transition());
        assert!(!fp.check_indexer_compressor_transition());

        // Resource install failpoint fires once.
        fp.arm_resource_install();
        assert!(fp.check_resource_install());
        assert!(!fp.check_resource_install());
    }

    #[test]
    fn disarm_all_clears_all_failpoints() {
        let fp = CudaFailpoints::default();
        fp.arm_allocation();
        fp.arm_arena_acquire();
        fp.arm_d2d_preserve();
        fp.arm_expert_upload();
        fp.arm_main_compressor_transition();
        fp.arm_indexer_compressor_transition();
        fp.arm_resource_install();
        fp.disarm_all();

        assert!(!fp.check_allocation());
        assert!(!fp.check_arena_acquire());
        assert!(!fp.check_d2d_preserve());
        assert!(!fp.check_expert_upload());
        assert!(!fp.check_main_compressor_transition());
        assert!(!fp.check_indexer_compressor_transition());
        assert!(!fp.check_resource_install());
    }
}
