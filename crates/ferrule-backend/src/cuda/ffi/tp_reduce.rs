//! Internal opaque native communicator ABI; no Rust CUDA/NCCL dynamic loading.
use std::ffi::c_char;
#[repr(C)]
pub(crate) struct Library {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Comm {
    _private: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct UniqueId {
    pub(crate) bytes: [u8; 128],
}
pub(crate) const OK: i32 = 0;
pub(crate) const PENDING: i32 = 3;
unsafe extern "C" {
    pub(crate) fn ferrule_tp_library_host_bytes() -> usize;
    pub(crate) fn ferrule_tp_comm_host_bytes() -> usize;
    pub(crate) fn ferrule_tp_library_open(path: *const c_char, out: *mut *mut Library) -> i32;
    pub(crate) fn ferrule_tp_library_close(library: *mut Library);
    pub(crate) fn ferrule_tp_unique_id(library: *mut Library, out: *mut UniqueId) -> i32;
    pub(crate) fn ferrule_tp_comm_create(
        library: *mut Library,
        id: *const UniqueId,
        rank: u32,
        stream: u64,
        storage: u64,
        bytes: u64,
        out: *mut *mut Comm,
    ) -> i32;
    pub(crate) fn ferrule_tp_comm_async_status(comm: *mut Comm) -> i32;
    pub(crate) fn ferrule_tp_comm_warmup(comm: *mut Comm) -> i32;
    pub(crate) fn ferrule_tp_allgather_bf16_two(
        comm: *mut Comm,
        input: u64,
        bytes: u64,
        count: u32,
    ) -> i32;
    pub(crate) fn ferrule_tp_comm_progress(comm: *mut Comm) -> i32;
    pub(crate) fn ferrule_tp_comm_read_status(comm: *mut Comm, out: *mut u32) -> i32;
    pub(crate) fn ferrule_tp_comm_abort(comm: *mut Comm) -> i32;
    pub(crate) fn ferrule_tp_comm_destroy(comm: *mut Comm) -> i32;
}

// Diagnostics are reachable only by backend unit tests, not production operators.
#[cfg(test)]
unsafe extern "C" {
    pub(crate) fn ferrule_tp_probe_primary(ordinal: u32, flags: *mut u32, active: *mut i32) -> i32;
    pub(crate) fn ferrule_tp_probe_core(stream: u64, storage: u64, bytes: u64) -> i32;
    pub(crate) fn ferrule_tp_probe_core_status(storage: u64, status: *mut u32) -> i32;
    pub(crate) fn ferrule_tp_probe_trace_start(path: *const c_char) -> i32;
    pub(crate) fn ferrule_tp_probe_counts(out: *mut u64, capacity: u32) -> i32;
}
