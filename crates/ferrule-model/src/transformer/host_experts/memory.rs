//! Cache-specific error boundary for the shared, device-neutral OS memory probe.
use super::{HostMemorySnapshot, error};
use ferrule_backend::resources::host_memory;
use ferrule_common::Result;

pub(super) fn snapshot() -> Result<HostMemorySnapshot> {
    host_memory::snapshot().map_err(|source| error(source.into()))
}
