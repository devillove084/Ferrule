//! Application error mapping over the canonical metadata-only resource probe.
use ferrule_backend::resources::host_memory::{self, HostMemoryProbeError, HostMemorySnapshot};
use ferrule_common::Result;
use ferrule_model::transformer::host_experts::HostExpertCacheError;

fn map_probe_error(source: HostMemoryProbeError) -> ferrule_common::Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(HostExpertCacheError::from(source)),
    }
}

pub(super) fn snapshot() -> Result<HostMemorySnapshot> {
    host_memory::snapshot().map_err(map_probe_error)
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/host_memory.rs"]
mod tests;
