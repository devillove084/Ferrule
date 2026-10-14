//! CLI policy is stricter than the library's family-dependent Auto selection.
use ferrule_composition::BackendSelection;

pub(super) fn select(value: Option<&str>) -> anyhow::Result<BackendSelection> {
    match value
        .map(BackendSelection::parse)
        .transpose()?
        .unwrap_or_default()
    {
        BackendSelection::Auto | BackendSelection::Cuda => Ok(BackendSelection::Cuda),
        BackendSelection::Cpu if cfg!(feature = "cpu-reference") => Ok(BackendSelection::Cpu),
        BackendSelection::Cpu => anyhow::bail!(
            "CPU execution is test-only; rebuild with --features cpu-reference for reference tests, or select --backend cuda with a --features cuda build"
        ),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/backend_policy.rs"]
mod tests;
