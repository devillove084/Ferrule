//! Device-independent expert execution interface.

use crate::moe::streaming::ExpertComputeBundle;
use ferrule_common::Result;

/// Executes a single routed expert for one activation vector.
pub trait ExpertExecutor {
    fn execute(
        &self,
        bundle: &ExpertComputeBundle,
        input: &[f32],
        route_weight: f32,
    ) -> Result<Vec<f32>>;
}

#[cfg(any(test, feature = "cpu-reference"))]
#[path = "../../tests/support/expert_reference.rs"]
mod reference;

#[cfg(any(test, feature = "cpu-reference"))]
pub use reference::{CpuReferenceExpertExecutor, reference_linear};
