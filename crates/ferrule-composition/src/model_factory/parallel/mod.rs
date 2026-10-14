#![cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "CUDA provider planning remains available for CPU-only metadata tests"
    )
)]
//! Parallel composition: immutable geometry, provider budgets and shared owner assembly.
//! Runtime executors retain transaction, publication and quarantine authority.
#[cfg(feature = "cuda")]
use crate::model_factory::PipelineRankBackend;
use crate::model_factory::{
    NumericFp8Profile, PipelineBuildOptions, ResolvedModelRequest, invalid, physical_page_bytes,
};
use crate::{Error, Result};
use ferrule_common::ParallelRankId;
#[cfg(test)]
use ferrule_common::ParallelTopologyId;
#[cfg(any(feature = "cuda", test))]
use ferrule_model::ModelExecutionBackend;
use inputs::NumericProviderInputs;
pub(in crate::model_factory) mod inputs;
#[cfg(unix)]
pub(in crate::model_factory) mod process;
pub(in crate::model_factory) mod reporting;
pub(in crate::model_factory) mod resident;
fn provider_unsupported(reason: impl Into<String>) -> Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
            "numeric_expert_provider",
            reason,
        )),
    }
    .into()
}

use ferrule_model::TokenizerHandle;
#[cfg(feature = "cuda")]
use ferrule_model::decoder::GenericDecoderOptions;
use ferrule_model::decoder::HybridStateSchema;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::host_experts::{
    HostExpertCacheOptions, HostExpertWarmPlan, StandardHostStartupMemory,
};
use ferrule_model::transformer::hybrid_parallel::{
    HybridExpertAttachment, HybridExpertScope, HybridParallelPlan,
};
use ferrule_model::transformer::{BoundDecoderResources, LayerSegmentPlan};
use ferrule_runtime::engine::ResidentKvPagePlan;
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;
#[cfg(any(feature = "cuda", test))]
use ferrule_runtime::parallel::pipeline::HybridDeviceBudget;
use ferrule_runtime::parallel::pipeline::PipelineParallelExecutor;
use ferrule_runtime::parallel::pipeline::{
    HybridPipelineLayout, HybridStageBudget, PhysicalKvLayout, PipelineConfig,
};
pub(super) mod assembly;
pub(super) mod bindings;
pub(super) mod budget;
pub(super) mod experts;
pub(super) mod plan;
#[cfg(feature = "cuda")]
mod provider;
pub(super) mod requirements;
const MARGIN: u64 = 64 << 20;
const EXPERT_WORKSPACE: usize = 64 << 20;
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .filter(|n| *n <= isize::MAX as u64)
        .ok_or_else(|| invalid("parallel budget overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .filter(|n| *n <= isize::MAX as u64)
        .ok_or_else(|| invalid("parallel budget overflow"))
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/parallel_segment.rs"]
mod segment_tests;
#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/parallel_tensor.rs"]
mod tensor_tests;

#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/parallel_assembly.rs"]
mod assembly_tests;

#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/neutral_provider.rs"]
mod neutral_provider_tests;
