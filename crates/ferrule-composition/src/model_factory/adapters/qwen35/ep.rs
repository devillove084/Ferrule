//! Strict resident EP adapter. Owner construction is model-neutral.
#[cfg(any(feature = "cuda", test))]
use super::*;
#[cfg(test)]
use crate::model_factory::parallel::experts::OwnerPlan;
#[cfg(test)]
use ferrule_model::{nn::ParameterResidency, transformer::FeedForward};
#[cfg(test)]
const OWNER_WORKSPACE: usize = 64 << 20;
#[cfg(test)]
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| ep_unsupported("EP memory estimate overflow"))
}
#[cfg(test)]
#[path = "../../../../tests/unit/engine/model_factory/qwen35_ep_scratch.rs"]
mod scratch_budget_tests;
#[cfg(all(test, feature = "cuda"))]
#[path = "../../../../tests/unit/engine/model_factory/qwen35_ep_strict_tests.rs"]
pub(in crate::model_factory) mod strict;
#[cfg(test)]
#[path = "../../../../tests/unit/engine/model_factory/qwen35_ep_tests.rs"]
pub(in crate::model_factory) mod tests;

#[cfg(feature = "cuda")]
pub(super) fn build(request: ResolvedModelRequest) -> Result<BoxedSessionInferenceEngine> {
    if request.family != ModelFamily::Qwen35Moe
        || request.backend != ModelExecutionBackend::Cuda
        || request.pipeline.is_some()
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires a CUDA resident root, not CPU or Pipeline",
        ));
    }
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    if metadata.config().profile() != ferrule_model::models::qwen35::Qwen35Profile::Moe35BA3Bfp8 {
        return Err(ep_unsupported(
            "resident EP requires the strict Qwen3.5 MoE FP8 profile",
        ));
    }
    let inputs = super::binding::resident_inputs(&request, &resources)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    crate::model_factory::parallel::resident::build_bound(request, resources, tokenizer, inputs)
}
#[cfg(all(test, feature = "cuda"))]
pub(in crate::model_factory) fn prepare_bound(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<crate::model_factory::parallel::resident::PreparedRoot> {
    let inputs = super::binding::resident_inputs(request, &resources)?;
    crate::model_factory::parallel::resident::prepare_bound(request, resources, tokenizer, inputs)
}

#[cfg(all(test, feature = "cuda"))]
use ferrule_model::decoder::{
    GenericDecoderRunner, HybridCudaDecoder, HybridCudaDevice, HybridCudaMemoryBudget,
};
#[cfg(all(test, feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(test, feature = "cuda"))]
fn build_bound(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<BoxedSessionInferenceEngine> {
    let inputs = super::binding::resident_inputs(&request, &resources)?;
    crate::model_factory::parallel::resident::build_bound(request, resources, tokenizer, inputs)
}

#[cfg(all(test, unix))]
#[path = "../../../../tests/unit/engine/model_factory"]
mod process {
    #[path = "qwen35_ep_process.rs"]
    mod tests;
}
