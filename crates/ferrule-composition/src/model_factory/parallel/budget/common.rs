//! Checked provider calculations shared by resident, pipeline and process owners.
use super::super::inputs::NumericCapacityLimits;
use crate::Result;
use crate::model_factory::{NumericFp8Profile, invalid};
use ferrule_model::{
    ModelFamily, WeightSource, decoder::GenericDecoderOptions, execution::ExecutionPrecisionPolicy,
    transformer::BoundDecoderResources,
};

pub(in crate::model_factory) fn decoder_options(
    resources: &BoundDecoderResources,
    family: ModelFamily,
    limits: NumericCapacityLimits,
    max_tensor_bytes: u64,
    profile: NumericFp8Profile,
) -> Result<GenericDecoderOptions> {
    Ok(GenericDecoderOptions::standard_cpu(
        resources.spec(),
        family,
        WeightSource::Safetensors,
        16,
        limits.max_positions,
        limits.max_batch_tokens,
        limits.max_sequences,
        max_tensor_bytes,
        ExecutionPrecisionPolicy::f32(),
    )?
    .with_hybrid_cuda_numeric_fp8_precision(
        ferrule_model::transformer::ExpertCacheLimits {
            max_experts: limits.max_experts,
            max_bytes: limits.max_bytes,
        },
        limits.scratch_bytes,
        profile.precision(),
    )?)
}

pub(in crate::model_factory) fn expert_payload_envelope(
    weights: usize,
    activation: usize,
) -> Result<usize> {
    weights
        .checked_add(super::super::EXPERT_WORKSPACE)
        .and_then(|n| n.checked_add(activation))
        .ok_or_else(|| invalid("expert payload envelope overflow"))
}

pub(in crate::model_factory) fn process_child_overhead(
    metadata: u64,
    frame: u64,
    owners: u64,
) -> Result<u64> {
    metadata
        .checked_mul(4)
        .and_then(|n| n.checked_add(16 << 20))
        .and_then(|n| frame.checked_mul(32).and_then(|f| n.checked_add(f)))
        .and_then(|n| n.checked_mul(owners))
        .ok_or_else(|| invalid("process metadata/IPC overflow"))
}

/// Metadata-only host ledger for the actual bound image. Kept CPU-testable;
/// no CUDA device, tokenizer or weight payload has been created at this point.
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "host metadata remains compiled independently of CUDA construction"
    )
)]
pub(in crate::model_factory) fn host_startup_options(
    resources: &BoundDecoderResources,
    mut options: ferrule_model::transformer::host_experts::HostExpertCacheOptions,
    max_positions: usize,
    max_tensor_bytes: u64,
) -> Result<ferrule_model::transformer::host_experts::HostExpertCacheOptions> {
    use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, StandardHostStartupMemory};
    if options.mode == ExpertPrewarmMode::Full {
        let base =
            StandardHostStartupMemory::for_resources(resources, max_positions, max_tensor_bytes)?;
        // Fresh construction: no base payload credit. Catalog/planning allocations
        // already live in RSS/current and must not be charged a second time.
        options.memory_budget = base.memory_budget()?;
        tracing::info!(
            ?base,
            "numeric remaining host base ledger (after expert warm, before ready)"
        );
    }
    Ok(options)
}
