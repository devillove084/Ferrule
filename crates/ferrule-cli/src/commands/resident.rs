#[cfg(feature = "cuda")]
use std::future::Future;
use std::num::NonZeroU32;

use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::{MultiSessionRunner, ResidentModelRunner};
use ferrule_runtime::cache::KvPageManager;
use ferrule_runtime::{
    FixedSequenceSlotPool, ResidentSchedulerConfig, ResidentTopKDriver, ResidentTopKDriverConfig,
};

/// Run a non-`Send` inference owner on the calling OS thread.
///
/// CUDA context creation, completion reactors, stream callbacks, and every model
/// step therefore remain attached to one owner thread.
#[cfg(feature = "cuda")]
pub(crate) fn block_on_local_inference<F, T>(future: F) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, future)
}

/// Default per-position proposal confidence threshold used by every
/// interactive/benchmark entry point until the calibrated batch-wide
/// scheduler lands.
pub(crate) const DEFAULT_PROPOSAL_CONFIDENCE_THRESHOLD: f32 = 0.2;

/// Driver configuration shared by all resident-model entry points.
pub(crate) fn resident_driver_config(
    ctx_size: usize,
    stop_at_eos: bool,
) -> ResidentTopKDriverConfig {
    ResidentTopKDriverConfig {
        ctx_size,
        stop_at_eos,
        proposal_confidence_threshold: DEFAULT_PROPOSAL_CONFIDENCE_THRESHOLD,
    }
}

/// Scheduler configuration for the single-sequence interactive/benchmark path.
/// Mixed batching is disabled explicitly so the decision is not left to a
/// default that may drift.
#[cfg(feature = "cuda")]
pub(crate) fn single_sequence_scheduler_config(
    prefill_chunk_size: usize,
) -> ResidentSchedulerConfig {
    ResidentSchedulerConfig {
        prefill_chunk_size: prefill_chunk_size.max(1),
        max_active_sequences: 1,
        max_decode_batch: 1,
        allow_mixed_batches: false,
        ..ResidentSchedulerConfig::default()
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn build_resident_topk_driver<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> anyhow::Result<ResidentTopKDriver<R, FixedSequenceSlotPool>>
where
    R: MultiSessionRunner + ResidentModelRunner,
{
    build_resident_topk_driver_with_page_limit(
        runner,
        schema,
        scheduler_config,
        driver_config,
        None,
    )
}

pub(crate) fn build_resident_topk_driver_with_page_limit<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
    max_page_limit: Option<usize>,
) -> anyhow::Result<ResidentTopKDriver<R, FixedSequenceSlotPool>>
where
    R: MultiSessionRunner + ResidentModelRunner,
{
    if driver_config.ctx_size == 0 {
        anyhow::bail!("resident driver ctx_size must be greater than zero");
    }
    if driver_config.ctx_size > schema.max_sequence_len() {
        anyhow::bail!(
            "resident driver ctx_size {} exceeds model KV limit {}",
            driver_config.ctx_size,
            schema.max_sequence_len()
        );
    }

    let max_active_sequences = scheduler_config.max_active_sequences.max(1);
    let full_capacity_pages = schema
        .pages_for_tokens(driver_config.ctx_size)
        .checked_mul(max_active_sequences)
        .filter(|pages| *pages > 0)
        .ok_or_else(|| anyhow::anyhow!("resident driver KV page capacity overflow"))?;
    let max_pages = match max_page_limit {
        Some(0) => anyhow::bail!("resident driver KV page limit must be greater than zero"),
        Some(limit) => full_capacity_pages.min(limit),
        None => full_capacity_pages,
    };

    ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(max_active_sequences),
        scheduler_config,
        NonZeroU32::new(1).expect("top-k one is non-zero"),
        driver_config,
    )
    .try_with_page_manager(KvPageManager::new(schema, max_pages))
    .map_err(Into::into)
}
