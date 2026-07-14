use std::num::NonZeroU32;

use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::MultiSessionRunner;
use ferrule_runtime::cache::KvPageManager;
use ferrule_runtime::{
    FixedSequenceSlotPool, PageManagedDiagnosticHarness, ResidentSchedulerConfig,
    ResidentTopKDriver, ResidentTopKDriverConfig,
};

pub(crate) fn build_page_managed_diagnostic_harness<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    max_tokens: usize,
    max_sequences: usize,
) -> anyhow::Result<PageManagedDiagnosticHarness<R>>
where
    R: MultiSessionRunner,
{
    PageManagedDiagnosticHarness::new(runner, schema, max_tokens, max_sequences).map_err(Into::into)
}

pub(crate) fn build_resident_topk_driver<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> anyhow::Result<ResidentTopKDriver<R, FixedSequenceSlotPool>>
where
    R: MultiSessionRunner,
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
    R: MultiSessionRunner,
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
