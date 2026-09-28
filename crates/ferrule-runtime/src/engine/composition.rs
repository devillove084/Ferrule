//! Model-neutral composition of resident runners, KV ownership, and scheduling.

use std::num::NonZeroU32;

use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::ResidentModelRunner;

use crate::cache::KvPageManager;
use crate::scheduling::{FixedSequenceSlotPool, ResidentSchedulerConfig};
use crate::{Error, Result};

use super::{
    BoxedSessionInferenceEngine, ResidentInferenceEngine, ResidentTopKDriver,
    ResidentTopKDriverConfig,
};

/// How the authoritative physical KV pool is bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentKvPageAccounting {
    /// Provision enough pages for every configured active sequence at `ctx_size`.
    ContextCapacity,
    /// Reserve full context capacity plus one shadow per committed page.
    /// Both sets are charged against the same physical byte budget.
    TransactionCapacity {
        page_bytes: u64,
        budget_bytes: Option<u64>,
    },
    /// Apply an explicit physical page ceiling.
    PageLimit(usize),
    /// Derive the physical page ceiling from a byte budget and backend page size.
    PhysicalBytes { budget_bytes: u64, page_bytes: u64 },
}

/// Resolved logical and physical KV capacity for one resident engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentKvPagePlan {
    pub full_capacity_pages: usize,
    pub configured_pages: usize,
    pub page_bytes: Option<u64>,
    pub configured_bytes: Option<u64>,
}

/// Read-only effective context envelope, not a reservation or a second ledger.
/// Byte requirements are owner estimates/caps, never free-memory gauges. Unknown
/// model memory remains `None`; zero is reserved for an explicitly empty domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCapacityProfile {
    pub ctx_size: usize,
    pub max_active_sequences: usize,
    /// None preserves the scheduler's legacy unlimited batch-token policy.
    pub max_batch_tokens: Option<usize>,
    pub prefill_chunk_size: usize,
    /// Full-context demand versus the configured pool (including shadows).
    pub kv: Option<ResidentKvPagePlan>,
    pub logical_kv_bytes: Option<u64>,
    pub recurrent_bytes: Option<usize>,
    pub workspace_bytes: Option<usize>,
    /// Reported root total, including weights/cache/margins when applicable.
    /// A schema target is still an estimate, not owner admission. This is not
    /// the sum of per-card EP requirements or a device reservation.
    pub root_required_bytes: Option<usize>,
}

impl ContextCapacityProfile {
    pub fn from_configs(
        scheduler: ResidentSchedulerConfig,
        driver: ResidentTopKDriverConfig,
    ) -> Result<Self> {
        if driver.ctx_size == 0 || driver.ctx_size > u32::MAX as usize {
            return Err(Error::InvalidRequest {
                message: "context capacity must fit nonzero execution positions".into(),
            });
        }
        let max_batch_tokens =
            (scheduler.max_batch_tokens != 0).then_some(scheduler.max_batch_tokens);
        Ok(Self {
            ctx_size: driver.ctx_size,
            max_active_sequences: scheduler.max_active_sequences.max(1),
            max_batch_tokens,
            prefill_chunk_size: scheduler
                .prefill_chunk_size
                .max(1)
                .min(max_batch_tokens.unwrap_or(usize::MAX)),
            kv: None,
            logical_kv_bytes: None,
            recurrent_bytes: None,
            workspace_bytes: None,
            root_required_bytes: None,
        })
    }

    /// Check an absolute retained/fork position, prompt and requested output.
    /// This does not inspect or reserve currently available KV/sequence credits.
    pub fn validate_request(
        &self,
        position: usize,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<()> {
        validate_context_envelope(self.ctx_size, position, prompt_tokens, max_new_tokens)
    }

    /// Attach the existing owner's budget, without recomputing its weight,
    /// cache, numeric-scratch or allocator accounting. Reject inconsistent sums.
    pub fn with_memory_requirements(
        mut self,
        recurrent_bytes: usize,
        workspace_bytes: usize,
        root_required_bytes: usize,
    ) -> Result<Self> {
        let kv_bytes = self
            .kv
            .and_then(|kv| kv.configured_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Error::InvalidRequest {
                message: "root memory profile requires known physical KV bytes".into(),
            })?;
        let minimum = recurrent_bytes
            .checked_add(workspace_bytes)
            .and_then(|bytes| bytes.checked_add(kv_bytes))
            .ok_or_else(|| Error::InvalidRequest {
                message: "context memory profile overflow".into(),
            })?;
        if minimum > root_required_bytes {
            return Err(Error::InvalidRequest {
                message: "context memory domains exceed owner root requirement".into(),
            });
        }
        self.recurrent_bytes = Some(recurrent_bytes);
        self.workspace_bytes = Some(workspace_bytes);
        self.root_required_bytes = Some(root_required_bytes);
        Ok(self)
    }
}

impl ResidentKvPagePlan {
    /// Report an already resolved KV plan; this does not grow or reserve its pool.
    pub fn context_capacity_profile(
        self,
        scheduler: ResidentSchedulerConfig,
        driver: ResidentTopKDriverConfig,
    ) -> Result<ContextCapacityProfile> {
        let mut profile = ContextCapacityProfile::from_configs(scheduler, driver)?;
        let bytes = |pages: usize, page_bytes: u64| {
            u64::try_from(pages)
                .ok()
                .and_then(|n| n.checked_mul(page_bytes))
                .filter(|bytes| *bytes > 0)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "context KV profile byte capacity overflow".into(),
                })
        };
        if self.full_capacity_pages == 0 || self.configured_pages == 0 {
            return Err(Error::InvalidRequest {
                message: "context KV profile requires nonzero page capacities".into(),
            });
        }
        let physical_bytes = self
            .page_bytes
            .map(|size| bytes(self.configured_pages, size))
            .transpose()?;
        if physical_bytes != self.configured_bytes {
            return Err(Error::InvalidRequest {
                message: "context KV profile differs from owner byte accounting".into(),
            });
        }
        profile.logical_kv_bytes = self
            .page_bytes
            .map(|size| bytes(self.full_capacity_pages, size))
            .transpose()?;
        profile.kv = Some(self);
        Ok(profile)
    }
}

/// Validate the permanent request envelope only. Physical KV and sequence
/// availability remain on the existing waiting/transaction owners.
pub(crate) fn validate_context_envelope(
    context: usize,
    position: usize,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let limit = context.min(u32::MAX as usize);
    let prompt_end = position.checked_add(prompt_tokens);
    if prompt_end.is_none_or(|end| end > limit) {
        return Err(super::RuntimeAdmissionError::InvalidPosition {
            position,
            prompt_tokens,
            context,
        }
        .into());
    }
    // Reuse the existing typed position error with the input context remaining
    // after reserving the requested output envelope. No KV grant is acquired.
    let input_context = limit.checked_sub(max_new_tokens);
    if input_context.is_none_or(|available| prompt_end.unwrap() > available) {
        return Err(super::RuntimeAdmissionError::InvalidPosition {
            position,
            prompt_tokens,
            context: input_context.unwrap_or(0),
        }
        .into());
    }
    Ok(())
}

/// Resolve the physical KV capacity without constructing model or backend state.
pub fn plan_resident_kv_pages(
    schema: &dyn KvLayoutSchema,
    accounting: ResidentKvPageAccounting,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> Result<ResidentKvPagePlan> {
    if driver_config.ctx_size == 0 || driver_config.ctx_size > u32::MAX as usize {
        return Err(Error::InvalidRequest {
            message: "resident engine ctx_size must fit nonzero execution positions".into(),
        });
    }
    if schema.page_size() == 0 {
        return Err(Error::InvalidRequest {
            message: "resident engine KV page size must be greater than zero".into(),
        });
    }
    if driver_config.ctx_size > schema.max_sequence_len() {
        return Err(Error::InvalidRequest {
            message: format!(
                "resident engine ctx_size {} exceeds model KV limit {}",
                driver_config.ctx_size,
                schema.max_sequence_len()
            ),
        });
    }

    let max_active_sequences = scheduler_config.max_active_sequences.max(1);
    let full_capacity_pages = schema
        .pages_for_tokens(driver_config.ctx_size)
        .checked_mul(max_active_sequences)
        .filter(|pages| *pages > 0)
        .ok_or_else(|| Error::InvalidRequest {
            message: "resident engine KV page capacity overflow".into(),
        })?;

    if let ResidentKvPageAccounting::TransactionCapacity {
        page_bytes,
        budget_bytes,
    } = accounting
    {
        let configured_pages =
            full_capacity_pages
                .checked_mul(2)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "transaction KV capacity overflow".into(),
                })?;
        let configured_bytes = u64::try_from(configured_pages)
            .ok()
            .and_then(|pages| pages.checked_mul(page_bytes))
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::InvalidRequest {
                message: "transaction KV byte capacity overflow".into(),
            })?;
        if budget_bytes.is_some_and(|budget| configured_bytes > budget) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "hybrid CUDA KV budget must cover full contexts plus transaction shadows: required {configured_bytes} bytes, budget {budget_bytes:?}; reduce ctx-size or max-active-sequences"
                ),
            });
        }
        return Ok(ResidentKvPagePlan {
            full_capacity_pages,
            configured_pages,
            page_bytes: Some(page_bytes),
            configured_bytes: Some(configured_bytes),
        });
    }
    let (page_limit, page_bytes) = match accounting {
        ResidentKvPageAccounting::TransactionCapacity { .. } => unreachable!(),
        ResidentKvPageAccounting::ContextCapacity => (full_capacity_pages, None),
        ResidentKvPageAccounting::PageLimit(0) => {
            return Err(Error::InvalidRequest {
                message: "resident engine KV page limit must be greater than zero".into(),
            });
        }
        ResidentKvPageAccounting::PageLimit(limit) => (limit, None),
        ResidentKvPageAccounting::PhysicalBytes {
            budget_bytes,
            page_bytes: 0,
        } => {
            return Err(Error::InvalidRequest {
                message: format!(
                    "physical KV page size must be greater than zero for budget {budget_bytes}"
                ),
            });
        }
        ResidentKvPageAccounting::PhysicalBytes {
            budget_bytes,
            page_bytes,
        } => {
            let pages = budget_bytes / page_bytes;
            if pages == 0 {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "physical KV budget ({budget_bytes} bytes) is smaller than one page ({page_bytes} bytes)"
                    ),
                });
            }
            let pages = usize::try_from(pages).map_err(|_| Error::InvalidRequest {
                message: "physical KV page budget exceeds usize".into(),
            })?;
            (pages, Some(page_bytes))
        }
    };

    let configured_pages = full_capacity_pages.min(page_limit);
    let configured_bytes = page_bytes
        .map(|page_bytes| {
            u64::try_from(configured_pages)
                .ok()
                .and_then(|pages| pages.checked_mul(page_bytes))
                .ok_or_else(|| Error::InvalidRequest {
                    message: "resident engine KV byte estimate overflow".into(),
                })
        })
        .transpose()?;

    Ok(ResidentKvPagePlan {
        full_capacity_pages,
        configured_pages,
        page_bytes,
        configured_bytes,
    })
}

/// Pipeline admission is full-context only. CPU never inherits CUDA shadows;
/// both paths report and charge the physical bytes even without an explicit cap.
pub(super) fn plan_pipeline_kv_pages(
    schema: &dyn KvLayoutSchema,
    page_bytes: u64,
    budget_bytes: Option<u64>,
    cuda: bool,
    scheduler: ResidentSchedulerConfig,
    driver: ResidentTopKDriverConfig,
) -> Result<ResidentKvPagePlan> {
    let mut plan = plan_resident_kv_pages(
        schema,
        ResidentKvPageAccounting::ContextCapacity,
        scheduler,
        driver,
    )?;
    plan.configured_pages = plan
        .full_capacity_pages
        .checked_mul(if cuda { 2 } else { 1 })
        .ok_or_else(|| Error::InvalidRequest {
            message: "pipeline physical KV capacity overflow".into(),
        })?;
    let bytes = u64::try_from(plan.configured_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(page_bytes))
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| Error::InvalidRequest {
            message: "pipeline physical KV byte capacity overflow".into(),
        })?;
    if budget_bytes.is_some_and(|budget| bytes > budget) {
        return Err(Error::InvalidRequest {
            message: format!(
                "pipeline KV budget cannot cover full contexts and physical transaction slots: required {bytes} bytes, budget {budget_bytes:?}"
            ),
        });
    }
    plan.page_bytes = Some(page_bytes);
    plan.configured_bytes = Some(bytes);
    Ok(plan)
}

/// Build the object-safe resident inference boundary used by interactive,
/// benchmark, and serving frontends.
pub fn build_resident_engine<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    accounting: ResidentKvPageAccounting,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> Result<BoxedSessionInferenceEngine>
where
    R: ResidentModelRunner + 'static,
    R::SequenceState: 'static,
{
    Ok(Box::new(compose_resident_engine(
        runner,
        schema,
        accounting,
        scheduler_config,
        driver_config,
    )?))
}

/// Preserve the concrete owner for factory-specific read-only shutdown reports.
pub(crate) fn compose_resident_engine<R>(
    runner: R,
    schema: Box<dyn KvLayoutSchema>,
    accounting: ResidentKvPageAccounting,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> Result<ResidentInferenceEngine<R, FixedSequenceSlotPool>>
where
    R: ResidentModelRunner + 'static,
    R::SequenceState: 'static,
{
    let plan =
        plan_resident_kv_pages(schema.as_ref(), accounting, scheduler_config, driver_config)?;
    // Enrich only the read-only report when legacy accounting omitted bytes.
    // Keep the original page plan and its ownership/accounting policy unchanged.
    let page_bytes = plan
        .page_bytes
        .or_else(|| {
            schema
                .checked_page_bytes()
                .and_then(|bytes| u64::try_from(bytes).ok())
        })
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| Error::InvalidRequest {
            message: "context KV page bytes overflow or empty layout".into(),
        })?;
    let configured_bytes = u64::try_from(plan.configured_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(page_bytes))
        .ok_or_else(|| Error::InvalidRequest {
            message: "context physical KV bytes overflow".into(),
        })?;
    let profile = ResidentKvPagePlan {
        page_bytes: Some(page_bytes),
        configured_bytes: Some(configured_bytes),
        ..plan
    }
    .context_capacity_profile(scheduler_config, driver_config)?;
    tracing::info!(
        ?profile,
        "resident effective context capacity (not free memory)"
    );
    let max_active_sequences = scheduler_config.max_active_sequences.max(1);
    let driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(max_active_sequences),
        scheduler_config,
        NonZeroU32::new(1).expect("top-k one is non-zero"),
        driver_config,
    )
    .try_with_page_manager(KvPageManager::new(schema, plan.configured_pages))?;

    Ok(ResidentInferenceEngine::with_kv_page_plan(driver, plan))
}

#[cfg(test)]
mod tests {
    use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPlaneDescriptor};

    use super::*;

    static TEST_PLANE: KvPlaneDescriptor = KvPlaneDescriptor::new("test", 1, 1, KvElementType::F32);

    #[derive(Debug)]
    struct TestSchema;

    impl KvLayoutSchema for TestSchema {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            std::slice::from_ref(&TEST_PLANE)
        }

        fn page_size(&self) -> usize {
            4
        }

        fn max_sequence_len(&self) -> usize {
            64
        }
    }

    #[test]
    fn pipeline_capacity_exact_physical_budget_and_checked_overflow() {
        let scheduler = ResidentSchedulerConfig {
            max_active_sequences: 1,
            ..Default::default()
        };
        let driver = ResidentTopKDriverConfig {
            ctx_size: 2,
            ..Default::default()
        };
        for cuda in [false, true] {
            let pages = if cuda { 2 } else { 1 };
            let bytes = pages as u64 * 16;
            let plan =
                plan_pipeline_kv_pages(&TestSchema, 16, Some(bytes), cuda, scheduler, driver)
                    .unwrap();
            assert_eq!(plan.full_capacity_pages, 1);
            assert_eq!(plan.configured_pages, pages);
            assert_eq!(plan.configured_bytes, Some(bytes));
            assert_eq!(
                plan,
                plan_pipeline_kv_pages(&TestSchema, 16, None, cuda, scheduler, driver).unwrap()
            );
            assert!(
                plan_pipeline_kv_pages(&TestSchema, 16, Some(bytes - 1), cuda, scheduler, driver)
                    .is_err()
            );
            assert!(plan_pipeline_kv_pages(&TestSchema, 0, None, cuda, scheduler, driver).is_err());
        }
        assert!(
            plan_pipeline_kv_pages(&TestSchema, u64::MAX, None, true, scheduler, driver).is_err()
        );
        let huge = ResidentSchedulerConfig {
            max_active_sequences: usize::MAX,
            ..scheduler
        };
        assert!(plan_pipeline_kv_pages(&TestSchema, 1, None, true, huge, driver).is_err());
        assert!(
            plan_pipeline_kv_pages(
                &TestSchema,
                1,
                None,
                false,
                huge,
                ResidentTopKDriverConfig {
                    ctx_size: 8,
                    ..driver
                }
            )
            .is_err()
        );
    }

    #[test]
    fn transaction_capacity_charges_shadows_inside_hard_budget() {
        let scheduler = ResidentSchedulerConfig {
            max_active_sequences: 3,
            ..Default::default()
        };
        let driver = ResidentTopKDriverConfig {
            ctx_size: 10,
            ..Default::default()
        };
        let accounting = |budget| ResidentKvPageAccounting::TransactionCapacity {
            page_bytes: 10,
            budget_bytes: budget,
        };
        let plan =
            plan_resident_kv_pages(&TestSchema, accounting(Some(180)), scheduler, driver).unwrap();
        assert_eq!(plan.full_capacity_pages, 9);
        assert_eq!(plan.configured_pages, 18);
        assert_eq!(plan.configured_bytes, Some(180));
        assert!(
            plan_resident_kv_pages(&TestSchema, accounting(Some(179)), scheduler, driver).is_err()
        );
    }

    #[test]
    fn physical_byte_accounting_caps_full_context_capacity() {
        let scheduler = ResidentSchedulerConfig {
            max_active_sequences: 3,
            ..ResidentSchedulerConfig::default()
        };
        let driver = ResidentTopKDriverConfig {
            ctx_size: 10,
            ..ResidentTopKDriverConfig::default()
        };
        let plan = plan_resident_kv_pages(
            &TestSchema,
            ResidentKvPageAccounting::PhysicalBytes {
                budget_bytes: 50,
                page_bytes: 10,
            },
            scheduler,
            driver,
        )
        .unwrap();

        assert_eq!(plan.full_capacity_pages, 9);
        assert_eq!(plan.configured_pages, 5);
        assert_eq!(plan.configured_bytes, Some(50));
    }

    #[test]
    fn physical_byte_accounting_rejects_sub_page_budget() {
        let error = plan_resident_kv_pages(
            &TestSchema,
            ResidentKvPageAccounting::PhysicalBytes {
                budget_bytes: 9,
                page_bytes: 10,
            },
            ResidentSchedulerConfig::default(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                ..ResidentTopKDriverConfig::default()
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("smaller than one page"));
    }
}

#[cfg(test)]
mod long_context_tests {
    use super::*;
    use ferrule_common::execution::{KvElementType, KvPlaneDescriptor};

    #[derive(Debug)]
    struct Schema;
    impl KvLayoutSchema for Schema {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            static PLANE: KvPlaneDescriptor =
                KvPlaneDescriptor::new("kv", 10, 1024, KvElementType::F32);
            std::slice::from_ref(&PLANE)
        }
        fn page_size(&self) -> usize {
            16
        }
        fn max_sequence_len(&self) -> usize {
            usize::MAX
        }
    }

    #[test]
    fn long_context_profile_1k_2k_4k_logical_physical_and_budget_boundaries() {
        for ctx in [1024usize, 2048, 4096] {
            for active in [1, 4] {
                let scheduler = ResidentSchedulerConfig {
                    max_active_sequences: active,
                    max_batch_tokens: 32,
                    prefill_chunk_size: 16,
                    ..Default::default()
                };
                let driver = ResidentTopKDriverConfig {
                    ctx_size: ctx,
                    ..Default::default()
                };
                let logical = ctx / 16 * active;
                for shadows in [false, true] {
                    let physical = logical * if shadows { 2 } else { 1 };
                    let page_bytes = Schema.checked_page_bytes().unwrap() as u64;
                    let total = physical as u64 * page_bytes;
                    let plan = plan_pipeline_kv_pages(
                        &Schema,
                        page_bytes,
                        Some(total),
                        shadows,
                        scheduler,
                        driver,
                    )
                    .unwrap();
                    let profile = plan.context_capacity_profile(scheduler, driver).unwrap();
                    assert_eq!(profile.ctx_size, ctx);
                    assert_eq!(profile.max_active_sequences, active);
                    assert_eq!(profile.max_batch_tokens, Some(32));
                    assert_eq!(profile.prefill_chunk_size, 16);
                    assert_eq!(profile.kv.unwrap().full_capacity_pages, logical);
                    assert_eq!(profile.kv.unwrap().configured_pages, physical);
                    assert_eq!(profile.logical_kv_bytes, Some(logical as u64 * page_bytes));
                    assert_eq!(profile.kv.unwrap().configured_bytes, Some(total));
                    assert!(profile.root_required_bytes.is_none());
                    assert!(
                        plan_pipeline_kv_pages(
                            &Schema,
                            page_bytes,
                            Some(total - 1),
                            shadows,
                            scheduler,
                            driver
                        )
                        .is_err()
                    );
                    let root = total as usize + 100 + 200 + 300;
                    let memory = profile.with_memory_requirements(100, 200, root).unwrap();
                    assert_eq!(memory.root_required_bytes, Some(root));
                    assert!(
                        profile
                            .with_memory_requirements(100, 200, total as usize + 299)
                            .is_err()
                    );
                    assert!(
                        profile
                            .with_memory_requirements(usize::MAX, 1, usize::MAX)
                            .is_err()
                    );
                    eprintln!("long-context arithmetic only: {memory:?}");
                }
                let partial = plan_resident_kv_pages(
                    &Schema,
                    ResidentKvPageAccounting::PhysicalBytes {
                        page_bytes: 10,
                        budget_bytes: 30,
                    },
                    scheduler,
                    driver,
                )
                .unwrap()
                .context_capacity_profile(scheduler, driver)
                .unwrap();
                assert_eq!(partial.kv.unwrap().configured_pages, 3);
                assert_eq!(partial.max_active_sequences, active);
                assert!(
                    partial.validate_request(0, ctx - 32, 32).is_ok(),
                    "transient capacity is not permanent rejection"
                );
            }
        }
    }

    #[test]
    fn long_context_profile_checked_overflow_and_legacy_unlimited_batch() {
        let scheduler = ResidentSchedulerConfig {
            max_batch_tokens: 0,
            max_active_sequences: 0,
            prefill_chunk_size: 0,
            ..Default::default()
        };
        let driver = ResidentTopKDriverConfig::default();
        let profile = ContextCapacityProfile::from_configs(scheduler, driver).unwrap();
        assert_eq!(profile.max_batch_tokens, None);
        assert_eq!(profile.max_active_sequences, 1);
        assert_eq!(profile.prefill_chunk_size, 1);
        assert!(profile.with_memory_requirements(0, 0, 0).is_err());
        for ctx in [0, usize::MAX] {
            assert!(
                ContextCapacityProfile::from_configs(
                    scheduler,
                    ResidentTopKDriverConfig {
                        ctx_size: ctx,
                        ..driver
                    }
                )
                .is_err()
            );
        }
        for accounting in [
            ResidentKvPageAccounting::ContextCapacity,
            ResidentKvPageAccounting::TransactionCapacity {
                page_bytes: 1,
                budget_bytes: None,
            },
        ] {
            assert!(
                plan_resident_kv_pages(
                    &Schema,
                    accounting,
                    ResidentSchedulerConfig {
                        max_active_sequences: usize::MAX,
                        ..scheduler
                    },
                    driver
                )
                .is_err()
            );
        }
        assert!(plan_pipeline_kv_pages(&Schema, u64::MAX, None, true, scheduler, driver).is_err());
        for plan in [
            ResidentKvPagePlan {
                full_capacity_pages: usize::MAX,
                configured_pages: 1,
                page_bytes: Some(2),
                configured_bytes: Some(2),
            },
            ResidentKvPagePlan {
                full_capacity_pages: 1,
                configured_pages: 1,
                page_bytes: Some(2),
                configured_bytes: Some(1),
            },
            ResidentKvPagePlan {
                full_capacity_pages: 1,
                configured_pages: 0,
                page_bytes: None,
                configured_bytes: None,
            },
        ] {
            assert!(plan.context_capacity_profile(scheduler, driver).is_err());
        }
    }

    #[test]
    fn long_context_envelope_boundaries_retained_positions_and_overflow() {
        for ctx in [1024, 2048, 4096] {
            for position in [0, 17, ctx - 32] {
                let prompt = ctx - position - 32;
                assert!(validate_context_envelope(ctx, position, prompt, 32).is_ok());
                assert!(matches!(
                    validate_context_envelope(ctx, position, prompt, 33),
                    Err(Error::Admission {
                        source: super::super::RuntimeAdmissionError::InvalidPosition { .. }
                    })
                ));
            }
            assert!(validate_context_envelope(ctx, 0, ctx, 0).is_ok());
            for (position, prompt, generated) in [
                (0, ctx, 1),
                (0, ctx + 1, 0),
                (usize::MAX, 1, 0),
                (1, usize::MAX, 1),
                (0, 1, usize::MAX),
                (ctx, 0, 1),
            ] {
                assert!(matches!(
                    validate_context_envelope(ctx, position, prompt, generated),
                    Err(Error::Admission {
                        source: super::super::RuntimeAdmissionError::InvalidPosition { .. }
                    })
                ));
            }
        }
        let protocol_max = u32::MAX as usize;
        assert!(validate_context_envelope(usize::MAX, protocol_max - 1, 1, 1).is_err());
    }
}
