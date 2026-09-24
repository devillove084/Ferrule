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

/// Resolve the physical KV capacity without constructing model or backend state.
pub fn plan_resident_kv_pages(
    schema: &dyn KvLayoutSchema,
    accounting: ResidentKvPageAccounting,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
) -> Result<ResidentKvPagePlan> {
    if driver_config.ctx_size == 0 {
        return Err(Error::InvalidRequest {
            message: "resident engine ctx_size must be greater than zero".into(),
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
    let plan =
        plan_resident_kv_pages(schema.as_ref(), accounting, scheduler_config, driver_config)?;
    let max_active_sequences = scheduler_config.max_active_sequences.max(1);
    let driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(max_active_sequences),
        scheduler_config,
        NonZeroU32::new(1).expect("top-k one is non-zero"),
        driver_config,
    )
    .try_with_page_manager(KvPageManager::new(schema, plan.configured_pages))?;

    Ok(Box::new(ResidentInferenceEngine::with_kv_page_plan(
        driver, plan,
    )))
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
