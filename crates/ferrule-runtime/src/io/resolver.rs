//! Model-facing exact-key resolver backed by the shared physical provider.
//!
//! The resolver owns no logical demand, waiter, publication, residency, lease,
//! cancellation, or eviction state. Those lifetimes belong exclusively to
//! [`super::registry::LoadRegistry`].

use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::{Error, Result};
use ferrule_model::{MaterializationPlacement, MaterializationRequest, MaterializationResolver};

use super::provider::SharedMaterializationProvider;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeMaterializationResolverStats {
    pub resolves: u64,
}

#[derive(Debug)]
struct ResolverState {
    placement: MaterializationPlacement,
    provider: Option<SharedMaterializationProvider>,
    stats: RuntimeMaterializationResolverStats,
}

/// Cloneable view installed in a model runner. Clones address the same serialized
/// provider and metrics but do not cache provider or registry state.
#[derive(Debug, Clone)]
pub struct RuntimeMaterializationResolver {
    state: Arc<Mutex<ResolverState>>,
}

impl RuntimeMaterializationResolver {
    pub fn new(
        placement: MaterializationPlacement,
        provider: Option<SharedMaterializationProvider>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(ResolverState {
                placement,
                provider,
                stats: RuntimeMaterializationResolverStats::default(),
            })),
        }
    }

    pub fn stats(&self) -> RuntimeMaterializationResolverStats {
        self.lock().stats
    }

    fn lock(&self) -> MutexGuard<'_, ResolverState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl MaterializationResolver for RuntimeMaterializationResolver {
    fn placement(&self) -> MaterializationPlacement {
        self.lock().placement
    }

    fn resolve(
        &mut self,
        request: MaterializationRequest,
    ) -> Result<ferrule_common::MaterializationKey> {
        let provider = {
            let mut state = self.lock();
            state.stats.resolves = state.stats.resolves.saturating_add(1);
            if request.model() != state.placement.model()
                || request.backend() != state.placement.backend()
                || request.device() != state.placement.device()
            {
                return Err(Error::Execution(
                    "materialization request does not match the runtime placement".into(),
                ));
            }
            state.provider.clone().ok_or_else(|| {
                Error::Execution(
                    "runner requested materialization without a physical provider".into(),
                )
            })?
        };
        provider
            .prepare(request)
            .map(|preparation| preparation.key())
            .map_err(|reason| {
                Error::Execution(format!(
                    "physical materialization preparation failed: {reason:?}"
                ))
            })
    }
}
