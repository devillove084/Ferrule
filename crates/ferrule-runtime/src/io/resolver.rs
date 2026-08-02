//! Model-facing exact-key resolver backed by the shared physical provider.
//!
//! The resolver owns no logical demand, waiter, publication, residency, lease,
//! cancellation, or eviction state. Those lifetimes belong exclusively to
//! [`super::registry::LoadRegistry`].

use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::{
    MaterializationKey, MaterializationPurpose, MaterializationResolveError,
    MaterializationResolveResult,
};
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

    pub fn prepare_prefetch(
        &self,
        request: MaterializationRequest,
    ) -> MaterializationResolveResult<MaterializationKey> {
        self.prepare(request, MaterializationPurpose::Prefetch)
    }

    fn prepare(
        &self,
        request: MaterializationRequest,
        purpose: MaterializationPurpose,
    ) -> MaterializationResolveResult<MaterializationKey> {
        let provider = {
            let state = self.lock();
            let placement = state.placement;
            if request.model() != placement.model()
                || request.backend() != placement.backend()
                || request.device() != placement.device()
            {
                return Err(MaterializationResolveError::PlacementMismatch {
                    purpose,
                    request_model: request.model(),
                    request_backend: request.backend(),
                    request_device: request.device(),
                    resolver_model: placement.model(),
                    resolver_backend: placement.backend(),
                    resolver_device: placement.device(),
                });
            }
            state
                .provider
                .clone()
                .ok_or(MaterializationResolveError::ProviderUnavailable {
                    purpose,
                    model: placement.model(),
                    backend: placement.backend(),
                    device: placement.device(),
                })?
        };
        provider
            .prepare(request, purpose)
            .map(|preparation| preparation.key())
            .map_err(|source| MaterializationResolveError::Provider { purpose, source })
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
    ) -> MaterializationResolveResult<MaterializationKey> {
        {
            let mut state = self.lock();
            state.stats.resolves = state.stats.resolves.saturating_add(1);
        }
        self.prepare(request, MaterializationPurpose::Execution)
    }
}
