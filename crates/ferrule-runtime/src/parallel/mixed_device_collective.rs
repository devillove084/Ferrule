//! Typed metadata participation on the existing endpoint and rendezvous loop.
//! No data ownership, consumer-fence evidence, or production enablement.
use super::*;
use crate::parallel::pipeline::tensor_scope::TensorCollectiveShape;
use ferrule_common::error::ErrorChain;
use ferrule_model::decoder::CompletionUnknown;
use ferrule_model::transformer::hybrid_parallel::{
    HybridCollectiveKind, HybridCollectiveSemantic, HybridCollectiveShape, HybridCollectiveSite,
};

pub(super) fn supported_site(site: &HybridCollectiveSite) -> bool {
    site.kind == HybridCollectiveKind::ReduceSum
        && site.global_layer.is_some()
        && matches!(
            site.semantic,
            HybridCollectiveSemantic::AttentionOutput
                | HybridCollectiveSemantic::RoutedExpertOutput
        )
}

fn canonical_site_matches(site: &HybridCollectiveSite, expected: &TensorCollectiveSite) -> bool {
    let TensorCollectiveOperator::Protocol {
        operation,
        kind,
        shape,
        ..
    } = &expected.operator
    else {
        return false;
    };
    let semantic = match operation {
        TensorCollectiveSemantic::AttentionOutput => HybridCollectiveSemantic::AttentionOutput,
        TensorCollectiveSemantic::RoutedExpertOutput => {
            HybridCollectiveSemantic::RoutedExpertOutput
        }
        _ => return false,
    };
    let (fixed, per_sequence, per_row) = match *shape {
        TensorCollectiveShape::Fixed(n) => (n, 0, 0),
        TensorCollectiveShape::PerRow(n) => (0, 0, n),
        TensorCollectiveShape::Rows { fixed, per_row } => (fixed, 0, per_row),
        TensorCollectiveShape::PerSequence {
            fixed,
            per_sequence,
            per_row,
        } => (fixed, per_sequence, per_row),
    };
    supported_site(site)
        && site.ordinal == expected.site
        && site.global_layer == expected.layer
        && site.semantic == semantic
        && *kind == HostCollectiveKind::AllReduceSumBf16
        && site.shape
            == (HybridCollectiveShape {
                fixed,
                per_sequence,
                per_row,
            })
}

impl MixedRankCollective {
    fn device_protocol_error(&self, source: Error, enqueued: bool) -> Error {
        let mut state = self
            .endpoint
            .shared
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        state.device_unknown |= enqueued
            || ErrorChain::new(&source)
                .downcast_ref::<CompletionUnknown>()
                .is_some();
        fail(&mut state, source.to_string());
        self.endpoint.shared.wake.notify_all();
        let shared = error(state.failure.as_deref().expect("failed device protocol"));
        if state.device_unknown {
            CompletionUnknown::wrap(Error::with_cleanup(
                "Mixed device protocol",
                shared,
                Err(source),
            ))
        } else {
            shared
        }
    }

    fn authenticate_device_identity(
        &self,
        state: &mut State,
        identity: &DevicePayloadIdentity,
    ) -> Result<()> {
        check_open(state)?;
        let index = self.endpoint.local;
        let packed = state
            .identity
            .as_ref()
            .ok_or_else(|| error("device vote lacks sealed batch"))?;
        // sealed_binding acquires this mutex itself; use the authenticated plan.
        if state.failure.is_some()
            || !self.device_participation
            || state.mixed_plan.as_deref() != Some(self.plan.as_ref())
            || !state.registered[index]
            || !state.validated[index]
            || self.plan.members.len() != 2
            || self.plan.max_rows > 8
            || packed.binding().participants().len() != 2
            || state.execution.as_ref().is_none_or(|(tx, flag)| {
                *tx != identity.transaction || flag.load(Ordering::Acquire)
            })
            || identity.epoch != self.plan.epoch
            || identity.owner != self.owner()
            || identity.members.as_slice() != self.plan.members.as_slice()
            || identity.rows == 0
            || identity.rows > self.plan.max_rows
            || identity.rows != packed.rows()
            || identity.sequences > self.plan.max_sequences
            || identity.sequences != packed.sequences()
            || self
                .plan
                .sites
                .get(state.completed[index])
                .is_none_or(|expected| !canonical_site_matches(&identity.site, expected))
            || identity.elements == 0
            || identity.elements
                != identity
                    .site
                    .shape
                    .elements(identity.sequences, identity.rows)?
        {
            return Err(error(
                "device vote differs from canonical sealed execution/site",
            ));
        }
        check_data_route(state, index, identity.site.ordinal)?;
        Ok(())
    }

    pub(super) fn participate_device_inner(
        &mut self,
        identity: &DevicePayloadIdentity,
    ) -> Result<()> {
        let result = (|| {
            {
                let mut state = self
                    .endpoint
                    .shared
                    .state
                    .lock()
                    .map_err(|_| error("collective mutex poisoned"))?;
                self.authenticate_device_identity(&mut state, identity)?;
                if state.device_permissions[self.endpoint.local].is_some() {
                    return Err(error("duplicate device participation"));
                }
            }
            // Identity stays in typed slots. A zero-sized token only drives the
            // existing sequence/Ready/drain synchronization; it allocates no data
            // buffers and cannot agree with a nonzero host data contribution.
            let payload = self.endpoint.exchange_inner(
                identity.transaction,
                identity.site.ordinal,
                HostCollectiveKind::AllReduceSumBf16,
                CollectivePayload::Bf16(Vec::new()),
                Some(identity),
            )?;
            if !matches!(payload, CollectivePayload::Bf16(ref words) if words.is_empty()) {
                return Err(error("device participation returned data storage"));
            }
            Ok(())
        })();
        result.map_err(|source| self.device_protocol_error(source, false))
    }

    pub(super) fn check_device_payload_permit_inner(
        &mut self,
        permit: &DevicePayloadPermit,
    ) -> Result<()> {
        let result = (|| {
            let mut state = self
                .endpoint
                .shared
                .state
                .lock()
                .map_err(|_| error("collective mutex poisoned"))?;
            self.authenticate_device_identity(&mut state, permit.identity())?;
            if state.device_permissions[self.endpoint.local].as_ref() != Some(permit.identity()) {
                return Err(error("device permit is not the issued pending site"));
            }
            Ok(())
        })();
        result.map_err(|source| self.device_protocol_error(source, false))
    }

    pub(super) fn device_payload_enqueued_inner(
        &mut self,
        permit: DevicePayloadPermit,
    ) -> Result<()> {
        let result = (|| {
            let mut state = self
                .endpoint
                .shared
                .state
                .lock()
                .map_err(|_| error("collective mutex poisoned"))?;
            let index = self.endpoint.local;
            // This call is post-enqueue, even if its acknowledgement is invalid.
            state.device_uncompleted[index] = true;
            self.authenticate_device_identity(&mut state, permit.identity())?;
            if state.device_permissions[index].as_ref() != Some(permit.identity()) {
                return Err(error("device enqueue lacks its issued permit"));
            }
            state.device_permissions[index] = None;
            state.completed[index] += 1;
            self.endpoint.shared.wake.notify_all();
            Ok(())
        })();
        result.map_err(|source| self.device_protocol_error(source, true))
    }

    pub(super) fn complete_device_payloads_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        completion: Result<()>,
    ) -> Result<()> {
        let result = (|| {
            // Only the provider's actual consumer fence AND finite/async-error
            // outcome belongs here. Metadata Ready is never such evidence.
            completion?;
            let mut state = self
                .endpoint
                .shared
                .state
                .lock()
                .map_err(|_| error("collective mutex poisoned"))?;
            check_open(&mut state)?;
            let index = self.endpoint.local;
            if state.failure.is_some()
                || !self.device_participation
                || !state.validated[index]
                || state
                    .execution
                    .as_ref()
                    .is_none_or(|(tx, flag)| *tx != transaction || flag.load(Ordering::Acquire))
                || state.device_permissions[index].is_some()
                || !state.device_uncompleted[index]
                || state
                    .sites
                    .as_ref()
                    .is_none_or(|sites| state.completed[index] != sites.len())
            {
                return Err(error(
                    "device invocation completion is missing or premature",
                ));
            }
            state.device_uncompleted[index] = false;
            self.endpoint.shared.wake.notify_all();
            Ok(())
        })();
        result.map_err(|source| self.device_protocol_error(source, true))
    }
}

#[cfg(test)]
#[path = "mixed_device_collective_tests.rs"]
mod tests;
