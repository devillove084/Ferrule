//! Prepared BF16 rank communication over the existing decoder collective.
//! No model loop, GPU custody, KV vote, commit, or publication authority.
use super::*;
use ferrule_model::transformer::hybrid_parallel::{
    DataLocation, DevicePayloadIdentity, DevicePayloadPermit,
};
#[path = "mixed_device_collective.rs"]
mod device;
use ferrule_model::transformer::hybrid_parallel::{
    HybridMixedRouteAgreement, HybridMixedRouteSite, HybridMoeCombineOrder,
    HybridRoutedAccumulation,
};
#[path = "mixed_route_collective.rs"]
mod routes;
use crate::parallel::pipeline::tensor_scope::{TensorCollectiveOperator, TensorCollectiveSemantic};
pub(super) use routes::{PendingRoutes, check_data_route};

/// Complete payload allowance for the declared sequential host exchange pattern.
/// Metadata, pinned/device slots and provider allocations are separate resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixedCollectivePayloadBudget {
    pub service_peak_bytes: usize,
    pub input_bytes_per_owner: usize,
    pub output_bytes_per_owner: usize,
    /// Caller may retain one output and stage one input per owner. Retaining
    /// arbitrary historical outputs is outside this bounded exchange contract.
    pub staging_bytes_per_owner: usize,
    pub total_host_bytes: usize,
}

/// Typed route arrays are host control storage, never device/BF16 tensors.
/// Includes both caller records, both service copies and pending-state metadata;
/// sequential sites reuse the same bounded storage rather than accumulating by layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MixedRouteControlBudget {
    pub payload_bytes_per_owner: usize,
    pub service_payload_bytes: usize,
    pub caller_payload_bytes: usize,
    pub metadata_bytes: usize,
    pub total_host_bytes: usize,
}

/// Caller-supplied semantic schedule, not a translation of the F32 agreement
/// packet or a claim that a complete Mixed decoder is executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedRankCollectivePlan {
    epoch: ParallelTopologyId,
    group: ParallelGroupId,
    members: Vec<ParallelRankId>,
    source: ParallelRankId,
    sites: Vec<TensorCollectiveSite>,
    route_sites: Vec<HybridMixedRouteSite>,
    limits: HostCollectiveLimits,
    max_sequences: usize,
    max_rows: usize,
    budget: MixedCollectivePayloadBudget,
}
impl MixedRankCollectivePlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: ParallelTopologyId,
        group: ParallelGroupId,
        members: Vec<ParallelRankId>,
        source: ParallelRankId,
        sites: Vec<TensorCollectiveSite>,
        limits: HostCollectiveLimits,
        max_sequences: usize,
        max_rows: usize,
    ) -> Result<Self> {
        if epoch.get() == 0
            || group.get() == 0
            || members.len() != 2
            || !members.contains(&source)
            || sites.is_empty()
            || max_sequences == 0
            || max_rows == 0
            || max_sequences > max_rows
        {
            return Err(error("invalid Mixed rank collective identity or bounds"));
        }
        let service =
            HostCollectiveGroup::new(epoch, group, members.clone(), limits).map_err(error)?;
        let mut service_peak_bytes = 0;
        let mut output_bytes_per_owner = 0;
        let mut previous = None;
        let mut version = None;
        for site in &sites {
            site.validate(members.len())?;
            let TensorCollectiveOperator::Protocol {
                version: v,
                operation,
                kind,
                ..
            } = &site.operator
            else {
                return Err(error(
                    "Mixed rank requires explicit typed semantic sites, not F32 linear plans",
                ));
            };
            if *v == 0
                || version.is_some_and(|version| version != *v)
                || previous.is_some_and(|previous| previous >= site.site)
                || matches!(
                    operation,
                    TensorCollectiveSemantic::InputAgreement | TensorCollectiveSemantic::Frontier
                )
                || !matches!(
                    kind,
                    HostCollectiveKind::AllReduceSumBf16 | HostCollectiveKind::AllGatherBf16
                )
                || (matches!(
                    operation,
                    TensorCollectiveSemantic::OutputHead | TensorCollectiveSemantic::KvReplication
                ) && *kind != HostCollectiveKind::AllGatherBf16)
            {
                return Err(error(
                    "invalid Mixed precision/site order; legacy control packets are not BF16 activations",
                ));
            }
            previous = Some(site.site);
            version = Some(*v);
            let memory = service
                .memory_for(*kind, site.elements(max_sequences, max_rows)?)
                .map_err(error)?;
            service_peak_bytes = service_peak_bytes.max(memory.peak_bytes);
            output_bytes_per_owner =
                output_bytes_per_owner.max(memory.output_bytes / members.len());
        }
        let bytes = |a: usize, b: usize| {
            a.checked_add(b)
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| error("Mixed host payload allowance overflow"))
        };
        let input_bytes_per_owner = limits
            .max_elements_per_rank
            .checked_mul(2)
            .ok_or_else(|| error("Mixed input allowance overflow"))?;
        let staging_bytes_per_owner = bytes(input_bytes_per_owner, output_bytes_per_owner)?;
        let total_staging = staging_bytes_per_owner
            .checked_mul(members.len())
            .ok_or_else(|| error("Mixed staging allowance overflow"))?;
        let budget = MixedCollectivePayloadBudget {
            service_peak_bytes,
            input_bytes_per_owner,
            output_bytes_per_owner,
            staging_bytes_per_owner,
            total_host_bytes: bytes(service_peak_bytes, total_staging)?,
        };
        Ok(Self {
            epoch,
            group,
            members,
            source,
            sites,
            route_sites: Vec::new(),
            limits,
            max_sequences,
            max_rows,
            budget,
        })
    }
    /// Adapt explicit model-owned DATA sites, not the legacy schedule's F32
    /// agreement/completion packet. The caller must supply its actual block
    /// schedule; this method never infers layer math or MoE combine order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_hybrid_sites(
        rank_plan: &ferrule_model::transformer::hybrid_parallel::HybridParallelPlan,
        group: ParallelGroupId,
        source: ParallelRankId,
        version: u64,
        sites: &[ferrule_model::transformer::hybrid_parallel::HybridCollectiveSite],
        limits: HostCollectiveLimits,
        max_sequences: usize,
        max_rows: usize,
    ) -> Result<Self> {
        use crate::parallel::pipeline::tensor_scope::TensorCollectiveShape;
        use ferrule_model::transformer::hybrid_parallel::{
            HybridCollectiveKind as K, HybridCollectivePrecision, HybridCollectiveSemantic as S,
        };
        if rank_plan.precision().collective() != HybridCollectivePrecision::Bf16PartialsAndReduction
        {
            return Err(error(
                "Mixed collective requires model BF16 partial/reduction contract",
            ));
        }
        let translated = sites
            .iter()
            .map(|site| {
                if site
                    .global_layer
                    .is_some_and(|layer| !rank_plan.schema().layer_range().contains(&layer))
                {
                    return Err(error("Mixed collective site is outside owner layer scope"));
                }
                let operation = match site.semantic {
                    S::AttentionOutput => TensorCollectiveSemantic::AttentionOutput,
                    S::DenseMlpDown => TensorCollectiveSemantic::DenseMlpOutput,
                    S::SharedExpertDown => TensorCollectiveSemantic::SharedExpertOutput,
                    S::RoutedExpertOutput => TensorCollectiveSemantic::RoutedExpertOutput,
                    S::OutputHead => TensorCollectiveSemantic::OutputHead,
                    S::Agreement | S::Completion => {
                        return Err(error(
                            "legacy hybrid control packets cannot be reinterpreted as BF16",
                        ));
                    }
                };
                Ok(TensorCollectiveSite {
                    site: site.ordinal,
                    layer: site.global_layer,
                    operator: TensorCollectiveOperator::Protocol {
                        version,
                        operation,
                        kind: match site.kind {
                            K::ReduceSum => HostCollectiveKind::AllReduceSumBf16,
                            K::AllGather => HostCollectiveKind::AllGatherBf16,
                        },
                        shape: TensorCollectiveShape::PerSequence {
                            fixed: site.shape.fixed,
                            per_sequence: site.shape.per_sequence,
                            per_row: site.shape.per_row,
                        },
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(
            rank_plan.scope().topology,
            group,
            rank_plan.tensor_members().to_vec(),
            source,
            translated,
            limits,
            max_sequences,
            max_rows,
        )
    }

    /// Attach model-owned typed control sites before endpoint authentication.
    /// These events do not alter the BF16 data schedule or any IPC wire version.
    pub fn with_route_sites(mut self, sites: Vec<HybridMixedRouteSite>) -> Result<Self> {
        routes::validate_sites(&self, &sites)?;
        let old = self.route_control_budget()?.total_host_bytes;
        self.route_sites = sites;
        let new = self.route_control_budget()?.total_host_bytes;
        self.budget.total_host_bytes = self
            .budget
            .total_host_bytes
            .checked_sub(old)
            .and_then(|n| n.checked_add(new))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| error("Mixed route control budget overflow"))?;
        Ok(self)
    }
    pub fn route_sites(&self) -> &[HybridMixedRouteSite] {
        &self.route_sites
    }
    pub fn route_control_budget(&self) -> Result<MixedRouteControlBudget> {
        routes::budget(&self.route_sites, self.members.len())
    }

    pub fn epoch(&self) -> ParallelTopologyId {
        self.epoch
    }
    pub fn group(&self) -> ParallelGroupId {
        self.group
    }
    pub fn members(&self) -> &[ParallelRankId] {
        &self.members
    }
    pub fn source(&self) -> ParallelRankId {
        self.source
    }
    pub fn sites(&self) -> &[TensorCollectiveSite] {
        &self.sites
    }
    pub fn limits(&self) -> HostCollectiveLimits {
        self.limits
    }
    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub fn payload_budget(&self) -> MixedCollectivePayloadBudget {
        self.budget
    }
}

/// Non-cloneable owner capability. Intentionally does not implement the F32-only
/// StandardTensorCollective trait or expose a mutable/raw endpoint escape hatch.
/// Model code supplies completed BF16 host downloads and owns upload/fence custody.
pub struct MixedRankCollective {
    endpoint: DecoderTensorCollective,
    plan: Arc<MixedRankCollectivePlan>,
    // Kept private and disabled until provider quality and budgets are admitted.
    device_participation: bool,
}
impl MixedRankCollective {
    pub fn prepare(
        mut endpoint: DecoderTensorCollective,
        owner: ParallelRankId,
        plan: Arc<MixedRankCollectivePlan>,
    ) -> Result<Self> {
        let result = (|| {
            let state = endpoint
                .shared
                .state
                .lock()
                .map_err(|_| error("collective mutex poisoned"))?;
            if endpoint.owner() != owner
                || endpoint.epoch != plan.epoch
                || endpoint.group != plan.group
                || endpoint.members.as_ref() != plan.members.as_slice()
                || state.group.limits() != plan.limits
            {
                return Err(error(
                    "Mixed endpoint differs from sealed owner/epoch/group/capacity",
                ));
            }
            drop(state);
            endpoint.control().register_sites_inner(
                endpoint.owner(),
                plan.sites.clone(),
                Some(Arc::clone(&plan)),
            )
        })();
        if let Err(source) = result {
            endpoint.abort();
            return Err(source);
        }
        Ok(Self {
            endpoint,
            plan,
            device_participation: false,
        })
    }
    pub fn owner(&self) -> ParallelRankId {
        self.endpoint.owner()
    }
    pub fn members(&self) -> &[ParallelRankId] {
        self.plan.members()
    }
    pub fn source(&self) -> ParallelRankId {
        self.plan.source()
    }
    pub fn plan(&self) -> &MixedRankCollectivePlan {
        &self.plan
    }
    pub fn control(&self) -> DecoderTensorCollectiveControl {
        self.endpoint.control()
    }
    pub fn validate_owner_execution(&self, identity: &TensorBatchIdentity) -> Result<()> {
        self.control()
            .validate_owner_execution(self.owner(), identity)
    }
    /// Site lookup authenticates kind; callers cannot select different arithmetic.
    /// The existing coordinator authenticates order, transaction, packed shape,
    /// every owner, cancellation and sequence before consuming any input.
    pub fn exchange_bf16(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        words: Vec<u16>,
    ) -> Result<Vec<u16>> {
        let Some(entry) = self.plan.sites.iter().find(|entry| entry.site == site) else {
            self.endpoint.abort();
            return Err(error("unregistered Mixed collective site"));
        };
        match self.endpoint.exchange_payload(
            transaction,
            site,
            entry.kind(),
            CollectivePayload::Bf16(words),
        )? {
            CollectivePayload::Bf16(words) => Ok(words),
            _ => {
                self.endpoint.abort();
                Err(error("Mixed collective returned non-BF16 storage"))
            }
        }
    }
    /// Host schedule completion only, never a GPU fence or publication receipt.
    pub fn complete_owner(&self, transaction: ExecutionTransactionId) -> Result<()> {
        self.control()
            .check_owner_completion(self.owner(), transaction)
    }
    pub fn abort(&mut self) {
        self.endpoint.abort();
    }
}

impl ferrule_model::transformer::hybrid_parallel::HybridBf16Collective for MixedRankCollective {
    fn data_location(
        &self,
        site: &ferrule_model::transformer::hybrid_parallel::HybridCollectiveSite,
    ) -> DataLocation {
        if self.device_participation && device::supported_site(site) {
            DataLocation::Device
        } else {
            DataLocation::Host
        }
    }
    fn participate_device(&mut self, identity: &DevicePayloadIdentity) -> Result<()> {
        self.participate_device_inner(identity)
    }
    fn check_device_payload_permit(&mut self, permit: &DevicePayloadPermit) -> Result<()> {
        self.check_device_payload_permit_inner(permit)
    }
    fn device_payload_enqueued(&mut self, permit: DevicePayloadPermit) -> Result<()> {
        self.device_payload_enqueued_inner(permit)
    }
    fn complete_device_payloads(
        &mut self,
        transaction: ExecutionTransactionId,
        completion: Result<()>,
    ) -> Result<()> {
        self.complete_device_payloads_inner(transaction, completion)
    }

    fn owner(&self) -> ParallelRankId {
        MixedRankCollective::owner(self)
    }
    fn members(&self) -> &[ParallelRankId] {
        MixedRankCollective::members(self)
    }
    fn sealed_binding(
        &self,
    ) -> Result<ferrule_model::transformer::hybrid_parallel::HybridBf16CollectiveBinding> {
        use crate::parallel::pipeline::tensor_scope::TensorCollectiveShape as Shape;
        use ferrule_model::transformer::hybrid_parallel::{
            HybridBf16CollectiveBinding, HybridCollectiveKind as Kind, HybridCollectivePrecision,
            HybridCollectiveSemantic as Semantic, HybridCollectiveShape, HybridCollectiveSite,
        };
        // This is the plan authenticated by prepare/register_sites_inner, not a
        // model-provided expectation. Reading metadata neither begins execution
        // nor substitutes for every owner's packed-identity validation.
        let mut state = self
            .endpoint
            .shared
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        check_open(&mut state)?;
        if state.failure.is_some()
            || state.closed.iter().any(|closed| *closed)
            || !state.registered[self.endpoint.local]
            || state.mixed_plan.as_deref() != Some(self.plan.as_ref())
            || self.endpoint.members.as_ref() != self.plan.members.as_slice()
        {
            return Err(error(
                "Mixed binding requires its live authenticated owner plan",
            ));
        }
        let mut version = None;
        let sites = self
            .plan
            .sites
            .iter()
            .map(|site| {
                let TensorCollectiveOperator::Protocol {
                    version: current,
                    operation,
                    kind,
                    shape,
                } = &site.operator
                else {
                    return Err(error(
                        "Mixed binding cannot expose a legacy F32 linear site",
                    ));
                };
                if version.is_some_and(|version| version != *current) || *current == 0 {
                    return Err(error("Mixed binding has inconsistent protocol versions"));
                }
                version = Some(*current);
                let semantic = match operation {
                    TensorCollectiveSemantic::AttentionOutput => Semantic::AttentionOutput,
                    TensorCollectiveSemantic::DenseMlpOutput => Semantic::DenseMlpDown,
                    TensorCollectiveSemantic::SharedExpertOutput => Semantic::SharedExpertDown,
                    TensorCollectiveSemantic::RoutedExpertOutput => Semantic::RoutedExpertOutput,
                    TensorCollectiveSemantic::OutputHead => Semantic::OutputHead,
                    // The model has no exact representation for specialized KV/GQA/
                    // GDN sites. Do not silently relabel these or numeric control sites.
                    TensorCollectiveSemantic::GqaOutput
                    | TensorCollectiveSemantic::GatedDeltaOutput
                    | TensorCollectiveSemantic::KvReplication
                    | TensorCollectiveSemantic::InputAgreement
                    | TensorCollectiveSemantic::Frontier => {
                        return Err(error(
                            "Mixed binding semantic has no exact model data-site representation",
                        ));
                    }
                };
                let kind = match kind {
                    HostCollectiveKind::AllReduceSumBf16 => Kind::ReduceSum,
                    HostCollectiveKind::AllGatherBf16 => Kind::AllGather,
                    HostCollectiveKind::AllReduceSumF32
                    | HostCollectiveKind::AllGatherF32
                    | HostCollectiveKind::AllReduceSumF32ToBf16 => {
                        return Err(error(
                            "Mixed binding cannot advertise F32 payload arithmetic as BF16",
                        ));
                    }
                };
                let (fixed, per_sequence, per_row) = match *shape {
                    Shape::Fixed(n) => (n, 0, 0),
                    Shape::PerRow(n) => (0, 0, n),
                    Shape::Rows { fixed, per_row } => (fixed, 0, per_row),
                    Shape::PerSequence {
                        fixed,
                        per_sequence,
                        per_row,
                    } => (fixed, per_sequence, per_row),
                };
                Ok(HybridCollectiveSite {
                    ordinal: site.site,
                    global_layer: site.layer,
                    semantic,
                    kind,
                    shape: HybridCollectiveShape {
                        fixed,
                        per_sequence,
                        per_row,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(HybridBf16CollectiveBinding {
            epoch: self.plan.epoch,
            source: self.plan.source,
            members: self.plan.members.clone(),
            version: version.ok_or_else(|| error("Mixed binding has no data sites"))?,
            precision: HybridCollectivePrecision::Bf16PartialsAndReduction,
            sites,
            max_rows: self.plan.max_rows,
            max_sequences: self.plan.max_sequences,
        })
    }
    fn sealed_route_sites(&self) -> Result<Vec<HybridMixedRouteSite>> {
        self.sealed_binding()?;
        Ok(self.plan.route_sites.clone())
    }
    fn agree_routes(
        &mut self,
        transaction: ExecutionTransactionId,
        agreement: &HybridMixedRouteAgreement,
    ) -> Result<()> {
        let result = self.agree_routes_inner(transaction, agreement);
        if let Err(source) = &result {
            let mut state = self
                .endpoint
                .shared
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            fail(&mut state, source.to_string());
            self.endpoint.shared.wake.notify_all();
        }
        result
    }
    fn validate_execution(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &ferrule_model::decoder::PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        self.control()
            .check_owner_execution(self.owner(), transaction, batch, sessions)
    }
    fn complete_owner(&mut self, transaction: ExecutionTransactionId) -> Result<()> {
        MixedRankCollective::complete_owner(self, transaction)
    }
    fn exchange_bf16(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: ferrule_model::transformer::hybrid_parallel::HybridCollectiveKind,
        words: Vec<u16>,
    ) -> Result<Vec<u16>> {
        use ferrule_model::transformer::hybrid_parallel::HybridCollectiveKind;
        let expected = match kind {
            HybridCollectiveKind::ReduceSum => HostCollectiveKind::AllReduceSumBf16,
            HybridCollectiveKind::AllGather => HostCollectiveKind::AllGatherBf16,
        };
        if self
            .plan
            .sites
            .iter()
            .find(|entry| entry.site == site)
            .is_none_or(|entry| entry.kind() != expected)
        {
            self.abort();
            return Err(error(
                "model BF16 operation differs from sealed data site/kind; control sites are not numeric collectives",
            ));
        }
        MixedRankCollective::exchange_bf16(self, transaction, site, words)
    }
    fn abort(&mut self) {
        MixedRankCollective::abort(self);
    }
}

/// Closing an endpoint during a prepared execution loses owner completion
/// evidence, even if every host result has been drained. Keep the execution and
/// sequence watermarks intact; neither aborting host buffers nor waking waiters
/// authorizes device/KV reclamation.
pub(super) fn check_open(state: &mut State) -> Result<()> {
    if state.mixed_plan.is_some()
        && let Some((transaction, _)) = &state.execution
        && state.closed.iter().any(|closed| *closed)
    {
        let transaction = *transaction;
        fail(
            state,
            "Mixed collective owner disappeared before transaction completion".into(),
        );
        return Err(ferrule_model::decoder::CompletionUnknown::wrap(error(
            format!(
                "Mixed collective transaction {} lost owner completion evidence",
                transaction.get()
            ),
        )));
    }
    if state.device_unknown {
        return Err(ferrule_model::decoder::CompletionUnknown::wrap(error(
            state
                .failure
                .as_deref()
                .unwrap_or("device completion remains unknown"),
        )));
    }
    Ok(())
}
