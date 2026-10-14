//! Metadata-only physical accounting. Model estimates, not storage subtotals,
//! supply operation/auxiliary/provider claims. No CUDA discovery or allocation.
use super::plan::{ParallelExecutionPlan, ParallelExpertMode, ParallelRankCollectiveClaim};
use super::{add, invalid};
use crate::Result;
use ferrule_backend::cuda_plan::{bf16_attention::Bf16AttentionPlan, bf16_gemmex::Bf16GemmExPlan};
use ferrule_common::ParallelRankId;
use ferrule_model::ModelExecutionBackend;
use ferrule_model::transformer::BoundDecoderResources;
use ferrule_model::transformer::hybrid_parallel::{
    HybridRankLayout, HybridRankPrecision, HybridRankStorageBudget,
};
use ferrule_runtime::parallel::pipeline::{
    HybridDeviceBudget, HybridStageBudget, PhysicalKvLayout,
};
use ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelRankCapacity {
    pub page_size: usize,
    pub max_positions: usize,
    pub max_rows: usize,
    pub sessions: usize,
    /// Logical pages. Physical shadow/COW slots are derived by the existing runtime layout.
    pub logical_pages: usize,
}
impl ParallelRankCapacity {
    pub fn physical_pages(self) -> Result<usize> {
        Ok(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda)
            .physical_pages(self.logical_pages)?)
    }
    pub fn state_copies(self) -> Result<usize> {
        self.sessions
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("rank state copies overflow"))
    }
    fn validate(self) -> Result<()> {
        if self.page_size == 0
            || self.max_positions == 0
            || self.max_rows == 0
            || self.sessions == 0
            || self.sessions > self.max_rows
        {
            return Err(invalid("invalid parallel rank resource capacities"));
        }
        self.physical_pages()?;
        Ok(())
    }
}

/// Already-rounded caller storage plus distinct external-library reserves.
/// The same sequential provider family is pooled by maximum, NEVER layer sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelProviderPools {
    gemmex_storage_grants: u64,
    gemmex_external_reserve: u64,
    attention_storage_grants: u64,
    attention_external_reserve: u64,
}
impl ParallelProviderPools {
    pub fn attention_external_reserve(self) -> u64 {
        self.attention_external_reserve
    }
    pub fn attention_storage_grants(self) -> u64 {
        self.attention_storage_grants
    }
    pub fn gemmex_external_reserve(self) -> u64 {
        self.gemmex_external_reserve
    }
    pub fn gemmex_storage_grants(self) -> u64 {
        self.gemmex_storage_grants
    }
    /// Consume actual operator plans (no upload or provider initialization).
    /// These are resource facts only, not evidence for the model's arithmetic schedule.
    pub fn from_plans(gemmex: &[Bf16GemmExPlan], attention: &[Bf16AttentionPlan]) -> Result<Self> {
        if gemmex.is_empty() || attention.is_empty() {
            return Err(invalid(
                "Mixed rank requires both projection and attention provider estimates",
            ));
        }
        let pools = Self {
            gemmex_storage_grants: gemmex
                .iter()
                .map(|p| p.storage_grant_bytes() as u64)
                .max()
                .unwrap(),
            gemmex_external_reserve: gemmex
                .iter()
                .map(|p| p.provider_bytes() as u64)
                .max()
                .unwrap(),
            attention_storage_grants: attention
                .iter()
                .map(|p| p.storage_grant_bytes() as u64)
                .max()
                .unwrap(),
            attention_external_reserve: attention
                .iter()
                .map(|p| p.provider_bytes() as u64)
                .max()
                .unwrap(),
        };
        pools.total()?;
        Ok(pools)
    }
    pub fn total(self) -> Result<u64> {
        sum([
            self.gemmex_storage_grants,
            self.gemmex_external_reserve,
            self.attention_storage_grants,
            self.attention_external_reserve,
        ])
    }
}

/// Model-estimated roots NOT already in HybridRankStorageBudget. Native norm,
/// router and GDN vector weights belong to storage.parameters and must not be
/// charged twice. Derived RoPE and attention metadata have separate lifetimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelAuxiliaryRequirements {
    pub derived_rope_grants: u64,
    pub attention_metadata_grants: u64,
    pub recurrent_workspace_grants: u64,
    pub other_persistent_grants: u64,
}
impl ParallelAuxiliaryRequirements {
    pub fn total(self) -> Result<u64> {
        sum([
            self.derived_rope_grants,
            self.attention_metadata_grants,
            self.recurrent_workspace_grants,
            self.other_persistent_grants,
        ])
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelBoundaryRequirements {
    /// Additional pinned allocation, not an alias of pageable typed host vectors.
    pub pinned_host_bytes: u64,
    pub host_metadata_bytes: u64,
    /// Actual rounded device boundary/staging allocations; no implicit F32 mirror.
    pub device_grants: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelOwnerMargins {
    pub context_bytes: u64,
    pub allocator_bytes: u64,
}

/// Callback output bound to the exact model rank and capacities. None means
/// unknown, never zero. In particular storage-only plans cannot admit a device.
#[derive(Debug, Clone)]
pub struct ParallelOperatorEstimate {
    layout: HybridRankLayout,
    capacity: ParallelRankCapacity,
    pub providers: Option<ParallelProviderPools>,
    pub auxiliary: Option<ParallelAuxiliaryRequirements>,
    /// Simultaneously live activation/routing/reduction/operator roots, already
    /// allocation-rounded by the model estimator; excludes every category above.
    pub operation_grants: Option<u64>,
    pub boundary: Option<ParallelBoundaryRequirements>,
}
impl ParallelOperatorEstimate {
    /// Consume the model's pure full-program estimate, not a reconstructed TP1
    /// envelope. Storage and copy workspace are checked, then charged only by
    /// the existing storage ledger. The estimate is not an execution validator.
    pub fn from_program(
        resources: &BoundDecoderResources,
        program: &ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan,
        capacity: ParallelRankCapacity,
        storage: &HybridRankStorageBudget,
    ) -> Result<Self> {
        capacity.validate()?;
        let binding = program.collective_binding();
        if binding.max_rows != capacity.max_rows
            || binding.max_sequences != capacity.sessions
            || program.page_size() != capacity.page_size
            || program.max_positions() != capacity.max_positions
            || program.state_copies() != capacity.state_copies()?
        {
            return Err(invalid("model program differs from resource capacities"));
        }
        program.validate_spec(resources.spec())?;
        let estimate = program.operation_estimate(resources, capacity.physical_pages()?)?;
        if storage.precision != program.layout().precision()
            || storage.physical_pages != capacity.physical_pages()?
            || storage.state.copies != capacity.state_copies()?
            || estimate.storage_grants != storage.storage_grants()?
            || estimate.state_copy_workspace_grants != storage.state.copy_workspace_grants
        {
            return Err(invalid(
                "model operation estimate differs from storage ledger",
            ));
        }
        estimate.total_device_grants()?;
        let providers = ParallelProviderPools {
            gemmex_storage_grants: estimate.gemmex_storage_grants as u64,
            gemmex_external_reserve: estimate.gemmex_provider_bytes as u64,
            attention_storage_grants: estimate.attention_storage_grants as u64,
            attention_external_reserve: estimate.attention_provider_bytes as u64,
        };
        providers.total()?;
        Ok(Self {
            layout: program.layout().clone(),
            capacity,
            providers: Some(providers),
            auxiliary: Some(ParallelAuxiliaryRequirements {
                derived_rope_grants: estimate.derived_rope_grants as u64,
                attention_metadata_grants: estimate.attention_metadata_grants as u64,
                recurrent_workspace_grants: estimate.recurrent_workspace_grants as u64,
                other_persistent_grants: 0,
            }),
            operation_grants: Some(estimate.operation_grants as u64),
            boundary: Some(ParallelBoundaryRequirements {
                pinned_host_bytes: estimate.pinned_host_bytes as u64,
                host_metadata_bytes: estimate.host_staging_bytes as u64,
                device_grants: estimate.boundary_device_grants as u64,
            }),
        })
    }

    pub fn pending(layout: &HybridRankLayout, capacity: ParallelRankCapacity) -> Self {
        Self {
            layout: layout.clone(),
            capacity,
            providers: None,
            auxiliary: None,
            operation_grants: None,
            boundary: None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelRequirementGap {
    ProviderPools,
    Auxiliary,
    OperationWorkspace,
    Boundary,
    TypedSchedule,
}

#[derive(Debug)]
pub struct ParallelRankRequirements {
    pub owner: ParallelRankId,
    pub device: usize,
    pub storage: HybridRankStorageBudget,
    pub operators: ParallelOperatorEstimate,
    pub typed_host: Option<ParallelRankCollectiveClaim>,
    pub margins: ParallelOwnerMargins,
}
impl ParallelRankRequirements {
    pub fn missing(&self) -> Vec<ParallelRequirementGap> {
        [
            (
                self.operators.providers.is_none(),
                ParallelRequirementGap::ProviderPools,
            ),
            (
                self.operators.auxiliary.is_none(),
                ParallelRequirementGap::Auxiliary,
            ),
            (
                self.operators.operation_grants.is_none(),
                ParallelRequirementGap::OperationWorkspace,
            ),
            (
                self.operators.boundary.is_none(),
                ParallelRequirementGap::Boundary,
            ),
            (
                self.typed_host.is_none(),
                ParallelRequirementGap::TypedSchedule,
            ),
        ]
        .into_iter()
        .filter_map(|(missing, gap)| missing.then_some(gap))
        .collect()
    }
    /// Lower bound while missing() is nonempty. Never label this a full device budget.
    pub fn known_device_bytes(&self) -> Result<u64> {
        self.stage_budget()?.total().map_err(Into::into)
    }
    pub fn device_bytes(&self) -> Result<Option<u64>> {
        if self.missing().is_empty() {
            self.known_device_bytes().map(Some)
        } else {
            Ok(None)
        }
    }
    pub(super) fn stage_budget(&self) -> Result<HybridStageBudget> {
        let workspace = sum([
            self.storage.state.copy_workspace_grants as u64,
            self.operators
                .providers
                .map(|p| p.total())
                .transpose()?
                .unwrap_or(0),
            self.operators
                .auxiliary
                .map(|p| p.total())
                .transpose()?
                .unwrap_or(0),
            self.operators.operation_grants.unwrap_or(0),
            self.operators.boundary.map_or(0, |b| b.device_grants),
        ])?;
        Ok(HybridStageBudget {
            device: self.device,
            weights: self.storage.weight_grants as u64,
            state: self.storage.state.total_grants as u64,
            kv: self.storage.kv_grants as u64,
            workspace,
            margin: add(self.margins.context_bytes, self.margins.allocator_bytes)?,
        })
    }
}

/// Source-bound report; resource completeness is NOT model execution support.
/// References prevent exchanging the model or placement after requirements are collected.
pub struct ParallelResourceRequirements<'a> {
    plan: &'a ParallelExecutionPlan,
    resources: &'a BoundDecoderResources,
    owners: Vec<ParallelRankRequirements>,
    typed_host_bytes: u64,
    model_estimated: bool,
}
impl ParallelResourceRequirements<'_> {
    pub fn owners(&self) -> &[ParallelRankRequirements] {
        &self.owners
    }
    pub fn resource_complete(&self) -> bool {
        self.owners.iter().all(|owner| owner.missing().is_empty())
    }
    pub fn known_host_bytes(&self) -> Result<u64> {
        self.owners
            .iter()
            .try_fold(self.typed_host_bytes, |n, owner| {
                let boundary = owner.operators.boundary;
                sum([
                    n,
                    boundary.map_or(0, |b| b.pinned_host_bytes),
                    boundary.map_or(0, |b| b.host_metadata_bytes),
                ])
            })
    }
    /// Aggregate every owner on its physical card using the existing neutral
    /// device-admission implementation. Even an incomplete lower bound can reject.
    pub fn check_known_capacity(
        &self,
        capacities: &[(usize, u64)],
        host_capacity: u64,
    ) -> Result<Vec<HybridDeviceBudget>> {
        if !self.resources.state_dict().validate_source_identities() {
            return Err(invalid("parallel resource source seal is stale"));
        }
        if self.known_host_bytes()? > host_capacity {
            return Err(invalid("parallel host/pinned resource capacity exceeded"));
        }
        Ok(HybridDeviceBudget::admit(
            &self
                .owners
                .iter()
                .map(ParallelRankRequirements::stage_budget)
                .collect::<Result<Vec<_>>>()?,
            capacities,
        )?)
    }
    pub fn admit(
        &self,
        capacities: &[(usize, u64)],
        host_capacity: u64,
    ) -> Result<Vec<HybridDeviceBudget>> {
        let devices = self.check_known_capacity(capacities, host_capacity)?;
        if !self.resource_complete() {
            return Err(invalid(
                "parallel resource ledger incomplete: model operator/auxiliary/boundary or typed schedule requirements missing",
            ));
        }
        Ok(devices)
    }
    /// Internal experimental assembly hook. Only the source-bound model estimator,
    /// complete typed schedule and exact two-device topology can enter the closure.
    /// CUDA owner preparation still requires the model claim; public quality and
    /// CLI/profile validators remain unchanged.
    #[doc(hidden)]
    pub fn prepare_internal<T>(
        &self,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        prepare: impl FnOnce(&[ParallelRankRequirements]) -> Result<T>,
    ) -> Result<T> {
        self.admit(capacities, host_capacity)?;
        let p = self.plan.topology().plan();
        if self.plan.contract().precision().is_mixed()
            && (p.pipeline_parallel != 1
                || p.tensor_parallel != 2
                || p.expert_parallel != 1
                || self.plan.devices().len() != 2
                || self.plan.devices()[0] == self.plan.devices()[1])
        {
            return Err(invalid(
                "internal Mixed owners require PP1 TP2 EP1 on two distinct devices",
            ));
        }
        if !self.model_estimated {
            return Err(invalid(
                "internal Mixed construction requires the bound model operation estimator",
            ));
        }
        // Only this internal, source-bound experimental route bypasses the
        // advertised full-model quality gate. Owner claim validation is mandatory.
        prepare(&self.owners)
    }
}
impl ParallelExecutionPlan {
    /// Full segment programs, including token input and the vocabulary gather.
    /// Model geometry is payload-free and is NOT an execution capability.
    pub fn mixed_programs(
        &self,
        capacity: ParallelRankCapacity,
    ) -> Result<Vec<ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan>> {
        use ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan;
        capacity.validate()?;
        if self.contract().precision() != HybridRankPrecision::MixedBf16F32
            || self.contract().experts() != ParallelExpertMode::LocalTensorShards
            || self.external_owner_count() != 0
        {
            return Err(invalid("Mixed programs require local tensor expert shards"));
        }
        self.layout()
            .bind_tensor_scopes(self.topology(), self.segments(), &[])?;
        let copies = capacity.state_copies()?;
        self.layout()
            .stages()
            .iter()
            .zip(self.segments())
            .flat_map(|(stage, segment)| {
                (0..stage.tensor_members().len()).map(move |lane| (stage, segment, lane))
            })
            .map(|(stage, segment, lane)| {
                Ok(HybridMixedRankProgramPlan::for_segment(
                    std::sync::Arc::new(stage.rank_layout(lane)?),
                    segment.clone(),
                    capacity.max_rows,
                    capacity.sessions,
                    capacity.max_positions,
                    capacity.page_size,
                    copies,
                )?)
            })
            .collect()
    }

    /// Use exactly the model program's data and control schedules. No legacy F32
    /// conversion, omitted output head, or implicit route-control allowance.
    pub fn mixed_collective_plans(
        &self,
        capacity: ParallelRankCapacity,
        limits: ferrule_runtime::parallel::collective::HostCollectiveLimits,
    ) -> Result<Vec<MixedRankCollectivePlan>> {
        use ferrule_model::transformer::hybrid_parallel::HYBRID_AGREEMENT_VERSION;
        let programs = self.mixed_programs(capacity)?;
        self.layout()
            .stages()
            .iter()
            .enumerate()
            .map(|(stage, rank_plan)| {
                let program = programs
                    .iter()
                    .find(|p| p.layout().owner() == rank_plan.tensor_members()[0])
                    .ok_or_else(|| invalid("missing source rank program"))?;
                Ok(MixedRankCollectivePlan::from_hybrid_sites(
                    rank_plan,
                    ferrule_common::ParallelGroupId::new(stage as u32 + 1),
                    rank_plan.tensor_members()[0],
                    HYBRID_AGREEMENT_VERSION,
                    &program.sites(),
                    limits,
                    capacity.sessions,
                    capacity.max_rows,
                )?
                .with_route_sites(program.route_sites())?)
            })
            .collect()
    }

    /// Explicit cache configuration admitted by the model's complete ledger.
    #[doc(hidden)]
    pub fn mixed_programs_with_decoded_weight_cache(
        &self,
        resources: &BoundDecoderResources,
        capacity: ParallelRankCapacity,
        margins: ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        requested_cache_grants: usize,
    ) -> Result<Vec<ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan>> {
        if self.devices().len() != 2 || self.devices()[0] == self.devices()[1] {
            return Err(invalid("decoded cache requires two distinct owner devices"));
        }
        let margin = usize::try_from(add(margins.context_bytes, margins.allocator_bytes)?)
            .map_err(|_| invalid("cache owner margins exceed usize"))?;
        self.mixed_programs(capacity)?
            .into_iter()
            .enumerate()
            .map(|(lane, program)| {
                let device = self.devices()[lane];
                let mut matching = capacities.iter().filter(|(d, _)| *d == device);
                let (_, bytes) = matching
                    .next()
                    .ok_or_else(|| invalid("missing cache owner capacity"))?;
                if matching.next().is_some() {
                    return Err(invalid("duplicate cache owner capacity"));
                }
                let bytes = usize::try_from(*bytes)
                    .map_err(|_| invalid("cache owner capacity exceeds usize"))?;
                Ok(program.with_decoded_weight_cache(
                    resources,
                    capacity.physical_pages()?,
                    requested_cache_grants,
                    bytes,
                    margin,
                )?)
            })
            .collect()
    }

    /// Production estimate adapter. All categories come from the bound model
    /// program; unsupported geometry remains an error, never a zero allowance.
    pub fn resource_requirements_from_model<'a>(
        &'a self,
        resources: &'a BoundDecoderResources,
        capacity: ParallelRankCapacity,
        margins: ParallelOwnerMargins,
        collectives: &[MixedRankCollectivePlan],
    ) -> Result<ParallelResourceRequirements<'a>> {
        let programs = self.mixed_programs(capacity)?;
        self.resource_requirements_from_programs(
            resources,
            capacity,
            margins,
            collectives,
            &programs,
        )
    }

    /// Consume the same immutable programs that will enter runtime preparation.
    /// Available without CUDA so catalog metadata cannot omit candidate grants.
    #[doc(hidden)]
    pub fn resource_requirements_from_programs<'a>(
        &'a self,
        resources: &'a BoundDecoderResources,
        capacity: ParallelRankCapacity,
        margins: ParallelOwnerMargins,
        collectives: &[MixedRankCollectivePlan],
        programs: &[ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan],
    ) -> Result<ParallelResourceRequirements<'a>> {
        if programs.len() != self.root_count() {
            return Err(invalid("model program count differs from owner roster"));
        }
        let mut report = self.resource_requirements(
            resources,
            capacity,
            margins,
            collectives,
            |layout, capacity, storage| {
                let program = programs
                    .iter()
                    .find(|p| p.layout() == layout)
                    .ok_or_else(|| invalid("missing sealed model rank program"))?;
                ParallelOperatorEstimate::from_program(resources, program, capacity, storage)
            },
        )?;
        report.model_estimated = true;
        Ok(report)
    }

    /// Estimate the very programs carried into owner preparation, including any
    /// explicitly admitted decoded cache. Reconstructing native programs here
    /// would silently drop the candidate's workspace charge.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn resource_requirements_from_mixed_inputs<'a>(
        &'a self,
        resources: &'a BoundDecoderResources,
        capacity: ParallelRankCapacity,
        margins: ParallelOwnerMargins,
        inputs: &ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan,
    ) -> Result<ParallelResourceRequirements<'a>> {
        let config = inputs.config();
        if capacity
            != (ParallelRankCapacity {
                page_size: config.page_size,
                max_positions: config.max_positions,
                max_rows: config.max_batch_tokens,
                sessions: config.session_capacity,
                logical_pages: config.max_pages,
            })
        {
            return Err(invalid(
                "Mixed resource capacities differ from execution inputs",
            ));
        }
        let programs = (0..self.root_count())
            .map(|lane| inputs.program(lane).cloned().map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        self.resource_requirements_from_programs(
            resources,
            capacity,
            margins,
            &[inputs.collective().as_ref().clone()],
            &programs,
        )
    }

    /// Header-only aggregate. The callback must use model-owned geometry and
    /// actual operator plans, never TP1 estimates divided by rank count. Returning
    /// pending requirements is supported and explicitly prevents admission.
    pub fn resource_requirements<'a>(
        &'a self,
        resources: &'a BoundDecoderResources,
        capacity: ParallelRankCapacity,
        margins: ParallelOwnerMargins,
        collectives: &[MixedRankCollectivePlan],
        mut estimate: impl FnMut(
            &HybridRankLayout,
            ParallelRankCapacity,
            &HybridRankStorageBudget,
        ) -> Result<ParallelOperatorEstimate>,
    ) -> Result<ParallelResourceRequirements<'a>> {
        capacity.validate()?;
        if self.contract().precision() != HybridRankPrecision::MixedBf16F32
            || self.contract().experts() != ParallelExpertMode::LocalTensorShards
            || self.external_owner_count() != 0
            || margins.context_bytes == 0
            || margins.allocator_bytes == 0
        {
            return Err(invalid(
                "Mixed resource requirements need local TP2 roots and explicit context/allocator margins",
            ));
        }
        if !resources.state_dict().validate_source_identities() {
            return Err(invalid("parallel resource source seal is stale"));
        }
        if !collectives.is_empty() && collectives.len() != self.layout().stages().len() {
            return Err(invalid("typed schedule count differs from stage roster"));
        }
        let mut typed = Vec::new();
        let mut typed_host_bytes = 0;
        for (stage, collective) in collectives.iter().enumerate() {
            if collective.source() != self.layout().stages()[stage].tensor_members()[0]
                || collective.max_rows() != capacity.max_rows
                || collective.max_sequences() != capacity.sessions
            {
                return Err(invalid(
                    "typed schedule source/capacity differs from rank requirements",
                ));
            }
            // Resource completeness must not hide a missing route-control charge
            // or a partial data schedule. Derive the expected metadata from the
            // model's own program, without executing or materializing parameters.
            use ferrule_model::transformer::hybrid_parallel::{
                HYBRID_AGREEMENT_VERSION, HybridMixedRankProgramPlan,
            };
            let stage_plan = &self.layout().stages()[stage];
            let copies = capacity.state_copies()?;
            let model = HybridMixedRankProgramPlan::for_segment(
                std::sync::Arc::new(stage_plan.rank_layout(0)?),
                self.segments()[stage].clone(),
                capacity.max_rows,
                capacity.sessions,
                capacity.max_positions,
                capacity.page_size,
                copies,
            )?;
            let expected = MixedRankCollectivePlan::from_hybrid_sites(
                stage_plan,
                collective.group(),
                stage_plan.tensor_members()[0],
                HYBRID_AGREEMENT_VERSION,
                &model.sites(),
                collective.limits(),
                capacity.sessions,
                capacity.max_rows,
            )?
            .with_route_sites(model.route_sites())?;
            if collective != &expected {
                return Err(invalid(
                    "resource schedule differs from complete model data/route metadata",
                ));
            }
            let claims = self.collective_claims(stage, collective)?;
            typed_host_bytes = add(typed_host_bytes, claims.total_host_bytes as u64)?;
            typed.extend(claims.owners);
        }
        let mut owners = Vec::new();
        let claims = self.storage_claims(
            resources,
            capacity.page_size,
            capacity.max_positions,
            capacity.physical_pages()?,
            capacity.sessions,
        )?;
        for claim in claims {
            let stage = self
                .layout()
                .stages()
                .iter()
                .find(|stage| stage.tensor_members().contains(&claim.owner))
                .ok_or_else(|| invalid("resource owner outside rank roster"))?;
            let lane = stage
                .tensor_members()
                .iter()
                .position(|r| *r == claim.owner)
                .unwrap();
            let layout = stage.rank_layout(lane)?;
            let operators = estimate(&layout, capacity, &claim.storage)?;
            if operators.layout != layout || operators.capacity != capacity {
                return Err(invalid(
                    "model operator requirements differ from sealed rank/capacity",
                ));
            }
            let owner = ParallelRankRequirements {
                owner: claim.owner,
                device: claim.device,
                storage: claim.storage,
                operators,
                typed_host: typed.iter().find(|h| h.owner == claim.owner).cloned(),
                margins,
            };
            owner.known_device_bytes()?;
            owners.push(owner);
        }
        let report = ParallelResourceRequirements {
            plan: self,
            resources,
            owners,
            typed_host_bytes,
            model_estimated: false,
        };
        report.known_host_bytes()?;
        Ok(report)
    }
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0, add)
}
