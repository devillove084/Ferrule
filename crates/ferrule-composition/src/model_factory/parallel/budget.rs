//! Provider representation and concurrent host/device budget leaves.
use super::*;
mod common;
mod local;
pub(in crate::model_factory) use common::*;
pub(in crate::model_factory) use local::*;
mod resident;
use super::{
    experts::OwnerPlan,
    plan::{Geometry, Plan},
};
use ferrule_model::transformer::HybridSegmentMemoryEstimate;
#[cfg(feature = "cuda")]
use ferrule_model::{decoder::HybridCudaMemoryEstimate, transformer::CudaHybridRankMemory};
pub use resident::*;
pub(super) fn grant(n: u64) -> Result<u64> {
    usize::try_from(n)
        .ok()
        .and_then(ferrule_backend::cuda_allocation_grant_bytes)
        .map(|n| n as u64)
        .filter(|n| *n <= isize::MAX as u64)
        .ok_or_else(|| invalid("parallel allocation grant overflow"))
}
pub(super) fn wire_bytes(
    rows: usize,
    hidden: usize,
    vocab: usize,
    owners: &[OwnerPlan],
    roots: usize,
) -> Result<u64> {
    if [rows, hidden, vocab, roots].contains(&0) {
        return Err(invalid("empty parallel wire shape"));
    }
    let boundary = mul(mul(rows as u64, hidden.max(vocab) as u64)?, 4)?;
    owners
        .iter()
        .try_fold(mul(boundary, mul(8, roots as u64)?)?, |sum, owner| {
            add(sum, mul(owner.group.limits.max_bytes as u64, 8)?)
        })
}
pub(super) fn collective_limits(
    layout: &HybridExpertTopology,
    segments: &[LayerSegmentPlan],
    config: PipelineConfig,
) -> Result<HostCollectiveLimits> {
    let tp = layout.stages()[0].tensor_members().len();
    let mut elements = 0;
    for (plan, segment) in layout.stages().iter().zip(segments) {
        for site in ferrule_runtime::parallel::pipeline::translate_hybrid_schedule(plan, segment)? {
            elements =
                elements.max(site.elements(config.session_capacity, config.max_batch_tokens)?);
        }
    }
    let bytes = mul(mul(elements as u64, 4)?, mul(tp as u64, tp as u64 + 1)?)?;
    Ok(HostCollectiveLimits {
        max_ranks: tp,
        max_elements_per_rank: elements,
        max_host_bytes: bytes as usize,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn host_options(
    resources: &BoundDecoderResources,
    mut host: HostExpertCacheOptions,
    geometry: &Geometry,
    config: PipelineConfig,
    kv: ResidentKvPagePlan,
    wire_bytes: u64,
    collective: HostCollectiveLimits,
    tensor: bool,
) -> Result<HostExpertCacheOptions> {
    let base = StandardHostStartupMemory::for_resources(
        resources,
        config.max_positions,
        config.max_parameter_bytes,
    )?;
    let metadata = HybridSegmentMemoryEstimate::binding_metadata_bytes(resources)? as u64;
    let owners = geometry.devices.len() as u64;
    host.memory_budget.base_resident_bytes = [
        mul(base.resident_bytes()?, geometry.roots as u64)?,
        mul(metadata, owners + 1)?,
        wire_bytes,
        if tensor {
            mul(
                collective.max_host_bytes as u64,
                2 * geometry.segments.len() as u64,
            )?
        } else {
            0
        },
        mul(owners, 2 << 20)?,
        mul(
            mul(
                kv.configured_pages as u64,
                if tensor { owners + 1 } else { 3 },
            )?,
            4096,
        )?,
        mul(
            mul(
                config.session_capacity as u64,
                if tensor { owners } else { 1 },
            )?,
            16384,
        )?,
    ]
    .into_iter()
    .try_fold(0, add)?;
    host.memory_budget.base_already_resident_bytes = 0;
    host.memory_budget.base_temporary_bytes = mul(base.temporary_bytes(), geometry.roots as u64)?;
    HostExpertWarmPlan::new(resources.state_dict(), host, config.max_parameter_bytes)?;
    Ok(host)
}
/// Payload envelopes and allocator grants are distinct admission contracts.
/// Neither changes the payload/cache limits passed to expert startup.
#[derive(Clone, Copy, Debug)]
pub(super) enum ExpertBudgetRepresentation {
    PayloadEnvelope,
    PhysicalAllocationGrants,
}

pub(super) fn expert_budgets(
    resources: &BoundDecoderResources,
    plan: &Plan,
    representation: ExpertBudgetRepresentation,
) -> Result<Vec<HybridStageBudget>> {
    let mut result = Vec::new();
    for owner in &plan.owners {
        for (slot, &member) in owner.group.members.iter().enumerate() {
            let (weights, workspace) = match representation {
                ExpertBudgetRepresentation::PayloadEnvelope => (
                    owner.weights[slot] as u64,
                    expert_payload_envelope(0, owner.activation_bytes)? as u64,
                ),
                ExpertBudgetRepresentation::PhysicalAllocationGrants => {
                    use ferrule_model::nn::ParameterResidency;
                    let mut weights = 0;
                    for parameter in resources.state_dict().parameters() {
                        if let ParameterResidency::Expert { layer, expert } = parameter.residency()
                            && owner.group.layers.contains(layer)
                            && owner.group.placement.owner(*layer, *expert) == Some(member)
                        {
                            if parameter.numeric_fp8_encoding().is_none() {
                                return Err(invalid(
                                    "numeric experts require numeric E4M3 with paired scales",
                                ));
                            }
                            weights = add(weights, grant(parameter.weight().slice().bytes)?)?;
                            weights = add(
                                weights,
                                grant(
                                    parameter
                                        .scale()
                                        .ok_or_else(|| invalid("missing expert scale"))?
                                        .slice()
                                        .bytes,
                                )?,
                            )?;
                        }
                    }
                    // Independently rounded roots include growth/alias holds;
                    // payload cache counters cannot substitute for these grants.
                    let workspace = add(
                        grant(EXPERT_WORKSPACE as u64)?,
                        mul(grant(owner.activation_bytes as u64)?, 16)?,
                    )?;
                    (weights, workspace)
                }
            };
            result.push(HybridStageBudget {
                device: plan.devices[member.get() as usize],
                weights,
                state: 0,
                kv: 0,
                workspace,
                margin: MARGIN,
            });
        }
    }
    Ok(result)
}

#[cfg(feature = "cuda")]
fn segment_budgets(
    resources: &BoundDecoderResources,
    plan: &Plan,
    options: &GenericDecoderOptions,
) -> Result<Vec<HybridStageBudget>> {
    let config = plan.config();
    // Model currently exports a full external-root estimator, not a routed
    // segment estimator. Charge that conservative weight/workspace envelope
    // on EACH root; exact stage schema still determines state and KV. Never
    // fabricate a partial DecoderModelSpec or rewrite global expert identity.
    let envelope = HybridCudaMemoryEstimate::for_resources_with_routed_experts(
        resources,
        options,
        config.physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
    )?;
    let mut budgets = Vec::new();
    for (stage, segment) in plan.segments.iter().enumerate() {
        ferrule_model::transformer::CudaHybridDecoderSegment::validate_routed_preparation(
            resources, segment, options,
        )?;
        budgets.push(
            plan.segment_layout
                .as_ref()
                .expect("segment layout")
                .stage_budget(
                    stage,
                    config,
                    plan.devices[stage],
                    envelope.weight_bytes_upper_bound as u64,
                    envelope.workspace_bytes_upper_bound as u64,
                    MARGIN,
                    ModelExecutionBackend::Cuda,
                )?,
        );
    }
    if Some(budgets.iter().try_fold(0, |sum, b| add(sum, b.kv))?) != plan.kv.configured_bytes {
        return Err(invalid(
            "segment compact KV placement differs from admission",
        ));
    }
    budgets.extend(expert_budgets(
        resources,
        plan,
        ExpertBudgetRepresentation::PayloadEnvelope,
    )?);
    Ok(budgets)
}
#[cfg(feature = "cuda")]
fn tensor_budgets(
    resources: &BoundDecoderResources,
    plan: &mut Plan,
    options: &GenericDecoderOptions,
) -> Result<Vec<HybridStageBudget>> {
    let config = plan.config();
    let mut budgets = Vec::new();
    let mut pinned = 0;
    let mut boundary_envelope = 0;
    for (stage, parallel) in plan.layout.stages().iter().enumerate() {
        for lane in 0..parallel.tensor_members().len() {
            let memory = CudaHybridRankMemory::for_rank(
                resources,
                &plan.segments[stage],
                parallel,
                &parallel.rank_layout(lane)?,
                options,
                config.physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
            )?;
            memory.total_device_grants(config.session_capacity)?;
            if memory.max_collective_elements > plan.collective.max_elements_per_rank {
                return Err(invalid(
                    "model collective shape exceeds sealed wire admission",
                ));
            }
            pinned = add(pinned, memory.pinned_host_bytes as u64)?;
            boundary_envelope = boundary_envelope.max(memory.pinned_host_bytes as u64);
            let state = mul(
                mul(config.session_capacity as u64, 2)?
                    .checked_add(1)
                    .ok_or_else(|| invalid("state count overflow"))?,
                memory.per_sequence_state_grants as u64,
            )?;
            budgets.push(HybridStageBudget {
                device: plan.devices[budgets.len()],
                weights: memory.weight_grants as u64,
                state,
                kv: memory.physical_kv_grants as u64,
                workspace: add(
                    memory.numeric_workspace_grants as u64,
                    memory.operation_workspace_grants as u64,
                )?,
                margin: MARGIN,
            });
        }
    }
    // Child boundaries and source result adapters are additional to model TP.
    // The model exports a two-boundary envelope; conservatively charge that
    // full envelope again to every root/child rather than duplicate private slot constants.
    pinned = add(pinned, mul(boundary_envelope, plan.devices.len() as u64)?)?;
    plan.charge_pinned_host(pinned)?;
    budgets.extend(expert_budgets(
        resources,
        plan,
        ExpertBudgetRepresentation::PhysicalAllocationGrants,
    )?);
    HostExpertWarmPlan::new(
        resources.state_dict(),
        plan.host(),
        config.max_parameter_bytes,
    )?;
    Ok(budgets)
}
#[cfg(feature = "cuda")]
pub(in crate::model_factory) struct Admission {
    pub(in crate::model_factory) plan: Plan,
    pub(in crate::model_factory) options: GenericDecoderOptions,
    pub(in crate::model_factory) budgets: Vec<HybridStageBudget>,
}
#[cfg(feature = "cuda")]
impl Admission {
    pub(in crate::model_factory) fn new(
        resources: &BoundDecoderResources,
        mut plan: Plan,
    ) -> Result<Self> {
        let options = plan.decoder_options(resources)?;
        let budgets = if plan.segment_layout.is_some() {
            segment_budgets(resources, &plan, &options)?
        } else {
            tensor_budgets(resources, &mut plan, &options)?
        };
        Ok(Self {
            plan,
            options,
            budgets,
        })
    }
    pub(in crate::model_factory) fn admit_live(
        &self,
        resources: &BoundDecoderResources,
    ) -> Result<()> {
        HostExpertWarmPlan::new(
            resources.state_dict(),
            self.plan.host(),
            self.plan.config().max_parameter_bytes,
        )?
        .check_memory_snapshot(super::super::host_memory::snapshot()?)?;
        tracing::info!(budgets=?self.budgets, wire_bytes=self.plan.wire_bytes, profile=?self.plan.profile(), "aggregate parallel admission before warm/uploads");
        admit_devices(&self.budgets)
    }
}
#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn admit_devices(budgets: &[HybridStageBudget]) -> Result<()> {
    let mut capacities = Vec::new();
    for device in budgets
        .iter()
        .map(|b| b.device)
        .collect::<std::collections::BTreeSet<_>>()
    {
        let available = std::thread::spawn(move || -> ferrule_common::Result<u64> {
            let ops =
                ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(device)?;
            let (free, total) = ops.memory_info()?;
            Ok(free.min(total) as u64)
        })
        .join()
        .map_err(|_| invalid("parallel capacity probe panicked"))??;
        capacities.push((device, available));
    }
    HybridDeviceBudget::admit(budgets, &capacities)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/parallel_expert_budget.rs"]
mod tests;
