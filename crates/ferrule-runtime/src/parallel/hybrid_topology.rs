//! Payload-free, model-neutral PP/TP × EP topology validation.
//! Physical ranks are the union of stage TP and EP members, never `PP * TP * EP`.
//! Family/backend capability and quality evidence belong to model factory adapters;
//! constructing a topology alone grants neither execution nor quality acceptance.

use super::expert::ExpertGroup;
use ferrule_common::{Error, ParallelRankId, ParallelismPlan, Result};
use ferrule_model::decoder::HybridStateSchema;
use ferrule_model::transformer::hybrid_parallel::HybridParallelPlan;
use ferrule_model::transformer::{DecoderModelSpec, ExpertDispatchLimits};
use std::collections::BTreeSet;

/// Structural failures for the model-neutral owner/layout contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridTopologyError {
    Parallelism,
    KvOwners,
    ExpertOwners,
    MissingKvPlane,
}

impl std::fmt::Display for HybridTopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Parallelism => "hybrid topology parallelism does not match its roster",
            Self::KvOwners => "hybrid topology has invalid KV owner roster",
            Self::ExpertOwners => "hybrid topology has invalid expert owner roster",
            Self::MissingKvPlane => "hybrid topology stage has no full-attention KV plane",
        })
    }
}
impl std::error::Error for HybridTopologyError {}

#[derive(Clone, Debug)]
pub struct HybridExpertTopology {
    stages: Vec<HybridParallelPlan>,
    physical_members: BTreeSet<ParallelRankId>,
    has_attachments: bool,
}

fn error(message: &str) -> Error {
    Error::Execution {
        message: format!("hybrid topology: {message}"),
    }
}

impl HybridExpertTopology {
    /// One replica, an exact ordered PP partition, and explicit rank sets. A TP
    /// rank may also own experts in its own stage; cross-stage aliasing is rejected.
    pub fn new(spec: &DecoderModelSpec, stages: Vec<HybridParallelPlan>) -> Result<Self> {
        let first = stages
            .first()
            .ok_or_else(|| error("empty stage list"))?
            .scope();
        let schemas = stages
            .iter()
            .map(|s| s.schema().clone())
            .collect::<Vec<_>>();
        HybridStateSchema::validate_partition(spec, &schemas)?;
        let mut physical_members = BTreeSet::new();
        for (stage, plan) in stages.iter().enumerate() {
            plan.validate_spec(spec)?;
            let scope = plan.scope();
            if scope.topology != first.topology
                || scope.replica != first.replica
                || usize::try_from(scope.stage).ok() != Some(stage)
            {
                return Err(error("topology/replica/stage scope mismatch"));
            }
            for rank in plan.physical_members() {
                if !physical_members.insert(rank) {
                    return Err(error("physical rank is attached to multiple stages"));
                }
            }
        }
        Ok(Self {
            stages,
            physical_members,
            has_attachments: !spec.attachments().is_empty(),
        })
    }

    pub fn stages(&self) -> &[HybridParallelPlan] {
        &self.stages
    }
    pub fn physical_members(&self) -> &BTreeSet<ParallelRankId> {
        &self.physical_members
    }
    pub fn has_attachments(&self) -> bool {
        self.has_attachments
    }

    /// Bind the exact root mesh and optional external attachments before owner startup.
    /// Local tensor experts add neither a child owner nor a KV voter. This is
    /// structural authority only; provider/precision execution gates remain separate.
    pub fn bind_tensor_scopes(
        &self,
        topology: &ferrule_common::ValidatedParallelTopology,
        segments: &[ferrule_model::transformer::LayerSegmentPlan],
        groups: &[ExpertGroup],
    ) -> Result<ferrule_common::ParallelExecutionScopes> {
        use ferrule_common::topology::{ExpertDispatchMembers, ExpertSourceScope};
        ferrule_model::transformer::LayerSegmentPlan::validate_pipeline(segments)
            .map_err(|e| error(&e.to_string()))?;
        let p = topology.plan();
        if p.data_parallel != 1
            || p.sequence_parallel != 1
            || p.context_parallel != 1
            || self.stages.len() != segments.len()
            || segments.len() != p.pipeline_parallel
            || groups.len()
                != self
                    .stages
                    .iter()
                    .filter(|s| s.expert_attachment().is_some())
                    .count()
        {
            return Err(error("tensor stage/attachment count differs from mesh"));
        }
        let mut scopes = topology
            .execution_scopes(0)
            .map_err(|e| error(&format!("tensor execution scopes: {e:?}")))?;
        for (index, (stage, segment)) in self.stages.iter().zip(segments).enumerate() {
            let roots = scopes
                .tensor_collective_participants(index as u32, 0)
                .map_err(|e| error(&format!("tensor participants: {e:?}")))?;
            if stage.scope().topology != topology.topology_id()
                || stage.scope().replica != 0
                || stage.scope().stage as usize != index
                || stage.schema().layer_range() != segment.layers()
                || stage.schema().total_layers() != segment.total_layers()
                || stage.tensor_members() != roots.iter().collect::<Vec<_>>()
            {
                return Err(error("tensor root identity differs from sealed stage"));
            }
            let matching = groups
                .iter()
                .filter(|g| g.layers == segment.layers())
                .collect::<Vec<_>>();
            match (stage.expert_attachment(), matching.as_slice()) {
                (None, []) if p.expert_parallel == 1 => (),
                (Some(_), [group]) if p.expert_parallel > 1 => {
                    if **group != self.expert_group(index, group.limits)?
                        || group.members.len() != p.expert_parallel
                        || group
                            .members
                            .iter()
                            .any(|r| r.get() < topology.world_size())
                    {
                        return Err(error("external experts differ from sealed tensor stage"));
                    }
                    scopes
                        .attach_expert_dispatch_members(
                            ExpertDispatchMembers::new_with_scope(
                                topology,
                                0,
                                index as u32,
                                ExpertSourceScope::ExternalTensorSource,
                                group.source_rank,
                                group.members.clone(),
                            )
                            .map_err(|e| error(&format!("tensor expert scope: {e:?}")))?,
                        )
                        .map_err(|e| error(&format!("tensor expert attachment: {e:?}")))?;
                }
                _ => {
                    return Err(error(
                        "missing, duplicate or unexpected tensor expert attachment",
                    ));
                }
            }
        }
        Ok(scopes)
    }

    /// Validate contiguous PP-major TP roots in replica zero, followed by
    /// stage-ordered external expert owners sourced by each first TP lane.
    /// This is an explicit layout contract, not a restriction on `new`.
    /// Family selection, backend support, and numerical quality stay in the
    /// model-factory composition adapter.
    pub fn validate_external_expert_roster(
        &self,
        p: ParallelismPlan,
    ) -> std::result::Result<(), HybridTopologyError> {
        if self.stages.len() != p.pipeline_parallel
            || p.tensor_parallel == 0
            || p.expert_parallel == 0
            || p.data_parallel == 0
            || p.sequence_parallel == 0
            || p.context_parallel == 0
        {
            return Err(HybridTopologyError::Parallelism);
        }
        let roots = p
            .pipeline_parallel
            .checked_mul(p.tensor_parallel)
            .ok_or(HybridTopologyError::Parallelism)?;
        p.pipeline_parallel
            .checked_mul(p.expert_parallel)
            .and_then(|experts| roots.checked_add(experts))
            .ok_or(HybridTopologyError::Parallelism)?;
        for (index, stage) in self.stages.iter().enumerate() {
            let root_start = index * p.tensor_parallel;
            if stage.scope().replica != 0
                || stage.tensor_members().len() != p.tensor_parallel
                || stage
                    .tensor_members()
                    .iter()
                    .enumerate()
                    .any(|(lane, rank)| rank.get() as usize != root_start + lane)
            {
                return Err(HybridTopologyError::KvOwners);
            }
            let ep = stage
                .expert_attachment()
                .ok_or(HybridTopologyError::ExpertOwners)?;
            if ep.source_rank.get() as usize != root_start
                || ep.members.len() != p.expert_parallel
                || ep.members.iter().enumerate().any(|(slot, rank)| {
                    rank.get() as usize != roots + index * p.expert_parallel + slot
                })
            {
                return Err(HybridTopologyError::ExpertOwners);
            }
            if !stage.schema().layers().iter().any(|layer| {
                matches!(
                    layer,
                    ferrule_model::decoder::HybridLayerSchema::FullAttention { .. }
                )
            }) {
                return Err(HybridTopologyError::MissingKvPlane);
            }
        }
        Ok(())
    }

    /// Reuse the runtime's explicit owner/placement contract. This does not attach
    /// EP as another mesh axis or relax existing pipeline execution admission.
    pub fn expert_group(&self, stage: usize, limits: ExpertDispatchLimits) -> Result<ExpertGroup> {
        let plan = self
            .stages
            .get(stage)
            .ok_or_else(|| error("unknown stage"))?;
        let attachment = plan
            .expert_attachment()
            .ok_or_else(|| error("dense stage has no expert attachment"))?;
        let group = ExpertGroup {
            source_rank: attachment.source_rank,
            members: attachment.members.clone(),
            layers: plan.schema().layer_range(),
            placement: attachment.placement.clone(),
            limits,
        };
        group.validate()?;
        Ok(group)
    }

    pub fn optional_expert_group(
        &self,
        stage: usize,
        limits: ExpertDispatchLimits,
    ) -> Result<Option<ExpertGroup>> {
        let plan = self
            .stages
            .get(stage)
            .ok_or_else(|| error("unknown stage"))?;
        plan.expert_attachment()
            .map(|_| self.expert_group(stage, limits))
            .transpose()
    }
}
