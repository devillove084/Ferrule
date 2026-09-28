//! Stable identifiers and validated descriptors for parallel execution.

macro_rules! topology_id {
    ($name:ident) => {
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u32);
        impl $name {
            pub const fn new(value: u32) -> Self {
                Self(value)
            }
            pub const fn get(self) -> u32 {
                self.0
            }
        }
        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                Self::new(value)
            }
        }
        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }
    };
}
topology_id!(ParallelTopologyId);
topology_id!(ParallelRankId);
topology_id!(ParallelGroupId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ParallelismPlan {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub expert_parallel: usize,
    pub sequence_parallel: usize,
    pub context_parallel: usize,
    pub pipeline_parallel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelismPlanError {
    ZeroDimension { dimension: &'static str },
}

impl std::fmt::Display for ParallelismPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension { dimension } => {
                write!(
                    formatter,
                    "parallelism dimension {dimension} must be greater than zero"
                )
            }
        }
    }
}

impl std::error::Error for ParallelismPlanError {}

impl ParallelismPlan {
    pub fn validated(
        data_parallel: usize,
        tensor_parallel: usize,
        expert_parallel: usize,
        sequence_parallel: usize,
        context_parallel: usize,
        pipeline_parallel: usize,
    ) -> Result<Self, ParallelismPlanError> {
        let plan = Self {
            data_parallel,
            tensor_parallel,
            expert_parallel,
            sequence_parallel,
            context_parallel,
            pipeline_parallel,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), ParallelismPlanError> {
        for (dimension, value) in [
            ("data_parallel", self.data_parallel),
            ("tensor_parallel", self.tensor_parallel),
            ("expert_parallel", self.expert_parallel),
            ("sequence_parallel", self.sequence_parallel),
            ("context_parallel", self.context_parallel),
            ("pipeline_parallel", self.pipeline_parallel),
        ] {
            if value == 0 {
                return Err(ParallelismPlanError::ZeroDimension { dimension });
            }
        }
        Ok(())
    }
}

impl Default for ParallelismPlan {
    fn default() -> Self {
        Self {
            data_parallel: 1,
            tensor_parallel: 1,
            expert_parallel: 1,
            sequence_parallel: 1,
            context_parallel: 1,
            pipeline_parallel: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelTopologyError {
    ZeroId,
    ZeroWorldSize,
    RankOutOfBounds,
    ZeroFactor,
    UnsupportedParallelism,
    FactorOverflow,
    ReplicaOutOfBounds,
    EmptyParticipants,
    DuplicateParticipant,
    ParticipantTopologyMismatch,
    ExpertDispatchMemberOverlap,
    KvOwnerOverlap,
    ExpertDispatchMembersAlreadyAttached,
    ScopeReplicaMismatch,
    ScopeKindMismatch,
    ExpertDegreeMismatch,
    ExpertSourceNotMember,
    ExpertSourceNotStageCaller,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedParallelTopology {
    topology_id: ParallelTopologyId,
    world_size: u32,
    local_rank: ParallelRankId,
    plan: ParallelismPlan,
}

impl ValidatedParallelTopology {
    pub fn new(
        topology_id: ParallelTopologyId,
        world_size: u32,
        local_rank: ParallelRankId,
        plan: ParallelismPlan,
    ) -> Result<Self, ParallelTopologyError> {
        if topology_id.get() == 0 {
            return Err(ParallelTopologyError::ZeroId);
        }
        if world_size == 0 {
            return Err(ParallelTopologyError::ZeroWorldSize);
        }
        if local_rank.get() >= world_size {
            return Err(ParallelTopologyError::RankOutOfBounds);
        }
        plan.validate()
            .map_err(|_| ParallelTopologyError::ZeroFactor)?;
        let mesh_size = plan
            .data_parallel
            .checked_mul(plan.pipeline_parallel)
            .and_then(|size| size.checked_mul(plan.tensor_parallel))
            .ok_or(ParallelTopologyError::FactorOverflow)?;
        // EP is an attached activation group, not a world/mesh factor. SP and
        // CP remain outside this first-phase topology contract.
        if u32::try_from(mesh_size) != Ok(world_size)
            || plan.sequence_parallel != 1
            || plan.context_parallel != 1
        {
            return Err(ParallelTopologyError::UnsupportedParallelism);
        }
        Ok(Self {
            topology_id,
            world_size,
            local_rank,
            plan,
        })
    }
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.topology_id
    }
    pub const fn world_size(&self) -> u32 {
        self.world_size
    }
    pub const fn local_rank(&self) -> ParallelRankId {
        self.local_rank
    }
    pub const fn plan(&self) -> ParallelismPlan {
        self.plan
    }

    /// Create a topology-bound coordinate. Coordinates carry the topology
    /// epoch so they cannot be silently reused with a different mesh.
    pub fn coordinate(
        &self,
        replica: u32,
        stage: u32,
        tensor: u32,
    ) -> Result<MeshCoordinate, ParallelTopologyError> {
        self.rank_at(replica, stage, tensor)?;
        Ok(MeshCoordinate {
            topology_id: self.topology_id,
            world_size: self.world_size,
            plan: self.plan,
            replica,
            stage,
            tensor,
        })
    }

    pub fn coordinate_of(
        &self,
        rank: ParallelRankId,
    ) -> Result<MeshCoordinate, ParallelTopologyError> {
        let replica = self.replica_of(rank)?;
        self.coordinate(
            replica,
            rank.get() / self.plan.tensor_parallel as u32 % self.plan.pipeline_parallel as u32,
            rank.get() % self.plan.tensor_parallel as u32,
        )
    }

    /// Resolve a full, identity-checked coordinate, never a TP-local rank.
    pub fn rank_of(
        &self,
        coordinate: MeshCoordinate,
    ) -> Result<ParallelRankId, ParallelTopologyError> {
        if coordinate.topology_id != self.topology_id
            || coordinate.world_size != self.world_size
            || coordinate.plan != self.plan
        {
            return Err(ParallelTopologyError::ParticipantTopologyMismatch);
        }
        self.rank_at(coordinate.replica, coordinate.stage, coordinate.tensor)
    }

    /// Build the first-phase execution scopes for one data-parallel replica.
    /// KV owns the replica's PP×TP mesh; tensor collectives are stage-local.
    /// Expert members are attached activation owners and start empty.
    pub fn execution_scopes(
        &self,
        replica: u32,
    ) -> Result<ParallelExecutionScopes, ParallelTopologyError> {
        self.rank_at(replica, 0, 0)?;
        let kv = (0..self.plan.pipeline_parallel as u32).flat_map(|stage| {
            (0..self.plan.tensor_parallel as u32).map(move |tensor| {
                self.rank_at(replica, stage, tensor)
                    .expect("validated mesh")
            })
        });
        let kv = ParticipantSet::new(self, kv)?;
        let tensor_collectives = (0..self.plan.pipeline_parallel as u32)
            .map(|stage| {
                self.tensor_stage_participants(replica, stage)
                    .map(TensorCollectiveParticipants::from_participants)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ParallelExecutionScopes {
            topology_id: self.topology_id,
            plan: self.plan,
            replica,
            kv_participants: KvParticipants::from_participants(kv),
            tensor_collectives,
            expert_dispatch_members: std::collections::BTreeMap::new(),
        })
    }

    pub fn participants(&self) -> ParticipantSet {
        ParticipantSet {
            topology_id: self.topology_id,
            world_size: self.world_size,
            plan: self.plan,
            ranks: (0..self.world_size).map(ParallelRankId::new).collect(),
        }
    }

    /// TP peers at pipeline stage zero. Preserves the original DP × TP layout
    /// when PP=1; use `tensor_stage_participants` for other pipeline stages.
    pub fn tensor_participants(
        &self,
        replica: u32,
    ) -> Result<ParticipantSet, ParallelTopologyError> {
        self.tensor_stage_participants(replica, 0)
    }

    /// Rank layout: ((replica * PP) + pipeline stage) * TP + tensor index.
    /// EP is deliberately not multiplied into this mesh: it has its own executor.
    pub fn rank_at(
        &self,
        replica: u32,
        stage: u32,
        tensor: u32,
    ) -> Result<ParallelRankId, ParallelTopologyError> {
        if replica >= self.plan.data_parallel as u32 {
            return Err(ParallelTopologyError::ReplicaOutOfBounds);
        }
        if stage >= self.plan.pipeline_parallel as u32 || tensor >= self.plan.tensor_parallel as u32
        {
            return Err(ParallelTopologyError::RankOutOfBounds);
        }
        Ok(ParallelRankId::new(
            (replica * self.plan.pipeline_parallel as u32 + stage)
                * self.plan.tensor_parallel as u32
                + tensor,
        ))
    }

    pub fn tensor_stage_participants(
        &self,
        replica: u32,
        stage: u32,
    ) -> Result<ParticipantSet, ParallelTopologyError> {
        let start = self.rank_at(replica, stage, 0)?.get();
        ParticipantSet::new(
            self,
            (start..start + self.plan.tensor_parallel as u32).map(ParallelRankId::new),
        )
    }

    pub fn pipeline_participants(
        &self,
        replica: u32,
        tensor: u32,
    ) -> Result<ParticipantSet, ParallelTopologyError> {
        self.rank_at(replica, 0, tensor)?;
        ParticipantSet::new(
            self,
            (0..self.plan.pipeline_parallel as u32).map(|stage| {
                self.rank_at(replica, stage, tensor)
                    .expect("validated mesh coordinate")
            }),
        )
    }

    pub fn data_participants(
        &self,
        stage: u32,
        tensor: u32,
    ) -> Result<ParticipantSet, ParallelTopologyError> {
        self.rank_at(0, stage, tensor)?;
        ParticipantSet::new(
            self,
            (0..self.plan.data_parallel as u32).map(|replica| {
                self.rank_at(replica, stage, tensor)
                    .expect("validated mesh coordinate")
            }),
        )
    }

    pub fn replica_of(&self, rank: ParallelRankId) -> Result<u32, ParallelTopologyError> {
        if rank.get() >= self.world_size {
            return Err(ParallelTopologyError::RankOutOfBounds);
        }
        Ok(rank.get() / (self.plan.pipeline_parallel * self.plan.tensor_parallel) as u32)
    }

    /// Require the same topology epoch, world and mesh; local rank is not scope identity.
    pub fn validate_participants(
        &self,
        participants: &ParticipantSet,
    ) -> Result<(), ParallelTopologyError> {
        if participants.topology_id != self.topology_id
            || participants.world_size != self.world_size
            || participants.plan != self.plan
        {
            return Err(ParallelTopologyError::ParticipantTopologyMismatch);
        }
        if participants
            .ranks
            .iter()
            .any(|rank| rank.get() >= self.world_size)
        {
            return Err(ParallelTopologyError::RankOutOfBounds);
        }
        Ok(())
    }
}

/// A topology-bound coordinate in the DP×PP×TP mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshCoordinate {
    topology_id: ParallelTopologyId,
    world_size: u32,
    plan: ParallelismPlan,
    replica: u32,
    stage: u32,
    tensor: u32,
}

impl MeshCoordinate {
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.topology_id
    }
    pub const fn replica(&self) -> u32 {
        self.replica
    }
    pub const fn stage(&self) -> u32 {
        self.stage
    }
    pub const fn tensor(&self) -> u32 {
        self.tensor
    }
}

/// A KV-owner scope. It is intentionally distinct from a tensor collective
/// scope: a tensor rank is never implicitly promoted to a KV owner.
///
/// ```compile_fail
/// use ferrule_common::{KvParticipants, TensorCollectiveParticipants};
/// fn promote(tp: TensorCollectiveParticipants) -> KvParticipants {
///     tp.into()
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvParticipants(ParticipantSet);

impl KvParticipants {
    fn from_participants(participants: ParticipantSet) -> Self {
        Self(participants)
    }
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.0.topology_id()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn contains(&self, rank: ParallelRankId) -> bool {
        self.0.contains(rank)
    }
    pub fn iter(&self) -> std::vec::IntoIter<ParallelRankId> {
        self.0.iter()
    }
    pub fn as_participants(&self) -> &ParticipantSet {
        &self.0
    }
}

/// A stage-local TP collective scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCollectiveParticipants(ParticipantSet);

impl TensorCollectiveParticipants {
    fn from_participants(participants: ParticipantSet) -> Self {
        Self(participants)
    }
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.0.topology_id()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn contains(&self, rank: ParallelRankId) -> bool {
        self.0.contains(rank)
    }
    pub fn iter(&self) -> std::vec::IntoIter<ParallelRankId> {
        self.0.iter()
    }
    pub fn as_participants(&self) -> &ParticipantSet {
        &self.0
    }
}

/// The caller identity scope for an attached expert dispatch group.
///
/// `Member` is the legacy thread path: the caller is one of the expert
/// owners. `ExternalStage` is the process PP×EP path: the caller is the
/// attached PP stage rank and expert members are workers only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertSourceScope {
    Member,
    ExternalStage,
}

/// Attached activation owners for one replica/stage, shared by its TP peers.
/// This is not a `ParticipantSet`: expert owners cannot join a KV transaction.
/// Explicit member order is preserved because it defines expert dispatch slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertDispatchMembers {
    attachment: MeshCoordinate,
    source_scope: ExpertSourceScope,
    source_rank: ParallelRankId,
    members: Vec<ParallelRankId>,
}

impl ExpertDispatchMembers {
    /// Legacy member-source constructor. Thread callers retain this exact
    /// member-only contract.
    pub fn new(
        topology: &ValidatedParallelTopology,
        replica: u32,
        stage: u32,
        source_rank: ParallelRankId,
        members: impl IntoIterator<Item = ParallelRankId>,
    ) -> Result<Self, ParallelTopologyError> {
        Self::new_with_scope(
            topology,
            replica,
            stage,
            ExpertSourceScope::Member,
            source_rank,
            members,
        )
    }

    /// Construct an attachment with an explicit caller scope. EP members are
    /// always activation workers; ExternalStage never adds the caller to them.
    pub fn new_with_scope(
        topology: &ValidatedParallelTopology,
        replica: u32,
        stage: u32,
        source_scope: ExpertSourceScope,
        source_rank: ParallelRankId,
        members: impl IntoIterator<Item = ParallelRankId>,
    ) -> Result<Self, ParallelTopologyError> {
        let attachment = topology.coordinate(replica, stage, 0)?;
        let members: Vec<_> = members.into_iter().collect();
        if members.is_empty() {
            return Err(ParallelTopologyError::EmptyParticipants);
        }
        let unique: std::collections::BTreeSet<_> = members.iter().copied().collect();
        if unique.len() != members.len() {
            return Err(ParallelTopologyError::DuplicateParticipant);
        }
        // Exclude the whole mesh, including other DP replicas, not just this KV scope.
        if members.iter().any(|rank| rank.get() < topology.world_size) {
            return Err(ParallelTopologyError::ExpertDispatchMemberOverlap);
        }
        match source_scope {
            ExpertSourceScope::Member if !members.contains(&source_rank) => {
                return Err(ParallelTopologyError::ExpertSourceNotMember);
            }
            ExpertSourceScope::ExternalStage => {
                if topology.plan.tensor_parallel != 1 {
                    return Err(ParallelTopologyError::UnsupportedParallelism);
                }
                if source_rank != topology.rank_at(replica, stage, 0)? {
                    return Err(ParallelTopologyError::ExpertSourceNotStageCaller);
                }
            }
            ExpertSourceScope::Member => {}
        }
        // EP=1 keeps legacy explicitly configured groups compatible. A declared
        // EP>1 constrains attached groups; it never allocates implicit owners.
        if topology.plan.expert_parallel != 1 && members.len() != topology.plan.expert_parallel {
            return Err(ParallelTopologyError::ExpertDegreeMismatch);
        }
        Ok(Self {
            attachment,
            source_scope,
            source_rank,
            members,
        })
    }
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.attachment.topology_id
    }
    pub const fn replica(&self) -> u32 {
        self.attachment.replica
    }
    pub const fn stage(&self) -> u32 {
        self.attachment.stage
    }
    pub const fn source_scope(&self) -> ExpertSourceScope {
        self.source_scope
    }
    pub const fn source_rank(&self) -> ParallelRankId {
        self.source_rank
    }
    pub fn len(&self) -> usize {
        self.members.len()
    }
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
    pub fn contains(&self, rank: ParallelRankId) -> bool {
        self.members.contains(&rank)
    }
    pub fn iter(&self) -> std::vec::IntoIter<ParallelRankId> {
        self.members.clone().into_iter()
    }
}

/// Identity-checked scopes for one DP replica. This is metadata, not a build
/// plan, transaction coordinator, or authorization to execute a composed mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelExecutionScopes {
    topology_id: ParallelTopologyId,
    plan: ParallelismPlan,
    replica: u32,
    kv_participants: KvParticipants,
    tensor_collectives: Vec<TensorCollectiveParticipants>,
    expert_dispatch_members: std::collections::BTreeMap<u32, ExpertDispatchMembers>,
}

impl ParallelExecutionScopes {
    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.topology_id
    }
    pub const fn plan(&self) -> ParallelismPlan {
        self.plan
    }
    pub const fn replica(&self) -> u32 {
        self.replica
    }
    pub fn kv_participants(&self) -> &KvParticipants {
        &self.kv_participants
    }
    pub fn tensor_collective_participants(
        &self,
        stage: u32,
        replica: u32,
    ) -> Result<&TensorCollectiveParticipants, ParallelTopologyError> {
        if replica != self.replica {
            return Err(ParallelTopologyError::ScopeReplicaMismatch);
        }
        self.tensor_collectives
            .get(stage as usize)
            .ok_or(ParallelTopologyError::RankOutOfBounds)
    }
    pub fn expert_dispatch_members(
        &self,
        stage: u32,
        replica: u32,
    ) -> Result<Option<&ExpertDispatchMembers>, ParallelTopologyError> {
        self.tensor_collective_participants(stage, replica)?;
        Ok(self.expert_dispatch_members.get(&stage))
    }

    /// Validate identity without treating the topology's local rank as identity.
    pub fn validate_topology(
        &self,
        topology: &ValidatedParallelTopology,
    ) -> Result<(), ParallelTopologyError> {
        topology.validate_participants(self.kv_participants.as_participants())
    }

    /// Validate a legacy untyped cohort before using it for KV custody. A
    /// stage-local collective cannot stand in for the whole PP×TP transaction.
    pub fn validate_kv_participants(
        &self,
        participants: &ParticipantSet,
    ) -> Result<(), ParallelTopologyError> {
        let kv = self.kv_participants.as_participants();
        if participants.topology_id != kv.topology_id
            || participants.world_size != kv.world_size
            || participants.plan != kv.plan
        {
            return Err(ParallelTopologyError::ParticipantTopologyMismatch);
        }
        if participants != kv {
            return Err(ParallelTopologyError::ScopeKindMismatch);
        }
        Ok(())
    }

    /// KV owner addressing requires all mesh axes, not a TP-local slot. The
    /// returned global ID interoperates with the existing KV/transaction API.
    ///
    /// ```compile_fail
    /// use ferrule_common::{ParallelExecutionScopes, ParallelRankId};
    /// fn local_is_not_an_owner(scopes: &ParallelExecutionScopes, local: ParallelRankId) {
    ///     scopes.kv_owner(local).unwrap();
    /// }
    /// ```
    pub fn kv_owner(
        &self,
        coordinate: MeshCoordinate,
    ) -> Result<ParallelRankId, ParallelTopologyError> {
        if coordinate.topology_id != self.topology_id
            || coordinate.world_size != self.kv_participants.0.world_size
            || coordinate.plan != self.plan
        {
            return Err(ParallelTopologyError::ParticipantTopologyMismatch);
        }
        self.tensor_collective_participants(coordinate.stage, coordinate.replica)?
            .0
            .ranks
            .get(coordinate.tensor as usize)
            .copied()
            .ok_or(ParallelTopologyError::RankOutOfBounds)
    }

    /// Validate separate DP scopes before composing them. Each scope already
    /// excludes the entire mesh from expert ownership; this also rejects expert
    /// owner reuse between replicas. Call again after changing attachments.
    pub fn validate_disjoint_owners(&self, other: &Self) -> Result<(), ParallelTopologyError> {
        let left = self.kv_participants.as_participants();
        let right = other.kv_participants.as_participants();
        if left.topology_id != right.topology_id
            || left.world_size != right.world_size
            || left.plan != right.plan
        {
            return Err(ParallelTopologyError::ParticipantTopologyMismatch);
        }
        if left.iter().any(|rank| right.contains(rank)) {
            return Err(ParallelTopologyError::KvOwnerOverlap);
        }
        if self.expert_dispatch_members.values().any(|group| {
            other
                .expert_dispatch_members
                .values()
                .any(|peer| group.members.iter().any(|rank| peer.contains(*rank)))
        }) {
            return Err(ParallelTopologyError::ExpertDispatchMemberOverlap);
        }
        Ok(())
    }

    /// Attach explicit activation owners atomically. Duplicate stage attachment
    /// and owner reuse across stages are errors; neither widens the KV cohort.
    pub fn attach_expert_dispatch_members(
        &mut self,
        members: ExpertDispatchMembers,
    ) -> Result<(), ParallelTopologyError> {
        self.kv_owner(members.attachment)?;
        if self.expert_dispatch_members.contains_key(&members.stage()) {
            return Err(ParallelTopologyError::ExpertDispatchMembersAlreadyAttached);
        }
        if self
            .expert_dispatch_members
            .values()
            .any(|attached| members.members.iter().any(|rank| attached.contains(*rank)))
        {
            return Err(ParallelTopologyError::ExpertDispatchMemberOverlap);
        }
        self.expert_dispatch_members
            .insert(members.stage(), members);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticipantSet {
    topology_id: ParallelTopologyId,
    world_size: u32,
    plan: ParallelismPlan,
    ranks: Vec<ParallelRankId>,
}
impl ParticipantSet {
    /// Construct a nonempty, sorted scope. Duplicates are rejected, not deduplicated.
    /// Rank bounds are validated against the supplied topology.
    pub fn new(
        topology: &ValidatedParallelTopology,
        ranks: impl IntoIterator<Item = ParallelRankId>,
    ) -> Result<Self, ParallelTopologyError> {
        let mut ranks: Vec<_> = ranks.into_iter().collect();
        if ranks.is_empty() {
            return Err(ParallelTopologyError::EmptyParticipants);
        }
        ranks.sort_unstable();
        if ranks.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ParallelTopologyError::DuplicateParticipant);
        }
        let participants = Self {
            topology_id: topology.topology_id,
            world_size: topology.world_size,
            plan: topology.plan,
            ranks,
        };
        topology.validate_participants(&participants)?;
        Ok(participants)
    }

    pub const fn topology_id(&self) -> ParallelTopologyId {
        self.topology_id
    }
    pub const fn plan(&self) -> ParallelismPlan {
        self.plan
    }
    /// Size of the containing world, not the number of ranks in this scope.
    pub const fn world_size(&self) -> u32 {
        self.world_size
    }
    pub fn len(&self) -> usize {
        self.ranks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }
    pub fn contains(&self, rank: ParallelRankId) -> bool {
        self.ranks.binary_search(&rank).is_ok()
    }
    /// An owned iterator lets callers mutate a coordinator while visiting its ranks.
    pub fn iter(&self) -> std::vec::IntoIter<ParallelRankId> {
        self.ranks.clone().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(
        world_size: u32,
        plan: ParallelismPlan,
    ) -> Result<ValidatedParallelTopology, ParallelTopologyError> {
        ValidatedParallelTopology::new(
            ParallelTopologyId::new(1),
            world_size,
            ParallelRankId::new(0),
            plan,
        )
    }

    #[test]
    fn default_parallelism_plan_is_valid() {
        assert_eq!(
            ParallelismPlan::default(),
            ParallelismPlan::validated(1, 1, 1, 1, 1, 1).unwrap()
        );
        assert!(ParallelismPlan::default().validate().is_ok());
    }

    #[test]
    fn all_zero_dimensions_share_plan_validation() {
        for (index, dimension) in [
            "data_parallel",
            "tensor_parallel",
            "expert_parallel",
            "sequence_parallel",
            "context_parallel",
            "pipeline_parallel",
        ]
        .into_iter()
        .enumerate()
        {
            let mut factors = [1; 6];
            factors[index] = 0;
            let [
                data_parallel,
                tensor_parallel,
                expert_parallel,
                sequence_parallel,
                context_parallel,
                pipeline_parallel,
            ] = factors;
            let plan = ParallelismPlan {
                data_parallel,
                tensor_parallel,
                expert_parallel,
                sequence_parallel,
                context_parallel,
                pipeline_parallel,
            };
            let error = ParallelismPlanError::ZeroDimension { dimension };
            assert_eq!(plan.validate(), Err(error.clone()));
            assert_eq!(
                ParallelismPlan::validated(
                    data_parallel,
                    tensor_parallel,
                    expert_parallel,
                    sequence_parallel,
                    context_parallel,
                    pipeline_parallel
                ),
                Err(error.clone())
            );
            assert!(error.to_string().contains(dimension));
            for world in [1, 2] {
                assert_eq!(
                    topology(world, plan),
                    Err(ParallelTopologyError::ZeroFactor)
                );
            }
        }
    }

    #[test]
    fn validates_pure_dp_and_membership() {
        for world in [1, 2, 4] {
            let plan = ParallelismPlan {
                data_parallel: usize::try_from(world).unwrap(),
                ..ParallelismPlan::default()
            };
            let t = ValidatedParallelTopology::new(
                ParallelTopologyId::new(1),
                world,
                ParallelRankId::new(world - 1),
                plan,
            )
            .unwrap();
            assert_eq!(t.plan(), plan);
            assert_eq!(t.local_rank(), ParallelRankId::new(world - 1));
            assert_eq!(t.participants().world_size(), world);
            assert_eq!(
                t.participants().iter().collect::<Vec<_>>(),
                (0..world).map(ParallelRankId::new).collect::<Vec<_>>()
            );
            assert!(t.participants().contains(ParallelRankId::new(0)));
            assert!(!t.participants().contains(ParallelRankId::new(world)));
        }
    }

    #[test]
    fn dp_pp_tp_groups_follow_one_rank_layout() {
        let t = topology(12, ParallelismPlan::validated(2, 2, 1, 1, 1, 3).unwrap()).unwrap();
        let ids = |group: ParticipantSet| group.iter().map(ParallelRankId::get).collect::<Vec<_>>();
        assert_eq!(ids(t.tensor_participants(1).unwrap()), vec![6, 7]);
        assert_eq!(
            ids(t.tensor_stage_participants(1, 2).unwrap()),
            vec![10, 11]
        );
        assert_eq!(ids(t.pipeline_participants(1, 1).unwrap()), vec![7, 9, 11]);
        assert_eq!(ids(t.data_participants(2, 1).unwrap()), vec![5, 11]);
        for replica in 0..2 {
            for stage in 0..3 {
                for tensor in 0..2 {
                    let rank = t.rank_at(replica, stage, tensor).unwrap();
                    assert_eq!(rank.get(), (replica * 3 + stage) * 2 + tensor);
                    assert_eq!(t.replica_of(rank).unwrap(), replica);
                }
            }
        }
        assert!(t.rank_at(2, 0, 0).is_err());
        assert!(t.tensor_stage_participants(0, 3).is_err());
        assert!(t.pipeline_participants(0, 2).is_err());
        assert!(t.data_participants(3, 0).is_err());
        assert_eq!(
            topology(2, ParallelismPlan::validated(1, 1, 1, 1, 1, 2).unwrap())
                .unwrap()
                .pipeline_participants(0, 0)
                .unwrap()
                .len(),
            2,
        );
    }

    #[test]
    fn sp_cp_are_not_supported_mesh_factors() {
        for plan in [
            ParallelismPlan::validated(1, 1, 1, 2, 1, 1).unwrap(),
            ParallelismPlan::validated(1, 1, 1, 1, 2, 1).unwrap(),
        ] {
            for world in [1, 2] {
                assert_eq!(
                    topology(world, plan),
                    Err(ParallelTopologyError::UnsupportedParallelism)
                );
            }
        }
        assert_eq!(
            topology(
                1,
                ParallelismPlan::validated(usize::MAX, 2, 1, 1, 1, 2).unwrap()
            ),
            Err(ParallelTopologyError::FactorOverflow),
        );
    }

    #[test]
    fn rejects_unsupported_factors_and_mismatched_meshes_for_every_world_size() {
        for world in [1, 2] {
            // TP must match the world; SP/CP remain unsupported.
            for index in [1, 3, 4] {
                let mut factors = [1; 6];
                factors[0] = usize::try_from(world).unwrap();
                factors[index] = 8;
                let [dp, tp, ep, sp, cp, pp] = factors;
                let plan = ParallelismPlan::validated(dp, tp, ep, sp, cp, pp).unwrap();
                assert_eq!(
                    topology(world, plan),
                    Err(ParallelTopologyError::UnsupportedParallelism)
                );
            }
            for dp in [1, 2, 3] {
                if u32::try_from(dp) != Ok(world) {
                    let plan = ParallelismPlan {
                        data_parallel: dp,
                        ..ParallelismPlan::default()
                    };
                    assert_eq!(
                        topology(world, plan),
                        Err(ParallelTopologyError::UnsupportedParallelism)
                    );
                }
            }
        }
        assert_eq!(
            topology(1, ParallelismPlan::validated(2, 8, 1, 1, 1, 1).unwrap()),
            Err(ParallelTopologyError::UnsupportedParallelism)
        );
    }

    #[test]
    fn usize_plan_dimensions_are_not_truncated_to_world_size() {
        if let Ok(large) = usize::try_from(u64::from(u32::MAX) + 2) {
            let plan = ParallelismPlan::validated(large, 1, 1, 1, 1, 1).unwrap();
            assert_eq!(plan.data_parallel, large);
            assert_eq!(
                topology(1, plan),
                Err(ParallelTopologyError::UnsupportedParallelism)
            );
        }
    }

    #[test]
    fn rejects_invalid_topologies() {
        assert_eq!(
            ValidatedParallelTopology::new(
                ParallelTopologyId::new(0),
                1,
                ParallelRankId::new(0),
                ParallelismPlan::default(),
            ),
            Err(ParallelTopologyError::ZeroId)
        );
        assert_eq!(
            topology(0, ParallelismPlan::default()),
            Err(ParallelTopologyError::ZeroWorldSize)
        );
        assert_eq!(
            ValidatedParallelTopology::new(
                ParallelTopologyId::new(1),
                2,
                ParallelRankId::new(2),
                ParallelismPlan::validated(2, 1, 1, 1, 1, 1).unwrap(),
            ),
            Err(ParallelTopologyError::RankOutOfBounds)
        );
        assert_eq!(
            topology(2, ParallelismPlan::default()),
            Err(ParallelTopologyError::UnsupportedParallelism)
        );
    }

    #[test]
    fn rectangular_meshes_have_exact_tensor_scopes_and_replica_mapping() {
        for (dp, tp) in [(2, 2), (1, 8), (8, 1)] {
            let world = dp * tp;
            let plan = ParallelismPlan::validated(dp as usize, tp as usize, 1, 1, 1, 1).unwrap();
            for local in 0..world {
                let t = ValidatedParallelTopology::new(
                    ParallelTopologyId::new(1),
                    world,
                    ParallelRankId::new(local),
                    plan,
                )
                .unwrap();
                let mut all = Vec::new();
                for replica in 0..dp {
                    let scope = t.tensor_participants(replica).unwrap();
                    let expected: Vec<_> = (replica * tp..(replica + 1) * tp)
                        .map(ParallelRankId::new)
                        .collect();
                    assert_eq!(scope.iter().collect::<Vec<_>>(), expected);
                    assert_eq!(scope.len(), tp as usize);
                    assert_eq!(scope.world_size(), world);
                    assert!(!scope.is_empty());
                    for rank in 0..world {
                        let rank = ParallelRankId::new(rank);
                        assert_eq!(scope.contains(rank), t.replica_of(rank) == Ok(replica));
                    }
                    all.extend(scope.iter());
                }
                assert_eq!(all, t.participants().iter().collect::<Vec<_>>());
                assert_eq!(
                    t.tensor_participants(dp),
                    Err(ParallelTopologyError::ReplicaOutOfBounds)
                );
                assert_eq!(
                    t.tensor_participants(u32::MAX),
                    Err(ParallelTopologyError::ReplicaOutOfBounds)
                );
                assert_eq!(
                    t.replica_of(ParallelRankId::new(world)),
                    Err(ParallelTopologyError::RankOutOfBounds)
                );
            }
        }
    }

    #[test]
    fn rectangular_mesh_rejects_unequal_world_and_checked_overflow() {
        for (world, dp, tp) in [(3, 2, 2), (5, 2, 2), (7, 1, 8), (9, 1, 8)] {
            assert_eq!(
                topology(
                    world,
                    ParallelismPlan::validated(dp, tp, 1, 1, 1, 1).unwrap()
                ),
                Err(ParallelTopologyError::UnsupportedParallelism)
            );
        }
        for (dp, tp) in [(usize::MAX, 2), (2, usize::MAX)] {
            assert_eq!(
                topology(2, ParallelismPlan::validated(dp, tp, 1, 1, 1, 1).unwrap()),
                Err(ParallelTopologyError::FactorOverflow)
            );
        }
        // This product fits usize on 64-bit hosts, but must never truncate to u32.
        let overflow = topology(
            1,
            ParallelismPlan::validated(u32::MAX as usize, 2, 1, 1, 1, 1).unwrap(),
        );
        assert!(matches!(
            overflow,
            Err(ParallelTopologyError::UnsupportedParallelism
                | ParallelTopologyError::FactorOverflow)
        ));
        // Keep this legacy matrix focused on SP/CP.
        for index in 3..5 {
            let mut factors = [2, 2, 1, 1, 1, 1];
            factors[index] = 2;
            let [dp, tp, ep, sp, cp, pp] = factors;
            assert_eq!(
                topology(
                    4,
                    ParallelismPlan::validated(dp, tp, ep, sp, cp, pp).unwrap()
                ),
                Err(ParallelTopologyError::UnsupportedParallelism)
            );
        }
    }

    #[test]
    fn last_replica_at_u32_world_boundary_does_not_overflow() {
        let t = topology(
            u32::MAX,
            ParallelismPlan::validated(u32::MAX as usize, 1, 1, 1, 1, 1).unwrap(),
        )
        .unwrap();
        let rank = ParallelRankId::new(u32::MAX - 1);
        assert_eq!(t.replica_of(rank), Ok(u32::MAX - 1));
        assert_eq!(
            t.tensor_participants(u32::MAX - 1)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![rank]
        );
    }

    #[test]
    fn participant_construction_sorts_but_rejects_empty_duplicates_and_out_of_bounds() {
        let t = topology(4, ParallelismPlan::validated(2, 2, 1, 1, 1, 1).unwrap()).unwrap();
        let scope = ParticipantSet::new(&t, [3, 1].map(ParallelRankId::new)).unwrap();
        assert_eq!(
            scope.iter().collect::<Vec<_>>(),
            vec![ParallelRankId::new(1), ParallelRankId::new(3)]
        );
        assert_eq!(scope.len(), 2);
        assert_eq!(scope.world_size(), 4);
        assert!(!scope.contains(ParallelRankId::new(2)));
        assert_eq!(t.validate_participants(&scope), Ok(()));
        assert_eq!(
            ParticipantSet::new(&t, []),
            Err(ParallelTopologyError::EmptyParticipants)
        );
        assert_eq!(
            ParticipantSet::new(&t, [3, 1, 3].map(ParallelRankId::new)),
            Err(ParallelTopologyError::DuplicateParticipant)
        );
        for rank in [4, u32::MAX] {
            assert_eq!(
                ParticipantSet::new(&t, [ParallelRankId::new(rank)]),
                Err(ParallelTopologyError::RankOutOfBounds)
            );
        }
        let other = topology(2, ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap()).unwrap();
        assert_eq!(
            other.validate_participants(&scope),
            Err(ParallelTopologyError::ParticipantTopologyMismatch)
        );
    }

    #[test]
    fn participant_identity_includes_epoch_and_mesh_but_not_local_rank() {
        let plan = ParallelismPlan::validated(2, 2, 1, 1, 1, 1).unwrap();
        let t = topology(4, plan).unwrap();
        let scope = t.tensor_participants(0).unwrap();
        assert_eq!(scope.topology_id(), t.topology_id());
        assert_eq!(scope.plan(), plan);
        for (epoch, other_plan) in [
            (2, plan),
            (1, ParallelismPlan::validated(1, 4, 1, 1, 1, 1).unwrap()),
            (1, ParallelismPlan::validated(4, 1, 1, 1, 1, 1).unwrap()),
        ] {
            let other = ValidatedParallelTopology::new(
                ParallelTopologyId::new(epoch),
                4,
                ParallelRankId::new(0),
                other_plan,
            )
            .unwrap();
            let other_scope = ParticipantSet::new(&other, scope.iter()).unwrap();
            assert_ne!(scope, other_scope);
            assert_eq!(
                t.validate_participants(&other_scope),
                Err(ParallelTopologyError::ParticipantTopologyMismatch)
            );
            assert_eq!(
                other.validate_participants(&scope),
                Err(ParallelTopologyError::ParticipantTopologyMismatch)
            );
        }
        for local in 0..4 {
            let peer = ValidatedParallelTopology::new(
                t.topology_id(),
                4,
                ParallelRankId::new(local),
                plan,
            )
            .unwrap();
            assert_eq!(peer.validate_participants(&scope), Ok(()));
            assert_eq!(peer.tensor_participants(0).unwrap(), scope);
            assert_eq!(peer.participants(), t.participants());
        }
    }

    #[test]
    fn dp1_pp2_tp2_ep2_metadata_and_scopes_are_deterministic() {
        let plan = ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap();
        let t = topology(4, plan).unwrap();
        let mut scopes = t.execution_scopes(0).unwrap();
        let ids = |set: &ParticipantSet| set.iter().map(ParallelRankId::get).collect::<Vec<_>>();
        assert_eq!(scopes.plan(), plan);
        assert_eq!(scopes.kv_participants().as_participants().world_size(), 4);
        assert_eq!(
            ids(scopes.kv_participants().as_participants()),
            [0, 1, 2, 3]
        );
        for stage in 0..2 {
            let tp = scopes.tensor_collective_participants(stage, 0).unwrap();
            assert_eq!(ids(tp.as_participants()), [stage * 2, stage * 2 + 1]);
            assert_eq!(
                tp.as_participants(),
                &t.tensor_stage_participants(0, stage).unwrap()
            );
            assert_eq!(
                scopes.validate_kv_participants(tp.as_participants()),
                Err(ParallelTopologyError::ScopeKindMismatch)
            );
            for tensor in 0..2 {
                let coordinate = t.coordinate(0, stage, tensor).unwrap();
                let global = t.rank_at(0, stage, tensor).unwrap();
                assert_eq!(coordinate.replica(), 0);
                assert_eq!(coordinate.stage(), stage);
                assert_eq!(coordinate.tensor(), tensor);
                assert_eq!(coordinate.topology_id(), t.topology_id());
                assert_eq!(t.coordinate_of(global), Ok(coordinate));
                assert_eq!(t.rank_of(coordinate), Ok(global));
                assert_eq!(scopes.kv_owner(coordinate), Ok(global));
            }
            assert_eq!(scopes.expert_dispatch_members(stage, 0).unwrap(), None);
        }
        let groups = [0, 1].map(|stage| {
            ExpertDispatchMembers::new(
                &t,
                0,
                stage,
                ParallelRankId::new(100 + stage * 10),
                [101 + stage * 10, 100 + stage * 10].map(ParallelRankId::new),
            )
            .unwrap()
        });
        // Attachment order is not identity; member order is dispatch-slot identity.
        let mut reverse = scopes.clone();
        for group in groups.iter().rev() {
            reverse
                .attach_expert_dispatch_members(group.clone())
                .unwrap();
        }
        for group in &groups {
            scopes
                .attach_expert_dispatch_members(group.clone())
                .unwrap();
        }
        assert_eq!(scopes, reverse);
        assert_eq!(
            scopes.validate_kv_participants(scopes.kv_participants().as_participants()),
            Ok(())
        );
        for (stage, group) in groups.iter().enumerate() {
            assert_eq!(
                scopes.expert_dispatch_members(stage as u32, 0),
                Ok(Some(group))
            );
            assert_eq!(group.len(), 2);
            assert_eq!(
                group.iter().map(ParallelRankId::get).collect::<Vec<_>>(),
                [101 + stage as u32 * 10, 100 + stage as u32 * 10]
            );
            for member in group.iter() {
                assert!(!scopes.kv_participants().contains(member));
                assert_eq!(
                    t.coordinate_of(member),
                    Err(ParallelTopologyError::RankOutOfBounds)
                );
            }
        }
        assert_eq!(
            ids(scopes.kv_participants().as_participants()),
            [0, 1, 2, 3]
        );
        assert_eq!(
            topology(8, plan),
            Err(ParallelTopologyError::UnsupportedParallelism)
        );
    }

    #[test]
    fn execution_scope_identity_includes_epoch_world_and_all_factors_not_local_rank() {
        let plan = ParallelismPlan::validated(2, 2, 2, 1, 1, 2).unwrap();
        let t = topology(8, plan).unwrap();
        let scopes = t.execution_scopes(1).unwrap();
        assert_eq!(
            scopes
                .kv_participants()
                .iter()
                .map(ParallelRankId::get)
                .collect::<Vec<_>>(),
            [4, 5, 6, 7]
        );
        assert_eq!(
            scopes.kv_owner(t.coordinate(1, 1, 0).unwrap()),
            Ok(ParallelRankId::new(6))
        );
        assert_eq!(
            scopes.kv_owner(t.coordinate(0, 1, 0).unwrap()),
            Err(ParallelTopologyError::ScopeReplicaMismatch)
        );
        assert_eq!(
            scopes.validate_kv_participants(
                t.execution_scopes(0)
                    .unwrap()
                    .kv_participants()
                    .as_participants()
            ),
            Err(ParallelTopologyError::ScopeKindMismatch)
        );
        for local in 0..8 {
            let peer = ValidatedParallelTopology::new(
                t.topology_id(),
                8,
                ParallelRankId::new(local),
                plan,
            )
            .unwrap();
            assert_eq!(scopes.validate_topology(&peer), Ok(()));
            assert_eq!(scopes, peer.execution_scopes(1).unwrap());
            assert_eq!(t.coordinate(1, 1, 0), peer.coordinate(1, 1, 0));
        }
        for (epoch, world, other_plan) in [
            (2, 8, plan),
            (1, 4, ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap()),
            (1, 8, ParallelismPlan::validated(1, 2, 2, 1, 1, 4).unwrap()),
            (1, 8, ParallelismPlan::validated(2, 1, 2, 1, 1, 4).unwrap()),
            (1, 8, ParallelismPlan::validated(2, 2, 3, 1, 1, 2).unwrap()),
        ] {
            let other = ValidatedParallelTopology::new(
                ParallelTopologyId::new(epoch),
                world,
                ParallelRankId::new(0),
                other_plan,
            )
            .unwrap();
            let coordinate = other.coordinate(0, 0, 0).unwrap();
            let mismatch = Err(ParallelTopologyError::ParticipantTopologyMismatch);
            assert_eq!(scopes.validate_topology(&other), mismatch);
            assert_eq!(
                scopes.validate_kv_participants(
                    other
                        .execution_scopes(0)
                        .unwrap()
                        .kv_participants()
                        .as_participants()
                ),
                mismatch
            );
            assert_eq!(
                scopes.kv_owner(coordinate),
                Err(ParallelTopologyError::ParticipantTopologyMismatch)
            );
            assert_eq!(
                t.rank_of(coordinate),
                Err(ParallelTopologyError::ParticipantTopologyMismatch)
            );
            let group = ExpertDispatchMembers::new(
                &other,
                0,
                0,
                ParallelRankId::new(100),
                (100..100 + other_plan.expert_parallel as u32).map(ParallelRankId::new),
            )
            .unwrap();
            assert_eq!(
                scopes.clone().attach_expert_dispatch_members(group),
                mismatch
            );
        }
        assert_eq!(
            t.execution_scopes(2),
            Err(ParallelTopologyError::ReplicaOutOfBounds)
        );
        assert_eq!(
            t.coordinate(0, 2, 0),
            Err(ParallelTopologyError::RankOutOfBounds)
        );
        assert_eq!(
            t.coordinate(0, 0, 2),
            Err(ParallelTopologyError::RankOutOfBounds)
        );
        assert_eq!(
            scopes.tensor_collective_participants(2, 1),
            Err(ParallelTopologyError::RankOutOfBounds)
        );
        assert_eq!(
            scopes.expert_dispatch_members(0, 0),
            Err(ParallelTopologyError::ScopeReplicaMismatch)
        );
    }

    #[test]
    fn expert_attachments_reject_mesh_owners_duplicates_and_cross_stage_reuse_atomically() {
        let t = topology(8, ParallelismPlan::validated(2, 2, 2, 1, 1, 2).unwrap()).unwrap();
        let group = |stage, source, members: Vec<u32>| {
            ExpertDispatchMembers::new(
                &t,
                0,
                stage,
                ParallelRankId::new(source),
                members.into_iter().map(ParallelRankId::new),
            )
        };
        for (members, source, expected) in [
            (vec![], 100, ParallelTopologyError::EmptyParticipants),
            (
                vec![100, 100],
                100,
                ParallelTopologyError::DuplicateParticipant,
            ),
            (
                vec![100, 0],
                100,
                ParallelTopologyError::ExpertDispatchMemberOverlap,
            ),
            (
                vec![100, 7],
                100,
                ParallelTopologyError::ExpertDispatchMemberOverlap,
            ),
            (
                vec![100, 101],
                999,
                ParallelTopologyError::ExpertSourceNotMember,
            ),
            (vec![100], 100, ParallelTopologyError::ExpertDegreeMismatch),
            (
                vec![100, 101, 102],
                100,
                ParallelTopologyError::ExpertDegreeMismatch,
            ),
        ] {
            assert_eq!(group(0, source, members), Err(expected));
        }
        let mut scopes = t.execution_scopes(0).unwrap();
        let first = group(0, 100, vec![100, 101]).unwrap();
        scopes
            .attach_expert_dispatch_members(first.clone())
            .unwrap();
        let before = scopes.clone();
        assert_eq!(
            scopes.attach_expert_dispatch_members(first),
            Err(ParallelTopologyError::ExpertDispatchMembersAlreadyAttached)
        );
        assert_eq!(
            scopes.attach_expert_dispatch_members(group(1, 101, vec![101, 102]).unwrap()),
            Err(ParallelTopologyError::ExpertDispatchMemberOverlap)
        );
        assert_eq!(scopes, before);
        scopes
            .attach_expert_dispatch_members(group(1, 102, vec![102, 103]).unwrap())
            .unwrap();
        let other_replica = ExpertDispatchMembers::new(
            &t,
            1,
            0,
            ParallelRankId::new(200),
            [200, 201].map(ParallelRankId::new),
        )
        .unwrap();
        assert_eq!(
            scopes.attach_expert_dispatch_members(other_replica),
            Err(ParallelTopologyError::ScopeReplicaMismatch)
        );
    }

    #[test]
    fn dp_replica_composition_requires_disjoint_kv_and_expert_owners() {
        let t = topology(8, ParallelismPlan::validated(2, 2, 2, 1, 1, 2).unwrap()).unwrap();
        let mut left = t.execution_scopes(0).unwrap();
        let mut right = t.execution_scopes(1).unwrap();
        let group = |replica, stage, base| {
            ExpertDispatchMembers::new(
                &t,
                replica,
                stage,
                ParallelRankId::new(base),
                [base, base + 1].map(ParallelRankId::new),
            )
            .unwrap()
        };
        left.attach_expert_dispatch_members(group(0, 0, 100))
            .unwrap();
        right
            .attach_expert_dispatch_members(group(1, 0, 200))
            .unwrap();
        assert_eq!(left.validate_disjoint_owners(&right), Ok(()));
        assert_eq!(right.validate_disjoint_owners(&left), Ok(()));
        assert_eq!(
            left.validate_disjoint_owners(&left),
            Err(ParallelTopologyError::KvOwnerOverlap)
        );
        right
            .attach_expert_dispatch_members(group(1, 1, 101))
            .unwrap();
        assert_eq!(
            left.validate_disjoint_owners(&right),
            Err(ParallelTopologyError::ExpertDispatchMemberOverlap)
        );
        assert_eq!(
            right.validate_disjoint_owners(&left),
            Err(ParallelTopologyError::ExpertDispatchMemberOverlap)
        );
        let foreign = ValidatedParallelTopology::new(
            ParallelTopologyId::new(2),
            8,
            ParallelRankId::new(0),
            t.plan(),
        )
        .unwrap();
        assert_eq!(
            left.validate_disjoint_owners(&foreign.execution_scopes(1).unwrap()),
            Err(ParallelTopologyError::ParticipantTopologyMismatch)
        );
    }
}

#[cfg(test)]
mod external_expert_source_tests {
    use super::*;

    fn external_topology(tp: usize) -> ValidatedParallelTopology {
        ValidatedParallelTopology::new(
            ParallelTopologyId::new(31),
            (4 * tp) as u32,
            ParallelRankId::new(0),
            ParallelismPlan::validated(2, tp, 2, 1, 1, 2).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn external_stage_source_is_exact_and_never_an_expert_or_extra_kv_owner() {
        let topology = external_topology(1);
        for replica in 0..2 {
            let mut scopes = topology.execution_scopes(replica).unwrap();
            let kv = scopes.kv_participants().clone();
            for stage in 0..2 {
                let source = topology.rank_at(replica, stage, 0).unwrap();
                let base = 10 + replica * 4 + stage * 2;
                let members = [base + 1, base].map(ParallelRankId::new);
                let group = ExpertDispatchMembers::new_with_scope(
                    &topology,
                    replica,
                    stage,
                    ExpertSourceScope::ExternalStage,
                    source,
                    members,
                )
                .unwrap();
                assert_eq!(group.source_scope(), ExpertSourceScope::ExternalStage);
                assert_eq!(group.source_rank(), source);
                assert_eq!(group.iter().collect::<Vec<_>>(), members);
                assert!(!group.contains(source));
                assert!(kv.contains(source));
                assert!(members.iter().all(|rank| !kv.contains(*rank)));
                scopes.attach_expert_dispatch_members(group).unwrap();
                assert_eq!(scopes.kv_participants(), &kv);
                for wrong in [source.get() ^ 1, source.get() ^ 2, base, u32::MAX] {
                    assert_eq!(
                        ExpertDispatchMembers::new_with_scope(
                            &topology,
                            replica,
                            stage,
                            ExpertSourceScope::ExternalStage,
                            ParallelRankId::new(wrong),
                            members,
                        ),
                        Err(ParallelTopologyError::ExpertSourceNotStageCaller),
                    );
                }
            }
        }
    }

    #[test]
    fn external_scope_rejects_worker_overlap_and_does_not_enable_tp_ep() {
        let topology = external_topology(1);
        for members in [[0, 10], [2, 10], [10, 10], [10, 11]] {
            let source = ParallelRankId::new(0);
            let scope = if members == [10, 11] {
                ExpertSourceScope::Member
            } else {
                ExpertSourceScope::ExternalStage
            };
            assert!(
                ExpertDispatchMembers::new_with_scope(
                    &topology,
                    0,
                    0,
                    scope,
                    source,
                    members.map(ParallelRankId::new),
                )
                .is_err()
            );
        }
        let tp = external_topology(2);
        assert_eq!(
            ExpertDispatchMembers::new_with_scope(
                &tp,
                0,
                0,
                ExpertSourceScope::ExternalStage,
                ParallelRankId::new(0),
                [10, 11].map(ParallelRankId::new),
            ),
            Err(ParallelTopologyError::UnsupportedParallelism),
        );
    }

    #[test]
    fn legacy_constructor_remains_member_only() {
        let topology = external_topology(1);
        let members = [11, 10].map(ParallelRankId::new);
        let legacy = ExpertDispatchMembers::new(&topology, 0, 0, members[1], members).unwrap();
        assert_eq!(legacy.source_scope(), ExpertSourceScope::Member);
        assert_eq!(
            legacy,
            ExpertDispatchMembers::new_with_scope(
                &topology,
                0,
                0,
                ExpertSourceScope::Member,
                members[1],
                members,
            )
            .unwrap()
        );
        assert_eq!(
            ExpertDispatchMembers::new(&topology, 0, 0, ParallelRankId::new(0), members),
            Err(ParallelTopologyError::ExpertSourceNotMember),
        );
    }
}
