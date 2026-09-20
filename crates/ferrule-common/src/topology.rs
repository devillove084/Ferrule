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
        if u32::try_from(mesh_size) != Ok(world_size)
            || [
                plan.expert_parallel,
                plan.sequence_parallel,
                plan.context_parallel,
            ]
            .iter()
            .any(|factor| *factor != 1)
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
    fn sp_cp_and_independent_ep_are_not_mesh_factors() {
        for plan in [
            ParallelismPlan::validated(1, 1, 1, 2, 1, 1).unwrap(),
            ParallelismPlan::validated(1, 1, 1, 1, 2, 1).unwrap(),
            ParallelismPlan::validated(1, 1, 2, 1, 1, 1).unwrap(),
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
            // PP is a supported mesh factor now; SPCP/EP remain rejected here.
            for index in 1..5 {
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
        // Keep this legacy matrix focused on EP/SP/CP. PP has dedicated
        // rank-layout coverage above.
        for index in 2..5 {
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
}
