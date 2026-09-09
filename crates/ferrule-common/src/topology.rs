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
        // A single-rank world must obey the same pure-DP contract as larger worlds.
        if u32::try_from(plan.data_parallel) != Ok(world_size)
            || [
                plan.tensor_parallel,
                plan.expert_parallel,
                plan.sequence_parallel,
                plan.context_parallel,
                plan.pipeline_parallel,
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
            world_size: self.world_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticipantSet {
    world_size: u32,
}
impl ParticipantSet {
    pub const fn world_size(self) -> u32 {
        self.world_size
    }
    pub const fn contains(self, rank: ParallelRankId) -> bool {
        rank.get() < self.world_size
    }
    pub fn iter(self) -> impl Iterator<Item = ParallelRankId> {
        (0..self.world_size).map(ParallelRankId::new)
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
    fn rejects_non_dp_factors_for_every_world_size() {
        for world in [1, 2] {
            for index in 1..6 {
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
}
