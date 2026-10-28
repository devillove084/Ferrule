use crate::topology::ExpertSourceScope;
use crate::{
    ExpertDispatchMembers, ParallelRankId, ParallelTopologyError, ParallelTopologyId,
    ParallelismPlan, ParticipantSet, PipelineStageLayerRange, TopologyPlacement,
    ValidatedParallelTopology,
};

fn rank(value: u32) -> ParallelRankId {
    ParallelRankId::new(value)
}

fn layer_range(stage: u32, start: usize, end: usize) -> PipelineStageLayerRange {
    PipelineStageLayerRange::new(stage, start, end).unwrap()
}

fn pp2_tp2() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new_with_stage_ranges(
        ParallelTopologyId::new(900),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap(),
        4,
        [layer_range(0, 0, 2), layer_range(1, 2, 4)],
    )
    .unwrap()
}

#[test]
fn legacy_new_without_layer_metadata_remains_unsealed() {
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(904),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap(),
    )
    .unwrap();
    assert!(topology.stage_ranges().is_none());
}

#[test]
fn pp2_tp2_seals_stage_ranges_and_exact_mesh_scopes() {
    let topology = pp2_tp2();
    let ranges = topology.stage_ranges().unwrap();
    assert_eq!(ranges.total_layers(), 4);
    assert_eq!(
        ranges.iter().map(|range| range.stage()).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(ranges.get(0).unwrap().start(), 0);
    assert_eq!(ranges.get(1).unwrap().end(), 4);
    assert_eq!(topology.validate_stage_ranges(ranges), Ok(()));

    let scopes = topology.execution_scopes(0).unwrap();
    assert_eq!(
        scopes.kv_participants().iter().collect::<Vec<_>>(),
        [rank(0), rank(1), rank(2), rank(3)]
    );
    for stage in 0..2 {
        let collective = scopes.tensor_collective_participants(stage, 0).unwrap();
        let expected = topology.tensor_stage_participants(0, stage).unwrap();
        assert_eq!(collective.as_participants(), &expected);
        assert_eq!(
            topology.validate_tensor_stage_participants(0, stage, collective.as_participants()),
            Ok(())
        );
        for lane in 0..2 {
            let coordinate = topology.coordinate(0, stage, lane).unwrap();
            assert_eq!(
                topology.rank_of(coordinate),
                Ok(topology.rank_at(0, stage, lane).unwrap())
            );
            assert_eq!(
                topology
                    .validate_tensor_lane(coordinate, topology.rank_at(0, stage, lane).unwrap()),
                Ok(())
            );
        }
    }
    assert_eq!(scopes.validate_topology(&topology), Ok(()));
}

#[test]
fn stage_partition_rejects_zero_gap_overlap_missing_out_of_range_and_overflow() {
    let plan = ParallelismPlan::validated(1, 1, 1, 1, 1, 2).unwrap();
    let valid = |ranges: Vec<PipelineStageLayerRange>| {
        ValidatedParallelTopology::new_with_stage_ranges(
            ParallelTopologyId::new(901),
            2,
            rank(0),
            plan,
            4,
            ranges,
        )
    };

    assert_eq!(
        valid(vec![layer_range(0, 0, 1), layer_range(1, 2, 4)]),
        Err(ParallelTopologyError::LayerRangeGap)
    );
    assert_eq!(
        valid(vec![layer_range(0, 0, 3), layer_range(1, 2, 4)]),
        Err(ParallelTopologyError::LayerRangeOverlap)
    );
    assert_eq!(
        valid(vec![layer_range(0, 0, 4)]),
        Err(ParallelTopologyError::MissingStage)
    );
    assert_eq!(
        valid(vec![layer_range(0, 0, 2), layer_range(2, 2, 4)]),
        Err(ParallelTopologyError::StageOutOfBounds)
    );
    assert_eq!(
        valid(vec![layer_range(0, 0, 2), layer_range(0, 2, 4)]),
        Err(ParallelTopologyError::DuplicateStage)
    );
    assert_eq!(
        valid(vec![layer_range(0, 0, 2), layer_range(1, 2, 5)]),
        Err(ParallelTopologyError::LayerRangeOutOfBounds)
    );
    assert_eq!(
        ValidatedParallelTopology::new_with_stage_ranges(
            ParallelTopologyId::new(901),
            2,
            rank(0),
            plan,
            0,
            [layer_range(0, 0, 1), layer_range(1, 1, 2)],
        ),
        Err(ParallelTopologyError::ZeroLayerCount)
    );
    assert_eq!(
        PipelineStageLayerRange::new(0, 1, 1),
        Err(ParallelTopologyError::EmptyLayerRange)
    );
    assert_eq!(
        PipelineStageLayerRange::new(0, 2, 1),
        Err(ParallelTopologyError::ReversedLayerRange)
    );
    assert_eq!(
        PipelineStageLayerRange::from_len(0, usize::MAX, 1),
        Err(ParallelTopologyError::LayerRangeOverflow)
    );
}

#[test]
fn tp_scope_and_lane_identity_do_not_accept_other_mesh_members() {
    let topology = pp2_tp2();
    let wrong_stage_scope = ParticipantSet::new(&topology, [rank(0), rank(2)]).unwrap();
    assert_eq!(
        topology.validate_tensor_stage_participants(0, 0, &wrong_stage_scope),
        Err(ParallelTopologyError::ScopeKindMismatch)
    );
    assert_eq!(
        topology.validate_kv_participants(0, &wrong_stage_scope),
        Err(ParallelTopologyError::ScopeKindMismatch)
    );
    assert_eq!(
        topology.validate_tensor_lane(topology.coordinate(0, 1, 0).unwrap(), rank(1)),
        Err(ParallelTopologyError::RankLaneMismatch)
    );

    let placement = topology.placement(0, 1, 1, 7).unwrap();
    assert_eq!(placement.logical_rank(), rank(3));
    assert_eq!(placement.device_ordinal(), 7);
    assert_eq!(placement.coordinate().replica(), 0);
    assert_eq!(topology.validate_placement(&placement), Ok(()));
    let other =
        ValidatedParallelTopology::new(ParallelTopologyId::new(902), 4, rank(0), topology.plan())
            .unwrap();
    assert_eq!(
        other.validate_placement(&placement),
        Err(ParallelTopologyError::PlacementTopologyMismatch)
    );
    let _: TopologyPlacement = placement;
}

#[test]
fn ep_attachment_is_explicit_and_isolated_from_kv_and_tp() {
    let topology = pp2_tp2();
    let mut scopes = topology.execution_scopes(0).unwrap();
    let kv_before = scopes.kv_participants().clone();
    let attachment = ExpertDispatchMembers::new_with_scope(
        &topology,
        0,
        1,
        ExpertSourceScope::ExternalTensorSource,
        rank(3),
        [rank(101), rank(100)],
    )
    .unwrap();
    assert_eq!(attachment.source_rank(), rank(3));
    assert!(!attachment.contains(attachment.source_rank()));
    assert!(attachment.iter().all(|member| !kv_before.contains(member)));
    assert!(attachment.iter().all(|member| {
        !topology
            .tensor_stage_participants(0, 1)
            .unwrap()
            .contains(member)
    }));
    scopes
        .attach_expert_dispatch_members(attachment.clone())
        .unwrap();
    assert_eq!(scopes.kv_participants(), &kv_before);
    assert_eq!(
        scopes.expert_dispatch_members(1, 0).unwrap(),
        Some(&attachment)
    );
    assert_eq!(scopes.validate_topology(&topology), Ok(()));

    assert_eq!(
        ExpertDispatchMembers::new_with_scope(
            &topology,
            0,
            1,
            ExpertSourceScope::ExternalTensorSource,
            rank(1),
            [rank(102), rank(103)],
        ),
        Err(ParallelTopologyError::ExpertSourceNotStageCaller)
    );
    assert_eq!(
        ExpertDispatchMembers::new_with_scope(
            &topology,
            0,
            1,
            ExpertSourceScope::ExternalTensorSource,
            rank(3),
            [rank(0), rank(100)],
        ),
        Err(ParallelTopologyError::ExpertDispatchMemberOverlap)
    );
    assert_eq!(
        ExpertDispatchMembers::new_with_scope(
            &topology,
            0,
            1,
            ExpertSourceScope::ExternalTensorSource,
            rank(3),
            [rank(100), rank(100)],
        ),
        Err(ParallelTopologyError::DuplicateParticipant)
    );
    assert_eq!(
        ExpertDispatchMembers::new_with_scope(
            &topology,
            0,
            2,
            ExpertSourceScope::ExternalTensorSource,
            rank(3),
            [rank(100), rank(101)],
        ),
        Err(ParallelTopologyError::RankOutOfBounds)
    );
}

#[test]
fn expert_owner_reuse_and_data_replica_crossing_fail_closed() {
    let topology = pp2_tp2();
    let mut scopes = topology.execution_scopes(0).unwrap();
    scopes
        .attach_expert_dispatch_members(
            ExpertDispatchMembers::new_with_scope(
                &topology,
                0,
                0,
                ExpertSourceScope::ExternalTensorSource,
                rank(0),
                [rank(100), rank(101)],
            )
            .unwrap(),
        )
        .unwrap();
    let reused = ExpertDispatchMembers::new_with_scope(
        &topology,
        0,
        1,
        ExpertSourceScope::ExternalTensorSource,
        rank(2),
        [rank(101), rank(102)],
    )
    .unwrap();
    assert_eq!(
        scopes.attach_expert_dispatch_members(reused),
        Err(ParallelTopologyError::ExpertDispatchMemberOverlap)
    );

    let replica_topology = ValidatedParallelTopology::new_with_stage_ranges(
        ParallelTopologyId::new(903),
        4,
        rank(0),
        ParallelismPlan::validated(2, 1, 2, 1, 1, 2).unwrap(),
        4,
        [layer_range(0, 0, 2), layer_range(1, 2, 4)],
    )
    .unwrap();
    let foreign_attachment = ExpertDispatchMembers::new_with_scope(
        &replica_topology,
        1,
        0,
        ExpertSourceScope::ExternalStage,
        rank(2),
        [rank(200), rank(201)],
    )
    .unwrap();
    let mut replica_zero = replica_topology.execution_scopes(0).unwrap();
    assert_eq!(
        replica_zero.attach_expert_dispatch_members(foreign_attachment),
        Err(ParallelTopologyError::ScopeReplicaMismatch)
    );
    let replica_one = replica_topology.execution_scopes(1).unwrap();
    assert_eq!(
        replica_zero.kv_participants().iter().collect::<Vec<_>>(),
        [rank(0), rank(1)]
    );
    assert_eq!(
        replica_one.kv_participants().iter().collect::<Vec<_>>(),
        [rank(2), rank(3)]
    );
}
