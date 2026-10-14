use crate::{
    ExpertDispatchMembers, ParallelRankId, ParallelTopologyId, ParallelismPlan,
    ValidatedParallelTopology, topology::ExpertSourceScope,
};
fn rank(n: u32) -> ParallelRankId {
    ParallelRankId::new(n)
}
fn topology(tp: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(707),
        (4 * tp) as u32,
        rank(0),
        ParallelismPlan::validated(2, tp, 2, 1, 1, 2).unwrap(),
    )
    .unwrap()
}
#[test]
fn explicit_tensor_source_authenticates_stage_replica_and_keeps_experts_out_of_kv() {
    let t = topology(2);
    for replica in [0, 1] {
        let mut scopes = t.execution_scopes(replica).unwrap();
        let before = scopes.kv_participants().clone();
        for stage in [0, 1] {
            let source = t.rank_at(replica, stage, 1).unwrap();
            let members = [
                rank(101 + 100 * replica + 10 * stage),
                rank(99 + 100 * replica + 10 * stage),
            ];
            let group = ExpertDispatchMembers::new_with_scope(
                &t,
                replica,
                stage,
                ExpertSourceScope::ExternalTensorSource,
                source,
                members,
            )
            .unwrap();
            assert_eq!(group.iter().collect::<Vec<_>>(), members);
            scopes
                .attach_expert_dispatch_members(group.clone())
                .unwrap();
            assert_eq!(
                scopes.expert_dispatch_members(stage, replica).unwrap(),
                Some(&group)
            );
            assert_eq!(scopes.kv_participants(), &before);
            assert!(
                members
                    .iter()
                    .all(|r| !before.as_participants().contains(*r))
            );
        }
    }
    for source in [rank(0), rank(1), rank(4), rank(5), rank(99)] {
        assert!(
            ExpertDispatchMembers::new_with_scope(
                &t,
                0,
                1,
                ExpertSourceScope::ExternalTensorSource,
                source,
                [rank(99), rank(101)]
            )
            .is_err()
        );
    }
    for members in [
        vec![],
        vec![rank(99), rank(99)],
        vec![rank(1), rank(99)],
        vec![rank(99)],
    ] {
        assert!(
            ExpertDispatchMembers::new_with_scope(
                &t,
                0,
                1,
                ExpertSourceScope::ExternalTensorSource,
                rank(3),
                members
            )
            .is_err()
        );
    }
}
#[test]
fn external_stage_and_member_legacy_constraints_are_not_relaxed() {
    let t = topology(2);
    assert!(
        ExpertDispatchMembers::new_with_scope(
            &t,
            0,
            0,
            ExpertSourceScope::ExternalStage,
            rank(0),
            [rank(99), rank(101)]
        )
        .is_err()
    );
    assert!(
        ExpertDispatchMembers::new_with_scope(
            &t,
            0,
            0,
            ExpertSourceScope::Member,
            rank(0),
            [rank(99), rank(101)]
        )
        .is_err()
    );
    let t = topology(1);
    assert!(
        ExpertDispatchMembers::new_with_scope(
            &t,
            0,
            0,
            ExpertSourceScope::ExternalStage,
            rank(0),
            [rank(99), rank(101)]
        )
        .is_ok()
    );
}
