use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId};
use ferrule_runtime::parallel::collective::{
    HostCollectiveDescriptor, HostCollectiveError, HostCollectiveGroup, HostCollectiveKind,
    HostCollectiveLimits, HostCollectivePoll, HostCollectiveStatus,
};

fn rank(value: u32) -> ParallelRankId {
    ParallelRankId::new(value)
}

fn limits() -> HostCollectiveLimits {
    HostCollectiveLimits {
        max_ranks: 3,
        max_elements_per_rank: 4,
        max_host_bytes: 1024,
    }
}

fn group() -> HostCollectiveGroup {
    HostCollectiveGroup::new(
        ParallelTopologyId::new(7),
        ParallelGroupId::new(11),
        vec![rank(9), rank(2), rank(6)],
        limits(),
    )
    .unwrap()
}

fn descriptor(kind: HostCollectiveKind, count: usize, sequence: u64) -> HostCollectiveDescriptor {
    HostCollectiveDescriptor {
        epoch: ParallelTopologyId::new(7),
        group: ParallelGroupId::new(11),
        transaction: ExecutionTransactionId::new(13).unwrap(),
        sequence,
        kind,
        count,
    }
}

fn take(
    group: &mut HostCollectiveGroup,
    member: u32,
    expected: HostCollectiveDescriptor,
) -> Vec<f32> {
    match group.take_result(rank(member)).unwrap() {
        HostCollectivePoll::Ready(result) => {
            assert_eq!(result.descriptor, expected);
            result.values
        }
        HostCollectivePoll::Pending => panic!("expected ready output"),
    }
}

/// Rejection preserves all externally observable service state and returns the
/// very same input allocation, allowing an allocation-free corrected retry.
fn reject(
    group: &mut HostCollectiveGroup,
    member: u32,
    descriptor: HostCollectiveDescriptor,
    payload: Vec<f32>,
    expected: HostCollectiveError,
) -> Vec<f32> {
    let before = (
        group.active_descriptor(),
        group.sequence_watermark(),
        group.status(),
        group.owned_host_bytes(),
        group.reserved_host_bytes(),
    );
    let pointer = payload.as_ptr();
    let capacity = payload.capacity();
    let bits: Vec<_> = payload.iter().map(|v| v.to_bits()).collect();
    let rejected = group.submit(rank(member), descriptor, payload).unwrap_err();
    assert_eq!(rejected.error, expected);
    assert_eq!(rejected.payload.as_ptr(), pointer);
    assert_eq!(rejected.payload.capacity(), capacity);
    assert_eq!(
        rejected
            .payload
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
    assert_eq!(
        (
            group.active_descriptor(),
            group.sequence_watermark(),
            group.status(),
            group.owned_host_bytes(),
            group.reserved_host_bytes(),
        ),
        before
    );
    rejected.payload
}

#[test]
fn readiness_requires_every_member_and_results_can_be_consumed_in_any_rank_order() {
    let mut group = group();
    assert_eq!(HostCollectiveGroup::MAX_INFLIGHT_OPERATIONS, 1);
    assert_eq!(group.members(), &[rank(9), rank(2), rank(6)]);
    assert_eq!(group.limits(), limits());
    assert_eq!(group.status(), None);
    assert_eq!(
        group.take_result(rank(9)),
        Err(HostCollectiveError::NoOperation)
    );
    let op = descriptor(HostCollectiveKind::AllReduceSumF32, 2, 0);
    assert_eq!(
        group.submit(rank(6), op, vec![5.0, 6.0]).unwrap(),
        HostCollectiveStatus::Pending
    );
    for member in [9, 2, 6] {
        assert_eq!(
            group.take_result(rank(member)).unwrap(),
            HostCollectivePoll::Pending
        );
    }
    assert_eq!(
        group.submit(rank(9), op, vec![1.0, 2.0]).unwrap(),
        HostCollectiveStatus::Pending
    );
    assert_eq!(
        group.submit(rank(2), op, vec![3.0, 4.0]).unwrap(),
        HostCollectiveStatus::Ready
    );
    assert_eq!(group.status(), Some(HostCollectiveStatus::Ready));
    for member in [2, 6, 9] {
        assert_eq!(take(&mut group, member, op), vec![9.0, 12.0]);
    }
    assert_eq!(group.status(), None);
    assert_eq!(group.owned_host_bytes(), 0);
    assert_eq!(group.reserved_host_bytes(), 0);
    assert_eq!(group.sequence_watermark(), Some(0));
}

#[test]
fn descriptor_mismatch_duplicate_and_wrong_payload_are_retryable_without_mutation() {
    let mut group = group();
    let op = descriptor(HostCollectiveKind::AllReduceSumF32, 1, 10);
    group.submit(rank(6), op, vec![3.0]).unwrap();
    for changed in [
        HostCollectiveDescriptor {
            transaction: ExecutionTransactionId::new(99).unwrap(),
            ..op
        },
        HostCollectiveDescriptor {
            kind: HostCollectiveKind::AllGatherF32,
            ..op
        },
        HostCollectiveDescriptor { count: 2, ..op },
    ] {
        reject(
            &mut group,
            9,
            changed,
            vec![1.0],
            HostCollectiveError::DescriptorMismatch,
        );
    }
    reject(
        &mut group,
        9,
        HostCollectiveDescriptor {
            epoch: ParallelTopologyId::new(8),
            ..op
        },
        vec![1.0],
        HostCollectiveError::WrongEpoch,
    );
    reject(
        &mut group,
        9,
        HostCollectiveDescriptor {
            group: ParallelGroupId::new(12),
            ..op
        },
        vec![1.0],
        HostCollectiveError::WrongGroup,
    );
    reject(
        &mut group,
        0,
        op,
        vec![1.0],
        HostCollectiveError::WrongRank(rank(0)),
    );
    assert_eq!(
        group.take_result(rank(0)),
        Err(HostCollectiveError::WrongRank(rank(0)))
    );
    reject(
        &mut group,
        6,
        op,
        vec![999.0],
        HostCollectiveError::DuplicateSubmission(rank(6)),
    );
    reject(
        &mut group,
        9,
        op,
        vec![1.0, 2.0],
        HostCollectiveError::CountMismatch {
            expected: 1,
            actual: 2,
        },
    );
    let payload = reject(
        &mut group,
        9,
        HostCollectiveDescriptor {
            transaction: ExecutionTransactionId::new(99).unwrap(),
            ..op
        },
        vec![1.0],
        HostCollectiveError::DescriptorMismatch,
    );
    group.submit(rank(9), op, payload).unwrap();
    group.submit(rank(2), op, vec![2.0]).unwrap();
    for member in [9, 2, 6] {
        assert_eq!(take(&mut group, member, op), vec![6.0]);
    }
}

#[test]
fn allgather_uses_member_vector_order_not_arrival_or_numeric_rank_order() {
    let mut group = group();
    let op = descriptor(HostCollectiveKind::AllGatherF32, 2, 0);
    group.submit(rank(2), op, vec![20.0, 21.0]).unwrap();
    group.submit(rank(6), op, vec![60.0, 61.0]).unwrap();
    group.submit(rank(9), op, vec![90.0, 91.0]).unwrap();
    for member in [6, 9, 2] {
        assert_eq!(
            take(&mut group, member, op),
            vec![90.0, 91.0, 20.0, 21.0, 60.0, 61.0]
        );
    }
}

#[test]
fn allreduce_is_bitwise_deterministic_across_every_arrival_permutation() {
    for order in [
        [9, 2, 6],
        [9, 6, 2],
        [2, 9, 6],
        [2, 6, 9],
        [6, 9, 2],
        [6, 2, 9],
    ] {
        let mut group = group();
        let op = descriptor(HostCollectiveKind::AllReduceSumF32, 2, 0);
        for member in order {
            let value = match member {
                9 => 1.0e20_f32,
                2 => -1.0e20_f32,
                6 => 3.25,
                _ => unreachable!(),
            };
            group.submit(rank(member), op, vec![value, -0.0]).unwrap();
        }
        for member in [9, 2, 6] {
            let output = take(&mut group, member, op);
            assert_eq!(output[0].to_bits(), 3.25_f32.to_bits());
            assert_eq!(output[1].to_bits(), (-0.0_f32).to_bits());
        }
    }
}

#[test]
fn physical_capacity_counts_spare_input_capacity_and_all_rank_outputs() {
    let config = HostCollectiveLimits {
        max_host_bytes: 120,
        ..limits()
    };
    let mut group = HostCollectiveGroup::new(
        ParallelTopologyId::new(7),
        ParallelGroupId::new(11),
        vec![rank(9), rank(2), rank(6)],
        config,
    )
    .unwrap();
    let op = descriptor(HostCollectiveKind::AllGatherF32, 2, 0);
    let memory = group.memory_for(op.kind, op.count).unwrap();
    assert_eq!(memory.max_input_bytes, 48);
    assert_eq!(memory.output_elements_per_rank, 6);
    assert_eq!(memory.output_bytes, 72);
    assert_eq!(memory.peak_bytes, 120);
    let oversized = descriptor(HostCollectiveKind::AllGatherF32, 3, 0);
    reject(
        &mut group,
        9,
        oversized,
        vec![1.0; 3],
        HostCollectiveError::HostCapacity {
            required: 156,
            limit: 120,
        },
    );
    reject(
        &mut group,
        9,
        descriptor(HostCollectiveKind::AllReduceSumF32, 5, 0),
        vec![1.0; 5],
        HostCollectiveError::CountLimit,
    );
    let mut excessive_capacity = Vec::with_capacity(5);
    excessive_capacity.extend_from_slice(&[1.0, 2.0]);
    reject(
        &mut group,
        9,
        op,
        excessive_capacity,
        HostCollectiveError::InputCapacityLimit,
    );
    for (index, member) in [6, 9, 2].into_iter().enumerate() {
        let mut payload = Vec::with_capacity(4);
        payload.extend_from_slice(&[1.0, 2.0]);
        group.submit(rank(member), op, payload).unwrap();
        assert_eq!(group.reserved_host_bytes(), 120);
        assert_eq!(
            group.owned_host_bytes(),
            if index == 2 {
                72
            } else {
                72 + (index + 1) * 16
            }
        );
    }
    take(&mut group, 9, op);
    assert_eq!(group.owned_host_bytes(), 48);
    assert_eq!(group.reserved_host_bytes(), 120);
    take(&mut group, 2, op);
    take(&mut group, 6, op);
    assert_eq!(group.owned_host_bytes(), 0);
    assert_eq!(group.reserved_host_bytes(), 0);
}

#[test]
fn backpressure_lasts_until_final_result_and_drain_does_not_allow_replay() {
    let mut group = group();
    let op = descriptor(HostCollectiveKind::AllReduceSumF32, 1, 4);
    let next = HostCollectiveDescriptor {
        sequence: 8,
        transaction: ExecutionTransactionId::new(14).unwrap(),
        ..op
    };
    group.submit(rank(9), op, vec![1.0]).unwrap();
    reject(
        &mut group,
        2,
        next,
        vec![2.0],
        HostCollectiveError::Backpressure,
    );
    reject(
        &mut group,
        2,
        HostCollectiveDescriptor { sequence: 3, ..op },
        vec![2.0],
        HostCollectiveError::StaleSequence {
            watermark: 4,
            received: 3,
        },
    );
    group.submit(rank(2), op, vec![2.0]).unwrap();
    group.submit(rank(6), op, vec![3.0]).unwrap();
    take(&mut group, 9, op);
    assert_eq!(
        group.take_result(rank(9)),
        Err(HostCollectiveError::ResultAlreadyTaken(rank(9)))
    );
    reject(
        &mut group,
        9,
        op,
        vec![1.0],
        HostCollectiveError::DuplicateSubmission(rank(9)),
    );
    reject(
        &mut group,
        9,
        next,
        vec![1.0],
        HostCollectiveError::Backpressure,
    );
    take(&mut group, 2, op);
    reject(
        &mut group,
        9,
        next,
        vec![1.0],
        HostCollectiveError::Backpressure,
    );
    take(&mut group, 6, op);
    reject(
        &mut group,
        9,
        op,
        vec![1.0],
        HostCollectiveError::StaleSequence {
            watermark: 4,
            received: 4,
        },
    );
    // A new transaction cannot reuse an old group sequence either.
    reject(
        &mut group,
        9,
        HostCollectiveDescriptor {
            sequence: 4,
            ..next
        },
        vec![1.0],
        HostCollectiveError::StaleSequence {
            watermark: 4,
            received: 4,
        },
    );
    assert_eq!(
        group.submit(rank(9), next, vec![1.0]).unwrap(),
        HostCollectiveStatus::Pending
    );
    assert_eq!(group.sequence_watermark(), Some(8));
}

#[test]
fn abort_checks_full_identity_frees_cpu_data_and_keeps_sequence_watermark() {
    let mut group = group();
    let op = descriptor(HostCollectiveKind::AllGatherF32, 1, 0);
    assert_eq!(group.abort(op), Err(HostCollectiveError::NoOperation));
    group.submit(rank(9), op, vec![1.0]).unwrap();
    let bytes = group.owned_host_bytes();
    for wrong in [
        HostCollectiveDescriptor { sequence: 1, ..op },
        HostCollectiveDescriptor {
            transaction: ExecutionTransactionId::new(22).unwrap(),
            ..op
        },
        HostCollectiveDescriptor {
            kind: HostCollectiveKind::AllReduceSumF32,
            ..op
        },
        HostCollectiveDescriptor { count: 2, ..op },
        HostCollectiveDescriptor {
            epoch: ParallelTopologyId::new(8),
            ..op
        },
        HostCollectiveDescriptor {
            group: ParallelGroupId::new(12),
            ..op
        },
    ] {
        assert_eq!(
            group.abort(wrong),
            Err(HostCollectiveError::DescriptorMismatch)
        );
        assert_eq!(group.owned_host_bytes(), bytes);
        assert_eq!(group.active_descriptor(), Some(op));
    }
    group.abort(op).unwrap();
    assert_eq!(group.status(), None);
    assert_eq!(group.owned_host_bytes(), 0);
    assert_eq!(group.reserved_host_bytes(), 0);
    reject(
        &mut group,
        9,
        op,
        vec![1.0],
        HostCollectiveError::StaleSequence {
            watermark: 0,
            received: 0,
        },
    );
    let next = HostCollectiveDescriptor { sequence: 1, ..op };
    for member in [9, 2, 6] {
        group
            .submit(rank(member), next, vec![member as f32])
            .unwrap();
    }
    let returned = take(&mut group, 9, next);
    assert_eq!(
        group.abort(op),
        Err(HostCollectiveError::DescriptorMismatch)
    );
    group.abort(next).unwrap();
    assert_eq!(returned, vec![9.0, 2.0, 6.0]);
    assert_eq!(group.owned_host_bytes(), 0);
    assert_eq!(
        group.take_result(rank(2)),
        Err(HostCollectiveError::NoOperation)
    );
    for sequence in 2..256 {
        let next = HostCollectiveDescriptor { sequence, ..op };
        group.submit(rank(6), next, vec![1.0]).unwrap();
        group.abort(next).unwrap();
        assert_eq!(group.sequence_watermark(), Some(sequence));
        assert_eq!(group.reserved_host_bytes(), 0);
    }
}

#[test]
fn sequence_max_does_not_wrap_and_new_epoch_requires_explicit_new_group() {
    let mut old = group();
    let last = descriptor(HostCollectiveKind::AllReduceSumF32, 1, u64::MAX);
    old.submit(rank(9), last, vec![1.0]).unwrap();
    old.abort(last).unwrap();
    for sequence in [0, u64::MAX - 1, u64::MAX] {
        reject(
            &mut old,
            9,
            HostCollectiveDescriptor { sequence, ..last },
            vec![1.0],
            HostCollectiveError::StaleSequence {
                watermark: u64::MAX,
                received: sequence,
            },
        );
    }
    let next_epoch = HostCollectiveDescriptor {
        epoch: ParallelTopologyId::new(8),
        sequence: 0,
        ..last
    };
    reject(
        &mut old,
        9,
        next_epoch,
        vec![1.0],
        HostCollectiveError::WrongEpoch,
    );
    let mut rebuilt =
        HostCollectiveGroup::new(next_epoch.epoch, next_epoch.group, vec![rank(9)], limits())
            .unwrap();
    assert_eq!(
        rebuilt.submit(rank(9), next_epoch, vec![4.0]).unwrap(),
        HostCollectiveStatus::Ready
    );
    assert_eq!(take(&mut rebuilt, 9, next_epoch), vec![4.0]);
}

#[test]
fn validate_members_limits_and_allocation_arithmetic_before_allocating_payloads() {
    let new = |members, config| {
        HostCollectiveGroup::new(
            ParallelTopologyId::new(7),
            ParallelGroupId::new(11),
            members,
            config,
        )
    };
    assert_eq!(
        new(vec![], limits()).unwrap_err(),
        HostCollectiveError::EmptyMembers
    );
    assert_eq!(
        new(
            vec![rank(9)],
            HostCollectiveLimits {
                max_ranks: 0,
                ..limits()
            }
        )
        .unwrap_err(),
        HostCollectiveError::InvalidLimits
    );
    assert_eq!(
        new(vec![rank(9), rank(9)], limits()).unwrap_err(),
        HostCollectiveError::DuplicateMember(rank(9))
    );
    assert_eq!(
        new(vec![rank(0), rank(1), rank(2), rank(3)], limits()).unwrap_err(),
        HostCollectiveError::TooManyRanks
    );
    let mut spare_members = Vec::with_capacity(4);
    spare_members.push(rank(9));
    assert_eq!(
        new(spare_members, limits()).unwrap_err(),
        HostCollectiveError::TooManyRanks
    );
    assert_eq!(
        new(
            vec![rank(9)],
            HostCollectiveLimits {
                max_elements_per_rank: usize::MAX,
                ..limits()
            }
        )
        .unwrap_err(),
        HostCollectiveError::ArithmeticOverflow
    );
    assert_eq!(
        new(
            vec![rank(9)],
            HostCollectiveLimits {
                max_ranks: usize::MAX,
                ..limits()
            }
        )
        .unwrap_err(),
        HostCollectiveError::ArithmeticOverflow
    );
    let huge_count = isize::MAX as usize / std::mem::size_of::<f32>();
    let huge_limits = HostCollectiveLimits {
        max_elements_per_rank: huge_count,
        max_host_bytes: usize::MAX,
        ..limits()
    };
    let two = new(vec![rank(9), rank(2)], huge_limits).unwrap();
    assert_eq!(
        two.memory_for(HostCollectiveKind::AllGatherF32, huge_count),
        Err(HostCollectiveError::ArithmeticOverflow)
    );
    assert_eq!(
        two.memory_for(HostCollectiveKind::AllReduceSumF32, huge_count),
        Err(HostCollectiveError::ArithmeticOverflow)
    );
    let three = new(vec![rank(9), rank(2), rank(6)], huge_limits).unwrap();
    assert_eq!(
        three.memory_for(HostCollectiveKind::AllReduceSumF32, 0),
        Err(HostCollectiveError::ArithmeticOverflow)
    );
    assert_eq!(three.sequence_watermark(), None);
    assert_eq!(three.owned_host_bytes(), 0);
}

#[test]
fn zero_count_still_requires_all_ranks_and_both_kinds_support_single_member() {
    for kind in [
        HostCollectiveKind::AllGatherF32,
        HostCollectiveKind::AllReduceSumF32,
    ] {
        let mut empty = HostCollectiveGroup::new(
            ParallelTopologyId::new(7),
            ParallelGroupId::new(11),
            vec![rank(9), rank(2)],
            HostCollectiveLimits {
                max_elements_per_rank: 0,
                max_host_bytes: 0,
                ..limits()
            },
        )
        .unwrap();
        let op = descriptor(kind, 0, 0);
        assert_eq!(
            empty.submit(rank(2), op, vec![]).unwrap(),
            HostCollectiveStatus::Pending
        );
        assert_eq!(
            empty.take_result(rank(9)).unwrap(),
            HostCollectivePoll::Pending
        );
        assert_eq!(
            empty.submit(rank(9), op, vec![]).unwrap(),
            HostCollectiveStatus::Ready
        );
        assert!(take(&mut empty, 2, op).is_empty());
        assert!(take(&mut empty, 9, op).is_empty());
        assert_eq!(empty.reserved_host_bytes(), 0);
        let mut single =
            HostCollectiveGroup::new(op.epoch, op.group, vec![rank(9)], limits()).unwrap();
        let op = HostCollectiveDescriptor { count: 2, ..op };
        assert_eq!(
            single.submit(rank(9), op, vec![-0.0, 4.0]).unwrap(),
            HostCollectiveStatus::Ready
        );
        let output = take(&mut single, 9, op);
        assert_eq!(output[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(output[1], 4.0);
    }
}
