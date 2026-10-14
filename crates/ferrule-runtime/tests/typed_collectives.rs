//! Typed payloads use the existing sealed owner coordinator and lifetime.
use ferrule_common::error::ErrorChain;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ExecutionTransactionId, ForwardMode, ForwardPhase,
    KvBlockId, KvReservationView, KvWriteSlot, LogitsRequest, StateSlot,
};
use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CompletionUnknown, DecoderKvPageStatus, GenericDecoderSequenceState, KvCommitBinding,
    PackedDecoderBatch,
};
use ferrule_model::transformer::{LayerSegmentPlan, StandardTensorCollective};
use ferrule_runtime::parallel::collective::{
    CollectivePayload as P, HostCollectiveKind as K, HostCollectiveLimits,
};
use ferrule_runtime::parallel::pipeline::tensor_scope::*;
use ferrule_runtime::parallel::tensor::decoder_collective::DecoderTensorCollective;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;
fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}
fn topology() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        2,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
    )
    .unwrap()
}
fn binding() -> KvCommitBinding {
    KvCommitBinding::new(
        tx(1),
        topology().topology_id(),
        topology().participants(),
        17,
    )
    .unwrap()
}
fn description() -> ferrule_runtime::parallel::pipeline::PipelineStageDescription {
    ferrule_runtime::parallel::pipeline::PipelineStageDescription {
        plan: LayerSegmentPlan::new(1, 0..1, true, true).unwrap(),
        physical_pages: 8,
        config: ferrule_runtime::parallel::pipeline::PipelineConfig {
            page_size: 2,
            max_pages: 8,
            max_positions: 16,
            max_batch_tokens: 4,
            session_capacity: 2,
            max_parameter_bytes: 1 << 30,
            precision: ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
            max_ack_polls: 2,
        },
        hidden: 4,
        vocabulary: 4,
        kv_heads: 1,
        head_dim: 2,
        expert_group: None,
    }
}
fn packed(phase: ForwardPhase, local_slot: u32, generation: u64) -> PackedDecoderBatch {
    let description = description();
    let batch = ExecutionBatch::new(
        if phase == ForwardPhase::Prefill {
            ForwardMode::Prefill
        } else {
            ForwardMode::Decode
        },
        vec![3],
        vec![1],
        vec![Some(KvWriteSlot::new(15))],
        vec![LogitsRequest::Full],
        vec![ExecutionSequence::new(
            StateSlot::new(local_slot),
            phase,
            0..1,
            1,
            2,
            0..1,
        )],
        vec![KvBlockId::new(7)],
    );
    let reservation = KvReservationView {
        state_slot: StateSlot::new(9),
        execution_state_slot: StateSlot::new(local_slot),
        positions: 1..2,
        newly_allocated: vec![],
        generation,
        execution_generation: 0,
        cow_replacement: None,
    };
    let states = [
        GenericDecoderSequenceState::with_position(1, (), ()),
        GenericDecoderSequenceState::with_position(1, (), ()),
    ];
    PackedDecoderBatch::lower(
        &batch,
        &[reservation],
        &states,
        &description.execution_capabilities().unwrap(),
        2,
        &|_| DecoderKvPageStatus::Resident,
    )
    .unwrap()
}
fn identity() -> TensorBatchIdentity {
    TensorBatchIdentity::from_packed(&binding(), &packed(ForwardPhase::Decode, 0, 5), &[11])
        .unwrap()
}
fn raw_peers() -> Vec<DecoderTensorCollective> {
    DecoderTensorCollective::new_group_from_members(
        &[ParallelRankId::new(0), ParallelRankId::new(1)],
        ParallelTopologyId::new(501),
        ParallelGroupId::new(4),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 32,
            max_host_bytes: 4096,
        },
        Duration::from_secs(5),
    )
    .unwrap()
}
fn schedule(kind: K) -> Vec<TensorCollectiveSite> {
    vec![TensorCollectiveSite {
        site: 1,
        layer: Some(0),
        operator: TensorCollectiveOperator::Protocol {
            version: 2,
            operation: TensorCollectiveSemantic::GqaOutput,
            kind,
            shape: TensorCollectiveShape::PerRow(1),
        },
    }]
}
fn peers(kind: K) -> Vec<DecoderTensorCollective> {
    let peers = raw_peers();
    for peer in &peers {
        peer.control()
            .register_sites(peer.owner(), schedule(kind))
            .unwrap();
    }
    peers
}

#[test]
fn typed_tp2_runs_bf16_partials_and_f32_candidate_as_distinct_sealed_math() {
    // Rounding 1.003 to BF16 before summing differs from rounding 1.003+0.003.
    for (kind, inputs, expected) in [
        (
            K::AllReduceSumBf16,
            [P::Bf16(vec![0x3f80]), P::Bf16(vec![0x3b45])],
            P::Bf16(vec![0x3f80]),
        ),
        (
            K::AllReduceSumF32ToBf16,
            [P::F32(vec![1.003]), P::F32(vec![0.003])],
            P::Bf16(vec![0x3f81]),
        ),
        (
            K::AllGatherBf16,
            [P::Bf16(vec![0x8000]), P::Bf16(vec![0x7fc5])],
            P::Bf16(vec![0x8000, 0x7fc5]),
        ),
    ] {
        let mut peers = peers(kind);
        let control = peers[0].control();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        for peer in &peers {
            control
                .validate_owner_execution(peer.owner(), &identity())
                .unwrap();
        }
        std::thread::scope(|scope| {
            let handles = peers
                .iter_mut()
                .zip(inputs)
                .map(|(peer, input)| {
                    scope.spawn(move || peer.exchange_payload(tx(1), 1, kind, input))
                })
                .collect::<Vec<_>>();
            for h in handles {
                assert_eq!(h.join().unwrap().unwrap(), expected);
            }
        });
        control.finish(tx(1)).unwrap();
    }
}

#[test]
fn typed_schedule_mismatch_rejects_before_execution() {
    for mismatch in 0..3 {
        let peers = raw_peers();
        peers[0]
            .control()
            .register_sites(peers[0].owner(), schedule(K::AllReduceSumBf16))
            .unwrap();
        let mut changed = schedule(K::AllReduceSumBf16);
        let TensorCollectiveOperator::Protocol { version, kind, .. } = &mut changed[0].operator
        else {
            unreachable!()
        };
        match mismatch {
            0 => *kind = K::AllReduceSumF32ToBf16,
            1 => *version += 1,
            _ => changed[0].site += 1,
        }
        assert!(
            peers[1]
                .control()
                .register_sites(peers[1].owner(), changed)
                .is_err()
        );
        assert!(
            peers[0]
                .control()
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
    }
}

#[test]
fn typed_owner_source_and_generation_mismatch_never_admit_the_cohort() {
    for mismatch in 0..3 {
        let peers = peers(K::AllReduceSumBf16);
        let control = peers[0].control();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        let changed = if mismatch == 0 {
            TensorBatchIdentity::from_packed(&binding(), &packed(ForwardPhase::Decode, 0, 6), &[11])
                .unwrap()
        } else if mismatch == 1 {
            TensorBatchIdentity::from_packed(&binding(), &packed(ForwardPhase::Decode, 0, 5), &[12])
                .unwrap()
        } else {
            identity()
        };
        let owner = if mismatch == 2 {
            ParallelRankId::new(2)
        } else {
            peers[0].owner()
        };
        assert!(control.validate_owner_execution(owner, &changed).is_err());
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
}

#[test]
fn typed_exchange_cannot_bypass_schedule_payload_or_transaction_identity() {
    for mismatch in 0..4 {
        let mut peers = if mismatch == 0 {
            raw_peers()
        } else {
            peers(K::AllReduceSumBf16)
        };
        let control = peers[0].control();
        if mismatch == 0 {
            control
                .begin(tx(1), Arc::new(AtomicBool::new(false)))
                .unwrap();
        } else {
            control
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .unwrap();
            for peer in &peers {
                control
                    .validate_owner_execution(peer.owner(), &identity())
                    .unwrap();
            }
        }
        let input = if mismatch == 1 {
            P::F32(vec![1.0])
        } else {
            P::Bf16(vec![0x3f80])
        };
        let transaction = if mismatch == 2 { tx(2) } else { tx(1) };
        let kind = if mismatch == 3 {
            K::AllReduceSumF32ToBf16
        } else {
            K::AllReduceSumBf16
        };
        assert!(
            peers[0]
                .exchange_payload(transaction, 1, kind, input)
                .is_err()
        );
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
}

#[test]
fn typed_capacity_and_descriptor_checks_precede_mutation() {
    use ferrule_runtime::parallel::collective::{
        HostCollectiveDescriptor, HostCollectiveError as E, HostCollectiveGroup,
        HostCollectiveStatus, TypedCollectivePoll,
    };
    let limits = HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: 2,
        max_host_bytes: 16,
    };
    let mut group = HostCollectiveGroup::new(
        ParallelTopologyId::new(4),
        ParallelGroupId::new(9),
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        limits,
    )
    .unwrap();
    let op = HostCollectiveDescriptor {
        epoch: ParallelTopologyId::new(4),
        group: ParallelGroupId::new(9),
        transaction: tx(1),
        sequence: 1,
        kind: K::AllReduceSumBf16,
        count: 2,
    };
    assert_eq!(group.memory_for(op.kind, 2).unwrap().peak_bytes, 16);
    assert!(matches!(
        group.memory_for(K::AllReduceSumF32ToBf16, 2),
        Err(E::HostCapacity {
            required: 24,
            limit: 16
        })
    ));
    for (owner, descriptor, input, expected) in [
        (
            ParallelRankId::new(2),
            op,
            P::Bf16(vec![0; 2]),
            E::WrongRank(ParallelRankId::new(2)),
        ),
        (
            ParallelRankId::new(0),
            HostCollectiveDescriptor {
                epoch: ParallelTopologyId::new(5),
                ..op
            },
            P::Bf16(vec![0; 2]),
            E::WrongEpoch,
        ),
        (
            ParallelRankId::new(0),
            HostCollectiveDescriptor {
                group: ParallelGroupId::new(10),
                ..op
            },
            P::Bf16(vec![0; 2]),
            E::WrongGroup,
        ),
        (
            ParallelRankId::new(0),
            op,
            P::F32(vec![0.; 2]),
            E::PayloadTypeMismatch,
        ),
    ] {
        assert_eq!(
            group
                .submit_typed(owner, descriptor, input)
                .unwrap_err()
                .error,
            expected
        );
        assert!(group.active_descriptor().is_none());
        assert_eq!(group.owned_host_bytes(), 0);
        assert_eq!(group.sequence_watermark(), None);
    }
    // Both midpoint parities: nearest-even output, not truncation or ties-up.
    assert_eq!(
        group
            .submit_typed(ParallelRankId::new(0), op, P::Bf16(vec![0x3f80, 0x3f81]))
            .unwrap(),
        HostCollectiveStatus::Pending
    );
    assert_eq!(group.owned_host_bytes(), 12);
    assert_eq!(group.reserved_host_bytes(), 16);
    let wrong = HostCollectiveDescriptor {
        transaction: tx(2),
        ..op
    };
    assert_eq!(
        group
            .submit_typed(ParallelRankId::new(1), wrong, P::Bf16(vec![0; 2]))
            .unwrap_err()
            .error,
        E::DescriptorMismatch
    );
    assert_eq!(group.owned_host_bytes(), 12);
    assert_eq!(group.status(), Some(HostCollectiveStatus::Pending));
    group
        .submit_typed(ParallelRankId::new(1), op, P::Bf16(vec![0x3b80; 2]))
        .unwrap();
    assert_eq!(group.owned_host_bytes(), 8);
    assert_eq!(
        group.take_result(ParallelRankId::new(0)),
        Err(E::PayloadTypeMismatch)
    );
    for rank in [0, 1] {
        assert_eq!(
            group.take_typed_result(ParallelRankId::new(rank)).unwrap(),
            TypedCollectivePoll::Ready {
                descriptor: op,
                payload: P::Bf16(vec![0x3f80, 0x3f82])
            }
        );
    }
    assert_eq!(group.owned_host_bytes(), 0);
    assert!(matches!(
        group
            .submit_typed(ParallelRankId::new(0), op, P::Bf16(vec![0; 2]))
            .unwrap_err()
            .error,
        E::StaleSequence { .. }
    ));
}

use ferrule_runtime::parallel::tensor::decoder_collective::{
    MixedRankCollective, MixedRankCollectivePlan,
};

fn mixed_plan(source: u32, sites: Vec<TensorCollectiveSite>) -> Arc<MixedRankCollectivePlan> {
    Arc::new(
        MixedRankCollectivePlan::new(
            ParallelTopologyId::new(501),
            ParallelGroupId::new(4),
            vec![ParallelRankId::new(0), ParallelRankId::new(1)],
            ParallelRankId::new(source),
            sites,
            HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 32,
                max_host_bytes: 4096,
            },
            1,
            1,
        )
        .unwrap(),
    )
}
fn mixed_peers(plan: Arc<MixedRankCollectivePlan>) -> Vec<MixedRankCollective> {
    DecoderTensorCollective::new_group_from_members(
        plan.members(),
        plan.epoch(),
        plan.group(),
        plan.limits(),
        Duration::from_secs(5),
    )
    .unwrap()
    .into_iter()
    .map(|endpoint| {
        let owner = endpoint.owner();
        MixedRankCollective::prepare(endpoint, owner, Arc::clone(&plan)).unwrap()
    })
    .collect()
}
fn wait_for_peer(
    control: &ferrule_runtime::parallel::tensor::decoder_collective::DecoderTensorCollectiveControl,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while control.waiting_members() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "peer did not enter bounded wait"
        );
        std::thread::yield_now();
    }
}

#[test]
fn mixed_rank_capability_executes_only_bf16_schedule_and_never_finishes_early() {
    let mut sites = schedule(K::AllReduceSumBf16);
    sites.push(TensorCollectiveSite {
        site: 2,
        layer: Some(0),
        operator: TensorCollectiveOperator::Protocol {
            version: 2,
            operation: TensorCollectiveSemantic::KvReplication,
            kind: K::AllGatherBf16,
            shape: TensorCollectiveShape::PerRow(1),
        },
    });
    let mut peers = mixed_peers(mixed_plan(0, sites));
    let control = peers[0].control();
    assert!(
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in &peers {
        peer.validate_owner_execution(&identity()).unwrap();
    }
    assert!(control.finish(tx(1)).is_err());
    let (first, second) = peers.split_at_mut(1);
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| first[0].exchange_bf16(tx(1), 1, vec![0x3f81]));
        wait_for_peer(&control);
        assert!(control.finish(tx(1)).is_err());
        assert_eq!(
            second[0].exchange_bf16(tx(1), 1, vec![0x3b80]).unwrap(),
            vec![0x3f82]
        );
        assert_eq!(waiting.join().unwrap().unwrap(), vec![0x3f82]);
    });
    assert!(
        control.finish(tx(1)).is_err(),
        "pending KV gather cannot finish"
    );
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| first[0].exchange_bf16(tx(1), 2, vec![0x8000]));
        assert_eq!(
            second[0].exchange_bf16(tx(1), 2, vec![0x7fc5]).unwrap(),
            vec![0x8000, 0x7fc5]
        );
        assert_eq!(waiting.join().unwrap().unwrap(), vec![0x8000, 0x7fc5]);
    });
    for peer in &peers {
        peer.complete_owner(tx(1)).unwrap();
    }
    control.finish(tx(1)).unwrap();
    assert!(
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    // New transaction retains the same schedule and strictly advances operation sequence.
    let next_binding = KvCommitBinding::new(
        tx(2),
        topology().topology_id(),
        topology().participants(),
        18,
    )
    .unwrap();
    let next =
        TensorBatchIdentity::from_packed(&next_binding, &packed(ForwardPhase::Decode, 0, 6), &[11])
            .unwrap();
    control
        .begin_sealed(next.clone(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in &peers {
        peer.validate_owner_execution(&next).unwrap();
    }
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .map(|peer| {
                scope.spawn(|| {
                    assert_eq!(
                        peer.exchange_bf16(tx(2), 1, vec![0x3f80]).unwrap(),
                        vec![0x4000]
                    );
                    peer.exchange_bf16(tx(2), 2, vec![0x3f80]).unwrap();
                    peer.complete_owner(tx(2)).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    control.finish(tx(2)).unwrap();
}

#[test]
fn mixed_rank_preparation_seals_source_schedule_owner_and_limits() {
    for mismatch in 0..4 {
        let mut endpoints = raw_peers();
        let second = endpoints.pop().unwrap();
        let first = endpoints.pop().unwrap();
        let plan = mixed_plan(0, schedule(K::AllReduceSumBf16));
        let first = MixedRankCollective::prepare(first, ParallelRankId::new(0), plan).unwrap();
        let control = first.control();
        let mut sites = schedule(K::AllReduceSumBf16);
        if mismatch == 1 {
            sites[0].site += 1;
        }
        if mismatch == 3 {
            let TensorCollectiveOperator::Protocol { version, .. } = &mut sites[0].operator else {
                unreachable!()
            };
            *version += 1;
        }
        let plan = mixed_plan(if mismatch == 0 { 1 } else { 0 }, sites);
        let owner = ParallelRankId::new(if mismatch == 2 { 0 } else { 1 });
        assert!(MixedRankCollective::prepare(second, owner, plan).is_err());
        assert!(control.is_failed());
        assert!(
            control
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
    }
}

#[test]
fn mixed_rank_cancel_wakes_peers_and_unknown_is_not_a_completion_receipt() {
    let mut peers = mixed_peers(mixed_plan(0, schedule(K::AllReduceSumBf16)));
    let control = peers[0].control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in &peers {
        peer.validate_owner_execution(&identity()).unwrap();
    }
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| peers[0].exchange_bf16(tx(1), 1, vec![0x3f80]));
        wait_for_peer(&control);
        // This is host cancellation only, not proof that any GPU DMA quiesced.
        control.abort(tx(1), "owner completion Unknown").unwrap();
        assert!(waiting.join().unwrap().is_err());
    });
    assert!(control.is_failed());
    assert!(control.finish(tx(1)).is_err());
    assert!(peers[1].complete_owner(tx(1)).is_err());
    assert!(
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}

#[test]
fn mixed_rank_rejects_stale_identity_shape_site_and_non_bf16_plans() {
    for kind in [K::AllReduceSumF32, K::AllReduceSumF32ToBf16] {
        assert!(
            MixedRankCollectivePlan::new(
                ParallelTopologyId::new(501),
                ParallelGroupId::new(4),
                vec![ParallelRankId::new(0), ParallelRankId::new(1)],
                ParallelRankId::new(0),
                schedule(kind),
                HostCollectiveLimits {
                    max_ranks: 2,
                    max_elements_per_rank: 32,
                    max_host_bytes: 4096
                },
                1,
                1
            )
            .is_err()
        );
    }
    for mismatch in 0..3 {
        let mut peers = mixed_peers(mixed_plan(0, schedule(K::AllReduceSumBf16)));
        let control = peers[0].control();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        if mismatch == 0 {
            let stale = TensorBatchIdentity::from_packed(
                &binding(),
                &packed(ForwardPhase::Decode, 0, 4),
                &[11],
            )
            .unwrap();
            assert!(peers[0].validate_owner_execution(&stale).is_err());
        } else {
            for peer in &peers {
                peer.validate_owner_execution(&identity()).unwrap();
            }
            let (site, words) = if mismatch == 1 {
                (8, vec![0x3f80])
            } else {
                (1, vec![0x3f80; 2])
            };
            assert!(peers[0].exchange_bf16(tx(1), site, words).is_err());
        }
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
}

#[test]
fn mixed_peer_disappearance_is_unknown_while_pending_and_before_owner_completion() {
    // Pending: the surviving owner is asleep in the existing coordinator when
    // the other prepared owner disappears.
    let mut peers = mixed_peers(mixed_plan(0, schedule(K::AllReduceSumBf16)));
    let right = peers.pop().unwrap();
    let mut left = peers.pop().unwrap();
    let control = left.control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    left.validate_owner_execution(&identity()).unwrap();
    right.validate_owner_execution(&identity()).unwrap();
    let pending = std::thread::scope(|scope| {
        let waiting = scope.spawn(move || left.exchange_bf16(tx(1), 1, vec![0x3f80]));
        wait_for_peer(&control);
        drop(right);
        waiting.join().unwrap().unwrap_err()
    });
    assert!(
        ErrorChain::new(&pending)
            .downcast_ref::<CompletionUnknown>()
            .is_some(),
        "peer loss must retain typed completion custody"
    );
    assert!(control.finish(tx(1)).is_err());

    // Ready/drained: keep both endpoints alive through the exchanges, then
    // lose one either before local completion or between local and parent finish.
    for after_owner_completion in [false, true] {
        let mut peers = mixed_peers(mixed_plan(0, schedule(K::AllReduceSumBf16)));
        let mut right = peers.pop().unwrap();
        let mut left = peers.pop().unwrap();
        let control = left.control();
        let cancelled = Arc::new(AtomicBool::new(false));
        control
            .begin_sealed(identity(), Arc::clone(&cancelled))
            .unwrap();
        left.validate_owner_execution(&identity()).unwrap();
        right.validate_owner_execution(&identity()).unwrap();
        std::thread::scope(|scope| {
            let left_result = scope.spawn(|| left.exchange_bf16(tx(1), 1, vec![0x3f80]));
            assert_eq!(
                right.exchange_bf16(tx(1), 1, vec![0x3b80]).unwrap(),
                vec![0x3f80]
            );
            assert_eq!(left_result.join().unwrap().unwrap(), vec![0x3f80]);
        });
        if after_owner_completion {
            left.complete_owner(tx(1)).unwrap();
            right.complete_owner(tx(1)).unwrap();
        }
        drop(right);
        assert!(cancelled.load(std::sync::atomic::Ordering::Acquire));
        for completion in [
            left.complete_owner(tx(1)).unwrap_err(),
            control.finish(tx(1)).unwrap_err(),
        ] {
            assert!(
                ErrorChain::new(&completion)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        }
        assert!(control.is_failed());
        assert!(
            control
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
    }
}

#[test]
fn mixed_rank_timeout_is_terminal_and_preparation_rejects_endpoint_budget_mismatch() {
    let plan = mixed_plan(0, schedule(K::AllReduceSumBf16));
    let mut endpoints = DecoderTensorCollective::new_group_from_members(
        plan.members(),
        plan.epoch(),
        plan.group(),
        plan.limits(),
        Duration::from_millis(10),
    )
    .unwrap();
    let right = endpoints.pop().unwrap();
    let left = endpoints.pop().unwrap();
    let mut left =
        MixedRankCollective::prepare(left, ParallelRankId::new(0), Arc::clone(&plan)).unwrap();
    let right =
        MixedRankCollective::prepare(right, ParallelRankId::new(1), Arc::clone(&plan)).unwrap();
    let control = left.control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    left.validate_owner_execution(&identity()).unwrap();
    right.validate_owner_execution(&identity()).unwrap();
    assert!(
        left.exchange_bf16(tx(1), 1, vec![0x3f80])
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert!(control.is_failed());
    assert!(control.finish(tx(1)).is_err());
    assert!(right.complete_owner(tx(1)).is_err());
    let endpoints = DecoderTensorCollective::new_group_from_members(
        plan.members(),
        plan.epoch(),
        plan.group(),
        HostCollectiveLimits {
            max_elements_per_rank: 31,
            ..plan.limits()
        },
        Duration::from_secs(1),
    )
    .unwrap();
    let mut endpoints = endpoints.into_iter();
    let left = endpoints.next().unwrap();
    let control = left.control();
    assert!(MixedRankCollective::prepare(left, ParallelRankId::new(0), plan).is_err());
    assert!(control.is_failed());
}

fn model_block_plans() -> (
    Arc<MixedRankCollectivePlan>,
    Vec<ferrule_model::transformer::hybrid_parallel::HybridMixedGqaBlockPlan>,
) {
    use ferrule_model::models::qwen3::Qwen3DenseRecipe;
    use ferrule_model::transformer::{DecoderRecipe, hybrid_parallel::*};
    let spec = Qwen3DenseRecipe::new()
        .build_spec(&serde_json::json!({
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
            "hidden_act":"silu", "vocab_size":4, "hidden_size":4, "num_hidden_layers":1,
            "num_attention_heads":2, "num_key_value_heads":2, "head_dim":2, "intermediate_size":8,
            "max_position_embeddings":16, "rms_norm_eps":0.00001, "rope_theta":10000.0,
            "tie_word_embeddings":true, "attention_bias":false, "use_sliding_window":false,
            "attention_dropout":0.0, "use_cache":true, "max_window_layers":1,
            "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
        }))
        .unwrap();
    let parallel = HybridParallelPlan::new_with_execution(
        &spec,
        0..1,
        HybridExpertScope {
            topology: topology().topology_id(),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        HybridRoutedExpertExecution::None,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    let blocks = (0..2)
        .map(|lane| {
            HybridMixedGqaBlockPlan::new(
                Arc::new(parallel.rank_layout(lane).unwrap()),
                1,
                1,
                16,
                2,
                5,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let sites = blocks[0].sites();
    assert_eq!(sites.len(), 2);
    assert_eq!(sites[0].semantic, HybridCollectiveSemantic::AttentionOutput);
    assert_eq!(sites[1].semantic, HybridCollectiveSemantic::DenseMlpDown);
    assert!(
        sites
            .iter()
            .all(|site| site.kind == HybridCollectiveKind::ReduceSum)
    );
    let plan = Arc::new(
        MixedRankCollectivePlan::from_hybrid_sites(
            &parallel,
            ParallelGroupId::new(4),
            ParallelRankId::new(0),
            HYBRID_AGREEMENT_VERSION,
            &sites,
            HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 32,
                max_host_bytes: 4096,
            },
            1,
            1,
        )
        .unwrap(),
    );
    (plan, blocks)
}
fn model_block_collective_plan() -> Arc<MixedRankCollectivePlan> {
    model_block_plans().0
}

#[test]
fn model_trait_boxes_use_real_gqa_block_data_sites_and_control_completion() {
    use ferrule_model::transformer::hybrid_parallel::{HybridBf16Collective, HybridCollectiveKind};
    let (plan, blocks) = model_block_plans();
    let peers = mixed_peers(plan);
    for (block, peer) in blocks.iter().zip(&peers) {
        block.validate_collective(peer).unwrap();
        assert_eq!(peer.sealed_binding().unwrap(), block.collective_binding());
    }
    let control = peers[0].control();
    let owners = peers
        .iter()
        .map(MixedRankCollective::owner)
        .collect::<Vec<_>>();
    let mut peers: Vec<Box<dyn HybridBf16Collective + Send>> = peers
        .into_iter()
        .map(|peer| Box::new(peer) as Box<dyn HybridBf16Collective + Send>)
        .collect();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for owner in owners {
        control
            .validate_owner_execution(owner, &identity())
            .unwrap();
    }
    assert!(control.finish(tx(1)).is_err());
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .map(|peer| {
                scope.spawn(|| {
                    peer.validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 5), &[11])
                        .unwrap();
                    let input = if peer.owner().get() == 0 {
                        0x3f81
                    } else {
                        0x3b80
                    };
                    assert_eq!(
                        peer.exchange_bf16(
                            tx(1),
                            0,
                            HybridCollectiveKind::ReduceSum,
                            vec![input; 4]
                        )
                        .unwrap(),
                        vec![0x3f82; 4]
                    );
                    assert_eq!(
                        peer.exchange_bf16(
                            tx(1),
                            1,
                            HybridCollectiveKind::ReduceSum,
                            vec![0x3f80; 4]
                        )
                        .unwrap(),
                        vec![0x4000; 4]
                    );
                    peer.complete_owner(tx(1)).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for h in handles {
            h.join().unwrap();
        }
    });
    control.finish(tx(1)).unwrap();
    assert!(
        peers[0]
            .validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 5), &[11])
            .is_err()
    );
    assert!(control.is_failed());
}

#[test]
fn model_trait_rejects_kind_identity_unvalidated_owner_and_early_completion() {
    use ferrule_model::transformer::hybrid_parallel::{HybridBf16Collective, HybridCollectiveKind};
    for mismatch in 0..6 {
        let peers = mixed_peers(model_block_collective_plan());
        let control = peers[0].control();
        let owners = peers
            .iter()
            .map(MixedRankCollective::owner)
            .collect::<Vec<_>>();
        let mut peers: Vec<Box<dyn HybridBf16Collective>> = peers
            .into_iter()
            .map(|peer| Box::new(peer) as Box<dyn HybridBf16Collective>)
            .collect();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        if mismatch != 0 {
            for owner in owners {
                control
                    .validate_owner_execution(owner, &identity())
                    .unwrap();
            }
        }
        let result = match mismatch {
            0 => peers[0].validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 5), &[11]),
            1 => peers[0].validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 6), &[11]),
            2 => peers[0].validate_execution(tx(2), &packed(ForwardPhase::Decode, 0, 5), &[11]),
            3 => peers[0].complete_owner(tx(1)),
            4 => peers[0]
                .exchange_bf16(tx(1), 0, HybridCollectiveKind::AllGather, vec![0x3f80; 4])
                .map(|_| ()),
            _ => peers[0]
                .exchange_bf16(tx(1), 3, HybridCollectiveKind::ReduceSum, vec![0x3f80])
                .map(|_| ()),
        };
        assert!(result.is_err());
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
}

#[test]
fn model_trait_abort_wakes_waiters_and_timeout_cannot_complete() {
    use ferrule_model::transformer::hybrid_parallel::{HybridBf16Collective, HybridCollectiveKind};
    for timeout in [false, true] {
        let plan = model_block_collective_plan();
        let endpoints = DecoderTensorCollective::new_group_from_members(
            plan.members(),
            plan.epoch(),
            plan.group(),
            plan.limits(),
            if timeout {
                Duration::from_millis(10)
            } else {
                Duration::from_secs(5)
            },
        )
        .unwrap();
        let peers = endpoints
            .into_iter()
            .map(|endpoint| {
                let owner = endpoint.owner();
                MixedRankCollective::prepare(endpoint, owner, Arc::clone(&plan)).unwrap()
            })
            .collect::<Vec<_>>();
        let control = peers[0].control();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        for peer in &peers {
            peer.validate_owner_execution(&identity()).unwrap();
        }
        let mut peers: Vec<Box<dyn HybridBf16Collective + Send>> = peers
            .into_iter()
            .map(|peer| Box::new(peer) as Box<dyn HybridBf16Collective + Send>)
            .collect();
        let mut left = peers.remove(0);
        let mut right = peers.remove(0);
        left.validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 5), &[11])
            .unwrap();
        right
            .validate_execution(tx(1), &packed(ForwardPhase::Decode, 0, 5), &[11])
            .unwrap();
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| {
                left.exchange_bf16(tx(1), 0, HybridCollectiveKind::ReduceSum, vec![0x3f80; 4])
            });
            if !timeout {
                wait_for_peer(&control);
                right.abort();
            }
            assert!(waiting.join().unwrap().is_err());
        });
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
        assert!(right.complete_owner(tx(1)).is_err());
    }
}

#[test]
fn authenticated_runtime_binding_rejects_model_mismatches_before_any_enter_or_exchange() {
    use ferrule_model::transformer::hybrid_parallel::HybridBf16Collective;
    let (base, blocks) = model_block_plans();
    for mismatch in 0..18 {
        let mut sites = base.sites().to_vec();
        let mut epoch = base.epoch();
        let mut members = base.members().to_vec();
        let mut source = base.source();
        let mut rows = base.max_rows();
        let mut sequences = base.max_sequences();
        match mismatch {
            0 => sites[0].layer = Some(7),
            1 => sites[0].site = 9, // out-of-order ordinal rejected at construction
            2 => {
                sites[0].site = 4;
                sites[1].site = 5;
            }
            3 => sites.swap(0, 1),
            4 => source = ParallelRankId::new(1),
            5 => epoch = ParallelTopologyId::new(502),
            6 => members.reverse(),
            7 => {
                let TensorCollectiveOperator::Protocol { operation, .. } = &mut sites[0].operator
                else {
                    unreachable!()
                };
                *operation = TensorCollectiveSemantic::DenseMlpOutput;
            }
            8..=11 => {
                let TensorCollectiveOperator::Protocol {
                    version,
                    kind,
                    shape,
                    ..
                } = &mut sites[0].operator
                else {
                    unreachable!()
                };
                match mismatch {
                    8 => *kind = K::AllGatherBf16,
                    9 => *kind = K::AllReduceSumF32,
                    10 => *shape = TensorCollectiveShape::PerRow(5),
                    11 => *version += 1,
                    _ => unreachable!(),
                }
            }
            12 => {
                sites.pop();
            }
            13 => rows = 2,
            14 => {
                rows = 2;
                sequences = 2;
            }
            15 => {
                for site in &mut sites {
                    let TensorCollectiveOperator::Protocol { version, .. } = &mut site.operator
                    else {
                        unreachable!()
                    };
                    *version += 1;
                }
            }
            16 => {
                // Same element count for this cohort, different sealed geometry.
                let TensorCollectiveOperator::Protocol { shape, .. } = &mut sites[0].operator
                else {
                    unreachable!()
                };
                *shape = TensorCollectiveShape::PerSequence {
                    fixed: 0,
                    per_sequence: 1,
                    per_row: 3,
                };
            }
            17 => {
                sites.swap(0, 1);
                sites[0].site = 0;
                sites[1].site = 1;
            }
            _ => unreachable!(),
        }
        let plan = MixedRankCollectivePlan::new(
            epoch,
            base.group(),
            members,
            source,
            sites,
            base.limits(),
            sequences,
            rows,
        );
        if let Ok(plan) = plan {
            let peers = mixed_peers(Arc::new(plan));
            let peer = peers
                .iter()
                .find(|peer| peer.owner() == blocks[0].layout().owner())
                .unwrap();
            let control = peer.control();
            // Metadata validation cannot consume an operation or fabricate an execution.
            let before = peer.sealed_binding().unwrap();
            let mut physical_enters = 0;
            assert!(
                blocks[0]
                    .validate_collective(peer)
                    .map(|()| physical_enters += 1)
                    .is_err(),
                "mismatch {mismatch}"
            );
            assert_eq!(physical_enters, 0);
            assert_eq!(control.waiting_members(), 0);
            assert!(!control.is_failed());
            assert_eq!(peer.sealed_binding().unwrap(), before);
        } else {
            assert!(
                [1, 3, 9, 11].contains(&mismatch),
                "unexpected constructor rejection {mismatch}"
            );
        }
    }
}

#[test]
fn sealed_binding_preserves_three_data_sites_and_all_shape_coefficients() {
    use ferrule_model::transformer::hybrid_parallel::{
        HybridBf16Collective, HybridCollectiveKind, HybridCollectiveSemantic as S,
    };
    let sites = [
        (
            TensorCollectiveSemantic::AttentionOutput,
            TensorCollectiveShape::Fixed(1),
        ),
        (
            TensorCollectiveSemantic::SharedExpertOutput,
            TensorCollectiveShape::Rows {
                fixed: 1,
                per_row: 1,
            },
        ),
        (
            TensorCollectiveSemantic::RoutedExpertOutput,
            TensorCollectiveShape::PerSequence {
                fixed: 1,
                per_sequence: 1,
                per_row: 1,
            },
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (operation, shape))| TensorCollectiveSite {
        site: (i + 5) as u64,
        layer: Some(3),
        operator: TensorCollectiveOperator::Protocol {
            version: 2,
            operation,
            kind: K::AllReduceSumBf16,
            shape,
        },
    })
    .collect();
    let mut peers = mixed_peers(mixed_plan(1, sites));
    let description = peers[0].sealed_binding().unwrap();
    assert_eq!(description.source, ParallelRankId::new(1));
    assert_eq!(
        description.members,
        vec![ParallelRankId::new(0), ParallelRankId::new(1)]
    );
    for (i, semantic) in [
        S::AttentionOutput,
        S::SharedExpertDown,
        S::RoutedExpertOutput,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(description.sites[i].semantic, semantic);
        assert_eq!(description.sites[i].global_layer, Some(3));
        assert_eq!(description.sites[i].ordinal, (i + 5) as u64);
        assert_eq!(description.sites[i].kind, HybridCollectiveKind::ReduceSum);
        assert_eq!(
            description.sites[i].shape.elements(2, 3).unwrap(),
            [1, 4, 6][i]
        );
    }
    let control = peers[0].control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in &peers {
        peer.validate_owner_execution(&identity()).unwrap();
    }
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .map(|peer| {
                scope.spawn(|| {
                    for i in 0..3 {
                        let input = if peer.owner().get() == 0 {
                            0x3f81
                        } else {
                            0x3b80
                        };
                        assert_eq!(
                            peer.exchange_bf16(tx(1), (i + 5) as u64, vec![input; i + 1])
                                .unwrap(),
                            vec![0x3f82; i + 1]
                        );
                    }
                    peer.complete_owner(tx(1)).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    control.finish(tx(1)).unwrap();
    assert_eq!(peers[1].sealed_binding().unwrap(), description);
    peers[0].abort();
    assert!(
        peers[1].sealed_binding().is_err(),
        "aborted capability cannot prepare another model"
    );
}

#[test]
fn sealed_binding_gather_and_unrepresentable_semantics_never_use_f32_aliases() {
    use ferrule_model::transformer::hybrid_parallel::{
        HybridBf16Collective, HybridCollectiveKind, HybridCollectiveSemantic,
    };
    let mut sites = schedule(K::AllGatherBf16);
    sites[0].layer = None;
    let TensorCollectiveOperator::Protocol { operation, .. } = &mut sites[0].operator else {
        unreachable!()
    };
    *operation = TensorCollectiveSemantic::OutputHead;
    let peers = mixed_peers(mixed_plan(0, sites));
    let binding = peers[0].sealed_binding().unwrap();
    assert_eq!(
        binding.sites[0].semantic,
        HybridCollectiveSemantic::OutputHead
    );
    assert_eq!(binding.sites[0].kind, HybridCollectiveKind::AllGather);
    assert_eq!(binding.sites[0].shape.per_row, 1);
    for operation in [
        TensorCollectiveSemantic::GqaOutput,
        TensorCollectiveSemantic::GatedDeltaOutput,
        TensorCollectiveSemantic::KvReplication,
    ] {
        let mut sites = schedule(K::AllGatherBf16);
        let TensorCollectiveOperator::Protocol { operation: op, .. } = &mut sites[0].operator
        else {
            unreachable!()
        };
        *op = operation;
        let peers = mixed_peers(mixed_plan(0, sites));
        assert!(
            peers[0].sealed_binding().is_err(),
            "do not invent a model semantic for {operation:?}"
        );
    }
    for kind in [
        K::AllReduceSumF32,
        K::AllGatherF32,
        K::AllReduceSumF32ToBf16,
    ] {
        assert!(
            MixedRankCollectivePlan::new(
                ParallelTopologyId::new(501),
                ParallelGroupId::new(4),
                vec![ParallelRankId::new(0), ParallelRankId::new(1)],
                ParallelRankId::new(0),
                schedule(kind),
                peers[0].plan().limits(),
                1,
                1
            )
            .is_err()
        );
    }
}

#[test]
fn mock_without_authenticated_binding_still_fails_closed() {
    use ferrule_model::transformer::hybrid_parallel::{HybridBf16Collective, HybridCollectiveKind};
    struct Legacy(Vec<ParallelRankId>);
    impl HybridBf16Collective for Legacy {
        fn owner(&self) -> ParallelRankId {
            self.0[0]
        }
        fn members(&self) -> &[ParallelRankId] {
            &self.0
        }
        fn exchange_bf16(
            &mut self,
            _: ExecutionTransactionId,
            _: u64,
            _: HybridCollectiveKind,
            _: Vec<u16>,
        ) -> ferrule_common::Result<Vec<u16>> {
            panic!("preparation must not exchange")
        }
        fn abort(&mut self) {}
    }
    let (_, blocks) = model_block_plans();
    assert!(
        blocks[0]
            .validate_collective(&Legacy(blocks[0].layout().members().to_vec()))
            .is_err()
    );
}

#[path = "typed_collectives/routes.rs"]
mod route_tests;
