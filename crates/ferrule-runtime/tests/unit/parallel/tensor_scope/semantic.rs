//! Sealed semantic schedules use real wire shapes, not invented linear plans.
use super::tests::{identity, tx, wait};
use super::*;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
use ferrule_common::{ParallelGroupId, ParallelTopologyId};
use ferrule_model::transformer::{StandardTensorCollective, parallel::TensorParallelCollective};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

pub(super) fn raw_peers() -> Vec<DecoderTensorCollective> {
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
pub(super) fn site(
    site: u64,
    layer: Option<usize>,
    operation: TensorCollectiveSemantic,
    shape: TensorCollectiveShape,
) -> TensorCollectiveSite {
    TensorCollectiveSite {
        site,
        layer,
        operator: TensorCollectiveOperator::Protocol {
            version: 2,
            operation,
            shape,
            kind: if matches!(
                operation,
                TensorCollectiveSemantic::InputAgreement | TensorCollectiveSemantic::OutputHead
            ) {
                HostCollectiveKind::AllGatherF32
            } else {
                HostCollectiveKind::AllReduceSumF32
            },
        },
    }
}
fn schedule() -> Vec<TensorCollectiveSite> {
    use TensorCollectiveSemantic::*;
    use TensorCollectiveShape::*;
    vec![
        site(
            0,
            None,
            InputAgreement,
            Rows {
                fixed: 4,
                per_row: 2,
            },
        ),
        site(1, Some(8), GatedDeltaOutput, PerRow(4)),
        site(2, Some(8), SharedExpertOutput, PerRow(4)),
        site(3, Some(8), RoutedExpertOutput, PerRow(4)),
        site(4, None, Frontier, Fixed(1)),
    ]
}
fn register(peers: &[DecoderTensorCollective], schedule: &[TensorCollectiveSite]) {
    for peer in peers {
        peer.control()
            .register_sites(peer.owner(), schedule.to_vec())
            .unwrap();
    }
}
fn admit(peers: &[DecoderTensorCollective]) {
    let control = peers[0].control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in peers {
        control
            .validate_owner_execution(peer.owner(), &identity())
            .unwrap();
    }
}

#[test]
fn semantic_schedule_runs_in_order_and_sealed_boundaries_reject_exchange() {
    let mut peers = raw_peers();
    register(&peers, &schedule());
    admit(&peers);
    let control = peers[0].control();
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .map(|peer| {
                scope.spawn(|| {
                    for site in schedule() {
                        let count = site
                            .elements(identity().sequences(), identity().rows())
                            .unwrap();
                        let kind = if site.kind() == HostCollectiveKind::AllGatherF32 {
                            TensorParallelCollective::AllGather
                        } else {
                            TensorParallelCollective::Sum
                        };
                        let output = peer
                            .exchange(tx(1), site.site, kind, vec![1.; count])
                            .unwrap();
                        assert_eq!(
                            output.len(),
                            count
                                * if kind == TensorParallelCollective::AllGather {
                                    2
                                } else {
                                    1
                                }
                        );
                        assert!(output.iter().all(|&x| x
                            == if kind == TensorParallelCollective::AllGather {
                                1.
                            } else {
                                2.
                            }));
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    control.finish(tx(1)).unwrap();
    assert!(
        peers[0]
            .exchange(tx(1), 0, TensorParallelCollective::AllGather, vec![1.; 6])
            .is_err()
    );
    assert!(control.is_failed());
    let mut peers = raw_peers();
    register(&peers, &schedule());
    assert!(
        peers[0]
            .exchange(tx(1), 0, TensorParallelCollective::AllGather, vec![1.; 6])
            .is_err()
    );
    assert!(peers[0].control().is_failed());
}

#[test]
fn incomplete_registration_wrong_epoch_and_legacy_begin_cannot_bypass_seal() {
    let peers = raw_peers();
    let control = peers[0].control();
    assert!(
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    control
        .register_sites(peers[0].owner(), schedule())
        .unwrap();
    assert!(
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    assert!(
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    control
        .register_sites(peers[1].owner(), schedule())
        .unwrap();
    let topology = ferrule_common::ValidatedParallelTopology::new(
        ParallelTopologyId::new(502),
        2,
        ParallelRankId::new(0),
        ferrule_common::ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
    )
    .unwrap();
    let binding =
        KvCommitBinding::new(tx(1), topology.topology_id(), topology.participants(), 17).unwrap();
    let wrong = TensorBatchIdentity::from_packed(
        &binding,
        &super::tests::packed(ferrule_common::execution::ForwardPhase::Decode, 0, 5),
        &[11],
    )
    .unwrap();
    assert!(
        control
            .begin_sealed(wrong, Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    admit(&peers);
    assert!(control.finish(tx(1)).is_err());
    assert!(
        control
            .register_sites(peers[0].owner(), schedule())
            .is_err()
    );
    assert!(control.is_failed());
}

#[test]
fn semantic_registration_checks_operator_layer_shape_and_dense_degree() {
    for case in 0..5 {
        let peers = raw_peers();
        let control = peers[0].control();
        let original = vec![site(
            9,
            Some(8),
            TensorCollectiveSemantic::SharedExpertOutput,
            TensorCollectiveShape::PerRow(4),
        )];
        control
            .register_sites(peers[0].owner(), original.clone())
            .unwrap();
        let mut changed = original;
        match case {
            0 => changed[0].layer = Some(9),
            1 => {
                changed[0].operator = TensorCollectiveOperator::Protocol {
                    version: 2,
                    kind: HostCollectiveKind::AllReduceSumF32,
                    operation: TensorCollectiveSemantic::RoutedExpertOutput,
                    shape: TensorCollectiveShape::PerRow(4),
                }
            }
            2 => {
                changed[0].operator = TensorCollectiveOperator::Protocol {
                    version: 2,
                    kind: HostCollectiveKind::AllReduceSumF32,
                    operation: TensorCollectiveSemantic::SharedExpertOutput,
                    shape: TensorCollectiveShape::Fixed(4),
                }
            }
            3 => changed[0].site = 10,
            _ => {
                changed[0].operator =
                    TensorParallelLinearPlan::new(4, 4, 2, TensorParallelLinearPartition::Row)
                        .unwrap()
                        .into()
            }
        }
        assert!(control.register_sites(peers[1].owner(), changed).is_err());
        assert!(control.is_failed());
    }
    for bad in [
        site(
            0,
            None,
            TensorCollectiveSemantic::Frontier,
            TensorCollectiveShape::PerRow(1),
        ),
        site(
            0,
            Some(8),
            TensorCollectiveSemantic::InputAgreement,
            TensorCollectiveShape::Fixed(1),
        ),
        site(
            0,
            None,
            TensorCollectiveSemantic::RoutedExpertOutput,
            TensorCollectiveShape::Fixed(1),
        ),
        site(
            0,
            None,
            TensorCollectiveSemantic::Frontier,
            TensorCollectiveShape::Fixed(0),
        ),
        TensorCollectiveSite {
            site: 0,
            layer: Some(8),
            operator: TensorParallelLinearPlan::new(4, 4, 1, TensorParallelLinearPartition::Row)
                .unwrap()
                .into(),
        },
    ] {
        let peers = raw_peers();
        assert!(
            peers[0]
                .control()
                .register_sites(peers[0].owner(), vec![bad])
                .is_err()
        );
    }
}

#[test]
fn sealed_semantic_wrong_site_kind_count_or_missing_owner_validation_wakes_peer() {
    for case in 0..4 {
        let mut peers = raw_peers();
        register(&peers, &schedule());
        let mut second = peers.pop().unwrap();
        let mut first = peers.pop().unwrap();
        let control = first.control();
        control
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        control
            .validate_owner_execution(first.owner(), &identity())
            .unwrap();
        if case != 3 {
            control
                .validate_owner_execution(second.owner(), &identity())
                .unwrap();
        }
        let waiter = std::thread::spawn(move || {
            first.exchange(tx(1), 0, TensorParallelCollective::AllGather, vec![1.; 6])
        });
        wait(&control);
        let kind = if case == 1 {
            TensorParallelCollective::Sum
        } else {
            TensorParallelCollective::AllGather
        };
        assert!(
            second
                .exchange(
                    tx(1),
                    u64::from(case == 0),
                    kind,
                    vec![1.; if case == 2 { 5 } else { 6 }]
                )
                .is_err()
        );
        assert!(waiter.join().unwrap().is_err());
        assert!(control.is_failed());
    }
}

#[test]
fn checked_shape_and_full_schedule_memory_preflight_reject_before_admission() {
    use TensorCollectiveShape::*;
    assert_eq!(
        Rows {
            fixed: 4,
            per_row: 3
        }
        .elements(1, 2)
        .unwrap(),
        10
    );
    for shape in [
        Fixed(0),
        PerRow(0),
        Rows {
            fixed: usize::MAX,
            per_row: 1,
        },
        PerRow(usize::MAX),
    ] {
        assert!(shape.elements(1, 2).is_err());
    }
    for count in [33, usize::MAX] {
        let peers = raw_peers();
        register(
            &peers,
            &[site(
                0,
                None,
                TensorCollectiveSemantic::Frontier,
                Fixed(count),
            )],
        );
        assert!(
            peers[0]
                .control()
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
        assert!(!peers[0].control().is_failed());
    }
    let peers = DecoderTensorCollective::new_group_from_members(
        &[ParallelRankId::new(0), ParallelRankId::new(1)],
        ParallelTopologyId::new(501),
        ParallelGroupId::new(4),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 32,
            max_host_bytes: 300,
        },
        Duration::from_secs(5),
    )
    .unwrap();
    register(
        &peers,
        &[site(
            0,
            None,
            TensorCollectiveSemantic::InputAgreement,
            Fixed(16),
        )],
    );
    assert!(
        peers[0]
            .control()
            .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}
