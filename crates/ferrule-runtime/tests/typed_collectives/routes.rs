use super::*;
use ferrule_model::transformer::{RouterRoutes, hybrid_parallel::*};
use std::sync::atomic::{AtomicUsize, Ordering};
fn site() -> HybridMixedRouteSite {
    HybridMixedRouteSite {
        layer: 0,
        output_site: 1,
        experts: 4,
        top_k: 2,
        max_rows: 1,
        accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
        combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
    }
}
fn plan() -> MixedRankCollectivePlan {
    let sites = [
        TensorCollectiveSemantic::AttentionOutput,
        TensorCollectiveSemantic::RoutedExpertOutput,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, operation)| TensorCollectiveSite {
        site: i as u64,
        layer: Some(0),
        operator: TensorCollectiveOperator::Protocol {
            version: HYBRID_AGREEMENT_VERSION,
            operation,
            kind: K::AllReduceSumBf16,
            shape: TensorCollectiveShape::PerRow(1),
        },
    })
    .collect();
    Arc::unwrap_or_clone(mixed_plan(0, sites))
        .with_route_sites(vec![site()])
        .unwrap()
}
fn record(
    site: HybridMixedRouteSite,
    ids: Vec<usize>,
    weights: Vec<f32>,
) -> HybridMixedRouteAgreement {
    HybridMixedRouteAgreement::new(
        site.clone(),
        &RouterRoutes::new(ids.len() / site.top_k, site.top_k, ids, weights).unwrap(),
    )
    .unwrap()
}
fn normal() -> HybridMixedRouteAgreement {
    record(site(), vec![3, 0], vec![0.5, 0.5])
}
fn attention(peers: &mut [MixedRankCollective], tx: ExecutionTransactionId) {
    std::thread::scope(|s| {
        let hs = peers
            .iter_mut()
            .map(|p| s.spawn(move || p.exchange_bf16(tx, 0, vec![0x3f80])))
            .collect::<Vec<_>>();
        for h in hs {
            assert_eq!(h.join().unwrap().unwrap(), vec![0x4000]);
        }
    });
}
fn admitted(timeout: Duration) -> Vec<MixedRankCollective> {
    let plan = Arc::new(plan());
    let mut peers = DecoderTensorCollective::new_group_from_members(
        plan.members(),
        plan.epoch(),
        plan.group(),
        plan.limits(),
        timeout,
    )
    .unwrap()
    .into_iter()
    .map(|e| {
        let owner = e.owner();
        MixedRankCollective::prepare(e, owner, plan.clone()).unwrap()
    })
    .collect::<Vec<_>>();
    peers[0]
        .control()
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for p in &peers {
        p.validate_owner_execution(&identity()).unwrap();
    }
    attention(&mut peers, tx(1));
    peers
}
#[test]
fn exact_routes_gate_experts_and_late_sum_across_transactions() {
    let mut peers = admitted(Duration::from_secs(5));
    let control = peers[0].control();
    assert_eq!(peers[0].sealed_route_sites().unwrap(), vec![site()]);
    for id in [1, 2] {
        if id == 2 {
            let binding = KvCommitBinding::new(
                tx(2),
                topology().topology_id(),
                topology().participants(),
                18,
            )
            .unwrap();
            let next = TensorBatchIdentity::from_packed(
                &binding,
                &packed(ForwardPhase::Decode, 0, 6),
                &[11],
            )
            .unwrap();
            control
                .begin_sealed(next.clone(), Arc::new(AtomicBool::new(false)))
                .unwrap();
            for p in &peers {
                p.validate_owner_execution(&next).unwrap();
            }
            attention(&mut peers, tx(id));
        }
        let experts = AtomicUsize::new(0);
        let (left, right) = peers.split_at_mut(1);
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                left[0].agree_routes(tx(id), &normal()).unwrap();
                experts.fetch_add(1, Ordering::SeqCst);
            });
            wait_for_peer(&control);
            assert_eq!(experts.load(Ordering::SeqCst), 0);
            assert!(control.finish(tx(id)).is_err());
            right[0].agree_routes(tx(id), &normal()).unwrap();
            h.join().unwrap();
        });
        std::thread::scope(|s| {
            let hs = peers
                .iter_mut()
                .map(|p| {
                    s.spawn(move || {
                        assert_eq!(
                            p.exchange_bf16(tx(id), 1, vec![0x3f81]).unwrap(),
                            vec![0x4001]
                        );
                        p.complete_owner(tx(id)).unwrap();
                    })
                })
                .collect::<Vec<_>>();
            for h in hs {
                h.join().unwrap();
            }
        });
        control.finish(tx(id)).unwrap();
    }
}
#[test]
fn mismatched_bits_ids_layer_topk_and_transaction_release_no_experts() {
    for fault in 0..6 {
        let mut peers = admitted(Duration::from_secs(5));
        let control = peers[0].control();
        let mut changed = site();
        let other = match fault {
            0 => record(
                changed,
                vec![3, 0],
                vec![f32::from_bits(0.5_f32.to_bits() + 1), 0.5],
            ),
            1 => record(changed, vec![0, 3], vec![0.5, 0.5]),
            2 => {
                changed.layer = 1;
                record(changed, vec![3, 0], vec![0.5, 0.5])
            }
            3 => {
                changed.top_k = 1;
                record(changed, vec![3], vec![1.0])
            }
            4 => {
                changed.output_site = 9;
                record(changed, vec![3, 0], vec![0.5, 0.5])
            }
            _ => normal(),
        };
        let (left, right) = peers.split_at_mut(1);
        std::thread::scope(|s| {
            let h = s.spawn(|| left[0].agree_routes(tx(1), &normal()));
            wait_for_peer(&control);
            assert!(
                right[0]
                    .agree_routes(tx(if fault == 5 { 2 } else { 1 }), &other)
                    .is_err()
            );
            assert!(h.join().unwrap().is_err());
        });
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
        assert!(peers[0].complete_owner(tx(1)).is_err());
        assert!(peers[1].exchange_bf16(tx(1), 1, vec![0x3f80]).is_err());
    }
}

#[test]
fn missing_rank_cancel_timeout_and_endpoint_loss_are_terminal() {
    for fault in 0..3 {
        let mut peers = admitted(if fault == 1 {
            Duration::from_millis(200)
        } else {
            Duration::from_secs(5)
        });
        let mut right = Some(peers.pop().unwrap());
        let mut left = peers.pop().unwrap();
        let control = left.control();
        let failure = std::thread::scope(|s| {
            let h = s.spawn(|| left.agree_routes(tx(1), &normal()));
            wait_for_peer(&control);
            if fault == 0 {
                control.abort(tx(1), "cancel before experts").unwrap();
            }
            if fault == 2 {
                drop(right.take());
            }
            h.join().unwrap().unwrap_err()
        });
        if fault == 2 {
            assert!(
                ErrorChain::new(&failure)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        }
        if fault == 1 {
            assert!(failure.to_string().contains("timed out"));
        }
        assert!(control.is_failed());
        assert!(left.complete_owner(tx(1)).is_err());
        assert!(control.finish(tx(1)).is_err());
        assert!(
            control
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
    }
}
#[test]
fn skipping_duplicate_or_early_route_fails_closed() {
    for duplicate in [false, true] {
        let mut peers = admitted(Duration::from_secs(5));
        let control = peers[0].control();
        if duplicate {
            std::thread::scope(|s| {
                let hs = peers
                    .iter_mut()
                    .map(|p| s.spawn(|| p.agree_routes(tx(1), &normal())))
                    .collect::<Vec<_>>();
                for h in hs {
                    h.join().unwrap().unwrap();
                }
            });
            assert!(peers[0].agree_routes(tx(1), &normal()).is_err());
        } else {
            assert!(peers[0].exchange_bf16(tx(1), 1, vec![0x3f80]).is_err());
        }
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
    let mut peers = mixed_peers(Arc::new(plan()));
    let control = peers[0].control();
    control
        .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for p in &peers {
        p.validate_owner_execution(&identity()).unwrap();
    }
    assert!(peers[0].agree_routes(tx(1), &normal()).is_err());
    assert!(control.is_failed());
}
#[test]
fn invalid_model_records_never_reach_experts_and_control_bytes_are_not_zero() {
    for (ids, weights) in [
        (vec![0, 0], vec![0.5, 0.5]),
        (vec![4, 0], vec![0.5, 0.5]),
        (vec![0, 1], vec![f32::NAN, 0.5]),
        (vec![0, 1], vec![f32::INFINITY, 0.5]),
        (vec![0, 1], vec![-0.5, 0.5]),
    ] {
        let routes = RouterRoutes::new(1, 2, ids, weights).unwrap();
        assert!(HybridMixedRouteAgreement::new(site(), &routes).is_err());
        // The actual model failure boundary aborts its transport when record
        // construction rejects a route. That must wake an already waiting rank.
        let mut peers = admitted(Duration::from_secs(5));
        let control = peers[0].control();
        let (left, right) = peers.split_at_mut(1);
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| left[0].agree_routes(tx(1), &normal()));
            wait_for_peer(&control);
            right[0].abort();
            assert!(waiting.join().unwrap().is_err());
        });
        assert!(control.is_failed());
        assert!(control.finish(tx(1)).is_err());
    }
    let p = plan();
    let b = p.route_control_budget().unwrap();
    assert_eq!(
        b.payload_bytes_per_owner,
        2 * (std::mem::size_of::<usize>() + 4)
    );
    assert_eq!(b.service_payload_bytes, 2 * b.payload_bytes_per_owner);
    assert_eq!(b.caller_payload_bytes, b.service_payload_bytes);
    assert!(b.metadata_bytes > 0);
    assert!(p.payload_budget().total_host_bytes > b.total_host_bytes);
    for fault in 0..4 {
        let mut changed = site();
        match fault {
            0 => changed.layer = 1,
            1 => changed.output_site = 0,
            2 => changed.max_rows = 2,
            _ => changed.top_k = 5,
        }
        assert!(p.clone().with_route_sites(vec![changed]).is_err());
    }
    assert!(p.clone().with_route_sites(vec![]).is_err());
    assert!(p.with_route_sites(vec![site(), site()]).is_err());
}
#[test]
fn actual_model_full40_route_sites_bind_to_production_adapter_not_a_private_transport() {
    use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};
    let config = Qwen35Config::from_value(
        &serde_json::from_str(include_str!(
            "../../../ferrule-model/tests/qwen35_35b_fp8_config.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let spec = Qwen35Recipe::spec(&config).unwrap();
    let parallel = HybridParallelPlan::new_with_execution(
        &spec,
        0..40,
        HybridExpertScope {
            topology: topology().topology_id(),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    let blocks = (0..2)
        .map(|lane| {
            HybridMixedRankProgramPlan::from_layout(
                Arc::new(parallel.rank_layout(lane).unwrap()),
                32,
                2,
                1024,
                16,
                5,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let routes = blocks[0].route_sites();
    assert_eq!(routes.len(), 40);
    let plan = Arc::new(
        MixedRankCollectivePlan::from_hybrid_sites(
            &parallel,
            ParallelGroupId::new(4),
            ParallelRankId::new(0),
            HYBRID_AGREEMENT_VERSION,
            &blocks[0].sites(),
            HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 32 * 2048,
                max_host_bytes: 1 << 20,
            },
            2,
            32,
        )
        .unwrap()
        .with_route_sites(routes.clone())
        .unwrap(),
    );
    let budget = plan.route_control_budget().unwrap();
    assert_eq!(
        budget.payload_bytes_per_owner,
        32 * 8 * (std::mem::size_of::<usize>() + 4)
    );
    let peers = mixed_peers(plan);
    for (block, peer) in blocks.iter().zip(&peers) {
        block.validate_collective(peer).unwrap();
        assert_eq!(peer.sealed_route_sites().unwrap(), routes);
    }
    assert!(
        parallel.validate_rank_execution().is_err(),
        "metadata is not full-model execution evidence"
    );
}
