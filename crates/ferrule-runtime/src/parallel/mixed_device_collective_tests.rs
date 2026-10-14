use super::*;
use crate::parallel::pipeline::tensor_scope::{TensorBatchIdentity, tests::identity};
use ferrule_model::transformer::{RouterRoutes, hybrid_parallel::HybridBf16Collective};

struct Fixture {
    peers: Vec<MixedRankCollective>,
    control: DecoderTensorCollectiveControl,
    packed: TensorBatchIdentity,
    cancellation: Arc<AtomicBool>,
}
impl Fixture {
    fn new(enabled: bool) -> Self {
        let packed = identity();
        let members = vec![ParallelRankId::new(0), ParallelRankId::new(1)];
        let route = HybridMixedRouteSite {
            layer: 0,
            output_site: 1,
            experts: 4,
            top_k: 2,
            max_rows: 1,
            accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
        };
        let sites = [
            TensorCollectiveSemantic::AttentionOutput,
            TensorCollectiveSemantic::RoutedExpertOutput,
            TensorCollectiveSemantic::OutputHead,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, operation)| TensorCollectiveSite {
            site: i as u64,
            layer: (i < 2).then_some(0),
            operator: TensorCollectiveOperator::Protocol {
                version: 2,
                operation,
                kind: if i == 2 {
                    HostCollectiveKind::AllGatherBf16
                } else {
                    HostCollectiveKind::AllReduceSumBf16
                },
                shape: TensorCollectiveShape::PerRow(1),
            },
        })
        .collect();
        let plan = Arc::new(
            MixedRankCollectivePlan::new(
                packed.binding().topology_id(),
                ParallelGroupId::new(4),
                members.clone(),
                members[0],
                sites,
                HostCollectiveLimits {
                    max_ranks: 2,
                    max_elements_per_rank: 4,
                    max_host_bytes: 1024,
                },
                1,
                1,
            )
            .unwrap()
            .with_route_sites(vec![route])
            .unwrap(),
        );
        let peers = DecoderTensorCollective::new_group_from_members(
            &members,
            plan.epoch(),
            plan.group(),
            plan.limits(),
            Duration::from_secs(5),
        )
        .unwrap()
        .into_iter()
        .map(|endpoint| {
            let owner = endpoint.owner();
            let mut peer = MixedRankCollective::prepare(endpoint, owner, plan.clone()).unwrap();
            // This is an internal CPU fault seam, not a production enable API.
            peer.device_participation = enabled;
            peer
        })
        .collect::<Vec<_>>();
        let control = peers[0].control();
        let cancellation = Arc::new(AtomicBool::new(false));
        control
            .begin_sealed(packed.clone(), cancellation.clone())
            .unwrap();
        for peer in &peers {
            peer.validate_owner_execution(&packed).unwrap();
        }
        Self {
            peers,
            control,
            packed,
            cancellation,
        }
    }
    fn tx(&self) -> ExecutionTransactionId {
        self.packed.binding().transaction()
    }
    fn request(&self, owner: usize, site: usize) -> DevicePayloadIdentity {
        let binding = self.peers[owner].sealed_binding().unwrap();
        let site = binding.sites[site].clone();
        let rows = self.packed.rows();
        let sequences = self.packed.sequences();
        DevicePayloadIdentity {
            epoch: binding.epoch,
            transaction: self.tx(),
            owner: self.peers[owner].owner(),
            members: [binding.members[0], binding.members[1]],
            elements: site.shape.elements(sequences, rows).unwrap(),
            site,
            rows,
            sequences,
        }
    }
    fn permits(&mut self, site: usize) -> Vec<DevicePayloadPermit> {
        let requests = [self.request(0, site), self.request(1, site)];
        std::thread::scope(|s| {
            let handles = self
                .peers
                .iter_mut()
                .zip(requests)
                .map(|(peer, request)| s.spawn(move || peer.permit_device_payload(request)))
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap().unwrap())
                .collect()
        })
    }
    fn enqueue_pair(&mut self, site: usize) {
        let permits = self.permits(site);
        for (peer, permit) in self.peers.iter_mut().zip(permits) {
            peer.check_device_payload_permit(&permit).unwrap();
            peer.device_payload_enqueued(permit).unwrap();
        }
    }
    fn routes(&mut self) {
        let route = self.peers[0].plan.route_sites[0].clone();
        let agreement = HybridMixedRouteAgreement::new(
            route,
            &RouterRoutes::new(1, 2, vec![0, 1], vec![0.5, 0.5]).unwrap(),
        )
        .unwrap();
        let tx = self.tx();
        std::thread::scope(|s| {
            let handles = self
                .peers
                .iter_mut()
                .map(|peer| {
                    let agreement = &agreement;
                    s.spawn(move || peer.agree_routes(tx, agreement))
                })
                .collect::<Vec<_>>();
            for h in handles {
                h.join().unwrap().unwrap();
            }
        });
    }
    fn host(&mut self, site: u64) {
        let tx = self.tx();
        std::thread::scope(|s| {
            let handles = self
                .peers
                .iter_mut()
                .map(|peer| s.spawn(move || peer.exchange_bf16(tx, site, vec![0x3f80])))
                .collect::<Vec<_>>();
            for h in handles {
                let words = h.join().unwrap().unwrap();
                assert_eq!(
                    words,
                    if site == 2 {
                        vec![0x3f80, 0x3f80]
                    } else {
                        vec![0x4000]
                    }
                );
            }
        });
    }
    fn device_schedule(&mut self) {
        self.enqueue_pair(0);
        self.routes();
        self.enqueue_pair(1);
        self.host(2);
    }
    fn assert_retained(&self) {
        let state = self.control.0.state.lock().unwrap();
        assert_eq!(state.execution.as_ref().map(|e| e.0), Some(self.tx()));
        assert_eq!(state.execution_watermark, Some(self.tx()));
        assert!(state.failure.is_some());
    }
}
fn wait_for_waiter(control: &DecoderTensorCollectiveControl) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while control.waiting_members() == 0 {
        assert!(Instant::now() < deadline, "metadata owner did not wait");
        std::thread::yield_now();
    }
}
fn unknown(source: &Error) {
    assert!(
        ErrorChain::new(source)
            .downcast_ref::<CompletionUnknown>()
            .is_some(),
        "{source}"
    );
}

#[test]
fn default_host_preserves_vec_and_exact_route_schedule() {
    let mut f = Fixture::new(false);
    assert_eq!(
        f.peers[0].data_location(&f.request(0, 0).site),
        DataLocation::Host
    );
    f.host(0);
    f.routes();
    f.host(1);
    f.host(2);
    for peer in &f.peers {
        peer.complete_owner(f.tx()).unwrap();
    }
    f.control.finish(f.tx()).unwrap();
}

#[test]
fn metadata_needs_no_data_vec_and_only_enqueue_advances() {
    let mut f = Fixture::new(true);
    let request0 = f.request(0, 0);
    let request1 = f.request(1, 0);
    let (first, second) = f.peers.split_at_mut(1);
    let permits = std::thread::scope(|s| {
        let h = s.spawn(|| first[0].permit_device_payload(request0));
        wait_for_waiter(&f.control);
        {
            let state = f.control.0.state.lock().unwrap();
            assert_eq!(state.group.active_descriptor().unwrap().count, 0);
            assert_eq!(state.completed, [0, 0]);
            assert!(state.device_permissions.iter().all(Option::is_none));
        }
        let p1 = second[0].permit_device_payload(request1).unwrap();
        vec![h.join().unwrap().unwrap(), p1]
    });
    {
        let state = f.control.0.state.lock().unwrap();
        assert_eq!(state.completed, [0, 0]);
        assert!(state.group.active_descriptor().is_none());
        assert!(state.device_permissions.iter().all(Option::is_some));
        assert_eq!(state.device_uncompleted, [false, false]);
    }
    let mut permits = permits.into_iter();
    f.peers[0]
        .device_payload_enqueued(permits.next().unwrap())
        .unwrap();
    assert_eq!(f.control.0.state.lock().unwrap().completed, [1, 0]);
    f.peers[1]
        .device_payload_enqueued(permits.next().unwrap())
        .unwrap();
    assert_eq!(f.control.0.state.lock().unwrap().completed, [1, 1]);
}

#[test]
fn wrong_identity_wakes_both_with_same_failure_before_permission() {
    for case in 0..16 {
        let mut f = Fixture::new(true);
        let first_request = f.request(0, 0);
        let mut bad = f.request(1, 0);
        match case {
            0 => bad.site.ordinal = 99,
            1 => bad.site.shape.per_row += 1,
            2 => bad.epoch = ParallelTopologyId::new(bad.epoch.get() + 1),
            3 => bad.elements = usize::MAX,
            4 => bad.owner = ParallelRankId::new(99),
            5 => bad.members.reverse(),
            6 => bad.site.semantic = HybridCollectiveSemantic::OutputHead,
            7 => bad.site.kind = HybridCollectiveKind::AllGather,
            8 => bad.site.global_layer = Some(99),
            9 => bad.rows = 9,
            10 => bad.sequences += 1,
            11 => bad.transaction = ExecutionTransactionId::new(bad.transaction.get() + 1).unwrap(),
            12 => bad.site = f.request(1, 1).site,
            13 => bad.elements = 0,
            14 => f.peers[1].device_participation = false,
            15 => f.control.0.state.lock().unwrap().validated[1] = false,
            _ => unreachable!(),
        }
        let (first, second) = f.peers.split_at_mut(1);
        std::thread::scope(|s| {
            let h = s.spawn(|| first[0].permit_device_payload(first_request));
            wait_for_waiter(&f.control);
            let rejected = second[0].permit_device_payload(bad).unwrap_err();
            let waiting = h.join().unwrap().unwrap_err();
            assert_eq!(waiting.to_string(), rejected.to_string(), "case {case}");
        });
        let state = f.control.0.state.lock().unwrap();
        assert_eq!(state.completed, [0, 0]);
        assert!(state.device_permissions.iter().all(Option::is_none));
        assert!(state.group.active_descriptor().is_none());
        drop(state);
        f.assert_retained();
    }
}

#[test]
fn host_and_metadata_cannot_agree_on_one_data_site() {
    let mut f = Fixture::new(true);
    let request = f.request(0, 0);
    let tx = f.tx();
    let (first, second) = f.peers.split_at_mut(1);
    std::thread::scope(|s| {
        let h = s.spawn(|| first[0].permit_device_payload(request));
        wait_for_waiter(&f.control);
        let host = second[0].exchange_bf16(tx, 0, vec![0x3f80]).unwrap_err();
        let metadata = h.join().unwrap().unwrap_err();
        assert_eq!(host.to_string(), metadata.to_string());
    });
    assert_eq!(f.control.0.state.lock().unwrap().completed, [0, 0]);
    f.assert_retained();
}

#[test]
fn each_owner_reports_real_completion_before_parent_finish() {
    let mut f = Fixture::new(true);
    f.device_schedule();
    let tx = f.tx();
    assert_eq!(
        f.control.0.state.lock().unwrap().device_uncompleted,
        [true, true]
    );
    f.peers[0].complete_device_payloads(tx, Ok(())).unwrap();
    assert_eq!(
        f.control.0.state.lock().unwrap().device_uncompleted,
        [false, true]
    );
    f.peers[0].complete_owner(tx).unwrap();
    f.peers[1].complete_device_payloads(tx, Ok(())).unwrap();
    f.peers[1].complete_owner(tx).unwrap();
    f.control.finish(tx).unwrap();
    let state = f.control.0.state.lock().unwrap();
    assert!(state.execution.is_none());
    assert!(!state.device_unknown);
    assert_eq!(state.execution_watermark, Some(tx));
}

#[test]
fn full_schedule_without_all_device_completions_cannot_mark_ready() {
    for owner_or_parent in 0..3 {
        let mut f = Fixture::new(true);
        f.device_schedule();
        let tx = f.tx();
        let failure = match owner_or_parent {
            0 => f.peers[0].complete_owner(tx).unwrap_err(),
            1 => f.control.finish(tx).unwrap_err(),
            _ => {
                f.peers[0].complete_device_payloads(tx, Ok(())).unwrap();
                f.peers[0].complete_owner(tx).unwrap();
                f.control.finish(tx).unwrap_err()
            }
        };
        unknown(&failure);
        unknown(&f.peers[1].complete_device_payloads(tx, Ok(())).unwrap_err());
        unknown(&f.peers[0].complete_owner(tx).unwrap_err());
        f.assert_retained();
        assert!(f.control.0.state.lock().unwrap().device_uncompleted[1]);
    }
}

#[test]
fn per_site_completion_cannot_fake_invocation_fence() {
    let mut f = Fixture::new(true);
    f.enqueue_pair(0);
    let tx = f.tx();
    unknown(&f.peers[0].complete_device_payloads(tx, Ok(())).unwrap_err());
    unknown(&f.control.finish(tx).unwrap_err());
    assert_eq!(
        f.control.0.state.lock().unwrap().device_uncompleted,
        [true, true]
    );
    f.assert_retained();
}

#[test]
fn finite_error_is_unknown_shared_and_preserves_original_source() {
    let mut f = Fixture::new(true);
    f.device_schedule();
    let tx = f.tx();
    let report = Err(Error::ModelSource {
        source: Box::new(std::io::Error::other("device finite flag failed")),
    });
    let failure = f.peers[0].complete_device_payloads(tx, report).unwrap_err();
    unknown(&failure);
    assert!(
        ErrorChain::new(&failure)
            .downcast_ref::<std::io::Error>()
            .is_some()
    );
    unknown(&f.peers[1].complete_owner(tx).unwrap_err());
    unknown(&f.control.finish(tx).unwrap_err());
    assert_eq!(
        f.control.0.state.lock().unwrap().device_uncompleted,
        [true, true]
    );
    f.assert_retained();
}

#[test]
fn duplicate_vote_and_foreign_enqueue_fail_jointly_without_advancing() {
    for duplicate_vote in [true, false] {
        let mut f = Fixture::new(true);
        let request = f.request(0, 0);
        let mut permits = f.permits(0);
        let second = permits.pop().unwrap();
        let first = permits.pop().unwrap();
        let failure = if duplicate_vote {
            f.peers[0].permit_device_payload(request).unwrap_err()
        } else {
            f.peers[0].device_payload_enqueued(second).unwrap_err()
        };
        unknown(&failure);
        unknown(&f.peers[1].check_device_payload_permit(&first).unwrap_err());
        let state = f.control.0.state.lock().unwrap();
        assert_eq!(state.completed, [0, 0]);
        assert!(state.device_permissions.iter().all(Option::is_some));
        drop(state);
        f.assert_retained();
    }
}

#[test]
fn cancellation_before_permission_is_a_common_rejection() {
    for shared_abort in [false, true] {
        let mut f = Fixture::new(true);
        let a = f.request(0, 0);
        let b = f.request(1, 0);
        let tx = f.tx();
        let (first, second) = f.peers.split_at_mut(1);
        std::thread::scope(|s| {
            let h = s.spawn(|| first[0].permit_device_payload(a));
            wait_for_waiter(&f.control);
            if shared_abort {
                f.control.abort(tx, "CPU metadata cancellation").unwrap();
            } else {
                f.cancellation.store(true, Ordering::Release);
            }
            let rejected = second[0].permit_device_payload(b).unwrap_err();
            assert_eq!(
                h.join().unwrap().unwrap_err().to_string(),
                rejected.to_string()
            );
        });
        assert_eq!(f.control.0.state.lock().unwrap().completed, [0, 0]);
        f.assert_retained();
    }
}

#[test]
fn peer_drop_before_permission_is_terminal_unknown() {
    let mut f = Fixture::new(true);
    let request = f.request(0, 0);
    let second = f.peers.pop().unwrap();
    std::thread::scope(|s| {
        let h = s.spawn(|| f.peers[0].permit_device_payload(request));
        wait_for_waiter(&f.control);
        drop(second);
        unknown(&h.join().unwrap().unwrap_err());
    });
    unknown(&f.control.finish(f.tx()).unwrap_err());
    assert_eq!(f.control.0.state.lock().unwrap().completed, [0, 0]);
    f.assert_retained();
}

#[test]
fn drop_or_cancel_after_permission_or_enqueue_retains_custody() {
    for enqueued in [false, true] {
        for drop_peer in [false, true] {
            let mut f = Fixture::new(true);
            let permits = f.permits(0);
            if enqueued {
                for (peer, permit) in f.peers.iter_mut().zip(permits) {
                    peer.device_payload_enqueued(permit).unwrap();
                }
            } else {
                drop(permits);
            }
            if drop_peer {
                drop(f.peers.pop().unwrap());
            } else {
                f.control
                    .abort(f.tx(), "CPU cancellation after permission")
                    .unwrap();
            }
            unknown(&f.peers[0].complete_owner(f.tx()).unwrap_err());
            unknown(&f.control.finish(f.tx()).unwrap_err());
            let state = f.control.0.state.lock().unwrap();
            assert_eq!(state.completed, [usize::from(enqueued); 2]);
            if enqueued {
                assert_eq!(state.device_uncompleted, [true, true]);
            } else {
                assert!(state.device_permissions.iter().all(Option::is_some));
            }
            drop(state);
            f.assert_retained();
        }
    }
}

#[test]
fn terminal_peer_loss_dominates_ready_and_drained_metadata() {
    for drained in [false, true] {
        for close_transition in [false, true] {
            let mut f = Fixture::new(true);
            let first = f.request(0, 0);
            let second = f.request(1, 0);
            let second_owner = f.peers[1].owner();
            std::thread::scope(|s| {
                let h = s.spawn(|| f.peers[0].permit_device_payload(first));
                wait_for_waiter(&f.control);
                {
                    let mut state = f.control.0.state.lock().unwrap();
                    let descriptor = state.group.active_descriptor().unwrap();
                    state.device_votes[1] = Some(second);
                    state
                        .group
                        .submit_typed(
                            second_owner,
                            descriptor,
                            CollectivePayload::Bf16(Vec::new()),
                        )
                        .unwrap();
                    assert_eq!(state.group.status(), Some(HostCollectiveStatus::Ready));
                    if drained {
                        for member in descriptor_members() {
                            assert!(matches!(
                                state.group.take_typed_result(member).unwrap(),
                                TypedCollectivePoll::Ready { .. }
                            ));
                        }
                        assert!(state.group.active_descriptor().is_none());
                    }
                    if close_transition {
                        close_peer(&mut state, 1);
                    } else {
                        state.closed[1] = true;
                    }
                    f.control.0.wake.notify_all();
                }
                unknown(&h.join().unwrap().unwrap_err());
            });
            unknown(&f.control.finish(f.tx()).unwrap_err());
            let state = f.control.0.state.lock().unwrap();
            assert_eq!(state.completed, [0, 0]);
            assert!(state.device_permissions.iter().all(Option::is_none));
            drop(state);
            f.assert_retained();
        }
    }
}
fn descriptor_members() -> [ParallelRankId; 2] {
    [ParallelRankId::new(0), ParallelRankId::new(1)]
}

#[test]
fn routed_output_requires_exact_route_agreement_before_permission() {
    let mut f = Fixture::new(true);
    f.enqueue_pair(0);
    let requests = [f.request(0, 1), f.request(1, 1)];
    for (peer, request) in f.peers.iter_mut().zip(requests) {
        unknown(&peer.permit_device_payload(request).unwrap_err());
    }
    let state = f.control.0.state.lock().unwrap();
    assert_eq!(state.completed, [1, 1]);
    assert!(state.device_permissions.iter().all(Option::is_none));
    drop(state);
    f.assert_retained();
}
