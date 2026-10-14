//! Model-independent endpoint construction and lifetime regressions.
use super::*;

fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
fn rank(id: u32) -> ParallelRankId {
    ParallelRankId::new(id)
}
fn limits() -> HostCollectiveLimits {
    HostCollectiveLimits {
        max_ranks: 4,
        max_elements_per_rank: 16,
        max_host_bytes: 4096,
    }
}
fn group(members: &[ParallelRankId]) -> Vec<DecoderTensorCollective> {
    DecoderTensorCollective::new_group_from_members(
        members,
        ParallelTopologyId::new(501),
        ParallelGroupId::new(19),
        limits(),
        Duration::from_secs(5),
    )
    .unwrap()
}
fn wait(control: &DecoderTensorCollectiveControl) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while control.waiting_members() == 0 {
        assert!(Instant::now() < deadline, "peer did not park");
        std::thread::yield_now();
    }
}

#[test]
fn constructor_rejects_empty_duplicate_limits_and_invalid_timeout() {
    let create = |members: &[ParallelRankId], limits, timeout| {
        DecoderTensorCollective::new_group_from_members(
            members,
            ParallelTopologyId::new(501),
            ParallelGroupId::new(19),
            limits,
            timeout,
        )
    };
    for members in [vec![], vec![rank(7), rank(7)], (0..5).map(rank).collect()] {
        assert!(create(&members, limits(), Duration::from_secs(1)).is_err());
    }
    for timeout in [Duration::ZERO, Duration::MAX] {
        assert!(create(&[rank(7)], limits(), timeout).is_err());
    }
    for bound in [
        HostCollectiveLimits {
            max_ranks: 0,
            ..limits()
        },
        HostCollectiveLimits {
            max_ranks: usize::MAX,
            ..limits()
        },
        HostCollectiveLimits {
            max_elements_per_rank: usize::MAX,
            ..limits()
        },
    ] {
        assert!(create(&[rank(7)], bound, Duration::from_secs(1)).is_err());
    }
}

#[test]
fn sparse_members_preserve_declared_order_not_arrival_or_numeric_order() {
    let members = [rank(91), rank(3), rank(700)];
    let mut peers = group(&members);
    let control = peers[0].control();
    for (peer, member) in peers.iter().zip(members) {
        assert_eq!(peer.owner(), member);
        assert_eq!(peer.members(), members);
    }
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .rev()
            .map(|peer| {
                scope.spawn(|| {
                    peer.exchange(
                        tx(1),
                        9,
                        TensorParallelCollective::AllGather,
                        vec![peer.owner().get() as f32],
                    )
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_eq!(handle.join().unwrap().unwrap(), vec![91., 3., 700.]);
        }
    });
    control.finish(tx(1)).unwrap();
}

#[test]
fn legacy_exchange_also_requires_active_begin_and_cannot_resume_after_finish() {
    for after_finish in [false, true] {
        let mut peers = group(&[rank(7)]);
        let control = peers[0].control();
        if after_finish {
            control
                .begin(tx(1), Arc::new(AtomicBool::new(false)))
                .unwrap();
            assert_eq!(
                peers[0]
                    .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![2.])
                    .unwrap(),
                vec![2.]
            );
            control.finish(tx(1)).unwrap();
        }
        assert!(
            peers[0]
                .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![2.])
                .is_err()
        );
        assert!(control.is_failed());
        assert!(
            control
                .begin(tx(2), Arc::new(AtomicBool::new(false)))
                .is_err()
        );
    }
}

#[test]
fn stale_abort_cannot_cancel_new_execution_and_cancelled_finish_is_not_success() {
    let mut peers = group(&[rank(7)]);
    let control = peers[0].control();
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    peers[0]
        .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![1.])
        .unwrap();
    control.finish(tx(1)).unwrap();
    assert!(
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    let cancellation = Arc::new(AtomicBool::new(false));
    control.begin(tx(2), cancellation.clone()).unwrap();
    assert!(control.abort(tx(1), "stale").is_err());
    assert!(!control.is_failed());
    assert!(!cancellation.load(Ordering::Acquire));
    peers[0]
        .exchange(tx(2), 0, TensorParallelCollective::Sum, vec![1.])
        .unwrap();
    cancellation.store(true, Ordering::Release);
    assert!(control.finish(tx(2)).is_err());
    control.abort(tx(2), "cancelled").unwrap();
    assert!(control.is_failed());
    assert!(
        control
            .begin(tx(3), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}

#[test]
fn peer_drop_and_abort_wake_waiters_and_permanently_fail_lifetime() {
    for drop_peer in [false, true] {
        let mut peers = group(&[rank(7), rank(9)]);
        let second = peers.pop().unwrap();
        let mut first = peers.pop().unwrap();
        let control = first.control();
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .unwrap();
        let worker = std::thread::spawn(move || {
            first.exchange(tx(1), 1, TensorParallelCollective::Sum, vec![1.])
        });
        wait(&control);
        if drop_peer {
            drop(second);
        } else {
            control.abort(tx(1), "inflight cancellation").unwrap();
        }
        assert!(worker.join().unwrap().is_err());
        assert!(control.is_failed());
        assert_eq!(control.waiting_members(), 0);
        assert!(control.finish(tx(1)).is_err());
    }
}

#[test]
fn timeout_and_host_capacity_failure_are_terminal_not_completion() {
    let mut peers = DecoderTensorCollective::new_group_from_members(
        &[rank(7), rank(9)],
        ParallelTopologyId::new(501),
        ParallelGroupId::new(19),
        limits(),
        Duration::from_millis(10),
    )
    .unwrap();
    let control = peers[0].control();
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    assert!(
        peers[0]
            .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![1.])
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert!(control.is_failed());
    for bound in [
        HostCollectiveLimits {
            max_elements_per_rank: 1,
            ..limits()
        },
        HostCollectiveLimits {
            max_host_bytes: 1,
            ..limits()
        },
    ] {
        let mut peers = DecoderTensorCollective::new_group_from_members(
            &[rank(7)],
            ParallelTopologyId::new(501),
            ParallelGroupId::new(19),
            bound,
            Duration::from_secs(1),
        )
        .unwrap();
        let control = peers[0].control();
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert!(
            peers[0]
                .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![1.; 2])
                .is_err()
        );
        assert!(control.is_failed());
    }
}

#[test]
fn dense_constructor_is_an_order_preserving_compatibility_wrapper() {
    use ferrule_model::models::qwen3::Qwen3DenseRecipe;
    use ferrule_model::transformer::{DecoderRecipe, StandardTensorPlacement};
    let spec = Qwen3DenseRecipe::new()
        .build_spec(&serde_json::json!({
            "architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","torch_dtype":"bfloat16",
            "hidden_act":"silu","vocab_size":4,"hidden_size":4,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":2,"intermediate_size":8,
            "max_position_embeddings":16,"rms_norm_eps":0.00001,"rope_theta":10000.0,
            "tie_word_embeddings":true,"attention_bias":false,"use_sliding_window":false,
            "attention_dropout":0.0,"use_cache":true,"max_window_layers":1,
            "initializer_range":0.02,"bos_token_id":1,"eos_token_id":2
        }))
        .unwrap();
    let members = [rank(91), rank(3)];
    let plan = StandardTensorPlan::new(
        &spec,
        members
            .iter()
            .enumerate()
            .map(|(device, &owner)| StandardTensorPlacement { owner, device })
            .collect(),
    )
    .unwrap();
    let old = DecoderTensorCollective::new_group(
        &plan,
        ParallelTopologyId::new(501),
        ParallelGroupId::new(19),
        limits(),
        Duration::from_secs(5),
    )
    .unwrap();
    let new = group(&members);
    for (old, new) in old.iter().zip(&new) {
        assert_eq!(old.owner(), new.owner());
        assert_eq!(old.members(), new.members());
    }
}

#[test]
fn pre_admission_cancellation_does_not_consume_watermark_or_poison_group() {
    let mut peers = group(&[rank(7)]);
    let control = peers[0].control();
    assert!(
        control
            .begin(tx(1), Arc::new(AtomicBool::new(true)))
            .is_err()
    );
    assert!(!control.is_failed());
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    peers[0]
        .exchange(tx(1), 0, TensorParallelCollective::Sum, vec![1.])
        .unwrap();
    control.finish(tx(1)).unwrap();
}

#[test]
fn mixed_ready_peer_close_cannot_release_result_or_reset_watermarks() {
    use crate::parallel::pipeline::tensor_scope::{
        TensorCollectiveOperator, TensorCollectiveSemantic, TensorCollectiveShape, tests::identity,
    };
    use ferrule_common::error::ErrorChain;
    use ferrule_model::decoder::CompletionUnknown;

    // The state mutex makes Ready -> close -> survivor wake deterministic. Use
    // the same close transition as Drop, not a second coordinator/test transport.
    for fail_on_close in [true, false] {
        let sites = vec![TensorCollectiveSite {
            site: 1,
            layer: Some(0),
            operator: TensorCollectiveOperator::Protocol {
                version: 2,
                operation: TensorCollectiveSemantic::GqaOutput,
                kind: HostCollectiveKind::AllReduceSumBf16,
                shape: TensorCollectiveShape::PerRow(1),
            },
        }];
        let plan = Arc::new(
            MixedRankCollectivePlan::new(
                ParallelTopologyId::new(501),
                ParallelGroupId::new(19),
                vec![rank(0), rank(1)],
                rank(0),
                sites,
                limits(),
                1,
                1,
            )
            .unwrap(),
        );
        let mut peers = group(&[rank(0), rank(1)]).into_iter();
        let mut first =
            MixedRankCollective::prepare(peers.next().unwrap(), rank(0), Arc::clone(&plan))
                .unwrap();
        let second = MixedRankCollective::prepare(peers.next().unwrap(), rank(1), plan).unwrap();
        let control = first.control();
        let cancelled = Arc::new(AtomicBool::new(false));
        control
            .begin_sealed(identity(), Arc::clone(&cancelled))
            .unwrap();
        first.validate_owner_execution(&identity()).unwrap();
        second.validate_owner_execution(&identity()).unwrap();
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| first.exchange_bf16(tx(1), 1, vec![0x3f80]));
            wait(&control);
            {
                let mut state = control.0.state.lock().unwrap();
                let descriptor = state.group.active_descriptor().unwrap();
                assert_eq!(state.group.status(), Some(HostCollectiveStatus::Pending));
                state
                    .group
                    .submit_typed(rank(1), descriptor, CollectivePayload::Bf16(vec![0x3b80]))
                    .unwrap();
                assert_eq!(state.group.status(), Some(HostCollectiveStatus::Ready));
                if fail_on_close {
                    close_peer(&mut state, 1);
                    assert!(state.failure.is_some());
                    assert!(cancelled.load(Ordering::Acquire));
                    assert!(state.group.active_descriptor().is_none());
                } else {
                    // Isolate the read-side guard: closure itself must dominate
                    // Ready, even without a pre-existing failure/cancel marker.
                    state.closed[1] = true;
                    assert!(state.failure.is_none());
                    assert!(!cancelled.load(Ordering::Acquire));
                }
                control.0.wake.notify_all();
            }
            let failure = waiting.join().unwrap().unwrap_err();
            assert!(
                ErrorChain::new(&failure)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        });
        for failure in [
            first.complete_owner(tx(1)).unwrap_err(),
            control.finish(tx(1)).unwrap_err(),
            control
                .begin_sealed(identity(), Arc::new(AtomicBool::new(false)))
                .unwrap_err(),
        ] {
            assert!(
                ErrorChain::new(&failure)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        }
        assert!(cancelled.load(Ordering::Acquire));
        assert_eq!(control.waiting_members(), 0);
        let state = control.0.state.lock().unwrap();
        assert_eq!(
            state.completed,
            vec![0, 0],
            "Ready was never published to an owner"
        );
        assert_eq!(state.group.sequence_watermark(), Some(1));
        assert_eq!(state.execution_watermark, Some(tx(1)));
        assert_eq!(state.execution.as_ref().map(|(id, _)| *id), Some(tx(1)));
        assert!(state.identity.is_some());
        assert!(state.group.active_descriptor().is_none());
    }
}

#[test]
fn legacy_f32_peer_close_after_host_drain_keeps_existing_finish_policy() {
    let mut peers = group(&[rank(7), rank(9)]);
    let mut second = peers.pop().unwrap();
    let mut first = peers.pop().unwrap();
    let control = first.control();
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    std::thread::scope(|scope| {
        let waiting =
            scope.spawn(|| first.exchange(tx(1), 1, TensorParallelCollective::Sum, vec![1.]));
        assert_eq!(
            second
                .exchange(tx(1), 1, TensorParallelCollective::Sum, vec![2.])
                .unwrap(),
            vec![3.]
        );
        assert_eq!(waiting.join().unwrap().unwrap(), vec![3.]);
    });
    drop(second);
    assert!(!control.is_failed());
    control.finish(tx(1)).unwrap();
}
