use super::*;

#[test]
fn failed_completion_error_retains_materializer_admission_until_custody_drops() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    f.replace();
    let error = sequence.demand(0, 0).unwrap_err();
    drop(sequence);
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    drop(error);
    // Releasing the failed completion cannot refresh the immutable source image.
    let error = materializer
        .expert_read_sequence(IMAGE, &[&f.metadata[0]])
        .unwrap_err();
    assert!(error.to_string().contains("stale source"));
    assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
}

#[test]
fn owner_unwind_drains_and_reuses_real_host_worker_without_cuda_claims() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sequence = f.sequence(&materializer);
        sequence.lookahead((0, 0), Some((0, 1)), false);
        assert_eq!(sequence.pipeline().budget().reserved_bytes, EXPERT_BYTES);
        panic!("controlled owner unwind after read submission");
    }));
    assert!(result.is_err());
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_ok()
    );
    let stats = materializer.expert_read_pipeline_stats();
    assert_eq!(stats.wasted_reads, 1);
    assert_eq!(stats.unknown_bytes, 0);
}

#[test]
fn escaped_host_artifact_blocks_admission_until_all_payload_aliases_drop() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
    sequence.track_artifacts(&value).unwrap();
    let PreparedLinearStorage::NumericFp8(artifact) = value.gate().storage() else {
        panic!()
    };
    let alias = Arc::clone(artifact);
    drop(value);
    drop(sequence);
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    drop(alias);
    // Host allocation retirement is not an Unknown GPU completion proof.
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_ok()
    );
}

#[test]
fn overflow_and_stack_options_fail_before_any_checkpoint_io() {
    let mut materializer = StateDictMaterializer::new(128).unwrap();
    for value in [
        ExpertReadPipelineOptions {
            max_metadata_bytes: u64::MAX,
            ..options(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            worker_stack_bytes: 0,
            ..options(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            worker_stack_bytes: usize::MAX,
            ..options(ExpertReadSchedule::Blocking)
        },
    ] {
        assert!(materializer.configure_expert_read_pipeline(value).is_err());
        assert_eq!(materializer.expert_read_pipeline_options(), None);
    }
    assert_eq!(
        materializer.checkpoint_read_counters(),
        CheckpointReadCounters::default()
    );
}

#[test]
fn every_projection_by_value_artifact_and_payload_clone_blocks_replacement_and_drop_admission() {
    for projection in 0..3 {
        let f = Fixture::new(2);
        let materializer = f.materializer(ExpertReadSchedule::Lookahead);
        let mut sequence = f.sequence(&materializer);
        let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
        sequence.track_artifacts(&value).unwrap();
        let linear = [value.gate(), value.up(), value.down()][projection];
        let artifact = linear.numeric_fp8().unwrap().clone();
        let payload = artifact.validated_payload().clone();
        assert_eq!(
            artifact.weight_bytes().as_ptr(),
            payload.weight_bytes().as_ptr()
        );
        drop(value);
        let before = materializer.checkpoint_read_counters();
        assert!(sequence.demand(0, 1).is_err());
        assert_eq!(sequence.artifact_bytes, EXPERT_BYTES);
        assert_eq!(sequence.pipeline().budget().reserved_bytes, EXPERT_BYTES);
        assert_eq!(materializer.checkpoint_read_counters(), before);
        drop(artifact);
        assert!(sequence.demand(0, 1).is_err());
        let join = sequence.pipeline().join_handoff();
        drop(sequence);
        assert_eq!(join.budget().reserved_bytes, 0);
        assert_eq!(join.budget().worker_stack.reserved_bytes, STACK);
        assert!(
            materializer
                .expert_read_sequence(IMAGE, &[&f.metadata[1]])
                .is_err()
        );
        drop(payload);
        let next = materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[1]])
            .unwrap()
            .unwrap();
        assert_eq!(next.join, join);
        drop(next);
        assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
        assert_eq!(materializer.expert_read_pipeline_stats().worker_reuses, 1);
    }
}

#[test]
fn payload_liveness_tracks_each_original_raw_arc_without_retaining_allocations() {
    use ferrule_common::numeric_fp8::{
        ImmutableValidatedNumericFp8Payload, NumericFp8Layout, NumericFp8ScaleType,
    };
    for retain_weight in [false, true] {
        let weight: Arc<[u8]> = Arc::from([0x38; 4]);
        let scale: Arc<[u8]> = Arc::from(1.0f32.to_le_bytes());
        let payload = ImmutableValidatedNumericFp8Payload::new(
            NumericFp8Layout {
                n: 2,
                k: 2,
                row_origin: 0,
                column_origin: 0,
                scale_type: NumericFp8ScaleType::F32,
            },
            Arc::clone(&weight),
            Arc::clone(&scale),
        )
        .unwrap();
        let live = payload.allocation_liveness();
        let alias = if retain_weight {
            Arc::clone(&weight)
        } else {
            Arc::clone(&scale)
        };
        drop(weight);
        drop(scale);
        let clone = payload.clone();
        drop(payload);
        assert!(live.is_live());
        drop(clone);
        assert!(live.is_live());
        drop(alias);
        assert!(!live.is_live());
    }
}

#[test]
fn all_projection_aliases_must_retire_before_the_next_demand() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let mut sequence = f.sequence(&materializer);
    let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
    sequence.track_artifacts(&value).unwrap();
    let mut aliases = [value.gate(), value.up(), value.down()]
        .map(|linear| Some(linear.numeric_fp8().unwrap().validated_payload().clone()));
    drop(value);
    for alias in &mut aliases {
        assert!(sequence.demand(0, 1).is_err());
        drop(alias.take());
    }
    let next = prepared(&materializer, sequence.demand(0, 1).unwrap());
    sequence.track_artifacts(&next).unwrap();
    drop(next);
    assert!(sequence.release_current().is_ok());
}

#[test]
fn worker_side_sequence_drop_hands_off_join_before_materializer_admission_reopens() {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::mpsc;
    use std::task::{Context, Wake, Waker};
    use std::time::Duration;

    struct Callback(Box<dyn Fn() + Send + Sync>);
    impl Wake for Callback {
        fn wake(self: Arc<Self>) {
            (self.0)();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            (self.0)();
        }
    }
    for panic_after_drop in [false, true] {
        let f = Fixture::new(1);
        let materializer = Arc::new(f.materializer(ExpertReadSchedule::Lookahead));
        let sequence = f.sequence(&materializer);
        let handoff = sequence.pipeline().join_handoff();
        let mut listener = sequence.pipeline().listen();
        let slot = Arc::new(Mutex::new(Some(sequence)));
        let callback_slot = Arc::clone(&slot);
        let callback_materializer = Arc::clone(&materializer);
        let callback_handoff = handoff.clone();
        let (reported, report) = mpsc::sync_channel(1);
        let (released, release) = mpsc::sync_channel(1);
        let release = Mutex::new(release);
        let waker = Waker::from(Arc::new(Callback(Box::new(move || {
            let sequence = callback_slot.lock().unwrap().take();
            if let Some(sequence) = sequence {
                drop(sequence);
                // A worker-side attempt cannot reopen quota by self-joining.
                assert!(
                    callback_materializer
                        .expert_read_sequence(IMAGE, &[])
                        .is_err()
                );
                reported.send(callback_handoff.join()).unwrap();
                release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
                assert!(
                    !panic_after_drop,
                    "controlled panic after worker-side sequence Drop"
                );
            }
        }))));
        assert!(
            Pin::new(&mut listener)
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        slot.lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .lookahead((0, 0), Some((0, 0)), false);
        let pending = report.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(pending.join_pending);
        assert!(!pending.join_failed);
        assert!(pending.unknown.is_empty());
        assert_eq!(pending.budget.reserved_bytes, EXPERT_BYTES);
        assert_eq!(pending.budget.worker_stack.reserved_bytes, STACK);
        let observed = materializer.expert_read_pipeline_observation();
        assert_eq!(observed.status, "join_pending_at_scope_end");
        assert_eq!(observed.stats.sequences_finished, 1);
        assert_eq!(observed.stats.join_pending_sequences, 1);
        assert_eq!(observed.stats.unobserved_terminal_reads, 1);
        assert_eq!(observed.stats.completed_reads, 0);
        let pending_identity = observed.pending_read.unwrap();
        assert_eq!(pending_identity.image_generation, IMAGE);
        assert_eq!((pending_identity.layer, pending_identity.expert), (0, 0));
        assert_eq!(observed.current_charge.packed_bytes, EXPERT_BYTES);
        assert_eq!(observed.current_charge.requested_stack_bytes, STACK as u64);
        assert_eq!(materializer.expert_read_pipeline_observation(), observed);
        released.send(()).unwrap();
        // Admission itself performs the authorized off-worker join of its retained
        // handoff; no direct join/refund is needed from this test's supervisor.
        let next = materializer.expert_read_sequence(IMAGE, &[&f.metadata[0]]);
        if panic_after_drop {
            assert!(next.is_err());
            assert!(handoff.observe_worker().custody_unknown);
            assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
            assert!(
                materializer
                    .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                    .is_err()
            );
        } else {
            drop(next.unwrap());
            assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 2);
        }
        assert_eq!(materializer.expert_read_pipeline_stats().workers_joined, 1);
        assert_eq!(handoff.budget().reserved_bytes, 0);
        assert_eq!(handoff.budget().worker_stack.reserved_bytes, 0);
    }
}
