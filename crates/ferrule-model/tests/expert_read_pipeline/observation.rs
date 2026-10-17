use super::*;
use crate::resource_observability::{
    PhysicalResourceObservation, ResourceEvidence, ResourceRelease, resource_summary_json,
};
use crate::transformer::ExpertReadPipelineStats;

#[test]
fn configured_mode_is_not_a_positive_worker_read_qualification() {
    let default = StateDictMaterializer::new(128).unwrap();
    let disabled = default.expert_read_pipeline_observation();
    assert!(!disabled.enabled);
    assert_eq!(disabled.requested_mode, None);
    assert_eq!(disabled.effective_mode, None);
    assert_eq!(disabled.status, "disabled");
    // Removed upload policies must not expand the historical default schema.
    let charge = serde_json::to_value(disabled.current_charge).unwrap();
    for field in [
        "upload_reserved_packed_bytes",
        "pinned_upload_bytes",
        "upload_ticket_metadata_bytes",
        "upload_policy_headroom_bytes",
        "unknown_upload_bytes",
    ] {
        assert!(charge.get(field).is_none());
    }
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let materializer = f.materializer(schedule);
        let configured = materializer.expert_read_pipeline_observation();
        assert!(configured.enabled);
        assert_eq!(configured.requested_mode, Some(schedule));
        assert_eq!(configured.effective_mode, None);
        assert!(!configured.worker_read_observed);
        assert_eq!(configured.current_charge.host_charged_bytes, 0);
        let sequence = f.sequence(&materializer);
        let active = materializer.expert_read_pipeline_observation();
        assert_eq!(active.effective_mode, Some(schedule));
        assert!(!active.worker_read_observed);
        assert_eq!(active.stats.submitted_reads, 0);
        assert_eq!(active.current_charge.metadata_bytes, METADATA);
        assert_eq!(active.current_charge.requested_stack_bytes, STACK as u64);
        drop(sequence);
        let closed = materializer.expert_read_pipeline_observation();
        assert_eq!(closed.stats.sequences_finished, 1);
        assert_eq!(
            closed.current_charge.host_charged_bytes,
            METADATA + STACK as u64
        );
        assert!(!closed.worker_read_observed);
    }
}

#[test]
fn receipt_identity_clock_and_all_charge_partitions_come_from_real_demand() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(2);
        let materializer = f.materializer(schedule);
        let mut sequence = f.sequence(&materializer);
        let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
        sequence.track_artifacts(&value).unwrap();
        let before = materializer.checkpoint_read_counters();
        let budget = sequence.pipeline().budget();
        let observed = materializer.expert_read_pipeline_observation();
        for _ in 0..4 {
            assert_eq!(materializer.expert_read_pipeline_observation(), observed);
        }
        assert_eq!(sequence.pipeline().budget(), budget);
        assert_eq!(materializer.checkpoint_read_counters(), before);
        assert!(observed.worker_read_observed);
        assert_eq!(observed.stats.submitted_reads, 1);
        assert_eq!(observed.stats.completed_reads, 1);
        assert_eq!(observed.stats.verified_reads, 1);
        assert_eq!(observed.stats.reader_interval_calls, 6);
        assert_eq!(observed.stats.reader_interval_bytes, EXPERT_BYTES);
        assert_eq!(observed.stats.proved_read_bytes, EXPERT_BYTES);
        assert_eq!(observed.stats.demand_wait_calls, 1);
        let receipt = observed.last_receipt.unwrap();
        assert_eq!(observed.current_read, Some(receipt.identity));
        assert_eq!(receipt.identity.image_generation, IMAGE);
        assert_eq!(receipt.identity.layer, 0);
        assert_eq!(receipt.identity.expert, 0);
        assert_eq!(
            receipt.identity.operation,
            sequence.current.as_ref().unwrap().ticket.operation().get()
        );
        let ordered = [
            receipt.submitted_ns,
            receipt.worker_started_ns.unwrap(),
            receipt.io_started_ns.unwrap(),
            receipt.io_finished_ns.unwrap(),
            receipt.source_verified_ns.unwrap(),
            receipt.completed_ns.unwrap(),
            receipt.taken_ns.unwrap(),
        ];
        assert!(ordered.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(
            observed.stats.io_duration_ns,
            receipt.io_finished_ns.unwrap() - receipt.io_started_ns.unwrap()
        );
        let charge = observed.current_charge;
        assert_eq!(charge.packed_bytes, EXPERT_BYTES);
        assert_eq!(charge.artifact_bytes, EXPERT_BYTES);
        assert_eq!(charge.metadata_bytes, METADATA);
        assert_eq!(charge.requested_stack_bytes, STACK as u64);
        assert_eq!(charge.payload_bytes, 2 * EXPERT_BYTES);
        assert!(charge.host_charged_bytes <= options(schedule).max_host_bytes);
        assert_eq!(observed.stats.peak_packed_bytes, EXPERT_BYTES);
        assert_eq!(observed.stats.peak_artifact_bytes, EXPERT_BYTES);
        assert_eq!(observed.stats.peak_metadata_bytes, METADATA);
        assert_eq!(observed.stats.peak_requested_stack_bytes, STACK as u64);
        drop(value);
        drop(sequence);
        let closed = materializer.expert_read_pipeline_observation();
        assert_eq!(closed.current_read, None);
        assert_eq!(
            closed.current_charge.host_charged_bytes,
            METADATA + STACK as u64
        );
        assert!(closed.worker_read_observed);
        assert_eq!(closed.stats.sequences_finished, 1);
    }
}

#[test]
fn closed_artifact_payload_alias_charge_is_sampled_without_refunding_or_io() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
    sequence.track_artifacts(&value).unwrap();
    let artifact = value.down().numeric_fp8().unwrap().clone();
    let payload = artifact.validated_payload().clone();
    drop(value);
    drop(sequence);
    let before = materializer.checkpoint_read_counters();
    let retained = materializer.expert_read_pipeline_observation();
    assert_eq!(retained.status, "idle_retained");
    assert_eq!(retained.current_charge.packed_bytes, 0);
    assert_eq!(retained.current_charge.requested_stack_bytes, STACK as u64);
    assert_eq!(retained.current_charge.artifact_bytes, EXPERT_BYTES);
    assert_eq!(retained.current_charge.metadata_bytes, METADATA);
    assert_eq!(materializer.expert_read_pipeline_observation(), retained);
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    drop(artifact);
    assert_eq!(materializer.expert_read_pipeline_observation(), retained);
    drop(payload);
    // Passive sampling cannot release the Weak witnesses' backing allocations.
    assert_eq!(materializer.expert_read_pipeline_observation(), retained);
    drop(f.sequence(&materializer));
    let ended = materializer.expert_read_pipeline_observation();
    assert_eq!(
        ended.current_charge.host_charged_bytes,
        METADATA + STACK as u64
    );
    assert_eq!(ended.stats.workers_created, 1);
    assert_eq!(ended.stats.worker_reuses, 1);
    assert_eq!(materializer.checkpoint_read_counters(), before);
}

#[test]
fn wasted_completed_io_reports_actual_reader_interval_not_assumed_equal_schedules() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    sequence.lookahead((0, 0), Some((0, 1)), false);
    let ticket = sequence.pending.as_ref().unwrap().ticket.as_ref().unwrap();
    sequence.pipeline().wait(ticket).unwrap(); // ensure actual IO, then abandon the scope.
    drop(sequence);
    let observed = materializer.expert_read_pipeline_observation();
    assert!(observed.worker_read_observed);
    assert_eq!(observed.stats.wasted_reads, 1);
    assert_eq!(observed.stats.cancelled_reads, 1);
    assert_eq!(observed.stats.wasted_read_bytes, EXPERT_BYTES);
    assert_eq!(observed.stats.wasted_reader_interval_calls, 6);
    assert_eq!(observed.stats.wasted_reader_interval_bytes, EXPERT_BYTES);
    assert_eq!(observed.stats.prefetched_reads_taken, 0);
    assert_eq!(observed.last_receipt.unwrap().identity.expert, 1);
    assert!(observed.last_receipt.unwrap().cancellation_requested);
    assert_eq!(
        observed.current_charge.host_charged_bytes,
        METADATA + STACK as u64
    );
}

#[test]
fn failed_scope_finalizes_counters_but_keeps_caller_held_completion_charge() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let mut sequence = f.sequence(&materializer);
    f.replace();
    let error = sequence.demand(0, 0).unwrap_err();
    drop(sequence);
    let retained = materializer.expert_read_pipeline_observation();
    assert_eq!(retained.stats.failed_reads, 1);
    assert_eq!(retained.stats.demand_failures, 1);
    assert_eq!(retained.stats.sequences_finished, 1);
    assert_eq!(retained.stats.completed_reads, 1);
    assert_eq!(retained.stats.unknown_reads, 0);
    assert_eq!(retained.current_charge.packed_bytes, EXPERT_BYTES);
    assert_eq!(retained.last_receipt.unwrap().outcome, "failed");
    assert_eq!(retained.last_receipt.unwrap().io_bytes, None);
    assert!(!retained.worker_read_observed);
    drop(error);
    assert_eq!(
        materializer
            .expert_read_pipeline_observation()
            .current_charge
            .host_charged_bytes,
        METADATA + STACK as u64
    );
    assert_eq!(materializer.expert_read_pipeline_stats(), retained.stats);
}

#[test]
fn counter_delta_preserves_lifetime_peaks_and_marks_resets_missing() {
    let before = ExpertReadPipelineStats {
        submitted_reads: 8,
        peak_host_charged_bytes: 64,
        ..Default::default()
    };
    let after = ExpertReadPipelineStats {
        submitted_reads: 10,
        peak_host_charged_bytes: 80,
        ..Default::default()
    };
    let delta = after.checked_delta(before).unwrap();
    assert_eq!(delta.submitted_reads, 2);
    assert_eq!(delta.peak_host_charged_bytes, 80);
    assert!(before.checked_delta(after).is_none());
}

#[test]
fn physical_json_merges_host_observation_without_certifying_release_from_zero() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    drop(f.sequence(&materializer));
    let observation = materializer.expert_read_pipeline_observation();
    assert_eq!(
        observation.current_charge.host_charged_bytes,
        METADATA + STACK as u64
    );
    let physical = PhysicalResourceObservation {
        expert_read_pipeline: ResourceEvidence::Observed(observation),
        ..Default::default()
    };
    let merged = PhysicalResourceObservation::default()
        .snapshot_with_live(&physical)
        .unwrap();
    assert_eq!(merged.expert_read_pipeline, physical.expert_read_pipeline);
    assert_eq!(
        merged.release,
        ResourceEvidence::<ResourceRelease>::Unobserved
    );
    let json: serde_json::Value =
        serde_json::from_str(&resource_summary_json(&merged).unwrap()).unwrap();
    let direct = serde_json::to_value(&merged).unwrap();
    assert_eq!(json, direct);
    assert_eq!(direct["expert_read_pipeline"]["value"]["enabled"], true);
    assert_eq!(
        direct["expert_read_pipeline"]["value"]["effective_mode"],
        "blocking"
    );
    assert_eq!(direct["release"]["state"], "unobserved");
}

#[test]
fn real_worker_notification_panic_finalizes_unknown_without_refund_or_healing() {
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
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let sequence = f.sequence(&materializer);
    let mut listener = sequence.pipeline().listen();
    let slot = Arc::new(Mutex::new(Some(sequence)));
    let callback_slot = Arc::clone(&slot);
    let (sent, received) = mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(Callback(Box::new(move || {
        // The first read is already Complete on this worker. Its real callback
        // queues the next accepted read, then panics before that worker can
        // execute it. The core quarantines the queued ticket; no fake Unknown.
        callback_slot
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .lookahead((0, 0), Some((0, 1)), false);
        sent.send(()).unwrap();
        panic!("controlled real-worker notification failure");
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
    received.recv_timeout(Duration::from_secs(5)).unwrap();
    drop(slot.lock().unwrap().take()); // joins before sampling the final Unknown.
    let observed = materializer.expert_read_pipeline_observation();
    assert_eq!(observed.status, "quarantined");
    assert_eq!(observed.stats.sequences_finished, 1);
    assert_eq!(observed.stats.submitted_reads, 2);
    assert_eq!(observed.stats.completed_reads, 1);
    assert_eq!(observed.stats.unknown_reads, 1);
    assert_eq!(observed.stats.unknown_bytes, EXPERT_BYTES);
    assert_eq!(observed.current_charge.unknown_packed_bytes, EXPERT_BYTES);
    assert_eq!(observed.current_charge.packed_bytes, EXPERT_BYTES);
    assert_eq!(observed.current_charge.metadata_bytes, METADATA);
    assert_eq!(observed.current_charge.requested_stack_bytes, 0);
    let receipt = observed.last_receipt.unwrap();
    assert_eq!(receipt.outcome, "unknown");
    assert_eq!(receipt.identity.image_generation, IMAGE);
    assert_eq!(receipt.identity.expert, 1);
    assert_eq!(receipt.io_bytes, None);
    for _ in 0..3 {
        assert_eq!(materializer.expert_read_pipeline_observation(), observed);
    }
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    assert_eq!(materializer.expert_read_pipeline_observation(), observed);
}

#[test]
fn abandoned_lookahead_reports_extra_io_while_kept_weights_and_worker_bounds_match_blocking() {
    let f = Fixture::new(2);
    let mut runs = Vec::new();
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let materializer = f.materializer(schedule);
        let mut sequence = f.sequence(&materializer);
        let worker_options = sequence.pipeline().options();
        let payload_limit = sequence.options.limits().unwrap();
        let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
        sequence.track_artifacts(&value).unwrap();
        let kept = bytes(&value);
        drop(value);
        sequence.lookahead((0, 0), Some((0, 1)), false);
        if let Some(pending) = &sequence.pending {
            sequence
                .pipeline()
                .wait(pending.ticket.as_ref().unwrap())
                .unwrap();
        }
        drop(sequence);
        let observed = materializer.expert_read_pipeline_observation();
        assert!(observed.worker_read_observed);
        assert!(observed.stats.peak_host_charged_bytes <= options(schedule).max_host_bytes);
        runs.push((kept, worker_options, payload_limit, observed.stats));
    }
    assert_eq!(
        (&runs[0].0, runs[0].1, runs[0].2),
        (&runs[1].0, runs[1].1, runs[1].2)
    );
    assert_eq!(runs[0].3.reader_interval_calls, 6);
    assert_eq!(runs[0].3.reader_interval_bytes, EXPERT_BYTES);
    assert_eq!(runs[0].3.wasted_reads, 0);
    assert_eq!(runs[1].3.reader_interval_calls, 12);
    assert_eq!(runs[1].3.reader_interval_bytes, 2 * EXPERT_BYTES);
    assert_eq!(runs[1].3.wasted_reader_interval_calls, 6);
    assert_eq!(runs[1].3.wasted_reader_interval_bytes, EXPERT_BYTES);
}
