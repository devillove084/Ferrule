use super::*;

#[test]
fn sequential_layers_reuse_one_real_worker_with_monotonic_ids_and_counters() {
    let f = Fixture::layers(4, 2);
    let mut runs = Vec::new();
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let materializer = f.materializer(schedule);
        let mut handoff = None;
        let mut observation = None;
        let mut operations = Vec::new();
        let mut payloads = Vec::new();
        let mut previous = materializer.expert_read_pipeline_stats();
        for invocation in 0..3 {
            for layer in 0..4 {
                let selected = [&f.metadata[layer * 2], &f.metadata[layer * 2 + 1]];
                let mut sequence = materializer
                    .expert_read_sequence(IMAGE, &selected)
                    .unwrap()
                    .unwrap();
                if let Some(first) = &handoff {
                    assert_eq!(first, &sequence.join);
                } else {
                    handoff = Some(sequence.join.clone());
                    observation = sequence.stats_sink.clone();
                }
                for expert in 0..2 {
                    let read = sequence.demand(layer, expert).unwrap();
                    operations.push(read.ticket.operation().get());
                    let value = prepared(&materializer, read);
                    sequence.track_artifacts(&value).unwrap();
                    payloads.push(bytes(&value));
                    drop(value);
                    sequence.lookahead((layer, expert), (expert == 0).then_some((layer, 1)), false);
                }
                drop(sequence);
                let observed = materializer.expert_read_pipeline_observation();
                let delta = observed.stats.checked_delta(previous).unwrap();
                assert_eq!(
                    delta.workers_created,
                    u64::from(invocation == 0 && layer == 0)
                );
                assert_eq!(
                    delta.worker_reuses,
                    u64::from(invocation != 0 || layer != 0)
                );
                assert_eq!(delta.workers_joined, 0);
                assert_eq!(delta.sequences_started, 1);
                assert_eq!(delta.sequences_finished, 1);
                assert_eq!(delta.verified_reads, 2);
                assert_eq!(delta.reader_interval_calls, 12);
                assert_eq!(delta.reader_interval_bytes, 2 * EXPERT_BYTES);
                assert_eq!(delta.unobserved_reader_intervals, 0);
                assert_eq!(observed.stats.workers_created, 1);
                assert_eq!(
                    observed.stats.worker_reuses,
                    (invocation * 4 + layer) as u64
                );
                assert!(observed.worker_alive);
                assert!(!observed.worker_join_pending);
                assert!(!observed.worker_unknown);
                assert_eq!(observed.status, "idle_retained");
                assert_eq!(observed.current_charge.packed_bytes, 0);
                assert_eq!(observed.current_charge.artifact_bytes, 0);
                assert_eq!(observed.current_charge.requested_stack_bytes, STACK as u64);
                assert!(observed.stats.peak_host_charged_bytes <= options(schedule).max_host_bytes);
                previous = observed.stats;
            }
        }
        assert!(operations.windows(2).all(|ids| ids[0] < ids[1]));
        let counts = materializer.checkpoint_read_counters();
        assert_eq!(counts.open_calls, 1);
        assert_eq!(counts.source_checks, 2 * 24); // one-shot pre/post on EVERY read
        assert_eq!(counts.read_calls, 6 * 24);
        assert_eq!(counts.bytes_read, EXPERT_BYTES * 24);
        let view = observation.unwrap().lock().unwrap().clone();
        let handoff = handoff.unwrap();
        drop(materializer);
        let ended = view.snapshot();
        assert_eq!(ended.stats.workers_created, 1);
        assert_eq!(ended.stats.worker_reuses, 11);
        assert_eq!(ended.stats.workers_joined, 1);
        assert!(!ended.worker_alive);
        assert!(!ended.worker_join_pending);
        assert_eq!(ended.current_charge.host_charged_bytes, 0);
        assert!(handoff.observe_worker().joined);
        assert_eq!(
            handoff.observe_budget().budget.worker_stack.reserved_bytes,
            0
        );
        runs.push((
            payloads,
            counts.read_calls,
            counts.bytes_read,
            previous.peak_host_charged_bytes,
        ));
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn owner_rejects_image_options_reader_identity_and_limits_drift_without_new_worker() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let first = f.sequence(&materializer);
    let reader = first.reader.clone();
    let owner_state = Arc::clone(&first.admission.as_ref().unwrap().0.state);
    let handoff = first.join.clone();
    drop(first);
    let original = options(ExpertReadSchedule::Blocking);
    let selected = [&f.metadata[0]];
    let changed = [
        ExpertReadPipelineOptions {
            schedule: ExpertReadSchedule::Lookahead,
            ..original
        },
        ExpertReadPipelineOptions {
            max_host_bytes: original.max_host_bytes + 1,
            ..original
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: original.max_metadata_bytes + 1,
            ..original
        },
        ExpertReadPipelineOptions {
            worker_stack_bytes: original.worker_stack_bytes / 2,
            ..original
        },
    ];
    for options in changed {
        let admission = ExpertReadAdmission::take(&owner_state).unwrap();
        assert!(
            ExpertReadSequence::for_owner(options, IMAGE, &selected, reader.clone(), admission)
                .is_err()
        );
    }
    for other in [
        CheckpointTensorReader::new(128),
        CheckpointTensorReader::with_max_shards(128, 1).unwrap(),
        CheckpointTensorReader::new(256),
    ] {
        let admission = ExpertReadAdmission::take(&owner_state).unwrap();
        assert!(
            ExpertReadSequence::for_owner(original, IMAGE, &selected, other, admission).is_err()
        );
    }
    assert!(
        materializer
            .expert_read_sequence(IMAGE + 1, &selected)
            .is_err()
    );
    let next = f.sequence(&materializer);
    assert_eq!(next.join, handoff);
    drop(next);
    let stats = materializer.expert_read_pipeline_stats();
    assert_eq!(stats.workers_created, 1);
    assert_eq!(stats.worker_reuses, 1);
    assert_eq!(stats.workers_joined, 0);
    assert_eq!(stats.submitted_reads, 0);
}

#[test]
fn changed_source_snapshot_or_stale_original_cannot_refresh_the_retained_owner() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    drop(f.sequence(&materializer));
    f.replace();
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap_err()
            .to_string()
            .contains("stale source")
    );
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for parameter in f.bound.parameters() {
        let weight = parameter.weight().slice();
        let scale = parameter.scale().unwrap().slice();
        mapper
            .insert(
                weight.name.clone(),
                NameMapping::weight(parameter.path().clone()),
            )
            .unwrap();
        mapper
            .insert(
                scale.name.clone(),
                NameMapping::scale(parameter.path().clone()),
            )
            .unwrap();
        slices.extend([weight.clone(), scale.clone()]);
    }
    let rebound = StateDictBinder::new(f.bound.schema(), &mapper)
        .bind_slices(slices)
        .unwrap();
    let metadata = ExpertMetadata::new(
        0,
        0,
        rebound.parameters().to_vec().try_into().unwrap(),
        None,
    )
    .unwrap();
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&metadata])
            .unwrap_err()
            .to_string()
            .contains("source snapshot changed")
    );
    let observed = materializer.expert_read_pipeline_observation();
    assert_eq!(observed.stats.workers_created, 1);
    assert_eq!(observed.stats.worker_reuses, 0);
    assert_eq!(observed.stats.workers_joined, 0);
    assert!(observed.worker_alive);
    assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
}

#[test]
fn poisoned_idle_owner_observation_neither_repairs_nor_replaces_the_worker() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let sequence = f.sequence(&materializer);
    let handoff = sequence.join.clone();
    let owner_state = Arc::clone(&sequence.admission.as_ref().unwrap().0.state);
    drop(sequence);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = owner_state.lock().unwrap();
            panic!("controlled owner ledger poison");
        }))
        .is_err()
    );
    let observed = materializer.expert_read_pipeline_observation();
    assert_eq!(observed.status, "unknown_ledger_poisoned");
    assert!(observed.worker_unknown);
    assert_eq!(observed.current_charge.unknown_stack_bytes, STACK as u64);
    for _ in 0..3 {
        assert_eq!(materializer.expert_read_pipeline_observation(), observed);
        assert!(
            materializer
                .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                .is_err()
        );
        assert!(owner_state.is_poisoned());
    }
    assert_eq!(observed.stats.workers_created, 1);
    assert_eq!(observed.stats.workers_joined, 0);
    drop(owner_state);
    drop(materializer);
    assert!(handoff.observe_worker().joined);
}

#[cfg(unix)]
#[test]
fn same_inode_restored_mtime_cannot_reuse_a_stale_prepared_image() {
    use std::io::Write;
    use std::os::unix::fs::MetadataExt;
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let mut sequence = f.sequence(&materializer);
    sequence.demand(0, 0).unwrap();
    let handoff = sequence.join.clone();
    drop(sequence);
    let before = std::fs::metadata(&f.path).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&f.path)
        .unwrap();
    file.write_all(&[0x39]).unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(before.modified().unwrap()))
        .unwrap();
    let after = file.metadata().unwrap();
    assert_eq!(before.ino(), after.ino());
    assert_eq!(before.len(), after.len());
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    assert_ne!(
        (before.ctime(), before.ctime_nsec()),
        (after.ctime(), after.ctime_nsec())
    );
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
    assert_eq!(materializer.expert_read_pipeline_stats().worker_reuses, 0);
    assert_eq!(
        materializer.checkpoint_read_counters().bytes_read,
        EXPERT_BYTES
    );
    assert!(handoff.observe_worker().alive);
}

#[test]
fn poisoned_observation_stays_passive_and_cannot_restart_owner_counters() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let sequence = f.sequence(&materializer);
    let sink = sequence.stats_sink.as_ref().unwrap().clone();
    let handoff = sequence.join.clone();
    drop(sequence);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = sink.lock().unwrap();
            panic!("controlled observation poison");
        }))
        .is_err()
    );
    let observed = materializer.expert_read_pipeline_observation();
    assert_eq!(observed.status, "unknown_observation_poisoned");
    assert!(observed.worker_unknown);
    for _ in 0..3 {
        assert_eq!(materializer.expert_read_pipeline_observation(), observed);
        assert!(
            materializer
                .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                .is_err()
        );
        assert!(sink.is_poisoned());
    }
    assert_eq!(observed.stats.workers_created, 1);
    assert_eq!(observed.stats.worker_reuses, 0);
    drop(materializer);
    assert!(handoff.observe_worker().joined);
}

#[test]
fn passive_observer_arc_never_inherits_worker_teardown_from_materializer_drop() {
    let f = Fixture::new(1);
    for active in [false, true] {
        let materializer = f.materializer(ExpertReadSchedule::Blocking);
        let mut sequence = f.sequence(&materializer);
        sequence.demand(0, 0).unwrap();
        let handoff = sequence.join.clone();
        let view = sequence
            .stats_sink
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .clone();
        // Model the temporary upgrade performed by a concurrent passive getter.
        let observer = view.admission_state.as_ref().unwrap().upgrade().unwrap();
        let mut sequence = Some(sequence);
        if !active {
            drop(sequence.take());
        }
        drop(materializer);
        assert_eq!(handoff.observe_worker().joined, !active);
        drop(sequence);
        assert!(handoff.observe_worker().joined);
        assert!(!handoff.observe_worker().alive);
        assert!(observer.lock().unwrap().worker.is_none());
        let observed = handoff.observe_budget();
        assert_eq!(observed.budget.requested_host_reservation_bytes(), 0);
        drop(observer);
        assert_eq!(handoff.observe_budget(), observed);
    }
}
