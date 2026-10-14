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
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_ok()
    );
}

#[test]
fn owner_unwind_joins_and_drains_real_host_worker_without_cuda_claims() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sequence = f.sequence(&materializer);
        sequence.lookahead((0, 0), Some((0, 1)), false);
        assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
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
fn escaped_host_artifact_does_not_refund_a_second_materializer_scope() {
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
    // A later alias drop does not retroactively prove correct scope retirement.
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
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
