use super::*;

#[cfg(unix)]
#[test]
fn same_inode_restored_mtime_fails_session_before_and_after_open() {
    for after_open in [false, true] {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::with_max_shards(32, 1).unwrap();
        let original = fixture.plan.clone();
        let session = after_open.then(|| reader.verified_read_session(&original).unwrap());
        metadata_support::overwrite_restoring_mtime(&fixture.path);
        assert!(original.validate_source_identity().is_err());
        assert!(reader.verified_read_session(&original).is_err());
        if let Some(session) = session {
            assert!(session.read().is_err());
        }
        let refreshed = CheckpointReadPlan::new(
            original.extents().iter().cloned(),
            Arc::<[CheckpointSourceFileIdentity]>::from([CheckpointSourceFileIdentity::capture(
                &fixture.path,
            )
            .unwrap()]),
        )
        .unwrap();
        assert_ne!(original.source_files(), refreshed.source_files());
        assert_eq!(
            reader
                .verified_read_session(&refreshed)
                .unwrap()
                .read()
                .unwrap(),
            [10, 11, 12, 13, 1, 2, 3, 4]
        );
        assert_eq!(reader.read_counters().peak_handles, 1);
    }
}

#[cfg(unix)]
#[test]
fn same_inode_restored_mtime_pipeline_rejects_before_and_during_read() {
    for phase in [
        TestPhase::Dequeued,
        TestPhase::SessionOpened,
        TestPhase::PayloadRead,
    ] {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::new(32);
        let path = fixture.path.clone();
        let mut pipeline = fixture.pipeline(
            &reader,
            Some(hook(move |current, _| {
                if current == phase {
                    metadata_support::overwrite_restoring_mtime(&path);
                }
            })),
        );
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        let receipt = ready(&pipeline, &ticket);
        assert!(receipt.source_verified_at.is_none());
        assert_eq!(receipt.payload_bytes, 0);
        if phase == TestPhase::Dequeued {
            assert_eq!(reader.read_counters().read_calls, 0);
        } else {
            assert_eq!(receipt.io_bytes, Some(8));
        }
        let completion = pipeline.take(&ticket).unwrap();
        assert!(matches!(completion.outcome(), ReadOutcome::Failed(_)));
        assert_eq!(pipeline.budget().reserved_bytes, 8);
        drop(completion);
        assert_eq!(pipeline.budget().reserved_bytes, 0);
        assert!(pipeline.shutdown().unknown.is_empty());
    }
}
