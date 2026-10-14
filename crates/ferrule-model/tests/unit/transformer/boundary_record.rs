use super::*;
use ferrule_common::Error;
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, PartialEq)]
enum RecordFailure {
    Producer,
    ImmediateCleanup,
}

impl fmt::Display for RecordFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected {self:?}")
    }
}

impl StdError for RecordFailure {}

fn typed_failure(failure: RecordFailure) -> Error {
    Error::Backend {
        source: Box::new(failure),
    }
}

fn assert_primary(error: &Error, cleanup_failed: bool) {
    let shard = crate::transformer::parallel_transfer::CudaShardError::from_error(error)
        .expect("typed quarantine result");
    assert!(shard.needs_quarantine());
    assert!(!shard.is_quiescent());
    let mut current: &(dyn StdError + 'static) = error;
    let mut found_cleanup = false;
    loop {
        if let Some(Error::Cleanup {
            source, cleanup, ..
        }) = current.downcast_ref::<Error>()
        {
            let Error::Backend { source } = source.as_ref() else {
                panic!("typed primary")
            };
            assert_eq!(
                source.downcast_ref::<RecordFailure>(),
                Some(&RecordFailure::Producer)
            );
            let Error::Backend { source } = cleanup.as_ref() else {
                panic!("typed secondary")
            };
            assert_eq!(
                source.downcast_ref::<RecordFailure>(),
                Some(&RecordFailure::ImmediateCleanup)
            );
            found_cleanup = true;
        }
        if let Some(marker) = current.downcast_ref::<RecordFailure>() {
            assert_eq!(
                marker,
                &RecordFailure::Producer,
                "cleanup never replaces primary"
            );
            assert_eq!(found_cleanup, cleanup_failed);
            break;
        }
        current = current
            .source()
            .expect("first typed producer error survives wrappers");
    }
}

#[test]
#[ignore = "requires CUDA; missing producer event is sticky despite a later successful event proof"]
fn failed_boundary_producer_record_keeps_scope_custody_and_primary_error() {
    for precision in precisions() {
        for cleanup_failed in [false, true] {
            let fixture = Fixture::new();
            let mut provider = fixture.provider(true);
            let expert = prepared(&mut provider);
            let mut owner = fixture.owner(precision, 32768);
            let input = owner.bind_rows(input(1)).unwrap();
            warm(&mut owner, &expert, &input);
            let proofs = owner.consumer_proofs;
            let workspace = owner.numeric_workspace_stats().unwrap();
            let before = owner.expert_cache_stats().unwrap();
            // Same typed result as backend record_event: immediate cleanup
            // success still returns the producer error; failure keeps both.
            owner.boundary.fail_next_producer_record = Some(Error::with_cleanup(
                "record producer event",
                typed_failure(RecordFailure::Producer),
                if cleanup_failed {
                    Err(typed_failure(RecordFailure::ImmediateCleanup))
                } else {
                    Ok(())
                },
            ));
            let error = owner
                .synchronous(|this| {
                    this.swiglu_consume(&expert, &input, None, |this, output| {
                        let failure = this
                            .boundary
                            .download(&this.ops, this.f32(&output)?)
                            .unwrap_err();
                        assert!(this.boundary.needs_quarantine());
                        assert!(this.needs_quarantine(), "sticky before admission cleanup");
                        // Real subsequent event record AND wait succeed. They must
                        // not substitute for the producer proof that was lost.
                        this.ops.record_compute_event()?.synchronize()?;
                        this.ops.sync_upload_stream()?;
                        this.boundary.drain()?;
                        assert!(this.consumer_proof().is_err());
                        assert_eq!(
                            this.consumer_proofs, proofs,
                            "quarantine prevents another logical proof"
                        );
                        assert_eq!(this.expert_artifacts.len(), 3);
                        assert_eq!(this.expert_hold.len(), 5);
                        assert!(this.bindings.expert_admission.is_some());
                        Err::<(), _>(failure)
                    })
                })
                .unwrap_err();
            assert_primary(&error, cleanup_failed);
            assert!(owner.needs_quarantine());
            let frozen = owner.expert_cache_stats().unwrap();
            assert!(frozen.quarantined);
            assert_eq!(frozen.unknown_quarantine_bytes, frozen.charged_bytes());
            assert_eq!(frozen.resident_bytes, before.resident_bytes);
            assert_eq!(frozen.pending_upload_bytes, before.pending_upload_bytes);
            assert!(frozen.scratch_bytes > before.scratch_bytes);
            let after = owner.numeric_workspace_stats().unwrap();
            assert_eq!(after.submissions, workspace.submissions + 3);
            assert_eq!(after.allocations, workspace.allocations);
            let counters = owner.ops.counters();
            assert!(owner.quiesce().is_err());
            assert!(owner.finish_cache_operation().is_err());
            assert!(owner.prepare_expert(&expert).is_err());
            assert!(owner.dense_swiglu(&expert, &input, None).is_err());
            let buffer = input.cuda().unwrap().f32_buffer().unwrap();
            assert!(owner.boundary.download(&owner.ops, buffer).is_err());
            assert!(owner.boundary.upload(&owner.ops, &[0.0; 8]).is_err());
            assert_eq!(
                owner.ops.counters(),
                counters,
                "disabled owner cannot submit or allocate"
            );
            assert_eq!(owner.numeric_workspace_stats().unwrap(), after);
            assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
            assert_eq!(owner.expert_artifacts.len(), 3);
            assert_eq!(owner.expert_hold.len(), 5);
            let held = owner.ops.allocator_metrics().live_requested_bytes;
            let ops = Rc::clone(&owner.ops);
            drop(input);
            drop(owner);
            ops.trim_device_allocator().unwrap();
            assert!(ops.allocator_metrics().live_requested_bytes >= held);
        }
    }
}

#[test]
#[ignore = "requires CUDA; ordinary boundary allocation reject does not poison the owner"]
fn ordinary_boundary_allocation_reject_remains_reusable() {
    let fixture = Fixture::new();
    let mut owner = fixture.owner(NumericFp8Precision::Bf16RneF32Accumulate, 32768);
    owner.ops.failpoints().arm_allocation();
    assert!(
        owner
            .synchronous(|this| this.boundary.upload(&this.ops, &[1.0]))
            .is_err()
    );
    assert!(!owner.boundary.needs_quarantine());
    assert!(!owner.needs_quarantine());
    let output = owner
        .synchronous(|this| this.boundary.upload(&this.ops, &[1.0]))
        .unwrap();
    let values = owner
        .synchronous(|this| this.boundary.download(&this.ops, &output))
        .unwrap();
    assert_eq!(values, vec![1.0]);
    assert!(!owner.needs_quarantine());
}
