//! Driver-free tests of the exact completion receipt used by the CUDA owner.
#[path = "../src/transformer/standard/cuda/upload_proof.rs"]
mod upload_proof;
use ferrule_common::execution::ExecutionTransactionId;
use upload_proof::{MixedServingScope, SubmissionGeneration, TailCompletion};

fn scope(image: u64, transaction: u64) -> Option<MixedServingScope> {
    Some(MixedServingScope {
        image_generation: image,
        transaction: ExecutionTransactionId::new(transaction).unwrap(),
    })
}
fn complete() -> TailCompletion {
    TailCompletion {
        operation_succeeded: true,
        boundary_drained: true,
        compute_synchronized: true,
        owner_healthy: true,
    }
}
#[test]
fn no_op_is_not_a_completed_consumer_proof() {
    let generation = SubmissionGeneration::new(scope(1, 1));
    assert!(!generation.covers_tail_upload(None, scope(1, 1), complete()));
    let proof = generation.prove::<()>(Ok(()), Ok(())).unwrap();
    assert!(generation.covers_tail_upload(Some(proof), scope(1, 1), complete()));
}
#[test]
fn failed_compute_or_upload_cannot_issue_a_receipt() {
    let generation = SubmissionGeneration::new(scope(1, 1));
    assert!(generation.prove(Err("compute"), Ok(())).is_err());
    assert!(generation.prove(Ok(()), Err("upload")).is_err());
    assert!(matches!(
        generation.prove(Err("compute"), Err("upload")),
        Err("compute")
    ));
}
#[test]
fn receipt_cannot_cross_nested_submission_image_or_transaction() {
    let first = SubmissionGeneration::new(scope(1, 1));
    let nested = SubmissionGeneration::new(scope(1, 1));
    let proof = first.prove::<()>(Ok(()), Ok(())).unwrap();
    assert!(!nested.covers_tail_upload(Some(proof), scope(1, 1), complete()));
    for current in [None, scope(2, 1), scope(1, 2)] {
        let proof = first.prove::<()>(Ok(()), Ok(())).unwrap();
        assert!(!first.covers_tail_upload(Some(proof), current, complete()));
    }
}
#[test]
fn excluded_owner_cannot_reuse_even_an_executed_proof() {
    // F32, single-owner, diagnostics, preparation and inactive/failed Mixed
    // invocations all supply None, not a reusable serving scope.
    let generation = SubmissionGeneration::new(None);
    let proof = generation.prove::<()>(Ok(()), Ok(())).unwrap();
    assert!(!generation.covers_tail_upload(Some(proof), None, complete()));
    let proof = generation.prove::<()>(Ok(()), Ok(())).unwrap();
    assert!(!generation.covers_tail_upload(Some(proof), scope(1, 1), complete()));
}
#[test]
fn every_failure_and_unknown_keeps_the_tail_upload_wait() {
    let generation = SubmissionGeneration::new(scope(1, 1));
    for tail in [
        TailCompletion {
            operation_succeeded: false,
            ..complete()
        },
        TailCompletion {
            boundary_drained: false,
            ..complete()
        },
        TailCompletion {
            compute_synchronized: false,
            ..complete()
        },
        TailCompletion {
            owner_healthy: false,
            ..complete()
        },
    ] {
        let proof = generation.prove::<()>(Ok(()), Ok(())).unwrap();
        assert!(!generation.covers_tail_upload(Some(proof), scope(1, 1), tail));
    }
}

#[test]
fn synchronous_wiring_keeps_compute_cleanup_and_error_fallback() {
    let source = include_str!("../src/transformer/standard/cuda.rs");
    let body = source
        .split("fn synchronous<T>")
        .nth(1)
        .unwrap()
        .split("fn f32<'a>")
        .next()
        .unwrap();
    let positions = [
        "self.consumer_completion(&generation)",
        "self.boundary.drain()",
        "self.ops.sync_stream()",
        "generation.covers_tail_upload(",
        "let upload = if reuse_upload",
        "self.ops.sync_upload_stream()",
        "let failures = [event, drain, compute, upload]",
        "self.recurrent_scratch.complete()",
        "self.finish_cache_operation()?",
    ]
    .map(|needle| body.find(needle).unwrap());
    assert!(positions.windows(2).all(|p| p[0] < p[1]));
    let eligibility = source
        .split("fn mixed_upload_proof_scope")
        .nth(1)
        .unwrap()
        .split("fn synchronous<T>")
        .next()
        .unwrap();
    for required in [
        "self.diagnostic.is_enabled()",
        "self.needs_quarantine()",
        "self.root_submission.is_some()",
        "self.bounded_experts.is_some()",
        "self.hybrid_rank.is_some()",
        "self.bindings.tensor.is_some()",
        "Bf16ModelMixedF32Accumulate",
        "upload_proof_transaction()?",
    ] {
        assert!(
            eligibility.contains(required),
            "missing exclusion {required}"
        );
    }
    let mixed = include_str!("../src/transformer/standard/cuda/hybrid_rank/mixed_layer.rs");
    let active = mixed
        .split("fn upload_proof_transaction")
        .nth(1)
        .unwrap()
        .split("pub(crate) fn diagnostic_layout")
        .next()
        .unwrap();
    for required in [
        "channel.failed",
        "channel.completion_unknown()",
        "self.device.needs_quarantine()",
        "members().len() <= 1",
        "channel.active.map",
    ] {
        assert!(
            active.contains(required),
            "missing invocation check {required}"
        );
    }
}
