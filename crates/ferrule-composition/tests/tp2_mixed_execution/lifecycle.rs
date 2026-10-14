//! Bounded self-consistency/lifecycle checks, NOT independent multistep numerical quality.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "lifecycle_recipe.rs"]
mod recipe;
fn sequence(pipeline: &mut PipelineParallelExecutor, prompt: u32, decode: [u32; 3]) -> Vec<f32> {
    let a = SessionId(91);
    let b = SessionId(92);
    let initial = pipeline
        .forward(a, &[prompt], ForwardPhase::Prefill)
        .unwrap()
        .logits
        .values()
        .to_vec();
    pipeline.fork_session(a, b).unwrap();
    let cancel = AtomicBool::new(false);
    let before = pipeline.coordinator().publication_count();
    let mut cancelled_ready = false;
    let result =
        pipeline.forward_observed(a, &[decode[0]], ForwardPhase::Decode, &cancel, |progress| {
            if progress.state == ferrule_runtime::TransactionState::Preparing {
                // This synchronous report follows all ready votes, before decision.
                // pending_ranks tracks post-decision ACKs, not prepare readiness.
                assert_eq!(progress.pending_ranks.len(), 2);
                assert!(!cancelled_ready, "one prepared barrier per forward");
                assert_eq!(progress.publications, before);
                cancelled_ready = true;
                cancel.store(true, Ordering::Release);
            }
        });
    assert!(cancelled_ready, "prepared observer absent: {result:?}");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cancelled before decision")
    );
    assert_eq!(pipeline.coordinator().publication_count(), before);
    assert!(
        !pipeline.is_quarantined(),
        "proven prepared rollback must remain reusable"
    );
    for token in decode {
        let parent = pipeline.forward(a, &[token], ForwardPhase::Decode).unwrap();
        let child = pipeline.forward(b, &[token], ForwardPhase::Decode).unwrap();
        assert_eq!(
            parent.logits, child.logits,
            "fork must preserve rollback and state frontier"
        );
    }
    pipeline.release_session(a).unwrap();
    pipeline.release_session(b).unwrap();
    let recreated = pipeline
        .forward(a, &[prompt], ForwardPhase::Prefill)
        .unwrap();
    assert_eq!(
        initial,
        recreated.logits.values(),
        "same-ID recreation must have fresh state/KV"
    );
    pipeline.release_session(a).unwrap();
    assert_eq!(pipeline.outstanding(), 0);
    assert!(
        pipeline
            .owner_stats()
            .unwrap()
            .iter()
            .all(|o| o.sessions == 0)
    );
    initial
}
#[test]
#[ignore = "GPU owner only: pinned paper recipe replay and native publication-joined capture; CPU canonical comparison still required"]
fn full40_bounded_multistep_fork_recreate_cold_owner() {
    recipe::run();
}

#[test]
#[ignore = "two-GPU owner only: prepared cancellation rollback, multistep/fork and same-ID recreation"]
fn tiny_prepared_cancel_multistep_and_recreation() {
    grant();
    let fixture = super::tiny::base::Fixture::new();
    let mut pipeline = build(
        super::tiny::resources(&fixture),
        PipelineConfig {
            page_size: 2,
            max_pages: 16,
            max_positions: 32,
            max_batch_tokens: 2,
            max_parameter_bytes: 1 << 20,
            ..config()
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sequence(&mut pipeline, 1, [2, 3, 4])
    }));
    pipeline.shutdown().unwrap();
    result.unwrap();
}

#[test]
#[ignore = "sole GPU owner: explicit decoded cache and unchanged six pinned lifecycle recipes"]
fn full40_decoded_cache_bounded_multistep_fork_recreate_cold_owner() {
    recipe::run_with_cache(Some(3 << 30));
}

#[test]
#[ignore = "sole GPU owner: unchanged six recipes, cache-on, no numeric observer or committed tensor readback"]
fn full40_decoded_cache_unobserved_fused_lifecycle() {
    recipe::run_unobserved_with_cache(Some(3 << 30));
}
