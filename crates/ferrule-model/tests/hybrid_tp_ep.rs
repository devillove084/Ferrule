//! Model-local TP/EP math and contracts; no cross-owner journal proof.
#![cfg(feature = "cpu-reference")]
#[path = "support/hybrid_tp_ep_fixture.rs"]
mod fixture;
#[path = "hybrid_parallel/model_contract.rs"]
mod support;
use support::*;
#[test]
fn local_state_lifecycle_rejects_foreign_lane_and_accounts_physical_heads() {
    use ferrule_model::decoder::{DecoderSequenceCheckout, DecoderSequenceLifecycle};
    use ferrule_model::transformer::{LayerSegmentPlan, hybrid_parallel::HybridTensorCpuSegment};
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let plan = plan(&resources);
    assert_eq!(plan.local_state_bytes(0).unwrap(), 336);
    assert_eq!(plan.local_state_bytes(1).unwrap(), 336);
    assert_eq!(plan.schema().per_sequence_state_bytes().unwrap(), 672);
    assert!(plan.local_state_bytes(2).is_err());
    let modules = endpoints()
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            HybridTensorCpuSegment::prepare(
                &resources,
                LayerSegmentPlan::new(4, 0..4, true, true).unwrap(),
                plan.clone(),
                lane,
                Box::new(endpoint),
                32,
                1 << 20,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut first = modules[0].lifecycle();
    let mut second = modules[1].lifecycle();
    let state = first.create().unwrap();
    let other = second.create().unwrap();
    let checkout = DecoderSequenceCheckout::new(tx(1), 0, 0);
    let copy = first.checkout(checkout, &state).unwrap();
    assert_eq!(state, copy);
    assert!(first.checkout(checkout, &other).is_err());
    let error = first.try_release(other).unwrap_err();
    let (_, other) = error.into_parts();
    second.try_release(other).unwrap();
    first.try_release(copy).unwrap();
    first.try_release(state).unwrap();
}

#[path = "hybrid_parallel/rank_image.rs"]
mod rank_image;
#[path = "hybrid_parallel/review.rs"]
mod review;

#[test]
fn mismatched_tokens_reject_before_local_state_kv_or_expert_writes() {
    use ferrule_model::{decoder::*, transformer::*};
    use std::sync::{Arc, atomic::AtomicBool};
    let fixture = Arc::new(fixture::Fixture::new());
    let members = [rank(7), rank(19)];
    let endpoints = endpoints_for(&members);
    let control = endpoints[0].control();
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    let threads = endpoints
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            let fixture = Arc::clone(&fixture);
            std::thread::spawn(move || {
                let resources = fixture.resources();
                let p = parallel(&resources, 0..4, members.to_vec(), 1, 0);
                let mut tensor = Tensor::new(&resources, p, lane, endpoint);
                let mut work = tensor.enter(1, &[1 + lane as u32]);
                let before = work.states.clone();
                let kv_before = (0..2)
                    .map(|layer| work.view.history(layer, 0, 0, 1, 8).unwrap())
                    .collect::<Vec<_>>();
                let error = tensor.execute(&mut work, SegmentInput::Tokens).unwrap_err();
                assert!(
                    error.to_string().contains("disagree") || error.to_string().contains("aborted")
                );
                assert_eq!(work.states, before);
                assert!(tensor.experts.as_ref().is_none_or(|ep| ep.calls.is_empty()));
                for (layer, before) in kv_before.iter().enumerate() {
                    assert_eq!(&work.view.history(layer, 0, 0, 1, 8).unwrap(), before);
                }
                assert!(tensor.module.is_failed());
                assert_eq!(tensor.local.backend.capacity().active_transactions, 1);
                tensor.rollback_local(work);
                tensor.release();
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(control.is_failed());
}
