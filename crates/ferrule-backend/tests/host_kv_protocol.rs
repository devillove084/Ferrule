//! Driver-free tests of the production completion/budget protocol.
mod cpu {
    pub use ferrule_backend::cpu::{
        PagedKvBatch, PagedKvPrepare, StreamingF32GqaConfig, StreamingF32GqaPartial,
        StreamingF32GqaRequest, streaming_f32_gqa_partial,
    };
}
pub use ferrule_backend::cuda_allocation_grant_bytes;
#[path = "streaming_attention/append_partials.rs"]
mod append_composition;
#[path = "host_kv/append_dma.rs"]
mod append_dma;
#[path = "../src/cuda/operators/kv/transaction/append_stream.rs"]
mod append_stream;
#[allow(dead_code)]
#[path = "../src/cuda/operators/kv/transaction/plan.rs"]
mod plan;
#[path = "../src/cuda/operators/kv/transaction/prefix_read.rs"]
mod prefix_read;
#[path = "../src/cuda/operators/kv/read_protocol.rs"]
mod reaper;
#[path = "host_kv/transaction_prefix.rs"]
mod transaction_prefix;

use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};

#[test]
fn admission_is_atomic_in_all_six_domains_and_drop_is_not_ack() {
    let c = cpu::StreamingF32GqaConfig {
        rows: 1,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        block_tokens: 5,
        softmax_scale: 0.5,
        sliding_window: None,
    };
    let credits = reaper::committed_gqa_credits(c, 2).unwrap();
    let budget = PlacementBudget::new(credits);
    let held = budget.reserve_resources(credits).unwrap();
    assert_eq!(budget.in_use(), credits);
    for one in [
        PlacementCredits {
            pageable_host_bytes: 1,
            ..Default::default()
        },
        PlacementCredits {
            pinned_host_bytes: 1,
            ..Default::default()
        },
        PlacementCredits {
            staging_bytes: 1,
            ..Default::default()
        },
        PlacementCredits {
            staging_slots: 1,
            ..Default::default()
        },
        PlacementCredits {
            transfers: 1,
            ..Default::default()
        },
        PlacementCredits {
            consumers: 1,
            ..Default::default()
        },
    ] {
        assert!(budget.reserve_resources(one).is_err());
        assert_eq!(budget.in_use(), credits);
    }
    held.release();
    assert_eq!(budget.in_use(), PlacementCredits::default());
    let abandoned = budget.reserve_resources(credits).unwrap();
    drop(abandoned);
    assert_eq!(budget.in_use(), credits);
}

#[test]
fn late_acks_and_partial_reaps_cannot_clear_unknown_or_refund_budget() {
    let credits = PlacementCredits {
        pageable_host_bytes: 32,
        pinned_host_bytes: 32,
        staging_bytes: 256,
        staging_slots: 1,
        transfers: 2,
        consumers: 1,
    };
    let budget = PlacementBudget::new(credits);
    let charge = budget.reserve_resources(credits).unwrap();
    let mut custody = reaper::Custody::default();
    custody.submit_consumer().unwrap();
    custody.quarantine();
    for _ in 0..2 {
        custody.dma_ack();
        custody.consumer_ack();
        custody.cancel();
        assert!(custody.unknown());
        assert!(!custody.quiescent());
        assert!(custody.submit_dma().is_err());
        assert!(custody.submit_consumer().is_err());
        assert!(
            custody
                .complete_reap(Ok(()), Err(reaper::error("compute")))
                .is_err()
        );
        assert!(
            custody
                .complete_reap(Err(reaper::error("DMA")), Ok(()))
                .is_err()
        );
        assert_eq!(budget.in_use(), credits);
        assert!(budget.clone().reserve_resources(credits).is_err());
    }
    custody.complete_reap(Ok(()), Ok(())).unwrap();
    assert!(custody.quiescent());
    assert!(custody.canceled());
    assert!(!custody.unknown());
    assert!(custody.submit_dma().is_err());
    // Reap proves quiescence; the owner's explicit finish still owns the charge.
    assert_eq!(budget.in_use(), credits);
    charge.release();
    assert_eq!(budget.in_use(), PlacementCredits::default());
}
