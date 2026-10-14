use super::*;
use std::cell::Cell;
#[test]
fn native_unknown_is_sticky_without_retry_fence() {
    let poisoned = Cell::new(false);
    assert!(
        initialization_completion(
            9,
            || poisoned.set(true),
            || panic!("later fence cannot recover Unknown")
        )
        .is_err()
    );
    assert!(poisoned.get());
}
#[test]
fn ordinary_setup_failure_still_requires_completion() {
    let poisoned = Cell::new(false);
    let fenced = Cell::new(false);
    initialization_completion(
        7,
        || poisoned.set(true),
        || {
            fenced.set(true);
            Ok(())
        },
    )
    .unwrap();
    assert!(fenced.get());
    assert!(!poisoned.get());
    assert!(status(7).is_err());
}
#[test]
fn failed_rust_completion_quarantines_successful_native_setup() {
    let poisoned = Cell::new(false);
    assert!(
        initialization_completion(0, || poisoned.set(true), || Err(invalid("lost fence"))).is_err()
    );
    assert!(poisoned.get());
}
#[test]
fn native_plan_preserves_every_checked_region() {
    let p = Bf16AttentionPlan::new(
        &[Bf16AttentionGroup {
            row_start: 0,
            rows: 5,
            sequence: 0,
            tokens: 129,
        }],
        16,
        2,
        256,
        64 << 20,
        16 << 20,
    )
    .unwrap();
    let n = NativePlan::from(&p);
    assert_eq!(n.bytes, p.storage_bytes() as u64);
    assert_eq!(n.offsets, p.offsets().map(|n| n as u64));
    assert_eq!(
        (n.heads, n.kv_heads, n.dim, n.max_rows, n.max_tokens),
        (16, 2, 256, 5, 129)
    );
}

#[path = "review.rs"]
mod review;
