//! Metadata-only planning tests: no GPU allocation or tensor payloads.
use super::*;
use crate::transformer::hybrid_parallel::mixed_decoded::DecodedIdentity;
use std::sync::Arc;
const MIB: usize = 1 << 20;
fn d(id: u32, bytes: usize, layer: Option<usize>) -> DecodedDescriptor {
    DecodedDescriptor {
        identity: Arc::new(DecodedIdentity {
            owner: id,
            geometry: [bytes / 2, 1],
            parts: vec![],
        }),
        grants: bytes,
        routed_layer: layer,
        operands: vec![(id as usize, 0)],
    }
}
#[test]
fn three_gib_plan_has_disjoint_fixed_and_at_least_577_complete_experts() {
    let mut ds = vec![d(0, 1340 * MIB, None)];
    for layer in 0..40 {
        for expert in 0..32 {
            let id = 1 + (layer * 32 + expert) as u32 * 2;
            ds.push(d(id, 2 * MIB, Some(layer)));
            ds.push(d(id + 1, MIB, Some(layer)));
        }
    }
    let budget = HybridMixedDecodedCacheBudget::new(3072 * MIB, &ds).unwrap();
    let p = DecodedArenaPlan::new(budget, &ds).unwrap();
    assert_eq!(p, DecodedArenaPlan::new(budget, &ds).unwrap());
    assert_eq!(p.fixed.len(), 1);
    assert_eq!(p.fixed[0].offset_words, 0);
    let mut end = 1340 * MIB;
    for c in &p.classes {
        assert_eq!(c.offset_words * 2, end);
        assert_eq!(c.layer_minimum, 32);
        assert_eq!(c.count, if c.slot_bytes == MIB { 578 } else { 577 });
        end += c.count * c.slot_bytes;
    }
    assert_eq!(end + p.unused_bytes, p.arena_grants);
    assert_eq!(p.routed_slot_bytes + p.unused_bytes, 1732 * MIB);
    assert_eq!(p.slot_padding_bytes, 0);
    assert_eq!(p.unused_bytes, 0);
}
#[test]
fn exact_class_minima_cover_all_layer_shapes_not_just_largest_layer_bytes() {
    let ds = vec![
        d(0, 60 * MIB, None),
        d(1, 4 * MIB, Some(0)),
        d(2, 2 * MIB, Some(1)),
    ];
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB, &ds).unwrap();
    assert!(DecodedArenaPlan::new(budget, &ds).is_err());
}
#[test]
fn malformed_extent_and_non_grant_capacity_are_rejected() {
    let mut ds = vec![d(1, MIB, Some(0))];
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB, &ds).unwrap();
    ds[0].identity = Arc::new(DecodedIdentity {
        owner: 1,
        geometry: [usize::MAX, 2],
        parts: vec![],
    });
    assert!(DecodedArenaPlan::new(budget, &ds).is_err());
    let ds = vec![d(1, MIB, Some(0))];
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB + 1, &ds).unwrap();
    assert!(DecodedArenaPlan::new(budget, &ds).is_err());
}

#[test]
fn inventory_capped_pair_redistributes_all_remaining_slots_to_down() {
    let mut ds = vec![d(0, 2 * MIB, Some(0))];
    for i in 1..=80 {
        ds.push(d(i, MIB, Some(i as usize)));
    }
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB, &ds).unwrap();
    let p = DecodedArenaPlan::new(budget, &ds).unwrap();
    assert_eq!(
        p.classes
            .iter()
            .map(|c| (c.slot_bytes, c.count, c.layer_minimum))
            .collect::<Vec<_>>(),
        vec![(MIB, 62, 1), (2 * MIB, 1, 1)]
    );
    assert_eq!(p.unused_bytes, 0);
}
