use super::*;
use crate::transformer::hybrid_parallel::mixed_decoded_arena::{FixedSlot, SlotClass};
#[test]
fn decoded_claim_compares_full_admission_even_when_grants_match() {
    let f = Fixture::new();
    let (_, native) = claim_inputs(&f);
    let program = f
        .plan
        .clone()
        .with_decoded_weight_cache(&f.resources, 16, 64 << 20, 1 << 30, 128 << 20)
        .unwrap();
    let (_, enabled) = program
        .cuda_claim_inputs(&f.resources, 16, 0, 1 << 20, &native.attention_shapes)
        .unwrap();
    // Hold every derived field identical; only admission identity varies.
    assert_ne!(native, enabled);
    assert!(native.validate_for(&enabled).is_err());
    assert_eq!(enabled, enabled.clone());
    for field in 0..13 {
        let mut other = enabled.clone();
        let a = other.program.decoded_cache.as_mut().unwrap();
        match field {
            0 => a.device_capacity -= 1,
            1 => a.owner_margins += 1,
            2 => a.physical_pages += 1,
            3 => a.budget.capacity_grants += 1,
            4 => a.budget.pinned_grants += 1,
            5 => a.budget.routed_working_set_grants += 1,
            6 => a.budget.routed_entries += 1,
            7 => Arc::make_mut(&mut a.arena).unused_bytes += 1,
            8 => Arc::make_mut(&mut a.arena).routed_slot_bytes += 1,
            9 => Arc::make_mut(&mut a.arena).slot_padding_bytes += 1,
            10 => Arc::make_mut(&mut a.arena).fixed.push(FixedSlot {
                descriptor: 0,
                offset_words: 0,
                elements: 32,
                slot_bytes: 64,
            }),
            11 => Arc::make_mut(&mut a.arena).classes.push(SlotClass {
                offset_words: 0,
                elements: 32,
                slot_bytes: 64,
                count: 1,
                layer_minimum: 1,
            }),
            _ => other.program.decoded_cache = None,
        }
        assert_eq!(enabled.estimate, other.estimate);
        assert_eq!(enabled.projection_limits, other.projection_limits);
        assert_ne!(enabled, other);
        assert!(enabled.validate_for(&other).is_err());
    }
}
