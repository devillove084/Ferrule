use super::*;
use crate::checkpoint::{CheckpointReadExtent, CheckpointSourceFileIdentity};
fn d(bytes: usize, layer: Option<usize>) -> DecodedDescriptor {
    DecodedDescriptor {
        identity: Arc::new(DecodedIdentity {
            owner: 0,
            geometry: [128, 128],
            parts: vec![],
        }),
        grants: bytes,
        routed_layer: layer,
        operands: vec![(0, 0)],
    }
}
#[test]
fn actual_historical_ledger_has_three_gib_not_unbounded_headroom() {
    let b =
        HybridMixedDecodedCacheBudget::new(3 << 30, &[d(1340 << 20, None), d(768 << 20, Some(0))])
            .unwrap();
    let total = b.require_capacity(19_362_869_248, 0, 22 << 30).unwrap();
    assert_eq!(total, 22_584_094_720);
    assert_eq!((22 << 30) - total, 1_038_225_408);
    assert!(b.require_capacity(19_362_869_248, 0, total - 1).is_err());
    assert!(b.require_capacity(usize::MAX, 1, usize::MAX).is_err());
    assert!(
        b.require_capacity(19_362_869_248, 1_038_225_409, 22 << 30)
            .is_err()
    );
}
#[test]
fn descriptor_grants_not_payloads_and_largest_borrowed_layer_are_admitted() {
    let ds = [
        d(256, None),
        d(256, Some(0)),
        d(512, Some(0)),
        d(512, Some(1)),
    ];
    assert!(HybridMixedDecodedCacheBudget::new(1023, &ds).is_err());
    let b = HybridMixedDecodedCacheBudget::new(1024, &ds).unwrap();
    assert_eq!(b.pinned_grants, 256);
    assert_eq!(b.routed_working_set_grants, 768);
    assert_eq!(b.routed_entries, 3);
    assert!(HybridMixedDecodedCacheBudget::new(0, &[]).is_err());
    assert!(
        HybridMixedDecodedCacheBudget::new(usize::MAX, &[d(usize::MAX, None), d(1, None)]).is_err()
    );
}
fn source(name: &str, incarnation: &str, offset: u64) -> DecodedSource {
    let path = std::path::PathBuf::from(name);
    let identity = CheckpointSourceFileIdentity::for_test(path.clone(), incarnation.into(), 1024);
    DecodedSource(
        CheckpointReadPlan::new(
            [CheckpointReadExtent::new(path, offset, 16).unwrap()],
            Arc::<[_]>::from([identity]),
        )
        .unwrap(),
    )
}
#[test]
fn keys_bind_both_sources_generation_owner_geometry_and_order_without_io() {
    let before = CheckpointSourceFileIdentity::counters();
    let part = DecodedPart {
        slot: 1,
        segment: 0,
        parameter: ParameterId::new(1),
        role: TensorRole::RoutedExpertGate,
        weight: source("w", "w-v1", 0),
        scale: source("s", "s-v1", 0),
        rows: (0, 128),
        columns: (0, 128),
    };
    let mut up = part.clone();
    up.role = TensorRole::RoutedExpertUp;
    up.slot = 2;
    let key = DecodedKey {
        generation: 7,
        identity: Arc::new(DecodedIdentity {
            owner: 0,
            geometry: [256, 128],
            parts: vec![part, up],
        }),
    };
    let mut variants = Vec::new();
    let mut k = key.clone();
    k.generation += 1;
    variants.push(k);
    for change in 0..7 {
        let mut k = key.clone();
        let id = Arc::make_mut(&mut k.identity);
        match change {
            0 => id.owner += 1,
            1 => id.geometry[1] *= 2,
            2 => id.parts.swap(0, 1),
            3 => id.parts[0].weight = source("w", "w-v2", 0),
            4 => id.parts[0].scale = source("s", "s-v2", 0),
            5 => id.parts[0].scale = source("s", "s-v1", 16),
            _ => id.parts[0].columns = (128, 256),
        }
        variants.push(k);
    }
    for variant in variants {
        assert_ne!(key, variant);
        assert_ne!(key.cmp(&variant), Ordering::Equal);
    }
    assert_eq!(key, key.clone());
    let after = CheckpointSourceFileIdentity::counters();
    assert_eq!(before.capture_calls, after.capture_calls);
    assert_eq!(before.path_metadata_calls, after.path_metadata_calls);
}

#[test]
fn admitted_cache_rejects_changed_physical_pages_or_ledger() {
    let ds = [d(32768, None)];
    let budget = HybridMixedDecodedCacheBudget::new(64 << 20, &ds).unwrap();
    let arena =
        Arc::new(super::super::mixed_decoded_arena::DecodedArenaPlan::new(budget, &ds).unwrap());
    let admission = DecodedCacheAdmission {
        budget,
        arena,
        physical_pages: 64,
        device_capacity: 4096,
        owner_margins: 256,
    };
    assert!(admission.validate(64, 3840).is_ok());
    assert!(admission.validate(64, 3841).is_err());
    assert!(admission.validate(32, 3840).is_err());
    assert!(admission.validate(64, usize::MAX).is_err());
}
