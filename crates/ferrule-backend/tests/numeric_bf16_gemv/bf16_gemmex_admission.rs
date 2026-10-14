use super::*;
#[test]
fn owner_ledger_keeps_graph_or_unknown_charge_without_global_exclusion() {
    let mut l = Ledger::default();
    let c = Charge {
        storage_grant: 69_206_016,
        external: 16_777_216,
    };
    l.reserve((10, ProviderFamily::GemmEx), 0, c).unwrap();
    assert!(l.reserve((10, ProviderFamily::GemmEx), 0, c).is_err());
    // Owner10 can be graph-retained/Unknown without excluding owner11 or GPU1.
    l.reserve((11, ProviderFamily::GemmEx), 0, c).unwrap();
    l.reserve((12, ProviderFamily::GemmEx), 1, c).unwrap();
    assert_eq!(l.devices[&0], c.add(c).unwrap());
    assert_eq!(l.devices[&1], c);
    l.release((11, ProviderFamily::GemmEx));
    l.release((12, ProviderFamily::GemmEx));
    assert_eq!(l.devices[&0], c);
    assert!(!l.devices.contains_key(&1));
    assert!(l.owners.contains_key(&(10, ProviderFamily::GemmEx)));
    l.release((10, ProviderFamily::GemmEx));
    assert!(l.owners.is_empty() && l.devices.is_empty());
}
#[test]
fn owner_ledger_rejects_overflow_atomically() {
    let mut l = Ledger::default();
    let c = Charge {
        storage_grant: usize::MAX - 16,
        external: 16,
    };
    l.reserve((1, ProviderFamily::GemmEx), 0, c).unwrap();
    assert!(
        l.reserve(
            (2, ProviderFamily::GemmEx),
            0,
            Charge {
                storage_grant: 1,
                external: 0
            }
        )
        .is_err()
    );
    assert_eq!(l.owners.len(), 1);
    assert_eq!(l.devices[&0], c);
    assert!(
        Charge {
            storage_grant: usize::MAX,
            external: 1
        }
        .total()
        .is_err()
    );
}
#[test]
fn first_native_completion_failure_cannot_be_recovered_by_second_fence() {
    let poisoned = Cell::new(false);
    let fences = Cell::new(0);
    // Even if the next fence would pass, don't let it erase native status9.
    assert!(
        warmup_completion(
            COMPLETION_UNKNOWN,
            || poisoned.set(true),
            || {
                fences.set(fences.get() + 1);
                Ok(())
            }
        )
        .is_err()
    );
    assert!(poisoned.get());
    assert_eq!(fences.get(), 0);
}
#[test]
fn clean_native_preflight_and_pending_second_fence_are_distinct() {
    let poisoned = Cell::new(false);
    let fences = Cell::new(0);
    warmup_completion(
        1,
        || poisoned.set(true),
        || {
            fences.set(1);
            Ok(())
        },
    )
    .unwrap();
    assert!(!poisoned.get());
    assert_eq!(fences.get(), 1);
    assert!(
        warmup_completion(
            0,
            || poisoned.set(true),
            || Err(invalid("driver fence failed"))
        )
        .is_err()
    );
    assert!(poisoned.get());
}
