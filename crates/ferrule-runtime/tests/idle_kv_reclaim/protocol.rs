use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPlaneDescriptor, StateSlot};
use ferrule_runtime::{IdleKvReclaimConfig, KvPageManager, KvReservationCommit};
#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static P: [KvPlaneDescriptor; 2] = [
            KvPlaneDescriptor::new("standard_gqa.key", 1, 2, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 1, 2, KvElementType::F32),
        ];
        &P
    }
    fn page_size(&self) -> usize {
        4
    }
    fn max_sequence_len(&self) -> usize {
        16
    }
}
#[test]
fn admission_counts_original_shadow_staging_grants_and_full_host_frames() {
    let c = IdleKvReclaimConfig {
        staging_page_limit: 2,
        device_bytes: 8192,
        host_bytes: 1024,
        pinned_bytes: 1024,
    };
    let plan = c.plan(Schema.planes(), 4, 8).unwrap();
    assert_eq!(plan.page_payload_bytes, 64);
    assert_eq!(plan.page_grant_bytes, 512);
    assert_eq!(plan.physical_grant_bytes, 4096);
    assert_eq!(plan.staging_grant_bytes, 512);
    assert_eq!(plan.device_bytes, 4608);
    assert_eq!(plan.host_bytes, 512);
    for c in [
        IdleKvReclaimConfig {
            device_bytes: 4607,
            ..c
        },
        IdleKvReclaimConfig {
            host_bytes: 511,
            ..c
        },
        IdleKvReclaimConfig {
            pinned_bytes: 511,
            ..c
        },
        IdleKvReclaimConfig {
            staging_page_limit: 0,
            ..c
        },
    ] {
        assert!(c.plan(Schema.planes(), 4, 8).is_err());
    }
}
#[test]
fn same_manager_suspension_retains_ids_frontier_and_rejects_foreign_replay() {
    let mut manager = KvPageManager::new(Box::new(Schema), 8);
    let mut foreign = KvPageManager::new(Box::new(Schema), 8);
    let slot = StateSlot::new(0);
    manager.alloc_sequence(slot, 7).unwrap();
    let r = manager.reserve(slot, 7, 3).unwrap();
    let p = manager
        .prepare_commit(vec![KvReservationCommit::new(r, 3)])
        .unwrap();
    let retired = manager.publish_commit(p);
    manager.confirm_page_retirement(retired).unwrap();
    let before = manager.block_table(slot).unwrap().clone();
    let source = manager.committed_kv_source(slot, 7, 0..3).unwrap();
    let authority = manager.preempt_sequence(slot).unwrap();
    let replay = authority.clone();
    assert_eq!(authority.pages(), before.pages());
    assert_eq!(manager.allocated_pages(), before.pages().len());
    assert!(manager.validate_committed_source(&source).is_err());
    assert!(foreign.restore_sequence(slot, replay.clone()).is_err());
    manager.restore_sequence(slot, authority).unwrap();
    assert_eq!(manager.block_table(slot).unwrap().pages(), before.pages());
    assert_eq!(manager.block_table(slot).unwrap().committed_tokens(), 3);
    assert!(manager.restore_sequence(StateSlot::new(1), replay).is_err());
    assert!(manager.validate_committed_source(&source).is_err());
}

#[test]
fn snapshot_pins_prevent_shared_page_eviction_and_released_snapshot_cannot_replay() {
    let mut m = KvPageManager::new(Box::new(Schema), 8);
    let slot = StateSlot::new(0);
    m.alloc_sequence(slot, 0).unwrap();
    let r = m.reserve(slot, 0, 3).unwrap();
    let p = m
        .prepare_commit(vec![KvReservationCommit::new(r, 3)])
        .unwrap();
    let retirement = m.publish_commit(p);
    m.confirm_page_retirement(retirement).unwrap();
    let snapshot = m.capture_prefix_snapshot(slot, 3).unwrap();
    let prepared = m
        .prepare_fork_prefix_snapshot(snapshot, StateSlot::new(1), 0, 3)
        .unwrap();
    let suspension = m.preempt_sequence(slot).unwrap();
    assert!(
        suspension.evicted_pages().is_empty(),
        "immutable snapshot still owns these bytes"
    );
    let retirement = m.release_prefix_snapshot(snapshot).unwrap();
    m.confirm_page_retirement(retirement).unwrap();
    assert!(m.publish_fork_prefix_snapshot(prepared).is_err());
    assert!(m.release_prefix_snapshot(snapshot).is_err());
    m.restore_sequence(slot, suspension).unwrap();
    assert_eq!(m.block_table(slot).unwrap().committed_tokens(), 3);
}

#[test]
fn unknown_fixture_assembly_reserves_context_and_shadow_slots_without_relaxing_admission() {
    use ferrule_runtime::engine::{ResidentKvPageAccounting, plan_resident_kv_pages};
    use ferrule_runtime::{ResidentSchedulerConfig, ResidentTopKDriverConfig};

    #[derive(Debug)]
    struct FixtureSchema;
    impl KvLayoutSchema for FixtureSchema {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            Schema.planes()
        }
        fn page_size(&self) -> usize {
            2
        }
        fn max_sequence_len(&self) -> usize {
            16
        }
    }
    let scheduler = ResidentSchedulerConfig {
        max_active_sequences: 2,
        max_decode_batch: 2,
        prefill_chunk_size: 2,
        max_batch_tokens: 4,
        ..Default::default()
    };
    let driver = ResidentTopKDriverConfig {
        ctx_size: 16,
        enable_native_proposals: false,
        ..Default::default()
    };
    let schema = FixtureSchema;
    let page_bytes = schema.checked_page_bytes().unwrap() as u64;
    let capped = plan_resident_kv_pages(
        &schema,
        ResidentKvPageAccounting::PageLimit(32),
        scheduler,
        driver,
    )
    .unwrap();
    assert_eq!(capped.full_capacity_pages, 16);
    assert_eq!(
        capped.configured_pages, 16,
        "PageLimit is not additional shadow capacity"
    );

    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes,
        budget_bytes: None,
    };
    let exact = plan_resident_kv_pages(&schema, accounting, scheduler, driver).unwrap();
    assert_eq!(exact.full_capacity_pages, 16);
    assert_eq!(exact.configured_pages, 32);
    assert_eq!(exact.configured_bytes, Some(32 * page_bytes));
    assert!(
        plan_resident_kv_pages(
            &schema,
            ResidentKvPageAccounting::TransactionCapacity {
                page_bytes,
                budget_bytes: Some(32 * page_bytes - 1)
            },
            scheduler,
            driver,
        )
        .is_err(),
        "fixture repair must not weaken the production byte-budget check"
    );
}
