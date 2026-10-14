//! CPU logical authority regression and compile-only real append/attend fixture.
use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPlaneDescriptor, StateSlot};
use ferrule_runtime::{KvPageManager, KvReservationCommit};
#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static PLANES: [KvPlaneDescriptor; 2] = [
            KvPlaneDescriptor::new("standard_gqa.key", 2, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 2, 1, KvElementType::F32),
        ];
        &PLANES
    }
    fn page_size(&self) -> usize {
        4
    }
    fn max_sequence_len(&self) -> usize {
        16
    }
}
#[test]
fn actual_manager_reservation_is_required_and_does_not_add_another_pin() {
    use ferrule_common::execution::ExecutionTransactionId;
    let mut manager = KvPageManager::new(Box::new(Schema), 8);
    let mut foreign = KvPageManager::new(Box::new(Schema), 8);
    let slot = StateSlot::new(0);
    for m in [&mut manager, &mut foreign] {
        m.alloc_sequence(slot, 0).unwrap();
        let r = m.reserve(slot, 0, 3).unwrap();
        let p = m
            .prepare_commit(vec![KvReservationCommit::new(r, 3)])
            .unwrap();
        let retired = m.publish_commit(p);
        m.confirm_page_retirement(retired).unwrap();
    }
    let r = manager.reserve(slot, 0, 1).unwrap();
    let id = ExecutionTransactionId::new(2).unwrap();
    assert!(foreign.derive_append_prefix(&r, id, 0).is_err());
    let prefix = manager.derive_append_prefix(&r, id, 0).unwrap();
    prefix.validate().unwrap();
    assert_eq!(prefix.witness().source().committed_tokens, 3);
    assert_eq!(prefix.witness().positions(), 0..4);
    drop(prefix);
    let retired = manager.abort_reservations(vec![r]).unwrap();
    manager.confirm_page_retirement(retired).unwrap();
    assert_eq!(manager.block_table(slot).unwrap().committed_tokens(), 3);
}
#[cfg(feature = "cuda")]
#[path = "host_kv/append_bridge_cuda.rs"]
mod cuda;

#[path = "host_kv/runner_fixture.rs"]
mod runner;
