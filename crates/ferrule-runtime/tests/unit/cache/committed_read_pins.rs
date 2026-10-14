// Included inside the private pin index, not a standalone integration target.
use super::*;
use ferrule_common::execution::{KvElementType, KvLayoutSchemaIdentity, KvPlaneDescriptor};

fn source() -> CommittedKvSource {
    CommittedKvSource {
        owner: 7,
        state_slot: StateSlot::new(0),
        generation: 3,
        incarnation: 11,
        schema: KvLayoutSchemaIdentity::new(
            vec![KvPlaneDescriptor::new("kv", 2, 1, KvElementType::F32)],
            4,
            128,
        ),
        committed_tokens: 3,
        positions: 0..3,
        pages: vec![KvPageId(0)],
    }
}

#[test]
fn independent_pins_keep_page_and_slot_guards_until_last_release() {
    let mut pins = CommittedReadPins::new();
    let s = source();
    let first = pins.register(s.owner, &s).unwrap();
    let second = pins.register(s.owner, &s).unwrap();
    pins.release(s.owner, &first).unwrap();
    assert_eq!(pins.check_pages(&s.pages), Err(HostKvError::ReaderActive));
    assert_eq!(
        pins.check_slot(s.state_slot),
        Err(HostKvError::ReaderActive)
    );
    pins.check_pages(&[KvPageId(1)]).unwrap();
    pins.check_slot(StateSlot::new(1)).unwrap();
    assert_eq!(
        pins.release(s.owner, &first),
        Err(HostKvError::UnknownLease)
    );
    pins.release(s.owner, &second).unwrap();
    pins.check_pages(&s.pages).unwrap();
    pins.check_slot(s.state_slot).unwrap();
}

#[test]
fn foreign_and_changed_registration_rejections_leave_retryable_pin() {
    let mut pins = CommittedReadPins::new();
    let s = source();
    let mut lease = pins.register(s.owner, &s).unwrap();
    assert_eq!(
        pins.release(s.owner + 1, &lease),
        Err(HostKvError::ForeignOwner)
    );
    lease.source.incarnation += 1;
    assert_eq!(
        pins.release(s.owner, &lease),
        Err(HostKvError::UnknownLease)
    );
    assert_eq!(pins.readers.len(), 1);
    lease.source = s.clone();
    pins.release(s.owner, &lease).unwrap();
}

#[test]
fn reader_identity_never_reuses_released_registration() {
    let mut pins = CommittedReadPins::new();
    let s = source();
    let old = pins.register(s.owner, &s).unwrap();
    pins.release(s.owner, &old).unwrap();
    let new = pins.register(s.owner, &s).unwrap();
    assert_ne!(old.id, new.id);
    assert_eq!(pins.release(s.owner, &old), Err(HostKvError::UnknownLease));
    pins.validate(s.owner, &new).unwrap();
    assert_eq!(pins.check_pages(&s.pages), Err(HostKvError::ReaderActive));
}

#[test]
fn drop_is_not_ack_and_exhaustion_does_not_mutate_index() {
    let mut pins = CommittedReadPins::new();
    let s = source();
    let lease = pins.register(s.owner, &s).unwrap();
    drop(lease);
    assert_eq!(
        pins.check_slot(s.state_slot),
        Err(HostKvError::ReaderActive)
    );
    pins.next_id = u64::MAX;
    for _ in 0..2 {
        assert_eq!(
            pins.register(s.owner, &s).unwrap_err(),
            HostKvError::Overflow
        );
        assert_eq!(pins.next_id, u64::MAX);
        assert_eq!(pins.readers.len(), 1);
    }
}
