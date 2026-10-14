use super::*;
use crate::decoder::{DecoderSequenceState, StandardSequenceLifecycle};

#[test]
fn public_checkout_reexport_preserves_transaction_and_distinct_indices() {
    let transaction = ExecutionTransactionId::new(17).unwrap();
    let request: DecoderSequenceCheckout =
        crate::decoder::DecoderSequenceCheckout::new(transaction, 2, 7);
    assert_eq!(request.transaction(), transaction);
    assert_eq!(request.sequence_index(), 2);
    assert_eq!(request.source_index(), 7);
}

#[test]
fn state_parts_round_trip_without_clone_or_device_capability() {
    #[derive(Debug, PartialEq, Eq)]
    struct Attachment(u32);

    let state = DecoderSequenceState::with_position(7, Attachment(11), vec![13, 17]);
    let core = state.core().clone();
    let topology = state.topology_id();
    let (parts_core, parts_topology, attachment, kv) = state.into_parts();
    let restored = DecoderSequenceState::from_parts(parts_core, parts_topology, attachment, kv);
    assert_eq!(restored.core(), &core);
    assert_eq!(restored.topology_id(), topology);
    assert_eq!(restored.attachment(), &Attachment(11));
    assert_eq!(restored.kv_state(), &[13, 17]);
}

#[test]
fn public_contracts_accept_the_standard_state_lifecycle() {
    type State = DecoderSequenceState<(), ()>;
    let mut lifecycle = StandardSequenceLifecycle::<(), ()>::new();
    let lifecycle: &mut dyn crate::decoder::DecoderSequenceLifecycle<State> = &mut lifecycle;
    let original = lifecycle.create().unwrap();
    let request = DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 3);
    let working = lifecycle.checkout(request, &original).unwrap();
    let fork = lifecycle.logical_fork(&original, 0).unwrap();
    let projected: &dyn crate::decoder::DecoderSequence = &working;
    assert_eq!(projected.topology_id(), original.topology_id());
    assert_eq!(projected.core(), original.core());
    assert_ne!(fork.topology_id(), original.topology_id());
    lifecycle.try_release(working).unwrap();
    lifecycle.try_release(fork).unwrap();
    lifecycle.try_release(original).unwrap();
}
