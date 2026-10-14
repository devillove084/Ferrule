use super::semantic_tests::raw_peers;
use super::tests::{binding, description};
use super::*;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ForwardMode, ForwardPhase, KvBlockId, KvReservationView,
    KvWriteSlot, LogitsRequest, StateSlot,
};
use ferrule_model::decoder::{DecoderKvPageStatus, GenericDecoderSequenceState};
use ferrule_model::transformer::{StandardTensorCollective, parallel::TensorParallelCollective};
use std::sync::{Arc, atomic::AtomicBool};

fn two_sequences() -> TensorBatchIdentity {
    let batch = ExecutionBatch::new(
        ForwardMode::Decode,
        vec![3, 2],
        vec![1, 1],
        vec![Some(KvWriteSlot::new(15)), Some(KvWriteSlot::new(17))],
        vec![LogitsRequest::Full; 2],
        (0..2)
            .map(|i| {
                ExecutionSequence::new(
                    StateSlot::new(i),
                    ForwardPhase::Decode,
                    i..i + 1,
                    1,
                    2,
                    i..i + 1,
                )
            })
            .collect(),
        vec![KvBlockId::new(7), KvBlockId::new(8)],
    );
    let reservations = (0..2)
        .map(|i| KvReservationView {
            state_slot: StateSlot::new(i + 9),
            execution_state_slot: StateSlot::new(i),
            positions: 1..2,
            newly_allocated: vec![],
            generation: 5,
            execution_generation: 0,
            cow_replacement: None,
        })
        .collect::<Vec<_>>();
    let states = [
        GenericDecoderSequenceState::with_position(1, (), ()),
        GenericDecoderSequenceState::with_position(1, (), ()),
    ];
    let mut capabilities = description().execution_capabilities().unwrap();
    capabilities.max_sequences = 2;
    let packed =
        PackedDecoderBatch::lower(&batch, &reservations, &states, &capabilities, 2, &|_| {
            DecoderKvPageStatus::Resident
        })
        .unwrap();
    TensorBatchIdentity::from_packed(&binding(), &packed, &[11, 12]).unwrap()
}
fn site(shape: TensorCollectiveShape) -> TensorCollectiveSite {
    TensorCollectiveSite {
        site: 0,
        layer: None,
        operator: TensorCollectiveOperator::Protocol {
            version: 2,
            operation: TensorCollectiveSemantic::InputAgreement,
            kind: HostCollectiveKind::AllGatherF32,
            shape,
        },
    }
}

#[test]
fn sealed_exchange_and_memory_preflight_use_full_identity_sequence_cardinality() {
    let identity = two_sequences();
    assert_eq!(identity.sequences(), 2);
    assert_eq!(identity.rows(), 2);
    let shape = TensorCollectiveShape::PerSequence {
        fixed: 1,
        per_sequence: 3,
        per_row: 2,
    };
    let mut peers = raw_peers();
    let control = peers[0].control();
    for peer in &peers {
        control
            .register_sites(peer.owner(), vec![site(shape)])
            .unwrap();
    }
    control
        .begin_sealed(identity.clone(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for peer in &peers {
        control
            .validate_owner_execution(peer.owner(), &identity)
            .unwrap();
    }
    std::thread::scope(|scope| {
        let handles = peers
            .iter_mut()
            .map(|peer| {
                scope.spawn(|| {
                    peer.exchange(
                        binding().transaction(),
                        0,
                        TensorParallelCollective::AllGather,
                        vec![1.; 11],
                    )
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_eq!(handle.join().unwrap().len(), 22);
        }
    });
    control.finish(binding().transaction()).unwrap();
    let peers = raw_peers();
    for peer in &peers {
        peer.control()
            .register_sites(
                peer.owner(),
                vec![site(TensorCollectiveShape::PerSequence {
                    fixed: 0,
                    per_sequence: 17,
                    per_row: 0,
                })],
            )
            .unwrap();
    }
    // S=1 would fit the limit of 32; S=2 must fail before execution.
    assert!(
        peers[0]
            .control()
            .begin_sealed(identity, Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}

#[test]
fn per_sequence_arithmetic_rejects_each_overflow() {
    for shape in [
        TensorCollectiveShape::PerSequence {
            fixed: 0,
            per_sequence: usize::MAX,
            per_row: 0,
        },
        TensorCollectiveShape::PerSequence {
            fixed: usize::MAX,
            per_sequence: 1,
            per_row: 0,
        },
        TensorCollectiveShape::PerSequence {
            fixed: 0,
            per_sequence: 1,
            per_row: usize::MAX,
        },
        TensorCollectiveShape::PerSequence {
            fixed: 0,
            per_sequence: 0,
            per_row: 0,
        },
    ] {
        assert!(shape.elements(2, 2).is_err());
    }
}
