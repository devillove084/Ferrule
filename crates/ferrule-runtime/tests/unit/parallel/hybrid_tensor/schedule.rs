use super::*;
use crate::parallel::collective::HostCollectiveKind;
use crate::parallel::pipeline::tensor_scope::{
    TensorCollectiveOperator, TensorCollectiveSemantic as Op, TensorCollectiveShape as Shape,
};
use ferrule_model::transformer::StandardTensorCollective;
use ferrule_model::transformer::hybrid_parallel::{
    HYBRID_AGREEMENT_VERSION, HybridCollectiveKind, HybridCollectiveSemantic,
};

#[test]
fn runtime_translates_only_model_wire_metadata_for_tp1_tp2_tp4_and_both_boundaries() {
    for tp in [1, 2, 4] {
        let fixture = fixture::Fixture::for_degree(4, false);
        let resources = fixture.resources();
        let layout = layout(resources.spec(), 2, tp);
        for (stage, plan) in layout.stages().iter().enumerate() {
            let segment =
                LayerSegmentPlan::new(4, plan.schema().layer_range(), stage == 0, stage == 1)
                    .unwrap();
            let model = plan.collective_schedule(&segment).unwrap();
            let runtime = translate_hybrid_schedule(plan, &segment).unwrap();
            assert_eq!(model.agreement_version(), HYBRID_AGREEMENT_VERSION);
            assert_eq!(runtime.len(), model.wire_sites().len());
            assert_eq!(runtime.is_empty(), tp == 1);
            for (a, b) in runtime.iter().zip(model.wire_sites()) {
                assert_eq!(a.site, b.ordinal);
                assert_eq!(a.layer, b.global_layer);
                let TensorCollectiveOperator::Protocol {
                    version,
                    operation,
                    kind,
                    shape,
                } = a.operator
                else {
                    panic!("hybrid must not pretend to be dense Linear")
                };
                assert_eq!(version, model.agreement_version());
                assert_eq!(
                    kind,
                    match b.kind {
                        HybridCollectiveKind::AllGather => HostCollectiveKind::AllGatherF32,
                        HybridCollectiveKind::ReduceSum => HostCollectiveKind::AllReduceSumF32,
                    }
                );
                assert_eq!(
                    operation,
                    match b.semantic {
                        HybridCollectiveSemantic::Agreement => Op::InputAgreement,
                        HybridCollectiveSemantic::AttentionOutput => Op::AttentionOutput,
                        HybridCollectiveSemantic::DenseMlpDown => Op::DenseMlpOutput,
                        HybridCollectiveSemantic::SharedExpertDown => Op::SharedExpertOutput,
                        HybridCollectiveSemantic::RoutedExpertOutput => Op::RoutedExpertOutput,
                        HybridCollectiveSemantic::OutputHead => Op::OutputHead,
                        HybridCollectiveSemantic::Completion => Op::Frontier,
                    }
                );
                assert_eq!(
                    shape,
                    Shape::PerSequence {
                        fixed: b.shape.fixed,
                        per_sequence: b.shape.per_sequence,
                        per_row: b.shape.per_row
                    }
                );
                for (sequences, rows) in [(1, 1), (1, 3), (2, 5), (3, 9)] {
                    assert_eq!(
                        a.elements(sequences, rows).unwrap(),
                        b.shape.elements(sequences, rows).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn sealed_protocol_compares_version_kind_semantic_shape_ordinal_and_global_layer() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let layout = layout(resources.spec(), 2, 2);
    let plan = &layout.stages()[0];
    let segment = LayerSegmentPlan::new(4, 0..2, true, false).unwrap();
    let sites = translate_hybrid_schedule(plan, &segment).unwrap();
    for axis in 0..6 {
        let peers = DecoderTensorCollective::new_group_from_members(
            plan.tensor_members(),
            plan.scope().topology,
            ParallelGroupId::new(1),
            options().collective_limits,
            options().collective_timeout,
        )
        .unwrap();
        let control = peers[0].control();
        control
            .register_sites(peers[0].owner(), sites.clone())
            .unwrap();
        let mut changed = sites.clone();
        if axis == 4 {
            changed[0].site += 100;
        } else if axis == 5 {
            changed[1].layer = Some(99);
        } else {
            let TensorCollectiveOperator::Protocol {
                version,
                kind,
                operation,
                shape,
            } = &mut changed[0].operator
            else {
                unreachable!()
            };
            match axis {
                0 => *version += 1,
                1 => *kind = HostCollectiveKind::AllReduceSumF32,
                2 => *operation = Op::OutputHead,
                3 => {
                    // Same S=1 element count does not mean the same protocol.
                    let Shape::PerSequence {
                        fixed,
                        per_sequence,
                        ..
                    } = shape
                    else {
                        unreachable!()
                    };
                    *fixed += 1;
                    *per_sequence -= 1;
                    assert_eq!(
                        sites[0].elements(1, 3).unwrap(),
                        changed[0].elements(1, 3).unwrap()
                    );
                }
                _ => unreachable!(),
            }
        }
        assert!(
            control.register_sites(peers[1].owner(), changed).is_err(),
            "axis {axis}"
        );
        assert!(control.is_failed());
    }
}
