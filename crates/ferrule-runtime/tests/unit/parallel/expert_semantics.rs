use super::*;
use crate::parallel::expert::{ExpertRank, ExpertRankWorker};
use ferrule_model::moe::catalog::ValidatedExpertCatalog;
use ferrule_model::transformer::{
    DecoderLayer, DecoderModelParts, DecoderModelSpec, ExpertDispatchLimits, ExpertPlacement,
    FeedForward, Moe, UnsupportedOperator,
};

use crate::parallel::expert::numeric_test_fixture as fixture;

fn image() -> (tempfile::NamedTempFile, BoundDecoderResources) {
    let file = tempfile::NamedTempFile::new().unwrap();
    let resources = fixture::bind(file.path(), 4, false, true);
    (file, resources)
}

fn with_last_moe(
    resources: &BoundDecoderResources,
    change: impl FnOnce(&Moe) -> Moe,
) -> BoundDecoderResources {
    let spec = resources.spec();
    let mut layers = spec.layers().to_vec();
    let old = &layers[1];
    let FeedForward::Moe(moe) = old.feed_forward() else {
        panic!()
    };
    layers[1] = DecoderLayer::new(
        1,
        old.input_norm().clone(),
        old.attention().clone(),
        old.attention_residual().clone(),
        old.post_attention_norm().clone(),
        FeedForward::Moe(change(moe)),
        old.feed_forward_residual().clone(),
    )
    .unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: spec.architecture().into(),
        hidden_size: spec.hidden_size(),
        vocab_size: spec.vocab_size(),
        max_sequence_length: spec.max_sequence_length(),
        token_embedding: spec.token_embedding().clone(),
        layers,
        final_norm: spec.final_norm().clone(),
        output: spec.output().clone(),
        tie_word_embeddings: spec.tie_word_embeddings(),
    })
    .unwrap();
    BoundDecoderResources::new(spec, Arc::new(resources.state_dict().clone())).unwrap()
}

fn group(layers: std::ops::Range<usize>) -> ExpertGroup {
    ExpertGroup {
        source_rank: ParallelRankId::new(7),
        members: vec![ParallelRankId::new(109), ParallelRankId::new(42)],
        placement: ExpertPlacement::new(layers.clone().flat_map(|layer| {
            (0..4).map(move |expert| {
                (
                    layer,
                    expert,
                    ParallelRankId::new(if expert % 2 == 0 { 109 } else { 42 }),
                )
            })
        }))
        .unwrap(),
        layers,
        limits: ExpertDispatchLimits {
            max_tokens: 4,
            max_bytes: 128,
        },
    }
}

#[test]
fn expert_catalog_is_payload_free_and_preserves_global_roles_and_decoder_semantics() {
    let (file, resources) = image();
    let resources = with_last_moe(&resources, |moe| {
        moe.clone().with_activation_limit(0.125).unwrap()
    });
    // Even inaccessible payloads cannot prevent pure semantic catalog validation.
    file.close().unwrap();
    let catalog = ValidatedExpertCatalog::new(&resources, 1..2).unwrap();
    assert_eq!(catalog.layers().len(), 1);
    let layer = &catalog.layers()[0];
    assert_eq!(layer.layer(), 1);
    assert_eq!(layer.descriptor().router_spec().num_experts(), 4);
    assert_eq!(layer.descriptor().router().weight_shape(), [4, 8]);
    assert!(layer.shared().is_none());
    assert!(layer.shared_output_gate().is_none());
    assert!(layer.router_bias().is_none());
    assert_eq!(
        layer.router().role(),
        &ferrule_model::TensorRole::RouterLogits
    );
    assert_eq!(catalog.experts().len(), 4);
    assert_eq!(layer.router().spec().shape(), [4, 8]);
    for (expert, semantics) in catalog.experts().iter().enumerate() {
        assert_eq!(semantics.id(), ExpertId::new(1, expert));
        assert_eq!(semantics.activation_limit(), Some(0.125));
        for (parameter, (role, shape)) in semantics.parameters().iter().zip([
            (ferrule_model::TensorRole::RoutedExpertGate, [12, 8]),
            (ferrule_model::TensorRole::RoutedExpertUp, [12, 8]),
            (ferrule_model::TensorRole::RoutedExpertDown, [8, 12]),
        ]) {
            assert_eq!(parameter.role(), &role);
            assert_eq!(parameter.spec().shape(), shape);
            assert!(
                parameter
                    .shares_storage_with(resources.experts().require(1, expert, role).unwrap())
            );
        }
    }
    let selected = catalog
        .prepare_selected(&StateDictMaterializer::new(4096).unwrap(), |_| false)
        .unwrap();
    assert!(selected.is_empty());
    assert!(ValidatedExpertCatalog::new(&resources, 2..3).is_err());
    assert!(ValidatedExpertCatalog::new(&resources, 1..1).is_err());
}

fn local_prepare(
    resources: &BoundDecoderResources,
) -> Result<ferrule_model::transformer::CpuGqaMoeModule> {
    ferrule_model::transformer::CpuGqaMoeModule::prepare(
        Arc::new(resources.clone()),
        Arc::new(StateDictMaterializer::new(4096)?),
        ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
        16,
    )
}

#[test]
fn expert_thread_and_process_reject_late_shape_mismatch_before_any_payload() {
    let (file, resources) = image();
    let resources = with_last_moe(&resources, |moe| {
        Moe::new(8, 13, moe.router_spec().clone(), false).unwrap()
    });
    file.close().unwrap();
    let group = group(0..2);
    let owner = group.members[0];
    let thread = ExpertRankWorker::prepare_cpu(
        &resources,
        ExpertRank {
            slot: ParallelRankId::new(0),
            owner,
        },
        &group,
        4096,
    )
    .err()
    .unwrap();
    let process = prepare(&resources, &group, owner, 4096).err().unwrap();
    let local = local_prepare(&resources).unwrap_err();
    assert_eq!(local.to_string(), thread.to_string());
    assert_eq!(thread.to_string(), process.to_string());
    assert!(thread.to_string().contains("shape"), "{thread}");
}

#[test]
fn expert_thread_and_process_reject_unsupported_bias_without_payload() {
    let (file, resources) = image();
    let resources = with_last_moe(&resources, |moe| {
        Moe::new(8, 12, moe.router_spec().clone(), true).unwrap()
    });
    file.close().unwrap();
    let group = group(0..2);
    let owner = group.members[0];
    for error in [
        local_prepare(&resources).unwrap_err(),
        ExpertRankWorker::prepare_cpu(
            &resources,
            ExpertRank {
                slot: ParallelRankId::new(0),
                owner,
            },
            &group,
            4096,
        )
        .err()
        .unwrap(),
        prepare(&resources, &group, owner, 4096).err().unwrap(),
    ] {
        let ferrule_common::Error::ModelSource { source } = error else {
            panic!("{error}")
        };
        assert_eq!(
            source
                .downcast_ref::<UnsupportedOperator>()
                .unwrap()
                .operator,
            "routed_expert_lowering"
        );
    }
}

#[cfg(feature = "cpu-reference")]
#[test]
fn expert_thread_process_lowering_and_clamped_outputs_match_at_global_layer_one() {
    let (_file, resources) = image();
    let resources = with_last_moe(&resources, |moe| {
        moe.clone().with_activation_limit(0.01).unwrap()
    });
    let group = group(1..2);
    for (slot, &owner) in group.members.iter().enumerate() {
        let mut thread = ExpertRankWorker::prepare_cpu(
            &resources,
            ExpertRank {
                slot: ParallelRankId::new(slot as u32),
                owner,
            },
            &group,
            4096,
        )
        .unwrap();
        let mut process = prepare(&resources, &group, owner, 4096).unwrap();
        assert_eq!(process.0.len(), 2);
        let tokens = process
            .0
            .keys()
            .map(|&expert| ExpertToken {
                transaction: ExecutionTransactionId::new(37).unwrap(),
                sequence: 19,
                source_rank: group.source_rank,
                source_row: 0,
                route_slot: expert.expert,
                expert,
                weight: 0.75,
                payload: vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
            })
            .collect::<Vec<_>>();
        assert!(
            process
                .0
                .values()
                .all(|e| e.activation_limit() == Some(0.01))
        );
        let expected = thread.compute(&tokens).unwrap();
        let actual = CpuReferenceExpertWorker::new(owner, &group.placement, &mut process)
            .compute(&tokens)
            .unwrap();
        assert_eq!(actual, expected);
        assert!(
            actual
                .iter()
                .all(|r| r.expert.layer == 1 && r.owner_rank == owner)
        );
    }
}

#[test]
fn expert_catalog_validates_shared_roles_without_materializing_decoder_payloads() {
    use ferrule_model::TensorRole;
    use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
    use ferrule_model::nn::{
        ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
    };
    use ferrule_model::transformer::{
        ExactNameMapper, Linear, NameMapping, StateDictBinder, StateDictSchema, SwiGlu,
    };
    let (file, resources) = image();
    let resources = with_last_moe(&resources, |moe| {
        moe.clone()
            .with_shared_expert(SwiGlu::new(8, 3, false).unwrap())
            .unwrap()
            .with_shared_expert_gate(Linear::new(8, 1, false).unwrap())
            .unwrap()
    });
    assert!(
        ValidatedExpertCatalog::new(&resources, 1..2)
            .unwrap_err()
            .to_string()
            .contains("shared_expert_gate")
    );
    let shared_file = tempfile::NamedTempFile::new().unwrap();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    let mut next_id = 1;
    for parameter in resources.state_dict().parameters() {
        schema
            .register_with_role(parameter.spec().clone(), parameter.role().clone())
            .unwrap();
        mapper
            .insert(
                &parameter.weight().slice().name,
                NameMapping::weight(parameter.path().clone()),
            )
            .unwrap();
        slices.push(parameter.weight().slice().clone());
        next_id = next_id.max(parameter.id().get() + 1);
    }
    let mut offset = 0;
    for (index, (role, shape)) in [
        (TensorRole::SharedExpertGate, [3, 8]),
        (TensorRole::SharedExpertUp, [3, 8]),
        (TensorRole::SharedExpertDown, [8, 3]),
        (TensorRole::SharedExpertOutputGate, [1, 8]),
    ]
    .into_iter()
    .enumerate()
    {
        let name = format!("layers.1.shared.test_{index}");
        let parameter = ParameterSpec::new(
            ParameterId::new(next_id + index as u64),
            ModulePath::new(&name).unwrap(),
            ParameterDType::F32,
            shape.to_vec(),
            ParameterResidency::layer(1),
        )
        .unwrap();
        mapper
            .insert(&name, NameMapping::weight(parameter.path().clone()))
            .unwrap();
        schema.register_with_role(parameter, role.clone()).unwrap();
        let bytes = (shape[0] * shape[1] * 4) as u64;
        slices.push(CheckpointTensorSlice {
            name,
            role,
            path: shared_file.path().into(),
            offset,
            bytes,
            shape: shape.to_vec(),
            dtype: CheckpointDType::F32,
        });
        offset += bytes;
    }
    std::fs::write(shared_file.path(), vec![0u8; offset as usize]).unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap();
    let resources = BoundDecoderResources::new(resources.spec().clone(), Arc::new(bound)).unwrap();
    file.close().unwrap();
    shared_file.close().unwrap();
    let catalog = ValidatedExpertCatalog::new(&resources, 1..2).unwrap();
    let shared = catalog.layers()[0].shared().unwrap();
    assert_eq!(
        shared.each_ref().map(|p| p.role().clone()),
        [
            TensorRole::SharedExpertGate,
            TensorRole::SharedExpertUp,
            TensorRole::SharedExpertDown,
        ]
    );
    let gate = catalog.layers()[0].shared_output_gate().unwrap();
    assert_eq!(gate.role(), &TensorRole::SharedExpertOutputGate);
    assert_eq!(gate.spec().shape(), [1, 8]);
    assert!(
        catalog
            .prepare_selected(&StateDictMaterializer::new(4096).unwrap(), |_| false)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn local_thread_process_out_of_range_reject_before_payload() {
    use ferrule_model::transformer::{LayerSegmentPlan, StandardDecoderSegment};
    let (file, resources) = image();
    file.close().unwrap();
    let group = group(2..3);
    let owner = group.members[0];
    let thread = ExpertRankWorker::prepare_cpu(
        &resources,
        ExpertRank {
            slot: ParallelRankId::new(0),
            owner,
        },
        &group,
        4096,
    )
    .err()
    .unwrap();
    let process = prepare(&resources, &group, owner, 4096).err().unwrap();
    assert_eq!(thread.to_string(), process.to_string());
    assert!(thread.to_string().contains("outside decoder"));
    let local = StandardDecoderSegment::prepare(
        &resources,
        LayerSegmentPlan::new(3, 2..3, false, true).unwrap(),
        ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
        16,
        4096,
    )
    .unwrap_err();
    assert!(matches!(
        local,
        ferrule_model::transformer::SegmentError::LayerCount { .. }
    ));
}
