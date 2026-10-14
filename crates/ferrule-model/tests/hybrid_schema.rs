//! Strict production metadata, no checkpoint payloads or device execution.
use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::decoder::*;
use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};

fn spec() -> ferrule_model::transformer::DecoderModelSpec {
    let value = serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap();
    Qwen35Recipe::spec(&Qwen35Config::from_value(&value).unwrap()).unwrap()
}
#[test]
fn strict_08b_pp2_four_coordinates_and_exact_state_kv_bytes() {
    let spec = spec();
    let schemas = [
        HybridStateSchema::for_segment(&spec, 0..12).unwrap(),
        HybridStateSchema::for_segment(&spec, 12..24).unwrap(),
    ];
    HybridStateSchema::validate_partition(&spec, &schemas).unwrap();
    for (stage, schema) in schemas.iter().enumerate() {
        assert_eq!(schema.total_layers(), 24);
        assert_eq!(schema.layers().len(), 12);
        assert_eq!(schema.per_sequence_state_bytes().unwrap(), 10_321_920);
        assert_eq!(
            schema.kv_planes(16, 32).unwrap().checked_page_bytes(),
            Some(196_608)
        );
        assert!(schema.entry(stage * 12 + 12).is_none());
        for (local, entry) in schema.entries().iter().enumerate() {
            assert_eq!(entry.global_layer, stage * 12 + local);
            assert_eq!(entry.local_layer, local);
            assert_eq!(entry.state_slot, local);
            if local % 4 == 3 {
                assert_eq!(entry.global_kv_layer, Some(stage * 3 + local / 4));
                assert_eq!(entry.local_kv_layer, Some(local / 4));
            } else {
                assert_eq!(entry.global_kv_layer, None);
                assert_eq!(entry.local_kv_layer, None);
            }
        }
    }
    let full = HybridStateSchema::from_spec(&spec, 24).unwrap();
    assert_eq!(full.per_sequence_state_bytes().unwrap(), 20_643_840);
    assert_eq!(
        full.kv_planes(16, 32).unwrap().checked_page_bytes(),
        Some(393_216)
    );
}
#[test]
fn range_partition_and_recurrent_only_rejections_are_typed() {
    let spec = spec();
    for (start, end) in [(0, 0), (24, 25), (12, 11), (0, usize::MAX)] {
        let range = start..end;
        assert!(
            matches!(HybridStateSchema::for_segment(&spec, range).unwrap_err(),
            ferrule_common::Error::ModelSource { source } if matches!(source.downcast_ref::<HybridSchemaError>(), Some(HybridSchemaError::InvalidRange { .. })))
        );
    }
    assert!(
        matches!(HybridStateSchema::for_segment(&spec, 12..15).unwrap_err(),
        ferrule_common::Error::ModelSource { source } if matches!(source.downcast_ref::<HybridSchemaError>(), Some(HybridSchemaError::RecurrentOnly { .. })))
    );
    let a = HybridStateSchema::for_segment(&spec, 0..12).unwrap();
    let b = HybridStateSchema::for_segment(&spec, 12..24).unwrap();
    for bad in [
        vec![],
        vec![a.clone()],
        vec![b.clone(), a.clone()],
        vec![a.clone(), a],
        vec![HybridStateSchema::for_segment(&spec, 0..8).unwrap(), b],
    ] {
        assert!(HybridStateSchema::validate_partition(&spec, &bad).is_err());
    }
    let prefix = HybridStateSchema::from_spec(&spec, 3).unwrap();
    assert!(prefix.kv_planes(16, 32).is_err());
}
#[test]
fn nonzero_range_state_access_uses_global_identity_not_local_slot() {
    let schema = HybridStateSchema::for_segment(&spec(), 12..24).unwrap();
    let mut states = schema.create().unwrap();
    assert_eq!(states.layers().len(), 12);
    assert!(states.linear_mut(0).is_err());
    assert!(states.linear_mut(12).is_ok());
    assert!(states.linear_mut(15).is_err());
    assert!(states.linear_mut(24).is_err());
    states.validate(&schema, 0).unwrap();
    assert!(states.validate(&schema, 1).is_err());
    assert!(
        states
            .validate(&HybridStateSchema::for_segment(&spec(), 0..12).unwrap(), 0)
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn strict_08b_segment_admission_charges_output_workspace_and_tied_head_per_owner() {
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::models::qwen35::{Qwen35HfNameMapper, Qwen35TensorPartitionKind};
    use ferrule_model::transformer::{BoundDecoderResources, LayerSegmentPlan, StateDictBinder};
    use ferrule_model::{
        CheckpointDType, CheckpointTensorSlice, ModelFamily, TensorRole, WeightSource,
    };
    let value = serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap();
    let config = Qwen35Config::from_value(&value).unwrap();
    let spec = Qwen35Recipe::spec(&config).unwrap();
    let schema = Qwen35Recipe::schema(&config).unwrap();
    let mapper = Qwen35HfNameMapper::new(&config);
    let path = std::env::temp_dir().join(format!(
        "ferrule-hybrid-stage-budget-{}",
        std::process::id()
    ));
    struct Remove(std::path::PathBuf);
    impl Drop for Remove {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _remove = Remove(path.clone());
    let mut offset = 0;
    let slices = mapper
        .tensors()
        .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
        .map(|t| {
            let slice = CheckpointTensorSlice {
                name: t.external_name.clone(),
                role: TensorRole::Unknown,
                path: path.clone(),
                offset,
                bytes: t.bytes(),
                shape: t.shape.clone(),
                dtype: CheckpointDType::from_safetensors_dtype(t.dtype.as_str()),
            };
            offset += slice.bytes;
            slice
        })
        .collect::<Vec<_>>();
    // Sparse backing only supplies exact source identities to the real binder;
    // this test never reads weights, creates CUDA owners, or executes a kernel.
    std::fs::File::create_new(&path)
        .unwrap()
        .set_len(offset)
        .unwrap();
    let bound = StateDictBinder::new(&schema, &mapper)
        .bind_slices(slices)
        .unwrap();
    let resources = BoundDecoderResources::new(spec, std::sync::Arc::new(bound)).unwrap();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        16,
        32,
        2,
        1,
        4 << 30,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let first = LayerSegmentPlan::new(24, 0..12, true, false).unwrap();
    let last = LayerSegmentPlan::new(24, 12..24, false, true).unwrap();
    let a = HybridCudaMemoryEstimate::for_segment(&resources, &options, &first, 4).unwrap();
    let b = HybridCudaMemoryEstimate::for_segment(&resources, &options, &last, 4).unwrap();
    assert_eq!(a.per_sequence_state_bytes, 10_321_920);
    assert_eq!(b.per_sequence_state_bytes, 10_321_920);
    assert_eq!(a.kv_bytes, 196_608 * 4);
    assert_eq!(b.kv_bytes, 196_608 * 4);
    assert!(b.workspace_bytes_upper_bound > a.workspace_bytes_upper_bound * 10);
    let bare = HybridCudaMemoryEstimate::for_segment(
        &resources,
        &options,
        &LayerSegmentPlan::new(24, 12..24, false, false).unwrap(),
        4,
    )
    .unwrap();
    assert_eq!(
        b.weight_bytes_upper_bound - bare.weight_bytes_upper_bound,
        (248_320 * 1024 + 1024) * 8
    );
    let without_embedding = HybridCudaMemoryEstimate::for_segment(
        &resources,
        &options,
        &LayerSegmentPlan::new(24, 0..12, false, false).unwrap(),
        4,
    )
    .unwrap();
    assert_eq!(
        a.weight_bytes_upper_bound - without_embedding.weight_bytes_upper_bound,
        248_320 * 1024 * 8
    );
}
