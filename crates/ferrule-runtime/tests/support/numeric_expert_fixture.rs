//! Deterministic tiny numeric checkpoint, shared by the test and real CUDA child.
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::{ParameterDType, ParameterResidency, ParameterSpec};
use ferrule_model::transformer::*;
use std::{path::Path, sync::Arc};
pub fn bind(path: &Path, experts: usize, numeric: bool, write: bool) -> BoundDecoderResources {
    let path = path.to_path_buf();
    let config = serde_json::json!({
        "vocab_size": 8, "hidden_size": 8, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 12,
        "num_experts": experts, "experts_per_token": 4, "max_position_embeddings": 16,
        "rms_norm_eps": 0.00001, "rope_theta": 10000.0, "tie_word_embeddings": false
    });
    let output = SyntheticDecoderRecipe::new().build(&config).unwrap();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    let mut bytes = Vec::new();
    for p in output.schema().parameters() {
        let role = output.schema().role(p.id()).unwrap().clone();
        let compressed = numeric && matches!(p.residency(), ParameterResidency::Expert { .. });
        let parameter = if compressed {
            ParameterSpec::new(
                p.id(),
                p.path().clone(),
                ParameterDType::F8E4M3,
                p.shape().to_vec(),
                p.residency().clone(),
            )
            .unwrap()
            .with_required_scale(ParameterDType::Bf16, [1, 1])
            .unwrap()
        } else {
            p.clone()
        };
        schema.register_with_role(parameter, role.clone()).unwrap();
        let name = p.path().to_string();
        mapper
            .insert(&name, NameMapping::weight(p.path().clone()))
            .unwrap();
        let offset = bytes.len();
        for i in 0..p.shape().iter().product::<usize>() {
            let index = (i * 7 + i / 5 + p.id().get() as usize) % 4;
            if compressed {
                bytes.push([0x38, 0xb8, 0x30, 0xb0][index]);
            } else {
                bytes.extend([0.125f32, -0.125, 0.0625, -0.0625][index].to_le_bytes());
            }
        }
        slices.push(CheckpointTensorSlice {
            name: name.clone(),
            role: role.clone(),
            path: path.clone(),
            offset: offset as u64,
            bytes: (bytes.len() - offset) as u64,
            shape: p.shape().to_vec(),
            dtype: if compressed {
                CheckpointDType::F8E4M3
            } else {
                CheckpointDType::F32
            },
        });
        if compressed {
            let scale = format!("{name}.scale");
            mapper
                .insert(&scale, NameMapping::scale(p.path().clone()))
                .unwrap();
            slices.push(CheckpointTensorSlice {
                name: scale,
                role,
                path: path.clone(),
                offset: bytes.len() as u64,
                bytes: 2,
                shape: vec![1, 1],
                dtype: CheckpointDType::Bf16,
            });
            bytes.extend(0x3e00u16.to_le_bytes()); // exactly 0.125
        }
    }
    if write {
        std::fs::write(&path, bytes).unwrap();
    }
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap();
    BoundDecoderResources::new(output.spec().clone(), Arc::new(bound)).unwrap()
}
