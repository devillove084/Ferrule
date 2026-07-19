use std::env;
use std::path::{Path, PathBuf};

use ferrule_model::models::deepseek_v4::DeepSeekV4ArtifactModel;
use ferrule_model::{
    AttentionKind, EnginePlanStatus, HfSafetensorsIndex, HfSafetensorsInventory, ModelDescriptor,
    ModelFamily, PolicyArea, RouterKind, SpeculationMode, TensorClass, TensorRole, WeightSource,
};

#[test]
fn local_deepseek_v4_flash_dspark_descriptor_smoke_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let index = HfSafetensorsIndex::open(model_dir.join("model.safetensors.index.json"))
        .expect("local DeepSeek V4 safetensors index should parse");
    assert_eq!(index.tensor_count(), 72_317);
    assert_eq!(index.shard_count(), 48);
    assert_eq!(index.total_size, Some(166_878_536_440));
    assert!(
        index.missing_shards(&model_dir).is_empty(),
        "local DeepSeek V4 download is incomplete"
    );

    let inventory = HfSafetensorsInventory::from_index(&model_dir, ModelFamily::DeepSeekV4, &index)
        .expect("local DeepSeek V4 safetensors headers should scan without tensor payloads");
    assert_eq!(inventory.tensor_count, 72_317);
    assert_eq!(inventory.shard_count, 48);
    assert!(inventory.index_only_tensors.is_empty());
    assert!(inventory.header_only_tensors.is_empty());
    assert_eq!(inventory.class_count(&TensorClass::Unknown), 0);
    assert_eq!(inventory.dtype_bytes("BF16"), 2_967_134_976);
    assert_eq!(inventory.dtype_bytes("F32"), 150_966_520);
    assert_eq!(inventory.dtype_bytes("F8_E4M3"), 6_304_038_912);
    assert_eq!(inventory.dtype_bytes("F8_E8M0"), 9_261_408_000);
    assert_eq!(inventory.dtype_bytes("I64"), 18_616_320);
    assert_eq!(inventory.dtype_bytes("I8"), 148_176_371_712);
    assert_eq!(
        inventory.role_bytes(&TensorRole::RoutedExpertGate),
        52_479_131_648
    );
    assert_eq!(
        inventory.role_bytes(&TensorRole::RoutedExpertUp),
        52_479_131_648
    );
    assert_eq!(
        inventory.role_bytes(&TensorRole::RoutedExpertDown),
        52_479_131_648
    );
    assert!(inventory.role_bytes(&TensorRole::SpeculativeProjection) > 0);
    let routed_experts = inventory.routed_expert_tensors();
    assert_eq!(routed_experts.len(), 66_048);
    assert!(
        routed_experts
            .iter()
            .all(|tensor| tensor.descriptor.layer < 43 && tensor.descriptor.expert < 256)
    );
    assert!(
        routed_experts
            .iter()
            .all(|tensor| tensor.file_offset > tensor.data_offset)
    );

    let descriptor = ModelDescriptor::load(&model_dir)
        .expect("local DeepSeek V4 descriptor should load without tensor payloads");
    assert_eq!(descriptor.spec.family, ModelFamily::DeepSeekV4);
    assert_eq!(descriptor.spec.weight_source, WeightSource::Safetensors);
    assert_eq!(
        descriptor.spec.attention,
        AttentionKind::MultiLatentAttention
    );
    assert_eq!(descriptor.spec.hidden_size, Some(4096));
    assert_eq!(descriptor.spec.num_layers, Some(43));
    assert_eq!(descriptor.spec.vocab_size, Some(129_280));
    assert_eq!(descriptor.spec.num_heads, Some(64));
    assert_eq!(descriptor.spec.num_kv_heads, Some(1));
    assert_eq!(descriptor.spec.head_dim, Some(512));
    assert_eq!(descriptor.spec.moe.num_experts, Some(256));
    assert_eq!(descriptor.spec.moe.num_experts_per_tok, Some(6));
    assert_eq!(descriptor.spec.moe.router, RouterKind::HashAssistedTopK);
    assert!(descriptor.spec.moe.has_shared_experts);
    assert_eq!(descriptor.spec.tensor_count, Some(72_317));
    assert!(
        descriptor
            .spec
            .notes
            .iter()
            .any(|note| note.contains("DSpark attachment metadata"))
    );

    assert!(
        descriptor
            .spec
            .notes
            .iter()
            .any(|note| note.contains("HF safetensors header inventory"))
    );
    assert!(
        descriptor
            .spec
            .quantization
            .iter()
            .any(|item| item.format == "F8_E4M3")
    );

    assert_eq!(tensor_class_count(&descriptor, &TensorClass::Unknown), 0);
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::AttentionSink),
        46
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::SpeculativeProjection),
        2
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::SpeculativeMarkovHead),
        2
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::SpeculativeConfidenceHead),
        1
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::RoutedExpertGate),
        23_552
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::RoutedExpertUp),
        23_552
    );
    assert_eq!(
        tensor_class_count(&descriptor, &TensorClass::RoutedExpertDown),
        23_552
    );

    let plan = descriptor.engine_plan();
    assert_eq!(plan.status, EnginePlanStatus::MetadataOnly);
    assert_eq!(
        plan.policies.speculation.mode,
        SpeculationMode::MultiTokenPrediction
    );
    assert_missing(&plan, PolicyArea::Attention, "latent/compressed attention");
    assert_missing(&plan, PolicyArea::Attention, "attention sink");
    assert_missing(&plan, PolicyArea::Router, "hash-assisted routing");
    assert_missing(&plan, PolicyArea::Speculation, "speculative decoding");
    assert_missing(&plan, PolicyArea::Tokenizer, "external tokenizer/encoding");
}

#[test]
#[ignore = "reads the local DSpark attachment payloads"]
fn local_deepseek_v4_flash_dspark_mtp_attachment_loads() {
    let model_dir = local_deepseek_v4_dir().expect("local DeepSeek V4 checkpoint is required");
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .expect("local DeepSeek V4 artifact should load");
    let mtp = model
        .load_mtp()
        .expect("local DSpark attachment should bind")
        .expect("local checkpoint should contain an MTP attachment");

    assert_eq!(mtp.config.block_size, 5);
    assert_eq!(mtp.config.noise_token_id, Some(128_799));
    assert_eq!(mtp.config.target_layer_ids, vec![40, 41, 42]);
    assert_eq!(mtp.config.markov_rank, Some(256));
    assert_eq!(mtp.layers.len(), 3);
    assert!(mtp.prediction_heads.is_some());
    for (stage, layer) in mtp.layers.iter().enumerate() {
        assert_eq!(layer.mtp_index, stage);
        assert_eq!(layer.execution_layer, 43 + stage);
        assert_eq!(layer.expert_source_catalog.count(), 256);
        assert_eq!(layer.main_proj.is_some(), stage == 0);
        assert_eq!(layer.main_norm.is_some(), stage == 0);
    }
}

fn local_deepseek_v4_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("FERRULE_DEEPSEEK_V4_DIR") {
        let path = PathBuf::from(path);
        assert!(
            has_complete_deepseek_v4_artifact(&path),
            "FERRULE_DEEPSEEK_V4_DIR must point at a complete 48-shard HF model directory"
        );
        return Some(path);
    }

    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models")
        .join("DeepSeek-V4-Flash-DSpark");
    has_complete_deepseek_v4_artifact(&default).then_some(default)
}

fn has_complete_deepseek_v4_artifact(path: &Path) -> bool {
    path.join("config.json").is_file()
        && path.join("model.safetensors.index.json").is_file()
        && path.join("tokenizer.json").is_file()
        && (1..=48).all(|shard| {
            path.join(format!("model-{shard:05}-of-00048.safetensors"))
                .is_file()
        })
}

fn tensor_class_count(descriptor: &ModelDescriptor, class: &TensorClass) -> usize {
    descriptor
        .tensor_classes
        .iter()
        .find(|item| &item.class == class)
        .map(|item| item.tensors)
        .unwrap_or(0)
}

fn assert_missing(plan: &ferrule_model::EnginePlan, area: PolicyArea, needle: &str) {
    assert!(
        plan.missing
            .iter()
            .any(|item| item.area == area && item.reason.contains(needle)),
        "missing policy did not include area={area:?} reason containing {needle:?}; got {:#?}",
        plan.missing
    );
}
