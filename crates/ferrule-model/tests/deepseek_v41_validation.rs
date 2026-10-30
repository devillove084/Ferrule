#![recursion_limit = "512"]

//! Hermetic negative coverage for the DeepSeek-V4.1 metadata boundary.
//! No runtime, CUDA, or tensor payload is created or read.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrule_model::models::deepseek_v41::{
    DeepSeekV41Config, validate_header_inventory, validate_index, validate_real_index,
};
use ferrule_model::{
    HfSafetensorsIndex, HfSafetensorsInventory, HfSafetensorsShardSummary, HfSafetensorsTensorInfo,
    ModelFamily, TensorClass, TensorRole,
};
use serde_json::{Value, json};

#[allow(unused_attributes)]
#[path = "deepseek_v41_metadata.rs"]
mod fixture;

fn config() -> DeepSeekV41Config {
    DeepSeekV41Config::from_value(&fixture::config_value()).unwrap()
}

fn header_tensor(
    name: &str,
    dtype: &str,
    shape: &[usize],
    byte_size: u64,
) -> HfSafetensorsTensorInfo {
    HfSafetensorsTensorInfo {
        name: name.into(),
        shard: "model-00001-of-00001.safetensors".into(),
        dtype: dtype.into(),
        shape: shape.to_vec(),
        data_offset: 0,
        file_offset: 8,
        byte_size,
        class: TensorClass::Unknown,
        role: TensorRole::Unknown,
    }
}

static NEXT_HEADER_FIXTURE: AtomicUsize = AtomicUsize::new(0);

fn header_tensor_at(
    name: &str,
    dtype: &str,
    shape: &[usize],
    byte_size: u64,
    data_offset: u64,
) -> HfSafetensorsTensorInfo {
    let mut tensor = header_tensor(name, dtype, shape, byte_size);
    tensor.data_offset = data_offset;
    tensor.file_offset = 8 + data_offset;
    tensor
}

fn normalize_header_offsets(tensors: &mut [HfSafetensorsTensorInfo]) -> Result<(), String> {
    let mut next = 0u64;
    for tensor in tensors {
        tensor.data_offset = next;
        tensor.file_offset = 8 + next;
        next = next
            .checked_add(tensor.byte_size)
            .ok_or_else(|| "synthetic extent overflow".to_owned())?;
    }
    Ok(())
}

fn validate_synthetic_headers(mut tensors: Vec<HfSafetensorsTensorInfo>) -> Result<bool, String> {
    normalize_header_offsets(&mut tensors)?;
    validate_synthetic_headers_impl(tensors, true)
}

fn validate_synthetic_headers_with_offsets(
    tensors: Vec<HfSafetensorsTensorInfo>,
) -> Result<bool, String> {
    validate_synthetic_headers_impl(tensors, true)
}

fn validate_synthetic_header_metadata(
    mut tensors: Vec<HfSafetensorsTensorInfo>,
) -> Result<bool, String> {
    normalize_header_offsets(&mut tensors)?;
    validate_synthetic_headers_impl(tensors, false)
}

fn validate_synthetic_headers_impl(
    tensors: Vec<HfSafetensorsTensorInfo>,
    include_shard_extent: bool,
) -> Result<bool, String> {
    let shard = "model-00001-of-00001.safetensors";
    let total = tensors.iter().try_fold(0u64, |total, tensor| {
        total.checked_add(tensor.byte_size).ok_or("total overflow")
    })?;
    let mut weight_map = BTreeMap::new();
    for tensor in &tensors {
        weight_map.insert(tensor.name.clone(), shard.into());
    }
    let index = HfSafetensorsIndex {
        total_size: Some(total),
        weight_map,
    };
    let max_end = tensors
        .iter()
        .map(|tensor| {
            tensor
                .file_offset
                .checked_add(tensor.byte_size)
                .ok_or("file extent overflow")
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let directory = include_shard_extent.then(|| {
        std::env::temp_dir().join(format!(
            "ferrule-deepseek-v41-header-{}-{}",
            std::process::id(),
            NEXT_HEADER_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ))
    });
    if let Some(directory) = &directory {
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        let file = File::create(directory.join(shard)).map_err(|error| error.to_string())?;
        file.set_len(max_end).map_err(|error| error.to_string())?;
    }
    let shard_summaries = if include_shard_extent {
        vec![HfSafetensorsShardSummary {
            shard: shard.into(),
            tensors: index.tensor_count(),
            bytes: total,
        }]
    } else {
        Vec::new()
    };
    let inventory = HfSafetensorsInventory {
        family: ModelFamily::DeepSeekV41,
        total_size: Some(total),
        shard_count: 1,
        tensor_count: tensors.len(),
        tensors,
        dtype_counts: Vec::new(),
        class_counts: Vec::new(),
        role_counts: Vec::new(),
        shard_summaries,
        index_only_tensors: Vec::new(),
        header_only_tensors: Vec::new(),
    };
    let result = validate_header_inventory(
        &config(),
        &index,
        &inventory,
        directory.as_deref().unwrap_or_else(|| Path::new(".")),
    )
    .map_err(|error| error.to_string());
    if let Some(directory) = directory {
        std::fs::remove_file(directory.join(shard)).map_err(|error| error.to_string())?;
        std::fs::remove_dir(directory).map_err(|error| error.to_string())?;
    }
    result
}

#[test]
fn synthetic_header_contract_accepts_fp8_and_fp4_pairs() {
    let fp8 = validate_synthetic_headers(vec![
        header_tensor("layers.0.attn.wq_a.weight", "F8_E4M3", &[64, 64], 4096),
        header_tensor("layers.0.attn.wq_a.scale", "F8_E8M0", &[2, 2], 4),
    ]);
    assert!(fp8.is_ok(), "{fp8:?}");

    let fp4 = validate_synthetic_headers(vec![
        header_tensor("layers.0.ffn.experts.0.w1.weight", "I8", &[64, 16], 1024),
        header_tensor("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[64, 1], 64),
    ]);
    assert!(fp4.is_ok(), "{fp4:?}");
}

#[test]
fn synthetic_header_contract_covers_shared_fp8_and_real_engram_layouts() {
    let shared = validate_synthetic_headers(vec![
        header_tensor(
            "layers.0.ffn.shared_experts.w1.weight",
            "F8_E4M3",
            &[64, 64],
            4096,
        ),
        header_tensor(
            "layers.0.ffn.shared_experts.w1.scale",
            "F8_E8M0",
            &[2, 2],
            4,
        ),
    ]);
    assert_eq!(shared, Ok(true));

    let first = validate_synthetic_header_metadata(vec![
        header_tensor(
            "layers.1.engram.embed.weight",
            "F8_E4M3",
            &[384_006_168, 256],
            98_305_579_008,
        ),
        header_tensor(
            "layers.1.engram.embed.scale",
            "F8_E8M0",
            &[384_006_168, 8],
            3_072_049_344,
        ),
    ]);
    assert_eq!(first, Ok(true));

    let second = validate_synthetic_header_metadata(vec![
        header_tensor(
            "layers.14.engram.embed.weight",
            "F8_E4M3",
            &[384_016_682, 256],
            98_308_270_592,
        ),
        header_tensor(
            "layers.14.engram.embed.scale",
            "F8_E8M0",
            &[384_016_682, 8],
            3_072_133_456,
        ),
    ]);
    assert_eq!(second, Ok(true));

    let uncovered = validate_synthetic_header_metadata(vec![header_tensor(
        "layers.0.attn.attn_sink",
        "F32",
        &[64],
        256,
    )]);
    assert_eq!(uncovered, Ok(false));
}

#[test]
fn synthetic_header_contract_rejects_extent_overlap_and_shared_fp4() {
    let overlap = validate_synthetic_headers_with_offsets(vec![
        header_tensor_at("layers.0.attn.wq_a.weight", "F8_E4M3", &[64, 64], 4096, 0),
        header_tensor_at("layers.0.attn.wq_a.scale", "F8_E8M0", &[2, 2], 4, 4095),
    ]);
    let error = overlap.unwrap_err();
    assert!(error.contains("overlapping tensor extents"), "{error}");

    let shared_fp4 = validate_synthetic_headers(vec![
        header_tensor(
            "layers.0.ffn.shared_experts.w1.weight",
            "I8",
            &[64, 16],
            1024,
        ),
        header_tensor(
            "layers.0.ffn.shared_experts.w1.scale",
            "F8_E8M0",
            &[64, 1],
            64,
        ),
    ]);
    assert!(shared_fp4.is_err());

    let wrong_engram = validate_synthetic_header_metadata(vec![
        header_tensor(
            "layers.1.engram.embed.weight",
            "F8_E4M3",
            &[384_006_168, 255],
            97_921_573_840,
        ),
        header_tensor(
            "layers.1.engram.embed.scale",
            "F8_E8M0",
            &[384_006_168, 8],
            3_072_049_344,
        ),
    ]);
    assert!(wrong_engram.is_err());

    let wrong_engram_dtype = validate_synthetic_header_metadata(vec![
        header_tensor(
            "layers.14.engram.embed.weight",
            "I8",
            &[384_016_682, 256],
            98_308_270_592,
        ),
        header_tensor(
            "layers.14.engram.embed.scale",
            "F8_E8M0",
            &[384_016_682, 8],
            3_072_133_456,
        ),
    ]);
    assert!(wrong_engram_dtype.is_err());
}

#[test]
fn synthetic_header_contract_rejects_dtype_shape_and_pairing_errors() {
    for tensors in [
        vec![
            header_tensor("layers.0.attn.wq_a.weight", "F8_E4M3", &[64, 64], 4096),
            header_tensor("layers.0.attn.wq_a.scale", "BF16", &[2, 2], 8),
        ],
        vec![
            header_tensor("layers.0.attn.wq_a.weight", "BF16", &[64, 64], 8192),
            header_tensor("layers.0.attn.wq_a.scale", "F8_E8M0", &[2, 2], 4),
        ],
        vec![
            header_tensor("layers.0.attn.wq_a.weight", "F8_E4M3", &[64, 64], 4096),
            header_tensor("layers.0.attn.wq_a.scale", "F8_E8M0", &[1, 2], 2),
        ],
        vec![header_tensor(
            "layers.0.attn.wq_a.weight",
            "F8_E4M3",
            &[64, 64],
            4096,
        )],
        vec![
            header_tensor("layers.0.ffn.experts.0.w1.weight", "I8", &[64, 15], 960),
            header_tensor("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[64, 1], 64),
        ],
        vec![header_tensor(
            "layers.0.attn.wq_a.weight",
            "UNKNOWN",
            &[64, 64],
            4096,
        )],
    ] {
        assert!(validate_synthetic_headers(tensors).is_err());
    }
}

#[test]
fn synthetic_profile_cannot_be_used_as_real_checkpoint_contract() {
    let error = validate_real_index(&config(), &fixture::fixture_index())
        .unwrap_err()
        .to_string();
    assert!(error.contains("48 shards"), "{error}");
}

#[test]
fn valid_dynamic_schema_has_required_roles_and_scale_pairs() {
    let index = fixture::fixture_index();
    let summary = validate_index(&config(), &index).unwrap();
    assert_eq!(summary.tensor_count, index.tensor_count());
    assert_eq!(summary.shard_count, 1);
    assert!(summary.scale_pair_count > 40 * 384 * 3);
    assert!(!summary.headers_validated);
}

#[test]
fn missing_scale_pair_is_reported_as_a_schema_error() {
    let mut index = fixture::fixture_index();
    index.weight_map.remove("layers.0.attn.wq_a.scale").unwrap();
    let error = validate_index(&config(), &index).unwrap_err().to_string();
    assert!(
        error.contains("missing scale") || error.contains("missing required tensor"),
        "{error}"
    );
}

#[test]
fn source_layer_role_sets_are_exact_and_layer_specific() {
    let mut missing = fixture::fixture_index();
    missing
        .weight_map
        .remove("layers.2.attn.indexer.k_norm.weight")
        .unwrap();
    let error = validate_index(&config(), &missing).unwrap_err().to_string();
    assert!(
        error.contains("tensor role set") && error.contains("k_norm.weight"),
        "{error}"
    );

    let mut non_source = fixture::fixture_index();
    non_source.weight_map.insert(
        "layers.3.attn.compressor.norm.weight".into(),
        "model-00001-of-00001.safetensors".into(),
    );
    let error = validate_index(&config(), &non_source)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("tensor role set") && error.contains("compressor"),
        "{error}"
    );

    let mut wrong_layer_twenty = fixture::fixture_index();
    wrong_layer_twenty.weight_map.insert(
        "layers.20.attn.compressor.wgate.weight".into(),
        "model-00001-of-00001.safetensors".into(),
    );
    let error = validate_index(&config(), &wrong_layer_twenty)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("tensor role set") && error.contains("wgate.weight"),
        "{error}"
    );
}

#[test]
fn unknown_tensor_name_is_not_ignored() {
    let mut index = fixture::fixture_index();
    index.weight_map.insert(
        "layers.0.ffn.unknown.weight".into(),
        "model-00001-of-00001.safetensors".into(),
    );
    let error = validate_index(&config(), &index).unwrap_err().to_string();
    assert!(error.contains("unknown or unsupported"), "{error}");
}

#[test]
fn invalid_weight_map_and_shard_identity_fail_closed() {
    let empty = HfSafetensorsIndex {
        total_size: Some(1),
        weight_map: Default::default(),
    };
    assert!(validate_index(&config(), &empty).is_err());

    let mut bad_shard = fixture::fixture_index();
    *bad_shard.weight_map.values_mut().next().unwrap() = "../escape.safetensors".into();
    assert!(validate_index(&config(), &bad_shard).is_err());

    let mut non_contiguous = fixture::fixture_index();
    *non_contiguous.weight_map.values_mut().next().unwrap() =
        "model-00002-of-00002.safetensors".into();
    assert!(validate_index(&config(), &non_contiguous).is_err());
}

#[test]
fn absurd_shard_totals_are_rejected_before_allocation() {
    for shard in [
        "model-00001-of-999999999999.safetensors",
        "model-00001-of-999999999999999999999999.safetensors",
    ] {
        let mut index = fixture::fixture_index();
        *index.weight_map.values_mut().next().unwrap() = shard.into();
        let error = validate_index(&config(), &index).unwrap_err().to_string();
        assert!(
            error.contains("total") || error.contains("overflow"),
            "{error}"
        );
    }
}

#[test]
fn manually_constructed_inconsistent_config_cannot_enter_index_paths() {
    let mut config = config();
    config.text.compress_ratios.clear();
    let index = fixture::fixture_index();
    assert!(validate_index(&config, &index).is_err());
    assert!(
        ferrule_model::models::deepseek_v41::DeepSeekV41Metadata::from_config_and_index(
            Path::new("no-payload-fixture"),
            config,
            index,
        )
        .is_err()
    );
}

#[test]
fn exact_v41_semantics_and_optional_vision_ratio_fail_closed() {
    for (field, value) in [
        ("hc_eps", json!(2e-6)),
        ("dspark_noise_token_id", json!(128798)),
    ] {
        let mut changed = fixture::config_value();
        changed["text_config"][field] = value;
        assert!(DeepSeekV41Config::from_value(&changed).is_err(), "{field}");
    }

    let mut changed = fixture::config_value();
    changed["text_config"]["compress_ratios"][20] = 2.into();
    assert!(DeepSeekV41Config::from_value(&changed).is_err());

    let mut changed = fixture::config_value();
    changed["text_config"]["engram_num_embeddings"][0] = 384006169.into();
    assert!(DeepSeekV41Config::from_value(&changed).is_err());

    let mut positive = fixture::config_value();
    positive["vision_config"]["max_wh_ratio"] = 2.5.into();
    assert!(DeepSeekV41Config::from_value(&positive).is_ok());
    for value in [json!(0.0), json!(-1.0), json!(1e100)] {
        let mut changed = fixture::config_value();
        changed["vision_config"]["max_wh_ratio"] = value;
        assert!(DeepSeekV41Config::from_value(&changed).is_err());
    }
    let mut wrong_type = fixture::config_value();
    wrong_type["vision_config"]["max_wh_ratio"] = "2.5".into();
    assert!(DeepSeekV41Config::from_value(&wrong_type).is_err());

    let mut direct = config();
    direct.vision.as_mut().unwrap().max_wh_ratio = Some(f32::NAN);
    assert!(direct.validate().is_err());
    direct.vision.as_mut().unwrap().max_wh_ratio = Some(f32::INFINITY);
    assert!(direct.validate().is_err());
}

#[test]
fn configuration_errors_are_not_hidden_by_index_validation() {
    let mut value: Value = fixture::config_value();
    value["text_config"]["dspark_n_routed_experts"] = 384.into();
    assert!(DeepSeekV41Config::from_value(&value).is_err());
    let mut value = fixture::config_value();
    value["quantization_config"]["scale_fmt"] = "e8m0".into();
    assert!(DeepSeekV41Config::from_value(&value).is_err());
    assert!(Path::new("no-payload-fixture").is_relative());
}
