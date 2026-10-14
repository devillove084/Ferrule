//! Identical metadata contracts in default and CUDA builds; no GPU or payload reads.
use ferrule_composition::model_factory::Qwen35MoeCapacityLimits;
use ferrule_model::models::qwen35::Qwen35Config;

fn config() -> Qwen35Config {
    Qwen35Config::from_value(
        &serde_json::from_str(include_str!(
            "../../ferrule-model/tests/qwen35_35b_fp8_config.json"
        ))
        .unwrap(),
    )
    .unwrap()
}

#[path = "qwen35_memory_plan/metadata.rs"]
mod metadata;

#[path = "qwen35_memory_plan/mixed_attention.rs"]
mod mixed_attention;
