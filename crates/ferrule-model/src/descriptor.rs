use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferrule_core::{Error, Result};
use ferrule_gguf::{GgufFile, GgufValue};

use crate::artifact::{HfSafetensorsIndex, HfSafetensorsInventory};
use crate::config::OlmoeConfig;
use crate::families;
use crate::spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSpec,
    WeightSource,
};
use crate::support::{EnginePlan, ModelSupportContract};
use crate::tensor_policy::{GgufTensorPolicy, HfTensorPolicy, TensorClassCount};

/// Lightweight model descriptor used for inspection and runtime dispatch.
///
/// This intentionally does not load large tensor payloads. It is the boundary
/// between model-family detection and concrete execution backends.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    pub path: PathBuf,
    pub spec: TransformerSpec,
    pub tensor_classes: Vec<TensorClassCount>,
}

impl ModelDescriptor {
    pub fn load(path: &Path) -> Result<Self> {
        if path.is_file() {
            if is_gguf_file(path) {
                return Self::from_gguf(path);
            }
            return Err(Error::Model(format!(
                "unsupported model file '{}'; expected .gguf or a model directory",
                path.display()
            )));
        }

        if !path.is_dir() {
            return Err(Error::Model(format!(
                "model path not found: {}",
                path.display()
            )));
        }

        let config = path.join("config.json");
        if config.exists() {
            return Self::from_hf_dir(path);
        }

        if let Some(gguf) = find_single_gguf(path)? {
            return Self::from_gguf(&gguf);
        }

        Err(Error::Model(format!(
            "model directory '{}' has neither config.json nor a .gguf file",
            path.display()
        )))
    }

    /// Build the generic model support contract used by runtime planning.
    ///
    /// This keeps model-specific tensor names at the descriptor/binding boundary.
    /// Generic executors should consume semantic roles and policies from the
    /// returned contract instead of matching on source tensor names.
    pub fn support_contract(&self) -> ModelSupportContract {
        ModelSupportContract::from_spec(&self.spec, &self.tensor_classes)
    }

    /// Build the current execution plan for this model descriptor.
    pub fn engine_plan(&self) -> EnginePlan {
        self.support_contract().engine_plan()
    }

    fn from_hf_dir(model_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| Error::Model(format!("config: {e}")))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::Model(format!("json: {e}")))?;

        let architecture = hf_architecture(&json);
        let family = architecture
            .as_deref()
            .map(ModelFamily::from_architecture)
            .unwrap_or_else(|| ModelFamily::Unknown("hf-transformer".into()));

        let mut spec = if matches!(family, ModelFamily::Olmoe) {
            let config = OlmoeConfig::from_json(&json)?;
            TransformerSpec::from_olmoe_config(&config, WeightSource::Safetensors)
        } else {
            generic_hf_spec(&json, family, architecture.clone())
        };
        if spec.architecture.is_none() {
            spec.architecture = architecture;
        }
        spec.weight_source = WeightSource::Safetensors;

        let mut tensor_classes = Vec::new();
        if let Some(index) = read_hf_safetensors_index(model_dir)? {
            spec.tensor_count = Some(index.tensor_count());
            let policy = HfTensorPolicy::for_family(spec.family.clone());
            tensor_classes = policy.summarize(index.tensor_names());
            spec.notes.push(format!(
                "HF safetensors index: {} tensors across {} shards{}",
                index.tensor_count(),
                index.shard_count(),
                index
                    .total_size
                    .map(|bytes| format!(", total_size={} bytes", bytes))
                    .unwrap_or_default()
            ));
            let missing = index.missing_shards(model_dir);
            if missing.is_empty() {
                spec.notes
                    .push("all safetensors shards referenced by index are present".into());
                match HfSafetensorsInventory::from_index(model_dir, spec.family.clone(), &index) {
                    Ok(inventory) => {
                        spec.quantization = inventory
                            .dtype_counts
                            .iter()
                            .map(|item| QuantFormatCount {
                                format: item.dtype.clone(),
                                tensors: item.tensors,
                            })
                            .collect();
                        tensor_classes = inventory.class_counts;
                        spec.notes.push(format!(
                            "HF safetensors header inventory: {} tensors, {} dtype classes, {} role classes",
                            inventory.tensor_count,
                            inventory.dtype_counts.len(),
                            inventory.role_counts.len()
                        ));
                    }
                    Err(err) => spec.notes.push(format!(
                        "HF safetensors header inventory unavailable: {err}"
                    )),
                }
            } else {
                spec.notes.push(format!(
                    "missing {} safetensors shards referenced by index",
                    missing.len()
                ));
            }
        }
        families::refine_hf_spec(&mut spec, &json);

        Ok(Self {
            path: model_dir.to_path_buf(),
            spec,
            tensor_classes,
        })
    }

    fn from_gguf(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let architecture = gguf.architecture().map(ToOwned::to_owned);
        let family = architecture
            .as_deref()
            .map(ModelFamily::from_architecture)
            .unwrap_or_else(|| ModelFamily::Unknown("gguf".into()));

        let num_heads = gguf.num_heads().map(|v| v as usize);
        let num_kv_heads = gguf.num_kv_heads().map(|v| v as usize);
        let hidden_size = gguf.hidden_size().map(|v| v as usize);
        let head_dim = hidden_size
            .zip(num_heads)
            .and_then(|(h, n)| h.checked_div(n));
        let has_mla_tensors = families::has_mla_gguf_tensor_names(
            &family,
            gguf.tensors.iter().map(|t| t.name.as_str()),
        );

        let attention = if matches!(family, ModelFamily::DeepSeekV4) || has_mla_tensors {
            AttentionKind::MultiLatentAttention
        } else if matches!((num_heads, num_kv_heads), (Some(nh), Some(nkv)) if nkv < nh) {
            AttentionKind::GroupedQuery
        } else {
            AttentionKind::DenseMha
        };

        let has_shared_experts = gguf.tensors.iter().any(|t| t.name.contains("_shexp"));
        let has_hash_router = gguf
            .tensors
            .iter()
            .any(|t| t.name.contains("ffn_gate_tid2eid"));
        let has_routed_experts = gguf.tensors.iter().any(|t| {
            t.name.contains("ffn_gate_exps")
                || t.name.contains("ffn_up_exps")
                || t.name.contains("ffn_down_exps")
        });
        let router = if has_hash_router {
            RouterKind::HashAssistedTopK
        } else if has_routed_experts || gguf.expert_count().unwrap_or(0) > 0 {
            RouterKind::DenseTopK
        } else {
            RouterKind::None
        };

        let policy = GgufTensorPolicy::for_family(family.clone());
        let tensor_classes = policy.summarize(gguf.tensors.iter());

        let mut notes = Vec::new();
        families::append_gguf_notes(&family, &tensor_classes, &mut notes);
        if has_shared_experts {
            notes.push("shared experts detected".into());
        }
        if has_hash_router {
            notes.push("hash-routing tables detected".into());
        }

        let spec = TransformerSpec {
            family,
            architecture,
            weight_source: WeightSource::Gguf,
            hidden_size,
            num_layers: gguf.num_layers().map(|v| v as usize),
            vocab_size: gguf.vocab_size().map(|v| v as usize),
            num_heads,
            num_kv_heads,
            head_dim,
            attention,
            moe: MoeSpec {
                num_experts: gguf.expert_count().map(|v| v as usize),
                num_experts_per_tok: gguf.expert_used_count().map(|v| v as usize),
                has_shared_experts,
                router,
            },
            tensor_count: Some(gguf.tensors.len()),
            quantization: quant_counts(&gguf),
            notes,
        };

        Ok(Self {
            path: path.to_path_buf(),
            spec,
            tensor_classes,
        })
    }
}

fn generic_hf_spec(
    json: &serde_json::Value,
    family: ModelFamily,
    architecture: Option<String>,
) -> TransformerSpec {
    let hidden_size = usize_key(json, &["hidden_size", "n_embd", "d_model"]);
    let num_layers = usize_key(json, &["num_hidden_layers", "n_layer", "num_layers"]);
    let vocab_size = usize_key(json, &["vocab_size"]);
    let num_heads = usize_key(json, &["num_attention_heads", "n_head"]);
    let num_kv_heads = usize_key(json, &["num_key_value_heads", "num_kv_heads"]);
    let head_dim = usize_key(json, &["head_dim"]).or_else(|| {
        hidden_size
            .zip(num_heads)
            .and_then(|(hidden, heads)| hidden.checked_div(heads))
    });

    let attention = if has_any_key(json, &["kv_lora_rank", "q_lora_rank"]) {
        AttentionKind::MultiLatentAttention
    } else if matches!((num_heads, num_kv_heads), (Some(nh), Some(nkv)) if nkv < nh) {
        AttentionKind::GroupedQuery
    } else {
        AttentionKind::DenseMha
    };

    let num_experts = usize_key(
        json,
        &["num_experts", "n_routed_experts", "moe_num_experts"],
    );
    let num_experts_per_tok = usize_key(
        json,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "moe_top_k",
            "top_k",
        ],
    );
    let has_shared_experts = has_any_key(
        json,
        &[
            "n_shared_experts",
            "num_shared_experts",
            "shared_expert_intermediate_size",
        ],
    );
    let router = if usize_key(json, &["num_hash_layers"]).unwrap_or(0) > 0 {
        RouterKind::HashAssistedTopK
    } else if num_experts.unwrap_or(0) > 0 {
        RouterKind::DenseTopK
    } else {
        RouterKind::None
    };

    TransformerSpec {
        family,
        architecture,
        weight_source: WeightSource::Safetensors,
        hidden_size,
        num_layers,
        vocab_size,
        num_heads,
        num_kv_heads,
        head_dim,
        attention,
        moe: MoeSpec {
            num_experts,
            num_experts_per_tok,
            has_shared_experts,
            router,
        },
        tensor_count: None,
        quantization: Vec::new(),
        notes: Vec::new(),
    }
}

fn hf_architecture(json: &serde_json::Value) -> Option<String> {
    json.get("model_type")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            json.get("architectures")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned)
        })
}

fn usize_key(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| json.get(*key).and_then(|v| v.as_u64()).map(|v| v as usize))
}

fn has_any_key(json: &serde_json::Value, keys: &[&str]) -> bool {
    keys.iter()
        .any(|key| !json.get(*key).unwrap_or(&serde_json::Value::Null).is_null())
}

fn read_hf_safetensors_index(model_dir: &Path) -> Result<Option<HfSafetensorsIndex>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        Ok(Some(HfSafetensorsIndex::open(index_path)?))
    } else {
        Ok(None)
    }
}

fn quant_counts(gguf: &GgufFile) -> Vec<QuantFormatCount> {
    let mut counts = BTreeMap::<String, usize>::new();
    for tensor in &gguf.tensors {
        *counts
            .entry(format!("{:?}", tensor.quant_type))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(format, tensors)| QuantFormatCount { format, tensors })
        .collect()
}

fn is_gguf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

fn find_single_gguf(dir: &Path) -> Result<Option<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if is_gguf_file(&path) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files.into_iter().next())
}

#[allow(dead_code)]
fn gguf_array_len(value: Option<&GgufValue>) -> Option<usize> {
    match value {
        Some(GgufValue::Array(arr)) => Some(arr.len()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_architecture_prefers_model_type() {
        let json = serde_json::json!({
            "model_type": "olmoe",
            "architectures": ["Other"]
        });
        assert_eq!(hf_architecture(&json).as_deref(), Some("olmoe"));
    }

    #[test]
    fn generic_deepseek_hf_spec_detects_mla() {
        let json = serde_json::json!({
            "model_type": "deepseek4",
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "kv_lora_rank": 512,
            "n_routed_experts": 256,
            "moe_top_k": 8,
            "n_shared_experts": 1
        });
        let spec = generic_hf_spec(
            &json,
            ModelFamily::from_architecture("deepseek4"),
            Some("deepseek4".into()),
        );
        assert_eq!(spec.family, ModelFamily::DeepSeekV4);
        assert_eq!(spec.attention, AttentionKind::MultiLatentAttention);
        assert_eq!(spec.moe.num_experts, Some(256));
        assert!(spec.moe.has_shared_experts);
    }
}
