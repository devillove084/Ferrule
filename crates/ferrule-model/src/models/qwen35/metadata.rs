//! Header-only artifact validation. No tensor payloads or tokenizer are loaded.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use ferrule_common::Result;

use crate::{
    AttentionKind, HfSafetensorsIndex, HfSafetensorsInventory, ModelDescriptor, ModelFamily,
    MoeSpec, QuantFormatCount, RouterKind, TransformerSemantics, TransformerSpec, WeightSource,
};

use super::{Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartition, model_error};

fn validate_shard_extents(model_dir: &Path, inventory: &HfSafetensorsInventory) -> Result<()> {
    let mut shards = BTreeMap::<&str, Vec<&crate::HfSafetensorsTensorInfo>>::new();
    for tensor in &inventory.tensors {
        shards.entry(&tensor.shard).or_default().push(tensor);
    }
    for (shard, mut tensors) in shards {
        let mut file = std::fs::File::open(model_dir.join(shard))?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let data_start = u64::from_le_bytes(length)
            .checked_add(8)
            .ok_or_else(|| model_error("safetensors header length overflow"))?;
        // Unlike the lightweight inventory's alignment rounding, safetensors
        // offsets are relative to exactly 8 + the stored header length.
        tensors.sort_by_key(|tensor| tensor.data_offset);
        let mut offset = 0u64;
        for tensor in tensors {
            let expected_file_offset = data_start
                .checked_add(offset)
                .ok_or_else(|| model_error("tensor file offset overflow"))?;
            if tensor.data_offset != offset || tensor.file_offset != expected_file_offset {
                return Err(model_error(format!(
                    "non-contiguous or overlapping tensor extent for '{}'",
                    tensor.name
                )));
            }
            offset = offset
                .checked_add(tensor.byte_size)
                .ok_or_else(|| model_error("tensor extent overflow"))?;
        }
        let end = data_start
            .checked_add(offset)
            .ok_or_else(|| model_error("shard extent overflow"))?;
        if end != file.metadata()?.len() {
            return Err(model_error(format!(
                "shard '{shard}' length differs from header extents"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Qwen35Metadata {
    path: PathBuf,
    config: Qwen35Config,
    mapper: Qwen35HfNameMapper,
    inventory: HfSafetensorsInventory,
    partition: Qwen35TensorPartition,
}

impl Qwen35Metadata {
    pub fn open_hf(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)?;
        let value = serde_json::from_str(&text)
            .map_err(|error| model_error(format!("config '{}': {error}", path.display())))?;
        Self::from_value(model_dir, &value)
    }

    pub(crate) fn from_value(model_dir: &Path, value: &serde_json::Value) -> Result<Self> {
        // Reject unsupported profiles before opening even their shard headers.
        Self::from_config(model_dir, Qwen35Config::from_value(value)?)
    }

    pub(super) fn from_config(model_dir: &Path, config: Qwen35Config) -> Result<Self> {
        let mapper = Qwen35HfNameMapper::new(&config);
        let index_path = model_dir.join("model.safetensors.index.json");
        let index = match std::fs::symlink_metadata(&index_path) {
            Ok(_) => Some(HfSafetensorsIndex::open(&index_path)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if let Some(index) = &index {
            for shard in index.weight_map.values() {
                let mut parts = Path::new(shard).components();
                if !matches!(parts.next(), Some(Component::Normal(_)))
                    || parts.next().is_some()
                    || !shard.ends_with(".safetensors")
                    || shard.contains('\\')
                {
                    return Err(model_error(format!("invalid shard filename '{shard}'")));
                }
            }
        }
        let inventory = match &index {
            Some(index) => {
                HfSafetensorsInventory::from_index(model_dir, ModelFamily::Qwen35, index)?
            }
            None => HfSafetensorsInventory::from_single_file(model_dir, ModelFamily::Qwen35)?,
        };
        if let Some(index) = &index {
            for tensor in &inventory.tensors {
                if index.weight_map.get(&tensor.name) != Some(&tensor.shard) {
                    return Err(model_error(format!(
                        "index/header shard mismatch for '{}'",
                        tensor.name
                    )));
                }
            }
            if let Some(total) = index.total_size {
                let actual = inventory
                    .tensors
                    .iter()
                    .try_fold(0u64, |sum, t| sum.checked_add(t.byte_size))
                    .ok_or_else(|| model_error("total tensor bytes overflow"))?;
                if total != actual {
                    return Err(model_error(format!(
                        "index total_size {total} differs from header bytes {actual}"
                    )));
                }
            }
        }
        let partition = mapper.validate_inventory(&inventory)?;
        validate_shard_extents(model_dir, &inventory)?;
        Ok(Self {
            path: model_dir.to_path_buf(),
            config,
            mapper,
            inventory,
            partition,
        })
    }

    pub const fn config(&self) -> &Qwen35Config {
        &self.config
    }
    pub const fn name_mapper(&self) -> &Qwen35HfNameMapper {
        &self.mapper
    }
    pub const fn inventory(&self) -> &HfSafetensorsInventory {
        &self.inventory
    }
    pub const fn partition(&self) -> &Qwen35TensorPartition {
        &self.partition
    }

    /// Strict text-only runtime descriptor; semantic roles live in Qwen35Recipe.
    pub fn descriptor(&self) -> ModelDescriptor {
        let t = self.config.text();
        ModelDescriptor {
            path: self.path.clone(),
            spec: TransformerSpec {
                family: ModelFamily::Qwen35,
                architecture: Some(self.config.architecture().to_owned()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(t.hidden_size),
                num_layers: Some(t.num_hidden_layers),
                vocab_size: Some(t.vocab_size),
                num_heads: Some(t.num_attention_heads),
                num_kv_heads: Some(t.num_key_value_heads),
                head_dim: Some(t.head_dim),
                attention: AttentionKind::Unknown("qwen35_hybrid_linear_full".into()),
                moe: MoeSpec {
                    num_experts: None, num_experts_per_tok: None,
                    has_shared_experts: false, router: RouterKind::None,
                },
                semantics: TransformerSemantics {
                    norm_epsilon: Some(t.rms_norm_eps),
                    rope_theta: Some(t.rope_parameters.rope_theta),
                    rope_head_dim: Some(self.config.semantics().rotary_dimensions),
                    ..TransformerSemantics::default()
                },
                tensor_count: Some(self.inventory.tensor_count),
                quantization: self.inventory.dtype_counts.iter().map(|item| QuantFormatCount {
                    format: item.dtype.clone(), tensors: item.tensors,
                }).collect(),
                notes: vec![
                    "Qwen3.5-0.8B BF16 storage; F32 CPU/CUDA text-only resident execution; no prefix cache, speculation, partial retain or TP/PP".into(),
                    format!("validated text-only partition: {} text, {} visual, {} MTP tensors; attachments are not executed", self.partition.text().len(), self.partition.visual().len(), self.partition.mtp().len()),
                    "18 linear + 6 full layers; per-head sigmoid output gate, offset RMSNorm, 64-dimension partial RoPE, tied output head".into(),
                ],
            },
            tensor_classes: Vec::new(),
        }
    }
}
