use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ferrule_common::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSafetensorsIndex {
    pub total_size: Option<u64>,
    pub weight_map: BTreeMap<String, String>,
}

impl HfSafetensorsIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Model(format!("safetensors index json: {e}")))?;
        let total_size = json
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(|value| value.as_u64());
        let weight_map = json
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| Error::Model("safetensors index missing weight_map".into()))?
            .iter()
            .map(|(tensor, shard)| (tensor.clone(), shard.as_str().unwrap_or("").to_string()))
            .collect();
        Ok(Self {
            total_size,
            weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_names(&self) -> Vec<String> {
        self.weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn shard_count(&self) -> usize {
        self.shard_names().len()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn missing_shards(&self, model_dir: &Path) -> Vec<PathBuf> {
        self.shard_names()
            .into_iter()
            .map(|name| model_dir.join(name))
            .filter(|path| !path.exists())
            .collect()
    }

    pub fn tensors_in_shard(&self, shard: &str) -> Vec<&str> {
        self.weight_map
            .iter()
            .filter_map(|(tensor, mapped_shard)| {
                if mapped_shard == shard {
                    Some(tensor.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hf_safetensors_index_summary() {
        let index = HfSafetensorsIndex::from_json_str(
            r#"{
              "metadata": { "total_size": 1234 },
              "weight_map": {
                "embed.weight": "model-00001-of-00002.safetensors",
                "layers.0.attn.wq_a.weight": "model-00002-of-00002.safetensors",
                "layers.0.attn.wq_a.scale": "model-00002-of-00002.safetensors"
              }
            }"#,
        )
        .unwrap();
        assert_eq!(index.total_size, Some(1234));
        assert_eq!(index.tensor_count(), 3);
        assert_eq!(index.shard_count(), 2);
        assert_eq!(
            index.tensors_in_shard("model-00002-of-00002.safetensors"),
            vec!["layers.0.attn.wq_a.scale", "layers.0.attn.wq_a.weight"]
        );
    }
}
