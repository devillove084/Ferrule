use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use ferrule_common::{Error, Result};

use crate::families::{
    self, AttentionTensorRef, DenseLayerTensorRef, HyperConnectionTensorRef, RoutedExpertTensorRef,
    RouterTensorRef, SharedExpertTensorRef,
};
use crate::spec::ModelFamily;
use crate::support::{tensor_role_for_class, TensorRole};
use crate::tensor_policy::{HfTensorPolicy, TensorClass, TensorClassCount};

use super::index::HfSafetensorsIndex;

const MAX_SAFETENSORS_HEADER_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSafetensorsTensorInfo {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Offset from the safetensors data section, as recorded in the header.
    pub data_offset: u64,
    /// Absolute byte offset in the shard file. Streaming readers should use this.
    pub file_offset: u64,
    pub byte_size: u64,
    pub class: TensorClass,
    pub role: TensorRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRoutedExpertTensorInfo {
    pub descriptor: RoutedExpertTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSharedExpertTensorInfo {
    pub descriptor: SharedExpertTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRouterTensorInfo {
    pub descriptor: RouterTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfAttentionTensorInfo {
    pub descriptor: AttentionTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfDenseLayerTensorInfo {
    pub descriptor: DenseLayerTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfHyperConnectionTensorInfo {
    pub descriptor: HyperConnectionTensorRef,
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub file_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtypeCount {
    pub dtype: String,
    pub tensors: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRoleCount {
    pub role: TensorRole,
    pub tensors: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSafetensorsShardSummary {
    pub shard: String,
    pub tensors: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSafetensorsInventory {
    pub family: ModelFamily,
    pub total_size: Option<u64>,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub tensors: Vec<HfSafetensorsTensorInfo>,
    pub dtype_counts: Vec<DtypeCount>,
    pub class_counts: Vec<TensorClassCount>,
    pub role_counts: Vec<TensorRoleCount>,
    pub shard_summaries: Vec<HfSafetensorsShardSummary>,
    pub index_only_tensors: Vec<String>,
    pub header_only_tensors: Vec<String>,
}

impl HfSafetensorsInventory {
    pub fn open(model_dir: impl AsRef<Path>, family: ModelFamily) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index = HfSafetensorsIndex::open(model_dir.join("model.safetensors.index.json"))?;
        Self::from_index(model_dir, family, &index)
    }

    pub fn from_index(
        model_dir: &Path,
        family: ModelFamily,
        index: &HfSafetensorsIndex,
    ) -> Result<Self> {
        let policy = HfTensorPolicy::for_family(family.clone());
        let mut tensors = Vec::with_capacity(index.tensor_count());
        let mut shard_summaries = Vec::new();
        let mut header_tensor_to_shard = BTreeMap::<String, String>::new();

        for shard in index.shard_names() {
            let header = read_safetensors_shard_header(model_dir.join(&shard))?;
            let mut shard_bytes = 0u64;
            let mut shard_tensors = 0usize;
            for tensor in header.tensors {
                let class = policy.classify_name(&tensor.name);
                let role = tensor_role_for_class(&class);
                let file_offset = header.data_start.saturating_add(tensor.data_offset);
                shard_bytes = shard_bytes.saturating_add(tensor.byte_size);
                shard_tensors += 1;
                header_tensor_to_shard.insert(tensor.name.clone(), shard.clone());
                tensors.push(HfSafetensorsTensorInfo {
                    name: tensor.name,
                    shard: shard.clone(),
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    data_offset: tensor.data_offset,
                    file_offset,
                    byte_size: tensor.byte_size,
                    class,
                    role,
                });
            }
            shard_summaries.push(HfSafetensorsShardSummary {
                shard,
                tensors: shard_tensors,
                bytes: shard_bytes,
            });
        }

        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        shard_summaries.sort_by(|a, b| a.shard.cmp(&b.shard));

        let mut dtype_map = BTreeMap::<String, (usize, u64)>::new();
        let mut class_map = BTreeMap::<TensorClass, usize>::new();
        let mut role_map = BTreeMap::<TensorRole, (usize, u64)>::new();
        for tensor in &tensors {
            let dtype_entry = dtype_map.entry(tensor.dtype.clone()).or_default();
            dtype_entry.0 += 1;
            dtype_entry.1 = dtype_entry.1.saturating_add(tensor.byte_size);
            *class_map.entry(tensor.class.clone()).or_default() += 1;
            let role_entry = role_map.entry(tensor.role.clone()).or_default();
            role_entry.0 += 1;
            role_entry.1 = role_entry.1.saturating_add(tensor.byte_size);
        }

        let dtype_counts = dtype_map
            .into_iter()
            .map(|(dtype, (tensors, bytes))| DtypeCount {
                dtype,
                tensors,
                bytes,
            })
            .collect();
        let class_counts = class_map
            .into_iter()
            .map(|(class, tensors)| TensorClassCount { class, tensors })
            .collect();
        let role_counts = role_map
            .into_iter()
            .map(|(role, (tensors, bytes))| TensorRoleCount {
                role,
                tensors,
                bytes,
            })
            .collect();

        let index_only_tensors = index
            .tensor_names()
            .filter(|name| !header_tensor_to_shard.contains_key(*name))
            .map(ToOwned::to_owned)
            .collect();
        let header_only_tensors = header_tensor_to_shard
            .keys()
            .filter(|name| !index.weight_map.contains_key(*name))
            .cloned()
            .collect();

        Ok(Self {
            family,
            total_size: index.total_size,
            shard_count: index.shard_count(),
            tensor_count: tensors.len(),
            tensors,
            dtype_counts,
            class_counts,
            role_counts,
            shard_summaries,
            index_only_tensors,
            header_only_tensors,
        })
    }

    pub fn dtype_bytes(&self, dtype: &str) -> u64 {
        self.dtype_counts
            .iter()
            .find(|item| item.dtype == dtype)
            .map(|item| item.bytes)
            .unwrap_or(0)
    }

    pub fn role_bytes(&self, role: &TensorRole) -> u64 {
        self.role_counts
            .iter()
            .find(|item| &item.role == role)
            .map(|item| item.bytes)
            .unwrap_or(0)
    }

    pub fn class_count(&self, class: &TensorClass) -> usize {
        self.class_counts
            .iter()
            .find(|item| &item.class == class)
            .map(|item| item.tensors)
            .unwrap_or(0)
    }

    pub fn routed_expert_tensors(&self) -> Vec<HfRoutedExpertTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_routed_expert_tensor(&self.family, &tensor.name).map(
                    |descriptor| HfRoutedExpertTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    },
                )
            })
            .collect()
    }

    pub fn shared_expert_tensors(&self) -> Vec<HfSharedExpertTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_shared_expert_tensor(&self.family, &tensor.name).map(
                    |descriptor| HfSharedExpertTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    },
                )
            })
            .collect()
    }

    pub fn router_tensors(&self) -> Vec<HfRouterTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_router_tensor(&self.family, &tensor.name).map(|descriptor| {
                    HfRouterTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    }
                })
            })
            .collect()
    }

    pub fn attention_tensors(&self) -> Vec<HfAttentionTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_attention_tensor(&self.family, &tensor.name).map(|descriptor| {
                    HfAttentionTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    }
                })
            })
            .collect()
    }

    pub fn dense_layer_tensors(&self) -> Vec<HfDenseLayerTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_dense_layer_tensor(&self.family, &tensor.name).map(
                    |descriptor| HfDenseLayerTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    },
                )
            })
            .collect()
    }

    pub fn hyper_connection_tensors(&self) -> Vec<HfHyperConnectionTensorInfo> {
        self.tensors
            .iter()
            .filter_map(|tensor| {
                families::parse_hf_hyper_connection_tensor(&self.family, &tensor.name).map(
                    |descriptor| HfHyperConnectionTensorInfo {
                        descriptor,
                        name: tensor.name.clone(),
                        shard: tensor.shard.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        data_offset: tensor.data_offset,
                        file_offset: tensor.file_offset,
                        byte_size: tensor.byte_size,
                    },
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardHeaderTensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data_offset: u64,
    byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardHeader {
    data_start: u64,
    tensors: Vec<ShardHeaderTensor>,
}

fn read_safetensors_shard_header(path: impl AsRef<Path>) -> Result<ShardHeader> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .map_err(|e| Error::Model(format!("safetensors shard open '{}': {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        Error::Model(format!(
            "safetensors shard header length '{}': {e}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(Error::Model(format!(
            "safetensors shard header too large in '{}': {} bytes",
            path.display(),
            header_len
        )));
    }

    let data_start = ((8 + header_len + 7) & !7) as u64;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        Error::Model(format!(
            "safetensors shard header read '{}': {e}",
            path.display()
        ))
    })?;
    let header_text = std::str::from_utf8(&header_bytes).map_err(|e| {
        Error::Model(format!(
            "safetensors shard header utf8 '{}': {e}",
            path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(header_text).map_err(|e| {
        Error::Model(format!(
            "safetensors shard header json '{}': {e}",
            path.display()
        ))
    })?;
    parse_safetensors_header_json(&json, path, data_start)
}

fn parse_safetensors_header_json(
    json: &serde_json::Value,
    path: &Path,
    data_start: u64,
) -> Result<ShardHeader> {
    let object = json.as_object().ok_or_else(|| {
        Error::Model(format!(
            "safetensors shard header '{}' is not a JSON object",
            path.display()
        ))
    })?;
    let mut tensors = Vec::new();
    for (name, info) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                Error::Model(format!(
                    "safetensors tensor '{}' in '{}' missing dtype",
                    name,
                    path.display()
                ))
            })?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                Error::Model(format!(
                    "safetensors tensor '{}' in '{}' missing shape",
                    name,
                    path.display()
                ))
            })?
            .iter()
            .map(|value| {
                value.as_u64().map(|value| value as usize).ok_or_else(|| {
                    Error::Model(format!(
                        "safetensors tensor '{}' in '{}' has non-integer shape dim",
                        name,
                        path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                Error::Model(format!(
                    "safetensors tensor '{}' in '{}' missing data_offsets",
                    name,
                    path.display()
                ))
            })?;
        if offsets.len() != 2 {
            return Err(Error::Model(format!(
                "safetensors tensor '{}' in '{}' has invalid data_offsets",
                name,
                path.display()
            )));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            Error::Model(format!(
                "safetensors tensor '{}' in '{}' has invalid data offset start",
                name,
                path.display()
            ))
        })?;
        let end = offsets[1].as_u64().ok_or_else(|| {
            Error::Model(format!(
                "safetensors tensor '{}' in '{}' has invalid data offset end",
                name,
                path.display()
            ))
        })?;
        if end < start {
            return Err(Error::Model(format!(
                "safetensors tensor '{}' in '{}' has reversed data_offsets",
                name,
                path.display()
            )));
        }
        tensors.push(ShardHeaderTensor {
            name: name.clone(),
            dtype,
            shape,
            data_offset: start,
            byte_size: end - start,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(ShardHeader {
        data_start,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::support::TensorRole;

    #[test]
    fn parses_header_inventory_without_payload() {
        let dir = unique_temp_dir("ferrule-inventory-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_fake_safetensors(
            &dir.join(shard),
            serde_json::json!({
                "embed.weight": {
                    "dtype": "BF16",
                    "shape": [2, 3],
                    "data_offsets": [0, 12]
                },
                "layers.0.attn.wq_a.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [4, 5],
                    "data_offsets": [12, 32]
                },
                "layers.0.ffn.experts.3.w1.weight": {
                    "dtype": "I8",
                    "shape": [2, 4],
                    "data_offsets": [32, 40]
                },
                "mtp.0.main_proj.weight": {
                    "dtype": "I8",
                    "shape": [2, 2],
                    "data_offsets": [40, 44]
                }
            }),
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 36 },
                "weight_map": {
                    "embed.weight": shard,
                    "layers.0.attn.wq_a.weight": shard,
                    "layers.0.ffn.experts.3.w1.weight": shard,
                    "mtp.0.main_proj.weight": shard
                }
            })
            .to_string(),
        )
        .unwrap();

        let inventory = HfSafetensorsInventory::open(&dir, ModelFamily::DeepSeekV4).unwrap();
        assert_eq!(inventory.tensor_count, 4);
        assert_eq!(inventory.shard_count, 1);
        assert_eq!(inventory.dtype_bytes("BF16"), 12);
        assert_eq!(inventory.dtype_bytes("F8_E4M3"), 20);
        assert_eq!(inventory.dtype_bytes("I8"), 12);
        assert_eq!(
            inventory.class_count(&TensorClass::SpeculativeProjection),
            1
        );
        assert_eq!(inventory.role_bytes(&TensorRole::SpeculativeProjection), 4);
        let routed = inventory.routed_expert_tensors();
        assert_eq!(routed.len(), 1);
        assert_eq!(routed[0].descriptor.layer, 0);
        assert_eq!(routed[0].descriptor.expert, 3);
        assert_eq!(routed[0].data_offset, 32);
        assert!(routed[0].file_offset > routed[0].data_offset);
        assert_eq!(routed[0].byte_size, 8);
        assert!(inventory.index_only_tensors.is_empty());
        assert!(inventory.header_only_tensors.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_fake_safetensors(path: &Path, header: serde_json::Value) {
        let header = header.to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        while bytes.len() % 8 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0u8; 44]);
        std::fs::write(path, bytes).unwrap();
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
