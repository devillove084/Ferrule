//! Runtime-side object store for graph external bindings.
//!
//! `ExternalBindingPlan` describes semantic graph externals. This store is the
//! runtime materialization step that maps those externals to artifact slices, KV
//! handles, resident objects, or backend-owned opaque handles without embedding
//! storage objects inside graph nodes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::cache::kv::KvHandle;
use crate::graph::program::GraphProgram;
use crate::graph::runtime::{ArtifactGroupKind, ExternalBindingKind, ExternalBindingPlan};
use crate::graph::ExternalKey;
use ferrule_common::{Error, Result};
use ferrule_model::artifact::tensor::ArtifactTensorSlice;
use ferrule_model::moe::streaming::{
    ExpertId, ExpertLoadSource, ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey,
    ExpertTensorSlice,
};
use ferrule_model::semantic::{HyperConnectionStage, RoutedExpertMatrix, RoutedExpertTensorPart};
use ferrule_model::{
    HfRoutedExpertTensorInfo, HfSafetensorsInventory, HfSafetensorsTensorInfo, ModelFamily,
    TensorRole,
};

pub use ferrule_model::ArtifactObjectGroup;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertRegistryObject {
    pub layer: usize,
    pub experts: BTreeMap<ExpertId, ExpertLoadSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendObject {
    ArtifactTensor(ArtifactTensorSlice),
    ArtifactGroup(ArtifactObjectGroup),
    ExpertRegistry(ExpertRegistryObject),
    KvState(Option<KvHandle>),
    Opaque { kind: String, debug_name: String },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackendObjectStore {
    objects: BTreeMap<ExternalKey, BackendObject>,
}

impl BackendObjectStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: ExternalKey, object: BackendObject) -> Option<BackendObject> {
        self.objects.insert(key, object)
    }

    pub fn insert_required(&mut self, key: ExternalKey, object: BackendObject) -> Result<()> {
        if self.objects.insert(key.clone(), object).is_some() {
            return Err(Error::Graph(format!(
                "duplicate backend object for external '{}:{}'",
                key.namespace(),
                key.name()
            )));
        }
        Ok(())
    }

    pub fn get(&self, key: &ExternalKey) -> Option<&BackendObject> {
        self.objects.get(key)
    }

    pub fn objects(&self) -> &BTreeMap<ExternalKey, BackendObject> {
        &self.objects
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

pub fn materialize_dense_hf_externals(
    program: &GraphProgram,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
) -> Result<BackendObjectStore> {
    materialize_graph_hf_externals(program, inventory, model_dir)
}

pub fn materialize_graph_hf_externals(
    program: &GraphProgram,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
) -> Result<BackendObjectStore> {
    materialize_hf_externals_for_family(
        &program.bindings,
        inventory,
        model_dir,
        &program.runtime_plan.family,
    )
}

pub fn materialize_hf_externals(
    bindings: &ExternalBindingPlan,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
) -> Result<BackendObjectStore> {
    materialize_hf_externals_inner(bindings, inventory, model_dir, None)
}

pub fn materialize_hf_externals_for_family(
    bindings: &ExternalBindingPlan,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
    family: &ModelFamily,
) -> Result<BackendObjectStore> {
    materialize_hf_externals_inner(bindings, inventory, model_dir, Some(family))
}

fn materialize_hf_externals_inner(
    bindings: &ExternalBindingPlan,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
    family: Option<&ModelFamily>,
) -> Result<BackendObjectStore> {
    let mut store = BackendObjectStore::new();
    for binding in bindings.entries() {
        match &binding.kind {
            ExternalBindingKind::Weight | ExternalBindingKind::ArtifactTensor => {
                let role = binding.role.as_ref().ok_or_else(|| {
                    Error::Graph(format!(
                        "external '{}:{}' has weight/artifact binding without TensorRole",
                        binding.key.namespace(),
                        binding.key.name()
                    ))
                })?;
                let tensor =
                    resolve_hf_tensor_for_binding(inventory, role, binding.layer, &binding.key)?;
                store.insert_required(
                    binding.key.clone(),
                    BackendObject::ArtifactTensor(ArtifactTensorSlice::from_hf_inventory(
                        model_dir, tensor,
                    )),
                )?;
            }
            ExternalBindingKind::ArtifactGroup(group) => {
                let family = family.ok_or_else(|| {
                    Error::Graph(format!(
                        "artifact group external '{}:{}' requires a ModelFamily for materialization",
                        binding.key.namespace(),
                        binding.key.name()
                    ))
                })?;
                let tensors = materialize_artifact_group(
                    inventory,
                    model_dir,
                    family,
                    *group,
                    binding.layer,
                )?;
                store.insert_required(
                    binding.key.clone(),
                    BackendObject::ArtifactGroup(ArtifactObjectGroup {
                        kind: *group,
                        layer: binding.layer,
                        tensors,
                    }),
                )?;
            }
            ExternalBindingKind::ExpertRegistry => {
                let family = family.ok_or_else(|| {
                    Error::Graph(format!(
                        "expert registry external '{}:{}' requires a ModelFamily for materialization",
                        binding.key.namespace(),
                        binding.key.name()
                    ))
                })?;
                let layer = binding.layer.ok_or_else(|| {
                    Error::Graph(format!(
                        "expert registry external '{}:{}' is missing layer metadata",
                        binding.key.namespace(),
                        binding.key.name()
                    ))
                })?;
                store.insert_required(
                    binding.key.clone(),
                    BackendObject::ExpertRegistry(materialize_expert_registry(
                        inventory, model_dir, family, layer,
                    )?),
                )?;
            }
            ExternalBindingKind::KvState => {
                store.insert_required(binding.key.clone(), BackendObject::KvState(None))?;
            }
            other => {
                store.insert_required(
                    binding.key.clone(),
                    BackendObject::Opaque {
                        kind: format!("{other:?}"),
                        debug_name: binding.key.name().to_string(),
                    },
                )?;
            }
        }
    }
    Ok(store)
}

fn materialize_artifact_group(
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
    _family: &ModelFamily,
    group: ArtifactGroupKind,
    layer: Option<usize>,
) -> Result<Vec<ArtifactTensorSlice>> {
    let names = match group {
        ArtifactGroupKind::Attention => {
            let layer = require_group_layer(group, layer)?;
            inventory
                .attention_tensors()
                .into_iter()
                .filter(|tensor| tensor.descriptor.layer == layer)
                .map(|tensor| tensor.name)
                .collect::<Vec<_>>()
        }
        ArtifactGroupKind::LayerNorm => {
            let layer = require_group_layer(group, layer)?;
            let layer_markers = [
                format!("model.layers.{layer}."),
                format!("layers.{layer}."),
                format!("blk.{layer}."),
            ];
            inventory
                .tensors
                .iter()
                .filter(|tensor| {
                    matches!(
                        tensor.role,
                        TensorRole::AttentionNorm | TensorRole::FeedForwardNorm
                    ) && layer_markers
                        .iter()
                        .any(|marker| tensor.name.contains(marker))
                })
                .map(|tensor| tensor.name.clone())
                .collect::<Vec<_>>()
        }
        ArtifactGroupKind::HyperConnectionAttention
        | ArtifactGroupKind::HyperConnectionFeedForward => {
            let layer = require_group_layer(group, layer)?;
            let expected_stage = match group {
                ArtifactGroupKind::HyperConnectionAttention => HyperConnectionStage::Attention,
                ArtifactGroupKind::HyperConnectionFeedForward => HyperConnectionStage::FeedForward,
                _ => unreachable!(),
            };
            inventory
                .hyper_connection_tensors()
                .into_iter()
                .filter(|tensor| {
                    tensor.descriptor.layer == Some(layer)
                        && tensor.descriptor.stage == expected_stage
                })
                .map(|tensor| tensor.name)
                .collect::<Vec<_>>()
        }
        ArtifactGroupKind::HyperConnectionHead => inventory
            .hyper_connection_tensors()
            .into_iter()
            .filter(|tensor| {
                tensor.descriptor.layer.is_none()
                    && tensor.descriptor.stage == HyperConnectionStage::Head
            })
            .map(|tensor| tensor.name)
            .collect::<Vec<_>>(),
        ArtifactGroupKind::Router => {
            let layer = require_group_layer(group, layer)?;
            inventory
                .router_tensors()
                .into_iter()
                .filter(|tensor| tensor.descriptor.layer == layer)
                .map(|tensor| tensor.name)
                .collect::<Vec<_>>()
        }
        ArtifactGroupKind::SharedExpert => {
            let layer = require_group_layer(group, layer)?;
            inventory
                .shared_expert_tensors()
                .into_iter()
                .filter(|tensor| tensor.descriptor.layer == layer)
                .map(|tensor| tensor.name)
                .collect::<Vec<_>>()
        }
    };
    if names.is_empty() {
        return Err(Error::Graph(format!(
            "no artifact tensors found for group {} layer={layer:?}",
            group.as_str()
        )));
    }
    let mut slices = names
        .iter()
        .map(|name| artifact_slice_by_name(inventory, model_dir, name))
        .collect::<Result<Vec<_>>>()?;
    slices.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(slices)
}

fn require_group_layer(group: ArtifactGroupKind, layer: Option<usize>) -> Result<usize> {
    layer.ok_or_else(|| {
        Error::Graph(format!(
            "artifact group {} requires layer metadata",
            group.as_str()
        ))
    })
}

fn materialize_expert_registry(
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
    _family: &ModelFamily,
    layer: usize,
) -> Result<ExpertRegistryObject> {
    let mut grouped = BTreeMap::<ExpertId, Vec<ExpertTensorSlice>>::new();
    for tensor in inventory
        .routed_expert_tensors()
        .into_iter()
        .filter(|tensor| tensor.descriptor.layer == layer)
    {
        let expert = ExpertId::new(tensor.descriptor.layer, tensor.descriptor.expert);
        grouped
            .entry(expert)
            .or_default()
            .push(expert_tensor_slice(model_dir, tensor, expert));
    }
    if grouped.is_empty() {
        return Err(Error::Graph(format!(
            "no routed expert tensors found for layer {layer}"
        )));
    }
    let mut experts = BTreeMap::new();
    for (expert, mut tensors) in grouped {
        tensors.sort_by(|a, b| {
            a.key
                .matrix
                .cmp(&b.key.matrix)
                .then_with(|| a.component.cmp(&b.component))
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| a.offset.cmp(&b.offset))
        });
        experts.insert(expert, ExpertLoadSource::LocalTensorSet { tensors });
    }
    Ok(ExpertRegistryObject { layer, experts })
}

fn artifact_slice_by_name(
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
    name: &str,
) -> Result<ArtifactTensorSlice> {
    let tensor = inventory
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::Graph(format!("artifact tensor '{name}' not found in inventory")))?;
    Ok(ArtifactTensorSlice::from_hf_inventory(model_dir, tensor))
}

fn expert_tensor_slice(
    model_dir: &Path,
    tensor: HfRoutedExpertTensorInfo,
    expert: ExpertId,
) -> ExpertTensorSlice {
    ExpertTensorSlice {
        key: ExpertTensorKey {
            expert,
            matrix: expert_matrix_from_model(tensor.descriptor.matrix),
        },
        component: expert_component_from_model(tensor.descriptor.part),
        path: model_dir.join(tensor.shard),
        offset: tensor.file_offset,
        bytes: tensor.byte_size,
        dtype: tensor.dtype,
        shape: tensor.shape,
    }
}

fn expert_matrix_from_model(matrix: RoutedExpertMatrix) -> ExpertMatrixKind {
    match matrix {
        RoutedExpertMatrix::Gate => ExpertMatrixKind::Gate,
        RoutedExpertMatrix::Up => ExpertMatrixKind::Up,
        RoutedExpertMatrix::Down => ExpertMatrixKind::Down,
    }
}

fn expert_component_from_model(part: RoutedExpertTensorPart) -> ExpertTensorComponent {
    match part {
        RoutedExpertTensorPart::Weight => ExpertTensorComponent::Weight,
        RoutedExpertTensorPart::Scale => ExpertTensorComponent::Scale,
        RoutedExpertTensorPart::Other(value) => ExpertTensorComponent::Other(value),
    }
}

fn resolve_hf_tensor_for_binding<'a>(
    inventory: &'a HfSafetensorsInventory,
    role: &TensorRole,
    layer: Option<usize>,
    key: &ExternalKey,
) -> Result<&'a HfSafetensorsTensorInfo> {
    let candidates = inventory
        .tensors
        .iter()
        .filter(|tensor| &tensor.role == role)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(Error::Graph(format!(
            "no HF tensor found for external role {role} layer={layer:?}"
        )));
    }
    if let Some(layer) = layer {
        let layer_markers = [
            format!("model.layers.{layer}."),
            format!("layers.{layer}."),
            format!("blk.{layer}."),
        ];
        let semantic_markers = semantic_artifact_markers(key.name());
        let matches = candidates
            .iter()
            .copied()
            .filter(|tensor| {
                layer_markers
                    .iter()
                    .any(|marker| tensor.name.contains(marker))
                    && semantic_markers
                        .iter()
                        .any(|marker| tensor.name.contains(marker))
            })
            .collect::<Vec<_>>();
        return match matches.as_slice() {
            [tensor] => Ok(*tensor),
            [] => Err(Error::Graph(format!(
                "no HF tensor found for external role {role} layer={layer} key={}:{}",
                key.namespace(),
                key.name()
            ))),
            _ => Err(Error::Graph(format!(
                "ambiguous HF tensors for external role {role} layer={layer} key={}:{}: {} candidates",
                key.namespace(),
                key.name(),
                matches.len()
            ))),
        };
    }
    let semantic_markers = semantic_artifact_markers(key.name());
    let matches = candidates
        .iter()
        .copied()
        .filter(|tensor| {
            semantic_markers
                .iter()
                .any(|marker| tensor.name.contains(marker))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [tensor] => Ok(*tensor),
        [] if candidates.len() == 1 => Ok(candidates[0]),
        [] => Err(Error::Graph(format!(
            "no HF tensor found for model-level role {role} key={}:{}",
            key.namespace(),
            key.name()
        ))),
        _ => Err(Error::Graph(format!(
            "ambiguous HF tensors for model-level role {role} key={}:{}: {} candidates",
            key.namespace(),
            key.name(),
            matches.len()
        ))),
    }
}

fn semantic_artifact_markers(name: &str) -> Vec<&'static str> {
    if name == "token_embedding" {
        return vec![
            "embed_tokens",
            "tok_embeddings",
            "token_embd",
            "embed.weight",
        ];
    }
    if name == "output_norm" {
        return vec!["model.norm", "output_norm", "norm.weight", "final_norm"];
    }
    if name == "output_head" {
        return vec!["lm_head", "output.weight", "head.weight"];
    }
    if name.ends_with(".input_norm") {
        return vec!["input_layernorm", "attention_norm"];
    }
    if name.ends_with(".post_attention_norm") {
        return vec!["post_attention_layernorm", "ffn_norm"];
    }
    if name.ends_with(".attn.q") {
        return vec!["self_attn.q_proj", "attn_q"];
    }
    if name.ends_with(".attn.k") {
        return vec!["self_attn.k_proj", "attn_k"];
    }
    if name.ends_with(".attn.v") {
        return vec!["self_attn.v_proj", "attn_v"];
    }
    if name.ends_with(".attn.o") {
        return vec!["self_attn.o_proj", "attn_output"];
    }
    if name.ends_with(".ffn.gate") {
        return vec!["mlp.gate_proj", "ffn_gate"];
    }
    if name.ends_with(".ffn.up") {
        return vec!["mlp.up_proj", "ffn_up"];
    }
    if name.ends_with(".ffn.down") {
        return vec!["mlp.down_proj", "ffn_down"];
    }
    Vec::new()
}
