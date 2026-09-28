//! Semantic expert sources, descriptors, and payload encoding validation.
//!
//! Source discovery captures checkpoint metadata but never reads payload bytes.
//! Storage returns artifact payloads; constructing a compute bundle validates
//! their encoding independently of the transport. No residency or I/O authority
//! is owned here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::HfRoutedExpertTensorInfo;
use crate::checkpoint::{
    CheckpointReadExtent, CheckpointReadPlan, CheckpointSourceFileIdentity, CheckpointSourceTensor,
    checkpoint_resource_source,
};
use crate::materialization::{
    HF_SAFETENSORS_ROUTED_EXPERT_V1, MaterializationSourceCatalog, MaterializationSourceEntry,
    ResourceSource,
};
use crate::semantic::{RoutedExpertMatrix, RoutedExpertTensorPart, RoutedExpertTensorRef};
use ferrule_common::{
    Error, ExpertId as ProtocolExpertId, LayerId, MaterializedResourceId, Result, StaleReason,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: usize,
    pub expert: usize,
}

impl ExpertId {
    pub fn new(layer: usize, expert: usize) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertMatrixKind {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertTensorKey {
    pub expert: ExpertId,
    pub matrix: ExpertMatrixKind,
}

impl ExpertTensorKey {
    pub fn new(layer: usize, expert: usize, matrix: ExpertMatrixKind) -> Self {
        Self {
            expert: ExpertId::new(layer, expert),
            matrix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertTensorComponent {
    Weight,
    Scale,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensorSlice {
    pub key: ExpertTensorKey,
    pub component: ExpertTensorComponent,
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl ExpertTensorSlice {
    pub fn end_offset(&self) -> u64 {
        self.offset.saturating_add(self.bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertStorageTier {
    Gpu,
    Cpu,
    LocalStorage,
    Remote,
    Loading,
}

impl ExpertStorageTier {
    pub fn is_gpu_ready(self) -> bool {
        matches!(self, Self::Gpu)
    }

    pub fn is_streamable(self) -> bool {
        matches!(self, Self::Cpu | Self::LocalStorage | Self::Remote)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLoadSource {
    GpuResident,
    CpuResident,
    LocalShard {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
    LocalTensorSet {
        tensors: Vec<ExpertTensorSlice>,
    },
    /// HF tensor set with catalog-time source metadata snapshots.
    HfLocalTensorSet {
        tensors: Vec<ExpertTensorSlice>,
        source_files: Arc<[CheckpointSourceFileIdentity]>,
    },
    WeightPackChunk {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
    Remote {
        uri: String,
        offset: u64,
        bytes: u64,
    },
}

impl ExpertLoadSource {
    pub fn tier(&self) -> ExpertStorageTier {
        match self {
            Self::GpuResident => ExpertStorageTier::Gpu,
            Self::CpuResident => ExpertStorageTier::Cpu,
            Self::LocalShard { .. }
            | Self::LocalTensorSet { .. }
            | Self::HfLocalTensorSet { .. }
            | Self::WeightPackChunk { .. } => ExpertStorageTier::LocalStorage,
            Self::Remote { .. } => ExpertStorageTier::Remote,
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Self::GpuResident | Self::CpuResident => 0,
            Self::LocalShard { bytes, .. }
            | Self::WeightPackChunk { bytes, .. }
            | Self::Remote { bytes, .. } => *bytes,
            Self::LocalTensorSet { tensors } | Self::HfLocalTensorSet { tensors, .. } => {
                tensors.iter().map(|tensor| tensor.bytes).sum()
            }
        }
    }

    /// Compare the current filesystem metadata with the catalog snapshot.
    ///
    /// This hook performs no payload open or read. A mismatch is stale source
    /// identity and must abort physical materialization before publication.
    pub fn validate_source_identity(&self) -> std::result::Result<(), StaleReason> {
        if self
            .source_files()
            .iter()
            .all(CheckpointSourceFileIdentity::is_current)
        {
            Ok(())
        } else {
            Err(StaleReason::SourceIdentityChanged)
        }
    }

    fn source_files(&self) -> &[CheckpointSourceFileIdentity] {
        match self {
            Self::HfLocalTensorSet { source_files, .. } => source_files,
            _ => &[],
        }
    }
}

/// Immutable mapping from expert identity to model-neutral source metadata.
///
/// Catalogs are intended to be built once during artifact discovery and shared by
/// prepared plans and backend runtimes. The generic source type keeps the catalog
/// reusable for source representations beyond [`ExpertLoadSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSourceDenseLayout {
    pub first_layer: usize,
    pub layer_count: usize,
    pub experts_per_layer: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpertSourceCatalogEntry<S> {
    source: S,
    resource_source: Option<ResourceSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSourceCatalog<S = ExpertLoadSource> {
    sources: Vec<(ExpertId, ExpertSourceCatalogEntry<S>)>,
    dense_layout: Option<ExpertSourceDenseLayout>,
}

impl<S> ExpertSourceCatalog<S> {
    pub fn from_sources(sources: impl IntoIterator<Item = (ExpertId, S)>) -> Self {
        Self::from_entries(
            sources
                .into_iter()
                .map(|(expert, source)| (expert, source, None)),
        )
    }

    pub fn from_resource_sources(
        sources: impl IntoIterator<Item = (ExpertId, S, ResourceSource)>,
    ) -> Self {
        Self::from_entries(
            sources
                .into_iter()
                .map(|(expert, source, resource_source)| (expert, source, Some(resource_source))),
        )
    }

    pub(super) fn from_entries(
        sources: impl IntoIterator<Item = (ExpertId, S, Option<ResourceSource>)>,
    ) -> Self {
        let sources = sources
            .into_iter()
            .map(|(expert, source, resource_source)| {
                (
                    expert,
                    ExpertSourceCatalogEntry {
                        source,
                        resource_source,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>();
        let dense_layout = detect_dense_source_layout(&sources);
        Self {
            sources,
            dense_layout,
        }
    }

    pub fn source(&self, expert: ExpertId) -> Option<&S> {
        self.entry(expert).map(|entry| &entry.source)
    }

    pub fn resource_source(&self, expert: ExpertId) -> Option<ResourceSource> {
        self.entry(expert).and_then(|entry| entry.resource_source)
    }

    pub fn require_resource_source(&self, expert: ExpertId) -> Result<ResourceSource> {
        self.resource_source(expert).ok_or_else(|| Error::Model {
            message: format!(
                "expert source catalog has no materialization source for layer {} expert {}",
                expert.layer, expert.expert
            ),
        })
    }

    fn entry(&self, expert: ExpertId) -> Option<&ExpertSourceCatalogEntry<S>> {
        if let Some(dense) = self.dense_layout {
            let layer = expert.layer.checked_sub(dense.first_layer)?;
            if layer >= dense.layer_count || expert.expert >= dense.experts_per_layer {
                return None;
            }
            let index = layer
                .checked_mul(dense.experts_per_layer)?
                .checked_add(expert.expert)?;
            let (stored, entry) = self.sources.get(index)?;
            return (*stored == expert).then_some(entry);
        }
        self.sources
            .binary_search_by_key(&expert, |(stored, _)| *stored)
            .ok()
            .and_then(|index| self.sources.get(index).map(|(_, entry)| entry))
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&ExpertId, &S)> {
        self.sources
            .iter()
            .map(|(expert, entry)| (expert, &entry.source))
    }

    pub(super) fn iter_entries(
        &self,
    ) -> impl ExactSizeIterator<Item = (&ExpertId, &S, Option<ResourceSource>)> {
        self.sources
            .iter()
            .map(|(expert, entry)| (expert, &entry.source, entry.resource_source))
    }

    pub fn count(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub const fn dense_layout(&self) -> Option<ExpertSourceDenseLayout> {
        self.dense_layout
    }
}

impl ExpertSourceCatalog<ExpertLoadSource> {
    /// Convert expert-specific discovery output into the provider-neutral exact
    /// source catalog consumed by physical materialization backends.
    pub fn materialization_sources(
        &self,
    ) -> Result<MaterializationSourceCatalog<ExpertLoadSource>> {
        let entries = self
            .iter_entries()
            .map(|(expert, source, resource_source)| {
                let layer = u32::try_from(expert.layer).map_err(|_| {
                    Error::Model { message: format!(
                        "expert source layer {} exceeds materialization coordinates",
                        expert.layer
                    ) }
                })?;
                let expert_index = u32::try_from(expert.expert).map_err(|_| {
                    Error::Model { message: format!(
                        "expert source index {} exceeds materialization coordinates",
                        expert.expert
                    ) }
                })?;
                let resource_source = resource_source.ok_or_else(|| {
                    Error::Model { message: format!(
                        "expert source catalog has no materialization identity for layer {} expert {}",
                        expert.layer, expert.expert
                    ) }
                })?;
                MaterializationSourceEntry::new(
                    MaterializedResourceId::routed_expert(
                        LayerId::new(layer),
                        ProtocolExpertId::new(expert_index),
                    ),
                    resource_source,
                    checkpoint_read_plan_for_expert_source(*expert, source)?,
                    source.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        MaterializationSourceCatalog::new(entries)
    }
}

impl<S> Default for ExpertSourceCatalog<S> {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            dense_layout: None,
        }
    }
}

pub(super) fn checkpoint_read_plan_for_expert_source(
    expert: ExpertId,
    source: &ExpertLoadSource,
) -> Result<CheckpointReadPlan> {
    let (extents, source_files) = match source {
        ExpertLoadSource::HfLocalTensorSet {
            tensors,
            source_files,
        } => (
            tensors
                .iter()
                .map(|tensor| {
                    CheckpointReadExtent::new(tensor.path.clone(), tensor.offset, tensor.bytes)
                })
                .collect::<Result<Vec<_>>>()?,
            Arc::clone(source_files),
        ),
        ExpertLoadSource::LocalTensorSet { .. }
        | ExpertLoadSource::LocalShard { .. }
        | ExpertLoadSource::WeightPackChunk { .. } => {
            return Err(Error::Model {
                message: format!(
                    "expert source {}:{} has no catalog-time checkpoint snapshots",
                    expert.layer, expert.expert
                ),
            });
        }
        ExpertLoadSource::GpuResident
        | ExpertLoadSource::CpuResident
        | ExpertLoadSource::Remote { .. } => {
            return Err(Error::Model {
                message: format!(
                    "expert source {}:{} is not a local checkpoint read plan",
                    expert.layer, expert.expert
                ),
            });
        }
    };
    CheckpointReadPlan::new(extents, source_files)
}

fn detect_dense_source_layout<S>(
    sources: &[(ExpertId, ExpertSourceCatalogEntry<S>)],
) -> Option<ExpertSourceDenseLayout> {
    let first = sources.first()?.0;
    let last = sources.last()?.0;
    if first.expert != 0 {
        return None;
    }
    let layer_count = last.layer.checked_sub(first.layer)?.checked_add(1)?;
    let experts_per_layer = sources
        .iter()
        .take_while(|(expert, _)| expert.layer == first.layer)
        .count();
    if experts_per_layer == 0 || layer_count.checked_mul(experts_per_layer)? != sources.len() {
        return None;
    }
    for (index, (expert, _)) in sources.iter().enumerate() {
        let expected = ExpertId::new(
            first.layer + index / experts_per_layer,
            index % experts_per_layer,
        );
        if *expert != expected {
            return None;
        }
    }
    Some(ExpertSourceDenseLayout {
        first_layer: first.layer,
        layer_count,
        experts_per_layer,
    })
}

type HfCatalogTensor = (
    HfRoutedExpertTensorInfo,
    ExpertTensorSlice,
    CheckpointSourceFileIdentity,
);

impl ExpertSourceCatalog<ExpertLoadSource> {
    pub fn from_hf_routed_expert_tensor_sets(
        model_dir: &Path,
        tensors: impl IntoIterator<Item = HfRoutedExpertTensorInfo>,
    ) -> Result<Self> {
        Self::from_hf_routed_expert_tensor_sets_with_source_identity(
            model_dir,
            tensors,
            CheckpointSourceFileIdentity::capture,
        )
    }

    fn from_hf_routed_expert_tensor_sets_with_source_identity<F>(
        model_dir: &Path,
        tensors: impl IntoIterator<Item = HfRoutedExpertTensorInfo>,
        mut capture_source_identity: F,
    ) -> Result<Self>
    where
        F: FnMut(&Path) -> Result<CheckpointSourceFileIdentity>,
    {
        let mut grouped = BTreeMap::<ExpertId, Vec<HfCatalogTensor>>::new();
        let mut source_files = BTreeMap::<PathBuf, CheckpointSourceFileIdentity>::new();
        for tensor in tensors {
            let expert = expert_id_from_ref(&tensor.descriptor);
            let path = model_dir.join(&tensor.shard);
            let source_file = match source_files.get(&path) {
                Some(source_file) => source_file.clone(),
                None => {
                    let source_file = capture_source_identity(&path)?;
                    source_files.insert(path.clone(), source_file.clone());
                    source_file
                }
            };
            let slice = ExpertTensorSlice {
                key: ExpertTensorKey {
                    expert,
                    matrix: matrix_from_model(tensor.descriptor.matrix),
                },
                component: component_from_model(tensor.descriptor.part.clone()),
                path,
                offset: tensor.file_offset,
                bytes: tensor.byte_size,
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
            };
            let end_offset = slice
                .offset
                .checked_add(slice.bytes)
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "expert tensor extent overflows for layer {} expert {}",
                        expert.layer, expert.expert
                    ),
                })?;
            if end_offset > source_file.length() {
                return Err(Error::Model {
                    message: format!(
                        "expert tensor extent {}..{} exceeds shard metadata length {} for '{}'",
                        slice.offset,
                        end_offset,
                        source_file.length(),
                        slice.path.display()
                    ),
                });
            }
            grouped
                .entry(expert)
                .or_default()
                .push((tensor, slice, source_file));
        }

        let mut sources = Vec::new();
        for (expert, mut tensors) in grouped {
            tensors.sort_by(|(_, a, _), (_, b, _)| {
                a.key
                    .matrix
                    .cmp(&b.key.matrix)
                    .then_with(|| a.component.cmp(&b.component))
                    .then_with(|| a.dtype.cmp(&b.dtype))
                    .then_with(|| a.shape.cmp(&b.shape))
                    .then_with(|| a.bytes.cmp(&b.bytes))
                    .then_with(|| a.path.cmp(&b.path))
                    .then_with(|| a.offset.cmp(&b.offset))
            });
            if tensors.is_empty() {
                return Err(Error::Model {
                    message: format!(
                        "empty expert tensor set for layer {} expert {}",
                        expert.layer, expert.expert
                    ),
                });
            }
            let resource_source = hf_expert_resource_source(&tensors)?;
            let source_files = tensors
                .iter()
                .map(|(_, _, source_file)| source_file.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let tensors = tensors.into_iter().map(|(_, slice, _)| slice).collect();
            sources.push((
                expert,
                ExpertLoadSource::HfLocalTensorSet {
                    tensors,
                    source_files: Arc::from(source_files),
                },
                resource_source,
            ));
        }
        Ok(Self::from_resource_sources(sources))
    }
}

fn hf_expert_resource_source(tensors: &[HfCatalogTensor]) -> Result<ResourceSource> {
    let semantics = tensors
        .iter()
        .map(|(_, slice, _)| expert_tensor_semantic_descriptor(slice))
        .collect::<Vec<_>>();
    let descriptors = tensors
        .iter()
        .zip(&semantics)
        .map(
            |((_, slice, source_file), semantic)| CheckpointSourceTensor {
                semantic,
                path: &slice.path,
                offset: slice.offset,
                bytes: slice.bytes,
                dtype: &slice.dtype,
                shape: &slice.shape,
                source_file,
            },
        )
        .collect::<Vec<_>>();
    checkpoint_resource_source(
        b"hf-routed-expert-v3",
        HF_SAFETENSORS_ROUTED_EXPERT_V1,
        &descriptors,
    )
}

fn expert_tensor_semantic_descriptor(slice: &ExpertTensorSlice) -> Vec<u8> {
    let mut semantic = vec![match slice.key.matrix {
        ExpertMatrixKind::Gate => 0,
        ExpertMatrixKind::Up => 1,
        ExpertMatrixKind::Down => 2,
    }];
    match &slice.component {
        ExpertTensorComponent::Weight => semantic.push(0),
        ExpertTensorComponent::Scale => semantic.push(1),
        ExpertTensorComponent::Other(component) => {
            semantic.push(2);
            semantic.extend_from_slice(&(component.len() as u64).to_be_bytes());
            semantic.extend_from_slice(component.as_bytes());
        }
    }
    semantic
}

pub(super) fn slices_for_load_source(
    expert: ExpertId,
    load_source: &ExpertLoadSource,
) -> Result<Vec<ExpertTensorSlice>> {
    match load_source {
        ExpertLoadSource::LocalTensorSet { tensors }
        | ExpertLoadSource::HfLocalTensorSet { tensors, .. } => Ok(tensors.clone()),
        ExpertLoadSource::LocalShard {
            path,
            offset,
            bytes,
        } => Ok(vec![ExpertTensorSlice {
            key: ExpertTensorKey {
                expert,
                matrix: ExpertMatrixKind::Gate,
            },
            component: ExpertTensorComponent::Other("whole_expert_chunk".into()),
            path: path.clone(),
            offset: *offset,
            bytes: *bytes,
            dtype: "opaque".into(),
            shape: Vec::new(),
        }]),
        other => Err(Error::Model {
            message: format!(
                "expert streaming reader does not support artifact tier {:?} yet",
                other.tier()
            ),
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensorPayload {
    pub slice: ExpertTensorSlice,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertArtifactPayload {
    pub expert: ExpertId,
    pub tensors: Vec<ExpertTensorPayload>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertBundleFormat {
    Bf16,
    Fp4E2M1PackedWithE8M0Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBundleLayout {
    pub format: ExpertBundleFormat,
    pub input_features: usize,
    pub intermediate_features: usize,
    pub output_features: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLinearFormat {
    /// Unscaled row-major BF16 expert matrix.
    Bf16 {
        out_features: usize,
        in_features: usize,
    },
    /// Packed FP4 expert artifact format: `torch.float4_e2m1fn_x2` stored in
    /// safetensors as I8 bytes, with one `float8_e8m0fnu` scale per logical
    /// K block.
    Fp4E2M1PackedWithE8M0Scale {
        out_features: usize,
        in_features: usize,
        block_size: usize,
    },
    Opaque,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLinearPayload {
    pub matrix: ExpertMatrixKind,
    pub weight: ExpertTensorPayload,
    pub scale: Option<ExpertTensorPayload>,
    pub format: ExpertLinearFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertComputeBundle {
    pub expert: ExpertId,
    pub gate: ExpertLinearPayload,
    pub up: ExpertLinearPayload,
    pub down: ExpertLinearPayload,
}

impl ExpertComputeBundle {
    pub fn from_artifact_payload(payload: ExpertArtifactPayload) -> Result<Self> {
        let expert = payload.expert;
        let mut grouped = BTreeMap::<ExpertMatrixKind, Vec<ExpertTensorPayload>>::new();
        for tensor in payload.tensors {
            if tensor.slice.key.expert != expert {
                return Err(Error::Model {
                    message: format!(
                        "expert payload contains mismatched tensor: expected layer {} expert {}, got layer {} expert {}",
                        expert.layer,
                        expert.expert,
                        tensor.slice.key.expert.layer,
                        tensor.slice.key.expert.expert
                    ),
                });
            }
            grouped
                .entry(tensor.slice.key.matrix)
                .or_default()
                .push(tensor);
        }
        let bundle = Self {
            expert,
            gate: build_linear_payload(
                expert,
                ExpertMatrixKind::Gate,
                grouped.remove(&ExpertMatrixKind::Gate),
            )?,
            up: build_linear_payload(
                expert,
                ExpertMatrixKind::Up,
                grouped.remove(&ExpertMatrixKind::Up),
            )?,
            down: build_linear_payload(
                expert,
                ExpertMatrixKind::Down,
                grouped.remove(&ExpertMatrixKind::Down),
            )?,
        };
        validate_expert_compute_bundle(&bundle)?;
        Ok(bundle)
    }

    pub fn total_bytes(&self) -> u64 {
        linear_payload_bytes(&self.gate)
            .saturating_add(linear_payload_bytes(&self.up))
            .saturating_add(linear_payload_bytes(&self.down))
    }

    pub fn layout(&self) -> Result<ExpertBundleLayout> {
        typed_expert_bundle_layout(
            &self.gate.format,
            &self.up.format,
            &self.down.format,
            "expert artifact bundle",
        )
    }
}

fn build_linear_payload(
    expert: ExpertId,
    matrix: ExpertMatrixKind,
    tensors: Option<Vec<ExpertTensorPayload>>,
) -> Result<ExpertLinearPayload> {
    let tensors = tensors.ok_or_else(|| Error::Model {
        message: format!(
            "expert artifact bundle missing {:?} matrix for layer {} expert {}",
            matrix, expert.layer, expert.expert
        ),
    })?;
    let mut weight = None;
    let mut scale = None;
    for tensor in tensors {
        match tensor.slice.component {
            ExpertTensorComponent::Weight => {
                if weight.replace(tensor).is_some() {
                    return Err(Error::Model {
                        message: format!(
                            "expert artifact bundle has duplicate {:?} weight for layer {} expert {}",
                            matrix, expert.layer, expert.expert
                        ),
                    });
                }
            }
            ExpertTensorComponent::Scale => {
                if scale.replace(tensor).is_some() {
                    return Err(Error::Model {
                        message: format!(
                            "expert artifact bundle has duplicate {:?} scale for layer {} expert {}",
                            matrix, expert.layer, expert.expert
                        ),
                    });
                }
            }
            ExpertTensorComponent::Other(name) => {
                return Err(Error::Model {
                    message: format!(
                        "expert artifact bundle has unsupported {:?} component '{}' for layer {} expert {}",
                        matrix, name, expert.layer, expert.expert
                    ),
                });
            }
        }
    }
    let weight = weight.ok_or_else(|| Error::Model {
        message: format!(
            "expert artifact bundle missing {:?} weight for layer {} expert {}",
            matrix, expert.layer, expert.expert
        ),
    })?;
    let format = infer_linear_format(&weight, scale.as_ref())?;
    Ok(ExpertLinearPayload {
        matrix,
        weight,
        scale,
        format,
    })
}

fn infer_linear_format(
    weight: &ExpertTensorPayload,
    scale: Option<&ExpertTensorPayload>,
) -> Result<ExpertLinearFormat> {
    infer_expert_linear_format(
        &weight.slice,
        weight.bytes.len(),
        scale.map(|scale| (&scale.slice, scale.bytes.len())),
    )
}

pub(crate) fn infer_expert_linear_format(
    weight: &ExpertTensorSlice,
    weight_len: usize,
    scale: Option<(&ExpertTensorSlice, usize)>,
) -> Result<ExpertLinearFormat> {
    if weight.dtype == "BF16" {
        if let Some((scale, _)) = scale {
            return Err(Error::Model {
                message: format!(
                    "BF16 expert tensor has unexpected scale dtype={} shape={:?}",
                    scale.dtype, scale.shape
                ),
            });
        }
        let [out_features, in_features]: [usize; 2] =
            weight
                .shape
                .as_slice()
                .try_into()
                .map_err(|_| Error::Model {
                    message: format!(
                        "BF16 expert tensor expects a 2D weight shape, got {:?}",
                        weight.shape
                    ),
                })?;
        if out_features == 0 || in_features == 0 {
            return Err(Error::Model {
                message: format!(
                    "BF16 expert tensor dimensions must be non-zero, got {:?}",
                    weight.shape
                ),
            });
        }
        let expected_bytes = out_features
            .checked_mul(in_features)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Model {
                message: format!(
                    "BF16 expert tensor byte size overflows for shape {:?}",
                    weight.shape
                ),
            })?;
        if weight_len != expected_bytes || weight.bytes != expected_bytes as u64 {
            return Err(Error::Model {
                message: format!(
                    "BF16 expert tensor byte length mismatch: shape {:?} requires {expected_bytes}, metadata={}, payload={weight_len}",
                    weight.shape, weight.bytes
                ),
            });
        }
        return Ok(ExpertLinearFormat::Bf16 {
            out_features,
            in_features,
        });
    }
    let Some((scale, scale_len)) = scale else {
        return Ok(ExpertLinearFormat::Opaque);
    };
    if weight.dtype == "I8" && scale.dtype == "F8_E8M0" {
        let [out_features, packed_in_features]: [usize; 2] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::Model {
                message: format!(
                    "FP4 expert tensor expects a 2D weight shape, got {:?}",
                    weight.shape
                ),
            })?;
        let [scale_rows, scale_cols]: [usize; 2] =
            scale
                .shape
                .as_slice()
                .try_into()
                .map_err(|_| Error::Model {
                    message: format!(
                        "FP4 expert tensor expects a 2D scale shape, got {:?}",
                        scale.shape
                    ),
                })?;
        let in_features = packed_in_features
            .checked_mul(2)
            .ok_or_else(|| Error::Model {
                message: "FP4 expert packed input dimension overflow".into(),
            })?;
        if out_features == 0
            || in_features == 0
            || !in_features.is_multiple_of(32)
            || !in_features.is_multiple_of(2)
        {
            return Err(Error::Model {
                message: format!(
                    "FP4 expert tensor has invalid packed shape {:?}; logical input width must be a non-zero multiple of 32",
                    weight.shape
                ),
            });
        }
        let expected_weight_bytes =
            out_features
                .checked_mul(packed_in_features)
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "FP4 expert weight byte size overflows for shape {:?}",
                        weight.shape
                    ),
                })?;
        let expected_scale_cols = in_features / 32;
        let expected_scale_bytes =
            out_features
                .checked_mul(expected_scale_cols)
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "FP4 expert scale byte size overflows for weight shape {:?}",
                        weight.shape
                    ),
                })?;
        if scale_rows != out_features || scale_cols != expected_scale_cols {
            return Err(Error::Model {
                message: format!(
                    "FP4 expert scale shape mismatch: weight {:?} implies scale [{out_features}, {expected_scale_cols}], got {:?}",
                    weight.shape, scale.shape
                ),
            });
        }
        if weight_len != expected_weight_bytes
            || weight.bytes != expected_weight_bytes as u64
            || scale_len != expected_scale_bytes
            || scale.bytes != expected_scale_bytes as u64
        {
            return Err(Error::Model {
                message: format!(
                    "FP4 expert tensor byte length mismatch: weight metadata={} payload={weight_len} expected={expected_weight_bytes}; scale metadata={} payload={scale_len} expected={expected_scale_bytes}",
                    weight.bytes, scale.bytes
                ),
            });
        }
        return Ok(ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        });
    }
    Ok(ExpertLinearFormat::Opaque)
}

fn validate_expert_compute_bundle(bundle: &ExpertComputeBundle) -> Result<()> {
    let formats = [&bundle.gate.format, &bundle.up.format, &bundle.down.format];
    if formats
        .iter()
        .all(|format| matches!(format, ExpertLinearFormat::Opaque))
    {
        return Ok(());
    }
    typed_expert_bundle_layout(
        &bundle.gate.format,
        &bundle.up.format,
        &bundle.down.format,
        "expert artifact bundle",
    )?;
    Ok(())
}

pub(crate) fn typed_expert_bundle_layout(
    gate: &ExpertLinearFormat,
    up: &ExpertLinearFormat,
    down: &ExpertLinearFormat,
    context: &str,
) -> Result<ExpertBundleLayout> {
    let gate_dimensions = expert_linear_dimensions(gate);
    let up_dimensions = expert_linear_dimensions(up);
    let down_dimensions = expert_linear_dimensions(down);
    let (
        Some((gate_format, gate_out, gate_in)),
        Some((up_format, up_out, up_in)),
        Some((down_format, down_out, down_in)),
    ) = (gate_dimensions, up_dimensions, down_dimensions)
    else {
        return Err(Error::Model {
            message: format!(
                "{context} mixes opaque and typed linear formats: gate={gate:?} up={up:?} down={down:?}"
            ),
        });
    };
    if gate_format != up_format || gate_format != down_format {
        return Err(Error::Model {
            message: format!(
                "{context} mixes linear formats: gate={gate:?} up={up:?} down={down:?}"
            ),
        });
    }
    if (gate_out, gate_in) != (up_out, up_in) || down_in != gate_out || down_out != gate_in {
        return Err(Error::Model {
            message: format!(
                "{context} has inconsistent projection shapes: gate={gate_out}x{gate_in} up={up_out}x{up_in} down={down_out}x{down_in}"
            ),
        });
    }
    Ok(ExpertBundleLayout {
        format: gate_format,
        input_features: gate_in,
        intermediate_features: gate_out,
        output_features: down_out,
    })
}

fn expert_linear_dimensions(
    format: &ExpertLinearFormat,
) -> Option<(ExpertBundleFormat, usize, usize)> {
    match *format {
        ExpertLinearFormat::Bf16 {
            out_features,
            in_features,
        } => Some((ExpertBundleFormat::Bf16, out_features, in_features)),
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            ..
        } => Some((
            ExpertBundleFormat::Fp4E2M1PackedWithE8M0Scale,
            out_features,
            in_features,
        )),
        ExpertLinearFormat::Opaque => None,
    }
}

fn linear_payload_bytes(linear: &ExpertLinearPayload) -> u64 {
    linear.weight.bytes.len() as u64
        + linear
            .scale
            .as_ref()
            .map(|scale| scale.bytes.len() as u64)
            .unwrap_or(0)
}

fn expert_id_from_ref(value: &RoutedExpertTensorRef) -> ExpertId {
    ExpertId::new(value.layer, value.expert)
}

fn matrix_from_model(value: RoutedExpertMatrix) -> ExpertMatrixKind {
    match value {
        RoutedExpertMatrix::Gate => ExpertMatrixKind::Gate,
        RoutedExpertMatrix::Up => ExpertMatrixKind::Up,
        RoutedExpertMatrix::Down => ExpertMatrixKind::Down,
    }
}

fn component_from_model(value: RoutedExpertTensorPart) -> ExpertTensorComponent {
    match value {
        RoutedExpertTensorPart::Weight => ExpertTensorComponent::Weight,
        RoutedExpertTensorPart::Scale => ExpertTensorComponent::Scale,
        RoutedExpertTensorPart::Other(name) => ExpertTensorComponent::Other(name),
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage::ExpertStreamingReader;
    use super::super::tests::unique_temp_dir;
    use super::super::{ExpertStreamingPlanner, ExpertStreamingPolicy};
    use super::*;

    #[test]
    fn hf_catalog_uses_semantic_content_and_exact_source_identity() {
        let dir = unique_temp_dir("ferrule-expert-streaming-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "model-00001-of-00001.safetensors";
        let bytes = (0u8..80).collect::<Vec<_>>();
        std::fs::write(dir.join(shard), &bytes).unwrap();

        let tensors = vec![
            hf_tensor(
                0,
                3,
                RoutedExpertMatrix::Gate,
                RoutedExpertTensorPart::Weight,
                shard,
                8,
                4,
            ),
            hf_tensor(
                0,
                3,
                RoutedExpertMatrix::Gate,
                RoutedExpertTensorPart::Scale,
                shard,
                20,
                2,
            ),
            hf_tensor(
                0,
                3,
                RoutedExpertMatrix::Down,
                RoutedExpertTensorPart::Weight,
                shard,
                32,
                4,
            ),
        ];
        let catalog = Arc::new(
            ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(&dir, tensors.clone()).unwrap(),
        );
        assert_eq!(catalog.count(), 1);
        let expert = ExpertId::new(0, 3);
        let source = catalog.source(expert).unwrap();
        assert_eq!(source.bytes(), 10);
        let resource_source = catalog.require_resource_source(expert).unwrap();
        assert!(!resource_source.identity().is_zero());
        assert!(!resource_source.content_hash().is_zero());
        assert!(!resource_source.generation().is_zero());
        let materialization_sources = catalog.materialization_sources().unwrap();
        let resource =
            MaterializedResourceId::routed_expert(LayerId::new(0), ProtocolExpertId::new(3));
        let exact = materialization_sources
            .get(resource, resource_source)
            .expect("expert source must be indexed by exact materialization identity");
        assert_eq!(exact.storage_bytes(), 10);
        assert_eq!(exact.descriptor(), source);

        let mut planner = ExpertStreamingPlanner::from_catalog(
            ExpertStreamingPolicy::quality_first(1),
            Arc::clone(&catalog),
        );
        assert!(Arc::ptr_eq(planner.source_catalog(), &catalog));
        let step = planner.plan_layer_step(0, &[3], &[]).unwrap();
        let slices = if let ExpertLoadSource::HfLocalTensorSet { tensors, .. } =
            &step.loads[0].load_source
        {
            tensors
        } else {
            panic!(
                "expected snapshot-backed HF tensor set, got {:?}",
                step.loads[0].load_source
            );
        };
        assert_eq!(slices.len(), 3);
        let reader = ExpertStreamingReader::new(8);
        let payload = reader
            .read_load_source(expert, &step.loads[0].load_source)
            .unwrap();
        assert_eq!(payload.tensors[0].bytes, vec![8, 9, 10, 11]);
        assert_eq!(payload.tensors[1].bytes, vec![20, 21]);
        assert_eq!(payload.tensors[2].bytes, vec![32, 33, 34, 35]);

        let second_shard = "different-source.safetensors";
        let mut different_payload = bytes.clone();
        different_payload[8] ^= 0xff;
        std::fs::write(dir.join(second_shard), different_payload).unwrap();
        let same_descriptor_different_source = tensors
            .iter()
            .cloned()
            .map(|mut tensor| {
                tensor.shard = second_shard.into();
                tensor
            })
            .collect::<Vec<_>>();
        let second_catalog = ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(
            &dir,
            same_descriptor_different_source,
        )
        .unwrap();
        let second_source = second_catalog.require_resource_source(expert).unwrap();
        assert_eq!(resource_source.content_hash(), second_source.content_hash());
        assert_ne!(resource_source.identity(), second_source.identity());
        assert_ne!(resource_source.generation(), second_source.generation());

        let mut enlarged_payload = bytes.clone();
        enlarged_payload.push(80);
        std::fs::write(dir.join(shard), enlarged_payload).unwrap();
        assert_eq!(
            reader.validate_source_identity(source),
            Err(StaleReason::SourceIdentityChanged)
        );
        let stale_error = reader.read_load_source(expert, source).unwrap_err();
        assert!(
            stale_error
                .to_string()
                .contains("stale checkpoint source identity")
        );
        let changed_generation_catalog =
            ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(&dir, tensors.clone()).unwrap();
        let changed_generation_source = changed_generation_catalog
            .require_resource_source(expert)
            .unwrap();
        assert_eq!(
            resource_source.content_hash(),
            changed_generation_source.content_hash()
        );
        assert_ne!(
            resource_source.identity(),
            changed_generation_source.identity()
        );
        assert_ne!(
            resource_source.generation(),
            changed_generation_source.generation()
        );

        let mut changed_descriptor = tensors;
        changed_descriptor[0].dtype = "F8_E4M3".into();
        let changed_descriptor_catalog =
            ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(&dir, changed_descriptor)
                .unwrap();
        assert_ne!(
            resource_source.content_hash(),
            changed_descriptor_catalog
                .require_resource_source(expert)
                .unwrap()
                .content_hash()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_catalog_for_156_gib_descriptors_uses_one_metadata_snapshot_and_no_payload() {
        const LAYERS: usize = 43;
        const EXPERTS: usize = 256;
        const TENSORS_PER_EXPERT: usize = 6;
        const TOTAL_BYTES: u64 = 156 * 1024 * 1024 * 1024;
        const TENSOR_COUNT: usize = LAYERS * EXPERTS * TENSORS_PER_EXPERT;

        let model_dir = unique_temp_dir("ferrule-missing-156g-payload-test");
        let shard = "payload-does-not-exist.safetensors";
        let base_bytes = TOTAL_BYTES / TENSOR_COUNT as u64;
        let remainder = TOTAL_BYTES % TENSOR_COUNT as u64;
        let mut tensors = Vec::with_capacity(TENSOR_COUNT);
        let mut offset = 0u64;
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                for (matrix, part) in [
                    (RoutedExpertMatrix::Gate, RoutedExpertTensorPart::Weight),
                    (RoutedExpertMatrix::Gate, RoutedExpertTensorPart::Scale),
                    (RoutedExpertMatrix::Up, RoutedExpertTensorPart::Weight),
                    (RoutedExpertMatrix::Up, RoutedExpertTensorPart::Scale),
                    (RoutedExpertMatrix::Down, RoutedExpertTensorPart::Weight),
                    (RoutedExpertMatrix::Down, RoutedExpertTensorPart::Scale),
                ] {
                    let ordinal = tensors.len() as u64;
                    let bytes = base_bytes + u64::from(ordinal < remainder);
                    tensors.push(hf_tensor(layer, expert, matrix, part, shard, offset, bytes));
                    offset += bytes;
                }
            }
        }
        assert_eq!(offset, TOTAL_BYTES);
        assert!(!model_dir.join(shard).exists());

        let mut metadata_snapshots = 0usize;
        let catalog = ExpertSourceCatalog::from_hf_routed_expert_tensor_sets_with_source_identity(
            &model_dir,
            tensors,
            |catalog_path| {
                metadata_snapshots += 1;
                Ok(CheckpointSourceFileIdentity::for_test(
                    catalog_path.to_path_buf(),
                    PathBuf::from("/canonical/synthetic-156g.safetensors"),
                    TOTAL_BYTES,
                ))
            },
        )
        .unwrap();

        assert_eq!(metadata_snapshots, 1);
        assert_eq!(catalog.count(), LAYERS * EXPERTS);
        assert_eq!(
            catalog
                .iter()
                .map(|(_, source)| source.bytes())
                .sum::<u64>(),
            TOTAL_BYTES
        );
    }

    #[cfg(unix)]
    #[test]
    fn hf_catalog_does_not_require_payload_read_permission() {
        use std::os::unix::fs::PermissionsExt;

        let dir = unique_temp_dir("ferrule-permission-denied-catalog-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "unreadable.safetensors";
        let shard_path = dir.join(shard);
        std::fs::write(&shard_path, vec![0u8; 64]).unwrap();
        std::fs::set_permissions(&shard_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        assert_eq!(
            std::fs::metadata(&shard_path).unwrap().permissions().mode() & 0o444,
            0
        );

        let catalog = ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(
            &dir,
            [hf_tensor(
                0,
                0,
                RoutedExpertMatrix::Gate,
                RoutedExpertTensorPart::Weight,
                shard,
                8,
                16,
            )],
        )
        .unwrap();
        assert_eq!(catalog.count(), 1);

        std::fs::set_permissions(&shard_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_streaming_catalog_reads_unscaled_bf16_expert() {
        let dir = unique_temp_dir("ferrule-bf16-expert-catalog");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "experts.safetensors";
        std::fs::write(dir.join(shard), vec![0u8; 36]).unwrap();
        let expert = ExpertId::new(1, 4);
        let tensors = [
            hf_bf16_tensor(1, 4, RoutedExpertMatrix::Gate, shard, vec![2, 3], 0, 12),
            hf_bf16_tensor(1, 4, RoutedExpertMatrix::Up, shard, vec![2, 3], 12, 12),
            hf_bf16_tensor(1, 4, RoutedExpertMatrix::Down, shard, vec![3, 2], 24, 12),
        ];
        let catalog =
            ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(&dir, tensors).unwrap();
        let source = catalog.source(expert).unwrap();
        let artifact = ExpertStreamingReader::new(64)
            .read_load_source(expert, source)
            .unwrap();
        let bundle = ExpertComputeBundle::from_artifact_payload(artifact).unwrap();

        assert_eq!(
            bundle.gate.format,
            ExpertLinearFormat::Bf16 {
                out_features: 2,
                in_features: 3,
            }
        );
        assert_eq!(
            bundle.down.format,
            ExpertLinearFormat::Bf16 {
                out_features: 3,
                in_features: 2,
            }
        );
        assert!(bundle.gate.scale.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn builds_bf16_expert_compute_bundle_without_scale_slices() {
        let expert = ExpertId::new(1, 4);
        let payload = ExpertArtifactPayload {
            expert,
            tensors: vec![
                bf16_payload(expert, ExpertMatrixKind::Gate, vec![2, 3], vec![0; 12]),
                bf16_payload(expert, ExpertMatrixKind::Up, vec![2, 3], vec![0; 12]),
                bf16_payload(expert, ExpertMatrixKind::Down, vec![3, 2], vec![0; 12]),
            ],
        };
        let bundle = ExpertComputeBundle::from_artifact_payload(payload).unwrap();
        assert_eq!(
            bundle.gate.format,
            ExpertLinearFormat::Bf16 {
                out_features: 2,
                in_features: 3,
            }
        );
        assert_eq!(
            bundle.down.format,
            ExpertLinearFormat::Bf16 {
                out_features: 3,
                in_features: 2,
            }
        );
        assert!(bundle.gate.scale.is_none());
        assert_eq!(bundle.total_bytes(), 36);
    }

    #[test]
    fn rejects_malformed_bf16_expert_shape_and_byte_lengths() {
        let expert = ExpertId::new(0, 0);
        let malformed_shape = bf16_payload(expert, ExpertMatrixKind::Gate, vec![4], vec![0; 8]);
        let err = infer_linear_format(&malformed_shape, None).unwrap_err();
        assert!(err.to_string().contains("2D weight shape"));

        let zero_dim = bf16_payload(expert, ExpertMatrixKind::Gate, vec![0, 3], Vec::new());
        let err = infer_linear_format(&zero_dim, None).unwrap_err();
        assert!(err.to_string().contains("dimensions must be non-zero"));

        let short = bf16_payload(expert, ExpertMatrixKind::Gate, vec![2, 3], vec![0; 10]);
        let err = infer_linear_format(&short, None).unwrap_err();
        assert!(err.to_string().contains("byte length mismatch"));

        let weight = bf16_payload(expert, ExpertMatrixKind::Gate, vec![1, 1], vec![0; 2]);
        let scale = fp4_payload(
            expert,
            ExpertMatrixKind::Gate,
            ExpertTensorComponent::Scale,
            vec![1, 1],
            1,
        );
        let err = infer_linear_format(&weight, Some(&scale)).unwrap_err();
        assert!(err.to_string().contains("unexpected scale"));

        let overflow = ExpertTensorSlice {
            key: ExpertTensorKey::new(0, 0, ExpertMatrixKind::Gate),
            component: ExpertTensorComponent::Weight,
            path: PathBuf::from("synthetic.safetensors"),
            offset: 0,
            bytes: 0,
            dtype: "BF16".into(),
            shape: vec![usize::MAX, 2],
        };
        let err = infer_expert_linear_format(&overflow, 0, None).unwrap_err();
        assert!(err.to_string().contains("byte size overflows"));
    }

    #[test]
    fn builds_fp4_expert_compute_bundle_from_six_artifact_slices() {
        let expert = ExpertId::new(0, 3);
        let payload = ExpertArtifactPayload {
            expert,
            tensors: vec![
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Gate,
                    ExpertTensorComponent::Weight,
                    vec![32, 16],
                    32 * 16,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Gate,
                    ExpertTensorComponent::Scale,
                    vec![32, 1],
                    32,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Up,
                    ExpertTensorComponent::Weight,
                    vec![32, 16],
                    32 * 16,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Up,
                    ExpertTensorComponent::Scale,
                    vec![32, 1],
                    32,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Down,
                    ExpertTensorComponent::Weight,
                    vec![32, 16],
                    32 * 16,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Down,
                    ExpertTensorComponent::Scale,
                    vec![32, 1],
                    32,
                ),
            ],
        };
        let bundle = ExpertComputeBundle::from_artifact_payload(payload).unwrap();
        assert_eq!(bundle.expert, expert);
        assert_eq!(
            bundle.gate.format,
            ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 32,
                in_features: 32,
                block_size: 32,
            }
        );
        assert_eq!(
            bundle.down.format,
            ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 32,
                in_features: 32,
                block_size: 32,
            }
        );
        assert_eq!(bundle.total_bytes(), 3 * (32 * 16 + 32) as u64);
    }

    #[test]
    fn rejects_fp4_expert_bundle_with_bad_scale_shape() {
        let expert = ExpertId::new(0, 3);
        let payload = ExpertArtifactPayload {
            expert,
            tensors: vec![
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Gate,
                    ExpertTensorComponent::Weight,
                    vec![2048, 2048],
                    4,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Gate,
                    ExpertTensorComponent::Scale,
                    vec![2048, 127],
                    2,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Up,
                    ExpertTensorComponent::Weight,
                    vec![2048, 2048],
                    4,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Up,
                    ExpertTensorComponent::Scale,
                    vec![2048, 128],
                    2,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Down,
                    ExpertTensorComponent::Weight,
                    vec![4096, 1024],
                    4,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Down,
                    ExpertTensorComponent::Scale,
                    vec![4096, 64],
                    2,
                ),
            ],
        };
        let err = ExpertComputeBundle::from_artifact_payload(payload).unwrap_err();
        assert!(err.to_string().contains("scale shape mismatch"));
    }

    fn bf16_payload(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> ExpertTensorPayload {
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "BF16".into(),
                shape,
            },
            bytes,
        }
    }

    fn fp4_payload(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        component: ExpertTensorComponent,
        shape: Vec<usize>,
        len: usize,
    ) -> ExpertTensorPayload {
        let dtype = match component {
            ExpertTensorComponent::Weight => "I8",
            ExpertTensorComponent::Scale => "F8_E8M0",
            ExpertTensorComponent::Other(_) => "opaque",
        };
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: len as u64,
                dtype: dtype.into(),
                shape,
            },
            bytes: vec![1u8; len],
        }
    }

    fn hf_bf16_tensor(
        layer: usize,
        expert: usize,
        matrix: RoutedExpertMatrix,
        shard: &str,
        shape: Vec<usize>,
        file_offset: u64,
        byte_size: u64,
    ) -> HfRoutedExpertTensorInfo {
        HfRoutedExpertTensorInfo {
            descriptor: RoutedExpertTensorRef {
                layer,
                expert,
                matrix,
                part: RoutedExpertTensorPart::Weight,
            },
            name: format!("layers.{layer}.ffn.experts.{expert}.bf16"),
            shard: shard.into(),
            dtype: "BF16".into(),
            shape,
            data_offset: file_offset,
            file_offset,
            byte_size,
        }
    }

    fn hf_tensor(
        layer: usize,
        expert: usize,
        matrix: RoutedExpertMatrix,
        part: RoutedExpertTensorPart,
        shard: &str,
        file_offset: u64,
        byte_size: u64,
    ) -> HfRoutedExpertTensorInfo {
        HfRoutedExpertTensorInfo {
            descriptor: RoutedExpertTensorRef {
                layer,
                expert,
                matrix,
                part,
            },
            name: format!("layers.{layer}.ffn.experts.{expert}.synthetic"),
            shard: shard.into(),
            dtype: "I8".into(),
            shape: vec![byte_size as usize],
            data_offset: file_offset,
            file_offset,
            byte_size,
        }
    }

    #[test]
    fn source_lowering_is_metadata_only_and_preserves_opaque_chunk_semantics() {
        let path = unique_temp_dir("ferrule-expert-unopened-source").join("missing.bin");
        assert!(!path.exists());
        let expert = ExpertId::new(4, 9);
        let source = ExpertLoadSource::LocalShard {
            path: path.clone(),
            offset: 17,
            bytes: 5,
        };
        assert_eq!(
            slices_for_load_source(expert, &source).unwrap(),
            vec![ExpertTensorSlice {
                key: ExpertTensorKey {
                    expert,
                    matrix: ExpertMatrixKind::Gate
                },
                component: ExpertTensorComponent::Other("whole_expert_chunk".into()),
                path: path.clone(),
                offset: 17,
                bytes: 5,
                dtype: "opaque".into(),
                shape: Vec::new(),
            }]
        );
        for source in [
            ExpertLoadSource::GpuResident,
            ExpertLoadSource::CpuResident,
            ExpertLoadSource::WeightPackChunk {
                path,
                offset: 17,
                bytes: 5,
            },
            ExpertLoadSource::Remote {
                uri: "unopened://expert".into(),
                offset: 17,
                bytes: 5,
            },
        ] {
            let error = slices_for_load_source(expert, &source).unwrap_err();
            assert!(matches!(error, Error::Model { .. }));
            assert!(error.to_string().contains("does not support artifact tier"));
        }
    }
}
