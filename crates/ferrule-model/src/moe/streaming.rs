//! Expert streaming and residency planning.
//!
//! This module is intentionally model-family agnostic. A model adapter decides
//! which artifact tensors represent an expert; the runtime decides when an expert
//! should be GPU-resident, prefetched, evicted, or streamed from a slower tier.
//! Quality-first adapters can preserve artifact FP4/FP8 payloads and stream experts
//! instead of immediately re-quantizing them to fit.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::semantic::{RoutedExpertMatrix, RoutedExpertTensorPart, RoutedExpertTensorRef};
use crate::HfRoutedExpertTensorInfo;
use ferrule_common::{Error, Result};
use rayon::prelude::*;

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
    HostMmap,
    LocalStorage,
    Remote,
    Loading,
}

impl ExpertStorageTier {
    pub fn is_gpu_ready(self) -> bool {
        matches!(self, Self::Gpu)
    }

    pub fn is_streamable(self) -> bool {
        matches!(
            self,
            Self::Cpu | Self::HostMmap | Self::LocalStorage | Self::Remote
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLoadSource {
    GpuResident,
    CpuResident,
    HostMmap {
        artifact: PathBuf,
        offset: u64,
        bytes: u64,
    },
    LocalShard {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
    LocalTensorSet {
        tensors: Vec<ExpertTensorSlice>,
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
            Self::HostMmap { .. } => ExpertStorageTier::HostMmap,
            Self::LocalShard { .. }
            | Self::LocalTensorSet { .. }
            | Self::WeightPackChunk { .. } => ExpertStorageTier::LocalStorage,
            Self::Remote { .. } => ExpertStorageTier::Remote,
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Self::GpuResident | Self::CpuResident => 0,
            Self::HostMmap { bytes, .. }
            | Self::LocalShard { bytes, .. }
            | Self::WeightPackChunk { bytes, .. }
            | Self::Remote { bytes, .. } => *bytes,
            Self::LocalTensorSet { tensors } => tensors.iter().map(|tensor| tensor.bytes).sum(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLoadReason {
    Selected,
    Prefetch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLoadRequest {
    pub expert: ExpertId,
    pub load_source: ExpertLoadSource,
    pub reason: ExpertLoadReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertEvictRequest {
    pub expert: ExpertId,
    pub target: ExpertStorageTier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStreamingPolicy {
    /// Maximum concurrently GPU-resident experts per layer.
    ///
    /// For top-k MoE decode this must be at least `num_experts_per_tok` unless a
    /// later executor implements sequential per-expert load/compute/evict.
    pub gpu_slots_per_layer: usize,
    /// Best-effort predicted experts to load after selected experts are covered.
    pub prefetch_per_layer: usize,
    /// Keep the exact artifact payload/format; do not force a conversion policy here.
    pub preserve_artifact_quantization: bool,
    /// Whether CPU RAM staging is allowed. Disabling this models very constrained
    /// hosts where streaming should go directly from mmap/local/remote chunks.
    pub allow_cpu_staging: bool,
    /// Whether remote/object/LAN sources may satisfy expert loads.
    pub allow_remote_sources: bool,
}

impl ExpertStreamingPolicy {
    pub fn quality_first(num_experts_per_tok: usize) -> Self {
        let gpu_slots_per_layer = num_experts_per_tok
            .saturating_mul(2)
            .max(num_experts_per_tok);
        Self {
            gpu_slots_per_layer,
            prefetch_per_layer: num_experts_per_tok,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        }
    }

    pub fn quality_first_no_prefetch(num_experts_per_tok: usize) -> Self {
        Self {
            gpu_slots_per_layer: num_experts_per_tok
                .saturating_mul(8)
                .max(num_experts_per_tok),
            prefetch_per_layer: 0,
            ..Self::quality_first(num_experts_per_tok)
        }
    }

    pub fn quality_first_with_prefetch(
        num_experts_per_tok: usize,
        prefetch_per_layer: usize,
    ) -> Self {
        let no_prefetch_slots = num_experts_per_tok
            .saturating_mul(8)
            .max(num_experts_per_tok);
        let gpu_slots_per_layer = no_prefetch_slots.max(
            num_experts_per_tok
                .saturating_add(prefetch_per_layer)
                .max(num_experts_per_tok),
        );
        Self {
            gpu_slots_per_layer,
            prefetch_per_layer,
            ..Self::quality_first(num_experts_per_tok)
        }
    }

    pub fn quality_first_remote(num_experts_per_tok: usize) -> Self {
        Self {
            allow_remote_sources: true,
            ..Self::quality_first(num_experts_per_tok)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStreamingStep {
    pub layer: usize,
    pub selected: Vec<ExpertId>,
    pub prefetched: Vec<ExpertId>,
    pub loads: Vec<ExpertLoadRequest>,
    pub evictions: Vec<ExpertEvictRequest>,
}

impl ExpertStreamingStep {
    pub fn is_noop(&self) -> bool {
        self.loads.is_empty() && self.evictions.is_empty()
    }
}

#[derive(Debug, Clone)]
struct ExpertState {
    load_source: ExpertLoadSource,
    location: ExpertStorageTier,
    last_used_step: u64,
    selected_count: u64,
}

#[derive(Debug, Clone)]
pub struct ExpertStreamingPlanner {
    policy: ExpertStreamingPolicy,
    experts: BTreeMap<ExpertId, ExpertState>,
    step: u64,
}

impl ExpertStreamingPlanner {
    pub fn new(policy: ExpertStreamingPolicy) -> Self {
        Self {
            policy,
            experts: BTreeMap::new(),
            step: 0,
        }
    }

    pub fn policy(&self) -> &ExpertStreamingPolicy {
        &self.policy
    }

    pub fn register_load_source(&mut self, expert: ExpertId, load_source: ExpertLoadSource) {
        let location = load_source.tier();
        self.experts.insert(
            expert,
            ExpertState {
                load_source,
                location,
                last_used_step: 0,
                selected_count: 0,
            },
        );
    }

    pub fn register_hf_routed_expert_tensor_sets(
        &mut self,
        model_dir: &Path,
        tensors: impl IntoIterator<Item = HfRoutedExpertTensorInfo>,
    ) -> Result<usize> {
        let mut grouped = BTreeMap::<ExpertId, Vec<ExpertTensorSlice>>::new();
        for tensor in tensors {
            let expert = expert_id_from_ref(&tensor.descriptor);
            grouped.entry(expert).or_default().push(ExpertTensorSlice {
                key: ExpertTensorKey {
                    expert,
                    matrix: matrix_from_model(tensor.descriptor.matrix),
                },
                component: component_from_model(tensor.descriptor.part),
                path: model_dir.join(&tensor.shard),
                offset: tensor.file_offset,
                bytes: tensor.byte_size,
                dtype: tensor.dtype,
                shape: tensor.shape,
            });
        }

        let count = grouped.len();
        for (expert, mut tensors) in grouped {
            tensors.sort_by(|a, b| {
                a.key
                    .matrix
                    .cmp(&b.key.matrix)
                    .then_with(|| a.component.cmp(&b.component))
                    .then_with(|| a.path.cmp(&b.path))
                    .then_with(|| a.offset.cmp(&b.offset))
            });
            if tensors.is_empty() {
                return Err(Error::Model(format!(
                    "empty expert tensor set for layer {} expert {}",
                    expert.layer, expert.expert
                )));
            }
            self.register_load_source(expert, ExpertLoadSource::LocalTensorSet { tensors });
        }
        Ok(count)
    }

    pub fn mark_resident(&mut self, expert: ExpertId, location: ExpertStorageTier) -> Result<()> {
        let state = self.experts.get_mut(&expert).ok_or_else(|| {
            Error::Model(format!(
                "expert streaming load source missing for layer {} expert {}",
                expert.layer, expert.expert
            ))
        })?;
        state.location = location;
        Ok(())
    }

    pub fn location(&self, expert: ExpertId) -> Option<ExpertStorageTier> {
        self.experts.get(&expert).map(|state| state.location)
    }

    pub fn resident_experts(&self, layer: usize) -> Vec<ExpertId> {
        self.experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.location.is_gpu_ready()).then_some(*expert)
            })
            .collect()
    }

    fn resident_experts_by_hotness(&self, layer: usize) -> Vec<ExpertId> {
        let mut experts = self
            .experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.location.is_gpu_ready()).then_some((
                    *expert,
                    state.selected_count,
                    state.last_used_step,
                ))
            })
            .collect::<Vec<_>>();
        experts.sort_by(
            |(left, left_count, left_step), (right, right_count, right_step)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| right_step.cmp(left_step))
                    .then_with(|| left.expert.cmp(&right.expert))
            },
        );
        experts.into_iter().map(|(expert, _, _)| expert).collect()
    }

    /// Per-layer routing-aware hotset, ordered from hottest to coldest.
    ///
    /// This intentionally only returns experts that have actually been selected
    /// before. It avoids the old naive `0..N` prefetch pattern, which is not
    /// correlated with DSV4 routing and can increase residency churn.
    pub fn hot_experts(&self, layer: usize, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let mut experts = self
            .experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.selected_count > 0).then_some((
                    expert.expert,
                    state.selected_count,
                    state.last_used_step,
                ))
            })
            .collect::<Vec<_>>();
        experts.sort_by(
            |(left, left_count, left_step), (right, right_count, right_step)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| right_step.cmp(left_step))
                    .then_with(|| left.cmp(right))
            },
        );
        experts
            .into_iter()
            .take(count)
            .map(|(expert, _, _)| expert)
            .collect()
    }

    pub fn plan_layer_step(
        &mut self,
        layer: usize,
        selected: &[usize],
        predicted: &[usize],
    ) -> Result<ExpertStreamingStep> {
        self.step = self.step.saturating_add(1);
        let selected = unique_ids(layer, selected);
        if selected.len() > self.policy.gpu_slots_per_layer {
            return Err(Error::Model(format!(
                "expert streaming policy has {} GPU slots for layer {}, but {} selected experts must be available",
                self.policy.gpu_slots_per_layer,
                layer,
                selected.len()
            )));
        }
        for expert in &selected {
            if let Some(state) = self.experts.get_mut(expert) {
                state.selected_count = state.selected_count.saturating_add(1);
                state.last_used_step = self.step;
            }
        }

        let mut target = selected.iter().copied().collect::<BTreeSet<_>>();
        let mut prefetched = Vec::new();
        for expert in unique_ids(layer, predicted) {
            if target.contains(&expert) {
                continue;
            }
            if prefetched.len() >= self.policy.prefetch_per_layer {
                break;
            }
            if target.len() >= self.policy.gpu_slots_per_layer {
                break;
            }
            target.insert(expert);
            prefetched.push(expert);
        }

        let mut current_gpu = self.resident_experts_by_hotness(layer);
        for expert in current_gpu.iter().copied() {
            if target.len() >= self.policy.gpu_slots_per_layer {
                break;
            }
            target.insert(expert);
        }

        let mut evictions = Vec::new();
        for expert in current_gpu.drain(..) {
            if !target.contains(&expert) {
                evictions.push(ExpertEvictRequest {
                    expert,
                    target: self.load_source_tier_or_local(expert)?,
                });
            }
        }

        let mut loads = Vec::new();
        for expert in &selected {
            self.ensure_load_source_allowed(*expert)?;
            if !matches!(self.location(*expert), Some(ExpertStorageTier::Gpu)) {
                loads.push(ExpertLoadRequest {
                    expert: *expert,
                    load_source: self.load_source_for(*expert)?.clone(),
                    reason: ExpertLoadReason::Selected,
                });
            }
        }
        for expert in &prefetched {
            self.ensure_load_source_allowed(*expert)?;
            if !matches!(self.location(*expert), Some(ExpertStorageTier::Gpu)) {
                loads.push(ExpertLoadRequest {
                    expert: *expert,
                    load_source: self.load_source_for(*expert)?.clone(),
                    reason: ExpertLoadReason::Prefetch,
                });
            }
        }

        Ok(ExpertStreamingStep {
            layer,
            selected,
            prefetched,
            loads,
            evictions,
        })
    }

    pub fn commit_step(&mut self, step: &ExpertStreamingStep) -> Result<()> {
        for eviction in &step.evictions {
            self.mark_resident(eviction.expert, eviction.target)?;
        }
        for load in &step.loads {
            self.mark_resident(load.expert, ExpertStorageTier::Gpu)?;
        }
        Ok(())
    }

    fn load_source_for(&self, expert: ExpertId) -> Result<&ExpertLoadSource> {
        self.experts
            .get(&expert)
            .map(|state| &state.load_source)
            .ok_or_else(|| {
                Error::Model(format!(
                    "expert streaming load source missing for layer {} expert {}",
                    expert.layer, expert.expert
                ))
            })
    }

    fn load_source_tier_or_local(&self, expert: ExpertId) -> Result<ExpertStorageTier> {
        let tier = self.load_source_for(expert)?.tier();
        Ok(if tier == ExpertStorageTier::Gpu {
            ExpertStorageTier::LocalStorage
        } else {
            tier
        })
    }

    fn ensure_load_source_allowed(&self, expert: ExpertId) -> Result<()> {
        let load_source = self.load_source_for(expert)?;
        let tier = load_source.tier();
        if !tier.is_streamable() && tier != ExpertStorageTier::Gpu {
            return Err(Error::Model(format!(
                "expert load source for layer {} expert {} is not streamable: {:?}",
                expert.layer, expert.expert, tier
            )));
        }
        if tier == ExpertStorageTier::Remote && !self.policy.allow_remote_sources {
            return Err(Error::Model(format!(
                "remote expert load source for layer {} expert {} requires allow_remote_sources=true",
                expert.layer, expert.expert
            )));
        }
        if tier == ExpertStorageTier::Cpu && !self.policy.allow_cpu_staging {
            return Err(Error::Model(format!(
                "CPU-staged expert load source for layer {} expert {} requires allow_cpu_staging=true",
                expert.layer, expert.expert
            )));
        }
        Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLinearFormat {
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
                return Err(Error::Model(format!(
                    "expert payload contains mismatched tensor: expected layer {} expert {}, got layer {} expert {}",
                    expert.layer,
                    expert.expert,
                    tensor.slice.key.expert.layer,
                    tensor.slice.key.expert.expert
                )));
            }
            grouped
                .entry(tensor.slice.key.matrix)
                .or_default()
                .push(tensor);
        }
        Ok(Self {
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
        })
    }

    pub fn total_bytes(&self) -> u64 {
        linear_payload_bytes(&self.gate)
            .saturating_add(linear_payload_bytes(&self.up))
            .saturating_add(linear_payload_bytes(&self.down))
    }
}

/// Cache of mmap'd safetensors files, keyed by path.
/// The OS page cache provides automatic caching of file contents across reads.
/// Once a page is faulted in, subsequent reads of the same region are served
/// from the page cache without disk I/O.
static MMAP_CACHE: Mutex<Option<HashMap<PathBuf, Arc<memmap2::Mmap>>>> = Mutex::new(None);

fn get_or_open_mmap(path: &Path) -> Result<Arc<memmap2::Mmap>> {
    let mut guard = MMAP_CACHE.lock().expect("mmap cache poisoned");
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    let cache = guard.as_mut().expect("initialized above");
    if let Some(mmap) = cache.get(path) {
        return Ok(mmap.clone());
    }
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Model(format!("expert mmap open '{}': {e}", path.display())))?;
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::Model(format!("expert mmap map '{}': {e}", path.display())))?
    };
    let mmap = Arc::new(mmap);
    cache.insert(path.to_path_buf(), mmap.clone());
    Ok(mmap)
}

#[derive(Debug, Clone)]
pub struct ExpertStreamingReader {
    max_slice_bytes: u64,
    /// When true, reads use mmap'd file regions instead of pread.
    /// The mmap cache is shared across all reader instances (static).
    use_mmap: bool,
}

impl ExpertStreamingReader {
    pub fn new(max_slice_bytes: u64) -> Self {
        Self {
            max_slice_bytes,
            use_mmap: true,
        }
    }

    pub fn max_slice_bytes(&self) -> u64 {
        self.max_slice_bytes
    }

    pub fn read_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        let tensors = match load_source {
            ExpertLoadSource::LocalTensorSet { tensors } => tensors
                .iter()
                .map(Self::read_local_slice_positioned)
                .collect::<Result<Vec<_>>>()?,
            ExpertLoadSource::LocalShard {
                path,
                offset,
                bytes,
            } => {
                let slice = ExpertTensorSlice {
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
                };
                vec![Self::read_local_slice_positioned(&slice)?]
            }
            other => {
                return Err(Error::Model(format!(
                    "expert streaming reader does not support artifact tier {:?} yet",
                    other.tier()
                )))
            }
        };
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    pub fn read_local_slice(&self, slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        if slice.bytes > self.max_slice_bytes {
            return Err(Error::Model(format!(
                "expert tensor slice exceeds bounded read size: {} > {} bytes",
                slice.bytes, self.max_slice_bytes
            )));
        }
        let mut file = std::fs::File::open(&slice.path).map_err(|e| {
            Error::Model(format!(
                "expert tensor slice open '{}': {e}",
                slice.path.display()
            ))
        })?;
        file.seek(SeekFrom::Start(slice.offset)).map_err(|e| {
            Error::Model(format!(
                "expert tensor slice seek '{}': {e}",
                slice.path.display()
            ))
        })?;
        let mut bytes = vec![0u8; slice.bytes as usize];
        file.read_exact(&mut bytes).map_err(|e| {
            Error::Model(format!(
                "expert tensor slice read '{}': {e}",
                slice.path.display()
            ))
        })?;
        Ok(ExpertTensorPayload {
            slice: slice.clone(),
            bytes,
        })
    }

    /// Read a single tensor slice using positioned read (pread) — no seek needed.
    /// Uses `std::os::unix::fs::FileExt::read_exact_at` on Unix for a single
    /// syscall instead of open+seek+read (3 syscalls).
    ///
    /// When mmap is enabled, reads from the mmap'd file region instead.
    /// The OS page cache provides automatic caching across reads.
    fn read_local_slice_positioned(slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        Self::read_local_slice_positioned_with_mmap(slice, true)
    }

    fn read_local_slice_positioned_with_mmap(
        slice: &ExpertTensorSlice,
        use_mmap: bool,
    ) -> Result<ExpertTensorPayload> {
        if use_mmap {
            // Try mmap path first — OS page cache handles caching.
            match get_or_open_mmap(&slice.path) {
                Ok(mmap) => {
                    let end = slice.offset as usize + slice.bytes as usize;
                    if end > mmap.len() {
                        return Err(Error::Model(format!(
                            "expert mmap slice out of bounds: '{}': offset {} + bytes {} > mmap len {}",
                            slice.path.display(),
                            slice.offset,
                            slice.bytes,
                            mmap.len()
                        )));
                    }
                    // Copy from mmap'd region — OS page cache will serve repeated reads.
                    let bytes = mmap[slice.offset as usize..end].to_vec();
                    return Ok(ExpertTensorPayload {
                        slice: slice.clone(),
                        bytes,
                    });
                }
                Err(e) => {
                    // Fall back to pread if mmap fails (e.g. special filesystem).
                    tracing::trace!(
                        "expert mmap failed for '{}': {e}, falling back to pread",
                        slice.path.display()
                    );
                }
            }
        }

        // Fallback: positioned read (pread).
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(&slice.path).map_err(|e| {
            Error::Model(format!(
                "expert tensor slice open '{}': {e}",
                slice.path.display()
            ))
        })?;
        let mut bytes = vec![0u8; slice.bytes as usize];
        file.read_exact_at(&mut bytes, slice.offset).map_err(|e| {
            Error::Model(format!(
                "expert tensor slice read_at '{}': {e}",
                slice.path.display()
            ))
        })?;
        Ok(ExpertTensorPayload {
            slice: slice.clone(),
            bytes,
        })
    }

    /// Read all tensor slices for one expert concurrently using a tokio
    /// blocking thread pool. Each slice uses positioned read (pread) so
    /// multiple slices from the same file can be read in parallel without
    /// seeking.
    pub fn read_load_source_concurrent(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        let slices: Vec<ExpertTensorSlice> = match load_source {
            ExpertLoadSource::LocalTensorSet { tensors } => tensors.clone(),
            ExpertLoadSource::LocalShard {
                path,
                offset,
                bytes,
            } => {
                vec![ExpertTensorSlice {
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
                }]
            }
            other => {
                return Err(Error::Model(format!(
                    "expert streaming reader does not support artifact tier {:?} yet",
                    other.tier()
                )))
            }
        };
        for slice in &slices {
            if slice.bytes > self.max_slice_bytes {
                return Err(Error::Model(format!(
                    "expert tensor slice exceeds bounded read size: {} > {} bytes",
                    slice.bytes, self.max_slice_bytes
                )));
            }
        }

        // Fast path: single slice, no need for tokio overhead.
        if slices.len() <= 1 {
            let tensors = slices
                .iter()
                .map(|s| Self::read_local_slice_positioned_with_mmap(s, self.use_mmap))
                .collect::<Result<Vec<_>>>()?;
            return Ok(ExpertArtifactPayload { expert, tensors });
        }

        // Parallel path: use rayon for concurrent slice reads.
        let use_mmap = self.use_mmap;
        let results: Vec<Result<ExpertTensorPayload>> = slices
            .par_iter()
            .map(|s| Self::read_local_slice_positioned_with_mmap(s, use_mmap))
            .collect();
        let tensors = results.into_iter().collect::<Result<Vec<_>>>()?;
        Ok(ExpertArtifactPayload { expert, tensors })
    }
}

/// Read multiple experts concurrently. Each expert's slices are read in
/// parallel, and multiple experts are read in parallel too.
pub fn read_experts_concurrent(
    reader: &ExpertStreamingReader,
    loads: &[(ExpertId, ExpertLoadSource)],
) -> Result<Vec<ExpertArtifactPayload>> {
    if loads.is_empty() {
        return Ok(Vec::new());
    }
    // Fast path: single expert.
    if loads.len() == 1 {
        let (expert, source) = &loads[0];
        return Ok(vec![reader.read_load_source_concurrent(*expert, source)?]);
    }

    // Parallel path: use rayon scope (no runtime creation overhead).
    // The previous tokio-based code created a new runtime per call, which
    // added ~1ms overhead per layer per token.
    let results: Vec<Result<ExpertArtifactPayload>> = loads
        .par_iter()
        .map(|(expert, source)| {
            let r = ExpertStreamingReader::new(reader.max_slice_bytes);
            r.read_load_source_concurrent(*expert, source)
        })
        .collect();
    results.into_iter().collect()
}

fn build_linear_payload(
    expert: ExpertId,
    matrix: ExpertMatrixKind,
    tensors: Option<Vec<ExpertTensorPayload>>,
) -> Result<ExpertLinearPayload> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!(
            "expert artifact bundle missing {:?} matrix for layer {} expert {}",
            matrix, expert.layer, expert.expert
        ))
    })?;
    let mut weight = None;
    let mut scale = None;
    for tensor in tensors {
        match tensor.slice.component {
            ExpertTensorComponent::Weight => {
                if weight.replace(tensor).is_some() {
                    return Err(Error::Model(format!(
                        "expert artifact bundle has duplicate {:?} weight for layer {} expert {}",
                        matrix, expert.layer, expert.expert
                    )));
                }
            }
            ExpertTensorComponent::Scale => {
                if scale.replace(tensor).is_some() {
                    return Err(Error::Model(format!(
                        "expert artifact bundle has duplicate {:?} scale for layer {} expert {}",
                        matrix, expert.layer, expert.expert
                    )));
                }
            }
            ExpertTensorComponent::Other(name) => {
                return Err(Error::Model(format!(
                    "expert artifact bundle has unsupported {:?} component '{}' for layer {} expert {}",
                    matrix, name, expert.layer, expert.expert
                )));
            }
        }
    }
    let weight = weight.ok_or_else(|| {
        Error::Model(format!(
            "expert artifact bundle missing {:?} weight for layer {} expert {}",
            matrix, expert.layer, expert.expert
        ))
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
    let Some(scale) = scale else {
        return Ok(ExpertLinearFormat::Opaque);
    };
    if weight.slice.dtype == "I8" && scale.slice.dtype == "F8_E8M0" {
        if weight.slice.shape.len() != 2 || scale.slice.shape.len() != 2 {
            return Err(Error::Model(format!(
                "FP4 expert tensor expects 2D weight/scale shapes, got {:?} and {:?}",
                weight.slice.shape, scale.slice.shape
            )));
        }
        let out = weight.slice.shape[0];
        let packed_in = weight.slice.shape[1];
        let logical_in = packed_in
            .checked_mul(2)
            .ok_or_else(|| Error::Model("FP4 expert packed input dimension overflow".into()))?;
        let expected_scale_cols = logical_in / 32;
        if scale.slice.shape[0] != out || scale.slice.shape[1] != expected_scale_cols {
            return Err(Error::Model(format!(
                "FP4 expert scale shape mismatch: weight {:?} implies scale [{out}, {expected_scale_cols}], got {:?}",
                weight.slice.shape, scale.slice.shape
            )));
        }
        if weight.bytes.len() as u64 != weight.slice.bytes
            || scale.bytes.len() as u64 != scale.slice.bytes
        {
            return Err(Error::Model(
                "expert payload byte length does not match tensor slice metadata".into(),
            ));
        }
        return Ok(ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features: out,
            in_features: logical_in,
            block_size: 32,
        });
    }
    Ok(ExpertLinearFormat::Opaque)
}

fn linear_payload_bytes(linear: &ExpertLinearPayload) -> u64 {
    linear.weight.bytes.len() as u64
        + linear
            .scale
            .as_ref()
            .map(|scale| scale.bytes.len() as u64)
            .unwrap_or(0)
}

fn unique_ids(layer: usize, experts: &[usize]) -> Vec<ExpertId> {
    experts
        .iter()
        .copied()
        .map(|expert| ExpertId::new(layer, expert))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
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

// ── HostStagedExpertCache ──────────────────────────────────────────────────

/// A simple LRU cache of expert compute bundles staged in host RAM.
///
/// This avoids repeated disk reads for experts that are selected across
/// consecutive decode tokens but evicted from GPU memory between uses.
///
/// This directly targets the synthetic-fixture bottleneck of 243 disk reads
/// across 5 forward passes (ROADMAP P4).
#[derive(Debug)]
pub struct HostStagedExpertCache {
    entries: HashMap<ExpertId, ExpertComputeBundle>,
    order: VecDeque<ExpertId>,
    capacity: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl HostStagedExpertCache {
    /// Create a cache holding at most `capacity` expert bundles in host RAM.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a staged bundle, returning a clone if present.
    ///
    /// Cloning is intentional: the caller (CUDA upload path) needs an owned
    /// bundle to upload, while the cache retains its copy for future hits.
    /// Bundles are small (a few MB each for FP4 experts).
    pub fn get(&mut self, expert: ExpertId) -> Option<ExpertComputeBundle> {
        if self.entries.contains_key(&expert) {
            self.hits += 1;
            // Move to back (most-recently-used).
            self.order.retain(|id| *id != expert);
            self.order.push_back(expert);
            self.entries.get(&expert).cloned()
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a bundle, evicting the least-recently-used entry if at capacity.
    pub fn insert(&mut self, bundle: ExpertComputeBundle) {
        if self.capacity == 0 {
            return;
        }
        let expert = bundle.expert;
        if self.entries.contains_key(&expert) {
            self.order.retain(|id| *id != expert);
        } else if self.entries.len() >= self.capacity && self.capacity > 0 {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
                self.evictions += 1;
            }
        }
        self.entries.insert(expert, bundle);
        self.order.push_back(expert);
    }

    /// Load an expert bundle from disk via the reader, caching the result.
    ///
    /// On cache hit, skips the disk read entirely. On miss, reads from disk
    /// and inserts into the cache before returning.
    pub fn get_or_load(
        &mut self,
        expert: ExpertId,
        source: &ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> Result<ExpertComputeBundle> {
        if let Some(bundle) = self.get(expert) {
            return Ok(bundle);
        }
        let payload = reader.read_load_source(expert, source)?;
        let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
        self.insert(bundle.clone());
        Ok(bundle)
    }

    /// Number of currently staged bundles.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit count (experts served from host memory).
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count (experts that required a disk read).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Number of host-side evictions (LRU replacements).
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Total bytes of all staged bundles.
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|b| b.total_bytes()).sum()
    }
}

impl Default for HostStagedExpertCache {
    fn default() -> Self {
        // Default capacity: 256 expert bundles. At ~13 MB per FP4 expert
        // (gate+up+down for DeepSeek-V4 Flash), this is ~3.4 GB host RAM.
        // Larger cache reduces disk re-reads across decode tokens with
        // different expert routing.
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_first_policy_preserves_artifact_quantization() {
        let policy = ExpertStreamingPolicy::quality_first(6);
        assert!(policy.preserve_artifact_quantization);
        assert_eq!(policy.gpu_slots_per_layer, 12);
        assert_eq!(policy.prefetch_per_layer, 6);
        assert!(!policy.allow_cpu_staging);
        assert!(!policy.allow_remote_sources);
    }

    #[test]
    fn registers_hf_tensor_sets_and_reads_exact_slices() {
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
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
        assert_eq!(
            planner
                .register_hf_routed_expert_tensor_sets(&dir, tensors)
                .unwrap(),
            1
        );

        let step = planner.plan_layer_step(0, &[3], &[]).unwrap();
        assert_eq!(step.loads.len(), 1);
        let ExpertLoadSource::LocalTensorSet { tensors } = &step.loads[0].load_source else {
            panic!("expected LocalTensorSet load source");
        };
        assert_eq!(tensors.len(), 3);
        assert_eq!(step.loads[0].load_source.bytes(), 10);

        let reader = ExpertStreamingReader::new(8);
        let payload = reader
            .read_load_source(ExpertId::new(0, 3), &step.loads[0].load_source)
            .unwrap();
        assert_eq!(payload.tensors.len(), 3);
        assert_eq!(payload.tensors[0].bytes, vec![8, 9, 10, 11]);
        assert_eq!(payload.tensors[1].bytes, vec![20, 21]);
        assert_eq!(payload.tensors[2].bytes, vec![32, 33, 34, 35]);

        let _ = std::fs::remove_dir_all(&dir);
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
                    vec![2048, 2048],
                    4,
                ),
                fp4_payload(
                    expert,
                    ExpertMatrixKind::Gate,
                    ExpertTensorComponent::Scale,
                    vec![2048, 128],
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
        let bundle = ExpertComputeBundle::from_artifact_payload(payload).unwrap();
        assert_eq!(bundle.expert, expert);
        assert_eq!(
            bundle.gate.format,
            ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 2048,
                in_features: 4096,
                block_size: 32,
            }
        );
        assert_eq!(
            bundle.down.format,
            ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 4096,
                in_features: 2048,
                block_size: 32,
            }
        );
        assert_eq!(bundle.total_bytes(), 18);
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

    #[test]
    fn reader_rejects_slices_larger_than_bound() {
        let dir = unique_temp_dir("ferrule-expert-streaming-bound-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("slice.bin");
        std::fs::write(&shard, vec![0u8; 16]).unwrap();
        let slice = ExpertTensorSlice {
            key: ExpertTensorKey::new(0, 0, ExpertMatrixKind::Gate),
            component: ExpertTensorComponent::Weight,
            path: shard,
            offset: 0,
            bytes: 16,
            dtype: "I8".into(),
            shape: vec![16],
        };
        let err = ExpertStreamingReader::new(8)
            .read_local_slice(&slice)
            .unwrap_err();
        assert!(err.to_string().contains("bounded read size"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_selected_experts_from_local_shards_without_cpu_staging() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(2));
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from(format!("shard-{expert}.safetensors")),
                    offset: expert as u64 * 1024,
                    bytes: 1024,
                },
            );
        }

        let step = planner.plan_layer_step(0, &[1, 3], &[]).unwrap();
        assert_eq!(
            step.selected,
            vec![ExpertId::new(0, 1), ExpertId::new(0, 3)]
        );
        assert_eq!(step.loads.len(), 2);
        assert!(step
            .loads
            .iter()
            .all(|load| load.reason == ExpertLoadReason::Selected));
        planner.commit_step(&step).unwrap();
        assert_eq!(
            planner.location(ExpertId::new(0, 1)),
            Some(ExpertStorageTier::Gpu)
        );
        assert_eq!(
            planner.location(ExpertId::new(0, 3)),
            Some(ExpertStorageTier::Gpu)
        );
    }

    #[test]
    fn prefetch_uses_remaining_slots_but_never_duplicates_selected() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 3,
            prefetch_per_layer: 2,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..5 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let step = planner.plan_layer_step(0, &[2, 2], &[2, 3, 4]).unwrap();
        assert_eq!(step.selected, vec![ExpertId::new(0, 2)]);
        assert_eq!(
            step.prefetched,
            vec![ExpertId::new(0, 3), ExpertId::new(0, 4)]
        );
        assert_eq!(step.loads.len(), 3);
    }

    #[test]
    fn retains_recent_resident_experts_when_slots_are_available() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let first = planner.plan_layer_step(0, &[0], &[]).unwrap();
        planner.commit_step(&first).unwrap();
        let second = planner.plan_layer_step(0, &[2], &[]).unwrap();

        assert_eq!(second.loads.len(), 1);
        assert!(second.evictions.is_empty());
        planner.commit_step(&second).unwrap();
        assert_eq!(
            planner.resident_experts(0),
            vec![ExpertId::new(0, 0), ExpertId::new(0, 2)]
        );
    }

    #[test]
    fn hotset_reports_observed_experts_by_frequency() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 3,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let first = planner.plan_layer_step(0, &[1, 2], &[]).unwrap();
        planner.commit_step(&first).unwrap();
        let second = planner.plan_layer_step(0, &[1], &[]).unwrap();
        planner.commit_step(&second).unwrap();

        assert_eq!(planner.hot_experts(0, 3), vec![1, 2]);
        assert!(planner.hot_experts(1, 3).is_empty());
    }

    #[test]
    fn eviction_keeps_hot_resident_experts_before_cold_recent_ones() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        for selected in [[0], [0], [1]] {
            let step = planner.plan_layer_step(0, &selected, &[]).unwrap();
            planner.commit_step(&step).unwrap();
        }
        let step = planner.plan_layer_step(0, &[2], &[]).unwrap();

        assert_eq!(step.loads.len(), 1);
        assert_eq!(step.evictions.len(), 1);
        assert_eq!(step.evictions[0].expert, ExpertId::new(0, 1));
    }

    #[test]
    fn evicts_non_target_gpu_experts_to_artifact_tier() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }
        planner
            .mark_resident(ExpertId::new(0, 0), ExpertStorageTier::Gpu)
            .unwrap();
        planner
            .mark_resident(ExpertId::new(0, 1), ExpertStorageTier::Gpu)
            .unwrap();

        let step = planner.plan_layer_step(0, &[2, 3], &[]).unwrap();
        assert_eq!(step.loads.len(), 2);
        assert_eq!(step.evictions.len(), 2);
        assert!(step
            .evictions
            .iter()
            .all(|evict| evict.target == ExpertStorageTier::LocalStorage));
        planner.commit_step(&step).unwrap();
        assert_eq!(
            planner.resident_experts(0),
            vec![ExpertId::new(0, 2), ExpertId::new(0, 3)]
        );
    }

    #[test]
    fn rejects_remote_sources_until_policy_allows_them() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
        planner.register_load_source(
            ExpertId::new(0, 7),
            ExpertLoadSource::Remote {
                uri: "http://lan-node/experts/l0e7".into(),
                offset: 0,
                bytes: 2048,
            },
        );
        let err = planner.plan_layer_step(0, &[7], &[]).unwrap_err();
        assert!(err.to_string().contains("allow_remote_sources=true"));
    }

    #[test]
    fn supports_remote_sources_for_no_local_storage_mode() {
        let mut planner =
            ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first_remote(1));
        planner.register_load_source(
            ExpertId::new(0, 7),
            ExpertLoadSource::Remote {
                uri: "http://lan-node/experts/l0e7".into(),
                offset: 0,
                bytes: 2048,
            },
        );
        let step = planner.plan_layer_step(0, &[7], &[]).unwrap();
        assert_eq!(step.loads.len(), 1);
        assert_eq!(step.loads[0].load_source.tier(), ExpertStorageTier::Remote);
    }

    #[test]
    fn errors_when_selected_experts_exceed_concurrent_slots() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 1,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..2 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }
        let err = planner.plan_layer_step(0, &[0, 1], &[]).unwrap_err();
        assert!(err
            .to_string()
            .contains("selected experts must be available"));
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

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
