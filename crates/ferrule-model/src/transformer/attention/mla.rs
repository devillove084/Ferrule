//! Composable multi-latent attention, sequence state, and physical KV payloads.
//!
//! Generic transaction, page custody, COW, and provisional ownership live exclusively
//! in [`crate::decoder::PagedKvBackend`]. This module owns only MLA payload planes,
//! views, recurrent attachments, and prefix checkpoints.
#[cfg(feature = "cuda")]
use crate::attention_backend::SparseAttentionSpec;
use crate::checkpoint::LinearWeight;

use crate::decoder::{
    DecoderKvPrepare, DecoderSequenceCheckout, DecoderSequenceLifecycle, KvEndProgress,
    PackedDecoderBatch, PhysicalKvPool,
};
use crate::execution::SequenceTopologyId;
#[cfg(feature = "cuda")]
use crate::runner::SequenceStateReleaseError;
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::kv::{
    CudaCompressorRecurrentCheckpointSlab, CudaCompressorRecurrentState, KvPoolReservation,
};
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::linear::PROPOSAL_ROWS;
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::{
    attention::{
        self as cuda_attention, HybridMlaAttentionLayout,
        sparse::{DualPlanePagedSparseAttentionLayout, PagedSparseAttentionLayout},
    },
    linear as cuda_linear,
};
use ferrule_common::execution::{
    ExecutionIntent, ExecutionTransactionId, ForwardPhase, KvElementType, KvLayoutSchema, KvPageId,
    KvPlaneDescriptor,
};
use ferrule_common::{Error, Result};
#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use std::collections::{BTreeSet, HashMap};
#[cfg(feature = "cuda")]
use std::rc::Rc;
#[cfg(feature = "cuda")]
use std::time::Instant;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlaRopeConfig {
    pub theta: f32,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlaConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratio: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
}

impl MlaConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_heads == 0
            || self.head_dim == 0
            || self.o_groups == 0
            || self.o_lora_rank == 0
            || self.window_size == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
        {
            return Err(model_error(format!(
                "invalid shape: hidden={}, heads={}, head_dim={}, groups={}, o_rank={}, window={}, index_heads={}, index_dim={}, index_topk={}",
                self.hidden_size,
                self.num_heads,
                self.head_dim,
                self.o_groups,
                self.o_lora_rank,
                self.window_size,
                self.index_n_heads,
                self.index_head_dim,
                self.index_topk
            )));
        }
        if !self.num_heads.is_multiple_of(self.o_groups) {
            return Err(model_error(format!(
                "attention heads {} must be divisible by output groups {}",
                self.num_heads, self.o_groups
            )));
        }
        if self.rope_head_dim > self.head_dim || !self.rope_head_dim.is_multiple_of(2) {
            return Err(model_error(format!(
                "rope head dimension {} must be even and no greater than head dimension {}",
                self.rope_head_dim, self.head_dim
            )));
        }
        if self.norm_eps <= 0.0 || self.rope_theta <= 0.0 || self.compress_rope_theta <= 0.0 {
            return Err(model_error(
                "attention epsilon and theta values must be positive",
            ));
        }
        Ok(())
    }

    pub const fn q_full_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub const fn output_group_input_dim(&self) -> usize {
        self.q_full_dim() / self.o_groups
    }

    pub const fn output_latent_dim(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }

    pub fn sparse_spec_with_topk(&self, topk: usize) -> SparseAttentionSpec {
        SparseAttentionSpec {
            heads: self.num_heads,
            head_dim: self.head_dim,
            topk,
            softmax_scale: (self.head_dim as f32).powf(-0.5),
            has_attention_sink: true,
        }
    }

    pub const fn rope_params(&self) -> MlaRopeConfig {
        if self.compress_ratio == 0 {
            MlaRopeConfig {
                theta: self.rope_theta,
                original_seq_len: 0,
                factor: 1.0,
                beta_fast: 32,
                beta_slow: 1,
            }
        } else {
            MlaRopeConfig {
                theta: self.compress_rope_theta,
                original_seq_len: self.original_seq_len,
                factor: self.rope_factor,
                beta_fast: self.beta_fast,
                beta_slow: self.beta_slow,
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaKvLayout {
    layer_count: usize,
    window_size: usize,
    head_dim: usize,
    compress_ratios: Box<[usize]>,
    planes: Box<[KvPlaneDescriptor]>,
    page_size: usize,
    max_sequence_len: usize,
    compressed_plane: Option<usize>,
    index_plane: Option<usize>,
}

impl MlaKvLayout {
    pub fn new(configs: impl IntoIterator<Item = MlaConfig>, page_size: usize) -> Result<Self> {
        let configs = configs.into_iter().collect::<Vec<_>>();
        let first = configs
            .first()
            .copied()
            .ok_or_else(|| model_error("KV layout requires at least one execution layer"))?;
        if page_size == 0 {
            return Err(model_error("KV page size must be non-zero"));
        }
        for config in &configs {
            config.validate()?;
            if config.window_size != first.window_size
                || config.head_dim != first.head_dim
                || config.index_head_dim != first.index_head_dim
            {
                return Err(model_error("KV layout layers use incompatible dimensions"));
            }
        }
        let layer_count = configs.len();
        let has_compressed = configs.iter().any(|config| config.compress_ratio != 0);
        let has_indexer = configs.iter().any(|config| config.compress_ratio == 4);
        let mut planes = vec![KvPlaneDescriptor::new(
            "mla.window_latent_kv",
            first.head_dim,
            layer_count,
            KvElementType::F32,
        )];
        let compressed_plane = has_compressed.then(|| {
            planes.push(KvPlaneDescriptor::new(
                "mla.compressed_main_kv",
                first.head_dim,
                layer_count,
                KvElementType::F32,
            ));
            planes.len() - 1
        });
        let index_plane = has_indexer.then(|| {
            planes.push(KvPlaneDescriptor::new(
                "mla.indexer_kv",
                first.index_head_dim,
                layer_count,
                KvElementType::F32,
            ));
            planes.len() - 1
        });
        for plane in &planes {
            plane
                .checked_page_bytes(page_size)
                .ok_or_else(|| model_error("KV plane page size overflow"))?;
        }
        let max_sequence_len = configs
            .iter()
            .filter(|config| config.compress_ratio > 0 && config.compress_ratio != 4)
            .map(|config| config.compress_ratio.saturating_mul(config.index_topk))
            .min()
            .unwrap_or(u32::MAX as usize);
        Ok(Self {
            layer_count,
            window_size: first.window_size,
            head_dim: first.head_dim,
            compress_ratios: configs.iter().map(|config| config.compress_ratio).collect(),
            planes: planes.into_boxed_slice(),
            page_size,
            max_sequence_len,
            compressed_plane,
            index_plane,
        })
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn compress_ratios(&self) -> &[usize] {
        &self.compress_ratios
    }
}

impl KvLayoutSchema for MlaKvLayout {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        &self.planes
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn max_sequence_len(&self) -> usize {
        self.max_sequence_len
    }
}

impl crate::decoder::MlaPlaneStrategy for MlaKvLayout {
    fn latent_plane(&self) -> usize {
        0
    }

    fn index_plane(&self) -> Option<usize> {
        self.index_plane
    }

    fn compressed_plane(&self) -> Option<usize> {
        self.compressed_plane
    }
}

#[derive(Debug, Clone)]
pub struct MlaLayerState {
    config: MlaConfig,
    pub kv: MlaKvState,
}

impl MlaLayerState {
    pub fn new(config: MlaConfig) -> Self {
        Self {
            config,
            kv: MlaKvState::new(config),
        }
    }

    pub const fn config(&self) -> MlaConfig {
        self.config
    }

    pub fn reset_sequence(&mut self) {
        self.kv.reset_sequence();
    }

    pub fn release_sequence_capacity(&mut self) {
        self.kv = MlaKvState::new(self.config);
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
pub(crate) struct MlaPagedKvBinding {
    logical_pages: Vec<KvPageId>,
    pub(crate) physical_block_slots: Vec<i32>,
    pub(crate) sequence_len: usize,
    pub(crate) page_tokens: usize,
    pub(crate) layer_count: usize,
}

#[cfg(feature = "cuda")]
impl MlaPagedKvBinding {
    pub(crate) fn retain_sequence_len(&mut self, sequence_len: usize) -> Result<()> {
        if self.page_tokens == 0 {
            return Err(model_error("paged binding has zero page size"));
        }
        let retained_blocks = sequence_len.div_ceil(self.page_tokens);
        if retained_blocks > self.physical_block_slots.len() {
            return Err(model_error(format!(
                "paged binding prefix needs {retained_blocks} blocks but only {} are available",
                self.physical_block_slots.len()
            )));
        }
        self.logical_pages.truncate(retained_blocks);
        self.physical_block_slots.truncate(retained_blocks);
        self.sequence_len = sequence_len;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct MlaSequenceState {
    layers: Vec<MlaLayerState>,
    proposal_stages: Vec<MlaLayerState>,
    pub(crate) paged_kv_binding: Option<MlaPagedKvBinding>,
}

impl MlaSequenceState {
    pub fn new(
        layers: impl IntoIterator<Item = MlaConfig>,
        proposal_stages: impl IntoIterator<Item = MlaConfig>,
    ) -> Self {
        Self {
            layers: layers.into_iter().map(MlaLayerState::new).collect(),
            proposal_stages: proposal_stages
                .into_iter()
                .map(MlaLayerState::new)
                .collect(),
            paged_kv_binding: None,
        }
    }

    pub fn layers(&self) -> &[MlaLayerState] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [MlaLayerState] {
        &mut self.layers
    }

    pub fn proposal_stages(&self) -> &[MlaLayerState] {
        &self.proposal_stages
    }

    pub fn proposal_stages_mut(&mut self) -> &mut [MlaLayerState] {
        &mut self.proposal_stages
    }

    pub fn proposal_stage_count(&self) -> usize {
        self.proposal_stages.len()
    }

    pub(crate) fn paged_kv_binding(&self) -> Option<&MlaPagedKvBinding> {
        self.paged_kv_binding.as_ref()
    }

    pub fn reset_for_reuse(&mut self) {
        for state in &mut self.layers {
            state.reset_sequence();
        }
        for state in &mut self.proposal_stages {
            state.reset_sequence();
        }
        self.paged_kv_binding = None;
    }

    fn fresh_from_template(&self) -> Self {
        Self::new(
            self.layers.iter().map(MlaLayerState::config),
            self.proposal_stages.iter().map(MlaLayerState::config),
        )
    }

    fn clone_with_operators(
        &self,
        operators: Option<&ferrule_backend::cuda::operators::linear::CudaOperators>,
    ) -> Result<Self> {
        let clone_layers = |layers: &[MlaLayerState]| {
            layers
                .iter()
                .map(|layer| {
                    let mut clone = MlaLayerState {
                        config: layer.config,
                        kv: layer.kv.fork_paged_prefix_metadata(),
                    };
                    clone_layer_cuda_state(layer, &mut clone, operators)?;
                    Ok(clone)
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            layers: clone_layers(&self.layers)?,
            proposal_stages: clone_layers(&self.proposal_stages)?,
            paged_kv_binding: self.paged_kv_binding.clone(),
        })
    }
}

pub trait MlaSequenceAttachment: Clone + crate::decoder::DecoderSequenceAttachment {
    fn reset_mla_sequence(&mut self);
}

pub struct MlaSequenceLifecycle<A> {
    template: crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
    operators: Option<Rc<ferrule_backend::cuda::operators::linear::CudaOperators>>,
}

impl<A> std::fmt::Debug for MlaSequenceLifecycle<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlaSequenceLifecycle")
            .field("template_layers", &self.template.kv_state().layers.len())
            .field(
                "template_proposal_stages",
                &self.template.kv_state().proposal_stages.len(),
            )
            .finish_non_exhaustive()
    }
}

impl<A> MlaSequenceLifecycle<A>
where
    A: MlaSequenceAttachment,
{
    pub fn new(
        attachment: A,
        layers: impl IntoIterator<Item = MlaConfig>,
        proposal_stages: impl IntoIterator<Item = MlaConfig>,
        operators: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    ) -> Self {
        Self {
            template: crate::decoder::DecoderSequenceState::new(
                attachment,
                MlaSequenceState::new(layers, proposal_stages),
            ),
            operators: Some(operators),
        }
    }

    pub fn create_state(
        &self,
    ) -> Result<crate::decoder::DecoderSequenceState<A, MlaSequenceState>> {
        let mut attachment = self.template.attachment().clone();
        attachment.reset_mla_sequence();
        Ok(crate::decoder::DecoderSequenceState::new(
            attachment,
            self.template.kv_state().fresh_from_template(),
        ))
    }

    pub fn validate_state_shape(
        &self,
        state: &crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
    ) -> Result<()> {
        if state.kv_state().layers.len() != self.template.kv_state().layers.len()
            || state.kv_state().proposal_stages.len()
                != self.template.kv_state().proposal_stages.len()
        {
            return Err(model_error(
                "sequence MLA layout does not match its lifecycle template",
            ));
        }
        Ok(())
    }
}

impl<A> DecoderSequenceLifecycle<crate::decoder::DecoderSequenceState<A, MlaSequenceState>>
    for MlaSequenceLifecycle<A>
where
    A: MlaSequenceAttachment,
{
    fn create(&mut self) -> Result<crate::decoder::DecoderSequenceState<A, MlaSequenceState>> {
        self.create_state()
    }

    fn checkout(
        &mut self,
        _request: DecoderSequenceCheckout,
        source: &crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
    ) -> Result<crate::decoder::DecoderSequenceState<A, MlaSequenceState>> {
        source.core().begin_step()?;
        Ok(crate::decoder::DecoderSequenceState::from_parts(
            source.core().clone(),
            source.topology_id(),
            source.attachment().clone(),
            source
                .kv_state()
                .clone_with_operators(self.operators.as_deref())?,
        ))
    }

    fn logical_fork(
        &mut self,
        source: &crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
        expected_position: usize,
    ) -> Result<crate::decoder::DecoderSequenceState<A, MlaSequenceState>> {
        if source.core().position() != expected_position {
            return Err(execution_error(format!(
                "exact MLA fork expected committed position {expected_position}, source is at {}",
                source.core().position()
            )));
        }
        Ok(crate::decoder::DecoderSequenceState::from_parts(
            source.core().forked()?,
            SequenceTopologyId::take(),
            source.attachment().clone(),
            source
                .kv_state()
                .clone_with_operators(self.operators.as_deref())?,
        ))
    }

    fn reset(
        &mut self,
        state: &mut crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
    ) -> Result<()> {
        self.validate_state_shape(state)?;
        state.core_mut().reset();
        state.attachment_mut().reset_mla_sequence();
        state.kv_state_mut().reset_for_reuse();
        Ok(())
    }

    fn try_release(
        &mut self,
        state: crate::decoder::DecoderSequenceState<A, MlaSequenceState>,
    ) -> std::result::Result<
        (),
        SequenceStateReleaseError<crate::decoder::DecoderSequenceState<A, MlaSequenceState>>,
    > {
        if let Err(error) = self.validate_state_shape(&state) {
            return Err(SequenceStateReleaseError::new(error, state));
        }
        let (core, topology_id, attachment, mut kv_state) = state.into_parts();
        let release = match attachment.preflight_release() {
            Ok(release) => release,
            Err(error) => {
                let state = crate::decoder::DecoderSequenceState::from_parts(
                    core,
                    topology_id,
                    attachment,
                    kv_state,
                );
                return Err(SequenceStateReleaseError::new(error, state));
            }
        };
        for layer in &mut kv_state.layers {
            layer.release_sequence_capacity();
        }
        for stage in &mut kv_state.proposal_stages {
            stage.release_sequence_capacity();
        }
        kv_state.paged_kv_binding = None;
        attachment.release(release);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn clone_layer_cuda_state(
    source: &MlaLayerState,
    clone: &mut MlaLayerState,
    operators: Option<&ferrule_backend::cuda::operators::linear::CudaOperators>,
) -> Result<()> {
    let source_cuda = source.kv.window.cuda_state();
    let Some(operators) = operators else {
        if source_cuda.has_device_state() {
            return Err(execution_error(
                "MLA CUDA sequence checkout requires its CUDA operator context",
            ));
        }
        return Ok(());
    };
    *clone.kv.window.cuda_state_mut() = source_cuda.fork_paged_prefix(operators)?;
    Ok(())
}
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct MlaRecurrentState {
    pub(crate) main_compressor_recurrent: Option<CudaCompressorRecurrentState>,
    pub(crate) main_compressor_needs_reset: bool,
    pub(crate) indexer_compressor_recurrent: Option<CudaCompressorRecurrentState>,
    pub(crate) indexer_compressor_needs_reset: bool,
}
#[cfg(feature = "cuda")]
impl MlaRecurrentState {
    pub(crate) fn fork_paged_prefix(
        &self,
        operators: &ferrule_backend::cuda::operators::linear::CudaOperators,
    ) -> Result<Self> {
        Ok(Self {
            main_compressor_recurrent: self
                .main_compressor_recurrent
                .as_ref()
                .map(|state| operators.clone_compressor_recurrent_state(state))
                .transpose()?,
            main_compressor_needs_reset: self.main_compressor_needs_reset,
            indexer_compressor_recurrent: self
                .indexer_compressor_recurrent
                .as_ref()
                .map(|state| operators.clone_compressor_recurrent_state(state))
                .transpose()?,
            indexer_compressor_needs_reset: self.indexer_compressor_needs_reset,
        })
    }
    fn has_device_state(&self) -> bool {
        self.main_compressor_recurrent.is_some() || self.indexer_compressor_recurrent.is_some()
    }
    pub(crate) fn reset_for_reuse(&mut self) {
        self.main_compressor_needs_reset = self.main_compressor_recurrent.is_some();
        self.indexer_compressor_needs_reset = self.indexer_compressor_recurrent.is_some();
    }
}
#[cfg(feature = "cuda")]
impl std::fmt::Debug for MlaRecurrentState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlaRecurrentState")
            .field(
                "has_main_compressor_recurrent",
                &self.main_compressor_recurrent.is_some(),
            )
            .field(
                "has_indexer_compressor_recurrent",
                &self.indexer_compressor_recurrent.is_some(),
            )
            .finish()
    }
}
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MlaPrefixCheckpoint {
    pub(crate) window_len: usize,
    pub(crate) compressed_rows: usize,
    pub(crate) indexer_compressed_rows: usize,
    pub(crate) main_compressor_needs_reset: bool,
    pub(crate) indexer_compressor_needs_reset: bool,
}
#[derive(Default)]
struct MlaLayerPrefixCheckpoints {
    main: Option<CudaCompressorRecurrentCheckpointSlab>,
    indexer: Option<CudaCompressorRecurrentCheckpointSlab>,
    metadata: Vec<Option<MlaPrefixCheckpoint>>,
}

#[derive(Default)]
struct MlaSequencePrefixCheckpoints {
    start_position: usize,
    executed_rows: usize,
    layers: Vec<MlaLayerPrefixCheckpoints>,
}

#[derive(Default)]
struct MlaProvisionalPrefixCheckpoints {
    active: bool,
    sequences: Vec<MlaSequencePrefixCheckpoints>,
    row_to_sequence: Vec<usize>,
    row_to_local: Vec<usize>,
}

impl MlaProvisionalPrefixCheckpoints {
    /// Starts one transaction's per-row prefix checkpoints for a provisional
    /// verification batch. `sequence_shapes` carries `(start_position,
    /// executed_rows)` per packed sequence.
    fn begin(sequence_shapes: &[(usize, usize)], layer_count: usize) -> Result<Self> {
        if sequence_shapes.is_empty()
            || layer_count == 0
            || sequence_shapes.iter().any(|(_, rows)| *rows == 0)
        {
            return Err(model_error("MLA provisional checkpoint shape is empty"));
        }
        let total_rows = sequence_shapes
            .iter()
            .try_fold(0usize, |total, (_, rows)| {
                total
                    .checked_add(*rows)
                    .ok_or_else(|| model_error("MLA provisional row count overflow"))
            })?;
        let mut checkpoints = Self {
            active: true,
            ..Default::default()
        };
        checkpoints
            .sequences
            .resize_with(sequence_shapes.len(), MlaSequencePrefixCheckpoints::default);
        checkpoints.row_to_sequence.reserve(total_rows);
        checkpoints.row_to_local.reserve(total_rows);
        for (sequence_index, ((start_position, executed_rows), sequence)) in sequence_shapes
            .iter()
            .copied()
            .zip(&mut checkpoints.sequences)
            .enumerate()
        {
            sequence.start_position = start_position;
            sequence.executed_rows = executed_rows;
            sequence
                .layers
                .resize_with(layer_count, MlaLayerPrefixCheckpoints::default);
            for layer in &mut sequence.layers {
                layer.metadata.resize(executed_rows - 1, None);
            }
            checkpoints
                .row_to_sequence
                .extend(std::iter::repeat_n(sequence_index, executed_rows));
            checkpoints.row_to_local.extend(0..executed_rows);
        }
        Ok(checkpoints)
    }
}

#[derive(Debug)]
struct MlaStagedProvisionalRetain {
    states: Vec<Option<MlaSequenceState>>,
    retained_pages: BTreeSet<KvPageId>,
}

fn retained_paged_binding(
    state: &MlaSequenceState,
    sequence_len: usize,
) -> Result<Option<MlaPagedKvBinding>> {
    let Some(mut binding) = state.paged_kv_binding.clone() else {
        if sequence_len == 0 {
            return Ok(None);
        }
        return Err(execution_error(
            "retained MLA prefix has no paged KV binding",
        ));
    };
    binding.retain_sequence_len(sequence_len)?;
    Ok(Some(binding))
}

fn restore_recurrent_prefix(
    operators: Option<&cuda_linear::CudaOperators>,
    checkpoint: Option<&CudaCompressorRecurrentCheckpointSlab>,
    slot: usize,
    destination: &mut Option<CudaCompressorRecurrentState>,
    label: &str,
) -> Result<()> {
    match (checkpoint, destination.as_mut()) {
        (None, None) => Ok(()),
        (Some(checkpoint), Some(destination)) => {
            if slot >= checkpoint.slots() || !checkpoint.supports(destination, checkpoint.slots()) {
                return Err(model_error(format!(
                    "MLA {label} recurrent checkpoint shape is incompatible with its destination"
                )));
            }
            operators
                .ok_or_else(|| {
                    execution_error(format!(
                        "MLA {label} recurrent checkpoint requires CUDA operators"
                    ))
                })?
                .restore_compressor_recurrent_checkpoint(checkpoint, slot, destination)
        }
        _ => Err(model_error(format!(
            "MLA {label} recurrent checkpoint is missing or unexpected"
        ))),
    }
}

fn stage_provisional_retain<A>(
    operators: Option<&cuda_linear::CudaOperators>,
    checkpoints: Option<&MlaProvisionalPrefixCheckpoints>,
    sources: &[crate::decoder::DecoderSequenceState<A, MlaSequenceState>],
    working_states: &[crate::decoder::DecoderSequenceState<A, MlaSequenceState>],
    executed_rows: &[usize],
    retained_rows: &[usize],
) -> Result<MlaStagedProvisionalRetain> {
    let sequence_count = working_states.len();
    if sequence_count == 0
        || sources.len() != sequence_count
        || executed_rows.len() != sequence_count
        || retained_rows.len() != sequence_count
    {
        return Err(execution_error(
            "provisional MLA checkpoint cohort shape is inconsistent",
        ));
    }

    let mut states = Vec::with_capacity(sequence_count);
    let mut retained_pages = BTreeSet::new();
    for sequence_index in 0..sequence_count {
        let source = &sources[sequence_index];
        let working = &working_states[sequence_index];
        let executed = executed_rows[sequence_index];
        let retained = retained_rows[sequence_index];
        if retained > executed {
            return Err(execution_error(format!(
                "retained MLA rows exceed executed rows for sequence {sequence_index}"
            )));
        }
        let sequence_len = source
            .core()
            .position()
            .checked_add(retained)
            .ok_or_else(|| model_error("retained MLA sequence length overflow"))?;
        let binding_source = if retained == 0 {
            source.kv_state()
        } else {
            working.kv_state()
        };
        let binding = retained_paged_binding(binding_source, sequence_len)?;
        if let Some(binding) = &binding {
            retained_pages.extend(binding.logical_pages.iter().copied());
        }

        if retained == executed {
            if binding_source
                .paged_kv_binding
                .as_ref()
                .map(|binding| binding.sequence_len)
                != Some(sequence_len)
            {
                return Err(model_error(format!(
                    "full-width retained MLA binding is not exact for sequence {sequence_index}"
                )));
            }
            states.push(None);
            continue;
        }

        let checkpoint = checkpoints
            .ok_or_else(|| execution_error("MLA KV transaction has no checkpoints"))?
            .sequences
            .get(sequence_index)
            .ok_or_else(|| model_error("MLA provisional sequence checkpoint is missing"))?;
        if checkpoint.start_position != source.core().position()
            || checkpoint.executed_rows != executed
        {
            return Err(model_error(format!(
                "MLA provisional sequence {sequence_index} checkpoint identity is stale"
            )));
        }

        let mut staged = if retained == 0 {
            source.kv_state().clone_with_operators(operators)?
        } else {
            working.kv_state().clone_with_operators(operators)?
        };
        if retained != 0 {
            if staged.layers.len() != source.kv_state().layers.len()
                || checkpoint.layers.len() < staged.layers.len()
            {
                return Err(model_error(format!(
                    "MLA provisional layer checkpoint shape differs for sequence {sequence_index}"
                )));
            }
            let slot = retained - 1;
            for layer_index in 0..staged.layers.len() {
                let source_layer = &source.kv_state().layers[layer_index];
                let staged_layer = &mut staged.layers[layer_index];
                if staged_layer.config.compress_ratio == 0 {
                    staged_layer
                        .kv
                        .restore_uncompressed_prefix_from(&source_layer.kv, retained)?;
                    continue;
                }
                let layer_checkpoint = &checkpoint.layers[layer_index];
                let metadata = layer_checkpoint
                    .metadata
                    .get(slot)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        model_error(format!(
                            "MLA provisional metadata checkpoint is missing for sequence {sequence_index} layer {layer_index} row {retained}"
                        ))
                    })?;
                staged_layer
                    .kv
                    .restore_provisional_prefix_metadata(metadata)?;
                let recurrent = staged_layer.kv.window.cuda_state_mut();
                restore_recurrent_prefix(
                    operators,
                    layer_checkpoint.main.as_ref(),
                    slot,
                    &mut recurrent.main_compressor_recurrent,
                    "main compressor",
                )?;
                restore_recurrent_prefix(
                    operators,
                    layer_checkpoint.indexer.as_ref(),
                    slot,
                    &mut recurrent.indexer_compressor_recurrent,
                    "indexer compressor",
                )?;
                recurrent.main_compressor_needs_reset = metadata.main_compressor_needs_reset;
                recurrent.indexer_compressor_needs_reset = metadata.indexer_compressor_needs_reset;
            }
            if staged.proposal_stages.len() != source.kv_state().proposal_stages.len() {
                return Err(model_error(format!(
                    "MLA provisional proposal-stage shape differs for sequence {sequence_index}"
                )));
            }
            for (staged_stage, source_stage) in staged
                .proposal_stages
                .iter_mut()
                .zip(&source.kv_state().proposal_stages)
            {
                staged_stage
                    .kv
                    .restore_uncompressed_prefix_from(&source_stage.kv, retained)?;
            }
        }
        staged.paged_kv_binding = binding;
        states.push(Some(staged));
    }

    Ok(MlaStagedProvisionalRetain {
        states,
        retained_pages,
    })
}

fn reservation_page(reservation: &KvPoolReservation) -> Result<KvPageId> {
    match (reservation.pages(), reservation.cow_replacement()) {
        ([page], None) => Ok(*page),
        ([], Some(cow)) => Ok(cow.replacement),
        _ => Err(execution_error(
            "MLA physical reservation does not own exactly one page",
        )),
    }
}

fn validate_pending_reservation(
    pool: &ferrule_backend::cuda::operators::kv::CudaKvPagePool,
    reservation: &KvPoolReservation,
) -> Result<()> {
    let page = reservation_page(reservation)?;
    let pending = if reservation.cow_replacement().is_some() {
        pool.pending_replacement_slot(reservation)
    } else {
        pool.pending_slot(reservation, page)
    };
    if pending.is_none() {
        return Err(execution_error(format!(
            "MLA physical reservation for page {} is no longer pending",
            page.0
        )));
    }
    Ok(())
}

struct ActivePagedKvBinding {
    block_slots_device: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    block_offsets_device: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    kv_len_device: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    second_kv_len_device: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    row_sequence_ids_device: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    page_tokens: usize,
    layer_count: usize,
    sequence_count: usize,
}

pub struct MlaReservationBatch {
    transaction: ExecutionTransactionId,
    reservations: Vec<KvPoolReservation>,
    entered_batch: Option<PackedDecoderBatch>,
    entered_bindings: Option<Vec<MlaPagedKvBinding>>,
    provisional: Rc<RefCell<Option<MlaProvisionalPrefixCheckpoints>>>,
}

#[derive(Default)]
struct MlaPlaneStorage {
    page_pool: Option<ferrule_backend::cuda::operators::kv::CudaKvPagePool>,
    shutdown: bool,
}

/// CUDA MLA physical pool. Generic transaction and page custody live in
/// PagedKvBackend; this value owns only typed planes and MLA checkpoints.
pub struct MlaPhysicalPool<A> {
    operators: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    schema: MlaKvLayout,
    planes: Rc<RefCell<MlaPlaneStorage>>,
    attachment: std::marker::PhantomData<fn() -> A>,
}

impl<A> std::fmt::Debug for MlaPhysicalPool<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlaPhysicalPool")
            .field("configured_capacity", &self.pool_capacity())
            .finish_non_exhaustive()
    }
}

impl<A> MlaPhysicalPool<A> {
    pub fn new(
        operators: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
        schema: MlaKvLayout,
    ) -> Self {
        Self {
            operators,
            schema,
            planes: Rc::new(RefCell::new(MlaPlaneStorage::default())),
            attachment: std::marker::PhantomData,
        }
    }

    fn pool_capacity(&self) -> usize {
        self.planes
            .borrow()
            .page_pool
            .as_ref()
            .map_or(0, |pool| pool.stats().allocated_slots)
    }

    fn lower_binding(
        &self,
        transaction: Option<&MlaReservationBatch>,
        block_ids: &[ferrule_common::execution::KvBlockId],
        sequence_len: usize,
    ) -> Result<MlaPagedKvBinding> {
        let storage = self.planes.borrow();
        let pool = storage
            .page_pool
            .as_ref()
            .ok_or_else(|| model_error("MLA CUDA physical MLA pool is not configured"))?;
        let mut physical_block_slots = Vec::with_capacity(block_ids.len());
        for block in block_ids {
            let page = KvPageId(block.get());
            let slot = pool.physical_slot(page).or_else(|| {
                transaction.and_then(|transaction| {
                    transaction
                        .reservations
                        .iter()
                        .find_map(|reservation| pool.pending_slot(reservation, page))
                })
            });
            let slot = slot.ok_or_else(|| {
                model_error(format!(
                    "MLA KV page {} has no committed or provisional physical slot",
                    page.0
                ))
            })?;
            physical_block_slots.push(
                i32::try_from(slot)
                    .map_err(|_| model_error("MLA physical KV slot exceeds i32 ABI"))?,
            );
        }
        Ok(MlaPagedKvBinding {
            logical_pages: block_ids
                .iter()
                .map(|block| KvPageId(block.get()))
                .collect(),
            physical_block_slots,
            sequence_len,
            page_tokens: pool.page_tokens(),
            layer_count: pool.planes().first().map_or(0, |plane| plane.layer_count),
        })
    }

    fn transaction_bindings(
        &self,
        transaction: Option<&MlaReservationBatch>,
        batch: &PackedDecoderBatch,
    ) -> Result<Vec<MlaPagedKvBinding>> {
        batch
            .sequences()
            .iter()
            .map(|sequence| {
                let blocks = sequence
                    .block_table()
                    .iter()
                    .map(|page| ferrule_common::execution::KvBlockId::new(page.0))
                    .collect::<Vec<_>>();
                self.lower_binding(transaction, &blocks, sequence.sequence_len())
            })
            .collect()
    }

    fn view(
        &self,
        transaction: ExecutionTransactionId,
        batch: Option<PackedDecoderBatch>,
        bindings: &[MlaPagedKvBinding],
        row_sequence_ids: &[usize],
        provisional: Rc<RefCell<Option<MlaProvisionalPrefixCheckpoints>>>,
    ) -> Result<MlaKvView> {
        let binding_refs = bindings.iter().collect::<Vec<_>>();
        let active = build_active_paged_binding(&self.operators, &binding_refs, row_sequence_ids)?;
        // Provisional verification captures per-row prefix checkpoints while a
        // view is alive. The view begins them once per transaction, activates
        // them per forward pass, deactivates on drop, and commit/abort discard.
        if let Some(batch) = &batch
            && batch.intent() == ExecutionIntent::ProvisionalVerification
        {
            let mut cell = provisional.borrow_mut();
            if cell.is_none() {
                let shapes = batch
                    .sequences()
                    .iter()
                    .map(|sequence| (sequence.context_len(), sequence.query_len()))
                    .collect::<Vec<_>>();
                *cell = Some(MlaProvisionalPrefixCheckpoints::begin(
                    &shapes,
                    self.schema.layer_count(),
                )?);
            }
            if let Some(checkpoints) = cell.as_mut() {
                checkpoints.active = true;
            }
        }
        Ok(MlaKvView {
            transaction,
            operators: Rc::clone(&self.operators),
            planes: Rc::clone(&self.planes),
            active,
            batch,
            provisional,
            explicit_selection_workspaces: HashMap::new(),
        })
    }
}

impl<A> PhysicalKvPool for MlaPhysicalPool<A>
where
    A: MlaSequenceAttachment,
{
    type SequenceState = crate::decoder::DecoderSequenceState<A, MlaSequenceState>;
    type Transaction = MlaReservationBatch;
    type KvView = MlaKvView;

    fn configured_capacity(&self) -> usize {
        self.pool_capacity()
    }

    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        if max_pages == 0 {
            return Err(execution_error("MLA physical KV capacity must be non-zero"));
        }
        let page_pool = ferrule_backend::cuda::operators::kv::CudaKvPagePool::new(
            &self.operators,
            self.schema.planes(),
            self.schema.page_size(),
            max_pages,
        )?;
        let mut storage = self.planes.borrow_mut();
        if storage
            .page_pool
            .as_ref()
            .is_some_and(|pool| pool.stats().pending_pages != 0)
        {
            return Err(execution_error(
                "cannot reconfigure MLA physical KV with pending reservations",
            ));
        }
        storage.page_pool = Some(page_pool);
        storage.shutdown = false;
        Ok(())
    }

    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        let mut storage = self.planes.borrow_mut();
        if storage.shutdown {
            return Err(execution_error("MLA physical KV pool is shut down"));
        }
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let mut reservations =
            Vec::with_capacity(request.new_pages.len() + request.cow_replacements.len());
        for &page in request.new_pages {
            match pool.reserve(&self.operators, std::slice::from_ref(&page), None) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for reservation in reservations.drain(..).rev() {
                        if let Err(rollback_error) = pool.rollback(&self.operators, reservation) {
                            return Err(Error::context(
                                format!("MLA reservation cleanup after prepare failure: {error}"),
                                rollback_error,
                            ));
                        }
                    }
                    return Err(error);
                }
            }
        }
        for &cow in request.cow_replacements {
            match pool.reserve(&self.operators, &[], Some(cow)) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for reservation in reservations.drain(..).rev() {
                        if let Err(rollback_error) = pool.rollback(&self.operators, reservation) {
                            return Err(Error::context(
                                format!("MLA reservation cleanup after prepare failure: {error}"),
                                rollback_error,
                            ));
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(MlaReservationBatch {
            transaction: request.transaction,
            reservations,
            entered_batch: None,
            entered_bindings: None,
            provisional: Rc::new(RefCell::new(None)),
        })
    }

    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        if transaction.entered_bindings.is_some() || transaction.entered_batch.is_some() {
            return Err(execution_error(
                "MLA physical KV transaction is already entered",
            ));
        }
        if states.len() != batch.sequences().len() {
            return Err(execution_error(format!(
                "MLA physical KV received {} states for {} packed sequences",
                states.len(),
                batch.sequences().len()
            )));
        }
        let bindings = self.transaction_bindings(Some(transaction), batch)?;
        for (state, binding) in states.iter().zip(&bindings) {
            if state.kv_state().layers.len() != self.schema.layer_count()
                || binding.layer_count != self.schema.layer_count()
                || binding.page_tokens != self.schema.page_size()
            {
                return Err(model_error(
                    "packed sequence state does not match the MLA physical layout",
                ));
            }
        }
        for (state, binding) in states.iter_mut().zip(&bindings) {
            state.kv_state_mut().paged_kv_binding = Some(binding.clone());
        }
        transaction.entered_batch = Some(batch.clone());
        transaction.entered_bindings = Some(bindings);
        Ok(())
    }

    fn active_view(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView> {
        if transaction.entered_batch.as_ref() != Some(batch) {
            return Err(execution_error(
                "MLA physical KV transaction is not entered for this packed batch",
            ));
        }
        let bindings = transaction
            .entered_bindings
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV transaction is not entered"))?;
        self.view(
            transaction.transaction,
            Some(batch.clone()),
            bindings,
            batch.row_to_sequence(),
            Rc::clone(&transaction.provisional),
        )
    }

    fn proposal_view(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut Self::SequenceState,
    ) -> Result<Self::KvView> {
        let binding = state
            .kv_state()
            .paged_kv_binding
            .as_ref()
            .cloned()
            .ok_or_else(|| execution_error("MLA proposal state has no paged KV binding"))?;
        self.view(
            transaction,
            None,
            &[binding],
            &[0; PROPOSAL_ROWS],
            Rc::new(RefCell::new(None)),
        )
    }

    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        if transaction.entered_bindings.is_none() || transaction.entered_batch.is_none() {
            return Err(execution_error(
                "MLA physical KV transaction is not entered",
            ));
        }
        transaction.entered_bindings = None;
        transaction.entered_batch = None;
        Ok(())
    }

    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let pending = transaction
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV commit transaction is absent"))?;
        if pending.entered_bindings.is_some() || pending.entered_batch.is_some() {
            return Err(execution_error(
                "cannot commit an entered MLA physical KV transaction",
            ));
        }
        if self.planes.borrow().shutdown {
            return Err(execution_error("MLA physical KV pool is shut down"));
        }
        let pending = transaction
            .take()
            .expect("MLA physical KV commit transaction was validated above");
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        *pending.provisional.borrow_mut() = None;
        match pool.commit_many(pending.reservations) {
            Ok(()) => Ok(KvEndProgress::Complete),
            Err(_) => Ok(KvEndProgress::ConsumedRejected),
        }
    }

    fn abort(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let pending = transaction
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV abort transaction is absent"))?;
        if pending.entered_bindings.is_some() || pending.entered_batch.is_some() {
            return Err(execution_error(
                "cannot abort an entered MLA physical KV transaction",
            ));
        }
        if self.planes.borrow().shutdown {
            return Err(execution_error("MLA physical KV pool is shut down"));
        }
        let pending = transaction
            .take()
            .expect("MLA physical KV abort transaction was validated above");
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        *pending.provisional.borrow_mut() = None;
        for reservation in pending.reservations.into_iter().rev() {
            if pool.rollback(&self.operators, reservation).is_err() {
                return Ok(KvEndProgress::ConsumedRejected);
            }
        }
        Ok(KvEndProgress::Complete)
    }

    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        if pages
            .iter()
            .any(|page| pool.physical_slot(*page).is_none() && !pool.has_snapshot(*page))
        {
            return Err(execution_error(
                "cannot release an unknown MLA physical KV page",
            ));
        }
        for &page in pages {
            pool.release(&self.operators, page)?;
        }
        Ok(())
    }

    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        pool.preempt(&self.operators, pages)?;
        Ok(())
    }

    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        pool.restore(&self.operators, pages)
    }

    fn retain_provisional(
        &mut self,
        transaction: &mut Self::Transaction,
        sources: &[Self::SequenceState],
        working_states: &mut [Self::SequenceState],
        batch: &PackedDecoderBatch,
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<BTreeSet<KvPageId>> {
        if transaction.entered_bindings.is_some() || transaction.entered_batch.is_some() {
            return Err(execution_error(
                "cannot retain an entered MLA physical KV transaction",
            ));
        }
        if self.planes.borrow().shutdown {
            return Err(execution_error("MLA physical KV pool is shut down"));
        }
        if batch.sequences().len() != working_states.len() {
            return Err(execution_error(
                "provisional MLA batch and working-state cohorts differ",
            ));
        }

        let staged = {
            let provisional = transaction.provisional.borrow();
            if provisional
                .as_ref()
                .is_some_and(|checkpoints| checkpoints.active)
            {
                return Err(execution_error(
                    "cannot retain MLA checkpoints while capture is active",
                ));
            }
            stage_provisional_retain(
                Some(&self.operators),
                provisional.as_ref(),
                sources,
                working_states,
                executed_rows,
                retained_rows,
            )?
        };

        let mut keep_reservations = Vec::with_capacity(transaction.reservations.len());
        let mut rejected_reservations = Vec::new();
        for reservation in &transaction.reservations {
            let keep = staged
                .retained_pages
                .contains(&reservation_page(reservation)?);
            keep_reservations.push(keep);
            if !keep {
                rejected_reservations.push(reservation.clone());
            }
        }

        {
            let mut storage = self.planes.borrow_mut();
            let pool = storage
                .page_pool
                .as_mut()
                .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
            for reservation in &rejected_reservations {
                validate_pending_reservation(pool, reservation)?;
            }
            for reservation in rejected_reservations.into_iter().rev() {
                pool.rollback(&self.operators, reservation)?;
            }
        }

        transaction.reservations = std::mem::take(&mut transaction.reservations)
            .into_iter()
            .zip(keep_reservations)
            .filter_map(|(reservation, keep)| keep.then_some(reservation))
            .collect();
        for (working, staged) in working_states.iter_mut().zip(staged.states) {
            if let Some(staged) = staged {
                *working.kv_state_mut() = staged;
            }
        }
        *transaction.provisional.borrow_mut() = None;
        Ok(staged.retained_pages)
    }

    fn shutdown(&mut self) -> Result<()> {
        let mut storage = self.planes.borrow_mut();
        if storage
            .page_pool
            .as_ref()
            .is_some_and(|pool| pool.stats().pending_pages != 0)
        {
            return Err(execution_error(
                "cannot shut down MLA physical KV with pending reservations",
            ));
        }
        storage.page_pool = None;
        storage.shutdown = true;
        Ok(())
    }
}

fn build_active_paged_binding(
    operators: &cuda_linear::CudaOperators,
    bindings: &[&MlaPagedKvBinding],
    row_sequence_ids: &[usize],
) -> Result<ActivePagedKvBinding> {
    let first = bindings
        .first()
        .ok_or_else(|| model_error("active paged binding has no sequences"))?;
    if row_sequence_ids.is_empty() || first.page_tokens == 0 || first.layer_count == 0 {
        return Err(model_error("active paged binding shape is empty"));
    }
    let page_tokens = first.page_tokens;
    let layer_count = first.layer_count;
    let mut physical_block_slots = Vec::new();
    let mut block_offsets = Vec::with_capacity(bindings.len() + 1);
    let mut kv_lens = Vec::with_capacity(bindings.len());
    block_offsets.push(0i32);
    for binding in bindings {
        if binding.page_tokens != page_tokens || binding.layer_count != layer_count {
            return Err(model_error(
                "active paged bindings use incompatible physical layouts",
            ));
        }
        if binding.sequence_len.div_ceil(page_tokens) > binding.physical_block_slots.len() {
            return Err(model_error(
                "active paged binding has too few physical blocks",
            ));
        }
        physical_block_slots.extend_from_slice(&binding.physical_block_slots);
        block_offsets.push(
            i32::try_from(physical_block_slots.len())
                .map_err(|_| model_error("active paged block offsets exceed i32 ABI"))?,
        );
        kv_lens.push(
            i32::try_from(binding.sequence_len)
                .map_err(|_| model_error("active paged KV length exceeds i32 ABI"))?,
        );
    }
    if physical_block_slots.is_empty() {
        return Err(model_error("active paged binding has no physical blocks"));
    }
    let row_sequence_ids = row_sequence_ids
        .iter()
        .map(|&sequence| {
            if sequence >= bindings.len() {
                return Err(model_error(
                    "active paged row selector is outside the sequence cohort",
                ));
            }
            i32::try_from(sequence)
                .map_err(|_| model_error("active paged row selector exceeds i32 ABI"))
        })
        .collect::<Result<Vec<_>>>()?;
    let second_kv_lens = vec![0i32; bindings.len()];
    Ok(ActivePagedKvBinding {
        block_slots_device: operators.i32_host_mirror(&physical_block_slots)?,
        block_offsets_device: operators.i32_host_mirror(&block_offsets)?,
        kv_len_device: operators.i32_host_mirror(&kv_lens)?,
        second_kv_len_device: operators.i32_host_mirror(&second_kv_lens)?,
        row_sequence_ids_device: operators.i32_host_mirror(&row_sequence_ids)?,
        page_tokens,
        layer_count,
        sequence_count: bindings.len(),
    })
}

fn capture_recurrent_checkpoint(
    operators: &cuda_linear::CudaOperators,
    source: Option<&CudaCompressorRecurrentState>,
    checkpoints: &mut Option<CudaCompressorRecurrentCheckpointSlab>,
    slots: usize,
    slot: usize,
) -> Result<()> {
    let Some(source) = source else {
        *checkpoints = None;
        return Ok(());
    };
    if !checkpoints
        .as_ref()
        .is_some_and(|slab| slab.supports(source, slots))
    {
        *checkpoints = Some(operators.create_compressor_recurrent_checkpoint_slab(source, slots)?);
    }
    operators.capture_compressor_recurrent_checkpoint(
        source,
        checkpoints.as_mut().expect("checkpoint slab created above"),
        slot,
    )
}

/// Short-lived typed view with view-owned CUDA page descriptors and workspaces.
pub struct MlaKvView {
    transaction: ExecutionTransactionId,
    operators: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    planes: Rc<RefCell<MlaPlaneStorage>>,
    active: ActivePagedKvBinding,
    batch: Option<PackedDecoderBatch>,
    provisional: Rc<RefCell<Option<MlaProvisionalPrefixCheckpoints>>>,
    explicit_selection_workspaces: HashMap<
        (usize, usize),
        ferrule_backend::cuda::operators::attention::CudaHybridMlaExplicitSelectionWorkspace,
    >,
}

impl std::fmt::Debug for MlaKvView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlaKvView")
            .field("transaction", &self.transaction)
            .finish_non_exhaustive()
    }
}

impl Drop for MlaKvView {
    fn drop(&mut self) {
        if let Some(checkpoints) = self.provisional.borrow_mut().as_mut() {
            checkpoints.active = false;
        }
    }
}

impl MlaKvView {
    #[allow(clippy::too_many_arguments)]
    fn paged_scatter_rows_from_device(
        &mut self,
        plane: usize,
        layer: usize,
        values: &cuda_linear::CudaF32Buffer,
        positions: &cuda_attention::CudaI32Buffer,
        mask: Option<&cuda_attention::CudaI32Buffer>,
        elements_per_token: usize,
    ) -> Result<()> {
        let mut storage = self.planes.borrow_mut();
        let pool = storage
            .page_pool
            .as_mut()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let plane_storage = pool
            .plane_storage_mut(plane)
            .ok_or_else(|| model_error(format!("MLA KV plane {plane} is not F32 storage")))?;
        self.operators
            .paged_plane_scatter_selected_rows_from_device(
                values,
                positions,
                self.active.block_slots_device.device(),
                self.active.block_offsets_device.device(),
                self.active.row_sequence_ids_device.device(),
                mask,
                plane_storage,
                ferrule_backend::cuda::operators::kv::PagedPlaneLayout {
                    page_tokens: self.active.page_tokens,
                    elements_per_token,
                    layer_index: layer,
                    layer_count: self.active.layer_count,
                },
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_window_sparse_attention_rows_into(
        &mut self,
        query: &cuda_linear::CudaF32Buffer,
        visible_lens: &cuda_attention::CudaI32Buffer,
        topk: &mut cuda_attention::CudaI32Buffer,
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        attention_sink: &cuda_linear::CudaF32Buffer,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0 || visible_lens.len() != rows {
            return Err(model_error("MLA paged window attention row shape is empty"));
        }
        self.operators
            .fill_recent_rows_into(visible_lens, rows, spec.topk, topk)?;
        let storage = self.planes.borrow();
        let pool = storage
            .page_pool
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let plane = pool
            .plane_storage(0)
            .ok_or_else(|| model_error("MLA window KV plane is not F32 storage"))?;
        let layout = PagedSparseAttentionLayout {
            batch_size: rows,
            tokens_per_sequence: 1,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            page_tokens: self.active.page_tokens,
            elements_per_token: spec.head_dim,
            layer_index: layer,
            layer_count: self.active.layer_count,
            softmax_scale: spec.softmax_scale,
        };
        let workspace = self
            .explicit_selection_workspaces
            .entry((layer, spec.topk))
            .or_insert(self.operators.hybrid_mla_explicit_selection_workspace(
                layout.explicit_selection_layout(true)?,
            )?);
        self.operators
            .paged_sparse_attention_selected_rows_from_device_into(
                query,
                plane,
                self.active.block_slots_device.device(),
                self.active.block_offsets_device.device(),
                self.active.kv_len_device.device(),
                self.active.row_sequence_ids_device.device(),
                visible_lens,
                topk,
                attention_sink,
                layout,
                workspace,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_topk_indices_paged_indexer_rows_into(
        &mut self,
        query: &cuda_linear::CudaF32Buffer,
        weights: &cuda_linear::CudaF32Buffer,
        positions: &cuda_attention::CudaI32Buffer,
        window_lens: &cuda_attention::CudaI32Buffer,
        compressed_lens: &cuda_attention::CudaI32Buffer,
        layer: usize,
        window_size: usize,
        index_topk: usize,
        compress_ratio: usize,
        direct_compressed: bool,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        logical_indices: &mut cuda_attention::CudaI32Buffer,
        plane_selectors: &mut cuda_attention::CudaI32Buffer,
    ) -> Result<()> {
        let storage = self.planes.borrow();
        let pool = storage
            .page_pool
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let plane_index = if direct_compressed { 1 } else { 2 };
        let indexer_plane = pool
            .plane_storage(plane_index)
            .ok_or_else(|| model_error("MLA indexer KV plane is not F32 storage"))?;
        self.operators
            .dsv4_decode_topk_indices_paged_indexer_rows_from_device_into(
                query,
                weights,
                indexer_plane,
                self.active.block_slots_device.device(),
                self.active.block_offsets_device.device(),
                self.active.row_sequence_ids_device.device(),
                positions,
                window_lens,
                compressed_lens,
                positions.len(),
                window_size,
                index_topk,
                compress_ratio,
                direct_compressed,
                index_heads,
                index_head_dim,
                self.active.page_tokens,
                layer,
                self.active.layer_count,
                weight_scale,
                logical_indices,
                plane_selectors,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn dual_plane_paged_sparse_attention_rows_into(
        &mut self,
        query: &cuda_linear::CudaF32Buffer,
        topk: &cuda_attention::CudaI32Buffer,
        selectors: &cuda_attention::CudaI32Buffer,
        visible_lens: &cuda_attention::CudaI32Buffer,
        main_compressed_lens: &cuda_attention::CudaI32Buffer,
        sequence_main_compressed_lens: &[i32],
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        attention_sink: &cuda_linear::CudaF32Buffer,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        if sequence_main_compressed_lens.len() != self.active.sequence_count {
            return Err(model_error(
                "MLA dual-plane sequence compressed lengths do not match the active cohort",
            ));
        }
        self.operators.update_i32_host_mirror(
            sequence_main_compressed_lens,
            &mut self.active.second_kv_len_device,
        )?;
        let storage = self.planes.borrow();
        let pool = storage
            .page_pool
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let first = pool
            .plane_storage(0)
            .ok_or_else(|| model_error("MLA window KV plane is not F32 storage"))?;
        let second = pool
            .plane_storage(1)
            .ok_or_else(|| model_error("MLA compressed KV plane is not F32 storage"))?;
        let layout = DualPlanePagedSparseAttentionLayout {
            base: PagedSparseAttentionLayout {
                batch_size: rows,
                tokens_per_sequence: 1,
                heads: spec.heads,
                head_dim: spec.head_dim,
                topk: spec.topk,
                page_tokens: self.active.page_tokens,
                elements_per_token: spec.head_dim,
                layer_index: layer,
                layer_count: self.active.layer_count,
                softmax_scale: spec.softmax_scale,
            },
            second_elements_per_token: spec.head_dim,
        };
        let workspace = self
            .explicit_selection_workspaces
            .entry((layer, spec.topk))
            .or_insert(self.operators.hybrid_mla_explicit_selection_workspace(
                layout.explicit_selection_layout(true)?,
            )?);
        self.operators
            .dual_plane_paged_sparse_attention_selected_rows_from_device_into(
                query,
                first,
                second,
                self.active.block_slots_device.device(),
                self.active.block_offsets_device.device(),
                self.active.kv_len_device.device(),
                self.active.second_kv_len_device.device(),
                self.active.row_sequence_ids_device.device(),
                visible_lens,
                main_compressed_lens,
                topk,
                selectors,
                attention_sink,
                layout,
                workspace,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn proposal_hybrid_attention_device_into(
        &mut self,
        layer: usize,
        config: MlaConfig,
        sequence_tokens: usize,
        query: &cuda_linear::CudaF32Buffer,
        block_kv: &cuda_linear::CudaF32Buffer,
        attention_sink: &cuda_linear::CudaF32Buffer,
        output: &mut cuda_linear::CudaF32Buffer,
        workspace: &mut cuda_attention::CudaHybridMlaAttentionWorkspace,
    ) -> Result<()> {
        if self.active.sequence_count != 1 {
            return Err(model_error(
                "MLA proposal attention requires exactly one active sequence",
            ));
        }
        let block_slot_count = sequence_tokens.div_ceil(self.active.page_tokens);
        let storage = self.planes.borrow();
        let pool = storage
            .page_pool
            .as_ref()
            .ok_or_else(|| execution_error("MLA physical KV capacity is not configured"))?;
        let context_plane = pool
            .plane_storage(0)
            .ok_or_else(|| model_error("MLA proposal KV plane is not F32 storage"))?;
        self.operators.hybrid_mla_attention_into(
            query,
            context_plane,
            block_kv,
            self.active.block_slots_device.device(),
            attention_sink,
            HybridMlaAttentionLayout {
                sequence_tokens,
                page_tokens: self.active.page_tokens,
                elements_per_token: config.head_dim,
                layer_index: layer,
                layer_count: self.active.layer_count,
                block_slot_offset: 0,
                block_slot_count,
                softmax_scale: (config.head_dim as f32).powf(-0.5),
            },
            output,
            workspace,
        )
    }

    pub fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        if self.batch.as_ref() != Some(batch) {
            return Err(execution_error(
                "MLA KV view is bound to a different packed batch",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_provisional_prefix_checkpoint(
        &mut self,
        layer: usize,
        row: usize,
        state: &MlaRecurrentState,
        window_len: usize,
        compressed_rows: usize,
        indexer_compressed_rows: usize,
    ) -> Result<()> {
        let mut provisional = self.provisional.borrow_mut();
        let Some(checkpoints) = provisional.as_mut() else {
            return Ok(());
        };
        if !checkpoints.active {
            return Ok(());
        }
        let sequence_index = checkpoints
            .row_to_sequence
            .get(row)
            .copied()
            .ok_or_else(|| model_error("MLA provisional row is outside the cohort"))?;
        let local_row = checkpoints.row_to_local[row];
        let sequence = checkpoints
            .sequences
            .get_mut(sequence_index)
            .ok_or_else(|| model_error("MLA provisional sequence is missing"))?;
        if local_row + 1 >= sequence.executed_rows {
            return Ok(());
        }
        let layer_checkpoints = sequence
            .layers
            .get_mut(layer)
            .ok_or_else(|| model_error("MLA provisional layer is missing"))?;
        let slots = sequence.executed_rows - 1;
        capture_recurrent_checkpoint(
            &self.operators,
            state.main_compressor_recurrent.as_ref(),
            &mut layer_checkpoints.main,
            slots,
            local_row,
        )?;
        capture_recurrent_checkpoint(
            &self.operators,
            state.indexer_compressor_recurrent.as_ref(),
            &mut layer_checkpoints.indexer,
            slots,
            local_row,
        )?;
        layer_checkpoints.metadata[local_row] = Some(MlaPrefixCheckpoint {
            window_len,
            compressed_rows,
            indexer_compressed_rows,
            main_compressor_needs_reset: state.main_compressor_needs_reset,
            indexer_compressor_needs_reset: state.indexer_compressor_needs_reset,
        });
        Ok(())
    }
}

/// Backend-compiled compressor parameters consumed by generic MLA execution.
pub struct PreparedMlaCompressor {
    pub ape: cuda_linear::CudaF32Buffer,
    pub norm: cuda_linear::CudaF32Buffer,
    pub kv: crate::transformer::PreparedCudaLinear,
    pub gate: crate::transformer::PreparedCudaLinear,
}

/// Backend-compiled indexer parameters consumed by generic MLA execution.
pub struct PreparedMlaIndexer {
    pub compressor: PreparedMlaCompressor,
    pub query: crate::transformer::PreparedCudaLinear,
    pub weights: crate::transformer::PreparedCudaLinear,
}

/// Backend-compiled low-rank output projections.
pub struct PreparedMlaOutput {
    pub a: crate::transformer::PreparedCudaLinear,
    pub b: crate::transformer::PreparedCudaLinear,
}

/// Complete model-neutral device payload for one MLA execution layer.
pub struct PreparedMlaWeights {
    pub query_a: crate::transformer::PreparedCudaLinear,
    pub query_b: crate::transformer::PreparedCudaLinear,
    pub key_value: crate::transformer::PreparedCudaLinear,
    pub query_norm: cuda_linear::CudaF32Buffer,
    pub key_value_norm: cuda_linear::CudaF32Buffer,
    pub attention_sink: cuda_linear::CudaF32Buffer,
    pub main_compressor: Option<PreparedMlaCompressor>,
    pub indexer: Option<PreparedMlaIndexer>,
    pub output: PreparedMlaOutput,
}

impl PreparedMlaWeights {
    fn mla_linear(&self, linear: MlaLinearKind) -> Option<&crate::transformer::PreparedCudaLinear> {
        match linear {
            MlaLinearKind::QueryA => Some(&self.query_a),
            MlaLinearKind::QueryB => Some(&self.query_b),
            MlaLinearKind::KeyValue => Some(&self.key_value),
            MlaLinearKind::MainCompressorKv => self.main_compressor.as_ref().map(|value| &value.kv),
            MlaLinearKind::MainCompressorGate => {
                self.main_compressor.as_ref().map(|value| &value.gate)
            }
            MlaLinearKind::IndexerCompressorKv => {
                self.indexer.as_ref().map(|value| &value.compressor.kv)
            }
            MlaLinearKind::IndexerCompressorGate => {
                self.indexer.as_ref().map(|value| &value.compressor.gate)
            }
            MlaLinearKind::IndexerQuery => self.indexer.as_ref().map(|value| &value.query),
            MlaLinearKind::IndexerWeights => self.indexer.as_ref().map(|value| &value.weights),
        }
    }
}

/// Caller-owned buffers for packed proposal context preparation.
pub struct MlaProposalMainBuffers {
    pub(crate) target_taps: cuda_linear::CudaF32Buffer,
    pub(crate) positions: cuda_attention::CudaI32Buffer,
    pub(crate) activation: cuda_linear::CudaFp8ActivationPack,
    pub(crate) inv_rms: cuda_linear::CudaF32Buffer,
    pub(crate) main_x: cuda_linear::CudaF32Buffer,
    pub(crate) context_kv_raw: cuda_linear::CudaF32Buffer,
    pub(crate) context_kv: cuda_linear::CudaF32Buffer,
    pub(crate) context_linear_workspace: cuda_linear::CudaArtifactLinearWorkspace,
}

/// Caller-owned scratch for one proposal MLA attention launch.
pub struct MlaProposalAttentionBuffers {
    pub(crate) workspace: cuda_attention::CudaHybridMlaAttentionWorkspace,
}

struct MlaRopeTable {
    rope_dim: usize,
    rope: MlaRopeConfig,
    capacity: usize,
    cos: cuda_linear::CudaF32Buffer,
    sin: cuda_linear::CudaF32Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlaNormKind {
    Query,
    KeyValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlaCompressorKind {
    Main,
    Indexer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlaLinearKind {
    QueryA,
    QueryB,
    KeyValue,
    MainCompressorKv,
    MainCompressorGate,
    IndexerCompressorKv,
    IndexerCompressorGate,
    IndexerQuery,
    IndexerWeights,
}

/// Narrow concrete owner for model-neutral MLA device execution.
///
/// Model code assembles and installs typed payloads. Execution owns only the
/// backend operators, those payloads, RoPE tables, and profiling policy.
pub struct MlaExecution {
    pub(crate) ops: Rc<cuda_linear::CudaOperators>,
    prepared_layers: HashMap<usize, Rc<PreparedMlaWeights>>,
    rope_tables: HashMap<String, MlaRopeTable>,
    profile_enabled: bool,
    profile_sync: bool,
}

impl MlaExecution {
    pub fn new(
        ops: Rc<cuda_linear::CudaOperators>,
        profile_enabled: bool,
        profile_sync: bool,
    ) -> Self {
        Self {
            ops,
            prepared_layers: HashMap::new(),
            rope_tables: HashMap::new(),
            profile_enabled,
            profile_sync,
        }
    }

    pub(crate) fn install(&mut self, layer: usize, prepared: Rc<PreparedMlaWeights>) -> Result<()> {
        if self.ops.failpoints().check_resource_install() {
            return Err(Error::Internal {
                message: format!("deterministic failpoint: MLA layer {layer} resource install"),
            });
        }
        if self.prepared_layers.contains_key(&layer) {
            return Err(model_error(format!(
                "execution layer {layer} prepared resources are already installed"
            )));
        }
        self.prepared_layers.insert(layer, prepared);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.prepared_layers.clear();
        self.rope_tables.clear();
    }

    fn prepared_layer(&self, layer: usize) -> Result<&PreparedMlaWeights> {
        self.prepared_layers
            .get(&layer)
            .map(Rc::as_ref)
            .ok_or_else(|| model_error(format!("execution layer {layer} is not prepared")))
    }

    fn profile_start(&self) -> Option<Instant> {
        self.profile_enabled.then(Instant::now)
    }

    fn finish_profile_stage(&self, start: Option<Instant>) -> Result<Option<u64>> {
        let Some(start) = start else {
            return Ok(None);
        };
        if self.profile_sync {
            self.ops.sync_stream()?;
        }
        Ok(Some(
            start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
        ))
    }

    fn fail_compressor_transition_if_armed(&self, indexer: bool) -> Result<()> {
        let armed = if indexer {
            self.ops.failpoints().check_indexer_compressor_transition()
        } else {
            self.ops.failpoints().check_main_compressor_transition()
        };
        if armed {
            return Err(Error::Internal {
                message: format!(
                    "deterministic failpoint: MLA {} compressor transition",
                    if indexer { "indexer" } else { "main" }
                ),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn proposal_hybrid_attention_device_into(
        &self,
        kv: &mut MlaKvView,
        execution_layer: usize,
        config: MlaConfig,
        sequence_tokens: usize,
        query: &cuda_linear::CudaF32Buffer,
        block_kv: &cuda_linear::CudaF32Buffer,
        output: &mut cuda_linear::CudaF32Buffer,
        buffers: &mut MlaProposalAttentionBuffers,
    ) -> Result<()> {
        if config.num_heads != cuda_attention::HYBRID_MLA_ATTENTION_HEADS
            || config.head_dim != cuda_attention::HYBRID_MLA_ATTENTION_HEAD_DIM
            || config.window_size != cuda_attention::HYBRID_MLA_ATTENTION_WINDOW
            || config.compress_ratio != 0
        {
            return Err(model_error(format!(
                "proposal hybrid-attention shape mismatch: heads={} head_dim={} window={} compress_ratio={}",
                config.num_heads, config.head_dim, config.window_size, config.compress_ratio
            )));
        }
        kv.proposal_hybrid_attention_device_into(
            execution_layer,
            config,
            sequence_tokens,
            query,
            block_kv,
            &self.prepared_layer(execution_layer)?.attention_sink,
            output,
            &mut buffers.workspace,
        )
    }

    pub(crate) fn allocate_proposal_attention_buffers(
        &self,
    ) -> Result<MlaProposalAttentionBuffers> {
        Ok(MlaProposalAttentionBuffers {
            workspace: self.ops.hybrid_mla_attention_workspace()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn proposal_context_kv_stage_packed_device_into(
        &mut self,
        kv: &mut MlaKvView,
        stage: usize,
        execution_layer: usize,
        config: MlaConfig,
        rows: usize,
        max_position: usize,
        buffers: &mut MlaProposalMainBuffers,
    ) -> Result<()> {
        const ROPE_NAMES: [&str; 8] = [
            "rope_proposal_stage_0",
            "rope_proposal_stage_1",
            "rope_proposal_stage_2",
            "rope_proposal_stage_3",
            "rope_proposal_stage_4",
            "rope_proposal_stage_5",
            "rope_proposal_stage_6",
            "rope_proposal_stage_7",
        ];
        if rows == 0 || config.compress_ratio != 0 || buffers.positions.len() != rows {
            return Err(Error::Model {
                message: format!(
                    "MLA packed Proposal stage {stage} context-KV shape mismatch: rows={rows} positions={} compress_ratio={}",
                    buffers.positions.len(),
                    config.compress_ratio
                ),
            });
        }
        let rope_name = ROPE_NAMES.get(stage).copied().ok_or_else(|| Error::Model {
            message: format!("MLA Proposal stage {stage} exceeds the prepared RoPE identity table"),
        })?;
        if buffers.main_x.len() != rows.saturating_mul(config.hidden_size)
            || buffers.context_kv_raw.len() != rows.saturating_mul(config.head_dim)
            || buffers.context_kv.len() != rows.saturating_mul(config.head_dim)
        {
            return Err(Error::Model {
                message: format!(
                    "MLA packed Proposal stage {stage} context-KV buffer mismatch: main_x={}/{} raw={}/{} kv={}/{}",
                    buffers.main_x.len(),
                    rows.saturating_mul(config.hidden_size),
                    buffers.context_kv_raw.len(),
                    rows.saturating_mul(config.head_dim),
                    buffers.context_kv.len(),
                    rows.saturating_mul(config.head_dim)
                ),
            });
        }
        {
            let prepared = self.prepared_layer(execution_layer)?;
            self.ops
                .artifact_linear_rows_from_device_into_with_scratch(
                    &prepared.key_value.handle,
                    &buffers.main_x,
                    rows,
                    &mut buffers.context_kv_raw,
                    &mut buffers.context_linear_workspace,
                )?;
        }
        {
            let prepared = self.prepared_layer(execution_layer)?;
            self.ops.rms_norm_rows_from_device_into(
                &buffers.context_kv_raw,
                rows,
                &prepared.key_value_norm,
                config.norm_eps,
                &mut buffers.context_kv,
            )?;
        }
        let required_positions = max_position.checked_add(1).ok_or_else(|| Error::Model {
            message: "MLA Proposal context position overflow".into(),
        })?;
        self.ensure_rope_tables_with_params(
            rope_name,
            config.rope_head_dim,
            config.rope_params(),
            required_positions,
        )?;
        self.rope_tail_rows_indexed_from_device(
            rope_name,
            &mut buffers.context_kv,
            &buffers.positions,
            max_position,
            1,
            u32::try_from(config.head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal context head dimension exceeds u32".into(),
            })?,
            u32::try_from(config.rope_head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal context RoPE dimension exceeds u32".into(),
            })?,
            false,
        )?;
        self.ops.fp8_attention_kv_qat_quantize_buffer_in_place(
            &mut buffers.context_kv,
            config.head_dim,
            config.rope_head_dim,
        )?;
        kv.paged_scatter_rows_from_device(
            0,
            execution_layer,
            &buffers.context_kv,
            &buffers.positions,
            None,
            config.head_dim,
        )
    }
    fn readonly_prepared_linear(
        &self,
        layer: usize,
        linear: MlaLinearKind,
        input_len: usize,
    ) -> Result<&crate::transformer::PreparedCudaLinear> {
        let prepared = self.prepared_linear(layer, linear)?;
        let in_features = prepared.handle.shape().in_features();
        if input_len != in_features {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA readonly {linear:?} input length mismatch: expected {in_features}, got {input_len}"
                ),
            });
        }
        if prepared.activation_quantization.is_some() {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA readonly {linear:?} cannot skip activation quantization"
                ),
            });
        }
        if !matches!(
            prepared.handle.shape(),
            cuda_linear::CudaArtifactLinearShape::F32 { .. }
                | cuda_linear::CudaArtifactLinearShape::Bf16Bytes { .. }
        ) {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA readonly {linear:?} requires F32/BF16 weights, got {:?}",
                    prepared.handle.shape()
                ),
            });
        }
        Ok(prepared)
    }
    fn linear_pair_matvec_readonly_from_device_into(
        &self,
        layer: usize,
        first: MlaLinearKind,
        second: MlaLinearKind,
        input: &cuda_linear::CudaF32Buffer,
        first_output: &mut cuda_linear::CudaF32Buffer,
        second_output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        {
            if !matches!(
                (first, second),
                (
                    MlaLinearKind::MainCompressorKv,
                    MlaLinearKind::MainCompressorGate
                ) | (
                    MlaLinearKind::IndexerCompressorKv,
                    MlaLinearKind::IndexerCompressorGate
                )
            ) {
                return Err(model_error(format!(
                    "compressor bundle does not bind {first:?}+{second:?}"
                )));
            }
            let first = self.readonly_prepared_linear(layer, first, input.len())?;
            let second = self.readonly_prepared_linear(layer, second, input.len())?;
            self.ops.artifact_bf16_compressor_into(
                &first.handle,
                &second.handle,
                input,
                1,
                first_output,
                second_output,
            )
        }
    }
    fn rms_norm_layer_rows_device_into(
        &self,
        layer: usize,
        norm: MlaNormKind,
        input: &cuda_linear::CudaF32Buffer,
        rows: usize,
        eps: f32,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        let prepared = self.prepared_layer(layer)?;
        let weight = match norm {
            MlaNormKind::Query => &prepared.query_norm,
            MlaNormKind::KeyValue => &prepared.key_value_norm,
        };
        if rows == 0 || input.len() != rows * weight.len() || output.len() != input.len() {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA RMS rows length mismatch: rows={rows} input={} output={} weight={}",
                    input.len(),
                    output.len(),
                    weight.len()
                ),
            });
        }
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
    }
    fn rms_norm_compressor_rows_device_into(
        &self,
        layer: usize,
        compressor: MlaCompressorKind,
        input: &cuda_linear::CudaF32Buffer,
        rows: usize,
        eps: f32,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        let weight = &self.prepared_compressor(layer, compressor)?.norm;
        if rows == 0 || input.len() != rows * weight.len() || output.len() != input.len() {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA {compressor:?} compressor RMS rows length mismatch: rows={rows} input={} output={} weight={}",
                    input.len(),
                    output.len(),
                    weight.len()
                ),
            });
        }
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
    }
    pub(crate) fn rms_norm_heads_from_device_into(
        &self,
        input: &cuda_linear::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        self.ops
            .rms_norm_heads_from_device_into(input, heads, head_dim, eps, output)
    }
    #[allow(clippy::too_many_arguments)]
    fn mla_output_rows_from_device_into(
        &self,
        context: &cuda_linear::CudaF32Buffer,
        rows: usize,
        cfg: MlaConfig,
        layer: usize,
        latent: &mut cuda_linear::CudaBf16Buffer,
        workspace: &mut cuda_linear::CudaArtifactLinearWorkspace,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        {
            let prepared = self.prepared_layer(layer)?;
            self.ops.artifact_mla_output_into(
                context,
                rows,
                &prepared.output.a.handle,
                &prepared.output.b.handle,
                cfg.o_groups,
                cfg.output_group_input_dim(),
                cfg.o_lora_rank,
                latent,
                workspace,
                output,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn query_a_kv_from_prepared_fp8_into(
        &self,
        layer: usize,
        activation: &cuda_linear::CudaPreparedFp8Activation<'_>,
        query_a_output: &mut cuda_linear::CudaF32Buffer,
        key_value_output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        {
            let query_a = self.prepared_linear(layer, MlaLinearKind::QueryA)?;
            let key_value = self.prepared_linear(layer, MlaLinearKind::KeyValue)?;
            self.ops.artifact_fp8_query_a_kv_into(
                &query_a.handle,
                &key_value.handle,
                activation,
                query_a_output,
                key_value_output,
            )
        }
    }
    fn compressor_rows_from_device_into(
        &self,
        layer: usize,
        compressor: MlaCompressorKind,
        input: &cuda_linear::CudaF32Buffer,
        rows: usize,
        kv_output: &mut cuda_linear::CudaF32Buffer,
        gate_output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        {
            let prepared = self.prepared_compressor(layer, compressor)?;
            self.ops.artifact_bf16_compressor_into(
                &prepared.kv.handle,
                &prepared.gate.handle,
                input,
                rows,
                kv_output,
                gate_output,
            )
        }
    }
    fn linear_rows_from_device_into(
        &self,
        layer: usize,
        linear: MlaLinearKind,
        input: &cuda_linear::CudaF32Buffer,
        rows: usize,
        output: &mut cuda_linear::CudaF32Buffer,
        workspace: &mut cuda_linear::CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let prepared = self.prepared_linear(layer, linear)?;
        let in_features = prepared.handle.shape().in_features();
        if rows == 0 || input.len() != rows * in_features {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {layer} CUDA {linear:?} rows input length mismatch: rows={rows} expected {}, got {}",
                    rows * in_features,
                    input.len()
                ),
            });
        }
        if linear == MlaLinearKind::QueryB {
            return self
                .ops
                .artifact_fp8_projection_rows_from_device_into_with_scratch(
                    &prepared.handle,
                    input,
                    rows,
                    output,
                    workspace,
                );
        }
        self.ops.artifact_linear_rows_from_device_into_with_scratch(
            &prepared.handle,
            input,
            rows,
            output,
            workspace,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn compressor_recurrent_append_into(
        &self,
        layer: usize,
        compressor: MlaCompressorKind,
        state: &mut Option<CudaCompressorRecurrentState>,
        needs_reset: &mut bool,
        projected_kv: &cuda_linear::CudaF32Buffer,
        projected_score: &cuda_linear::CudaF32Buffer,
        position: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        compressed: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<bool> {
        let ape = &self.prepared_compressor(layer, compressor)?.ape;
        if state.is_none() {
            *state = Some(
                self.ops
                    .create_compressor_recurrent_state(ratio, head_dim, out_dim, overlap)?,
            );
        }
        let state = state.as_mut().expect("created above");
        if *needs_reset {
            self.ops.reset_compressor_recurrent_state(state)?;
            *needs_reset = false;
        }
        let boundary = self.ops.compressor_recurrent_append_projected(
            state,
            projected_kv,
            projected_score,
            ape,
            position,
        )?;
        if boundary {
            self.ops
                .compressor_recurrent_boundary_into(state, compressed)?;
        }
        Ok(boundary)
    }
    fn ensure_rope_tables_with_params(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope: MlaRopeConfig,
        required_positions: usize,
    ) -> Result<()> {
        validate_rope_table_request(name, rope_dim, rope, required_positions)?;
        if let Some(table) = self.rope_tables.get(name) {
            validate_rope_table_identity(name, table, rope_dim, rope)?;
            if table.capacity >= required_positions {
                return Ok(());
            }
        }
        let capacity = rope_table_capacity(required_positions)?;
        let rd2 = rope_dim / 2;
        let elements = capacity.checked_mul(rd2).ok_or_else(|| Error::Model {
            message: format!("MLA RoPE table '{name}' element count overflow: capacity={capacity} rope_dim={rope_dim}"),
        })?;
        let mut cos = Vec::new();
        cos.try_reserve_exact(elements).map_err(|error| Error::Model {
            message: format!("MLA RoPE cosine table '{name}' host allocation failed for {elements} elements: {error}"),
        })?;
        cos.resize(elements, 0.0f32);
        let mut sin = Vec::new();
        sin.try_reserve_exact(elements).map_err(|error| Error::Model {
            message: format!("MLA RoPE sine table '{name}' host allocation failed for {elements} elements: {error}"),
        })?;
        sin.resize(elements, 0.0f32);
        for position in 0..capacity {
            for pair in 0..rd2 {
                let freq = yarn_frequency(pair, rope_dim, rope);
                let angle = position as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[position * rd2 + pair] = c;
                sin[position * rd2 + pair] = s;
            }
        }
        let cos = self.ops.upload_f32_buffer(&cos)?;
        let sin = self.ops.upload_f32_buffer(&sin)?;
        self.rope_tables.insert(
            name.to_string(),
            MlaRopeTable {
                rope_dim,
                rope,
                capacity,
                cos,
                sin,
            },
        );
        Ok(())
    }
    pub(crate) fn require_rope_tables(
        &self,
        name: &str,
        rope_dim: usize,
        required_positions: usize,
    ) -> Result<()> {
        if required_positions == 0 {
            return Err(Error::Model {
                message: format!("MLA RoPE table '{name}' requires at least one position"),
            });
        }
        let table = self.rope_tables.get(name).ok_or_else(|| Error::Model {
            message: format!("MLA RoPE table '{name}' is not prepared"),
        })?;
        if table.rope_dim != rope_dim {
            return Err(Error::Model {
                message: format!(
                    "MLA RoPE table '{name}' dimension mismatch: cached={} requested={rope_dim}",
                    table.rope_dim
                ),
            });
        }
        validate_rope_table_capacity(name, table, required_positions)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_from_device(
        &mut self,
        name: &str,
        qk: &mut cuda_linear::CudaF32Buffer,
        start_position: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        self.rope_tail_rows_strided_from_device(
            name,
            qk,
            start_position,
            1,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_indexed_from_device(
        &mut self,
        name: &str,
        qk: &mut cuda_linear::CudaF32Buffer,
        positions: &cuda_attention::CudaI32Buffer,
        max_position: usize,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if positions.is_empty() || heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        let required_positions = max_position.checked_add(1).ok_or_else(|| Error::Model {
            message: "MLA indexed RoPE position overflow".into(),
        })?;
        self.require_rope_tables(name, rope_dim as usize, required_positions)?;
        let table = self
            .rope_tables
            .get(name)
            .expect("rope tables required immediately above");
        self.ops.rope_tail_rows_indexed_from_device(
            qk,
            &table.cos,
            &table.sin,
            positions,
            positions.len() as u32,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_strided_from_device(
        &mut self,
        name: &str,
        qk: &mut cuda_linear::CudaF32Buffer,
        start_position: u32,
        position_stride: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        let last_offset = (rows as usize - 1)
            .checked_mul(position_stride as usize)
            .ok_or_else(|| Error::Model {
                message: "MLA RoPE row-stride overflow".into(),
            })?;
        let required_positions = (start_position as usize)
            .checked_add(last_offset)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| Error::Model {
                message: "MLA RoPE position overflow".into(),
            })?;
        self.require_rope_tables(name, rope_dim as usize, required_positions)?;
        let table = self
            .rope_tables
            .get(name)
            .expect("rope tables required immediately above");
        self.ops.rope_tail_rows_strided_from_device(
            qk,
            &table.cos,
            &table.sin,
            start_position,
            position_stride,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }
    fn prepared_compressor(
        &self,
        layer: usize,
        compressor: MlaCompressorKind,
    ) -> Result<&PreparedMlaCompressor> {
        let prepared = self.prepared_layer(layer)?;
        let resources = match compressor {
            MlaCompressorKind::Main => prepared.main_compressor.as_ref(),
            MlaCompressorKind::Indexer => prepared.indexer.as_ref().map(|value| &value.compressor),
        };
        resources.ok_or_else(|| Error::Model {
            message: format!(
                "MLA layer {layer} CUDA {compressor:?} compressor resources are unavailable"
            ),
        })
    }
    fn prepared_linear(
        &self,
        layer: usize,
        linear: MlaLinearKind,
    ) -> Result<&crate::transformer::PreparedCudaLinear> {
        let prepared = self.prepared_layer(layer)?;
        let handle = prepared.mla_linear(linear);
        handle.ok_or_else(|| Error::Model {
            message: format!(
                "MLA layer {layer} CUDA {linear:?} ({:?}) linear is unavailable",
                linear
            ),
        })
    }
}

#[cfg(feature = "cuda")]
fn debug_cuda_attention_values(layer: usize, stage: &str, values: &[f32]) -> Result<()> {
    if std::env::var("FERRULE_DEBUG_ATTENTION_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        != Some(layer)
    {
        return Ok(());
    }
    if let Some(directory) = std::env::var_os("FERRULE_DEBUG_ATTENTION_DUMP_DIR") {
        std::fs::create_dir_all(&directory).map_err(|source| Error::Internal {
            message: format!("failed to create attention dump directory: {source}"),
        })?;
        let path = std::path::PathBuf::from(directory).join(format!("layer_{layer}_{stage}.f32"));
        if !path.exists() {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            std::fs::write(&path, bytes).map_err(|source| Error::Internal {
                message: format!(
                    "failed to write attention dump {}: {source}",
                    path.display()
                ),
            })?;
        }
    }
    let sum = values.iter().copied().sum::<f32>();
    let sumsq = values.iter().map(|value| value * value).sum::<f32>();
    let absmax = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "attention-stage layer={} name={} sum={} sumsq={} absmax={} samples={:?}",
        layer,
        stage,
        sum,
        sumsq,
        absmax,
        &values[..values.len().min(4)]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn debug_cuda_attention_stage(
    layer: usize,
    stage: &str,
    buffer: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    operators: &mut MlaExecution,
) -> Result<()> {
    if std::env::var("FERRULE_DEBUG_ATTENTION_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        != Some(layer)
    {
        return Ok(());
    }
    let values = operators.ops.download_f32_buffer(buffer)?;
    debug_cuda_attention_values(layer, stage, &values)
}

#[cfg(feature = "cuda")]
fn debug_cuda_attention_bf16_stage(
    layer: usize,
    stage: &str,
    buffer: &ferrule_backend::cuda::operators::linear::CudaBf16Buffer,
    operators: &mut MlaExecution,
) -> Result<()> {
    if std::env::var("FERRULE_DEBUG_ATTENTION_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        != Some(layer)
    {
        return Ok(());
    }
    let values = operators.ops.download_bf16_buffer(buffer)?;
    debug_cuda_attention_values(layer, stage, &values)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlaProfileStage {
    Qa,
    QNorm,
    Qb,
    QHeadNorm,
    QRope,
    KvProj,
    KvNorm,
    KvRopeQuant,

    SparseAttention,
    ContextRope,

    OutputB,
}

const COMPRESS_RATIOS: [usize; 2] = [4, 128];

#[derive(Debug, Clone, PartialEq)]
pub struct MlaWeights {
    pub layer: usize,
    pub query_a: LinearWeight,
    pub query_b: LinearWeight,
    pub key_value: LinearWeight,
    pub output_a: LinearWeight,
    pub output_b: LinearWeight,
    pub query_norm: Vec<f32>,
    pub key_value_norm: Vec<f32>,
    pub attention_sink: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaCompressor {
    pub compress_ratio: usize,
    pub head_dim: usize,
    pub overlap: bool,
    pub rotate_for_indexer: bool,
    pub ape: Vec<f32>,
    pub ape_rows: usize,
    pub ape_cols: usize,
    pub norm: Vec<f32>,
    pub wkv: LinearWeight,
    pub wgate: LinearWeight,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaIndexer {
    pub compressor: MlaCompressor,
    pub wq_b: LinearWeight,
    pub weights_proj: LinearWeight,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaCompression {
    pub compressor: MlaCompressor,
    pub indexer: Option<MlaIndexer>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMla {
    pub layer: usize,
    config: MlaConfig,
    pub payload: MlaWeights,
    pub compressed: Option<MlaCompression>,
}

#[cfg(feature = "cuda")]
struct MlaCompressorArena {
    kv: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    score: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    compressed: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    normalized: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
impl MlaCompressorArena {
    fn new(
        payload: &MlaCompressor,
        rows: usize,
        independent_sequences: bool,
        operators: &mut MlaExecution,
    ) -> Result<Self> {
        let ops = &operators.ops;
        let boundary_rows = if independent_sequences {
            rows
        } else {
            rows / payload.compress_ratio
        };
        Ok(Self {
            kv: ops.zero_f32_buffer(rows * payload.wkv.format.out_features())?,
            score: ops.zero_f32_buffer(rows * payload.wgate.format.out_features())?,
            compressed: ops.zero_f32_buffer(boundary_rows * payload.head_dim)?,
            normalized: ops.zero_f32_buffer(boundary_rows * payload.head_dim)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct MlaRowsTransitionArena {
    input: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    main_compressor: Option<MlaCompressorArena>,
    indexer_compressor: Option<MlaCompressorArena>,
}

#[cfg(feature = "cuda")]
impl MlaRowsTransitionArena {
    pub(crate) fn new(attention: &PreparedMla, operators: &mut MlaExecution) -> Result<Self> {
        let cfg = attention.config;
        let main_compressor = attention
            .compressed
            .as_ref()
            .map(|payload| MlaCompressorArena::new(&payload.compressor, 1, true, operators))
            .transpose()?;
        let indexer_compressor = attention
            .compressed
            .as_ref()
            .and_then(|payload| payload.indexer.as_ref())
            .map(|indexer| MlaCompressorArena::new(&indexer.compressor, 1, true, operators))
            .transpose()?;
        Ok(Self {
            input: operators.ops.zero_f32_buffer(cfg.hidden_size)?,
            main_compressor,
            indexer_compressor,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct MlaDecodeArena {
    q_latent: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    q_norm: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    q_indexer: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    query_raw: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    query: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    kv_raw: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    kv: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    index_query: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    index_weights: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    positions: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    window_lens: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    compressed_lens: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    main_compressed_lens: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    visible_lens: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    main_positions: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    main_mask: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    indexer_positions: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    indexer_mask: ferrule_backend::cuda::operators::attention::CudaI32HostMirror,
    window_topk: ferrule_backend::cuda::operators::attention::CudaI32Buffer,
    topk: ferrule_backend::cuda::operators::attention::CudaI32Buffer,
    topk_selectors: ferrule_backend::cuda::operators::attention::CudaI32Buffer,
    attention_sink: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    context: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    latent: ferrule_backend::cuda::operators::linear::CudaBf16Buffer,
    pub(crate) output: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(crate) linear_workspace:
        ferrule_backend::cuda::operators::linear::CudaArtifactLinearWorkspace,
    main_compressor: Option<MlaCompressorArena>,
    indexer_compressor: Option<MlaCompressorArena>,
}

#[cfg(feature = "cuda")]
impl MlaDecodeArena {
    pub(crate) fn new(
        attention: &PreparedMla,
        rows: usize,
        independent_sequences: bool,
        operators: &mut MlaExecution,
    ) -> Result<Self> {
        let cfg = attention.config;
        let main_compressor = attention
            .compressed
            .as_ref()
            .map(|payload| {
                MlaCompressorArena::new(&payload.compressor, rows, independent_sequences, operators)
            })
            .transpose()?;
        let indexer_compressor = attention
            .compressed
            .as_ref()
            .and_then(|payload| payload.indexer.as_ref())
            .map(|indexer| {
                MlaCompressorArena::new(&indexer.compressor, rows, independent_sequences, operators)
            })
            .transpose()?;
        let ops = &operators.ops;

        let max_linear_width = cfg
            .hidden_size
            .max(cfg.q_lora_rank)
            .max(cfg.q_full_dim())
            .max(cfg.head_dim)
            .max(cfg.output_latent_dim())
            .max(cfg.index_n_heads * cfg.index_head_dim);
        let zero_control_rows = vec![0i32; rows];
        Ok(Self {
            q_latent: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            q_norm: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            q_indexer: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            query_raw: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            query: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            kv_raw: ops.zero_f32_buffer(rows * cfg.head_dim)?,
            kv: ops.zero_f32_buffer(rows * cfg.head_dim)?,
            index_query: ops.zero_f32_buffer(rows * cfg.index_n_heads * cfg.index_head_dim)?,
            index_weights: ops.zero_f32_buffer(rows * cfg.index_n_heads)?,
            positions: ops.i32_host_mirror(&zero_control_rows)?,
            window_lens: ops.i32_host_mirror(&zero_control_rows)?,
            compressed_lens: ops.i32_host_mirror(&zero_control_rows)?,
            main_compressed_lens: ops.i32_host_mirror(&zero_control_rows)?,
            visible_lens: ops.i32_host_mirror(&zero_control_rows)?,
            main_positions: ops.i32_host_mirror(&zero_control_rows)?,
            main_mask: ops.i32_host_mirror(&zero_control_rows)?,
            indexer_positions: ops.i32_host_mirror(&zero_control_rows)?,
            indexer_mask: ops.i32_host_mirror(&zero_control_rows)?,
            window_topk: ops.zero_i32_buffer(rows * cfg.window_size)?,
            topk: ops.zero_i32_buffer(rows * (cfg.window_size + cfg.index_topk))?,
            topk_selectors: ops.zero_i32_buffer(rows * (cfg.window_size + cfg.index_topk))?,
            attention_sink: ops.upload_f32_buffer(&attention.payload.attention_sink)?,
            context: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            latent: ops.zero_bf16_buffer(rows * cfg.output_latent_dim())?,
            output: ops.zero_f32_buffer(rows * cfg.hidden_size)?,
            linear_workspace: ops.artifact_linear_workspace(rows, max_linear_width)?,
            main_compressor,
            indexer_compressor,
        })
    }
}

#[cfg(feature = "cuda")]
fn decode_metadata_i32(values: &[usize], label: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| Error::Model {
                message: format!("packed decode {label} exceeds i32 ABI"),
            })
        })
        .collect()
}

impl PreparedMla {
    #[cfg(test)]
    pub fn new(layer: usize, config: MlaConfig, payload: MlaWeights) -> Result<Self> {
        Self::new_with_compressed(layer, config, payload, None)
    }

    pub fn new_with_compressed(
        layer: usize,
        config: MlaConfig,
        payload: MlaWeights,
        compressed: Option<MlaCompression>,
    ) -> Result<Self> {
        config.validate()?;
        validate_ratio(config.compress_ratio)?;
        match (config.compress_ratio, compressed.as_ref()) {
            (0, None) => {}
            (4, Some(value)) if value.indexer.is_some() => {}
            (128, Some(value)) if value.indexer.is_none() => {}
            (0, Some(_)) => {
                return Err(model_error(format!(
                    "layer {layer} uncompressed MLA has a compressor"
                )));
            }
            (4, _) => {
                return Err(model_error(format!(
                    "layer {layer} ratio-4 MLA requires its tied indexer"
                )));
            }
            (128, _) => {
                return Err(model_error(format!(
                    "layer {layer} ratio-128 MLA requires only the main compressor"
                )));
            }
            _ => unreachable!("ratio validated above"),
        }
        let attention = Self {
            layer,
            config,
            payload,
            compressed,
        };
        attention.validate_shapes()?;
        Ok(attention)
    }

    pub const fn config(&self) -> MlaConfig {
        self.config
    }

    pub fn validate_shapes(&self) -> Result<()> {
        let config = self.config;
        check_linear(
            self.layer,
            "wq_a",
            &self.payload.query_a,
            config.q_lora_rank,
            config.hidden_size,
        )?;
        check_linear(
            self.layer,
            "wq_b",
            &self.payload.query_b,
            config.q_full_dim(),
            config.q_lora_rank,
        )?;
        check_linear(
            self.layer,
            "wkv",
            &self.payload.key_value,
            config.head_dim,
            config.hidden_size,
        )?;
        check_linear(
            self.layer,
            "wo_a",
            &self.payload.output_a,
            config.output_latent_dim(),
            config.output_group_input_dim(),
        )?;
        check_linear(
            self.layer,
            "wo_b",
            &self.payload.output_b,
            config.hidden_size,
            config.output_latent_dim(),
        )?;
        check_len(
            self.layer,
            "q_norm",
            self.payload.query_norm.len(),
            config.q_lora_rank,
        )?;
        check_len(
            self.layer,
            "kv_norm",
            self.payload.key_value_norm.len(),
            config.head_dim,
        )?;
        check_len(
            self.layer,
            "attention_sink",
            self.payload.attention_sink.len(),
            config.num_heads,
        )?;
        if let Some(compressed) = &self.compressed {
            validate_compressor(
                self.layer,
                &compressed.compressor,
                config.hidden_size,
                config.head_dim,
                false,
            )?;
            if let Some(indexer) = &compressed.indexer {
                validate_compressor(
                    self.layer,
                    &indexer.compressor,
                    config.hidden_size,
                    config.index_head_dim,
                    true,
                )?;
                check_linear(
                    self.layer,
                    "indexer.query",
                    &indexer.wq_b,
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                )?;
                check_linear(
                    self.layer,
                    "indexer.weights",
                    &indexer.weights_proj,
                    config.index_n_heads,
                    config.hidden_size,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn project_decode_rows_from_device_into(
        &self,
        hidden_dev: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        hidden_fp8: &ferrule_backend::cuda::operators::linear::CudaPreparedFp8Activation<'_>,
        max_position: usize,
        operators: &mut MlaExecution,
        arena: &mut MlaDecodeArena,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = arena.positions.len();
        if rows == 0 || hidden_dev.len() != rows * cfg.hidden_size {
            return Err(Error::Model {
                message: format!(
                    "MLA layer {} packed attention input mismatch: rows={rows} expected={} got={}",
                    self.layer,
                    rows * cfg.hidden_size,
                    hidden_dev.len()
                ),
            });
        }
        let required_positions = max_position.checked_add(1).ok_or_else(|| Error::Model {
            message: "MLA packed RoPE position overflow".into(),
        })?;
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");

        operators.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            required_positions,
        )?;

        let stage_start = operators.profile_start();
        operators.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        debug_cuda_attention_stage(self.layer, "q_latent", &arena.q_latent, operators)?;
        debug_cuda_attention_stage(self.layer, "kv_raw", &arena.kv_raw, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::Qa, stage_start)?;

        let stage_start = operators.profile_start();
        operators.rms_norm_layer_rows_device_into(
            self.layer,
            MlaNormKind::Query,
            &arena.q_latent,
            rows,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        debug_cuda_attention_stage(self.layer, "q_norm", &arena.q_norm, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::QNorm, stage_start)?;

        if self
            .compressed
            .as_ref()
            .is_some_and(|value| value.indexer.is_some())
        {
            operators
                .ops
                .copy_f32_into_slot(&arena.q_norm, &mut arena.q_indexer, 0)?;
        }

        let stage_start = operators.profile_start();
        operators.linear_rows_from_device_into(
            self.layer,
            MlaLinearKind::QueryB,
            &arena.q_norm,
            rows,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        debug_cuda_attention_stage(self.layer, "query_raw", &arena.query_raw, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::Qb, stage_start)?;
        let stage_start = operators.profile_start();
        operators.rms_norm_heads_from_device_into(
            &arena.query_raw,
            rows * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        debug_cuda_attention_stage(self.layer, "query_norm", &arena.query, operators)?;
        record_attention_stage(
            operators,
            self.layer,
            MlaProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.query,
            &arena.positions,
            max_position,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        debug_cuda_attention_stage(self.layer, "query", &arena.query, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::QRope, stage_start)?;

        let stage_start = operators.profile_start();
        record_attention_stage(operators, self.layer, MlaProfileStage::KvProj, stage_start)?;
        let stage_start = operators.profile_start();
        operators.rms_norm_layer_rows_device_into(
            self.layer,
            MlaNormKind::KeyValue,
            &arena.kv_raw,
            rows,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        debug_cuda_attention_stage(self.layer, "kv_norm", &arena.kv, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::KvNorm, stage_start)?;
        let stage_start = operators.profile_start();
        operators.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.kv,
            &arena.positions,
            max_position,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        debug_cuda_attention_stage(self.layer, "kv", &arena.kv, operators)?;
        record_attention_stage(
            operators,
            self.layer,
            MlaProfileStage::KvRopeQuant,
            stage_start,
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn project_decode_context_rows_from_device_into(
        &self,
        max_position: usize,
        operators: &mut MlaExecution,
        arena: &mut MlaDecodeArena,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = arena.positions.len();
        let rope_name = format!("rope_attn_L{}", self.layer);
        debug_cuda_attention_stage(self.layer, "context", &arena.context, operators)?;
        let stage_start = operators.profile_start();
        operators.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.context,
            &arena.positions,
            max_position,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        debug_cuda_attention_stage(
            self.layer,
            "context_inverse_rope",
            &arena.context,
            operators,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            MlaProfileStage::ContextRope,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.mla_output_rows_from_device_into(
            &arena.context,
            rows,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )?;
        debug_cuda_attention_bf16_stage(self.layer, "latent", &arena.latent, operators)?;
        debug_cuda_attention_stage(self.layer, "output", &arena.output, operators)?;
        record_attention_stage(operators, self.layer, MlaProfileStage::OutputB, stage_start)?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed_rows_from_device_into(
        &self,
        kv: &mut MlaKvView,
        caches: &mut [&mut MlaKvState],
        hidden_dev: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        hidden_fp8: &ferrule_backend::cuda::operators::linear::CudaPreparedFp8Activation<'_>,
        positions: &[usize],
        row_to_sequence: &[usize],
        sequence_major_rows: &[usize],
        sequence_phases: &[ForwardPhase],
        paged_bindings: &[MlaPagedKvBinding],
        operators: &mut MlaExecution,
        arena: &mut MlaDecodeArena,
        transition: &mut MlaRowsTransitionArena,
    ) -> Result<()> {
        let rows = positions.len();
        if rows == 0
            || row_to_sequence.len() != rows
            || sequence_major_rows.len() != rows
            || caches.len() != paged_bindings.len()
            || sequence_phases.len() != caches.len()
            || row_to_sequence
                .iter()
                .any(|sequence| *sequence >= caches.len())
        {
            return Err(Error::Model {
                message: "MLA packed attention row/sequence metadata is inconsistent".into(),
            });
        }

        let visible_lens = positions
            .iter()
            .map(|position| {
                position
                    .checked_add(1)
                    .and_then(|length| i32::try_from(length).ok())
                    .ok_or_else(|| Error::Model {
                        message: "packed attention visible length exceeds i32 ABI".into(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if self.config.compress_ratio == 0 {
            let cfg = self.config;
            let max_position = positions
                .iter()
                .copied()
                .max()
                .ok_or_else(|| Error::Model {
                    message: "packed decode positions are empty".into(),
                })?;
            let positions_i32 = decode_metadata_i32(positions, "position")?;
            {
                let ops = &operators.ops;
                ops.update_i32_host_mirror(&positions_i32, &mut arena.positions)?;
                ops.update_i32_host_mirror(&visible_lens, &mut arena.visible_lens)?;
            }
            self.project_decode_rows_from_device_into(
                hidden_dev,
                hidden_fp8,
                max_position,
                operators,
                arena,
            )?;
            for &row in sequence_major_rows {
                caches[row_to_sequence[row]].window.record_device_rows(1);
            }
            kv.paged_scatter_rows_from_device(
                0,
                self.layer,
                &arena.kv,
                &arena.positions,
                None,
                cfg.head_dim,
            )?;
            let attention_topk = positions
                .iter()
                .map(|position| position.saturating_add(1).min(cfg.window_size))
                .max()
                .ok_or_else(|| Error::Model {
                    message: "packed attention positions are empty".into(),
                })?;
            let window_topk_len = rows
                .checked_mul(attention_topk)
                .ok_or_else(|| Error::Model {
                    message: "packed window top-k size overflow".into(),
                })?;
            let mut window_topk = arena.window_topk.prefix(window_topk_len)?;
            let stage_start = operators.profile_start();
            kv.paged_window_sparse_attention_rows_into(
                &arena.query,
                &arena.visible_lens,
                &mut window_topk,
                rows,
                self.layer,
                SparseAttentionSpec {
                    heads: cfg.num_heads,
                    head_dim: cfg.head_dim,
                    topk: attention_topk,
                    softmax_scale: (cfg.head_dim as f32).powf(-0.5),
                    has_attention_sink: !self.payload.attention_sink.is_empty(),
                },
                &arena.attention_sink,
                &mut arena.context,
            )?;
            record_attention_stage(
                operators,
                self.layer,
                MlaProfileStage::SparseAttention,
                stage_start,
            )?;
            return self.project_decode_context_rows_from_device_into(
                max_position,
                operators,
                arena,
            );
        }

        let compressed = self.compressed.as_ref().ok_or_else(|| Error::Model {
            message: format!(
                "MLA layer {} has compressed config without payload",
                self.layer
            ),
        })?;
        let cfg = self.config;
        let rope = cfg.rope_params();
        let max_position = positions
            .iter()
            .copied()
            .max()
            .ok_or_else(|| Error::Model {
                message: "packed decode positions are empty".into(),
            })?;
        let positions_i32 = decode_metadata_i32(positions, "position")?;
        {
            let ops = &operators.ops;
            ops.update_i32_host_mirror(&positions_i32, &mut arena.positions)?;
            ops.update_i32_host_mirror(&visible_lens, &mut arena.visible_lens)?;
        }
        self.project_decode_rows_from_device_into(
            hidden_dev,
            hidden_fp8,
            max_position,
            operators,
            arena,
        )?;
        if sequence_phases.contains(&ForwardPhase::Prefill) {
            if compressed.indexer.is_some() {
                let projected = arena
                    .indexer_compressor
                    .as_mut()
                    .expect("indexer compressor arena exists");
                operators.compressor_rows_from_device_into(
                    self.layer,
                    MlaCompressorKind::Indexer,
                    hidden_dev,
                    rows,
                    &mut projected.kv,
                    &mut projected.score,
                )?;
            }
            let projected = arena
                .main_compressor
                .as_mut()
                .expect("main compressor arena exists");
            operators.compressor_rows_from_device_into(
                self.layer,
                MlaCompressorKind::Main,
                hidden_dev,
                rows,
                &mut projected.kv,
                &mut projected.score,
            )?;
        }
        (|| -> Result<()> {
            kv.paged_scatter_rows_from_device(
                0,
                self.layer,
                &arena.kv,
                &arena.positions,
                None,
                cfg.head_dim,
            )?;

            let mut main_positions = vec![0i32; rows];
            let mut main_mask = vec![0i32; rows];
            let mut indexer_positions = vec![0i32; rows];
            let mut indexer_mask = vec![0i32; rows];
            let mut window_lens = vec![0usize; rows];
            let mut compressed_lens = vec![0usize; rows];
            let mut main_compressed_lens = vec![0usize; rows];
            for &row in sequence_major_rows {
                let sequence = row_to_sequence[row];
                let cache = &mut caches[sequence];
                cache.window.record_device_rows(1);
                operators
                    .ops
                    .copy_f32_range(
                        hidden_dev,
                        row * cfg.hidden_size,
                        &mut transition.input,
                        0,
                        cfg.hidden_size,
                    )
                    .map_err(|error| {
                        Error::Internal { message: format!(
                            "MLA layer {} row {row} transition input copy failed: source_len={} destination_len={}: {error}",
                            self.layer,
                            hidden_dev.len(),
                            transition.input.len()
                        ) }
                    })?;

                if let Some(indexer) = compressed.indexer.as_ref() {
                    let new_indexer_kv = {
                        let (compressor_state, window) =
                            (&mut cache.indexer_compressor, &mut cache.window);
                        let state = compressor_state.as_mut().ok_or_else(|| Error::Model {
                            message: format!(
                                "MLA layer {} missing indexer compressor state",
                                self.layer
                            ),
                        })?;
                        let scratch = transition.indexer_compressor.as_mut().ok_or_else(|| {
                            Error::Internal {
                                message: format!(
                                    "MLA layer {} missing indexer compressor row arena",
                                    self.layer
                                ),
                            }
                        })?;
                        let cuda = window.cuda_state_mut();
                        if sequence_phases[sequence] == ForwardPhase::Prefill {
                            let projected = arena
                                .indexer_compressor
                                .as_ref()
                                .expect("indexer compressor arena exists");
                            state.append_projected_step_from_device_into(
                                &indexer.compressor,
                                self.layer,
                                MlaCompressorKind::Indexer,
                                &projected.kv,
                                &projected.score,
                                row,
                                positions[row],
                                cfg.rope_head_dim,
                                rope,
                                &format!("rope_indexer_compress_L{}", self.layer),
                                &mut cuda.indexer_compressor_recurrent,
                                &mut cuda.indexer_compressor_needs_reset,
                                operators,
                                scratch,
                            )?
                        } else {
                            state.append_step_from_device_into(
                                &indexer.compressor,
                                self.layer,
                                MlaCompressorKind::Indexer,
                                &transition.input,
                                positions[row],
                                cfg.rope_head_dim,
                                rope,
                                &format!("rope_indexer_compress_L{}", self.layer),
                                &mut cuda.indexer_compressor_recurrent,
                                &mut cuda.indexer_compressor_needs_reset,
                                operators,
                                scratch,
                            )?
                        }
                    };
                    operators.fail_compressor_transition_if_armed(true)?;
                    if new_indexer_kv {
                        cache.record_indexer_compressed_rows(1)?;
                        indexer_positions[row] =
                            i32::try_from(positions[row]).map_err(|_| Error::Model {
                                message: "packed indexer boundary position exceeds i32 ABI".into(),
                            })?;
                        indexer_mask[row] = 1;
                        let transition_normalized = &transition
                            .indexer_compressor
                            .as_ref()
                            .expect("indexer compressor row arena exists")
                            .normalized;
                        let packed_normalized = &mut arena
                            .indexer_compressor
                            .as_mut()
                            .expect("indexer compressor arena exists")
                            .normalized;
                        operators
                            .ops
                            .copy_f32_range(
                                transition_normalized,
                                0,
                                packed_normalized,
                                row * cfg.index_head_dim,
                                cfg.index_head_dim,
                            )
                            .map_err(|error| {
                                Error::Internal { message: format!(
                                    "MLA layer {} row {row} indexer normalized copy failed: source_len={} destination_len={} row_dim={}: {error}",
                                    self.layer,
                                    transition_normalized.len(),
                                    packed_normalized.len(),
                                    cfg.index_head_dim
                                ) }
                            })?;
                    }
                }

                let new_main_kv = {
                    let (compressor_state, window) =
                        (&mut cache.main_compressor, &mut cache.window);
                    let state = compressor_state.as_mut().ok_or_else(|| Error::Model {
                        message: format!("MLA layer {} missing main compressor state", self.layer),
                    })?;
                    let scratch =
                        transition
                            .main_compressor
                            .as_mut()
                            .ok_or_else(|| Error::Internal {
                                message: format!(
                                    "MLA layer {} missing main compressor row arena",
                                    self.layer
                                ),
                            })?;
                    let cuda = window.cuda_state_mut();
                    if sequence_phases[sequence] == ForwardPhase::Prefill {
                        let projected = arena
                            .main_compressor
                            .as_ref()
                            .expect("main compressor arena exists");
                        state.append_projected_step_from_device_into(
                            &compressed.compressor,
                            self.layer,
                            MlaCompressorKind::Main,
                            &projected.kv,
                            &projected.score,
                            row,
                            positions[row],
                            cfg.rope_head_dim,
                            rope,
                            &format!("rope_main_compress_L{}", self.layer),
                            &mut cuda.main_compressor_recurrent,
                            &mut cuda.main_compressor_needs_reset,
                            operators,
                            scratch,
                        )?
                    } else {
                        state.append_step_from_device_into(
                            &compressed.compressor,
                            self.layer,
                            MlaCompressorKind::Main,
                            &transition.input,
                            positions[row],
                            cfg.rope_head_dim,
                            rope,
                            &format!("rope_main_compress_L{}", self.layer),
                            &mut cuda.main_compressor_recurrent,
                            &mut cuda.main_compressor_needs_reset,
                            operators,
                            scratch,
                        )?
                    }
                };
                operators.fail_compressor_transition_if_armed(false)?;
                if new_main_kv {
                    cache.record_compressed_rows(1)?;
                    main_positions[row] =
                        i32::try_from(positions[row]).map_err(|_| Error::Model {
                            message: "packed main compressed boundary position exceeds i32 ABI"
                                .into(),
                        })?;
                    main_mask[row] = 1;
                    let transition_normalized = &transition
                        .main_compressor
                        .as_ref()
                        .expect("main compressor row arena exists")
                        .normalized;
                    let packed_normalized = &mut arena
                        .main_compressor
                        .as_mut()
                        .expect("main compressor arena exists")
                        .normalized;
                    operators
                        .ops
                        .copy_f32_range(
                            transition_normalized,
                            0,
                            packed_normalized,
                            row * cfg.head_dim,
                            cfg.head_dim,
                        )
                        .map_err(|error| {
                            Error::Internal { message: format!(
                                "MLA layer {} row {row} main normalized copy failed: source_len={} destination_len={} row_dim={}: {error}",
                                self.layer,
                                transition_normalized.len(),
                                packed_normalized.len(),
                                cfg.head_dim
                            ) }
                        })?;
                }
                window_lens[row] = cache.window.len;
                let main_compressed_rows = cache.compressed_len();
                main_compressed_lens[row] = if main_compressed_rows == 0 {
                    0
                } else {
                    positions[row].checked_add(1).ok_or_else(|| Error::Model {
                        message: "packed main compressed visible length overflow".into(),
                    })?
                };
                compressed_lens[row] = if compressed.indexer.is_some() {
                    cache.indexer_compressed_len(cfg.index_head_dim)
                } else {
                    main_compressed_rows
                };
                kv.capture_provisional_prefix_checkpoint(
                    self.layer,
                    row,
                    cache.window.cuda_state(),
                    cache.window.len,
                    cache.compressed_rows,
                    cache.indexer_compressed_rows,
                )?;
            }

            {
                let ops = &operators.ops;
                ops.update_i32_host_mirror(&main_positions, &mut arena.main_positions)?;
                ops.update_i32_host_mirror(&main_mask, &mut arena.main_mask)?;
                ops.update_i32_host_mirror(&indexer_positions, &mut arena.indexer_positions)?;
                ops.update_i32_host_mirror(&indexer_mask, &mut arena.indexer_mask)?;
            }
            kv.paged_scatter_rows_from_device(
                1,
                self.layer,
                &arena
                    .main_compressor
                    .as_ref()
                    .expect("main compressor arena exists")
                    .normalized,
                &arena.main_positions,
                Some(&arena.main_mask),
                cfg.head_dim,
            )?;
            if compressed.indexer.is_some() {
                kv.paged_scatter_rows_from_device(
                    2,
                    self.layer,
                    &arena
                        .indexer_compressor
                        .as_ref()
                        .expect("indexer compressor arena exists")
                        .normalized,
                    &arena.indexer_positions,
                    Some(&arena.indexer_mask),
                    cfg.index_head_dim,
                )?;
            }

            if main_compressed_lens.iter().all(|&length| length == 0)
                && compressed_lens.iter().all(|&length| length == 0)
            {
                let attention_topk =
                    window_lens
                        .iter()
                        .copied()
                        .max()
                        .ok_or_else(|| Error::Model {
                            message: "packed compressed attention window lengths are empty".into(),
                        })?;
                let window_topk_len =
                    rows.checked_mul(attention_topk)
                        .ok_or_else(|| Error::Model {
                            message: "packed compressed window top-k size overflow".into(),
                        })?;
                let mut window_topk = arena.window_topk.prefix(window_topk_len)?;
                let stage_start = operators.profile_start();
                kv.paged_window_sparse_attention_rows_into(
                    &arena.query,
                    &arena.visible_lens,
                    &mut window_topk,
                    rows,
                    self.layer,
                    SparseAttentionSpec {
                        heads: cfg.num_heads,
                        head_dim: cfg.head_dim,
                        topk: attention_topk,
                        softmax_scale: (cfg.head_dim as f32).powf(-0.5),
                        has_attention_sink: !self.payload.attention_sink.is_empty(),
                    },
                    &arena.attention_sink,
                    &mut arena.context,
                )?;
                record_attention_stage(
                    operators,
                    self.layer,
                    MlaProfileStage::SparseAttention,
                    stage_start,
                )?;
                return Ok(());
            }

            if compressed.indexer.is_some() {
                operators.linear_rows_from_device_into(
                    self.layer,
                    MlaLinearKind::IndexerQuery,
                    &arena.q_indexer,
                    rows,
                    &mut arena.index_query,
                    &mut arena.linear_workspace,
                )?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                let required_positions = positions
                    .iter()
                    .copied()
                    .max()
                    .and_then(|position| position.checked_add(1))
                    .ok_or_else(|| Error::Model {
                        message: "packed indexer RoPE position overflow".into(),
                    })?;
                operators.ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    required_positions,
                )?;
                operators.rope_tail_rows_indexed_from_device(
                    &index_rope_name,
                    &mut arena.index_query,
                    &arena.positions,
                    max_position,
                    cfg.index_n_heads as u32,
                    cfg.index_head_dim as u32,
                    index_rope_dim as u32,
                    false,
                )?;
                operators.ops.fp4_hadamard_qat_quantize_buffer_in_place(
                    &mut arena.index_query,
                    cfg.index_head_dim,
                )?;
                operators.linear_rows_from_device_into(
                    self.layer,
                    MlaLinearKind::IndexerWeights,
                    hidden_dev,
                    rows,
                    &mut arena.index_weights,
                    &mut arena.linear_workspace,
                )?;
            }
            let window_lens_i32 = decode_metadata_i32(&window_lens, "window length")?;
            let compressed_lens_i32 = decode_metadata_i32(&compressed_lens, "compressed length")?;
            let main_compressed_lens_i32 =
                decode_metadata_i32(&main_compressed_lens, "main compressed length")?;
            let sequence_main_compressed_lens_i32 = decode_metadata_i32(
                &caches
                    .iter()
                    .zip(paged_bindings)
                    .map(|(cache, binding)| {
                        if cache.compressed_len() == 0 {
                            0
                        } else {
                            binding.sequence_len
                        }
                    })
                    .collect::<Vec<_>>(),
                "sequence main compressed visible length",
            )?;
            {
                let ops = &operators.ops;
                ops.update_i32_host_mirror(&window_lens_i32, &mut arena.window_lens)?;
                ops.update_i32_host_mirror(&compressed_lens_i32, &mut arena.compressed_lens)?;
                ops.update_i32_host_mirror(
                    &main_compressed_lens_i32,
                    &mut arena.main_compressed_lens,
                )?;
            }
            let weight_scale =
                (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
            kv.decode_topk_indices_paged_indexer_rows_into(
                &arena.index_query,
                &arena.index_weights,
                &arena.positions,
                &arena.window_lens,
                &arena.compressed_lens,
                self.layer,
                cfg.window_size,
                cfg.index_topk,
                cfg.compress_ratio,
                compressed.indexer.is_none(),
                cfg.index_n_heads,
                cfg.index_head_dim,
                weight_scale,
                &mut arena.topk,
                &mut arena.topk_selectors,
            )?;
            kv.dual_plane_paged_sparse_attention_rows_into(
                &arena.query,
                &arena.topk,
                &arena.topk_selectors,
                &arena.visible_lens,
                &arena.main_compressed_lens,
                &sequence_main_compressed_lens_i32,
                rows,
                self.layer,
                cfg.sparse_spec_with_topk(cfg.window_size + cfg.index_topk),
                &arena.attention_sink,
                &mut arena.context,
            )?;

            Ok(())
        })()?;
        self.project_decode_context_rows_from_device_into(max_position, operators, arena)
    }

    /// Proposal attention is deliberately separate from ordinary packed
    /// target attention: all five rows see the complete ephemeral block, and the
    /// block KV is never appended to the committed page table.
    #[cfg(feature = "cuda")]
    pub(crate) fn proposal_block_from_device_into(
        &self,
        kv: &mut MlaKvView,
        stage: usize,
        hidden_fp8: &ferrule_backend::cuda::operators::linear::CudaPreparedFp8Activation<'_>,
        sequence_tokens: usize,
        operators: &mut MlaExecution,
        arena: &mut MlaDecodeArena,
        proposal: &mut MlaProposalAttentionBuffers,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = PROPOSAL_ROWS;
        if sequence_tokens == 0
            || cfg.compress_ratio != 0
            || arena.query.len() != rows.saturating_mul(cfg.q_full_dim())
            || arena.kv.len() != rows.saturating_mul(cfg.head_dim)
            || arena.context.len() != rows.saturating_mul(cfg.q_full_dim())
        {
            return Err(Error::Model {
                message: format!(
                    "MLA Proposal-attention shape mismatch at stage {stage}: sequence_tokens={sequence_tokens} compress_ratio={} query={} kv={} context={} rows={rows}",
                    cfg.compress_ratio,
                    arena.query.len(),
                    arena.kv.len(),
                    arena.context.len()
                ),
            });
        }
        let required_positions = sequence_tokens
            .checked_add(rows)
            .ok_or_else(|| Error::Model {
                message: "MLA Proposal position overflow".into(),
            })?;
        let rope_name = format!("rope_proposal_stage_{stage}");
        operators.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            required_positions,
        )?;

        let stage_start = operators.profile_start();
        operators.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(operators, self.layer, MlaProfileStage::Qa, stage_start)?;

        let stage_start = operators.profile_start();
        operators.rms_norm_layer_rows_device_into(
            self.layer,
            MlaNormKind::Query,
            &arena.q_latent,
            rows,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(operators, self.layer, MlaProfileStage::QNorm, stage_start)?;

        let stage_start = operators.profile_start();
        operators.linear_rows_from_device_into(
            self.layer,
            MlaLinearKind::QueryB,
            &arena.q_norm,
            rows,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        record_attention_stage(operators, self.layer, MlaProfileStage::Qb, stage_start)?;

        let stage_start = operators.profile_start();
        operators.rms_norm_heads_from_device_into(
            &arena.query_raw,
            rows * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            MlaProfileStage::QHeadNorm,
            stage_start,
        )?;
        operators.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.query,
            u32::try_from(sequence_tokens).map_err(|_| Error::Model {
                message: "MLA Proposal position exceeds u32".into(),
            })?,
            u32::try_from(rows).map_err(|_| Error::Model {
                message: "MLA Proposal row count exceeds u32".into(),
            })?,
            u32::try_from(cfg.num_heads).map_err(|_| Error::Model {
                message: "MLA Proposal head count exceeds u32".into(),
            })?,
            u32::try_from(cfg.head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal head dim exceeds u32".into(),
            })?,
            u32::try_from(cfg.rope_head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal RoPE dim exceeds u32".into(),
            })?,
            false,
        )?;

        let stage_start = operators.profile_start();
        operators.rms_norm_layer_rows_device_into(
            self.layer,
            MlaNormKind::KeyValue,
            &arena.kv_raw,
            rows,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        operators.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.kv,
            u32::try_from(sequence_tokens).map_err(|_| Error::Model {
                message: "MLA Proposal position exceeds u32".into(),
            })?,
            u32::try_from(rows).map_err(|_| Error::Model {
                message: "MLA Proposal row count exceeds u32".into(),
            })?,
            1,
            u32::try_from(cfg.head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal head dim exceeds u32".into(),
            })?,
            u32::try_from(cfg.rope_head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal RoPE dim exceeds u32".into(),
            })?,
            false,
        )?;
        operators
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(operators, self.layer, MlaProfileStage::KvNorm, stage_start)?;

        let stage_start = operators.profile_start();
        operators.proposal_hybrid_attention_device_into(
            kv,
            self.layer,
            cfg,
            sequence_tokens,
            &arena.query,
            &arena.kv,
            &mut arena.context,
            proposal,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            MlaProfileStage::SparseAttention,
            stage_start,
        )?;

        operators.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.context,
            u32::try_from(sequence_tokens).map_err(|_| Error::Model {
                message: "MLA Proposal position exceeds u32".into(),
            })?,
            u32::try_from(rows).map_err(|_| Error::Model {
                message: "MLA Proposal row count exceeds u32".into(),
            })?,
            u32::try_from(cfg.num_heads).map_err(|_| Error::Model {
                message: "MLA Proposal head count exceeds u32".into(),
            })?,
            u32::try_from(cfg.head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal head dim exceeds u32".into(),
            })?,
            u32::try_from(cfg.rope_head_dim).map_err(|_| Error::Model {
                message: "MLA Proposal RoPE dim exceeds u32".into(),
            })?,
            true,
        )?;
        operators.mla_output_rows_from_device_into(
            &arena.context,
            rows,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )
    }
}

fn record_attention_stage(
    operators: &mut MlaExecution,
    layer: usize,
    stage: MlaProfileStage,
    start: Option<Instant>,
) -> Result<()> {
    let Some(elapsed_us) = operators.finish_profile_stage(start).map_err(|error| {
        Error::Internal {
            message: format!(
                "MLA layer {layer} attention stage {stage:?} failed while synchronizing: {error}"
            ),
        }
    })?
    else {
        return Ok(());
    };
    let _ = (layer, stage, elapsed_us);
    Ok(())
}

#[cfg(feature = "cuda")]
fn required_rope_positions(
    start_position: usize,
    position_stride: usize,
    rows: usize,
) -> Result<usize> {
    if rows == 0 {
        return Err(Error::Model {
            message: "MLA CUDA RoPE launch requires at least one row".into(),
        });
    }
    for (field, value) in [
        ("start_position", start_position),
        ("position_stride", position_stride),
        ("rows", rows),
    ] {
        u32::try_from(value).map_err(|_| Error::Model {
            message: format!("MLA CUDA RoPE {field} exceeds the u32 kernel ABI: {value}"),
        })?;
    }
    let last_offset = (rows - 1)
        .checked_mul(position_stride)
        .ok_or_else(|| Error::Model {
            message: "MLA CUDA RoPE row-stride overflow".into(),
        })?;
    start_position
        .checked_add(last_offset)
        .and_then(|position| position.checked_add(1))
        .ok_or_else(|| Error::Model {
            message: "MLA CUDA RoPE position overflow".into(),
        })
}

fn validate_ratio(ratio: usize) -> Result<()> {
    if ratio == 0 || COMPRESS_RATIOS.contains(&ratio) {
        Ok(())
    } else {
        Err(model_error(format!(
            "compress_ratio must be 0, 4, or 128, got {ratio}"
        )))
    }
}

fn validate_compressor(
    layer: usize,
    payload: &MlaCompressor,
    hidden_size: usize,
    head_dim: usize,
    indexer: bool,
) -> Result<()> {
    let overlap = payload.compress_ratio == 4;
    let out_dim = head_dim * if overlap { 2 } else { 1 };
    if payload.head_dim != head_dim
        || payload.overlap != overlap
        || payload.rotate_for_indexer != indexer
        || (payload.ape_rows, payload.ape_cols) != (payload.compress_ratio, out_dim)
        || payload.ape.len() != payload.ape_rows * payload.ape_cols
    {
        return Err(model_error(format!(
            "layer {layer} compressor payload shape/rotation is inconsistent"
        )));
    }
    check_len(layer, "compressor.norm", payload.norm.len(), head_dim)?;
    check_linear(layer, "compressor.wkv", &payload.wkv, out_dim, hidden_size)?;
    check_linear(
        layer,
        "compressor.wgate",
        &payload.wgate,
        out_dim,
        hidden_size,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaKvState {
    pub window: MlaWindowKv,
    compressed_rows: usize,
    indexer_compressed_rows: usize,
    main_compressor: Option<MlaCompressorState>,
    indexer_compressor: Option<MlaCompressorState>,
}

impl MlaKvState {
    fn new(config: MlaConfig) -> Self {
        Self {
            window: MlaWindowKv::new(config.window_size, config.head_dim),
            compressed_rows: 0,
            indexer_compressed_rows: 0,
            main_compressor: (config.compress_ratio != 0)
                .then(|| MlaCompressorState::new(config.compress_ratio, config.head_dim)),
            indexer_compressor: (config.compress_ratio == 4)
                .then(|| MlaCompressorState::new(4, config.index_head_dim)),
        }
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn compressed_len(&self) -> usize {
        self.compressed_rows
    }

    pub fn indexer_compressed_len(&self, _head_dim: usize) -> usize {
        self.indexer_compressed_rows
    }

    pub fn reset_sequence(&mut self) {
        self.window.clear();
        self.compressed_rows = 0;
        self.indexer_compressed_rows = 0;
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn fork_paged_prefix_metadata(&self) -> Self {
        let mut window = MlaWindowKv::new(self.window.window_size, self.window.head_dim);
        window.len = self.window.len;
        Self {
            window,
            compressed_rows: self.compressed_rows,
            indexer_compressed_rows: self.indexer_compressed_rows,
            main_compressor: self.main_compressor.clone(),
            indexer_compressor: self.indexer_compressor.clone(),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn restore_provisional_prefix_metadata(
        &mut self,
        metadata: MlaPrefixCheckpoint,
    ) -> Result<()> {
        if metadata.window_len > self.window.window_size {
            return Err(Error::Model {
                message: format!(
                    "MLA restored window length {} exceeds capacity {}",
                    metadata.window_len, self.window.window_size
                ),
            });
        }
        self.window.len = metadata.window_len;
        self.compressed_rows = metadata.compressed_rows;
        self.indexer_compressed_rows = metadata.indexer_compressed_rows;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn restore_uncompressed_prefix_from(
        &mut self,
        source: &Self,
        retained_rows: usize,
    ) -> Result<()> {
        if self.main_compressor.is_some()
            || self.indexer_compressor.is_some()
            || source.main_compressor.is_some()
            || source.indexer_compressor.is_some()
        {
            return Err(Error::Model {
                message: "MLA uncompressed prefix restore received compressor state".into(),
            });
        }
        self.window.len = self
            .window
            .window_size
            .min(source.window.len.saturating_add(retained_rows));
        self.compressed_rows = source.compressed_rows;
        self.indexer_compressed_rows = source.indexer_compressed_rows;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn record_compressed_rows(&mut self, rows: usize) -> Result<()> {
        self.compressed_rows =
            self.compressed_rows
                .checked_add(rows)
                .ok_or_else(|| Error::Model {
                    message: "MLA compressed KV row count overflow".into(),
                })?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn record_indexer_compressed_rows(&mut self, rows: usize) -> Result<()> {
        self.indexer_compressed_rows =
            self.indexer_compressed_rows
                .checked_add(rows)
                .ok_or_else(|| Error::Model {
                    message: "MLA indexer compressed KV row count overflow".into(),
                })?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaCompressorState {
    ratio: usize,
    head_dim: usize,
    out_dim: usize,
    overlap: bool,
}

impl MlaCompressorState {
    fn new(ratio: usize, head_dim: usize) -> Self {
        let overlap = ratio == 4;
        let coefficient = if overlap { 2 } else { 1 };
        let out_dim = coefficient * head_dim;
        Self {
            ratio,
            head_dim,
            out_dim,
            overlap,
        }
    }

    #[cfg(feature = "cuda")]
    fn append_step_from_device_into(
        &mut self,
        payload: &MlaCompressor,
        layer: usize,
        compressor: MlaCompressorKind,
        hidden_dev: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        position: usize,
        rope_dim: usize,
        rope: MlaRopeConfig,
        rope_name: &str,
        recurrent_state: &mut Option<CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut MlaExecution,
        scratch: &mut MlaCompressorArena,
    ) -> Result<bool> {
        self.validate_append_payload(payload)?;
        let (kv_linear, gate_linear) = match compressor {
            MlaCompressorKind::Main => (
                MlaLinearKind::MainCompressorKv,
                MlaLinearKind::MainCompressorGate,
            ),
            MlaCompressorKind::Indexer => (
                MlaLinearKind::IndexerCompressorKv,
                MlaLinearKind::IndexerCompressorGate,
            ),
        };
        operators.linear_pair_matvec_readonly_from_device_into(
            layer,
            kv_linear,
            gate_linear,
            hidden_dev,
            &mut scratch.kv,
            &mut scratch.score,
        )?;
        self.finish_projected_step_from_device_into(
            layer,
            compressor,
            position,
            rope_dim,
            rope,
            rope_name,
            recurrent_state,
            recurrent_needs_reset,
            operators,
            scratch,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn append_projected_step_from_device_into(
        &mut self,
        payload: &MlaCompressor,
        layer: usize,
        compressor: MlaCompressorKind,
        projected_kv: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        projected_score: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        row: usize,
        position: usize,
        rope_dim: usize,
        rope: MlaRopeConfig,
        rope_name: &str,
        recurrent_state: &mut Option<CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut MlaExecution,
        scratch: &mut MlaCompressorArena,
    ) -> Result<bool> {
        self.validate_append_payload(payload)?;
        let offset = row.checked_mul(self.out_dim).ok_or_else(|| Error::Model {
            message: "compressor projected row offset overflow".into(),
        })?;
        if projected_kv.len() < offset + self.out_dim
            || projected_score.len() < offset + self.out_dim
        {
            return Err(Error::Model {
                message: "compressor projected packed rows are too short".into(),
            });
        }
        operators
            .ops
            .copy_f32_range(projected_kv, offset, &mut scratch.kv, 0, self.out_dim)?;
        operators.ops.copy_f32_range(
            projected_score,
            offset,
            &mut scratch.score,
            0,
            self.out_dim,
        )?;
        self.finish_projected_step_from_device_into(
            layer,
            compressor,
            position,
            rope_dim,
            rope,
            rope_name,
            recurrent_state,
            recurrent_needs_reset,
            operators,
            scratch,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn finish_projected_step_from_device_into(
        &mut self,
        layer: usize,
        compressor: MlaCompressorKind,
        position: usize,
        rope_dim: usize,
        rope: MlaRopeConfig,
        rope_name: &str,
        recurrent_state: &mut Option<CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut MlaExecution,
        scratch: &mut MlaCompressorArena,
    ) -> Result<bool> {
        let boundary = operators.compressor_recurrent_append_into(
            layer,
            compressor,
            recurrent_state,
            recurrent_needs_reset,
            &scratch.kv,
            &scratch.score,
            position,
            self.ratio,
            self.head_dim,
            self.out_dim,
            self.overlap,
            &mut scratch.compressed,
        )?;
        if !boundary {
            return Ok(false);
        }
        operators.rms_norm_compressor_rows_device_into(
            layer,
            compressor,
            &scratch.compressed,
            1,
            1e-6,
            &mut scratch.normalized,
        )?;
        let compressed_position = position + 1 - self.ratio;
        let effective_rope_dim = rope_dim.min(self.head_dim);
        let rope_positions = required_rope_positions(compressed_position, self.ratio, 1)?;
        operators.ensure_rope_tables_with_params(
            rope_name,
            effective_rope_dim,
            rope,
            rope_positions,
        )?;
        operators.rope_tail_rows_strided_from_device(
            rope_name,
            &mut scratch.normalized,
            compressed_position as u32,
            self.ratio as u32,
            1,
            1,
            self.head_dim as u32,
            effective_rope_dim as u32,
            false,
        )?;
        if compressor == MlaCompressorKind::Indexer {
            operators.ops.fp4_hadamard_qat_quantize_buffer_in_place(
                &mut scratch.normalized,
                self.head_dim,
            )?;
        } else {
            operators
                .ops
                .fp8_attention_kv_qat_quantize_buffer_in_place(
                    &mut scratch.normalized,
                    self.head_dim,
                    effective_rope_dim,
                )?;
        }
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn validate_append_payload(&self, payload: &MlaCompressor) -> Result<()> {
        if payload.compress_ratio != self.ratio
            || payload.head_dim != self.head_dim
            || payload.overlap != self.overlap
        {
            return Err(Error::Model {
                message: "MLA compressor payload/state shape mismatch".into(),
            });
        }
        Ok(())
    }
}

pub struct MlaWindowKv {
    pub(crate) window_size: usize,
    pub(crate) head_dim: usize,
    len: usize,
    cuda: MlaRecurrentState,
}

impl Clone for MlaWindowKv {
    fn clone(&self) -> Self {
        Self {
            window_size: self.window_size,
            head_dim: self.head_dim,
            len: self.len,
            cuda: MlaRecurrentState::default(),
        }
    }
}

impl PartialEq for MlaWindowKv {
    fn eq(&self, other: &Self) -> bool {
        self.window_size == other.window_size
            && self.head_dim == other.head_dim
            && self.len == other.len
    }
}

impl std::fmt::Debug for MlaWindowKv {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlaWindowKv")
            .field("window_size", &self.window_size)
            .field("head_dim", &self.head_dim)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl MlaWindowKv {
    pub fn new(window_size: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            head_dim,
            len: 0,
            cuda: MlaRecurrentState::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.cuda.reset_for_reuse();
    }

    pub(crate) fn record_device_rows(&mut self, rows: usize) {
        self.len = self.window_size.min(self.len.saturating_add(rows));
    }

    pub(crate) fn cuda_state(&self) -> &MlaRecurrentState {
        &self.cuda
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_state_mut(&mut self) -> &mut MlaRecurrentState {
        &mut self.cuda
    }
}

fn validate_rope_table_request(
    name: &str,
    rope_dim: usize,
    rope: MlaRopeConfig,
    required_positions: usize,
) -> Result<()> {
    if name.is_empty()
        || rope_dim == 0
        || !rope_dim.is_multiple_of(2)
        || required_positions == 0
        || !rope.theta.is_finite()
        || rope.theta <= 0.0
        || !rope.factor.is_finite()
        || rope.factor <= 0.0
    {
        return Err(model_error(format!(
            "invalid RoPE table request: name={name:?} dim={rope_dim} positions={required_positions} params={rope:?}"
        )));
    }
    Ok(())
}

fn validate_rope_table_identity(
    name: &str,
    table: &MlaRopeTable,
    rope_dim: usize,
    rope: MlaRopeConfig,
) -> Result<()> {
    if table.rope_dim != rope_dim || table.rope != rope {
        return Err(model_error(format!(
            "RoPE table {name:?} identity mismatch: cached_dim={} requested_dim={rope_dim} cached={:?} requested={rope:?}",
            table.rope_dim, table.rope
        )));
    }
    Ok(())
}

fn validate_rope_table_capacity(
    name: &str,
    table: &MlaRopeTable,
    required_positions: usize,
) -> Result<()> {
    if table.capacity < required_positions {
        return Err(model_error(format!(
            "RoPE table {name:?} capacity {} is smaller than required {required_positions}",
            table.capacity
        )));
    }
    Ok(())
}

fn rope_table_capacity(required_positions: usize) -> Result<usize> {
    required_positions
        .checked_next_power_of_two()
        .ok_or_else(|| model_error("RoPE table capacity overflow"))
}

fn yarn_frequency(pair: usize, rope_dim: usize, params: MlaRopeConfig) -> f32 {
    ferrule_backend::cpu::rotary_frequency(
        pair,
        rope_dim,
        ferrule_backend::cpu::RotaryFrequencyParams {
            theta: params.theta,
            original_sequence_length: params.original_seq_len,
            factor: params.factor,
            beta_fast: params.beta_fast,
            beta_slow: params.beta_slow,
        },
    )
}

fn check_linear(
    layer: usize,
    label: &str,
    linear: &LinearWeight,
    output: usize,
    input: usize,
) -> Result<()> {
    if linear.format.out_features() != output || linear.format.in_features() != input {
        return Err(model_error(format!(
            "layer {layer} {label} shape mismatch: got {}x{}, expected {output}x{input}",
            linear.format.out_features(),
            linear.format.in_features()
        )));
    }
    Ok(())
}

fn check_len(layer: usize, label: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(model_error(format!(
            "layer {layer} {label} length mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("MLA: {}", message.into()),
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("MLA: {}", message.into()),
    }
}

#[cfg(test)]
mod provisional_retain_tests {
    use super::*;
    use crate::decoder::DecoderSequenceAttachment;
    use ferrule_common::execution::{
        ExecutionBatch, ExecutionCapabilities, ExecutionIntent, ExecutionSequence, ForwardMode,
        KvBindingMode, KvBlockId, KvReservationView, KvWriteSlot, LogitsRequest, LogitsRowPolicy,
        StateSlot,
    };
    use std::num::NonZeroU32;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestAttachment {
        resets: usize,
    }

    impl DecoderSequenceAttachment for TestAttachment {
        type Release = ();

        fn preflight_release(&self) -> Result<Self::Release> {
            Ok(())
        }

        fn release(self, _release: Self::Release) {}
    }

    impl MlaSequenceAttachment for TestAttachment {
        fn reset_mla_sequence(&mut self) {
            self.resets += 1;
        }
    }

    fn compressed_config() -> MlaConfig {
        MlaConfig {
            hidden_size: 4,
            num_heads: 1,
            head_dim: 2,
            q_lora_rank: 2,
            rope_head_dim: 2,
            o_groups: 1,
            o_lora_rank: 2,
            window_size: 8,
            compress_ratio: 128,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
            compress_rope_theta: 10_000.0,
            original_seq_len: 8,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            index_n_heads: 1,
            index_head_dim: 2,
            index_topk: 1,
        }
    }

    fn source_state() -> crate::decoder::DecoderSequenceState<TestAttachment, MlaSequenceState> {
        crate::decoder::DecoderSequenceState::new(
            TestAttachment { resets: 0 },
            MlaSequenceState::new([compressed_config()], std::iter::empty()),
        )
    }

    fn working_state(
        pages: &[KvPageId],
    ) -> crate::decoder::DecoderSequenceState<TestAttachment, MlaSequenceState> {
        let mut state = source_state();
        let kv = state.kv_state_mut();
        kv.layers[0].kv.window.len = 5;
        kv.layers[0].kv.compressed_rows = 3;
        kv.layers[0].kv.indexer_compressed_rows = 2;
        kv.paged_kv_binding = Some(MlaPagedKvBinding {
            logical_pages: pages.to_vec(),
            physical_block_slots: (0..pages.len())
                .map(|slot| i32::try_from(slot).unwrap())
                .collect(),
            sequence_len: 5,
            page_tokens: 2,
            layer_count: 1,
        });
        state
    }

    fn checkpoints(
        main: Option<CudaCompressorRecurrentCheckpointSlab>,
    ) -> MlaProvisionalPrefixCheckpoints {
        let metadata = (1..5)
            .map(|retained| {
                Some(MlaPrefixCheckpoint {
                    window_len: retained,
                    compressed_rows: retained / 2,
                    indexer_compressed_rows: retained / 3,
                    main_compressor_needs_reset: retained.is_multiple_of(2),
                    indexer_compressor_needs_reset: false,
                })
            })
            .collect();
        MlaProvisionalPrefixCheckpoints {
            active: false,
            sequences: vec![MlaSequencePrefixCheckpoints {
                start_position: 0,
                executed_rows: 5,
                layers: vec![MlaLayerPrefixCheckpoints {
                    main,
                    indexer: None,
                    metadata,
                }],
            }],
            row_to_sequence: vec![0; 5],
            row_to_local: (0..5).collect(),
        }
    }

    #[test]
    fn retain_zero_partial_and_all_plan_exact_metadata_and_pages() {
        let pages = [KvPageId(10), KvPageId(11), KvPageId(12)];
        let sources = [source_state()];
        let working = [working_state(&pages)];
        let checkpoints = checkpoints(None);

        let zero =
            stage_provisional_retain(None, Some(&checkpoints), &sources, &working, &[5], &[0])
                .unwrap();
        let zero_state = zero.states[0].as_ref().unwrap();
        assert!(zero.retained_pages.is_empty());
        assert_eq!(zero_state.layers[0].kv.window.len, 0);
        assert_eq!(zero_state.layers[0].kv.compressed_rows, 0);
        assert!(zero_state.paged_kv_binding.is_none());

        let partial =
            stage_provisional_retain(None, Some(&checkpoints), &sources, &working, &[5], &[3])
                .unwrap();
        let partial_state = partial.states[0].as_ref().unwrap();
        assert_eq!(
            partial.retained_pages,
            BTreeSet::from([KvPageId(10), KvPageId(11)])
        );
        assert_eq!(partial_state.layers[0].kv.window.len, 3);
        assert_eq!(partial_state.layers[0].kv.compressed_rows, 1);
        assert_eq!(partial_state.layers[0].kv.indexer_compressed_rows, 1);
        let partial_binding = partial_state.paged_kv_binding.as_ref().unwrap();
        assert_eq!(partial_binding.logical_pages, [KvPageId(10), KvPageId(11)]);
        assert_eq!(partial_binding.sequence_len, 3);

        let all = stage_provisional_retain(None, None, &sources, &working, &[5], &[5]).unwrap();
        assert!(all.states[0].is_none());
        assert_eq!(all.retained_pages, BTreeSet::from(pages));
    }

    #[test]
    fn missing_partial_checkpoint_preserves_working_metadata() {
        let pages = [KvPageId(20), KvPageId(21), KvPageId(22)];
        let sources = [source_state()];
        let working = [working_state(&pages)];
        let before_window = working[0].kv_state().layers[0].kv.window.len;
        let before_compressed = working[0].kv_state().layers[0].kv.compressed_rows;
        let before_pages = working[0]
            .kv_state()
            .paged_kv_binding
            .as_ref()
            .unwrap()
            .logical_pages
            .clone();

        let error = stage_provisional_retain(None, None, &sources, &working, &[5], &[2])
            .expect_err("partial retain requires checkpoints");
        assert!(error.to_string().contains("no checkpoints"));
        assert_eq!(working[0].kv_state().layers[0].kv.window.len, before_window);
        assert_eq!(
            working[0].kv_state().layers[0].kv.compressed_rows,
            before_compressed
        );
        assert_eq!(
            working[0]
                .kv_state()
                .paged_kv_binding
                .as_ref()
                .unwrap()
                .logical_pages,
            before_pages
        );
    }

    fn packed_batch(
        source: &crate::decoder::DecoderSequenceState<TestAttachment, MlaSequenceState>,
        pages: &[KvPageId],
    ) -> PackedDecoderBatch {
        let positions = (0..5).collect::<Vec<u32>>();
        let write_slots = positions
            .iter()
            .map(|&position| {
                let position = usize::try_from(position).unwrap();
                let page = pages[position / 2];
                Some(
                    KvWriteSlot::try_from(usize::try_from(page.0).unwrap() * 2 + position % 2)
                        .unwrap(),
                )
            })
            .collect();
        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2, 3, 4, 5],
            positions,
            write_slots,
            vec![LogitsRequest::None; 5],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..5,
                0,
                5,
                0..3,
            )],
            pages.iter().map(|page| KvBlockId::new(page.0)).collect(),
        )
        .with_intent(ExecutionIntent::ProvisionalVerification);
        let reservations = [KvReservationView {
            state_slot: StateSlot::new(0),
            execution_state_slot: StateSlot::new(0),
            positions: 0..5,
            newly_allocated: pages.to_vec(),
            generation: source.core().generation(),
            execution_generation: source.core().generation(),
            cow_replacement: None,
        }];
        PackedDecoderBatch::lower(
            &batch,
            &reservations,
            std::slice::from_ref(source),
            &ExecutionCapabilities {
                max_batch_tokens: 5,
                max_sequences: 1,
                max_prefill_query_tokens_per_sequence: 5,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: NonZeroU32::new(1),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: false,
                full_logits_width: None,
                kv_binding_mode: KvBindingMode::Paged,
                logits_row_policy: LogitsRowPolicy::Any,
            },
            2,
            &|_| crate::decoder::DecoderKvPageStatus::Vacant,
        )
        .unwrap()
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_retain_failure_keeps_transaction_checkpoint_state_and_pages_retryable() {
        let Ok(operators) = cuda_linear::CudaOperators::new() else {
            return;
        };
        let operators = Rc::new(operators);
        let pages = [KvPageId(30), KvPageId(31), KvPageId(32)];
        let sources = [source_state()];
        let batch = packed_batch(&sources[0], &pages);
        let mut working = [working_state(&pages)];
        let recurrent = operators
            .create_compressor_recurrent_state(128, 2, 2, false)
            .unwrap();
        working[0].kv_state_mut().layers[0]
            .kv
            .window
            .cuda
            .main_compressor_recurrent = Some(recurrent);
        let recurrent = working[0].kv_state().layers[0]
            .kv
            .window
            .cuda
            .main_compressor_recurrent
            .as_ref()
            .unwrap();
        let mut slab = operators
            .create_compressor_recurrent_checkpoint_slab(recurrent, 4)
            .unwrap();
        for slot in 0..4 {
            operators
                .capture_compressor_recurrent_checkpoint(recurrent, &mut slab, slot)
                .unwrap();
        }

        let layout = MlaKvLayout::new([compressed_config()], 2).unwrap();
        let mut pool = MlaPhysicalPool::<TestAttachment>::new(Rc::clone(&operators), layout);
        PhysicalKvPool::configure_capacity(&mut pool, 3).unwrap();
        let transaction_id = ExecutionTransactionId::new(7).unwrap();
        let mut transaction = PhysicalKvPool::prepare(
            &mut pool,
            DecoderKvPrepare {
                transaction: transaction_id,
                sequences: &[],
                new_pages: &pages,
                writable_pages: &[],
                cow_replacements: &[],
                protected_pages: &pages,
                capacity: Default::default(),
                page_statuses: &[],
            },
        )
        .unwrap();
        *transaction.provisional.borrow_mut() = Some(checkpoints(Some(slab)));

        let before_window = working[0].kv_state().layers[0].kv.window.len;
        let before_pages = working[0]
            .kv_state()
            .paged_kv_binding
            .as_ref()
            .unwrap()
            .logical_pages
            .clone();
        let before_recurrent = operators
            .download_f32_buffer(
                working[0].kv_state().layers[0]
                    .kv
                    .window
                    .cuda
                    .main_compressor_recurrent
                    .as_ref()
                    .unwrap()
                    .kv_state(),
            )
            .unwrap();
        operators.failpoints().arm_allocation();

        PhysicalKvPool::retain_provisional(
            &mut pool,
            &mut transaction,
            &sources,
            &mut working,
            &batch,
            &[5],
            &[1],
        )
        .expect_err("injected recurrent clone allocation must fail");
        assert_eq!(transaction.reservations.len(), 3);
        assert!(transaction.provisional.borrow().is_some());
        assert_eq!(
            pool.planes
                .borrow()
                .page_pool
                .as_ref()
                .unwrap()
                .stats()
                .pending_pages,
            3
        );
        assert_eq!(working[0].kv_state().layers[0].kv.window.len, before_window);
        assert_eq!(
            working[0]
                .kv_state()
                .paged_kv_binding
                .as_ref()
                .unwrap()
                .logical_pages,
            before_pages
        );
        assert_eq!(
            operators
                .download_f32_buffer(
                    working[0].kv_state().layers[0]
                        .kv
                        .window
                        .cuda
                        .main_compressor_recurrent
                        .as_ref()
                        .unwrap()
                        .kv_state(),
                )
                .unwrap(),
            before_recurrent
        );

        let retained = PhysicalKvPool::retain_provisional(
            &mut pool,
            &mut transaction,
            &sources,
            &mut working,
            &batch,
            &[5],
            &[1],
        )
        .unwrap();
        assert_eq!(retained, BTreeSet::from([KvPageId(30)]));
        assert_eq!(transaction.reservations.len(), 1);
        assert!(transaction.provisional.borrow().is_none());
        assert_eq!(
            working[0]
                .kv_state()
                .paged_kv_binding
                .as_ref()
                .unwrap()
                .logical_pages,
            [KvPageId(30)]
        );
        assert_eq!(
            pool.planes
                .borrow()
                .page_pool
                .as_ref()
                .unwrap()
                .stats()
                .pending_pages,
            1
        );

        let mut custody = Some(transaction);
        assert_eq!(
            PhysicalKvPool::abort(&mut pool, &mut custody).unwrap(),
            KvEndProgress::Complete
        );
        assert!(custody.is_none());
    }
}
