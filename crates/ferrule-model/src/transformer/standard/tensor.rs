//! Dense tensor parallelism for the shared standard forward.
//!
//! Hidden/residual rows, norms and embedding are replicated. Q/K/V partition
//! whole heads; O and SwiGLU down reduce once before the residual. The vocabulary
//! head is column-sharded (including tied heads) and gathered in vocabulary order.
//! F32 execution is explicit, including when checkpoint storage is BF16.

use crate::decoder::StandardGqaPlanes;
use crate::support::TensorRole;
use crate::transformer::parallel::{
    TensorParallelCollective, TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
    TensorParallelSwiGluPlan,
};
use crate::transformer::{Attention, DecoderModelSpec, FeedForward, LayerSegmentPlan};
use ferrule_common::execution::{ExecutionTransactionId, KvElementType};
use ferrule_common::{Error, ParallelRankId, Result};

/// Local rank is the index, owner is a globally unique KV/collective identity,
/// device is a CUDA ordinal. These namespaces are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StandardTensorPlacement {
    pub owner: ParallelRankId,
    pub device: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StandardTensorPlan {
    spec: DecoderModelSpec,
    placements: Vec<StandardTensorPlacement>,
    kv_heads: usize,
    head_dim: usize,
}

impl StandardTensorPlan {
    /// Preflight without CUDA or weight reads. No KV replication, head padding,
    /// bias, MoE TP, quantized execution or shared devices are supported. PP
    /// ownership is validated separately against each layer segment.
    pub fn new(spec: &DecoderModelSpec, placements: Vec<StandardTensorPlacement>) -> Result<Self> {
        let degree = placements.len();
        if !matches!(degree, 1 | 2 | 4) {
            return Err(error("supported TP degrees are 1, 2 and 4"));
        }
        for (i, placement) in placements.iter().enumerate() {
            if placements[..i]
                .iter()
                .any(|p| p.owner == placement.owner || p.device == placement.device)
            {
                return Err(error("TP requires distinct KV owner IDs and CUDA devices"));
            }
        }
        super::validate_standard_descriptors(spec, spec.layers())?;
        let mut geometry = None;
        for layer in spec.layers() {
            let Attention::Gqa(gqa) = layer.attention() else {
                return Err(error("TP requires GQA"));
            };
            if !gqa.num_heads().is_multiple_of(degree) || !gqa.num_kv_heads().is_multiple_of(degree)
            {
                return Err(error(
                    "Q and KV heads must be divisible by TP; KV replication is unsupported",
                ));
            }
            let current = (gqa.num_kv_heads() / degree, gqa.head_dim());
            if geometry.is_some_and(|previous| previous != current) {
                return Err(error("TP requires uniform KV geometry"));
            }
            geometry = Some(current);
            let FeedForward::SwiGlu(mlp) = layer.feed_forward() else {
                return Err(error(
                    "MoE TP is unsupported; no replicated or local fallback",
                ));
            };
            TensorParallelSwiGluPlan::new(spec.hidden_size(), mlp.gate().out_features(), degree)?;
            for linear in [
                gqa.query(),
                gqa.key(),
                gqa.value(),
                gqa.output(),
                mlp.gate(),
                mlp.up(),
                mlp.down(),
            ] {
                if linear.has_bias() {
                    return Err(error("biased TP projections are unsupported"));
                }
                if linear
                    .in_features()
                    .checked_mul(linear.out_features())
                    .is_none_or(|n| n > i32::MAX as usize)
                {
                    return Err(error("TP projection exceeds CUDA i32 indexing"));
                }
            }
        }
        TensorParallelLinearPlan::new(
            spec.vocab_size(),
            spec.hidden_size(),
            degree,
            Partition::Column,
        )?;
        let (kv_heads, head_dim) = geometry.ok_or_else(|| error("empty decoder"))?;
        Ok(Self {
            spec: spec.clone(),
            placements,
            kv_heads,
            head_dim,
        })
    }
    pub fn ranks(&self) -> usize {
        self.placements.len()
    }

    /// Number of physical KV heads owned by each rank. KV is never replicated.
    pub const fn local_kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn global_kv_heads(&self) -> usize {
        self.kv_heads * self.ranks()
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn placements(&self) -> &[StandardTensorPlacement] {
        &self.placements
    }
    pub fn placement(&self, rank: ParallelRankId) -> Result<StandardTensorPlacement> {
        self.placements
            .get(rank.get() as usize)
            .copied()
            .ok_or_else(|| error("TP local rank out of range"))
    }
    pub fn validate_segment(
        &self,
        spec: &DecoderModelSpec,
        segment: &LayerSegmentPlan,
    ) -> Result<()> {
        if spec != &self.spec || segment.total_layers() != spec.layers().len() {
            return Err(error(
                "TP plan image/segment total layer count does not match decoder",
            ));
        }
        let layers = segment.layers();
        if layers.start >= layers.end || layers.end > spec.layers().len() {
            return Err(error("TP segment layer range is outside the decoder image"));
        }
        if segment.owns_embedding() && layers.start != 0 {
            return Err(error(
                "TP embedding ownership must match the segment's first layer",
            ));
        }
        if segment.owns_output() && layers.end != spec.layers().len() {
            return Err(error(
                "TP output ownership must match the segment's last layer",
            ));
        }
        Ok(())
    }

    /// Full decoder F32 K/V schema, retained for PP1 callers, with
    /// `local_kv_heads * head_dim` elements per token/layer.
    /// Each rank must construct a separate owner-local pool from this schema;
    /// logical page IDs may agree across ranks, physical slots must not be shared.
    pub fn kv_planes(&self, page_tokens: usize, max_positions: usize) -> Result<StandardGqaPlanes> {
        StandardGqaPlanes::new(
            self.spec.layers().len(),
            self.kv_heads,
            self.head_dim,
            page_tokens,
            max_positions,
            KvElementType::F32,
        )
    }

    /// Segment-local F32 K/V schema. The physical layer index passed to CUDA KV
    /// is local to this segment; the caller owns one schema/pool per PP×TP rank.
    pub fn kv_planes_for_segment(
        &self,
        segment: &LayerSegmentPlan,
        page_tokens: usize,
        max_positions: usize,
    ) -> Result<StandardGqaPlanes> {
        self.validate_segment(&self.spec, segment)?;
        StandardGqaPlanes::new(
            segment.layer_count(),
            self.kv_heads,
            self.head_dim,
            page_tokens,
            max_positions,
            KvElementType::F32,
        )
    }

    /// Validate the storage contract before reading any rank's weights.
    pub fn validate_resources(
        &self,
        resources: &crate::transformer::BoundDecoderResources,
    ) -> Result<()> {
        if resources.spec() != &self.spec {
            return Err(error("TP resources do not match the planned decoder"));
        }
        for binding in resources.state_dict().parameters() {
            self.validate_storage(binding)?;
            match binding.role() {
                TensorRole::TokenEmbedding
                | TensorRole::OutputNorm
                | TensorRole::AttentionNorm
                | TensorRole::FeedForwardNorm
                | TensorRole::AttentionQueryNorm
                | TensorRole::AttentionKeyNorm => {}
                role => {
                    self.parameter_plan(binding, role)?;
                }
            }
        }
        Ok(())
    }

    /// Metadata-only TP read preflight. Reuse the standard model's role/partition
    /// plans and the physical reader's rectangle validator; do not estimate local
    /// bytes as global bytes / TP (ragged partitions and tied heads differ).
    /// This does not replace source checks performed by the actual bounded reader.
    pub fn validate_read_limits(
        &self,
        resources: &crate::transformer::BoundDecoderResources,
        max_tensor_bytes: u64,
    ) -> Result<()> {
        use crate::checkpoint::CheckpointTensorReader;

        if max_tensor_bytes == 0 {
            return Err(error("TP read limit must be positive"));
        }
        self.validate_resources(resources)?;
        if !resources.state_dict().validate_source_identities() {
            return Err(error(
                "stale checkpoint source identity during TP preflight",
            ));
        }
        let reader = CheckpointTensorReader::new(max_tensor_bytes);
        for binding in resources.state_dict().parameters() {
            let weight = binding.weight();
            match binding.role() {
                TensorRole::TokenEmbedding
                | TensorRole::OutputNorm
                | TensorRole::AttentionNorm
                | TensorRole::FeedForwardNorm
                | TensorRole::AttentionQueryNorm
                | TensorRole::AttentionKeyNorm => {
                    if weight.slice().bytes > max_tensor_bytes {
                        return Err(error(&format!(
                            "replicated parameter '{}' is {} bytes, above its {}-byte TP read limit",
                            binding.path(),
                            weight.slice().bytes,
                            max_tensor_bytes,
                        )));
                    }
                }
                role => {
                    let plan = self.parameter_plan(binding, role)?;
                    for rank in 0..self.ranks() {
                        let range = plan.rank_range(ParallelRankId::new(rank as u32))?;
                        let (rows, columns) = match plan.partition() {
                            Partition::Column => (range, 0..plan.in_features()),
                            Partition::Row => (0..plan.out_features(), range),
                        };
                        reader.plan_2d_range(weight.slice(), rows, columns, weight.source_identity())
                            .map_err(|e| Error::context(format!(
                                "TP parameter '{}' rank {rank} read limit {max_tensor_bytes} bytes", binding.path(),
                            ), e))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_storage(&self, binding: &crate::transformer::BoundParameter) -> Result<()> {
        use crate::checkpoint::CheckpointDType;
        use crate::transformer::TensorTransform;
        let weight = binding.weight();
        if binding.scale().is_some()
            || !matches!(
                weight.slice().dtype,
                CheckpointDType::F32 | CheckpointDType::Bf16
            )
            || weight.transform() != &TensorTransform::Identity
            || weight.logical_shape() != weight.slice().shape
            || weight.physical_shape() != weight.logical_shape()
        {
            return Err(error(
                "TP requires identity, unscaled dense F32/BF16 checkpoint storage",
            ));
        }
        if weight.slice().element_count()? > i32::MAX as usize {
            return Err(error("TP parameter exceeds CUDA i32 indexing"));
        }
        Ok(())
    }

    pub(crate) fn parameter_plan(
        &self,
        binding: &crate::transformer::BoundParameter,
        role: &TensorRole,
    ) -> Result<TensorParallelLinearPlan> {
        self.validate_storage(binding)?;
        if binding.role() != role {
            return Err(error("TP parameter role does not match its binding"));
        }
        let [out, width] = binding.weight().logical_shape() else {
            return Err(error("TP linear must have a global 2D shape"));
        };
        self.shape_plan(role, *out, *width)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linear_plan(
        &self,
        linear: &crate::transformer::PreparedLinear,
    ) -> Result<TensorParallelLinearPlan> {
        if linear.bias().is_some() {
            return Err(error("TP linear bias is unsupported"));
        }
        self.shape_plan(linear.role(), linear.out_features(), linear.in_features())
    }

    fn shape_plan(
        &self,
        role: &TensorRole,
        out: usize,
        width: usize,
    ) -> Result<TensorParallelLinearPlan> {
        match role {
            TensorRole::DenseMlpGate | TensorRole::DenseMlpUp => {
                return TensorParallelSwiGluPlan::new(width, out, self.ranks())
                    .map(|p| p.gate_up_plan().clone());
            }
            TensorRole::DenseMlpDown => {
                return TensorParallelSwiGluPlan::new(out, width, self.ranks())
                    .map(|p| p.down_plan().clone());
            }
            _ => {}
        }
        let partition = match role {
            TensorRole::AttentionQuery
            | TensorRole::AttentionKey
            | TensorRole::AttentionValue
            | TensorRole::OutputHead => Partition::Column,
            TensorRole::AttentionOutput => Partition::Row,
            _ => return Err(error("unsupported TP linear role")),
        };
        TensorParallelLinearPlan::new(out, width, self.ranks(), partition)
    }
}

/// Host-only communication seam. Implementations must bound waits and storage,
/// match transaction/site/sequence/shape across ranks, and wake peers on failure.
/// A returned host result is not a GPU fence or a KV commit/publication receipt.
pub trait StandardTensorCollective {
    fn owner(&self) -> ParallelRankId;
    fn members(&self) -> &[ParallelRankId];
    fn exchange(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: TensorParallelCollective,
        values: Vec<f32>,
    ) -> Result<Vec<f32>>;
    /// Fail closed: no subsequent collective may reuse this lifetime.
    fn abort(&mut self);
}

fn error(message: &str) -> Error {
    Error::Model {
        message: format!("standard decoder TP: {message}"),
    }
}
