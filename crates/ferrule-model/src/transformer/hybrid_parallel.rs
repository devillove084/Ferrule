//! Hybrid PP/TP geometry with an explicit, non-multiplicative EP attachment.
//!
//! Payload-free plans are available in production builds. Execution is only a
//! CPU-reference seam: `HybridTensorCpuSegment` executes local attention/state
//! with caller-supplied collectives; it owns no KV publication or GPU execution.
//! TP attention/shared outputs reduce before residuals, and EP dispatch occurs
//! once on the explicitly attached source rank. Production runtime admission remains explicitly gated.

#![cfg_attr(not(feature = "cpu-reference"), doc = include_str!("../../tests/compile/hybrid_parallel.md"))]

#[cfg(feature = "cpu-reference")]
mod tensor_reference;
#[cfg(feature = "cpu-reference")]
mod tensor_state;
#[cfg(feature = "cpu-reference")]
pub use tensor_reference::{HybridExpertCallBinding, HybridTensorCpuSegment};
#[cfg(feature = "cpu-reference")]
pub use tensor_state::{
    HybridTensorLayerStates, HybridTensorSequenceLifecycle, HybridTensorSequenceState,
};

use std::{collections::BTreeSet, ops::Range, sync::Arc};

use ferrule_backend::cpu::gated_delta::GatedDeltaShape;
use ferrule_common::{Error, ParallelRankId, ParallelTopologyId, Result};

use super::expert_parallel::ExpertPlacement;
use super::parallel::TensorParallelSwiGluPlan;
mod layout;
mod mixed_block;
pub(crate) mod mixed_decoded;
pub(crate) mod mixed_decoded_arena;
pub use mixed_decoded::HybridMixedDecodedCacheBudget;
mod mixed_estimate;
mod mixed_layer;
mod mixed_routes;
pub use mixed_estimate::HybridMixedRankOperationEstimate;
pub use mixed_layer::{
    DataLocation, DevicePayloadIdentity, DevicePayloadPermit, HybridBf16Collective,
    HybridBf16CollectiveBinding, HybridMixedGqaBlockPlan, HybridMixedRankProgramPlan,
};
pub use mixed_routes::{HybridMixedRouteAgreement, HybridMixedRouteSite};
#[cfg(any(feature = "cuda", test))]
pub(crate) mod mixed_channel;
mod precision;
pub use mixed_block::{HybridMixedBlockPlan, HybridMixedProjectionArithmetic};
mod schedule;
use super::{Attention, DecoderModelSpec, FeedForward};
use crate::{decoder::HybridStateSchema, moe::ExpertId};
pub use layout::{HybridParameterPartition, HybridRankLayout, HybridRankWeightBudget};
pub use precision::{
    HybridCollectivePrecision, HybridMoeCombineOrder, HybridRankParameterBudget,
    HybridRankPrecision, HybridRankStateBudget, HybridRankStorageBudget, HybridRoutedAccumulation,
    HybridRoutedExpertExecution,
};
pub use schedule::{
    HYBRID_AGREEMENT_VERSION, HybridCollectiveKind, HybridCollectiveSchedule,
    HybridCollectiveSemantic, HybridCollectiveShape, HybridCollectiveSite,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridExpertScope {
    pub topology: ParallelTopologyId,
    pub replica: u32,
    pub stage: u32,
}

/// Explicit routed-expert attachment. EP is not a mesh dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridExpertAttachment {
    pub source_rank: ParallelRankId,
    pub members: Vec<ParallelRankId>,
    pub placement: ExpertPlacement,
}

/// Global expert coordinates are meaningful only within their model/PP/DP scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopedHybridExpert {
    pub scope: HybridExpertScope,
    pub expert: ExpertId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridLocalAttention {
    Gqa {
        query_heads: Range<usize>,
        kv_heads: Range<usize>,
        /// Canonical TP owner for each local KV head. Other lanes retain replicas;
        /// this is not permission for concurrent writes to a shared physical page.
        kv_owners: Vec<ParallelRankId>,
        head_dim: usize,
    },
    GatedDelta {
        key_heads: Range<usize>,
        value_heads: Range<usize>,
        /// Packed local Q, K, V channel ranges in the global QKV projection.
        qkv_channels: [Range<usize>; 3],
        shape: GatedDeltaShape,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridParallelLayer {
    pub global_layer: usize,
    pub attention: Vec<HybridLocalAttention>,
    /// Shared experts are TP SwiGLU shards, NOT routed EP experts. Reduce their
    /// down-projection partials once; apply any scalar shared gate after reduction.
    pub shared_expert: Option<TensorParallelSwiGluPlan>,

    pub shared_output_gate: bool,
    pub dense_mlp: Option<TensorParallelSwiGluPlan>,
    routed_experts: usize,
    top_k: usize,
}

impl HybridParallelLayer {
    pub fn routed_experts(&self) -> usize {
        self.routed_experts
    }
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridParallelPlan {
    spec: Arc<DecoderModelSpec>,
    scope: HybridExpertScope,
    schema: HybridStateSchema,
    tensor_members: Vec<ParallelRankId>,
    routed_execution: HybridRoutedExpertExecution,
    precision: HybridRankPrecision,
    layers: Vec<HybridParallelLayer>,
    width: usize,
}

fn invalid(message: &str) -> Error {
    Error::Model {
        message: format!("hybrid parallel: {message}"),
    }
}

impl HybridParallelPlan {
    pub fn new(
        spec: &DecoderModelSpec,
        range: Range<usize>,
        scope: HybridExpertScope,
        tensor_members: Vec<ParallelRankId>,
        attachment: Option<HybridExpertAttachment>,
    ) -> Result<Self> {
        Self::new_with_execution(
            spec,
            range,
            scope,
            tensor_members,
            attachment.map_or(
                HybridRoutedExpertExecution::None,
                HybridRoutedExpertExecution::ExternalExpertAttachment,
            ),
            HybridRankPrecision::F32,
        )
    }

    /// Build rank geometry with explicit routed-expert ownership and numeric
    /// boundaries. This is an admission description, not runtime enablement.
    pub fn new_with_execution(
        spec: &DecoderModelSpec,
        range: Range<usize>,
        scope: HybridExpertScope,
        tensor_members: Vec<ParallelRankId>,
        routed_execution: HybridRoutedExpertExecution,
        precision: HybridRankPrecision,
    ) -> Result<Self> {
        let attachment = routed_execution.attachment();
        let tp = tensor_members.len();
        if tp == 0
            || u32::try_from(tp).is_err()
            || tensor_members.iter().collect::<BTreeSet<_>>().len() != tp
        {
            return Err(invalid("nonempty unique TP members required"));
        }
        if let Some(ep) = &attachment
            && (ep.members.is_empty()
                || !tensor_members.contains(&ep.source_rank)
                || ep.members.iter().collect::<BTreeSet<_>>().len() != ep.members.len())
        {
            return Err(invalid(
                "EP requires unique owners and an attached TP source",
            ));
        }
        let schema = HybridStateSchema::for_segment(spec, range.clone())?;
        let mut layers = Vec::new();
        for layer in &spec.layers()[range] {
            let attention = (0..tp).map(|lane| match layer.attention() {
                Attention::Gqa(gqa) => {
                    let (q, kv) = (gqa.num_heads(), gqa.num_kv_heads());
                    if !q.is_multiple_of(tp)
                        || !(kv.is_multiple_of(tp) || tp.is_multiple_of(kv))
                    {
                        return Err(invalid("GQA requires Q divisible by TP and aligned KV sharding/replication"));
                    }
                    let query_heads = lane * (q / tp)..(lane + 1) * (q / tp);
                    let kv_heads = query_heads.start / (q / kv)..(query_heads.end - 1) / (q / kv) + 1;
                    let kv_owners = kv_heads.clone().map(|h| {
                        let owner = if kv >= tp { h / (kv / tp) } else { h * (tp / kv) };
                        tensor_members[owner]
                    }).collect();
                    Ok(HybridLocalAttention::Gqa { query_heads, kv_heads, kv_owners, head_dim: gqa.head_dim() })
                }
                Attention::GatedDeltaNet(gdn) => {
                    let (kh, vh) = (gdn.num_key_heads(), gdn.num_value_heads());
                    // Replicating recurrent state is not implemented. Keep each
                    // key group and its value heads together, never ragged-split it.
                    if !kh.is_multiple_of(tp) || !vh.is_multiple_of(tp) {
                        return Err(invalid("GDN key/value heads must divide TP; replicated recurrent state unsupported"));
                    }
                    let keys = lane * (kh / tp)..(lane + 1) * (kh / tp);
                    let values = lane * (vh / tp)..(lane + 1) * (vh / tp);
                    let shape = GatedDeltaShape {
                        key_heads: kh / tp, value_heads: vh / tp,
                        key_dim: gdn.key_head_dim(), value_dim: gdn.value_head_dim(),
                        kernel: gdn.conv_kernel_dim(),
                    };
                    shape.sizes()?;
                    let kw = kh * shape.key_dim;
                    let q = keys.start * shape.key_dim..keys.end * shape.key_dim;
                    let k = kw + q.start..kw + q.end;
                    let v = 2 * kw + values.start * shape.value_dim..2 * kw + values.end * shape.value_dim;
                    Ok(HybridLocalAttention::GatedDelta { key_heads: keys, value_heads: values, qkv_channels: [q, k, v], shape })
                }
                _ => Err(invalid("only GQA/GDN attention supported")),
            }).collect::<Result<Vec<_>>>()?;
            let (shared_expert, shared_output_gate, routed_experts, top_k) =
                match layer.feed_forward() {
                    FeedForward::Moe(moe) => {
                        if matches!(routed_execution, HybridRoutedExpertExecution::None) {
                            return Err(invalid(
                                "routed layer requires an explicit expert ownership mode",
                            ));
                        }
                        if routed_execution.is_local_tensor_sharded() {
                            TensorParallelSwiGluPlan::new(
                                moe.expert().hidden_size(),
                                moe.expert().intermediate_size(),
                                tp,
                            )?;
                        }
                        if let Some(ep) = attachment {
                            for expert in 0..moe.router_spec().num_experts() {
                                if !ep
                                    .placement
                                    .owner(layer.index(), expert)
                                    .is_some_and(|o| ep.members.contains(&o))
                                {
                                    return Err(invalid(
                                        "every routed expert requires an explicit attached owner",
                                    ));
                                }
                            }
                        }
                        (
                            moe.shared_expert()
                                .map(|s| {
                                    TensorParallelSwiGluPlan::new(
                                        s.hidden_size(),
                                        s.intermediate_size(),
                                        tp,
                                    )
                                })
                                .transpose()?,
                            moe.shared_expert_gate().is_some(),
                            moe.router_spec().num_experts(),
                            moe.router_spec().experts_per_token(),
                        )
                    }
                    FeedForward::SwiGlu(_) => (None, false, 0, 0),
                };
            layers.push(HybridParallelLayer {
                global_layer: layer.index(),
                attention,
                shared_expert,

                shared_output_gate,
                dense_mlp: match layer.feed_forward() {
                    FeedForward::SwiGlu(m) => Some(TensorParallelSwiGluPlan::new(
                        m.hidden_size(),
                        m.intermediate_size(),
                        tp,
                    )?),
                    FeedForward::Moe(_) => None,
                },
                routed_experts,
                top_k,
            });
        }
        if !matches!(routed_execution, HybridRoutedExpertExecution::None)
            && layers.iter().all(|l| l.routed_experts == 0)
        {
            return Err(invalid("dense stage must not attach expert owners"));
        }
        if let Some(ep) = &attachment {
            let expected = ExpertPlacement::new(layers.iter().flat_map(|l| {
                (0..l.routed_experts).map(move |expert| {
                    (
                        l.global_layer,
                        expert,
                        ep.placement
                            .owner(l.global_layer, expert)
                            .expect("validated placement"),
                    )
                })
            }))?;
            if expected != ep.placement {
                return Err(invalid("EP placement contains experts outside this stage"));
            }
        }
        Ok(Self {
            spec: Arc::new(spec.clone()),
            scope,
            schema,
            tensor_members,
            routed_execution,
            precision,
            layers,
            width: spec.hidden_size(),
        })
    }

    pub fn scope(&self) -> HybridExpertScope {
        self.scope
    }
    pub fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }
    pub fn layers(&self) -> &[HybridParallelLayer] {
        &self.layers
    }
    pub fn tensor_members(&self) -> &[ParallelRankId] {
        &self.tensor_members
    }
    pub fn expert_members(&self) -> &[ParallelRankId] {
        self.expert_attachment()
            .map_or(&[], |ep| ep.members.as_slice())
    }
    pub fn expert_attachment(&self) -> Option<&HybridExpertAttachment> {
        self.routed_execution.attachment()
    }
    pub fn routed_execution(&self) -> &HybridRoutedExpertExecution {
        &self.routed_execution
    }
    pub fn precision(&self) -> HybridRankPrecision {
        self.precision
    }

    /// Mixed rank execution is intentionally fail-closed until its CUDA
    /// projection/state/collective seams and frozen TP numerical gate exist.
    pub fn validate_rank_execution(&self) -> Result<()> {
        if self.precision != HybridRankPrecision::F32 {
            return Err(invalid(
                "Mixed BF16 rank execution is not wired through typed projections, state/KV, and collectives",
            ));
        }
        if matches!(
            self.routed_execution,
            HybridRoutedExpertExecution::LocalTensorShards
        ) {
            return Err(invalid(
                "local tensor-sharded routed-expert execution is not wired; EP=1 is not external EP",
            ));
        }
        Ok(())
    }

    /// EP is an explicit owner set, never a multiplier of the TP/PP mesh.
    pub fn physical_members(&self) -> BTreeSet<ParallelRankId> {
        self.tensor_members
            .iter()
            .chain(self.expert_members())
            .copied()
            .collect()
    }

    pub fn expert_owner(&self, expert: ScopedHybridExpert) -> Result<ParallelRankId> {
        if expert.scope != self.scope {
            return Err(invalid("expert scope mismatch"));
        }
        let layer = self
            .layers
            .iter()
            .find(|l| l.global_layer == expert.expert.layer)
            .ok_or_else(|| invalid("expert outside stage"))?;
        if expert.expert.expert >= layer.routed_experts {
            return Err(invalid("expert outside layer"));
        }
        self.expert_attachment()
            .ok_or_else(|| invalid("stage has no external EP attachment"))?
            .placement
            .owner(expert.expert.layer, expert.expert.expert)
            .ok_or_else(|| invalid("expert missing placement"))
    }

    pub fn hidden_size(&self) -> usize {
        self.width
    }

    /// Physical typed recurrent + convolution bytes per sequence on one TP lane.
    /// KV replicas are separately described by `HybridLocalAttention::Gqa`.
    pub fn local_state_bytes(&self, lane: usize) -> Result<usize> {
        self.rank_layout(lane)?.state_bytes()
    }

    pub fn rank_layout(&self, lane: usize) -> Result<HybridRankLayout> {
        HybridRankLayout::new(self, lane)
    }

    /// Check admitted geometry, not just compact KV/state slots. Two descriptors
    /// can share a state schema while differing in query heads or shared-expert partitions.
    pub fn validate_spec(&self, spec: &DecoderModelSpec) -> Result<()> {
        let expected = Self::new_with_execution(
            spec,
            self.schema.layer_range(),
            self.scope,
            self.tensor_members.clone(),
            self.routed_execution.clone(),
            self.precision,
        )?;
        if self != &expected {
            return Err(invalid("model geometry differs from admitted plan"));
        }
        Ok(())
    }
}
