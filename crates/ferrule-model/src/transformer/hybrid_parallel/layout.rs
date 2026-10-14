//! Owner-local metadata, shared by preparation, state, KV and communication.
use super::*;
use crate::TensorRole;
use crate::decoder::HybridLayerSchema;
use crate::nn::ParameterResidency;
use crate::transformer::BoundParameter;
use crate::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridParameterPartition {
    pub rows: Vec<Range<usize>>,
    pub columns: Range<usize>,
    pub collective: Option<TensorParallelLinearPlan>,
}
impl HybridParameterPartition {
    pub fn local_shape(&self) -> Result<[usize; 2]> {
        fn width(range: &Range<usize>) -> Result<usize> {
            range
                .end
                .checked_sub(range.start)
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("empty or reversed parameter range"))
        }
        if self.rows.is_empty() {
            return Err(invalid("empty parameter row ranges"));
        }
        let rows = self.rows.iter().try_fold(0usize, |sum, range| {
            sum.checked_add(width(range)?)
                .ok_or_else(|| invalid("parameter row count overflow"))
        })?;
        Ok([rows, width(&self.columns)?])
    }
    pub fn bytes_f32(&self) -> Result<usize> {
        let [rows, cols] = self.local_shape()?;
        rows.checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| invalid("local parameter byte/address-space overflow"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridRankLayout {
    pub(super) spec: Arc<DecoderModelSpec>,
    precision: HybridRankPrecision,
    routed_execution: HybridRoutedExpertExecution,
    scope: HybridExpertScope,
    lane: usize,
    owner: ParallelRankId,
    members: Vec<ParallelRankId>,
    schema: HybridStateSchema,
    attention: Vec<HybridLocalAttention>,
    mlps: Vec<Option<TensorParallelSwiGluPlan>>,
}
impl HybridRankLayout {
    pub(super) fn new(plan: &HybridParallelPlan, lane: usize) -> Result<Self> {
        let owner = *plan
            .tensor_members()
            .get(lane)
            .ok_or_else(|| invalid("invalid TP lane"))?;
        let attention = plan
            .layers()
            .iter()
            .map(|l| l.attention[lane].clone())
            .collect::<Vec<_>>();
        let layers = plan
            .schema()
            .layers()
            .iter()
            .zip(&attention)
            .map(|(global, local)| match (global, local) {
                (
                    HybridLayerSchema::FullAttention { kv_layer, .. },
                    HybridLocalAttention::Gqa {
                        kv_heads, head_dim, ..
                    },
                ) => HybridLayerSchema::FullAttention {
                    kv_layer: *kv_layer,
                    kv_heads: kv_heads.len(),
                    head_dim: *head_dim,
                },
                (_, HybridLocalAttention::GatedDelta { shape, .. }) => {
                    HybridLayerSchema::GatedDeltaNet(*shape)
                }
                _ => unreachable!("validated attention schema"),
            })
            .collect();
        Ok(Self {
            spec: Arc::clone(&plan.spec),
            precision: plan.precision(),
            routed_execution: plan.routed_execution().clone(),
            scope: plan.scope(),
            lane,
            owner,
            members: plan.tensor_members().to_vec(),
            schema: plan.schema().with_local_layers(layers)?,
            attention,
            mlps: plan
                .layers()
                .iter()
                .map(|l| l.shared_expert.clone().or_else(|| l.dense_mlp.clone()))
                .collect(),
        })
    }
    pub fn precision(&self) -> HybridRankPrecision {
        self.precision
    }
    pub fn routed_execution(&self) -> &HybridRoutedExpertExecution {
        &self.routed_execution
    }
    pub fn scope(&self) -> HybridExpertScope {
        self.scope
    }
    pub fn lane(&self) -> usize {
        self.lane
    }
    pub fn owner(&self) -> ParallelRankId {
        self.owner
    }
    pub fn members(&self) -> &[ParallelRankId] {
        &self.members
    }
    pub fn state_schema(&self) -> &HybridStateSchema {
        &self.schema
    }
    pub fn state_bytes(&self) -> Result<usize> {
        Ok(self.state_budget(1)?.per_sequence_bytes)
    }
    pub fn kv_planes(
        &self,
        page_size: usize,
        max_positions: usize,
    ) -> Result<crate::decoder::StandardGqaPlanes> {
        self.schema
            .kv_planes_with_dtype(page_size, max_positions, self.precision.kv_dtype())
    }
    pub fn attention(&self, layer: usize) -> Result<&HybridLocalAttention> {
        let entry = self
            .schema
            .entry(layer)
            .ok_or_else(|| invalid("layer outside rank layout"))?;
        Ok(&self.attention[entry.local_layer])
    }
    pub fn parameter_partition(
        &self,
        binding: &BoundParameter,
    ) -> Result<HybridParameterPartition> {
        let shape = binding.weight().logical_shape();
        let role = binding.role();
        let layer = match binding.residency() {
            ParameterResidency::Layer { layer } => Some(*layer),
            ParameterResidency::Static => None,
            ParameterResidency::Expert { layer, expert }
                if self.routed_execution.is_local_tensor_sharded() =>
            {
                let Some(FeedForward::Moe(moe)) =
                    self.spec.layers().get(*layer).map(|l| l.feed_forward())
                else {
                    return Err(invalid("routed parameter belongs to a non-MoE layer"));
                };
                if *expert >= moe.router_spec().num_experts()
                    || !matches!(
                        role,
                        TensorRole::RoutedExpertGate
                            | TensorRole::RoutedExpertUp
                            | TensorRole::RoutedExpertDown
                    )
                {
                    return Err(invalid("routed parameter expert or role outside layer"));
                }
                Some(*layer)
            }
            _ => {
                return Err(invalid(
                    "rank image cannot materialize expert/attachment payload",
                ));
            }
        };
        let geometry = layer.map(|l| self.attention(l)).transpose()?;
        // Conv's physical [channels,1,kernel] and the head vectors are contiguous
        // matrices after flattening; their catalog byte extent is unchanged.
        let (out, width) = match (role, shape) {
            (TensorRole::LinearAttentionConv, [channels, 1, kernel]) => (*channels, *kernel),
            (TensorRole::LinearAttentionALog | TensorRole::LinearAttentionDtBias, [heads]) => {
                (*heads, 1)
            }
            (_, [out, width]) => (*out, *width),
            _ => return Err(invalid("rank parameter requires a matrix or GDN vector")),
        };
        let mut p = HybridParameterPartition {
            rows: std::iter::once(0..out).collect(),
            columns: 0..width,
            collective: None,
        };
        let rank = ParallelRankId::new(self.lane as u32);
        match (geometry, role) {
            (
                Some(HybridLocalAttention::Gqa {
                    query_heads,
                    head_dim,
                    ..
                }),
                TensorRole::AttentionQuery,
            ) => {
                let factor = out / (query_heads.len() * self.members.len() * head_dim);
                p.rows = std::iter::once(
                    query_heads.start * head_dim * factor..query_heads.end * head_dim * factor,
                )
                .collect();
            }
            (
                Some(HybridLocalAttention::Gqa {
                    kv_heads, head_dim, ..
                }),
                TensorRole::AttentionKey | TensorRole::AttentionValue,
            ) => {
                p.rows =
                    std::iter::once(kv_heads.start * head_dim..kv_heads.end * head_dim).collect();
            }
            (
                Some(HybridLocalAttention::GatedDelta { qkv_channels, .. }),
                TensorRole::LinearAttentionQkv | TensorRole::LinearAttentionConv,
            ) => p.rows = qkv_channels.to_vec(),
            (
                Some(HybridLocalAttention::GatedDelta {
                    value_heads, shape, ..
                }),
                TensorRole::LinearAttentionZ,
            ) => {
                p.rows = std::iter::once(
                    value_heads.start * shape.value_dim..value_heads.end * shape.value_dim,
                )
                .collect()
            }
            (
                Some(HybridLocalAttention::GatedDelta { value_heads, .. }),
                TensorRole::LinearAttentionA
                | TensorRole::LinearAttentionBeta
                | TensorRole::LinearAttentionALog
                | TensorRole::LinearAttentionDtBias,
            ) => p.rows = vec![value_heads.clone()],
            (_, TensorRole::AttentionOutput | TensorRole::OutputHead) => {
                let plan = TensorParallelLinearPlan::new(
                    out,
                    width,
                    self.members.len(),
                    if *role == TensorRole::OutputHead {
                        Partition::Column
                    } else {
                        Partition::Row
                    },
                )?;
                let (rows, columns) = plan.matrix_range(rank)?;
                p.rows = vec![rows];
                p.columns = columns;
                p.collective = Some(plan);
            }
            (
                _,
                TensorRole::DenseMlpGate
                | TensorRole::DenseMlpUp
                | TensorRole::DenseMlpDown
                | TensorRole::SharedExpertGate
                | TensorRole::SharedExpertUp
                | TensorRole::SharedExpertDown,
            ) => {
                let entry = self
                    .schema
                    .entry(layer.ok_or_else(|| invalid("MLP must be layer-local"))?)
                    .ok_or_else(|| invalid("MLP outside stage"))?;
                let mlp = self.mlps[entry.local_layer]
                    .as_ref()
                    .ok_or_else(|| invalid("missing MLP partition"))?;
                let down = matches!(
                    role,
                    TensorRole::DenseMlpDown | TensorRole::SharedExpertDown
                );
                let plan = if down {
                    mlp.down_plan()
                } else {
                    mlp.gate_up_plan()
                };
                if (plan.out_features(), plan.in_features()) != (out, width) {
                    return Err(invalid("MLP binding differs from plan"));
                }
                let (rows, columns) = plan.matrix_range(rank)?;
                p.rows = vec![rows];
                p.columns = columns;
                if down {
                    p.collective = Some(plan.clone());
                }
            }
            (
                _,
                TensorRole::RoutedExpertGate
                | TensorRole::RoutedExpertUp
                | TensorRole::RoutedExpertDown,
            ) if matches!(binding.residency(), ParameterResidency::Expert { .. }) => {
                let FeedForward::Moe(moe) = self.spec.layers()[layer.unwrap()].feed_forward()
                else {
                    return Err(invalid("routed parameter requires MoE"));
                };
                let mlp = TensorParallelSwiGluPlan::new(
                    moe.expert().hidden_size(),
                    moe.expert().intermediate_size(),
                    self.members.len(),
                )?;
                let down = *role == TensorRole::RoutedExpertDown;
                let plan = if down {
                    mlp.down_plan()
                } else {
                    mlp.gate_up_plan()
                };
                if (plan.out_features(), plan.in_features()) != (out, width) {
                    return Err(invalid("routed expert binding differs from TP plan"));
                }
                let (rows, columns) = plan.matrix_range(rank)?;
                p.rows = vec![rows];
                p.columns = columns;
                if down {
                    p.collective = Some(plan.clone());
                }
            }
            (
                _,
                TensorRole::TokenEmbedding
                | TensorRole::RouterLogits
                | TensorRole::SharedExpertOutputGate,
            ) => {}
            _ => return Err(invalid("unsupported rank parameter role")),
        }
        p.bytes_f32()?;
        Ok(p)
    }
}

/// Immutable owner-local payload accounting, not allocator admission. Expert
/// payloads belong to their external owners; workspace/KV/state are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridRankWeightBudget {
    pub matrix_bytes: usize,
    pub norm_bytes: usize,
    pub resident_bytes: usize,
    /// Largest aggregate local parameter, not the size of an individual Q/K/V range.
    pub max_parameter_bytes: usize,
}
impl HybridRankLayout {
    pub fn weight_budget(
        &self,
        resources: &crate::transformer::BoundDecoderResources,
        segment: &crate::transformer::LayerSegmentPlan,
    ) -> Result<HybridRankWeightBudget> {
        use crate::checkpoint::CheckpointDType;
        use crate::transformer::TensorTransform;
        if self.precision != HybridRankPrecision::F32
            || self.routed_execution.is_local_tensor_sharded()
        {
            return Err(invalid(
                "CPU rank image supports F32 with external experts only",
            ));
        }
        if resources.spec() != self.spec.as_ref() || segment.layers() != self.schema.layer_range() {
            return Err(invalid("weight budget segment differs from layout"));
        }
        let inventory =
            crate::transformer::segment_inventory::HybridSegmentInventory::new(resources, segment)?;
        let mut budget = HybridRankWeightBudget {
            matrix_bytes: 0,
            norm_bytes: 0,
            resident_bytes: 0,
            max_parameter_bytes: 0,
        };
        for binding in inventory.parameters() {
            if matches!(binding.residency(), ParameterResidency::Expert { .. }) {
                continue;
            }
            if binding.scale().is_some()
                || binding.weight().slice().dtype != CheckpointDType::F32
                || binding.weight().transform() != &TensorTransform::Identity
            {
                return Err(invalid(
                    "local CPU image requires identity unscaled F32 storage",
                ));
            }
            let norm = matches!(
                binding.role(),
                TensorRole::OutputNorm
                    | TensorRole::AttentionNorm
                    | TensorRole::FeedForwardNorm
                    | TensorRole::AttentionQueryNorm
                    | TensorRole::AttentionKeyNorm
                    | TensorRole::LinearAttentionNorm
            );
            let bytes = if norm {
                binding
                    .weight()
                    .slice()
                    .element_count()?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("norm size overflow"))?
            } else {
                self.parameter_partition(binding)?.bytes_f32()?
            };
            budget.max_parameter_bytes = budget.max_parameter_bytes.max(bytes);
            if norm {
                // Existing PreparedNorm retains its source parameter and decoded values.
                budget.norm_bytes = budget
                    .norm_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("norm budget overflow"))?;
            } else {
                budget.matrix_bytes = budget
                    .matrix_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("matrix budget overflow"))?;
            }
        }
        budget.resident_bytes = budget
            .norm_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(budget.matrix_bytes))
            .ok_or_else(|| invalid("resident weight budget overflow"))?;
        Ok(budget)
    }
}
