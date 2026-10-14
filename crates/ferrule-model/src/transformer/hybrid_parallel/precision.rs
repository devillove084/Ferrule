//! Native storage contracts, not execution permission or allocator reservations.
use super::{HybridExpertAttachment, HybridRankLayout, invalid};
use crate::TensorRole;
use crate::checkpoint::{CheckpointDType, NumericFp8Encoding};
use crate::decoder::HybridLayerSchema;
use crate::nn::{ParameterId, ParameterResidency};
use crate::transformer::{
    BoundDecoderResources, BoundParameter, LayerSegmentPlan, NumericFp8RankPlan, TensorTransform,
};
use ferrule_common::{
    Result,
    execution::{KvElementType, KvLayoutSchema},
};
use std::sync::Arc;

/// Native rank boundaries, independent of compressed weight operand arithmetic.
/// Existing F32/W8A16 rank execution uses F32; Mixed remains preparation-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HybridRankPrecision {
    #[default]
    F32,
    MixedBf16F32,
}

/// The communicated element type includes the local projection writeback.
/// This is a numerical contract, not a claim of vLLM kernel parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridCollectivePrecision {
    F32,
    /// Each local projection rounds to BF16 BEFORE a BF16 sum collective.
    Bf16PartialsAndReduction,
}

/// Installed vLLM MoERunner has two mutually exclusive combine paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridMoeCombineOrder {
    /// Unreduced routed kernel: BF16(shared + routed), then TP all-reduce.
    LocalBf16AddThenReduce,
    /// Already-reduced routed kernel: reduce shared separately, then BF16 add.
    SeparatelyReducedBf16Add,
}

/// Routed accumulation is independent of TP collective/combine ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridRoutedAccumulation {
    /// Existing model oracle: round product/add to BF16 in ascending expert order.
    /// vLLM fused MoE equivalence has NOT been established for this arithmetic.
    HfEagerBf16ExpertOrder,
}

impl HybridRankPrecision {
    pub const fn is_mixed(self) -> bool {
        matches!(self, Self::MixedBf16F32)
    }
    pub const fn activation_dtype(self) -> CheckpointDType {
        match self {
            Self::F32 => CheckpointDType::F32,
            Self::MixedBf16F32 => CheckpointDType::Bf16,
        }
    }
    pub const fn kv_dtype(self) -> KvElementType {
        match self {
            Self::F32 => KvElementType::F32,
            Self::MixedBf16F32 => KvElementType::Bf16,
        }
    }
    pub const fn conv_dtype(self) -> CheckpointDType {
        self.activation_dtype()
    }
    pub const fn ssm_dtype(self) -> CheckpointDType {
        CheckpointDType::F32
    }
    pub const fn collective(self) -> HybridCollectivePrecision {
        match self {
            Self::F32 => HybridCollectivePrecision::F32,
            Self::MixedBf16F32 => HybridCollectivePrecision::Bf16PartialsAndReduction,
        }
    }
    pub const fn bytes_per_activation(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::MixedBf16F32 => 2,
        }
    }
}

/// Orthogonal to TP: local shards hold a slice of EVERY expert (EP=1), whereas
/// external owners hold complete experts and dispatch once on the source rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridRoutedExpertExecution {
    None,
    ExternalExpertAttachment(HybridExpertAttachment),
    LocalTensorShards,
}
impl HybridRoutedExpertExecution {
    pub fn attachment(&self) -> Option<&HybridExpertAttachment> {
        match self {
            Self::ExternalExpertAttachment(a) => Some(a),
            _ => None,
        }
    }
    pub const fn is_local_tensor_sharded(&self) -> bool {
        matches!(self, Self::LocalTensorShards)
    }
}

/// One binding's independently allocated weight/scale roots, keyed by original
/// parameter identity. This is not a CUDA handle or residency lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridRankParameterBudget {
    pub parameter: ParameterId,
    pub role: TensorRole,
    pub weight_bytes: usize,
    pub scale_bytes: usize,
    pub source_bytes: usize,
    pub allocation_grant_bytes: usize,
    pub allocation_roots: usize,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HybridRankStateBudget {
    pub per_sequence_bytes: usize,
    pub per_sequence_grants: usize,
    /// Explicit simultaneous roots: committed, working, fork/default states.
    pub copies: usize,
    pub total_grants: usize,
    /// Existing BF16 lifecycle replaces one convolution root while copying.
    pub copy_workspace_grants: usize,
}

/// Parameter/state/KV storage ledger. Does NOT include derived RoPE tables,
/// operation/GemmEx/provider workspaces, transport, allocator pool slack, or
/// external experts; not a complete image estimate or execution admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridRankStorageBudget {
    pub precision: HybridRankPrecision,
    pub parameters: Vec<HybridRankParameterBudget>,
    pub weight_grants: usize,
    pub state: HybridRankStateBudget,
    pub kv_bytes: usize,
    pub kv_grants: usize,
    /// Includes all physical slots, including journal shadows/COW. Not tokens.
    pub physical_pages: usize,
}
impl HybridRankStorageBudget {
    pub fn storage_grants(&self) -> Result<usize> {
        add(
            add(self.weight_grants, self.state.total_grants)?,
            self.kv_grants,
        )
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("rank storage sum overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("rank storage product overflow"))
}
fn grant(bytes: usize) -> Result<usize> {
    ferrule_backend::cuda_allocation_grant_bytes(bytes)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("rank allocation grant overflow"))
}
fn replicated(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::TokenEmbedding
            | TensorRole::OutputNorm
            | TensorRole::AttentionNorm
            | TensorRole::FeedForwardNorm
            | TensorRole::AttentionQueryNorm
            | TensorRole::AttentionKeyNorm
            | TensorRole::LinearAttentionNorm
    )
}
impl HybridRankLayout {
    pub fn state_budget(&self, copies: usize) -> Result<HybridRankStateBudget> {
        if copies == 0 {
            return Err(invalid("state copies must include the default state"));
        }
        let mut result = HybridRankStateBudget {
            copies,
            ..Default::default()
        };
        for layer in self.state_schema().layers() {
            if let HybridLayerSchema::GatedDeltaNet(shape) = layer {
                let (_, conv, recurrent) = shape.sizes()?;
                let conv = mul(conv, self.precision().bytes_per_activation())?;
                let recurrent = mul(recurrent, 4)?;
                result.per_sequence_bytes = add(result.per_sequence_bytes, add(conv, recurrent)?)?;
                result.per_sequence_grants = add(
                    result.per_sequence_grants,
                    add(grant(conv)?, grant(recurrent)?)?,
                )?;
                if self.precision().is_mixed() {
                    result.copy_workspace_grants = result.copy_workspace_grants.max(grant(conv)?);
                }
            }
        }
        result.total_grants = mul(copies, result.per_sequence_grants)?;
        Ok(result)
    }

    pub fn parameter_budget(&self, binding: &BoundParameter) -> Result<HybridRankParameterBudget> {
        match binding.residency() {
            ParameterResidency::Layer { layer } | ParameterResidency::Expert { layer, .. }
                if self.state_schema().entry(*layer).is_none() =>
            {
                return Err(invalid("parameter outside rank layer range"));
            }
            ParameterResidency::Attachment { .. } => {
                return Err(invalid("rank attachment parameter unsupported"));
            }
            ParameterResidency::Expert { .. } => {
                self.parameter_partition(binding)?;
            }
            _ => {}
        }
        let mut result = HybridRankParameterBudget {
            parameter: binding.id(),
            role: binding.role().clone(),
            weight_bytes: 0,
            scale_bytes: 0,
            source_bytes: 0,
            allocation_grant_bytes: 0,
            allocation_roots: 0,
        };
        if binding.numeric_fp8_encoding().is_some() {
            if replicated(binding.role()) {
                return Err(invalid("numeric replicated vector/embedding unsupported"));
            }
            let plan = NumericFp8RankPlan::new(binding, Arc::new(self.clone()))?;
            let budget = plan.budget();
            result.weight_bytes = budget.weight_bytes;
            result.scale_bytes = budget.scale_bytes;
            result.source_bytes = budget.payload_bytes;
            result.allocation_grant_bytes = budget.allocation_grant_bytes;
            result.allocation_roots = mul(plan.reads().len(), 2)?;
        } else {
            self.validate_native_storage(binding)?;
            let elements = if replicated(binding.role()) {
                binding.weight().slice().element_count()?
            } else {
                let [n, k] = self.parameter_partition(binding)?.local_shape()?;
                mul(n, k)?
            };
            result.weight_bytes = mul(elements, self.precision().bytes_per_activation())?;
            result.source_bytes = mul(
                elements,
                if binding.weight().slice().dtype == CheckpointDType::F32 {
                    4
                } else {
                    2
                },
            )?;
            result.allocation_grant_bytes = grant(result.weight_bytes)?;
            result.allocation_roots = 1;
        }
        Ok(result)
    }

    pub(crate) fn validate_native_storage(&self, binding: &BoundParameter) -> Result<()> {
        let dtype = &binding.weight().slice().dtype;
        if binding.scale().is_some()
            || binding.weight().transform() != &TensorTransform::Identity
            || !matches!(dtype, CheckpointDType::F32 | CheckpointDType::Bf16)
            || (self.precision().is_mixed()
                && (*dtype != CheckpointDType::Bf16
                    && !(*dtype == CheckpointDType::F32
                        && matches!(
                            binding.role(),
                            TensorRole::LinearAttentionALog | TensorRole::LinearAttentionNorm
                        ))))
        {
            return Err(invalid(
                "rank native storage requires identity BF16; Mixed F32 source exception is GDN A_log/norm only",
            ));
        }
        Ok(())
    }

    pub fn storage_budget(
        &self,
        resources: &BoundDecoderResources,
        segment: &LayerSegmentPlan,
        page_size: usize,
        max_positions: usize,
        physical_pages: usize,
        state_copies: usize,
    ) -> Result<HybridRankStorageBudget> {
        if resources.spec() != self.spec.as_ref()
            || segment.layers() != self.state_schema().layer_range()
            || physical_pages == 0
        {
            return Err(invalid(
                "storage budget model/segment/physical page mismatch",
            ));
        }
        let inventory =
            crate::transformer::segment_inventory::HybridSegmentInventory::new(resources, segment)?;
        let mut parameters = Vec::new();
        let mut weight_grants = 0;
        for binding in inventory.parameters() {
            if matches!(binding.residency(), ParameterResidency::Expert { .. })
                && !self.routed_execution().is_local_tensor_sharded()
            {
                continue;
            }
            let budget = self.parameter_budget(binding)?;
            weight_grants = add(weight_grants, budget.allocation_grant_bytes)?;
            parameters.push(budget);
        }
        let planes = self.kv_planes(page_size, max_positions)?;
        let page = KvLayoutSchema::checked_page_bytes(&planes)
            .ok_or_else(|| invalid("rank KV bytes overflow"))?;
        let plane = mul(page / 2, physical_pages)?;
        Ok(HybridRankStorageBudget {
            precision: self.precision(),
            parameters,
            weight_grants,
            state: self.state_budget(state_copies)?,
            kv_bytes: mul(plane, 2)?,
            kv_grants: mul(grant(plane)?, 2)?,
            physical_pages,
        })
    }

    pub(crate) fn validate_numeric_storage(&self, encoding: NumericFp8Encoding) -> Result<()> {
        if self.precision().is_mixed() && encoding != NumericFp8Encoding::E4M3FnBlock128Bf16 {
            return Err(invalid("Mixed rank requires original BF16 block scales"));
        }
        Ok(())
    }
}
