//! Pure admission for an executable local block slice, not a decoder profile.
use super::*;
use crate::transformer::{HybridRankStorage, PreparedHybridRankParameter};
use crate::{TensorRole, nn::ParameterResidency};
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision, NumericFp8ScaleType,
    bf16_gemmex::Bf16GemmExPlan,
};

/// Chosen explicitly; unsupported GemmEx geometry never switches arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridMixedProjectionArithmetic {
    W8A16Bf16,
    NativeBf16GemmEx,
}
impl HybridMixedProjectionArithmetic {
    /// A packed local scale grid may be rebased only at complete block origins.
    /// Original global read/source provenance remains in the rank binding.
    pub fn packed_gemmex_layout(layout: NumericFp8Layout) -> Result<NumericFp8Layout> {
        layout.storage_lengths()?;
        if layout.scale_type != NumericFp8ScaleType::Bf16
            || !layout.row_origin.is_multiple_of(128)
            || !layout.column_origin.is_multiple_of(128)
            || !layout.n.is_multiple_of(128)
            || !layout.k.is_multiple_of(128)
        {
            return Err(invalid(
                "rank GemmEx requires complete BF16-scale blocks; no fallback",
            ));
        }
        Ok(NumericFp8Layout {
            row_origin: 0,
            column_origin: 0,
            ..layout
        })
    }
}

/// Exact static metadata for the internal routed-down M=1 candidate.
/// Provider availability remains a separate CUDA capability query at submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct HybridMixedRoutedDownM1Capability {
    pub n: usize,
    pub k: usize,
    pub kernel: ferrule_backend::cuda_plan::NumericFp8SelectedKernel,
    pub kernel_name: &'static str,
    pub launches: usize,
    pub workspace_bytes: usize,
    pub provider_bytes: usize,
    pub fused_shared_bytes_per_cta: Option<usize>,
    /// Native W8A16 is compiled for SM80+, even though M1 uses SIMT reduction.
    pub minimum_sm: u32,
    pub f32_accumulator_grants: usize,
    pub bf16_output_grants: usize,
    /// Separate F32-to-BF16 RNE kernel; not included in `launches`.
    pub bf16_writeback_launches: usize,
    pub production_validated: bool,
}

#[derive(Debug, Clone)]
pub struct HybridMixedBlockPlan {
    pub(crate) layout: Arc<HybridRankLayout>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) bindings: Vec<crate::transformer::BoundParameter>,
    pub(crate) arithmetic: HybridMixedProjectionArithmetic,
    pub(crate) max_rows: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) views: Vec<Bf16GemmExPlan>,
    pub(crate) expert_counts: std::collections::BTreeMap<usize, usize>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) decoded: Option<(
        HybridMixedDecodedCacheBudget,
        super::mixed_decoded_arena::DecodedArenaDescriptors,
    )>,
    // Compatibility observation of the same Arc carried by decoded descriptors.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) decoded_arena: Option<Arc<super::mixed_decoded_arena::DecodedArenaPlan>>,
    weight_grants: usize,
    weight_bytes: usize,
    workspace_grants: usize,
    provider_bytes: usize,
    routed_down_m1_accumulator_grants: Option<usize>,
    routed_down_bindings:
        std::collections::BTreeMap<crate::nn::ParameterId, (usize, NumericFp8Layout)>,
}
impl HybridMixedBlockPlan {
    /// Every expert of each included routed layer must be present (EP=1).
    /// Limits cover live weight/scale roots and the reusable provider workspace;
    /// activations, routing scratch, KV/state/COW and allocator slack are separate.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layout: Arc<HybridRankLayout>,
        parameters: &[PreparedHybridRankParameter],
        max_rows: usize,
        arithmetic: HybridMixedProjectionArithmetic,
        weight_limit: usize,
        workspace_limit: usize,
        provider_limit: usize,
    ) -> Result<Self> {
        if !layout.precision().is_mixed() || !(1..=32).contains(&max_rows) || parameters.is_empty()
        {
            return Err(invalid(
                "Mixed block requires native rank storage and M1..32",
            ));
        }
        let mut keys = BTreeSet::new();
        let mut experts = BTreeSet::new();
        let mut expert_counts = std::collections::BTreeMap::new();
        let mut views = Vec::new();
        let mut routed_down_bindings = std::collections::BTreeMap::new();
        let mut weight_grants = 0usize;
        let mut weight_bytes = 0usize;
        for (slot, p) in parameters.iter().enumerate() {
            p.validate_binding(p.binding(), &layout)?;
            if !keys.insert((p.binding().id(), p.binding().role().clone())) {
                return Err(invalid("duplicate Mixed rank projection"));
            }
            let [n, k] = p.local_shape();
            if p.binding().weight().logical_shape().len() != 2 {
                return Err(invalid("Mixed block directory accepts projections only"));
            }
            let budget = layout.parameter_budget(p.binding())?;
            let combined_qkv = match (arithmetic, p.storage()) {
                (
                    HybridMixedProjectionArithmetic::NativeBf16GemmEx,
                    HybridRankStorage::Numeric(n),
                ) => n.plan().mixed_gemmex_qkv_layout()?,
                _ => None,
            };
            weight_bytes = weight_bytes
                .checked_add(budget.weight_bytes)
                .and_then(|n| n.checked_add(budget.scale_bytes))
                .ok_or_else(|| invalid("Mixed rank physical payload overflow"))?;
            let physical_grants = if let Some(qkv) = combined_qkv {
                let (w, s) = qkv.storage_lengths()?;
                ferrule_backend::cuda_allocation_grant_bytes(w)
                    .and_then(|w| w.checked_add(ferrule_backend::cuda_allocation_grant_bytes(s)?))
                    .ok_or_else(|| invalid("Mixed QKV physical grants overflow"))?
            } else {
                budget.allocation_grant_bytes
            };
            weight_grants = weight_grants
                .checked_add(physical_grants)
                .filter(|n| *n <= weight_limit)
                .ok_or_else(|| invalid("Mixed rank resident weight grant"))?;
            if let ParameterResidency::Expert { layer, expert } = p.binding().residency() {
                let FeedForward::Moe(moe) = layout.spec.layers()[*layer].feed_forward() else {
                    return Err(invalid("expert outside MoE layer"));
                };
                expert_counts.insert(*layer, moe.router_spec().num_experts());
                if !experts.insert((*layer, *expert, p.binding().role().clone())) {
                    return Err(invalid("duplicate local expert role"));
                }
            }
            match p.storage() {
                HybridRankStorage::NumericBinding(_) => {
                    return Err(invalid(
                        "retired Mixed host source cannot prepare a new image",
                    ));
                }
                HybridRankStorage::Dense(_) => {
                    return Err(invalid("Mixed rejects resident F32 weights"));
                }
                HybridRankStorage::Bf16(_) => {
                    for rows in 1..=max_rows {
                        views.push(Bf16GemmExPlan::new_resident(
                            n,
                            k,
                            rows,
                            workspace_limit,
                            provider_limit,
                        )?);
                    }
                }
                HybridRankStorage::Numeric(prepared) => {
                    if let Some(qkv) = combined_qkv {
                        for rows in 1..=max_rows {
                            views.push(Bf16GemmExPlan::new(
                                qkv,
                                rows,
                                workspace_limit,
                                provider_limit,
                            )?);
                        }
                        continue;
                    }
                    for segment in prepared.segments() {
                        let l = segment.artifact().validated_payload().layout();
                        if l.scale_type != NumericFp8ScaleType::Bf16 {
                            return Err(invalid("Mixed requires original BF16 scales"));
                        }
                        if arithmetic == HybridMixedProjectionArithmetic::W8A16Bf16 {
                            ferrule_backend::cuda_plan::NumericFp8LinearPlan::new(l, max_rows, 0,
                                ferrule_backend::cuda_plan::NumericFp8Precision::W8A16Bf16F32Accumulate)?;
                        }
                        if arithmetic == HybridMixedProjectionArithmetic::NativeBf16GemmEx {
                            if p.binding().role() == &TensorRole::RoutedExpertDown
                                && prepared.segments().len() == 1
                            {
                                routed_down_bindings.insert(p.binding().id(), (slot, l));
                            }
                            let l = HybridMixedProjectionArithmetic::packed_gemmex_layout(l)?;
                            for rows in 1..=max_rows {
                                views.push(Bf16GemmExPlan::new(
                                    l,
                                    rows,
                                    workspace_limit,
                                    provider_limit,
                                )?);
                                if matches!(p.binding().role(), TensorRole::RoutedExpertGate) {
                                    let n =
                                        l.n.checked_mul(2)
                                            .ok_or_else(|| invalid("gate/up width overflow"))?;
                                    views.push(Bf16GemmExPlan::new(
                                        NumericFp8Layout { n, ..l },
                                        rows,
                                        workspace_limit,
                                        provider_limit,
                                    )?);
                                }
                            }
                        }
                    }
                }
            }
        }
        for (&layer, &count) in &expert_counts {
            for expert in 0..count {
                for role in [
                    TensorRole::RoutedExpertGate,
                    TensorRole::RoutedExpertUp,
                    TensorRole::RoutedExpertDown,
                ] {
                    if !experts.contains(&(layer, expert, role)) {
                        return Err(invalid(
                            "all local tensor expert shards must be resident; no offload/cache fallback",
                        ));
                    }
                }
            }
        }
        let mut unique = Vec::new();
        for view in views {
            if !unique.contains(&view) {
                unique.push(view);
            }
        }
        let views = unique;
        let workspace_grants = views
            .iter()
            .map(|p| p.storage_grant_bytes())
            .max()
            .unwrap_or(0);
        let provider_bytes = views.first().map_or(0, |p| p.provider_bytes());
        Ok(Self {
            layout,
            bindings: parameters.iter().map(|p| p.binding().clone()).collect(),
            arithmetic,
            max_rows,
            views,
            expert_counts,
            decoded: None,
            decoded_arena: None,
            weight_grants,
            weight_bytes,
            workspace_grants,
            provider_bytes,
            routed_down_m1_accumulator_grants: None,
            routed_down_bindings,
        })
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn with_decoded_budget(
        mut self,
        budget: HybridMixedDecodedCacheBudget,
    ) -> Result<Self> {
        if self.arithmetic != HybridMixedProjectionArithmetic::NativeBf16GemmEx
            || self.decoded.is_some()
            || self.routed_down_m1_accumulator_grants.is_some()
        {
            return Err(invalid(
                "decoded cache requires unchanged native arithmetic",
            ));
        }
        let descriptors = super::mixed_decoded::decoded_descriptors(&self.layout, &self.bindings)?;
        if HybridMixedDecodedCacheBudget::new(budget.capacity_grants, &descriptors)? != budget {
            return Err(invalid(
                "decoded prepared descriptors differ from admitted ledger",
            ));
        }
        let descriptors =
            super::mixed_decoded_arena::DecodedArenaDescriptors::new(budget, descriptors)?;
        self.decoded_arena = Some(descriptors.layout.clone());
        self.decoded = Some((budget, descriptors));
        Ok(self)
    }

    /// Internal operator candidate, NOT a validated production precision profile.
    /// Retains every native view for unsupported roles, shapes, rows and providers.
    /// No environment variable or generic model factory enables this candidate.
    #[doc(hidden)]
    pub fn with_routed_down_m1_candidate(mut self, accumulator_limit: usize) -> Result<Self> {
        if self.arithmetic != HybridMixedProjectionArithmetic::NativeBf16GemmEx
            || self.decoded.is_some()
            || self.layout.members().len() != 2
        {
            return Err(invalid(
                "routed-down M1 candidate requires native Mixed TP2",
            ));
        }
        let grants = Self::routed_down_m1_accumulator_grants(&self.layout)?;
        if grants > accumulator_limit {
            return Err(invalid(
                "routed-down M1 candidate accumulator grant exhausted",
            ));
        }
        self.routed_down_m1_accumulator_grants = Some(grants);
        Ok(self)
    }

    /// Additional live F32 outputs; BF16 writeback is already in the native
    /// operation envelope. Reserve all local experts because the block operator
    /// accepts explicit routes, not only the production router's configured top-k.
    pub fn candidate_accumulator_grant_bytes(&self) -> usize {
        self.routed_down_m1_accumulator_grants.unwrap_or(0)
    }

    pub(super) fn routed_down_m1_accumulator_grants(layout: &HybridRankLayout) -> Result<usize> {
        if !layout.precision().is_mixed() || layout.members().len() != 2 {
            return Err(invalid("routed-down M1 candidate requires Mixed TP2"));
        }
        let mut peak = 0;
        for layer in layout.state_schema().layer_range() {
            if let FeedForward::Moe(moe) = layout.spec.layers()[layer].feed_forward() {
                let bytes = layout
                    .spec
                    .hidden_size()
                    .checked_mul(4)
                    .ok_or_else(|| invalid("candidate accumulator size overflow"))?;
                let grant = ferrule_backend::cuda_allocation_grant_bytes(bytes)
                    .and_then(|n| n.checked_mul(moe.router_spec().num_experts()))
                    .ok_or_else(|| invalid("candidate accumulator grant overflow"))?;
                peak = peak.max(grant);
            }
        }
        Ok(peak)
    }

    /// Exact shape/role capability, not an equivalence or provider-availability
    /// claim. The plan identifies the vector kernel, zero scratch and one numeric
    /// launch; caller-owned F32 output and BF16 RNE writeback are additional.
    /// `invocation_rows` is the original routed input, NOT the gathered bucket.
    pub fn routed_down_m1_candidate_capability(
        &self,
        binding: &crate::transformer::BoundParameter,
        layout: NumericFp8Layout,
        invocation_rows: usize,
        projection_rows: usize,
    ) -> Option<HybridMixedRoutedDownM1Capability> {
        let plan =
            self.routed_down_m1_candidate_plan(binding, layout, invocation_rows, projection_rows)?;
        Some(HybridMixedRoutedDownM1Capability {
            n: plan.layout().n,
            k: plan.layout().k,
            kernel: plan.selected_kernel(),
            kernel_name: plan.selected_kernel().name(),
            launches: plan.kernel_launches(),
            workspace_bytes: plan.workspace_requirements().bytes as usize,
            provider_bytes: plan.provider_bytes(),
            fused_shared_bytes_per_cta: plan.fused_shared_bytes_per_cta(),
            minimum_sm: 80,
            f32_accumulator_grants: ferrule_backend::cuda_allocation_grant_bytes(
                plan.layout().n.checked_mul(4)?,
            )?,
            bf16_output_grants: ferrule_backend::cuda_allocation_grant_bytes(
                plan.layout().n.checked_mul(2)?,
            )?,
            bf16_writeback_launches: 1,
            production_validated: false,
        })
    }

    pub fn routed_down_m1_candidate_plan(
        &self,
        binding: &crate::transformer::BoundParameter,
        layout: NumericFp8Layout,
        invocation_rows: usize,
        projection_rows: usize,
    ) -> Option<NumericFp8LinearPlan> {
        if self.routed_down_m1_accumulator_grants.is_none()
            || binding.role() != &TensorRole::RoutedExpertDown
            || invocation_rows != 1
            || projection_rows != 1
            || layout.n != self.layout.spec.hidden_size()
        {
            return None;
        }
        let (layer, expert) = match binding.residency() {
            ParameterResidency::Expert { layer, expert } => (*layer, *expert),
            _ => return None,
        };
        // Metadata-only lookup survives host payload retirement. Shape equality,
        // a reused ID, or a tied storage root alone cannot authenticate a role.
        let &(slot, source_layout) = self.routed_down_bindings.get(&binding.id())?;
        let bound = self.bindings.get(slot)?;
        if bound.spec() != binding.spec()
            || bound.role() != binding.role()
            || bound.canonical_id() != binding.canonical_id()
            || !bound.shares_storage_with(binding)
            || expert >= *self.expert_counts.get(&layer)?
        {
            return None;
        }
        let FeedForward::Moe(_) = self.layout.spec.layers().get(layer)?.feed_forward() else {
            return None;
        };
        if self
            .layout
            .parameter_partition(bound)
            .ok()?
            .local_shape()
            .ok()?
            != [layout.n, layout.k]
        {
            return None;
        }
        let packed = HybridMixedProjectionArithmetic::packed_gemmex_layout(source_layout).ok()?;
        if layout != source_layout && layout != packed {
            return None;
        }
        NumericFp8LinearPlan::new(packed, 1, 0, NumericFp8Precision::W8A16Bf16F32Accumulate).ok()
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn weight_grant_bytes(&self) -> usize {
        self.weight_grants
    }
    pub fn workspace_grant_bytes(&self) -> usize {
        self.workspace_grants
    }
    pub fn provider_bytes(&self) -> usize {
        self.provider_bytes
    }
    pub fn all_expert_shards_resident(&self, layer: usize) -> bool {
        self.expert_counts.contains_key(&layer)
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub fn arithmetic(&self) -> HybridMixedProjectionArithmetic {
        self.arithmetic
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
}
