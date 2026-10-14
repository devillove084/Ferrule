//! Static hybrid CUDA profile/ABI validation and compressed-aware memory planning.
//! No provider discovery, device capability probe, allocation or execution permission.
use super::{GenericDecoderOptions, HybridLayerSchema, HybridStateSchema, IdleKvReclaimConfig};
use crate::checkpoint::CheckpointDType;
use crate::nn::ParameterResidency;
use crate::support::TensorRole;
use crate::transformer::{Attention, BoundDecoderResources, FeedForward, UnsupportedOperator};
use crate::transformer::{HybridSegmentMemoryEstimate, LayerSegmentPlan};
use crate::transformer::{RouterScoreFunction, RouterSelection, SwiGluScratchPlan};
use ferrule_backend::cuda_plan::{CausalConv1dLayout, GatedDeltaNetLayout};
use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType};
use ferrule_common::execution::KvLayoutSchema;
use ferrule_common::{Error, Result};
use std::collections::BTreeMap;

/// Byte/capacity requirements, not a reservation or a device capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridReservationRequirements {
    /// Physical slots including transaction shadows/COW, not just committed pages.
    pub kv_pages: usize,
    /// Default state, active sequences, forks and working copies all consume this cap.
    pub state_bytes: usize,
    /// Non-expert bindings plus the bounded expert cache less numeric scratch.
    pub weight_bytes: usize,
    /// Conservative operation workspace plus numeric scratch, counted once.
    pub workspace_bytes: usize,
}

/// Static F32-state memory plan, including explicit numeric operand profiles.
/// Native BF16 image admission remains on the CUDA-only implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridMemoryPlan {
    pub per_sequence_state_bytes: usize,
    pub kv_bytes: usize,
    pub weight_bytes_upper_bound: usize,
    pub workspace_bytes_upper_bound: usize,
}

fn error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_cuda", message)),
    }
}

fn validate_state_abi(schema: &HybridStateSchema) -> Result<()> {
    for layer in schema.layers() {
        if let HybridLayerSchema::GatedDeltaNet(s) = layer {
            CausalConv1dLayout {
                rows: 1,
                channels: s.sizes()?.0,
                kernel_size: s.kernel,
            }
            .validate()?;
            GatedDeltaNetLayout {
                rows: 1,
                key_heads: s.key_heads,
                value_heads: s.value_heads,
                key_dim: s.key_dim,
                value_dim: s.value_dim,
            }
            .validate()?;
        }
    }
    Ok(())
}

impl GenericDecoderOptions {
    pub(crate) fn validate_hybrid_cuda_profile(
        &self,
        resources: &BoundDecoderResources,
    ) -> Result<()> {
        self.validate(resources)?;
        if self.precision() != crate::execution::ExecutionPrecisionPolicy::f32() {
            return Err(unsupported(
                "hybrid CUDA requires F32 activation/state/output execution",
            ));
        }
        let numeric = self.hybrid_cuda_numeric_fp8();
        if !resources.spec().attachments().is_empty() {
            return Err(unsupported("hybrid CUDA attachments/proposals unsupported"));
        }
        for layer in resources.spec().layers() {
            match layer.feed_forward() {
                FeedForward::SwiGlu(_) => {}
                FeedForward::Moe(m) if numeric.is_some() => {
                    let r = m.router_spec();
                    if r.score_function() != RouterScoreFunction::Softmax
                        || r.selection() != &RouterSelection::TopK
                        || !r.normalize_selected()
                        || r.selection_bias()
                        || m.expert().gate().has_bias()
                    {
                        return Err(unsupported(
                            "numeric hybrid MoE requires bias-free experts and normalized softmax top-k without selection bias",
                        ));
                    }
                }
                _ => {
                    return Err(unsupported(
                        "legacy hybrid CUDA supports dense FFN only; routed MoE requires the explicit numeric FP8 profile",
                    ));
                }
            }
        }
        for p in resources.state_dict().parameters() {
            if let Some(encoding) = p.numeric_fp8_encoding() {
                if numeric.is_none() {
                    return Err(unsupported(
                        "numeric FP8 checkpoint requires the explicit hybrid CUDA numeric profile",
                    ));
                }
                // Validate geometry and source identities without reading payloads.
                let source = p.numeric_fp8_source(encoding)?;
                source.validate_source_identity()?;
                if source.expert_count().is_some() || !numeric_linear_role(p.role()) {
                    return Err(unsupported(
                        "numeric hybrid requires individual TP1 linear matrices; vectors, embeddings and packed expert tensors cannot use numeric FP8",
                    ));
                }
            } else if !matches!(
                p.weight().slice().dtype,
                CheckpointDType::F32 | CheckpointDType::Bf16
            ) || p.scale().is_some()
            {
                return Err(unsupported(
                    "hybrid CUDA checkpoint storage must be unscaled F32/BF16 or explicit numeric FP8",
                ));
            }
        }
        Ok(())
    }
}
pub(crate) fn numeric_linear_role(role: &TensorRole) -> bool {
    use TensorRole::*;
    matches!(
        role,
        OutputHead
            | AttentionQuery
            | AttentionKey
            | AttentionValue
            | AttentionOutput
            | LinearAttentionQkv
            | LinearAttentionZ
            | LinearAttentionBeta
            | LinearAttentionA
            | RouterLogits
            | RoutedExpertGate
            | RoutedExpertUp
            | RoutedExpertDown
            | SharedExpertGate
            | SharedExpertUp
            | SharedExpertDown
            | SharedExpertOutputGate
            | DenseMlpGate
            | DenseMlpUp
            | DenseMlpDown
    )
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("hybrid memory estimate overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("hybrid memory estimate overflow"))
}

impl HybridMemoryPlan {
    /// Finite shape set for one sequential Mixed provider. Full vocabulary and
    /// unsupported roles never contribute to its maximum staging allocation.
    pub fn bf16_gemmex_plans(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<Vec<ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan>> {
        let Some(precision) = options.hybrid_cuda_numeric_fp8_precision() else {
            return Ok(Vec::new());
        };
        if precision != crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate {
            return Ok(Vec::new());
        }
        let parameters = resources.state_dict().parameters();
        let mut plans = Vec::new();
        for p in parameters {
            let Some(encoding) = p.numeric_fp8_encoding() else {
                if p.weight().slice().dtype == CheckpointDType::Bf16
                    && let [n, k] = p.weight().slice().shape.as_slice()
                {
                    for rows in 1..=options.capabilities().max_batch_tokens.min(32) {
                        if let Some(plan) = precision.resident_bf16_gemmex_plan(
                            p.role(),
                            *n,
                            *k,
                            rows,
                            usize::MAX,
                            usize::MAX,
                        )? && !plans.contains(&plan)
                        {
                            plans.push(plan);
                        }
                    }
                }
                continue;
            };
            let source = p.numeric_fp8_source(encoding)?;
            source.validate_source_identity()?;
            let [n, k] = source.matrix_shape();
            let layout = NumericFp8Layout {
                n,
                k,
                row_origin: 0,
                column_origin: 0,
                scale_type: match encoding {
                    crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128Bf16 => {
                        NumericFp8ScaleType::Bf16
                    }
                    crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128F32 => {
                        NumericFp8ScaleType::F32
                    }
                },
            };
            let pair = if p.role() == &TensorRole::RoutedExpertGate {
                let up = parameters
                    .iter()
                    .find(|q| {
                        q.role() == &TensorRole::RoutedExpertUp && q.residency() == p.residency()
                    })
                    .ok_or_else(|| error("Mixed GemmEx gate missing paired up metadata"))?;
                if up.spec().shape() != p.spec().shape()
                    || up.numeric_fp8_encoding() != p.numeric_fp8_encoding()
                {
                    return Err(error("Mixed GemmEx incompatible gate/up metadata"));
                }
                let source = up.numeric_fp8_source(
                    up.numeric_fp8_encoding()
                        .ok_or_else(|| error("Mixed GemmEx missing up encoding"))?,
                )?;
                source.validate_source_identity()?;
                Some(layout)
            } else {
                None
            };
            for rows in 1..=options.capabilities().max_batch_tokens.min(32) {
                let plan = if let Some(up) = pair {
                    precision.gemmex_gate_up_plan(layout, up, rows, usize::MAX, usize::MAX)?
                } else {
                    precision.gemmex_plan(p.role(), layout, rows, usize::MAX, usize::MAX)?
                };
                if let Some(plan) = plan
                    && !plans.contains(&plan)
                {
                    plans.push(plan);
                }
            }
        }
        Ok(plans)
    }

    /// Logical full-matrix staging plus fixed native workspace, and one
    /// separately admitted external-library reserve, never per projection.
    pub fn bf16_gemmex_bytes(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<(usize, usize)> {
        let plans = Self::bf16_gemmex_plans(resources, options)?;
        Ok((
            plans.iter().map(|p| p.storage_bytes()).max().unwrap_or(0),
            plans.first().map_or(0, |p| p.provider_bytes()),
        ))
    }

    /// One prepared Mixed attention owner, shared sequentially by all GQA layers.
    /// Fixed head geometry is a backend contract, not permission to pad matrices.
    pub fn bf16_attention_plan(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<Option<ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionPlan>> {
        use ferrule_backend::cuda_plan::bf16_attention::{Bf16AttentionPlan, Bf16AttentionShape};
        if super::mixed_attention::MixedAttentionCapacity::for_resources(resources, options)?
            .is_none()
        {
            return Ok(None);
        }
        let mut geometry = None;
        for layer in resources.spec().layers() {
            if let Attention::Gqa(g) = layer.attention() {
                let shape = (g.num_heads(), g.num_kv_heads(), g.head_dim());
                if geometry.is_some_and(|previous| previous != shape) {
                    return Err(error(
                        "Mixed prepared attention requires uniform GQA head geometry",
                    ));
                }
                geometry = Some(shape);
            }
        }
        let (heads, kv_heads, dim) = geometry.expect("GQA capacity validated");
        let rows = options.capabilities().max_batch_tokens;
        let tokens = options.max_positions();
        let shapes = (1..=rows.min(tokens))
            .flat_map(|m| {
                (m..=tokens).map(move |n| Bf16AttentionShape {
                    rows: m as u32,
                    tokens: n as u32,
                })
            })
            .collect::<Vec<_>>();
        Bf16AttentionPlan::from_shapes(
            &shapes,
            rows,
            options.capabilities().max_sequences.min(rows),
            heads,
            kv_heads,
            dim,
            usize::MAX,
            usize::MAX,
        )
        .map(Some)
    }

    /// Physical status and worst-case simultaneously pending adapter metadata.
    /// The provider panels/external reserve are charged separately, once/image.
    pub fn bf16_attention_metadata_bytes(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<usize> {
        if super::mixed_attention::MixedAttentionCapacity::for_resources(resources, options)?
            .is_none()
        {
            return Ok(0);
        }
        let sequences = options.capabilities().max_sequences;
        let rows = options.capabilities().max_batch_tokens;
        let slots = mul(
            sequences,
            options.max_positions().div_ceil(options.page_size()),
        )?;
        let grant = |elements| {
            ferrule_backend::cuda_allocation_grant_bytes(mul(elements, 4)?)
                .ok_or_else(|| error("Mixed attention metadata grant overflow"))
        };
        let per_layer = add(
            add(grant(slots)?, grant(add(sequences, 1)?)?)?,
            mul(3, grant(rows)?)?,
        )?;
        let layers = resources
            .spec()
            .layers()
            .iter()
            .filter(|l| matches!(l.attention(), Attention::Gqa(_)))
            .count();
        add(grant(1)?, mul(layers, per_layer)?)
    }

    /// One image-owned chunk workspace, shared sequentially by GDN layers and
    /// sequence cohorts. This is logical payload, not a per-layer/state charge.
    pub fn bf16_chunk_workspace_bytes(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<usize> {
        let Some(precision) = options.hybrid_cuda_numeric_fp8_precision() else {
            return Ok(0);
        };
        if precision != crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate {
            return Ok(0);
        }
        let schema =
            HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?;
        schema.layers().iter().try_fold(0usize, |largest, layer| {
            let HybridLayerSchema::GatedDeltaNet(shape) = layer else {
                return Ok(largest);
            };
            let plan = precision.gdn_chunk_plan(
                GatedDeltaNetLayout {
                    rows: options.capabilities().max_batch_tokens,
                    key_heads: shape.key_heads,
                    value_heads: shape.value_heads,
                    key_dim: shape.key_dim,
                    value_dim: shape.value_dim,
                },
                false, // Every image must admit a fresh sequence, even at capacity one.
                usize::MAX,
            )?;
            Ok(largest.max(plan.map_or(0, |p| p.workspace_requirements().bytes as usize)))
        })
    }

    fn numeric_bf16_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        options.validate(resources)?;
        if options.precision() != crate::execution::ExecutionPrecisionPolicy::bf16_compatibility()
            || options
                .active_layers()
                .is_some_and(|n| n != resources.spec().layers().len())
        {
            return Err(unsupported(
                "numeric BF16 requires BF16 model precision and complete layers",
            ));
        }
        crate::transformer::NumericFp8Bf16HybridCudaProfile::validate(resources)?;
        // Fail at image admission, not after appending token 1025 to a live cache.
        super::mixed_attention::MixedAttentionCapacity::for_resources(resources, options)?;
        let (limits, scratch) = options
            .hybrid_cuda_numeric_fp8()
            .ok_or_else(|| error("missing numeric BF16 budget"))?;
        // Reuse the existing checked cache/route/ABI envelope. Its F32 temporary
        // sizes conservatively cover BF16 rows plus F32 accumulation/packing.
        let sizing = GenericDecoderOptions::new(
            options.family().clone(),
            options.weight_source(),
            options.capabilities(),
            options.page_size(),
            options.max_positions(),
            options.max_parameter_bytes(),
            crate::execution::ExecutionPrecisionPolicy::f32(),
        )
        .with_hybrid_cuda_numeric_fp8_precision(
            limits,
            scratch,
            match options.hybrid_cuda_numeric_fp8_precision() {
                Some(crate::transformer::NumericFp8Precision::Bf16ModelTiledF32Accumulate)
                | Some(crate::transformer::NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate)
                | Some(crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate) => {
                    crate::transformer::NumericFp8Precision::Bf16RneF32Accumulate
                }
                Some(crate::transformer::NumericFp8Precision::Bf16ModelF32Accumulate) => {
                    crate::transformer::NumericFp8Precision::W8A16Bf16F32Accumulate
                }
                _ => return Err(error("missing BF16 model arithmetic identity")),
            },
        )?;
        let mut plan = Self::for_resources_inner(resources, &sizing, kv_pages, false)?;
        // Validate each scheduled kernel, including M1 decode even when admission
        // capacity is multi-row. The base tiled envelope has identical global
        // scratch/launch geometry; runtime replaces (never doubles) the workspace.
        let precision = options
            .hybrid_cuda_numeric_fp8_precision()
            .expect("BF16 image");
        let mut provider_bytes = 0usize;
        for p in resources.state_dict().parameters() {
            if let Some(encoding) = p.numeric_fp8_encoding() {
                let source = p.numeric_fp8_source(encoding)?;
                let [n, k] = source.matrix_shape();
                for rows in [1, options.capabilities().max_batch_tokens] {
                    if precision
                        .gemmex_plan(
                            p.role(),
                            NumericFp8Layout {
                                n,
                                k,
                                row_origin: 0,
                                column_origin: 0,
                                scale_type: NumericFp8ScaleType::Bf16,
                            },
                            rows,
                            usize::MAX,
                            usize::MAX,
                        )?
                        .is_some()
                    {
                        continue;
                    }
                    let linear = precision.linear_plan(
                        p.role(),
                        NumericFp8Layout {
                            n,
                            k,
                            row_origin: 0,
                            column_origin: 0,
                            scale_type: NumericFp8ScaleType::Bf16,
                        },
                        rows,
                        scratch,
                    )?;
                    provider_bytes = provider_bytes.max(linear.provider_bytes());
                }
            }
        }
        let schema =
            HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?;
        plan.kv_bytes /= 2;
        plan.per_sequence_state_bytes = schema.layers().iter().try_fold(0usize, |n, l| {
            if let HybridLayerSchema::GatedDeltaNet(s) = l {
                let (_, c, r) = s.sizes()?;
                add(n, add(mul(c, 2)?, mul(r, 4)?)?)
            } else {
                Ok(n)
            }
        })?;
        let mut weights = limits.max_bytes - scratch;
        let mut unique = std::collections::BTreeSet::new();
        for p in resources.state_dict().parameters() {
            if matches!(p.residency(), ParameterResidency::Expert { .. })
                || !unique.insert(p.canonical_id())
            {
                continue;
            }
            weights = add(
                weights,
                usize::try_from(if p.weight().slice().dtype == CheckpointDType::F32 {
                    p.weight().slice().bytes / 2
                } else {
                    p.weight().slice().bytes
                })
                .map_err(|_| error("weight bytes overflow"))?,
            )?;
            if let Some(scale) = p.scale() {
                weights = add(
                    weights,
                    usize::try_from(scale.slice().bytes)
                        .map_err(|_| error("scale bytes overflow"))?,
                )?;
            }
        }
        for l in resources.spec().layers() {
            if let Attention::Gqa(g) = l.attention() {
                weights = add(
                    weights,
                    mul(
                        mul(options.max_positions(), g.rotary().region().dimensions())?,
                        2,
                    )?,
                )?;
            }
        }
        plan.weight_bytes_upper_bound = weights;
        let rows = options.capabilities().max_batch_tokens;
        for layer in resources.spec().layers() {
            let (expert, route) = match layer.feed_forward() {
                FeedForward::SwiGlu(f) => (f, 0),
                FeedForward::Moe(m) => (
                    m.expert(),
                    mul(
                        SwiGluScratchPlan::route(
                            rows,
                            m.router_spec().experts_per_token(),
                            resources.spec().hidden_size(),
                            rows,
                        )?
                        .total_bytes()?,
                        2,
                    )?,
                ),
            };
            let operation = mul(
                SwiGluScratchPlan::for_descriptor(expert, rows)?.total_bytes()?,
                2,
            )?;
            let mut expert_bytes = BTreeMap::<usize, usize>::new();
            for p in resources.state_dict().parameters() {
                if let ParameterResidency::Expert {
                    layer: l,
                    expert: id,
                } = p.residency()
                    && *l == layer.index()
                {
                    let bytes = p
                        .weight()
                        .slice()
                        .bytes
                        .checked_add(p.scale().map_or(0, |s| s.slice().bytes))
                        .ok_or_else(|| error("BF16 expert bytes overflow"))?;
                    let n = expert_bytes.entry(*id).or_default();
                    *n = add(
                        *n,
                        usize::try_from(bytes).map_err(|_| error("BF16 expert bytes overflow"))?,
                    )?;
                }
            }
            let peak = add(
                add(operation, route)?,
                expert_bytes.values().copied().max().unwrap_or(0),
            )?;
            if add(peak, scratch)? > limits.max_bytes {
                return Err(error(
                    "numeric BF16 typed expert/route scratch exceeds cache budget",
                ));
            }
        }
        // Native BF16 attention uses explicit F32 work buffers and typed KV;
        // include the existing native hybrid envelope and allocator slack.
        let width = resources
            .state_dict()
            .parameters()
            .iter()
            .flat_map(|p| p.spec().shape().iter().copied())
            .max()
            .unwrap_or(1)
            .max(resources.spec().vocab_size());
        let native_work = add(
            mul(mul(options.capabilities().max_batch_tokens, width)?, 128)?,
            add(64 << 20, scratch)?,
        )?;
        plan.workspace_bytes_upper_bound = add(
            plan.workspace_bytes_upper_bound.max(native_work),
            add(weights / 10, plan.kv_bytes / 10)?,
        )?;
        let scratch_grant = ferrule_backend::cuda_allocation_grant_bytes(scratch)
            .ok_or_else(|| error("numeric BF16 workspace allocation grant overflow"))?;
        plan.workspace_bytes_upper_bound = add(
            plan.workspace_bytes_upper_bound,
            add(scratch_grant - scratch, provider_bytes)?,
        )?;
        let (gemmex_storage, gemmex_provider) = Self::bf16_gemmex_bytes(resources, options)?;
        let gemmex_grant = ferrule_backend::cuda_allocation_grant_bytes(gemmex_storage)
            .ok_or_else(|| error("Mixed GemmEx allocation grant overflow"))?;
        plan.workspace_bytes_upper_bound = add(
            plan.workspace_bytes_upper_bound,
            add(gemmex_grant, gemmex_provider)?,
        )?;
        let chunk_bytes = Self::bf16_chunk_workspace_bytes(resources, options)?;
        let chunk_grant = ferrule_backend::cuda_allocation_grant_bytes(chunk_bytes)
            .ok_or_else(|| error("BF16 chunk workspace allocation grant overflow"))?;
        plan.workspace_bytes_upper_bound = add(plan.workspace_bytes_upper_bound, chunk_grant)?;
        if let Some(attention) = Self::bf16_attention_plan(resources, options)? {
            plan.workspace_bytes_upper_bound = add(
                plan.workspace_bytes_upper_bound,
                add(
                    attention.total_charge_bytes(),
                    Self::bf16_attention_metadata_bytes(resources, options)?,
                )?,
            )?;
        }
        Ok(plan)
    }

    /// No weight payload reads or CUDA allocations. Numeric experts are charged
    /// by their finite cache ceiling, not by the complete checkpoint size.
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        Self::for_resources_inner(resources, options, kv_pages, false)
    }

    /// Charges actual per-page allocator grants plus bounded Entry staging.
    /// No arithmetic/parallel profile changes; only dense local F32 all-GQA.
    pub fn for_resources_with_idle_kv_reclaim(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
        config: IdleKvReclaimConfig,
    ) -> Result<Self> {
        if options.hybrid_cuda_numeric_fp8().is_some()
            || !resources.spec().attachments().is_empty()
            || resources.spec().layers().is_empty()
            || options
                .active_layers()
                .is_some_and(|n| n != resources.spec().layers().len())
            || resources.spec().layers().iter().any(|l| {
                !matches!(l.attention(), Attention::Gqa(_))
                    || !matches!(l.feed_forward(), FeedForward::SwiGlu(_))
            })
        {
            return Err(unsupported(
                "idle KV reclaim requires full-depth local dense F32 all-GQA; no numeric/GDN/MoE/parallel",
            ));
        }
        let mut estimate = Self::for_resources(resources, options, kv_pages)?;
        let schema =
            HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?
                .kv_planes(options.page_size(), options.max_positions())?;
        estimate.kv_bytes = config
            .plan(
                KvLayoutSchema::planes(&schema),
                options.page_size(),
                kv_pages,
            )?
            .device_bytes;
        Ok(estimate)
    }

    /// Root-only budget for an external routed attachment. Expert payload/cache
    /// and expert matmul scratch belong to external owners. Numeric static
    /// workspace (including a configured 64 MiB reservation) is still charged.
    pub fn for_resources_with_routed_experts(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        Self::for_resources_inner(resources, options, kv_pages, true)
    }

    /// Metadata-only dense F32 TP1/EP1 stage estimate. `kv_pages` is the physical
    /// capacity (including CUDA shadows), not the logical page count. Endpoint
    /// storage is charged independently for each owner, including tied heads.
    /// State capacity and colocated-device aggregation belong to the PP planner.
    pub fn for_segment(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        plan: &LayerSegmentPlan,
        kv_pages: usize,
    ) -> Result<Self> {
        if kv_pages == 0 {
            return Err(error("hybrid KV page capacity must be nonzero"));
        }
        let estimate = HybridSegmentMemoryEstimate::for_cuda_segment(resources, options, plan)?;
        Ok(Self {
            per_sequence_state_bytes: estimate.per_sequence_state_bytes,
            kv_bytes: mul(estimate.per_page_kv_bytes, kv_pages)?,
            weight_bytes_upper_bound: estimate.device.weight_bytes()?,
            workspace_bytes_upper_bound: estimate.device.workspace_bytes,
        })
    }

    fn for_resources_inner(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
        external: bool,
    ) -> Result<Self> {
        if options
            .hybrid_cuda_numeric_fp8_precision()
            .is_some_and(crate::transformer::NumericFp8Precision::has_bf16_model_boundaries)
        {
            if external {
                return Err(unsupported("numeric BF16 requires local typed experts"));
            }
            return Self::numeric_bf16_resources(resources, options, kv_pages);
        }
        options.validate_hybrid_cuda_profile(resources)?;
        if kv_pages == 0 {
            return Err(error("hybrid KV page capacity must be nonzero"));
        }
        let spec = resources.spec();
        let layers = options.active_layers().unwrap_or(spec.layers().len());
        let schema = HybridStateSchema::from_spec(spec, layers)?;
        validate_state_abi(&schema)?;
        let planes = schema.kv_planes(options.page_size(), options.max_positions())?;
        let kv_bytes = KvLayoutSchema::checked_page_bytes(&planes)
            .and_then(|n| n.checked_mul(kv_pages))
            .ok_or_else(|| error("CUDA KV budget overflow"))?;
        let numeric = options.hybrid_cuda_numeric_fp8();
        let numeric_precision = options.hybrid_cuda_numeric_fp8_precision();
        let max_rows = options.capabilities().max_batch_tokens;
        let mut weights = 0;
        let mut experts = BTreeMap::<(usize, usize), usize>::new();
        let mut operation_scratch = 0usize;
        let mut width = spec.hidden_size().max(spec.vocab_size());
        for p in resources.state_dict().parameters() {
            if matches!(p.residency(), ParameterResidency::Layer { layer } | ParameterResidency::Expert { layer, .. } if *layer >= layers)
            {
                continue;
            }
            if external && matches!(p.residency(), ParameterResidency::Expert { .. }) {
                if p.id() != p.canonical_id() {
                    return Err(unsupported("bounded numeric expert aliases unsupported"));
                }
                continue;
            }
            let elements = p
                .weight()
                .slice()
                .shape
                .iter()
                .try_fold(1usize, |n, d| mul(n, *d))?;
            if numeric.is_some()
                && numeric_linear_role(p.role())
                && !matches!(p.residency(), ParameterResidency::Expert { .. })
                && let [n, k] = p.weight().slice().shape.as_slice()
            {
                // Include activation, result and possible broadcast bias/IDs.
                operation_scratch = operation_scratch.max(mul(
                    mul(
                        add(add(mul(*n, 2)?, *k)?, 1)?,
                        options.capabilities().max_batch_tokens,
                    )?,
                    4,
                )?);
            }
            let bytes = if let Some(encoding) = p.numeric_fp8_encoding() {
                let source = p.numeric_fp8_source(encoding)?;
                let [n, k] = source.matrix_shape();
                // Unique top-k routes can send every input row to one expert.
                // Validate the full bucket, not the old one-row execution plan.
                NumericFp8LinearPlan::new(
                    NumericFp8Layout {
                        n,
                        k,
                        row_origin: 0,
                        column_origin: 0,
                        scale_type: if source.scale().dtype == CheckpointDType::Bf16 {
                            NumericFp8ScaleType::Bf16
                        } else {
                            NumericFp8ScaleType::F32
                        },
                    },
                    max_rows,
                    numeric.expect("validated numeric profile").1,
                    numeric_precision
                        .expect("validated numeric precision")
                        .backend(),
                )?;
                usize::try_from(
                    source
                        .weight()
                        .bytes
                        .checked_add(source.scale().bytes)
                        .ok_or_else(|| error("numeric bytes overflow"))?,
                )
                .map_err(|_| error("numeric bytes overflow"))?
            } else {
                // Legacy vector/matrix alias duplication remains conservatively
                // covered. Expert payloads have exactly one F32 binding per role.
                mul(
                    elements,
                    if matches!(p.residency(), ParameterResidency::Expert { .. }) {
                        4
                    } else {
                        8
                    },
                )?
            };
            if numeric.is_some()
                && let ParameterResidency::Expert { layer, expert } = p.residency()
            {
                if p.id() != p.canonical_id() {
                    return Err(unsupported("bounded numeric expert aliases unsupported"));
                }
                let entry = experts.entry((*layer, *expert)).or_default();
                *entry = add(*entry, bytes)?;
            } else {
                weights = add(weights, bytes)?;
            }
        }
        let mut scores = 0;
        let mut expert_peak = 0;
        for layer in &spec.layers()[..layers] {
            match layer.attention() {
                Attention::Gqa(g) => {
                    width = width.max(g.query().out_features());
                    weights = add(
                        weights,
                        mul(
                            mul(options.max_positions(), g.rotary().region().dimensions())?,
                            4,
                        )?,
                    )?;
                    scores = scores.max(mul(g.num_heads(), options.max_positions())?);
                }
                Attention::GatedDeltaNet(d) => {
                    width = width.max(d.conv_dim()).max(d.z().out_features());
                }
                _ => return Err(unsupported("CUDA hybrid supports GQA/GatedDeltaNet only")),
            }
            match layer.feed_forward() {
                FeedForward::SwiGlu(ff) => {
                    width = width.max(ff.gate().out_features());
                    operation_scratch = operation_scratch
                        .max(SwiGluScratchPlan::for_descriptor(ff, max_rows)?.total_bytes()?);
                }
                FeedForward::Moe(m) => {
                    width = width.max(m.router_spec().num_experts());
                    if !external {
                        width = width.max(m.expert().gate().out_features());
                    }
                    if let Some(shared) = m.shared_expert() {
                        width = width.max(shared.gate().out_features());
                        operation_scratch = operation_scratch.max(
                            SwiGluScratchPlan::for_descriptor(shared, max_rows)?.total_bytes()?,
                        );
                    }
                    let h = spec.hidden_size();
                    let route = SwiGluScratchPlan::route(
                        max_rows,
                        m.router_spec().experts_per_token(),
                        h,
                        max_rows,
                    )?;
                    let route_bytes = route.total_bytes()?;
                    if external {
                        // Root gather/result-combine still needs bounded storage.
                        operation_scratch = operation_scratch.max(route_bytes);
                        continue;
                    }
                    let intermediate = m.expert().gate().out_features();
                    if mul(max_rows, intermediate)? > i32::MAX as usize {
                        return Err(error("expert bucket exceeds i32 ABI"));
                    }
                    let expert = SwiGluScratchPlan::for_descriptor(m.expert(), max_rows)?
                        .add(route)?
                        .total_bytes()?;
                    let largest = experts
                        .range((layer.index(), 0)..=(layer.index(), usize::MAX))
                        .map(|(_, bytes)| *bytes)
                        .max()
                        .unwrap_or(0);
                    expert_peak = expert_peak.max(add(largest, expert)?);
                }
            }
        }
        let mut workspace = mul(
            mul(
                add(mul(width, 32)?, scores)?,
                options.capabilities().max_batch_tokens,
            )?,
            4,
        )?;
        workspace = add(
            workspace,
            HybridSegmentMemoryEstimate::recurrent_workspace_bytes(
                &spec.layers()[..layers],
                max_rows,
            )?,
        )?;
        if let Some((limits, scratch)) = numeric {
            if SwiGluScratchPlan::reserved_bytes(expert_peak.max(operation_scratch), 0, scratch)?
                > limits.max_bytes
            {
                return Err(error(format!(
                    "numeric admission requires workspace={scratch} + max(expert weights + bucket route/activation={expert_peak}, non-expert temporaries={operation_scratch}), cache limit={}",
                    limits.max_bytes
                )));
            }
            if external {
                workspace = add(workspace, operation_scratch)?;
            } else {
                weights = add(weights, limits.max_bytes - scratch)?;
            }
            workspace = add(workspace, scratch)?;
        }
        Ok(Self {
            per_sequence_state_bytes: schema.per_sequence_state_bytes()?,
            kv_bytes,
            weight_bytes_upper_bound: weights,
            workspace_bytes_upper_bound: workspace,
        })
    }
}

impl HybridSegmentMemoryEstimate {
    /// Dense segment ledger with the existing CUDA profile/ABI checks. No GPU
    /// context is created; host prepared bindings remain a separate charge.
    pub fn for_cuda_segment(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        plan: &LayerSegmentPlan,
    ) -> Result<Self> {
        options.validate_hybrid_cuda_profile(resources)?;
        let estimate = Self::for_segment(resources, options, plan)?;
        validate_state_abi(estimate.schema())?;
        Ok(estimate)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/decoder/hybrid_memory_plan.rs"]
mod tests;
