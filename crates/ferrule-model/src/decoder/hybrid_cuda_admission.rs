//! Metadata-only hybrid profile validation and compressed-aware admission.
use super::*;
use crate::checkpoint::CheckpointDType;
use crate::nn::ParameterResidency;
use crate::support::TensorRole;
use crate::transformer::{RouterScoreFunction, RouterSelection, SwiGluScratchPlan};
use ferrule_backend::cuda::operators::linear::{
    NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType,
};
use std::collections::BTreeMap;

impl GenericDecoderOptions {
    pub(crate) fn validate_hybrid_cuda(&self, resources: &BoundDecoderResources) -> Result<()> {
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
fn numeric_linear_role(role: &TensorRole) -> bool {
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

impl HybridCudaMemoryEstimate {
    /// No weight payload reads or CUDA allocations. Numeric experts are charged
    /// by their finite cache ceiling, not by the complete checkpoint size.
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        Self::for_resources_inner(resources, options, kv_pages, false)
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

    fn for_resources_inner(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
        external: bool,
    ) -> Result<Self> {
        options.validate_hybrid_cuda(resources)?;
        if kv_pages == 0 {
            return Err(error("hybrid KV page capacity must be nonzero"));
        }
        let spec = resources.spec();
        let layers = options.active_layers().unwrap_or(spec.layers().len());
        let schema = HybridStateSchema::from_spec(spec, layers)?;
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
            per_sequence_state_bytes: state_bytes(&schema)?,
            kv_bytes,
            weight_bytes_upper_bound: weights,
            workspace_bytes_upper_bound: workspace,
        })
    }
}

#[cfg(test)]
mod bucket_admission_tests {
    use super::*;
    use crate::transformer::CudaStandardDecoderOperators;

    #[test]
    fn qwen_bucket_table_and_gpu_indices_are_charged_exactly() {
        let route = CudaStandardDecoderOperators::routed_bucket_scratch_bytes;
        let rows = 32;
        let top_k = 8;
        let hidden = 2048;
        let table = rows * top_k * hidden * 4;
        assert_eq!(table, 2 * 1024 * 1024);
        let expected = table + 2 * rows * hidden * 4 + (2 * rows * top_k + 2 * rows) * 4;
        assert_eq!(route(rows, top_k, hidden, rows).unwrap(), expected);
        assert_eq!(
            route(rows, top_k, hidden, 7).unwrap(),
            expected - 2 * (rows - 7) * 4
        );
        assert_eq!(
            route(rows, top_k + 1, hidden, rows).unwrap() - expected,
            rows * (hidden + 2) * 4
        );
    }

    #[test]
    fn bucket_geometry_overflow_and_i32_abi_fail_without_cuda() {
        let route = CudaStandardDecoderOperators::routed_bucket_scratch_bytes;
        for (rows, top_k, hidden, bucket) in [
            (usize::MAX, 2, 1, 0),
            (2, 1, usize::MAX, 0),
            (i32::MAX as usize, 2, 1, 0),
            (1, 1, i32::MAX as usize + 1, 0),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 1, 8, 2),
        ] {
            assert!(route(rows, top_k, hidden, bucket).is_err());
        }
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 4).is_err());
    }
}
