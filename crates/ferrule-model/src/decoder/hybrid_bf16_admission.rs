//! Opt-in native BF16 hybrid image admission. No numeric/TP/HostKV policy changes.
use super::*;
use ferrule_backend::cuda::operators::linear::{Bf16GemmLayout, bf16_gemm_workspace_requirements};
use ferrule_common::execution::KvElementType;

impl GenericDecoderOptions {
    pub(super) fn validate_bf16_hybrid(&self, resources: &BoundDecoderResources) -> Result<()> {
        self.validate(resources)?;
        if self
            .active_layers()
            .is_some_and(|n| n != resources.spec().layers().len())
        {
            return Err(unsupported(
                "native BF16 hybrid requires complete image; numeric profiles remain model-F32",
            ));
        }
        if self
            .hybrid_cuda_numeric_fp8_precision()
            .is_some_and(crate::transformer::NumericFp8Precision::has_bf16_model_boundaries)
        {
            crate::transformer::NumericFp8Bf16HybridCudaProfile::validate(resources)?;
        } else {
            if self.hybrid_cuda_numeric_fp8().is_some() {
                return Err(unsupported(
                    "model-F32 numeric profile cannot select BF16 boundaries",
                ));
            }
            crate::transformer::Bf16HybridCudaProfile::validate(resources)?;
        }
        Ok(())
    }
}
impl HybridCudaMemoryEstimate {
    pub(super) fn bf16_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
        external: bool,
    ) -> Result<Self> {
        options.validate_bf16_hybrid(resources)?;
        if external || kv_pages == 0 {
            return Err(unsupported(
                "native BF16 hybrid requires resident local experts and nonzero KV capacity",
            ));
        }
        if options.hybrid_cuda_numeric_fp8().is_some() {
            return HybridMemoryPlan::for_resources(resources, options, kv_pages).map(Into::into);
        }
        let spec = resources.spec();
        let schema = HybridStateSchema::from_spec(spec, spec.layers().len())?;
        let planes = schema.kv_planes_with_dtype(
            options.page_size(),
            options.max_positions(),
            KvElementType::Bf16,
        )?;
        let kv_bytes = planes
            .checked_page_bytes()
            .and_then(|n| n.checked_mul(kv_pages))
            .ok_or_else(|| error("BF16 KV budget overflow"))?;
        let mut weights = 0usize;
        let mut width = spec.hidden_size().max(spec.vocab_size());
        let mut unique = std::collections::BTreeSet::new();
        for p in resources.state_dict().parameters() {
            let shape = &p.weight().slice().shape;
            let elements = shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d)
                    .ok_or_else(|| error("BF16 weight size overflow"))
            })?;
            if unique.insert(p.canonical_id()) {
                weights = add(weights, mul(elements, 2)?)?;
            }
            if shape.len() == 2 && p.role() != &crate::support::TensorRole::LinearAttentionConv {
                width = width.max(shape[0]).max(shape[1]);
                if p.role() != &crate::support::TensorRole::TokenEmbedding {
                    bf16_gemm_workspace_requirements(Bf16GemmLayout::contiguous(
                        options.capabilities().max_batch_tokens,
                        shape[0],
                        shape[1],
                    ))?;
                }
            }
        }
        let mut top_k = 1;
        let mut largest_history = 0;
        for layer in spec.layers() {
            match layer.attention() {
                Attention::Gqa(g) => {
                    // Preparation can retain distinct tables per descriptor; overcharge sharing.
                    weights = add(
                        weights,
                        mul(
                            mul(options.max_positions(), g.rotary().region().dimensions())?,
                            2,
                        )?,
                    )?;
                    width = width.max(g.query().out_features());
                }
                Attention::GatedDeltaNet(g) => {
                    width = width.max(g.qkv().out_features()).max(g.z().out_features());
                    largest_history =
                        largest_history.max(mul(g.qkv().out_features(), g.conv_kernel_dim())?);
                }
                _ => unreachable!("validated attention"),
            }
            if let FeedForward::Moe(m) = layer.feed_forward() {
                top_k = top_k.max(m.router_spec().experts_per_token());
                width = width.max(m.router_spec().num_experts());
            }
        }
        // Conservative simultaneous live rows: residual/QKV/projections, BF16
        // recurrent scratch and F32 exact packing/norm/GEMM/export temporaries.
        // Full route concat is 4B work + 2B BF16 plus 8B metadata per route.
        let rows = options.capabilities().max_batch_tokens;
        let activations = mul(mul(rows, width)?, 128)?;
        let route = mul(mul(rows, top_k)?, add(mul(spec.hidden_size(), 6)?, 8)?)?;
        let history_copy = mul(largest_history, 2)?;
        let workspace = add(add(activations, route)?, history_copy)?;
        // Account for allocator fragmentation/transfer staging without relabeling
        // logical state bytes. State capacity includes all live checkouts/forks.
        let workspace = add(
            workspace,
            add(
                mul(resources.state_dict().parameters().len(), 256)?,
                64 << 20,
            )?,
        )?;
        Ok(Self {
            per_sequence_state_bytes: bf16_state::state_bytes(&schema)?,
            kv_bytes,
            weight_bytes_upper_bound: weights,
            workspace_bytes_upper_bound: add(workspace, add(weights / 10, kv_bytes / 10)?)?,
        })
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("BF16 budget overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("BF16 budget overflow"))
}
