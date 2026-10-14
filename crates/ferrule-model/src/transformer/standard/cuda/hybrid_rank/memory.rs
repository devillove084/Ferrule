//! Host-only local geometry and physical grant admission. Pool slack and external
//! EP owners remain separate device-placement charges, never logical KV pages.
use super::*;
use crate::checkpoint::CheckpointDType;
use crate::transformer::segment_inventory::HybridSegmentInventory;
use ferrule_common::execution::KvLayoutSchema;

#[derive(Debug, Clone)]
pub struct CudaHybridRankMemory {
    pub precision: crate::transformer::NumericFp8Precision,
    pub numeric_scratch_limit: usize,
    pub weight_grants: usize,
    pub numeric_workspace_grants: usize,
    pub operation_workspace_grants: usize,
    pub per_sequence_state_bytes: usize,
    pub per_sequence_state_grants: usize,
    pub physical_kv_grants: usize,
    pub physical_pages: usize,
    /// Largest model-owned collective contribution; transport must admit this.
    pub max_collective_elements: usize,
    /// Two independent bounded pinned boundaries (root and external EP).
    pub pinned_host_bytes: usize,
    pub recurrent_workspace_grants: usize,
    pub gqa_metadata_grants: usize,
    pub external_hold_grants: usize,
    pub output_workspace_grants: usize,
    pub page_size: usize,
    pub max_positions: usize,
    pub max_rows: usize,
    pub max_sequences: usize,
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| cuda_error("rank budget overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| cuda_error("rank budget overflow"))
}
fn grant(n: usize) -> Result<usize> {
    ferrule_backend::cuda_allocation_grant_bytes(n)
        .ok_or_else(|| cuda_error("rank physical grant overflow"))
}
impl CudaHybridRankMemory {
    pub fn for_rank(
        resources: &BoundDecoderResources,
        segment: &LayerSegmentPlan,
        parallel: &HybridParallelPlan,
        layout: &HybridRankLayout,
        options: &GenericDecoderOptions,
        physical_pages: usize,
    ) -> Result<Self> {
        options.validate(resources)?;
        parallel.validate_spec(resources.spec())?;
        if !matches!(parallel.tensor_members().len(), 1 | 2 | 4)
            || parallel.rank_layout(layout.lane())? != *layout
            || segment.layers() != layout.state_schema().layer_range()
            || physical_pages == 0
            || options.precision() != ExecutionPrecisionPolicy::f32()
            || !resources.spec().attachments().is_empty()
            || options
                .active_layers()
                .is_some_and(|n| n != resources.spec().layers().len())
        {
            return Err(cuda_error(
                "CUDA rank model/layout/options/physical capacity mismatch",
            ));
        }
        parallel.collective_schedule(segment)?;
        super::super::super::validate_hybrid_descriptors(
            resources.spec(),
            &resources.spec().layers()[segment.layers()],
        )?;
        let precision = options
            .hybrid_cuda_numeric_fp8_precision()
            .unwrap_or(crate::transformer::NumericFp8Precision::F32Tf32x3);
        if !matches!(
            precision,
            crate::transformer::NumericFp8Precision::F32Tf32x3
                | crate::transformer::NumericFp8Precision::W8A16Bf16F32Accumulate
        ) {
            return Err(cuda_error(
                "CUDA rank admits explicit F32(default) or W8A16; no precision fallback",
            ));
        }
        let scratch = options
            .hybrid_cuda_numeric_fp8()
            .map_or(64 << 20, |(_, n)| n);
        let rows = options.capabilities().max_batch_tokens;
        let inventory = HybridSegmentInventory::new(resources, segment)?;
        let mut weights = 0;
        let mut workspaces = 0;
        let mut width = resources.spec().hidden_size();
        for binding in inventory.parameters() {
            if matches!(binding.residency(), ParameterResidency::Expert { .. }) {
                continue;
            }
            let replicated = matches!(
                binding.role(),
                TensorRole::TokenEmbedding
                    | TensorRole::OutputNorm
                    | TensorRole::AttentionNorm
                    | TensorRole::FeedForwardNorm
                    | TensorRole::AttentionQueryNorm
                    | TensorRole::AttentionKeyNorm
                    | TensorRole::LinearAttentionNorm
            );
            if binding.numeric_fp8_encoding().is_some() {
                if replicated {
                    return Err(cuda_error("numeric vectors/embeddings unsupported"));
                }
                let plan = NumericFp8RankPlan::new(binding, Arc::new(layout.clone()))?;
                if plan.budget().payload_bytes as u64 > options.max_parameter_bytes() {
                    return Err(cuda_error(
                        "rank paired parameter exceeds local read budget",
                    ));
                }
                let budget = plan.cuda_execution_budget(rows, scratch, precision)?;
                weights = add(weights, plan.budget().allocation_grant_bytes)?;
                workspaces = add(workspaces, budget.workspace_grant_bytes)?;
                if binding.role() != &TensorRole::OutputHead {
                    width = width.max(plan.local_shape()[0]).max(plan.local_shape()[1]);
                }
            } else {
                if binding.scale().is_some()
                    || binding.weight().transform()
                        != &crate::transformer::TensorTransform::Identity
                    || !matches!(
                        binding.weight().slice().dtype,
                        CheckpointDType::F32 | CheckpointDType::Bf16
                    )
                {
                    return Err(cuda_error(
                        "rank storage requires identity F32/BF16 or numeric FP8",
                    ));
                }
                let bytes = if replicated {
                    mul(binding.weight().slice().element_count()?, 4)?
                } else {
                    let partition = layout.parameter_partition(binding)?;
                    let shape = partition.local_shape()?;
                    if binding.role() != &TensorRole::OutputHead {
                        width = width.max(shape[0]).max(shape[1]);
                    }
                    partition.bytes_f32()?
                };
                if bytes as u64 > options.max_parameter_bytes() {
                    return Err(cuda_error("rank dense parameter exceeds local read budget"));
                }
                weights = add(weights, grant(bytes)?)?;
            }
        }
        for layer in &resources.spec().layers()[segment.layers()] {
            let projections: Vec<&crate::transformer::Linear> = match layer.attention() {
                crate::transformer::Attention::Gqa(a) => {
                    super::super::validate_rope(a.rotary())?;
                    let bytes = mul(
                        mul(
                            options.max_positions(),
                            a.rotary().region().dimensions() / 2,
                        )?,
                        4,
                    )?;
                    weights = add(weights, mul(grant(bytes)?, 2)?)?;
                    vec![a.query(), a.key(), a.value(), a.output()]
                }
                crate::transformer::Attention::GatedDeltaNet(a) => {
                    vec![a.qkv(), a.z(), a.a(), a.beta(), a.output()]
                }
                _ => return Err(cuda_error("rank attention unsupported")),
            };
            if projections.iter().any(|p| p.has_bias()) {
                return Err(cuda_error("biased rank attention unsupported"));
            }
            let mlp = match layer.feed_forward() {
                crate::transformer::FeedForward::SwiGlu(m) => Some(m),
                crate::transformer::FeedForward::Moe(m) => {
                    let router = m.router_spec();
                    if m.router().has_bias()
                        || router.score_function()
                            != crate::transformer::RouterScoreFunction::Softmax
                        || router.selection() != &crate::transformer::RouterSelection::TopK
                        || !router.normalize_selected()
                        || router.selection_bias()
                    {
                        return Err(cuda_error(
                            "rank EP requires normalized softmax top-k router",
                        ));
                    }
                    m.shared_expert()
                }
            };
            if mlp.is_some_and(|m| [m.gate(), m.up(), m.down()].iter().any(|p| p.has_bias())) {
                return Err(cuda_error("biased rank MLP unsupported"));
            }
        }
        if resources.spec().output().has_bias() {
            return Err(cuda_error("biased rank head unsupported"));
        }
        let mut states = 0;
        let mut recurrent_capacity = [0usize; 8];
        for layer in layout.state_schema().layers() {
            if let HybridLayerSchema::GatedDeltaNet(shape) = layer {
                let (channels, conv, recurrent) = shape.sizes()?;
                ferrule_backend::cuda::operators::recurrent::GatedDeltaNetLayout {
                    rows,
                    key_heads: shape.key_heads,
                    value_heads: shape.value_heads,
                    key_dim: shape.key_dim,
                    value_dim: shape.value_dim,
                }
                .validate()?;
                width = width.max(channels);
                let scratch_plan =
                    crate::transformer::segment_inventory::RecurrentScratchPlan::new(
                        rows,
                        channels,
                        shape.value_heads,
                        mul(shape.value_heads, shape.value_dim)?,
                    )?;
                for (capacity, required) in recurrent_capacity.iter_mut().zip(scratch_plan.elements)
                {
                    *capacity = (*capacity).max(required);
                }
                states = add(
                    states,
                    add(grant(mul(conv, 4)?)?, grant(mul(recurrent, 4)?)?)?,
                )?;
            }
        }
        let planes = layout.kv_planes(options.page_size(), options.max_positions())?;
        // Pool stores separate K and V roots; the two physical plane grants are
        // rounded independently, not after summing logical page bytes.
        let per_page = KvLayoutSchema::checked_page_bytes(&planes)
            .ok_or_else(|| cuda_error("rank KV byte overflow"))?;
        let kv = mul(grant(mul(per_page / 2, physical_pages)?)?, 2)?;
        let recurrent_workspace_grants = mul(
            recurrent_capacity
                .into_iter()
                .try_fold(0, |sum, n| add(sum, grant(mul(n, 4)?)?))?,
            2,
        )?;
        // The existing paged F32 kernel allocates six integer roots, not a
        // dense [rows,heads,context] score matrix. Shared prefix pages can appear
        // in multiple sequence tables; bound table entries per sequence.
        let sequences = options.capabilities().max_sequences;
        let slots = mul(
            sequences,
            options.max_positions().div_ceil(options.page_size()),
        )?;
        let gqa_metadata_grants = add(
            add(grant(mul(slots, 4)?)?, grant(mul(add(sequences, 1)?, 4)?)?)?,
            add(mul(grant(mul(rows, 4)?)?, 3)?, grant(4)?)?,
        )?;
        // Conservative live row-root envelope for the shared loop (including
        // ragged vocabulary gather and segmented projection pack), NOT an
        // allocator reservation. Persistent/growth recurrence is separate.
        let routed_layers = parallel
            .layers()
            .iter()
            .filter(|layer| layer.routed_experts() > 0)
            .count();
        let external_hold_grants = if parallel
            .expert_attachment()
            .is_some_and(|ep| ep.source_rank == layout.owner())
        {
            mul(
                mul(routed_layers, 2)?,
                grant(mul(mul(rows, resources.spec().hidden_size())?, 4)?)?,
            )?
        } else {
            0
        };
        // Vocabulary outputs are separate from the layer-loop row envelope:
        // charging every layer temporary at vocabulary width makes otherwise
        // usable large-vocabulary ranks impossible to admit. Four roots bound
        // the local output/segment pack, gathered upload, and returned logits.
        let output_workspace_grants = if segment.owns_output() {
            mul(
                grant(mul(
                    mul(
                        rows,
                        add(
                            resources.spec().vocab_size(),
                            parallel.tensor_members().len(),
                        )?,
                    )?,
                    4,
                )?)?,
                4,
            )?
        } else {
            0
        };
        let operation = add(
            add(
                mul(grant(mul(mul(rows, width)?, 4)?)?, 64)?,
                add(recurrent_workspace_grants, gqa_metadata_grants)?,
            )?,
            add(external_hold_grants, output_workspace_grants)?,
        )?;
        let max_collective_elements = parallel
            .collective_schedule(segment)?
            .sites()
            .iter()
            .try_fold(0usize, |n, site| {
                Ok::<_, Error>(n.max(site.shape.elements(sequences, rows)?))
            })?;
        let config = crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG;
        let pinned_host_bytes = mul(
            mul(
                config.chunk_elements,
                add(config.tx_slots, config.rx_slots)?,
            )?,
            8,
        )?;
        Ok(Self {
            precision,
            numeric_scratch_limit: scratch,
            weight_grants: weights,
            numeric_workspace_grants: workspaces,
            operation_workspace_grants: operation,
            per_sequence_state_bytes: layout.state_bytes()?,
            per_sequence_state_grants: states,
            physical_kv_grants: kv,
            physical_pages,
            max_collective_elements,
            pinned_host_bytes,
            recurrent_workspace_grants,
            gqa_metadata_grants,
            external_hold_grants,
            output_workspace_grants,
            page_size: options.page_size(),
            max_positions: options.max_positions(),
            max_rows: rows,
            max_sequences: options.capabilities().max_sequences,
        })
    }
    /// Shared factory admission units. Rank state uses allocator grants rather
    /// than logical tensor bytes; the same exclusive claim covers all owners.
    pub fn admission_estimate(&self) -> Result<crate::decoder::HybridCudaMemoryEstimate> {
        Ok(crate::decoder::HybridCudaMemoryEstimate {
            per_sequence_state_bytes: self.per_sequence_state_grants,
            kv_bytes: self.physical_kv_grants,
            weight_bytes_upper_bound: self.weight_grants,
            workspace_bytes_upper_bound: add(
                self.numeric_workspace_grants,
                self.operation_workspace_grants,
            )?,
        })
    }
    pub fn total_device_grants(&self, sequences: usize) -> Result<usize> {
        add(
            add(
                self.weight_grants,
                add(
                    self.numeric_workspace_grants,
                    self.operation_workspace_grants,
                )?,
            )?,
            add(
                self.physical_kv_grants,
                mul(add(mul(sequences, 2)?, 1)?, self.per_sequence_state_grants)?,
            )?,
        )
    }
    pub fn validate_device_budget(
        &self,
        budget: crate::decoder::HybridCudaMemoryBudget,
        sequences: usize,
    ) -> Result<()> {
        if budget.kv_pages != self.physical_pages
            || sequences != self.max_sequences
            || self.weight_grants > budget.weight_bytes
            || add(
                self.numeric_workspace_grants,
                self.operation_workspace_grants,
            )? > budget.workspace_bytes
            || mul(add(mul(sequences, 2)?, 1)?, self.per_sequence_state_grants)?
                > budget.state_bytes
        {
            return Err(cuda_error(
                "rank local physical grants exceed device budget",
            ));
        }
        Ok(())
    }
}
