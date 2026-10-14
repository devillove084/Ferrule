//! Pure Mixed rank-program resource estimation. No payload reads or CUDA owners.
use super::*;
use crate::TensorRole;
#[cfg(feature = "cuda")]
use crate::decoder::{HybridCudaMemoryEstimate, HybridCudaPreparationContext};
use crate::transformer::segment_inventory::HybridSegmentInventory;
use crate::transformer::{BoundDecoderResources, NumericFp8RankPlan};
use ferrule_backend::cuda_plan::bf16_attention::{Bf16AttentionPlan, Bf16AttentionShape};
use ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan;
use ferrule_common::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridMixedRankOperationEstimate {
    pub storage_grants: usize,
    pub gemmex_storage_grants: usize,
    pub gemmex_provider_bytes: usize,
    pub attention_storage_grants: usize,
    pub attention_provider_bytes: usize,
    pub derived_rope_grants: usize,
    pub attention_metadata_grants: usize,
    pub recurrent_workspace_grants: usize,
    pub state_copy_workspace_grants: usize,
    pub operation_grants: usize,
    pub boundary_device_grants: usize,
    pub pinned_host_bytes: usize,
    pub host_staging_bytes: usize,
}
impl HybridMixedRankOperationEstimate {
    pub fn total_device_grants(self) -> Result<usize> {
        [
            self.storage_grants,
            self.gemmex_storage_grants,
            self.attention_storage_grants,
            self.gemmex_provider_bytes,
            self.attention_provider_bytes,
            self.derived_rope_grants,
            self.attention_metadata_grants,
            self.recurrent_workspace_grants,
            self.state_copy_workspace_grants,
            self.operation_grants,
            self.boundary_device_grants,
        ]
        .into_iter()
        .try_fold(0usize, |n, v| {
            n.checked_add(v)
                .ok_or_else(|| invalid("Mixed estimate overflow"))
        })
    }
    /// External provider headroom is included, but driver/context and allocator
    /// reserve margins belong to the owner and must be supplied explicitly.
    pub fn require_device_capacity(self, capacity: usize, owner_margins: usize) -> Result<usize> {
        let total = add(self.total_device_grants()?, owner_margins)?;
        if total > capacity {
            return Err(invalid(
                "Mixed complete rank estimate exceeds device capacity",
            ));
        }
        Ok(total)
    }
}
fn grant(bytes: usize) -> Result<usize> {
    if bytes == 0 {
        return Ok(0);
    }
    ferrule_backend::cuda_allocation_grant_bytes(bytes)
        .ok_or_else(|| invalid("Mixed estimate grant overflow"))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Mixed estimate overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Mixed estimate overflow"))
}

pub(super) fn mixed_projection_role(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::AttentionQuery
            | TensorRole::AttentionKey
            | TensorRole::AttentionValue
            | TensorRole::AttentionOutput
            | TensorRole::LinearAttentionQkv
            | TensorRole::LinearAttentionZ
            | TensorRole::LinearAttentionA
            | TensorRole::LinearAttentionBeta
            | TensorRole::DenseMlpGate
            | TensorRole::DenseMlpUp
            | TensorRole::DenseMlpDown
            | TensorRole::RouterLogits
            | TensorRole::SharedExpertOutputGate
            | TensorRole::SharedExpertGate
            | TensorRole::SharedExpertUp
            | TensorRole::SharedExpertDown
            | TensorRole::RoutedExpertGate
            | TensorRole::RoutedExpertUp
            | TensorRole::RoutedExpertDown
            | TensorRole::OutputHead
    )
}

impl HybridMixedRankProgramPlan {
    /// Build the exact claim input consumed by Mixed CUDA preparation. Storage
    /// state/KV/weights come from the authoritative rank ledger; every other
    /// model-owned grant is one compiled-program workspace pool.
    #[cfg(feature = "cuda")]
    pub fn cuda_claim_inputs(
        &self,
        resources: &BoundDecoderResources,
        physical_pages: usize,
        device_ordinal: usize,
        max_parameter_bytes: u64,
        attention_shapes: &[Bf16AttentionShape],
    ) -> Result<(HybridCudaMemoryEstimate, HybridCudaPreparationContext)> {
        let estimate = self.operation_estimate(resources, physical_pages)?;
        let storage = self.layout.storage_budget(
            resources,
            &self.segment,
            self.page_size,
            self.max_positions,
            physical_pages,
            self.state_copies,
        )?;
        let expected_copies = self
            .max_sequences
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("Mixed claim state-copy overflow"))?;
        if self.state_copies != expected_copies
            || estimate.storage_grants != storage.storage_grants()?
        {
            return Err(invalid(
                "Mixed claim context differs from compiled storage ledger",
            ));
        }
        let attention_limits = [
            estimate.attention_storage_grants,
            estimate.attention_provider_bytes,
        ];
        if self.gqa_layer().is_some() {
            self.attention_plan(attention_shapes, attention_limits[0], attention_limits[1])?;
        } else if !attention_shapes.is_empty() {
            return Err(invalid(
                "recurrent-only claim must not reserve attention views",
            ));
        }
        if max_parameter_bytes == 0
            || storage
                .parameters
                .iter()
                .any(|p| p.source_bytes as u64 > max_parameter_bytes)
        {
            return Err(invalid(
                "Mixed claim parameter read limit exhausted before allocation",
            ));
        }
        let projection_grants = storage
            .parameters
            .iter()
            .filter(|p| mixed_projection_role(&p.role))
            .try_fold(0, |n, p| add(n, p.allocation_grant_bytes))?;
        let projection_limits = [
            projection_grants,
            estimate.gemmex_storage_grants,
            estimate.gemmex_provider_bytes,
        ];
        let auxiliary_grants = add(
            storage
                .weight_grants
                .checked_sub(projection_grants)
                .ok_or_else(|| invalid("Mixed vector ledger underflow"))?,
            add(
                estimate.derived_rope_grants,
                add(
                    estimate.recurrent_workspace_grants,
                    estimate.state_copy_workspace_grants,
                )?,
            )?,
        )?;
        let total = estimate.total_device_grants()?;
        let workspace = total
            .checked_sub(estimate.storage_grants)
            .ok_or_else(|| invalid("Mixed claim workspace underflow"))?;
        Ok((
            HybridCudaMemoryEstimate {
                per_sequence_state_bytes: storage.state.per_sequence_grants,
                kv_bytes: storage.kv_grants,
                weight_bytes_upper_bound: storage.weight_grants,
                workspace_bytes_upper_bound: workspace,
            },
            HybridCudaPreparationContext {
                program: self.clone(),
                parameters: HybridSegmentInventory::new(resources, &self.segment)?
                    .parameters()
                    .cloned()
                    .collect(),
                device_ordinal,
                physical_pages,
                max_parameter_bytes,
                estimate: HybridCudaMemoryEstimate {
                    per_sequence_state_bytes: storage.state.per_sequence_grants,
                    kv_bytes: storage.kv_grants,
                    weight_bytes_upper_bound: storage.weight_grants,
                    workspace_bytes_upper_bound: workspace,
                },
                projection_limits,
                auxiliary_grants,
                attention_shapes: attention_shapes.to_vec(),
                attention_limits,
            },
        ))
    }

    /// Internal block candidate envelope. Keep native fallback workspace/provider
    /// reserves; do not claim a decode-only workspace saving for a mixed schedule.
    /// This does not enable the candidate in production preparation/claim inputs.
    #[doc(hidden)]
    pub fn operation_estimate_with_routed_down_m1_candidate(
        &self,
        resources: &BoundDecoderResources,
        physical_pages: usize,
    ) -> Result<HybridMixedRankOperationEstimate> {
        let mut estimate = self.operation_estimate(resources, physical_pages)?;
        estimate.operation_grants = add(
            estimate.operation_grants,
            HybridMixedBlockPlan::routed_down_m1_accumulator_grants(&self.layout)?,
        )?;
        Ok(estimate)
    }

    /// Estimate all model-owned categories before any payload materialization.
    /// Storage ledger categories are included once; state/KV are authoritative
    /// in `HybridRankStorageBudget` and are not re-added as operation scratch.
    pub fn operation_estimate(
        &self,
        resources: &BoundDecoderResources,
        physical_pages: usize,
    ) -> Result<HybridMixedRankOperationEstimate> {
        if self
            .decoded_cache
            .as_ref()
            .is_some_and(|a| a.physical_pages != physical_pages)
        {
            return Err(invalid("decoded cache physical-page admission changed"));
        }
        self.validate_spec(resources.spec())?;
        let segment = self.segment.clone();
        let storage = self.layout.storage_budget(
            resources,
            &segment,
            self.page_size,
            self.max_positions,
            physical_pages,
            self.state_copies,
        )?;
        let inventory = HybridSegmentInventory::new(resources, &segment)?;
        let mut gemmex_storage = 0;
        let mut gemmex_provider = 0;
        let mut packed_qkv_host_bytes = 0;
        for binding in inventory.parameters() {
            if !matches!(binding.weight().logical_shape(), [_, _])
                || !mixed_projection_role(binding.role())
            {
                continue;
            }
            let local = self.layout.parameter_partition(binding)?.local_shape()?;
            let mut plans = Vec::new();
            if let Some(encoding) = binding.numeric_fp8_encoding() {
                let scale_type = match encoding.scale_dtype() {
                    crate::checkpoint::CheckpointDType::Bf16 => {
                        ferrule_backend::cuda_plan::NumericFp8ScaleType::Bf16
                    }
                    crate::checkpoint::CheckpointDType::F32 => {
                        return Err(invalid(
                            "Mixed GemmEx requires BF16 numeric FP8 scales; F32 scale source is not relabelled",
                        ));
                    }
                    _ => return Err(invalid("unsupported numeric FP8 scale dtype")),
                };
                let numeric = NumericFp8RankPlan::new(binding, self.layout.clone())?;
                if let Some(qkv) = numeric.mixed_gemmex_qkv_layout()? {
                    let plan = Bf16GemmExPlan::new(qkv, self.max_rows, usize::MAX, usize::MAX)?;
                    gemmex_storage = gemmex_storage.max(plan.storage_grant_bytes());
                    gemmex_provider = gemmex_provider.max(plan.provider_bytes());
                    // Encoded Vec-to-Arc publication can overlap two packed copies
                    // while original source segments remain owned. No host decode
                    // or device F32 mirror is introduced.
                    packed_qkv_host_bytes =
                        packed_qkv_host_bytes.max(mul(numeric.budget().payload_bytes, 2)?);
                    continue;
                }
                for read in numeric.reads() {
                    let [n, k] = read.local_shape();
                    let layout = HybridMixedProjectionArithmetic::packed_gemmex_layout(
                        ferrule_backend::cuda_plan::NumericFp8Layout {
                            n,
                            k,
                            row_origin: read.weight_read().rows().start,
                            column_origin: read.weight_read().columns().start,
                            scale_type,
                        },
                    )?;
                    plans.push(Bf16GemmExPlan::new(
                        layout,
                        self.max_rows,
                        usize::MAX,
                        usize::MAX,
                    )?);
                    if matches!(binding.role(), TensorRole::RoutedExpertGate) {
                        plans.push(Bf16GemmExPlan::new(
                            ferrule_backend::cuda_plan::NumericFp8Layout {
                                n: mul(n, 2)?,
                                ..layout
                            },
                            self.max_rows,
                            usize::MAX,
                            usize::MAX,
                        )?);
                    }
                }
            } else {
                plans.push(Bf16GemmExPlan::new_resident(
                    local[0],
                    local[1],
                    self.max_rows,
                    usize::MAX,
                    usize::MAX,
                )?);
            }
            for plan in plans {
                gemmex_storage = gemmex_storage.max(plan.storage_grant_bytes());
                gemmex_provider = gemmex_provider.max(plan.provider_bytes());
            }
        }
        let mut attention_storage = 0;
        let mut attention_provider = 0;
        if let Some(layer) = self.gqa_layer() {
            let HybridLocalAttention::Gqa {
                query_heads,
                kv_heads,
                head_dim,
                ..
            } = self.layout.attention(layer)?
            else {
                unreachable!()
            };
            let plan = Bf16AttentionPlan::from_shapes(
                &[Bf16AttentionShape {
                    rows: self.max_rows as u32,
                    tokens: self.max_positions as u32,
                }],
                self.max_rows,
                self.max_sequences,
                query_heads.len(),
                kv_heads.len(),
                *head_dim,
                usize::MAX,
                usize::MAX,
            )?;
            attention_storage = plan.storage_grant_bytes();
            attention_provider = plan.provider_bytes();
        }
        let mut ropes = Vec::new();
        let mut derived_rope_grants = 0;
        let mut layer_peak = 0usize;
        let mut recurrent_peak = 0usize;
        let mut route_peak = 0usize;
        let mut routed_slots_host_bytes = 0usize;
        let mut routed_batch_host_peak = 0usize;
        for layer in self.layers() {
            // Conservatively charge an image table header for every local layer.
            routed_slots_host_bytes =
                add(routed_slots_host_bytes, 7 * std::mem::size_of::<usize>())?;
            let descriptor = &self.layout.spec.layers()[layer];
            let mut width = self.width;
            match descriptor.attention() {
                Attention::Gqa(gqa) => {
                    if !ropes.contains(&gqa.rotary()) {
                        ropes.push(gqa.rotary());
                        let bytes = mul(self.max_positions, gqa.rotary().region().dimensions())?;
                        derived_rope_grants = add(derived_rope_grants, mul(grant(bytes)?, 2)?)?;
                    }
                    let HybridLocalAttention::Gqa {
                        query_heads,
                        head_dim,
                        ..
                    } = self.layout.attention(layer)?
                    else {
                        unreachable!()
                    };
                    width = width.max(mul(
                        mul(query_heads.len(), *head_dim)?,
                        if gqa.gated_query() { 2 } else { 1 },
                    )?);
                }
                Attention::GatedDeltaNet(_) => {
                    let HybridLocalAttention::GatedDelta { shape, .. } =
                        self.layout.attention(layer)?
                    else {
                        unreachable!()
                    };
                    let channels = shape.sizes()?.0;
                    width = width.max(channels);
                    // The chunk operation retains cohort operands through its outer
                    // fence. Sum live cohort elements, plus per-cohort grant rounding;
                    // recurrent state itself is borrowed, not a second F32 mirror.
                    recurrent_peak = recurrent_peak.max(add(
                        mul(grant(mul(mul(self.max_rows, channels)?, 4)?)?, 16)?,
                        mul(self.max_sequences, 16 * 256)?,
                    )?);
                }
                _ => unreachable!(),
            }
            let ff = match descriptor.feed_forward() {
                FeedForward::SwiGlu(m) => m,
                FeedForward::Moe(m) => m.expert(),
            };
            width = width.max(ff.intermediate_size().div_ceil(self.layout.members().len()));
            // Conservative 32-root envelope for hidden/residual, norms, GQA
            // projections/gates, FFN intermediates and segmented BF16/F32 packs.
            // Vocabulary is intentionally excluded from this layer-local peak.
            layer_peak = layer_peak.max(mul(grant(mul(mul(self.max_rows, width)?, 4)?)?, 32)?);
            if let FeedForward::Moe(m) = descriptor.feed_forward() {
                let routes = mul(self.max_rows, m.router_spec().experts_per_token())?;
                let selected = m.router_spec().num_experts().min(routes);
                let inner = m
                    .expert()
                    .intermediate_size()
                    .div_ceil(self.layout.members().len());
                // Retain the conservative legacy envelope: gather, gate, up,
                // SwiGLU, down and exact expansion through the outer fence.
                // Routed BF16 down uses packed-output aliases now; do not spend
                // those savings or shorten activation/cache/workspace lifetimes.
                let payload = add(
                    mul(mul(routes, self.width)?, 16)?,
                    mul(mul(routes, inner)?, 8)?,
                )?;
                let roots = mul(selected, 12 * 256)?;
                let router = mul(
                    grant(mul(mul(self.max_rows, m.router_spec().num_experts())?, 4)?)?,
                    3,
                )?;
                // Do not spend the legacy activation envelope. Reserve the one
                // opaque rows/F32-bits root separately from the old envelope.
                let route_metadata = grant(mul(routes, 8)?)?;
                route_peak = route_peak.max(add(
                    add(add(grant(payload)?, roots)?, router)?,
                    route_metadata,
                )?);
                routed_batch_host_peak = routed_batch_host_peak.max(add(
                    // Plan IDs/weights overlap the opaque immutable H2D source.
                    mul(routes, 16)?,
                    mul(selected, 3 * std::mem::size_of::<usize>())?,
                )?);
                // One compact layer header and original [gate,up,down] indices.
                routed_slots_host_bytes = add(
                    routed_slots_host_bytes,
                    add(
                        3 * std::mem::size_of::<usize>(),
                        mul(
                            m.router_spec().num_experts(),
                            3 * std::mem::size_of::<usize>(),
                        )?,
                    )?,
                )?;
            }
        }
        let metadata_entries = mul(
            self.max_sequences,
            self.max_positions.div_ceil(self.page_size),
        )?;
        let metadata = add(
            grant(mul(metadata_entries, 4)?)?,
            add(
                grant(mul(add(self.max_sequences, 1)?, 4)?)?,
                mul(grant(mul(self.max_rows, 4)?)?, 3)?,
            )?,
        )?;
        let operation = add(
            layer_peak
                .max(add(
                    route_peak,
                    mul(grant(mul(mul(self.max_rows, self.width)?, 2)?)?, 4)?,
                )?)
                .max(recurrent_peak),
            metadata,
        )?;
        let max_site = self.sites().iter().try_fold(0usize, |peak, site| {
            let n = site.shape.elements(self.max_sequences, self.max_rows)?;
            let n = if site.kind == HybridCollectiveKind::AllGather {
                mul(n, self.layout.members().len())?
            } else {
                n
            };
            Ok::<_, ferrule_common::Error>(peak.max(n))
        })?;
        // Download source, returned upload and legacy F32 export can overlap.
        // Typed host Vec<u16> staging is pageable; no pinned transfer ring is used.
        let boundary = mul(grant(mul(max_site, 4)?)?, 3)?;
        let estimate = HybridMixedRankOperationEstimate {
            storage_grants: storage.storage_grants()?,
            gemmex_storage_grants: gemmex_storage,
            gemmex_provider_bytes: gemmex_provider,
            attention_storage_grants: attention_storage,
            attention_provider_bytes: attention_provider,
            derived_rope_grants,
            attention_metadata_grants: if self.gqa_layer().is_some() {
                grant(4)?
            } else {
                0
            },
            recurrent_workspace_grants: grant(self.recurrent_workspace_bytes()?)?,
            state_copy_workspace_grants: storage.state.copy_workspace_grants,
            operation_grants: add(operation, self.decoded_cache_grants())?,
            boundary_device_grants: boundary,
            pinned_host_bytes: 0,
            host_staging_bytes: add(
                add(mul(max_site, 8)?, packed_qkv_host_bytes)?,
                add(routed_slots_host_bytes, routed_batch_host_peak)?,
            )?,
        };
        if let Some(admission) = &self.decoded_cache {
            admission.validate(physical_pages, estimate.total_device_grants()?)?;
        }
        Ok(estimate)
    }
}
