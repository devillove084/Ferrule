use super::*;
/// Full-depth text-only single-copy target, distinct from model admission.
/// This is neither a device reservation nor proof that an artifact can execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeCapacityPlan {
    pub numeric_profile: NumericFp8Profile,
    pub limits: Qwen35MoeCapacityLimits,
    pub resident_f32_bytes: usize,
    pub compressed_projection_bytes: usize,
    /// All routed experts on disk, excluded from the resident weight sum.
    pub routed_expert_storage_bytes: usize,
    pub one_expert_bytes: usize,
    pub rotary_bytes: usize,
    pub per_sequence_state_bytes: usize,
    pub state_bytes: usize,
    pub kv_pages: usize,
    pub kv_bytes: usize,
    pub minimum_workspace_bytes: usize,
    pub minimum_scratch_bytes: usize,
    /// Includes full cache/workspace caps and allocator margin, not all experts.
    pub total_device_bytes: usize,
}

impl Qwen35MoeCapacityPlan {
    pub const PLANNED_BACKEND_PROFILE: &'static str =
        "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-35b-a3b";

    /// Uses the exact immutable profile schema; does not read payloads, initialize
    /// CUDA, or change Qwen35Config::supports_execution(). Strict artifact checks
    /// must still run via Qwen35Metadata before any future owner construction.
    pub fn from_config(
        config: &ferrule_model::models::qwen35::Qwen35Config,
        limits: Qwen35MoeCapacityLimits,
    ) -> Result<Self> {
        Self::from_config_with_profile(config, limits, NumericFp8Profile::F32)
    }

    pub fn from_config_with_profile(
        config: &ferrule_model::models::qwen35::Qwen35Config,
        limits: Qwen35MoeCapacityLimits,
        numeric_profile: NumericFp8Profile,
    ) -> Result<Self> {
        use ferrule_model::decoder::HybridLayerSchema;
        use ferrule_model::models::qwen35::{
            Qwen35HfNameMapper, Qwen35Profile, Qwen35Recipe, Qwen35TensorPartitionKind,
        };
        use ferrule_model::nn::{ParameterDType, ParameterPart};
        if config.profile() != Qwen35Profile::Moe35BA3Bfp8 {
            return Err(invalid(
                "35B capacity planning requires the exact MoE FP8 profile",
            ));
        }
        let t = config.text();
        if [
            limits.max_experts,
            limits.max_bytes,
            limits.scratch_bytes,
            limits.workspace_bytes,
            limits.kv_bytes,
            limits.device_bytes,
            limits.max_positions,
            limits.max_sequences,
            limits.max_batch_tokens,
        ]
        .contains(&0)
            || limits.max_positions > t.max_position_embeddings
            || limits.max_batch_tokens > 32
            || limits.max_batch_tokens > limits.max_positions
            || limits.max_experts < t.num_experts_per_tok.unwrap()
            || limits.max_experts > t.num_hidden_layers * t.num_experts.unwrap()
        {
            return Err(invalid(
                "35B requires finite nonzero budgets, full-route expert capacity, and token-serial batch <= 32 within context",
            ));
        }
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| invalid("35B capacity overflow"))
        };
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("35B capacity overflow"))
        };
        let mapper = Qwen35HfNameMapper::new(config);
        let mut resident_f32_bytes = 0;
        let mut compressed_projection_bytes = 0;
        let mut routed_expert_storage_bytes = 0;
        let mut one_expert_bytes = 0;
        let mut minimum_scratch_bytes = 0;
        let mut width = t.vocab_size.max(t.hidden_size);
        for tensor in mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
        {
            let bytes = usize::try_from(tensor.bytes())
                .map_err(|_| invalid("35B tensor bytes overflow"))?;
            // Names come from the strict profile mapper, not untrusted prefix matching.
            let routed = tensor.external_name.contains(".mlp.experts.");
            if routed {
                routed_expert_storage_bytes = add(routed_expert_storage_bytes, bytes)?;
                if tensor.external_name.contains(".layers.0.mlp.experts.0.") {
                    one_expert_bytes = add(one_expert_bytes, bytes)?;
                }
            } else if tensor.part == ParameterPart::Scale || tensor.dtype == ParameterDType::F8E4M3
            {
                compressed_projection_bytes = add(compressed_projection_bytes, bytes)?;
            } else {
                // Includes untied embedding AND head, router, shared output gate,
                // norms, convolution and every other nonexpert BF16/F32 tensor.
                let elements = tensor.shape.iter().try_fold(1, |n, d| mul(n, *d))?;
                resident_f32_bytes = add(resident_f32_bytes, mul(elements, 4)?)?;
            }
            if tensor.part == ParameterPart::Weight && tensor.shape.len() == 2 {
                width = width.max(tensor.shape[0]).max(tensor.shape[1]);
                // F32 numeric storage bridge needs at least one decoded weight
                // row. Activation/output buffers belong to operator workspace.
                let padded_k = mul(tensor.shape[1].div_ceil(8), 8)?;
                if numeric_profile == NumericFp8Profile::F32 {
                    minimum_scratch_bytes = minimum_scratch_bytes.max(mul(padded_k, 4)?);
                }
            }
        }
        let cache_required = add(
            mul(limits.max_experts, one_expert_bytes)?,
            limits.scratch_bytes,
        )?;
        if one_expert_bytes == 0
            || cache_required > limits.max_bytes
            || minimum_scratch_bytes > limits.scratch_bytes
        {
            return Err(invalid(format!(
                "35B bounded cache requires {cache_required} bytes (experts + shared scratch), scratch minimum {minimum_scratch_bytes}; limits={limits:?}"
            )));
        }
        let spec = Qwen35Recipe::spec(config).map_err(|e| invalid(e.to_string()))?;
        let schema = HybridStateSchema::from_spec(&spec, t.num_hidden_layers)?;
        let planes = schema.kv_planes(16, limits.max_positions)?;
        let pages = plan_resident_kv_pages(
            &planes,
            ResidentKvPageAccounting::TransactionCapacity {
                page_bytes: physical_page_bytes(&planes)?,
                budget_bytes: Some(limits.kv_bytes as u64),
            },
            ResidentSchedulerConfig {
                max_active_sequences: limits.max_sequences,
                ..Default::default()
            },
            ResidentTopKDriverConfig {
                ctx_size: limits.max_positions,
                ..Default::default()
            },
        )?;
        let kv_bytes = usize::try_from(pages.configured_bytes.unwrap())
            .map_err(|_| invalid("35B KV bytes overflow"))?;
        let mut per_sequence_state_bytes = 0;
        let mut rotary_bytes = 0;
        for layer in schema.layers() {
            match layer {
                HybridLayerSchema::GatedDeltaNet(shape) => {
                    let (channels, conv, recurrent) = shape.sizes()?;
                    width = width.max(channels);
                    per_sequence_state_bytes =
                        add(per_sequence_state_bytes, mul(add(conv, recurrent)?, 4)?)?;
                }
                HybridLayerSchema::FullAttention { .. } => {
                    rotary_bytes = add(
                        rotary_bytes,
                        mul(
                            mul(limits.max_positions, config.semantics().rotary_dimensions)?,
                            4,
                        )?,
                    )?;
                }
            }
        }
        let state_bytes = mul(
            add(mul(limits.max_sequences, 2)?, 1)?,
            per_sequence_state_bytes,
        )?;
        // Conservative F32 operator/workspace envelope, separate from conversion
        // scratch. Model-side allocation accounting must validate this target.
        let minimum_workspace_bytes = mul(
            mul(
                add(
                    mul(width, 32)?,
                    mul(t.num_attention_heads, limits.max_positions)?,
                )?,
                limits.max_batch_tokens,
            )?,
            4,
        )?;
        if minimum_workspace_bytes > limits.workspace_bytes {
            return Err(invalid(format!(
                "35B workspace requires at least {minimum_workspace_bytes} bytes"
            )));
        }
        let total_device_bytes = [
            resident_f32_bytes,
            compressed_projection_bytes,
            rotary_bytes,
            state_bytes,
            kv_bytes,
            limits.workspace_bytes,
            limits.max_bytes,
            limits.allocator_margin_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        if total_device_bytes > limits.device_bytes {
            return Err(invalid(format!(
                "35B capacity requires {total_device_bytes} device bytes, budget {}",
                limits.device_bytes
            )));
        }
        Ok(Self {
            numeric_profile,
            limits,
            resident_f32_bytes,
            compressed_projection_bytes,
            routed_expert_storage_bytes,
            one_expert_bytes,
            rotary_bytes,
            per_sequence_state_bytes,
            state_bytes,
            kv_pages: pages.configured_pages,
            kv_bytes,
            minimum_workspace_bytes,
            minimum_scratch_bytes,
            total_device_bytes,
        })
    }

    /// Schema target only, not the model owner's authoritative CUDA estimate.
    /// Prefill is supplied separately because the legacy capacity DTO only carries
    /// the packed-token bound. Cache and scratch stay charged exactly once.
    pub fn context_capacity_profile(
        &self,
        prefill_chunk_size: usize,
    ) -> Result<ContextCapacityProfile> {
        let logical_pages = self
            .limits
            .max_positions
            .div_ceil(16)
            .checked_mul(self.limits.max_sequences)
            .filter(|pages| *pages > 0)
            .ok_or_else(|| invalid("35B context profile page overflow"))?;
        if self.kv_pages == 0
            || logical_pages.checked_mul(2) != Some(self.kv_pages)
            || self.kv_bytes == 0
            || !self.kv_bytes.is_multiple_of(self.kv_pages)
        {
            return Err(invalid(
                "35B context profile differs from planned KV geometry",
            ));
        }
        let pages = ferrule_runtime::engine::ResidentKvPagePlan {
            full_capacity_pages: logical_pages,
            configured_pages: self.kv_pages,
            page_bytes: Some(
                u64::try_from(self.kv_bytes / self.kv_pages)
                    .map_err(|_| invalid("35B context page bytes overflow"))?,
            ),
            configured_bytes: Some(
                u64::try_from(self.kv_bytes)
                    .map_err(|_| invalid("35B context KV bytes overflow"))?,
            ),
        };
        pages
            .context_capacity_profile(
                ResidentSchedulerConfig {
                    max_active_sequences: self.limits.max_sequences,
                    max_batch_tokens: self.limits.max_batch_tokens,
                    prefill_chunk_size,
                    ..Default::default()
                },
                ResidentTopKDriverConfig {
                    ctx_size: self.limits.max_positions,
                    ..Default::default()
                },
            )?
            .with_memory_requirements(
                self.state_bytes,
                self.limits.workspace_bytes,
                self.total_device_bytes,
            )
    }

    /// A pure check; the future model owner must repeat it against actual free
    /// memory before allocation and enforce the individual caps during execution.
    pub fn check_available_device_bytes(&self, free_bytes: usize) -> Result<()> {
        if self.total_device_bytes > free_bytes {
            return Err(invalid(format!(
                "35B requires {} device bytes including margin, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }
}

/// Metadata-only bridge to the real model numeric options and memory estimator.
/// Payload loading remains on the consuming factory owner, not this plan.
#[derive(Debug)]
pub struct Qwen35MoeMemoryPlan {
    target: Qwen35MoeCapacityPlan,
    pub(super) options: GenericDecoderOptions,
    pub(super) estimate: ferrule_model::decoder::HybridMemoryPlan,
    pub(super) budget: ferrule_model::decoder::HybridReservationRequirements,
    pub(super) total_device_bytes: usize,
}

impl Qwen35MoeMemoryPlan {
    /// Strict headers and source identity only: no payload, tokenizer, device or
    /// runner is created. The returned plan is not a serving capability.
    pub fn open_hf(
        model_path: &std::path::Path,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        Self::open_hf_with_profile(model_path, limits, max_tensor_bytes, NumericFp8Profile::F32)
    }

    pub fn open_hf_with_profile(
        model_path: &std::path::Path,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
        numeric_profile: NumericFp8Profile,
    ) -> Result<Self> {
        let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(model_path)?;
        let target = Qwen35MoeCapacityPlan::from_config_with_profile(
            metadata.config(),
            limits,
            numeric_profile,
        )?;
        Self::for_resources(target, &resources, max_tensor_bytes)
    }

    pub(super) fn for_resources(
        target: Qwen35MoeCapacityPlan,
        resources: &BoundDecoderResources,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        use crate::model_factory::parallel::{
            budget::{BoundMemoryPlan, ResidentCapacityTarget},
            inputs::NumericProviderInputs,
        };
        let inputs = NumericProviderInputs {
            model_label: "Qwen3.5-35B",
            family: ModelFamily::Qwen35Moe,
            limits: target.limits.into(),
            numeric_profile: target.numeric_profile,
            parallel_execution: crate::model_factory::ParallelExecutionContract::external_f32(),
            host_cache: Default::default(),
            placement: None,
        };
        let bound = BoundMemoryPlan::new(
            ResidentCapacityTarget {
                kv_pages: target.kv_pages,
                kv_bytes: target.kv_bytes,
                state_bytes: target.state_bytes,
                per_sequence_state_bytes: target.per_sequence_state_bytes,
                total_device_bytes: target.total_device_bytes,
            },
            &inputs,
            resources,
            max_tensor_bytes,
        )?;
        Ok(Self {
            target,
            options: bound.options,
            estimate: bound.estimate,
            budget: bound.budget,
            total_device_bytes: bound.total_device_bytes,
        })
    }

    /// Owner-estimator requirements, not the lighter schema target or free VRAM.
    pub fn context_capacity_profile(
        &self,
        prefill_chunk_size: usize,
    ) -> Result<ContextCapacityProfile> {
        self.target
            .context_capacity_profile(prefill_chunk_size)?
            .with_memory_requirements(
                self.budget.state_bytes,
                self.budget.workspace_bytes,
                self.total_device_bytes,
            )
    }

    pub fn target(&self) -> &Qwen35MoeCapacityPlan {
        &self.target
    }
    pub fn runner_options(&self) -> &GenericDecoderOptions {
        &self.options
    }
    pub fn model_estimate(&self) -> ferrule_model::decoder::HybridMemoryPlan {
        self.estimate
    }
    pub fn model_budget(&self) -> ferrule_model::decoder::HybridReservationRequirements {
        self.budget
    }
    pub fn total_device_bytes(&self) -> usize {
        self.total_device_bytes
    }

    /// Compare against memory_info().0 on the actual owner after creating its
    /// CUDA context and BEFORE allocating weights/KV/state. The runner must
    /// still perform its own admission; this check reserves no memory.
    pub fn check_available_device_bytes(&self, free_bytes: usize) -> Result<()> {
        if self.total_device_bytes > free_bytes {
            return Err(invalid(format!(
                "35B model admission requires {} bytes including margin, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }
}
