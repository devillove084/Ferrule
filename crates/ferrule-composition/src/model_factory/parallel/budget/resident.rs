//! Resident external-routing root and colocated physical-card admission.
use crate::Result;
use crate::model_factory::parallel::{inputs::ExpertPlacementConfig, provider_unsupported};
use ferrule_model::{decoder::GenericDecoderOptions, transformer::BoundDecoderResources};
use ferrule_runtime::parallel::pipeline::{HybridDeviceBudget, HybridStageBudget};

/// Actual external-routing root estimate supplied by the model estimator.
/// `weights_bytes` excludes routed experts and any local routed-cache ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRootMemory {
    pub weights_bytes: usize,
    pub state_bytes: usize,
    pub kv_bytes: usize,
    pub workspace_bytes: usize,
}

impl ExpertRootMemory {
    /// Authoritative model estimate for external routing, not the local cache
    /// estimator. Root retains router/shared/static weights and numeric scratch;
    /// all routed weights and owner workspaces are charged separately.
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        if !matches!(
            options.hybrid_cuda_numeric_fp8_precision(),
            Some(
                ferrule_model::transformer::NumericFp8Precision::F32Tf32x3
                    | ferrule_model::transformer::NumericFp8Precision::W8A16Bf16F32Accumulate
            )
        ) {
            return Err(provider_unsupported(
                "numeric EP root requires a production numeric FP8 profile with F32 model boundaries",
            ));
        }
        let estimate = ferrule_model::decoder::HybridMemoryPlan::for_resources_with_routed_experts(
            resources, options, kv_pages,
        )?;
        let copies = options
            .capabilities()
            .max_sequences
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| provider_unsupported("numeric root state copy count overflow"))?;
        let state_bytes = estimate
            .per_sequence_state_bytes
            .checked_mul(copies)
            .ok_or_else(|| provider_unsupported("numeric root state budget overflow"))?;
        Ok(Self {
            weights_bytes: estimate.weight_bytes_upper_bound,
            state_bytes,
            kv_bytes: estimate.kv_bytes,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        })
    }
}

/// One admission ledger per physical card, not per logical CUDA context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalCardMemory {
    pub ordinal: usize,
    pub root_weights_bytes: usize,
    pub expert_weights_bytes: usize,
    pub state_bytes: usize,
    pub kv_bytes: usize,
    pub workspace_bytes: usize,
    pub contexts: usize,
    pub allocator_margin_bytes: usize,
    pub required_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalCardCapacity {
    pub ordinal: usize,
    pub free_bytes: usize,
    pub limit_bytes: usize,
}

/// Immutable pre-allocation proof. This is not a CUDA reservation: callers must
/// sample every selected physical device before starting any owner uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPhysicalPlan {
    cards: Vec<PhysicalCardMemory>,
    budgets: Vec<HybridStageBudget>,
}

impl ExpertPhysicalPlan {
    /// `owner_weights_bytes` must come from exact owned compressed bindings,
    /// including scales, not the single-device routed-cache cap or disk size.
    pub fn new(
        placement: &ExpertPlacementConfig,
        root: ExpertRootMemory,
        owner_weights_bytes: &[usize],
        owner_workspace_bytes: usize,
        allocator_margin_per_context: usize,
    ) -> Result<Self> {
        if owner_weights_bytes.len() != placement.expert_parallel()
            || owner_weights_bytes.contains(&0)
            || root.weights_bytes == 0
            || root.workspace_bytes == 0
            || owner_workspace_bytes == 0
            || allocator_margin_per_context == 0
        {
            return Err(provider_unsupported(
                "numeric EP needs complete nonzero actual root/owner estimates and per-context allocator margins",
            ));
        }
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| provider_unsupported("numeric physical-card budget overflow"))
        };
        let mut cards = Vec::with_capacity(placement.expert_parallel());
        for (owner, (&ordinal, &expert_weights_bytes)) in placement
            .devices
            .iter()
            .zip(owner_weights_bytes)
            .enumerate()
        {
            let is_root = owner == 0;
            let contexts = if is_root { 2 } else { 1 };
            let mut card = PhysicalCardMemory {
                ordinal,
                root_weights_bytes: if is_root { root.weights_bytes } else { 0 },
                expert_weights_bytes,
                state_bytes: if is_root { root.state_bytes } else { 0 },
                kv_bytes: if is_root { root.kv_bytes } else { 0 },
                workspace_bytes: add(
                    owner_workspace_bytes,
                    if is_root { root.workspace_bytes } else { 0 },
                )?,
                contexts,
                allocator_margin_bytes: if is_root {
                    add(allocator_margin_per_context, allocator_margin_per_context)?
                } else {
                    allocator_margin_per_context
                },
                required_bytes: 0,
            };
            card.required_bytes = [
                card.root_weights_bytes,
                card.expert_weights_bytes,
                card.state_bytes,
                card.kv_bytes,
                card.workspace_bytes,
                card.allocator_margin_bytes,
            ]
            .into_iter()
            .try_fold(0, add)?;
            cards.push(card);
        }
        let mut budgets = vec![HybridStageBudget {
            device: placement.root_device(),
            weights: root.weights_bytes as u64,
            state: root.state_bytes as u64,
            kv: root.kv_bytes as u64,
            workspace: root.workspace_bytes as u64,
            margin: allocator_margin_per_context as u64,
        }];
        budgets.extend(
            placement
                .expert_devices()
                .iter()
                .zip(owner_weights_bytes)
                .map(|(&device, &weights)| HybridStageBudget {
                    device,
                    weights: weights as u64,
                    state: 0,
                    kv: 0,
                    workspace: owner_workspace_bytes as u64,
                    margin: allocator_margin_per_context as u64,
                }),
        );
        Ok(Self { cards, budgets })
    }

    pub fn cards(&self) -> &[PhysicalCardMemory] {
        &self.cards
    }

    /// Reject incomplete, duplicate or foreign device samples. Log every card,
    /// including the root+expert0 sum, before returning any capacity failure.
    pub fn check_capacity(&self, capacities: &[PhysicalCardCapacity]) -> Result<()> {
        if capacities.len() != self.cards.len()
            || capacities
                .iter()
                .map(|c| c.ordinal)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != capacities.len()
            || self
                .cards
                .iter()
                .any(|card| !capacities.iter().any(|c| c.ordinal == card.ordinal))
        {
            return Err(provider_unsupported(
                "numeric EP requires exactly one capacity sample per selected physical card",
            ));
        }
        let mut failure = None;
        for card in &self.cards {
            let capacity = capacities
                .iter()
                .find(|c| c.ordinal == card.ordinal)
                .unwrap();
            tracing::info!(
                ordinal = card.ordinal,
                root_weights_bytes = card.root_weights_bytes,
                expert_weights_bytes = card.expert_weights_bytes,
                state_bytes = card.state_bytes,
                kv_bytes = card.kv_bytes,
                workspace_bytes = card.workspace_bytes,
                contexts = card.contexts,
                allocator_margin_bytes = card.allocator_margin_bytes,
                required_bytes = card.required_bytes,
                free_bytes = capacity.free_bytes,
                limit_bytes = capacity.limit_bytes,
                "numeric EP physical-card admission (before uploads)"
            );
            if card.required_bytes > capacity.free_bytes.min(capacity.limit_bytes)
                && failure.is_none()
            {
                failure = Some(provider_unsupported(format!(
                    "numeric EP CUDA ordinal {} requires {} bytes across {} contexts including margins; free {}, limit {}",
                    card.ordinal,
                    card.required_bytes,
                    card.contexts,
                    capacity.free_bytes,
                    capacity.limit_bytes
                )));
            }
        }
        let samples = capacities
            .iter()
            .map(|c| (c.ordinal, c.free_bytes.min(c.limit_bytes) as u64))
            .collect::<Vec<_>>();
        let admission = HybridDeviceBudget::admit(&self.budgets, &samples);
        failure.map_or_else(|| admission.map(|_| ()).map_err(Into::into), Err)
    }
}
