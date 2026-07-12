use ferrule_common::execution::{ExecutionCapabilities, PreparedModelPlan};

/// Immutable, generation-stamped model preparation result.
///
/// Model-family implementations own their concrete prepared resources in `R`.
/// Resources are intentionally exposed only through shared references so a
/// published plan cannot be mutated behind the executor's back.
#[derive(Debug)]
pub struct PreparedModel<R> {
    generation: u64,
    capabilities: ExecutionCapabilities,
    resources: R,
}

impl<R> PreparedModel<R> {
    pub const fn new(generation: u64, capabilities: ExecutionCapabilities, resources: R) -> Self {
        Self {
            generation,
            capabilities,
            resources,
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    pub const fn resources(&self) -> &R {
        &self.resources
    }
}

impl<R> PreparedModelPlan for PreparedModel<R> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use ferrule_common::execution::{KvBindingMode, LogitsRowPolicy};

    use super::*;

    #[test]
    fn prepared_model_publishes_generation_capabilities_and_immutable_resources() {
        let capabilities = ExecutionCapabilities {
            max_batch_tokens: 8,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: 8,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: NonZeroU32::new(4),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: false,
            full_logits_width: NonZeroU32::new(16),
            kv_binding_mode: KvBindingMode::None,
            logits_row_policy: LogitsRowPolicy::LastPerSequence,
        };
        let plan = PreparedModel::new(7, capabilities, vec!["bound"]);

        assert_eq!(plan.generation(), 7);
        assert_eq!(plan.capabilities(), &capabilities);
        assert_eq!(PreparedModelPlan::capabilities(&plan), &capabilities);
        assert_eq!(plan.resources(), &["bound"]);
    }
}
