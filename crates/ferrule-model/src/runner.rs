use crate::{EnginePlan, ModelDescriptor};
pub use ferrule_common::execution::TokenLogit;
use ferrule_common::Result;

// ── ModelInfo ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: crate::ModelFamily,
    pub architecture: Option<String>,
    pub attention: crate::AttentionKind,
    pub weight_source: crate::WeightSource,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub backend: &'static str,
}

impl ModelInfo {
    pub fn from_descriptor(descriptor: &ModelDescriptor, backend: &'static str) -> Self {
        let spec = &descriptor.spec;
        Self {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            attention: spec.attention.clone(),
            weight_source: spec.weight_source,
            hidden_size: spec.hidden_size.unwrap_or(0),
            num_layers: spec.num_layers.unwrap_or(0),
            num_experts: spec.moe.num_experts.unwrap_or(0),
            num_experts_per_tok: spec.moe.num_experts_per_tok.unwrap_or(0),
            vocab_size: spec.vocab_size.unwrap_or(0),
            backend,
        }
    }
}

// ── ModelRunner trait ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    /// Correctness-first prompt processing. Backends may use a true batched
    /// prefill implementation or a reference fallback.
    Batched,
    /// Append all prompt tokens except the last without materializing logits,
    /// then return top-k for the last token. This is useful for interactive chat
    /// and resident serving workers.
    Interactive,
}

pub trait ModelRunner {
    fn model_info(&self) -> ModelInfo;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>>;
    fn reset_session(&mut self) -> Result<()>;
    fn eos_token_id(&self) -> Option<u32>;
    /// Optional count of model layers/materialized execution states currently bound
    /// into the runner. Useful for lazy artifact-backed runners; dense or eagerly
    /// bound runners may return `None`.
    fn bound_layer_count(&self) -> Option<usize> {
        None
    }

    /// Optional expert activation report (MoE models only).
    fn expert_report(&self) -> Option<String> {
        None
    }
}

/// Optional capability for model runners that can produce top-k logits without
/// materializing a full vocabulary logits vector.
///
/// `ferrule-runtime` owns generation/session algorithms over this trait; concrete
/// model families only implement the primitive operations. This keeps runtime code
/// generic and prevents it from depending on DeepSeek/Qwen/etc. runner types.
pub trait TopKModelRunner: ModelRunner {
    fn position(&self) -> usize;
    fn feed_token(&mut self, token_id: u32) -> Result<()>;

    /// Maximum top-k request the active backend can validate before mutating
    /// sequence state. Backends with a smaller kernel limit must override this.
    fn max_top_k(&self) -> usize {
        usize::MAX
    }

    /// Append prompt tokens without requiring logits for the last row.
    ///
    /// The default keeps existing implementations correct by feeding tokens one
    /// at a time. Model runners with real segment/chunked prefill should override
    /// this so runtime can execute non-final prefill chunks without materializing
    /// hidden states or lm_head logits.
    fn prefill_tokens(&mut self, token_ids: &[u32], _mode: PrefillMode) -> Result<()> {
        for &token_id in token_ids {
            self.feed_token(token_id)?;
        }
        Ok(())
    }

    fn prefill_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
        mode: PrefillMode,
    ) -> Result<Vec<TokenLogit>>;
    fn decode_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>>;
}

// ── Engine-plan helpers ──────────────────────────────────────────────────

pub fn unsupported_runtime_message(plan: &EnginePlan) -> String {
    let mut message = format!(
        "{} metadata is recognized, but the current executable backend cannot run it yet (engine plan: {})",
        plan.family, plan.status
    );
    if !plan.missing.is_empty() {
        message.push_str(". Missing policies:");
        for item in &plan.missing {
            message.push_str(&format!(" {}: {};", item.area, item.reason));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AttentionKind, ModelFamily, ModelSupportContract, MoeSpec, QuantFormatCount, RouterKind,
        TransformerSpec, WeightSource,
    };

    #[test]
    fn unsupported_runtime_message_reports_engine_plan_gaps() {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("deepseek4".into()),
            weight_source: WeightSource::Gguf,
            hidden_size: Some(7168),
            num_layers: Some(1),
            vocab_size: Some(129280),
            num_heads: Some(128),
            num_kv_heads: None,
            head_dim: None,
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(256),
                num_experts_per_tok: Some(8),
                has_shared_experts: true,
                router: RouterKind::HashAssistedTopK,
            },
            semantics: Default::default(),
            tensor_count: Some(1),
            quantization: vec![QuantFormatCount {
                format: "Q4_K".into(),
                tensors: 1,
            }],
            notes: Vec::new(),
        };
        let plan = ModelSupportContract::from_spec(&spec, &[]).engine_plan();
        let message = unsupported_runtime_message(&plan);
        assert!(message.contains("DeepSeek-V4"));
        assert!(message.contains("engine plan: metadata-only"));
        assert!(message.contains("attention:"));
        assert!(message.contains("quantization:"));
        assert!(message.contains("router:"));
    }
}
