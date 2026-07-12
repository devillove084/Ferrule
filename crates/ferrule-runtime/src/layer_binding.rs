//! Generic per-layer artifact binding and reference execution.
//!
//! This is the first executable vertical slice for artifact-bound HC/MLA/MoE layers.
//! It keeps concrete artifact tensor names in `ferrule-model` bindings: runtime
//! consumes semantic payloads only (attention linears, HC weights, router/shared
//! FFN payloads, and expert streaming handles). The reference executor is
//! deliberately scalar and correctness-oriented; CUDA backends can replace
//! individual steps behind the same bundle/state boundary.

use std::path::Path;

use ferrule_common::{Error, Result};
use ferrule_model::semantic::HyperConnectionStage;
use ferrule_model::{
    HfAttentionTensorInfo, HfHyperConnectionTensorInfo, HfRouterTensorInfo,
    HfSharedExpertTensorInfo, RouterKind,
};

use crate::graph::layer_binding::GraphLayerObjects;
use ferrule_model::artifact::binding::{
    bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_layer_norms_from_artifact_group, bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
    MlaAttentionArtifactPayload, RouterArtifactPayload,
};
use ferrule_model::artifact::tensor::ArtifactTensorReader;
use ferrule_model::attention_backend::{
    sliding_window_topk_indices, sparse_attention_reference, SparseAttentionSpec,
};
use ferrule_model::ffn::SwiGluFfnPayload;
use ferrule_model::hyper_connection::{
    hc_post_reference, hc_pre_reference, HyperConnectionConfig, HyperConnectionWeights,
};
use ferrule_model::moe::executor::CpuReferenceExpertExecutor;
use ferrule_model::moe::handle::CpuExpertHandleStore;
use ferrule_model::moe::routed::{
    execute_routed_moe_with_artifact_router_reference_with_handles, RoutedMoeStepOutput,
};
use ferrule_model::moe::routing::ExpertRouterPolicy;
use ferrule_model::moe::streaming::{
    ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
};
use ferrule_model::semantic_plan::{FeedForwardSemantic, TransformerLayerSemantic};

#[derive(Debug, Clone, PartialEq)]
pub struct GraphLayerBindingOptions {
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub attention_topk: Option<usize>,
    pub expert_policy: ExpertStreamingPolicy,
}

impl Default for GraphLayerBindingOptions {
    fn default() -> Self {
        Self {
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            hc_sinkhorn_iters: 4,
            swiglu_limit: 0.0,
            route_scale: 1.0,
            attention_topk: None,
            expert_policy: ExpertStreamingPolicy::quality_first(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerArtifactBinding {
    pub layer: usize,
    pub attention: MlaAttentionArtifactPayload,
    pub attention_norm: Option<Vec<f32>>,
    pub feed_forward_norm: Option<Vec<f32>>,
    pub hc_attention: HyperConnectionWeights,
    pub hc_feed_forward: HyperConnectionWeights,
    pub router: RouterArtifactPayload,
    pub shared_ffn: Option<SwiGluFfnPayload>,
    pub router_policy: ExpertRouterPolicy,
    pub attention_spec: SparseAttentionSpec,
}

impl LayerArtifactBinding {
    pub fn inferred_hc_config(
        &self,
        eps: f32,
        norm_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<HyperConnectionConfig> {
        let mix_hc = self.hc_attention.base.len();
        let hc_mult = infer_hc_mult_from_mix_hc(mix_hc).ok_or_else(|| {
            Error::Model(format!(
                "layer {} cannot infer HC multiplier from HC mix length {mix_hc}",
                self.layer
            ))
        })?;
        Ok(HyperConnectionConfig {
            hc_mult,
            hidden_size: self.attention.query_a.format.in_features(),
            sinkhorn_iters,
            eps,
            norm_eps,
        })
    }

    pub fn bind_from_graph_objects(
        objects: &GraphLayerObjects<'_>,
        layer_plan: &TransformerLayerSemantic,
        reader: &ArtifactTensorReader,
        options: GraphLayerBindingOptions,
    ) -> Result<Self> {
        if objects.layer != layer_plan.index {
            return Err(Error::Model(format!(
                "graph layer object mismatch: objects layer={} plan layer={}",
                objects.layer, layer_plan.index
            )));
        }
        let attention = bind_attention_from_artifact_group(objects.attention, reader)?;
        let layer_norms = objects
            .layer_norms
            .map(|group| bind_layer_norms_from_artifact_group(group, reader))
            .transpose()?;
        let hc_attention_group = objects.hc_attention.ok_or_else(|| {
            Error::Model(format!(
                "layer {} graph objects are missing HC attention artifacts",
                objects.layer
            ))
        })?;
        let hc_mult = infer_hc_mult(hc_attention_group)?;
        let hidden_size = attention.query_a.format.in_features();
        let hc_config = HyperConnectionConfig {
            hc_mult,
            hidden_size,
            sinkhorn_iters: options.hc_sinkhorn_iters,
            eps: options.hc_eps,
            norm_eps: options.norm_eps,
        };
        let hc_attention =
            bind_hyper_connection_from_artifact_group(hc_attention_group, reader, hc_config)?;
        let hc_feed_forward = bind_hyper_connection_from_artifact_group(
            objects.hc_feed_forward.ok_or_else(|| {
                Error::Model(format!(
                    "layer {} graph objects are missing HC feed-forward artifacts",
                    objects.layer
                ))
            })?,
            reader,
            hc_config,
        )?;
        let router = bind_router_from_artifact_group(
            objects.router.ok_or_else(|| {
                Error::Model(format!(
                    "layer {} graph objects are missing router artifacts",
                    objects.layer
                ))
            })?,
            reader,
        )?;
        let shared_ffn = objects
            .shared_expert
            .map(|group| {
                bind_shared_swiglu_ffn_from_artifact_group(group, reader, options.swiglu_limit)
            })
            .transpose()?;
        let attention_spec = infer_attention_spec(layer_plan, &attention, options.attention_topk)?;
        attention_spec.validate()?;
        Ok(Self {
            layer: objects.layer,
            attention,
            attention_norm: layer_norms
                .as_ref()
                .and_then(|norms| norms.attention_norm.clone()),
            feed_forward_norm: layer_norms
                .as_ref()
                .and_then(|norms| norms.feed_forward_norm.clone()),
            hc_attention,
            hc_feed_forward,
            router_policy: router_policy_from_plan(
                &layer_plan.feed_forward,
                &router,
                options.route_scale,
            ),
            router,
            shared_ffn,
            attention_spec,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_from_hf(
        model_dir: &Path,
        layer: usize,
        attention_tensors: &[HfAttentionTensorInfo],
        hyper_connection_tensors: &[HfHyperConnectionTensorInfo],
        router_tensors: &[HfRouterTensorInfo],
        shared_expert_tensors: &[HfSharedExpertTensorInfo],
        reader: &ArtifactTensorReader,
        hc_config: HyperConnectionConfig,
        swiglu_limit: f32,
        router_policy: ExpertRouterPolicy,
        attention_spec: SparseAttentionSpec,
    ) -> Result<Self> {
        let attention = bind_attention_from_hf(model_dir, layer, attention_tensors, reader)?;
        let hc_attention = bind_hyper_connection_from_hf(
            model_dir,
            layer,
            HyperConnectionStage::Attention,
            hyper_connection_tensors,
            reader,
            hc_config,
        )?;
        let hc_feed_forward = bind_hyper_connection_from_hf(
            model_dir,
            layer,
            HyperConnectionStage::FeedForward,
            hyper_connection_tensors,
            reader,
            hc_config,
        )?;
        let router = bind_router_from_hf(model_dir, layer, router_tensors, reader)?;
        let has_shared = shared_expert_tensors
            .iter()
            .any(|tensor| tensor.descriptor.layer == layer);
        let shared_ffn = has_shared
            .then(|| {
                bind_shared_swiglu_ffn_from_hf(
                    model_dir,
                    layer,
                    shared_expert_tensors,
                    reader,
                    swiglu_limit,
                )
            })
            .transpose()?;
        attention_spec.validate()?;
        Ok(Self {
            layer,
            attention,
            attention_norm: None,
            feed_forward_norm: None,
            hc_attention,
            hc_feed_forward,
            router,
            shared_ffn,
            router_policy,
            attention_spec,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_layer_artifact_from_hf(
    model_dir: &Path,
    layer: usize,
    attention_tensors: &[HfAttentionTensorInfo],
    hyper_connection_tensors: &[HfHyperConnectionTensorInfo],
    router_tensors: &[HfRouterTensorInfo],
    shared_expert_tensors: &[HfSharedExpertTensorInfo],
    reader: &ArtifactTensorReader,
    hc_config: HyperConnectionConfig,
    swiglu_limit: f32,
    router_policy: ExpertRouterPolicy,
    attention_spec: SparseAttentionSpec,
) -> Result<LayerArtifactBinding> {
    LayerArtifactBinding::bind_from_hf(
        model_dir,
        layer,
        attention_tensors,
        hyper_connection_tensors,
        router_tensors,
        shared_expert_tensors,
        reader,
        hc_config,
        swiglu_limit,
        router_policy,
        attention_spec,
    )
}

pub fn bind_layer_artifact_from_graph_objects(
    objects: &GraphLayerObjects<'_>,
    layer_plan: &TransformerLayerSemantic,
    reader: &ArtifactTensorReader,
    options: GraphLayerBindingOptions,
) -> Result<LayerArtifactBinding> {
    LayerArtifactBinding::bind_from_graph_objects(objects, layer_plan, reader, options)
}

pub fn new_layer_expert_runtime_from_graph_objects(
    objects: &GraphLayerObjects<'_>,
    policy: ExpertStreamingPolicy,
) -> Result<LayerExpertRuntime> {
    let mut planner = ExpertStreamingPlanner::new(policy);
    if let Some(registry) = objects.expert_registry {
        for (expert, load_source) in &registry.experts {
            planner.register_load_source(*expert, load_source.clone());
        }
    }
    Ok(LayerExpertRuntime::new(planner))
}

fn infer_hc_mult(group: &crate::backend_object_store::ArtifactObjectGroup) -> Result<usize> {
    let mix_hc = group
        .tensors
        .iter()
        .filter(|tensor| tensor.shape.len() == 1)
        .filter_map(|tensor| tensor.shape.first().copied())
        .max()
        .ok_or_else(|| {
            Error::Model(format!(
                "cannot infer HC multiplier from artifact group {} layer={:?}",
                group.kind.as_str(),
                group.layer
            ))
        })?;
    infer_hc_mult_from_mix_hc(mix_hc).ok_or_else(|| {
        Error::Model(format!(
            "invalid HC mix length {mix_hc} in artifact group {} layer={:?}",
            group.kind.as_str(),
            group.layer
        ))
    })
}

fn infer_hc_mult_from_mix_hc(mix_hc: usize) -> Option<usize> {
    (1..=mix_hc).find(|hc_mult| (hc_mult + 2) * hc_mult == mix_hc)
}

fn infer_attention_spec(
    layer_plan: &TransformerLayerSemantic,
    attention: &MlaAttentionArtifactPayload,
    topk_override: Option<usize>,
) -> Result<SparseAttentionSpec> {
    let heads = layer_plan
        .attention
        .num_heads
        .or(Some(attention.attention_sink.len()))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Error::Model(format!(
                "layer {} missing attention heads",
                layer_plan.index
            ))
        })?;
    let head_dim = layer_plan
        .attention
        .head_dim
        .or_else(|| attention.key_value_norm.len().checked_div(1))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Error::Model(format!(
                "layer {} missing attention head_dim",
                layer_plan.index
            ))
        })?;
    let inferred_topk = layer_plan
        .attention
        .window_size
        .or(layer_plan.attention.index_topk)
        .unwrap_or_else(|| attention.attention_sink.len().max(1));
    Ok(SparseAttentionSpec {
        heads,
        head_dim,
        topk: topk_override.unwrap_or(inferred_topk),
        softmax_scale: (head_dim as f32).powf(-0.5),
        has_attention_sink: !attention.attention_sink.is_empty(),
    })
}

fn router_policy_from_plan(
    plan: &FeedForwardSemantic,
    router: &RouterArtifactPayload,
    route_scale: f32,
) -> ExpertRouterPolicy {
    let top_k = plan
        .num_experts_per_tok
        .or_else(|| (router.hash_cols > 0).then_some(router.hash_cols))
        .unwrap_or(1)
        .max(1);
    if matches!(&plan.router, RouterKind::HashAssistedTopK) {
        ExpertRouterPolicy::sqrt_softplus_hash(top_k, route_scale)
    } else {
        ExpertRouterPolicy::sqrt_softplus_score_topk(top_k, route_scale)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvState {
    head_dim: usize,
    values: Vec<f32>,
}

impl LayerKvState {
    pub fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            values: Vec::new(),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn len(&self) -> usize {
        self.values.len() / self.head_dim
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn append(&mut self, value: &[f32]) -> Result<()> {
        if value.len() != self.head_dim {
            return Err(Error::Model(format!(
                "layer KV value length mismatch: expected {}, got {}",
                self.head_dim,
                value.len()
            )));
        }
        self.values.extend_from_slice(value);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.values.clear();
    }
}

#[derive(Debug, Clone)]
pub struct LayerExecutionState {
    pub kv: LayerKvState,
}

impl LayerExecutionState {
    pub fn new(head_dim: usize) -> Self {
        Self {
            kv: LayerKvState::new(head_dim),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerExpertRuntime {
    pub expert_planner: ExpertStreamingPlanner,
    pub expert_handles: CpuExpertHandleStore,
}

impl LayerExpertRuntime {
    pub fn new(expert_planner: ExpertStreamingPlanner) -> Self {
        Self {
            expert_planner,
            expert_handles: CpuExpertHandleStore::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerStepOutput {
    pub attention_hidden: Vec<f32>,
    pub feed_forward_hidden: Vec<f32>,
    pub moe: RoutedMoeStepOutput,
    pub hc_state: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ReferenceLayerExecutor {
    pub hc_config: HyperConnectionConfig,
    pub expert_reader: ExpertStreamingReader,
    pub expert_executor: CpuReferenceExpertExecutor,
}

impl ReferenceLayerExecutor {
    pub fn new(
        hc_config: HyperConnectionConfig,
        expert_reader: ExpertStreamingReader,
        expert_executor: CpuReferenceExpertExecutor,
    ) -> Self {
        Self {
            hc_config,
            expert_reader,
            expert_executor,
        }
    }

    pub fn execute_decode_step(
        &self,
        binding: &LayerArtifactBinding,
        state: &mut LayerExecutionState,
        expert_runtime: &mut LayerExpertRuntime,
        hc_state: &[f32],
        token_id: u32,
        predicted_experts: &[usize],
    ) -> Result<LayerStepOutput> {
        if hc_state.len() != self.hc_config.hc_hidden_size() {
            return Err(Error::Model(format!(
                "layer {} HC input length mismatch: expected {}, got {}",
                binding.layer,
                self.hc_config.hc_hidden_size(),
                hc_state.len()
            )));
        }

        let attention_pre = hc_pre_reference(hc_state, 1, self.hc_config, &binding.hc_attention)?;
        let attention_input = apply_optional_norm(
            &attention_pre.hidden,
            binding.attention_norm.as_deref(),
            self.hc_config.norm_eps,
            "attention_norm",
        )?;
        let attention_hidden = execute_attention_decode_reference(
            &binding.attention,
            &mut state.kv,
            &attention_input,
            binding.attention_spec,
        )?;
        let after_attention = hc_post_reference(
            &attention_hidden,
            hc_state,
            self.hc_config,
            &attention_pre.split,
        )?;

        let ffn_pre = hc_pre_reference(
            &after_attention,
            1,
            self.hc_config,
            &binding.hc_feed_forward,
        )?;
        let ffn_input = apply_optional_norm(
            &ffn_pre.hidden,
            binding.feed_forward_norm.as_deref(),
            self.hc_config.norm_eps,
            "feed_forward_norm",
        )?;
        let moe = execute_routed_moe_with_artifact_router_reference_with_handles(
            binding.layer,
            &ffn_input,
            token_id,
            &binding.router,
            predicted_experts,
            &binding.router_policy,
            &mut expert_runtime.expert_planner,
            &self.expert_reader,
            &mut expert_runtime.expert_handles,
            &self.expert_executor,
            binding.shared_ffn.as_ref(),
        )?;
        let hc_state = hc_post_reference(
            &moe.output,
            &after_attention,
            self.hc_config,
            &ffn_pre.split,
        )?;

        Ok(LayerStepOutput {
            attention_hidden,
            feed_forward_hidden: moe.output.clone(),
            moe,
            hc_state,
        })
    }
}

fn execute_attention_decode_reference(
    attention: &MlaAttentionArtifactPayload,
    kv: &mut LayerKvState,
    input: &[f32],
    spec: SparseAttentionSpec,
) -> Result<Vec<f32>> {
    spec.validate()?;
    if kv.head_dim() != spec.head_dim {
        return Err(Error::Model(format!(
            "attention KV head_dim mismatch: state={}, spec={}",
            kv.head_dim(),
            spec.head_dim
        )));
    }

    let query_latent = attention.query_a.reference_matvec(input)?;
    let query_latent =
        rms_norm_with_weight(&query_latent, &attention.query_norm, 1e-6, "query_norm")?;
    let query = attention.query_b.reference_matvec(&query_latent)?;
    if query.len() != spec.heads * spec.head_dim {
        return Err(Error::Model(format!(
            "attention query length mismatch: expected {}, got {}",
            spec.heads * spec.head_dim,
            query.len()
        )));
    }

    let kv_value = attention.key_value.reference_matvec(input)?;
    let kv_value =
        rms_norm_with_weight(&kv_value, &attention.key_value_norm, 1e-6, "key_value_norm")?;
    kv.append(&kv_value)?;
    let kv_len = kv.len();
    let topk_indices = sliding_window_topk_indices(spec.topk, 1, kv_len.saturating_sub(1));
    let context = sparse_attention_reference(
        &query,
        kv.values(),
        &topk_indices,
        Some(&attention.attention_sink),
        1,
        kv_len,
        spec,
    )?;
    let projected = attention.output_a.reference_matvec(&context)?;
    attention.output_b.reference_matvec(&projected)
}

fn apply_optional_norm(
    input: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
    label: &str,
) -> Result<Vec<f32>> {
    match weight {
        Some(weight) => rms_norm_with_weight(input, weight, eps, label),
        None => Ok(input.to_vec()),
    }
}

fn rms_norm_with_weight(input: &[f32], weight: &[f32], eps: f32, label: &str) -> Result<Vec<f32>> {
    if input.len() != weight.len() {
        return Err(Error::Model(format!(
            "{label} length mismatch: input={}, weight={}",
            input.len(),
            weight.len()
        )));
    }
    if input.is_empty() {
        return Err(Error::Model(format!(
            "{label} cannot normalize an empty vector"
        )));
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use ferrule_model::TensorRole;

    use super::*;
    use ferrule_model::artifact::linear::ArtifactLinearPayload;
    use ferrule_model::artifact::tensor::{
        ArtifactDType, ArtifactTensorPayload, ArtifactTensorSlice,
    };
    use ferrule_model::moe::executor::CpuReferenceExpertExecutor;
    use ferrule_model::moe::streaming::{
        ExpertId, ExpertLoadSource, ExpertMatrixKind, ExpertStorageTier, ExpertStreamingPolicy,
        ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
    };

    #[test]
    fn generic_mla_layer_vertical_slice_runs_hc_attention_moe_shared_hc() {
        let dir = unique_temp_dir("ferrule-mla-layer-vertical-slice");
        std::fs::create_dir_all(&dir).unwrap();

        let config = HyperConnectionConfig {
            hc_mult: 2,
            hidden_size: 32,
            sinkhorn_iters: 3,
            eps: 1e-6,
            norm_eps: 1e-6,
        };
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 1,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);

        let binding = tiny_layer_binding(config);
        let executor = ReferenceLayerExecutor::new(
            config,
            ExpertStreamingReader::new(4096),
            CpuReferenceExpertExecutor::default(),
        );
        let mut state = LayerExecutionState::new(32);
        let mut expert_runtime = LayerExpertRuntime::new(planner);
        let mut input = vec![0.0f32; config.hc_hidden_size()];
        input[0] = 2.0;
        input[33] = 3.0;

        let first = executor
            .execute_decode_step(&binding, &mut state, &mut expert_runtime, &input, 0, &[])
            .unwrap();
        assert_eq!(first.attention_hidden.len(), 32);
        assert_eq!(first.feed_forward_hidden.len(), 32);
        assert_eq!(first.hc_state.len(), config.hc_hidden_size());
        assert_eq!(state.kv.len(), 1);
        assert_eq!(first.moe.routes.len(), 1);
        assert_eq!(first.moe.routes[0].expert, 0);
        assert_eq!(first.moe.streaming.loads.len(), 1);
        assert!(first.hc_state.iter().all(|value| value.is_finite()));
        assert!(first.feed_forward_hidden[0] > 0.0);

        let second = executor
            .execute_decode_step(&binding, &mut state, &mut expert_runtime, &input, 0, &[])
            .unwrap();
        assert_eq!(state.kv.len(), 2);
        assert_eq!(second.moe.streaming.loads.len(), 0);
        assert_eq!(
            expert_runtime.expert_planner.location(ExpertId::new(0, 0)),
            Some(ExpertStorageTier::Gpu)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layer_kv_state_validates_head_dim() {
        let mut kv = LayerKvState::new(4);
        kv.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(kv.len(), 1);
        let err = kv.append(&[1.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("KV value length mismatch"));
    }

    fn tiny_layer_binding(config: HyperConnectionConfig) -> LayerArtifactBinding {
        LayerArtifactBinding {
            layer: 0,
            attention: tiny_attention_payload(),
            attention_norm: Some(vec![1.0; 32]),
            feed_forward_norm: Some(vec![1.0; 32]),
            hc_attention: zero_hc_weights(config),
            hc_feed_forward: zero_hc_weights(config),
            router: RouterArtifactPayload {
                layer: 0,
                weight: f32_linear(TensorRole::RouterLogits, "router", 1, 32, &vec![0.0; 32]),
                bias: None,
                hash_table: Some(vec![0]),
                hash_rows: 1,
                hash_cols: 1,
            },
            shared_ffn: Some(tiny_shared_ffn()),
            router_policy: ExpertRouterPolicy::sqrt_softplus_hash(1, 1.0),
            attention_spec: SparseAttentionSpec {
                heads: 1,
                head_dim: 32,
                topk: 2,
                softmax_scale: 1.0 / (32.0f32).sqrt(),
                has_attention_sink: true,
            },
        }
    }

    fn tiny_attention_payload() -> MlaAttentionArtifactPayload {
        MlaAttentionArtifactPayload {
            layer: 0,
            query_a: f32_linear(
                TensorRole::AttentionLatentQueryA,
                "wq_a",
                4,
                32,
                &one_hot_rows(4, 32, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
            ),
            query_b: f32_linear(
                TensorRole::AttentionLatentQueryB,
                "wq_b",
                32,
                4,
                &one_hot_rows(32, 4, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
            ),
            key_value: f32_linear(
                TensorRole::AttentionLatentKv,
                "wkv",
                32,
                32,
                &one_hot_rows(
                    32,
                    32,
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
            ),
            output_a: f32_linear(
                TensorRole::AttentionLatentOutputA,
                "wo_a",
                4,
                32,
                &one_hot_rows(4, 32, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
            ),
            output_b: f32_linear(
                TensorRole::AttentionLatentOutputB,
                "wo_b",
                32,
                4,
                &one_hot_rows(32, 4, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
            ),
            query_norm: vec![1.0; 4],
            key_value_norm: vec![1.0; 32],
            attention_sink: vec![0.0],
            auxiliary: Vec::new(),
        }
    }

    fn zero_hc_weights(config: HyperConnectionConfig) -> HyperConnectionWeights {
        HyperConnectionWeights {
            function: vec![0.0; config.mix_hc() * config.hc_hidden_size()],
            scale: vec![1.0, 1.0, 1.0],
            base: vec![0.0; config.mix_hc()],
        }
    }

    fn tiny_shared_ffn() -> SwiGluFfnPayload {
        SwiGluFfnPayload {
            gate: f32_linear(
                TensorRole::SharedExpertGate,
                "shared_gate",
                1,
                32,
                &one_hot_rows(1, 32, &[(0, 0, 1.0)]),
            ),
            up: f32_linear(
                TensorRole::SharedExpertUp,
                "shared_up",
                1,
                32,
                &one_hot_rows(1, 32, &[(0, 1, 1.0)]),
            ),
            down: f32_linear(
                TensorRole::SharedExpertDown,
                "shared_down",
                32,
                1,
                &one_hot_rows(32, 1, &[(0, 0, 1.0)]),
            ),
            swiglu_limit: 10.0,
        }
    }

    fn f32_linear(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        values: &[f32],
    ) -> ArtifactLinearPayload {
        assert_eq!(values.len(), out * input);
        ArtifactLinearPayload::from_weight_and_scale(
            role,
            ArtifactTensorPayload {
                slice: ArtifactTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: ArtifactDType::F32,
                    shape: vec![out, input],
                },
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
            None,
        )
        .unwrap()
    }

    fn one_hot_rows(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut values = vec![0.0f32; rows * cols];
        for &(row, col, value) in entries {
            values[row * cols + col] = value;
        }
        values
    }

    fn register_tiny_expert(
        dir: &Path,
        planner: &mut ExpertStreamingPlanner,
        layer: usize,
        expert: usize,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) {
        let expert_id = ExpertId::new(layer, expert);
        let path = dir.join(format!("l{layer}e{expert}.bin"));
        let tensors = tiny_expert_tensors(expert_id, &path, gate_byte, up_byte, down_byte);
        let mut bytes = Vec::new();
        for tensor in &tensors {
            bytes.extend(&tensor.bytes);
        }
        std::fs::write(&path, bytes).unwrap();
        let mut offset = 0u64;
        let slices = tensors
            .into_iter()
            .map(|tensor| {
                let bytes = tensor.bytes.len() as u64;
                let slice = ExpertTensorSlice {
                    offset,
                    bytes,
                    ..tensor.slice
                };
                offset += bytes;
                slice
            })
            .collect();
        planner.register_load_source(
            expert_id,
            ExpertLoadSource::LocalTensorSet { tensors: slices },
        );
    }

    fn tiny_expert_tensors(
        expert: ExpertId,
        path: &Path,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) -> Vec<ExpertTensorPayload> {
        vec![
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Gate, gate_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Gate),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Up, up_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Up),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Down, down_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Down),
        ]
    }

    fn tiny_fp4_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
        first_byte: u8,
    ) -> ExpertTensorPayload {
        let mut bytes = vec![0u8; 32 * 16];
        bytes[0] = first_byte;
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "I8".into(),
                shape: vec![32, 16],
            },
            bytes,
        }
    }

    fn tiny_scale_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
    ) -> ExpertTensorPayload {
        let bytes = vec![127u8; 32];
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Scale,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "F8_E8M0".into(),
                shape: vec![32, 1],
            },
            bytes,
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
