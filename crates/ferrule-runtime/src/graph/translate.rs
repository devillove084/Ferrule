//! Model/semantic-plan to graph translation.
//!
//! The first concrete translator is intentionally narrow: dense decoder models
//! with MHA/GQA attention and dense SwiGLU FFN. It emits opaque graph nodes and a
//! semantic external binding plan without exposing raw HF tensor names to graph
//! nodes or backend APIs.

use std::collections::BTreeMap;

use crate::graph::{
    AttributeMap, AttributeValue, ComputeGraph, DataType, Dim, ExternalKey, TensorShape, ValueId,
    ValueMeta,
};
use ferrule_common::{Error, Result};
use ferrule_model::semantic::{ArtifactTensorPart, DenseLayerTensorKind};
use ferrule_model::{
    AttentionKind, FeedForwardKind, HfDenseLayerTensorInfo, HfSafetensorsInventory,
    HfSafetensorsTensorInfo, TensorRole,
};

use crate::graph::dialects::{tensor_ops, transformer_ops};
use crate::graph::external_bindings::{
    ArtifactGroupKind, ExternalBinding, ExternalBindingKind, ExternalBindingPlan, ExternalResidency,
};
use crate::graph::program::{GraphProgram, GraphProgramProfile};
use ferrule_model::semantic_plan::{TransformerLayerSemantic, TransformerSemanticPlan};

#[derive(Debug, Clone, PartialEq)]
pub struct DenseGraphTranslationOptions {
    pub norm_epsilon: f64,
    pub rope_theta: Option<f64>,
    pub profile: GraphProgramProfile,
}

impl Default for DenseGraphTranslationOptions {
    fn default() -> Self {
        Self {
            norm_epsilon: 1e-6,
            rope_theta: Some(10_000.0),
            profile: GraphProgramProfile::default(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SemanticGraphTranslationOptions {
    pub profile: GraphProgramProfile,
}

pub fn build_semantic_transformer_graph_program(
    plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
) -> Result<GraphProgram> {
    build_semantic_transformer_graph_program_with_options(
        plan,
        inventory,
        SemanticGraphTranslationOptions::default(),
    )
}

pub fn build_semantic_transformer_graph_program_with_options(
    plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
    options: SemanticGraphTranslationOptions,
) -> Result<GraphProgram> {
    validate_semantic_plan(plan)?;
    let hidden_size = plan.hidden_size.ok_or_else(|| {
        Error::Graph(
            "semantic graph translation requires hidden_size in TransformerSemanticPlan".into(),
        )
    })?;
    let vocab_size = plan.vocab_size.ok_or_else(|| {
        Error::Graph(
            "semantic graph translation requires vocab_size in TransformerSemanticPlan".into(),
        )
    })?;

    let mut graph = ComputeGraph::with_name(format!("{} semantic transformer", plan.family));
    let mut bindings = ExternalBindingPlan::new();
    let token_shape = TensorShape::new(vec![Dim::Symbol(options.profile.token_dim_symbol.clone())]);
    let hidden_meta = activation_meta(vec![
        Dim::Symbol(options.profile.token_dim_symbol.clone()),
        Dim::Known(hidden_size),
    ]);
    let state_meta = activation_meta(vec![
        Dim::Symbol(options.profile.token_dim_symbol.clone()),
        Dim::Dynamic,
    ]);

    let token_ids = graph.add_input("token_ids", ValueMeta::token_ids(token_shape.clone()))?;
    let positions = graph.add_input(
        "positions",
        ValueMeta::tensor(DataType::U32, token_shape.clone()),
    )?;
    let embedding = add_model_weight(
        &mut graph,
        &mut bindings,
        "token_embedding",
        plan.prologue.token_embedding.clone(),
        find_model_tensor(inventory, &plan.prologue.token_embedding)?,
    )?;
    let hidden = add_node_one(
        &mut graph,
        transformer_ops::token_embedding()?,
        vec![token_ids, embedding],
        attrs([("role", AttributeValue::String("token_embedding".into()))]),
        hidden_meta,
    )?;
    let mut state = add_node_one(
        &mut graph,
        transformer_ops::transformer_state_init()?,
        vec![hidden, token_ids, positions],
        attrs([("state", AttributeValue::String("model_activation".into()))]),
        state_meta.clone(),
    )?;

    for layer in &plan.layers {
        let mut inputs = vec![state, token_ids, positions];
        if !layer.pre_norm_roles.is_empty() {
            inputs.push(add_artifact_group(
                &mut graph,
                &mut bindings,
                format!("layers.{}.layer_norm_artifacts", layer.index),
                ArtifactGroupKind::LayerNorm,
                Some(layer.index),
                ExternalResidency::BackendManaged,
            )?);
        }
        inputs.push(add_artifact_group(
            &mut graph,
            &mut bindings,
            format!("layers.{}.attention_artifacts", layer.index),
            ArtifactGroupKind::Attention,
            Some(layer.index),
            ExternalResidency::BackendManaged,
        )?);
        if layer_uses_hyper_connection(layer) {
            inputs.push(add_artifact_group(
                &mut graph,
                &mut bindings,
                format!("layers.{}.hc_attention_artifacts", layer.index),
                ArtifactGroupKind::HyperConnectionAttention,
                Some(layer.index),
                ExternalResidency::BackendManaged,
            )?);
            inputs.push(add_artifact_group(
                &mut graph,
                &mut bindings,
                format!("layers.{}.hc_feed_forward_artifacts", layer.index),
                ArtifactGroupKind::HyperConnectionFeedForward,
                Some(layer.index),
                ExternalResidency::BackendManaged,
            )?);
        }
        if matches!(
            layer.feed_forward.kind,
            FeedForwardKind::RoutedExperts | FeedForwardKind::RoutedAndSharedExperts
        ) {
            inputs.push(add_artifact_group(
                &mut graph,
                &mut bindings,
                format!("layers.{}.router_artifacts", layer.index),
                ArtifactGroupKind::Router,
                Some(layer.index),
                ExternalResidency::BackendManaged,
            )?);
            if layer.feed_forward.has_shared_experts {
                inputs.push(add_artifact_group(
                    &mut graph,
                    &mut bindings,
                    format!("layers.{}.shared_expert_artifacts", layer.index),
                    ArtifactGroupKind::SharedExpert,
                    Some(layer.index),
                    ExternalResidency::BackendManaged,
                )?);
            }
            inputs.push(add_expert_registry(
                &mut graph,
                &mut bindings,
                format!("layers.{}.routed_expert_registry", layer.index),
                layer.index,
                ExternalResidency::Streamable,
            )?);
        }
        inputs.push(add_kv_state(&mut graph, &mut bindings, layer.index)?);
        state = add_node_one(
            &mut graph,
            transformer_ops::transformer_layer()?,
            inputs,
            layer_attrs(layer),
            state_meta.clone(),
        )?;
    }

    let mut projection_inputs = vec![state];
    if plan.epilogue.output_norm.is_some() {
        projection_inputs.push(add_model_weight(
            &mut graph,
            &mut bindings,
            "output_norm",
            TensorRole::OutputNorm,
            find_model_tensor(inventory, &TensorRole::OutputNorm)?,
        )?);
    }
    if inventory
        .hyper_connection_tensors()
        .iter()
        .any(|tensor| tensor.descriptor.layer.is_none())
    {
        projection_inputs.push(add_artifact_group(
            &mut graph,
            &mut bindings,
            "hc_head_artifacts",
            ArtifactGroupKind::HyperConnectionHead,
            None,
            ExternalResidency::BackendManaged,
        )?);
    }
    let output_head_role = plan
        .epilogue
        .output_head
        .clone()
        .unwrap_or(TensorRole::OutputHead);
    projection_inputs.push(add_model_weight(
        &mut graph,
        &mut bindings,
        "output_head",
        output_head_role.clone(),
        find_model_tensor(inventory, &output_head_role)?,
    )?);
    let logits_meta = activation_meta(vec![
        Dim::Symbol(options.profile.token_dim_symbol.clone()),
        Dim::Known(vocab_size),
    ]);
    let mut projection_attrs = attrs([("vocab_size", AttributeValue::UInt(vocab_size as u64))]);
    insert_optional_usize(
        &mut projection_attrs,
        "output_projection_groups",
        plan.policies.semantics.output_projection_groups,
    );
    insert_optional_usize(
        &mut projection_attrs,
        "output_projection_rank",
        plan.policies.semantics.output_projection_rank,
    );
    let logits = add_node_one(
        &mut graph,
        transformer_ops::output_projection()?,
        projection_inputs,
        projection_attrs,
        logits_meta.clone(),
    )?;
    let selected_logits = add_node_one(
        &mut graph,
        transformer_ops::logits_select()?,
        vec![logits],
        attrs([(
            "selection",
            AttributeValue::String("execution_batch".into()),
        )]),
        logits_meta,
    )?;
    graph.set_outputs(vec![selected_logits])?;

    GraphProgram::new(graph, bindings, plan.clone(), options.profile)
}

pub fn build_dense_decoder_graph_program(
    plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
) -> Result<GraphProgram> {
    build_dense_decoder_graph_program_with_options(
        plan,
        inventory,
        DenseGraphTranslationOptions::default(),
    )
}

pub fn build_dense_decoder_graph_program_with_options(
    plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
    options: DenseGraphTranslationOptions,
) -> Result<GraphProgram> {
    validate_dense_plan(plan)?;
    let hidden_size = plan.hidden_size.ok_or_else(|| {
        Error::Graph(
            "dense graph translation requires hidden_size in TransformerSemanticPlan".into(),
        )
    })?;
    let vocab_size = plan.vocab_size.ok_or_else(|| {
        Error::Graph(
            "dense graph translation requires vocab_size in TransformerSemanticPlan".into(),
        )
    })?;

    let dense_tensors = dense_tensor_map(inventory, plan)?;
    let mut graph = ComputeGraph::with_name(format!("{} dense decoder", plan.family));
    let mut bindings = ExternalBindingPlan::new();

    let token_shape = TensorShape::new(vec![Dim::Symbol(options.profile.token_dim_symbol.clone())]);
    let hidden_meta = activation_meta(vec![
        Dim::Symbol(options.profile.token_dim_symbol.clone()),
        Dim::Known(hidden_size),
    ]);

    let token_ids = graph.add_input("token_ids", ValueMeta::token_ids(token_shape.clone()))?;
    let positions = graph.add_input(
        "positions",
        ValueMeta::tensor(DataType::U32, token_shape.clone()),
    )?;

    let embedding = add_model_weight(
        &mut graph,
        &mut bindings,
        "token_embedding",
        TensorRole::TokenEmbedding,
        find_model_tensor(inventory, &TensorRole::TokenEmbedding)?,
    )?;
    let hidden = add_node_one(
        &mut graph,
        transformer_ops::token_embedding()?,
        vec![token_ids, embedding],
        attrs([("role", AttributeValue::String("token_embedding".into()))]),
        hidden_meta.clone(),
    )?;

    let mut hidden = hidden;
    for layer in &plan.layers {
        hidden = translate_dense_layer(
            &mut graph,
            &mut bindings,
            &dense_tensors,
            layer,
            hidden,
            positions,
            &hidden_meta,
            &options,
        )?;
    }

    if let Some(role) = &plan.epilogue.output_norm {
        let output_norm = add_model_weight(
            &mut graph,
            &mut bindings,
            "output_norm",
            role.clone(),
            find_model_tensor(inventory, role)?,
        )?;
        hidden = add_node_one(
            &mut graph,
            transformer_ops::rms_norm()?,
            vec![hidden, output_norm],
            attrs([
                ("role", AttributeValue::String(role.as_str().into())),
                ("epsilon", AttributeValue::Float(options.norm_epsilon)),
            ]),
            hidden_meta.clone(),
        )?;
    }

    let logits_meta = activation_meta(vec![
        Dim::Symbol(options.profile.token_dim_symbol.clone()),
        Dim::Known(vocab_size),
    ]);
    let output_head_role = plan
        .epilogue
        .output_head
        .clone()
        .unwrap_or(TensorRole::OutputHead);
    let output_head = add_model_weight(
        &mut graph,
        &mut bindings,
        "output_head",
        output_head_role.clone(),
        find_model_tensor(inventory, &output_head_role)?,
    )?;
    let logits = add_node_one(
        &mut graph,
        transformer_ops::linear()?,
        vec![hidden, output_head],
        attrs([
            (
                "role",
                AttributeValue::String(output_head_role.as_str().into()),
            ),
            ("projection", AttributeValue::String("lm_head".into())),
        ]),
        logits_meta.clone(),
    )?;
    let selected_logits = add_node_one(
        &mut graph,
        transformer_ops::logits_select()?,
        vec![logits],
        attrs([(
            "selection",
            AttributeValue::String("execution_batch".into()),
        )]),
        logits_meta,
    )?;
    graph.set_outputs(vec![selected_logits])?;

    GraphProgram::new(graph, bindings, plan.clone(), options.profile)
}

fn validate_semantic_plan(plan: &TransformerSemanticPlan) -> Result<()> {
    if !uses_semantic_artifact_groups(plan) {
        return Err(Error::Graph(
            "semantic graph translator requires non-dense attention or routed experts".into(),
        ));
    }
    for layer in &plan.layers {
        match layer.attention.kind {
            AttentionKind::DenseMha
            | AttentionKind::GroupedQuery
            | AttentionKind::MultiLatentAttention => {}
            _ => {
                return Err(Error::Graph(format!(
                    "semantic graph translator does not support attention kind {} in layer {}",
                    layer.attention.kind, layer.index
                )))
            }
        }
        match layer.feed_forward.kind {
            FeedForwardKind::DenseMlp
            | FeedForwardKind::RoutedExperts
            | FeedForwardKind::RoutedAndSharedExperts => {}
            _ => {
                return Err(Error::Graph(format!(
                    "semantic graph translator does not support feed-forward kind {} in layer {}",
                    layer.feed_forward.kind, layer.index
                )))
            }
        }
    }
    Ok(())
}

pub fn uses_semantic_artifact_groups(plan: &TransformerSemanticPlan) -> bool {
    plan.layers.iter().any(|layer| {
        matches!(layer.attention.kind, AttentionKind::MultiLatentAttention)
            || matches!(
                layer.attention.kv_shape,
                ferrule_model::KvCacheShape::LatentOrCompressed
            )
            || matches!(
                layer.feed_forward.kind,
                FeedForwardKind::RoutedExperts | FeedForwardKind::RoutedAndSharedExperts
            )
    })
}

fn layer_uses_hyper_connection(layer: &TransformerLayerSemantic) -> bool {
    matches!(layer.attention.kind, AttentionKind::MultiLatentAttention)
        && matches!(
            layer.feed_forward.kind,
            FeedForwardKind::RoutedExperts | FeedForwardKind::RoutedAndSharedExperts
        )
}

fn layer_attrs(layer: &TransformerLayerSemantic) -> AttributeMap {
    let mut values = attrs([
        ("layer", AttributeValue::UInt(layer.index as u64)),
        (
            "attention",
            AttributeValue::String(layer.attention.kind.as_str().into()),
        ),
        (
            "kv_shape",
            AttributeValue::String(layer.attention.kv_shape.as_str().into()),
        ),
        (
            "feed_forward",
            AttributeValue::String(layer.feed_forward.kind.as_str().into()),
        ),
        (
            "router",
            AttributeValue::String(layer.feed_forward.router.as_str().into()),
        ),
        (
            "norm_epsilon",
            AttributeValue::Float(layer.norm_epsilon as f64),
        ),
        (
            "hc_epsilon",
            AttributeValue::Float(layer.hyper_connection_epsilon as f64),
        ),
        (
            "hc_sinkhorn_iters",
            AttributeValue::UInt(layer.hyper_connection_sinkhorn_iters as u64),
        ),
    ]);
    insert_optional_float(&mut values, "rope_theta", layer.attention.rope_theta);
    insert_optional_usize(&mut values, "rope_head_dim", layer.attention.rope_head_dim);
    insert_optional_float(&mut values, "rope_factor", layer.attention.rope_factor);
    insert_optional_usize(
        &mut values,
        "rope_original_max_position_embeddings",
        layer.attention.rope_original_max_position_embeddings,
    );
    insert_optional_usize(
        &mut values,
        "rope_beta_fast",
        layer.attention.rope_beta_fast,
    );
    insert_optional_usize(
        &mut values,
        "rope_beta_slow",
        layer.attention.rope_beta_slow,
    );
    insert_optional_float(
        &mut values,
        "compress_rope_theta",
        layer.attention.compress_rope_theta,
    );
    insert_optional_usize(
        &mut values,
        "attention_window_size",
        layer.attention.window_size,
    );
    insert_optional_usize(
        &mut values,
        "attention_index_topk",
        layer.attention.index_topk,
    );
    insert_optional_usize(
        &mut values,
        "attention_index_num_heads",
        layer.attention.index_num_heads,
    );
    insert_optional_usize(
        &mut values,
        "attention_index_head_dim",
        layer.attention.index_head_dim,
    );
    insert_optional_usize(
        &mut values,
        "attention_compress_ratio",
        layer.attention.compress_ratio,
    );
    insert_optional_float(&mut values, "swiglu_limit", layer.feed_forward.swiglu_limit);
    insert_optional_float(&mut values, "route_scale", layer.feed_forward.route_scale);
    values
}

fn validate_dense_plan(plan: &TransformerSemanticPlan) -> Result<()> {
    for layer in &plan.layers {
        match layer.attention.kind {
            AttentionKind::DenseMha | AttentionKind::GroupedQuery => {}
            _ => {
                return Err(Error::Graph(format!(
                    "dense graph translator does not support attention kind {} in layer {}",
                    layer.attention.kind, layer.index
                )))
            }
        }
        if layer.feed_forward.kind != FeedForwardKind::DenseMlp {
            return Err(Error::Graph(format!(
                "dense graph translator does not support feed-forward kind {} in layer {}",
                layer.feed_forward.kind, layer.index
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn translate_dense_layer(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    tensors: &DenseTensorMap,
    layer: &TransformerLayerSemantic,
    hidden: ValueId,
    positions: ValueId,
    hidden_meta: &ValueMeta,
    options: &DenseGraphTranslationOptions,
) -> Result<ValueId> {
    let layer_index = layer.index;
    let input_norm = add_layer_weight(
        graph,
        bindings,
        layer_index,
        DenseLayerTensorKind::InputNorm,
        TensorRole::LayerNorm,
        require_dense_tensor(tensors, layer_index, DenseLayerTensorKind::InputNorm)?,
    )?;
    let normed = add_node_one(
        graph,
        transformer_ops::rms_norm()?,
        vec![hidden, input_norm],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            ("role", AttributeValue::String("input_norm".into())),
            ("epsilon", AttributeValue::Float(options.norm_epsilon)),
        ]),
        hidden_meta.clone(),
    )?;

    let q = translate_linear(
        graph,
        bindings,
        tensors,
        layer_index,
        DenseLayerTensorKind::AttentionQuery,
        TensorRole::AttentionQuery,
        "attention_query",
        normed,
    )?;
    let k = translate_linear(
        graph,
        bindings,
        tensors,
        layer_index,
        DenseLayerTensorKind::AttentionKey,
        TensorRole::AttentionKey,
        "attention_key",
        normed,
    )?;
    let v = translate_linear(
        graph,
        bindings,
        tensors,
        layer_index,
        DenseLayerTensorKind::AttentionValue,
        TensorRole::AttentionValue,
        "attention_value",
        normed,
    )?;

    let mut rope_attrs = attrs([
        ("layer", AttributeValue::UInt(layer_index as u64)),
        (
            "attention",
            AttributeValue::String(layer.attention.kind.as_str().into()),
        ),
    ]);
    if let Some(theta) = options.rope_theta {
        rope_attrs.insert("theta".into(), AttributeValue::Float(theta));
    }
    let (_rope_node, rope_outputs) = graph.add_node(
        transformer_ops::rope()?,
        vec![q, k, positions],
        rope_attrs,
        vec![
            graph.value(q)?.meta().clone(),
            graph.value(k)?.meta().clone(),
        ],
    )?;
    let q = rope_outputs[0];
    let k = rope_outputs[1];

    let kv_state = add_kv_state(graph, bindings, layer_index)?;
    let attention_hidden = add_node_one(
        graph,
        transformer_ops::causal_attention()?,
        vec![q, k, v, kv_state],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            (
                "attention",
                AttributeValue::String(layer.attention.kind.as_str().into()),
            ),
            (
                "kv_shape",
                AttributeValue::String(layer.attention.kv_shape.as_str().into()),
            ),
        ]),
        hidden_meta.clone(),
    )?;

    let attention_output = translate_linear(
        graph,
        bindings,
        tensors,
        layer_index,
        DenseLayerTensorKind::AttentionOutput,
        TensorRole::AttentionOutput,
        "attention_output",
        attention_hidden,
    )?;
    let after_attention = add_node_one(
        graph,
        tensor_ops::residual_add()?,
        vec![hidden, attention_output],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            ("residual", AttributeValue::String("attention".into())),
        ]),
        hidden_meta.clone(),
    )?;

    let post_norm = add_layer_weight(
        graph,
        bindings,
        layer_index,
        DenseLayerTensorKind::PostAttentionNorm,
        TensorRole::LayerNorm,
        require_dense_tensor(
            tensors,
            layer_index,
            DenseLayerTensorKind::PostAttentionNorm,
        )?,
    )?;
    let ffn_input = add_node_one(
        graph,
        transformer_ops::rms_norm()?,
        vec![after_attention, post_norm],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            ("role", AttributeValue::String("post_attention_norm".into())),
            ("epsilon", AttributeValue::Float(options.norm_epsilon)),
        ]),
        hidden_meta.clone(),
    )?;

    let gate = add_layer_weight(
        graph,
        bindings,
        layer_index,
        DenseLayerTensorKind::DenseMlpGate,
        TensorRole::DenseMlpGate,
        require_dense_tensor(tensors, layer_index, DenseLayerTensorKind::DenseMlpGate)?,
    )?;
    let up = add_layer_weight(
        graph,
        bindings,
        layer_index,
        DenseLayerTensorKind::DenseMlpUp,
        TensorRole::DenseMlpUp,
        require_dense_tensor(tensors, layer_index, DenseLayerTensorKind::DenseMlpUp)?,
    )?;
    let down = add_layer_weight(
        graph,
        bindings,
        layer_index,
        DenseLayerTensorKind::DenseMlpDown,
        TensorRole::DenseMlpDown,
        require_dense_tensor(tensors, layer_index, DenseLayerTensorKind::DenseMlpDown)?,
    )?;
    let ffn_output = add_node_one(
        graph,
        transformer_ops::swiglu_ffn()?,
        vec![ffn_input, gate, up, down],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            ("activation", AttributeValue::String("silu".into())),
        ]),
        hidden_meta.clone(),
    )?;
    add_node_one(
        graph,
        tensor_ops::residual_add()?,
        vec![after_attention, ffn_output],
        attrs([
            ("layer", AttributeValue::UInt(layer_index as u64)),
            ("residual", AttributeValue::String("ffn".into())),
        ]),
        hidden_meta.clone(),
    )
}

fn translate_linear(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    tensors: &DenseTensorMap,
    layer: usize,
    kind: DenseLayerTensorKind,
    role: TensorRole,
    attr_role: &str,
    input: ValueId,
) -> Result<ValueId> {
    let tensor = require_dense_tensor(tensors, layer, kind)?;
    let weight = add_layer_weight(graph, bindings, layer, kind, role.clone(), tensor)?;
    let output_meta = linear_output_meta(graph.value(input)?.meta(), &external_meta(tensor));
    add_node_one(
        graph,
        transformer_ops::linear()?,
        vec![input, weight],
        attrs([
            ("layer", AttributeValue::UInt(layer as u64)),
            ("role", AttributeValue::String(attr_role.into())),
            ("tensor_role", AttributeValue::String(role.as_str().into())),
        ]),
        output_meta,
    )
}

type DenseTensorKey = (usize, DenseLayerTensorKind, ArtifactTensorPart);
type DenseTensorMap = BTreeMap<DenseTensorKey, HfDenseLayerTensorInfo>;

fn dense_tensor_map(
    inventory: &HfSafetensorsInventory,
    _plan: &TransformerSemanticPlan,
) -> Result<DenseTensorMap> {
    let mut tensors = BTreeMap::new();
    for tensor in inventory.dense_layer_tensors() {
        let key = (
            tensor.descriptor.layer,
            tensor.descriptor.kind,
            tensor.descriptor.part,
        );
        if tensors.insert(key, tensor).is_some() {
            return Err(Error::Graph(format!(
                "duplicate dense tensor binding for layer={} kind={:?} part={:?}",
                key.0, key.1, key.2
            )));
        }
    }
    Ok(tensors)
}

fn require_dense_tensor(
    tensors: &DenseTensorMap,
    layer: usize,
    kind: DenseLayerTensorKind,
) -> Result<&HfDenseLayerTensorInfo> {
    tensors
        .get(&(layer, kind, ArtifactTensorPart::Weight))
        .ok_or_else(|| {
            Error::Graph(format!(
                "missing dense tensor binding for layer={layer} kind={kind:?} part=Weight"
            ))
        })
}

fn find_model_tensor<'a>(
    inventory: &'a HfSafetensorsInventory,
    role: &TensorRole,
) -> Result<&'a HfSafetensorsTensorInfo> {
    inventory
        .tensors
        .iter()
        .find(|tensor| &tensor.role == role)
        .ok_or_else(|| Error::Graph(format!("missing model tensor for role {role}")))
}

fn add_model_weight(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    semantic_name: &str,
    role: TensorRole,
    tensor: &HfSafetensorsTensorInfo,
) -> Result<ValueId> {
    let key = ExternalKey::new("weights", semantic_name)?;
    let meta = tensor_meta(&tensor.dtype, &tensor.shape);
    let value = graph.add_external(semantic_name, key.clone(), meta.clone())?;
    bindings.push(ExternalBinding::weight(
        key,
        role,
        None,
        meta,
        ExternalResidency::BackendManaged,
    ))?;
    Ok(value)
}

fn add_layer_weight(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    layer: usize,
    kind: DenseLayerTensorKind,
    role: TensorRole,
    tensor: &HfDenseLayerTensorInfo,
) -> Result<ValueId> {
    let semantic_name = layer_weight_name(layer, kind);
    let key = ExternalKey::new("weights", semantic_name.clone())?;
    let meta = external_meta(tensor);
    let value = graph.add_external(semantic_name, key.clone(), meta.clone())?;
    bindings.push(ExternalBinding::weight(
        key,
        role,
        Some(layer),
        meta,
        ExternalResidency::BackendManaged,
    ))?;
    Ok(value)
}

fn add_artifact_group(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    semantic_name: impl Into<String>,
    group: ArtifactGroupKind,
    layer: Option<usize>,
    residency: ExternalResidency,
) -> Result<ValueId> {
    let semantic_name = semantic_name.into();
    let key = ExternalKey::new("artifacts", semantic_name.clone())?;
    let meta = ValueMeta::external_state(format!("artifact_group:{}", group.as_str()));
    let value = graph.add_external(semantic_name, key.clone(), meta.clone())?;
    bindings.push(ExternalBinding::artifact_group(
        key, group, layer, meta, residency,
    ))?;
    Ok(value)
}

fn add_expert_registry(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    semantic_name: impl Into<String>,
    layer: usize,
    residency: ExternalResidency,
) -> Result<ValueId> {
    let semantic_name = semantic_name.into();
    let key = ExternalKey::new("experts", semantic_name.clone())?;
    let meta = ValueMeta::external_state("expert_registry");
    let value = graph.add_external(semantic_name, key.clone(), meta.clone())?;
    bindings.push(ExternalBinding::expert_registry(
        key, layer, meta, residency,
    ))?;
    Ok(value)
}

fn add_kv_state(
    graph: &mut ComputeGraph,
    bindings: &mut ExternalBindingPlan,
    layer: usize,
) -> Result<ValueId> {
    let semantic_name = format!("layers.{layer}.kv_state");
    let key = ExternalKey::new("state", semantic_name.clone())?;
    let meta = ValueMeta::external_state("kv_cache");
    let value = graph.add_external(semantic_name, key.clone(), meta.clone())?;
    bindings.push(ExternalBinding::state(
        key,
        ExternalBindingKind::KvState,
        meta,
        ExternalResidency::BackendManaged,
    ))?;
    Ok(value)
}

fn add_node_one(
    graph: &mut ComputeGraph,
    op: crate::graph::OpKey,
    inputs: Vec<ValueId>,
    attrs: AttributeMap,
    output: ValueMeta,
) -> Result<ValueId> {
    let (_node, outputs) = graph.add_node(op, inputs, attrs, vec![output])?;
    Ok(outputs[0])
}

fn layer_weight_name(layer: usize, kind: DenseLayerTensorKind) -> String {
    let suffix = match kind {
        DenseLayerTensorKind::InputNorm => "input_norm",
        DenseLayerTensorKind::PostAttentionNorm => "post_attention_norm",
        DenseLayerTensorKind::AttentionQuery => "attn.q",
        DenseLayerTensorKind::AttentionKey => "attn.k",
        DenseLayerTensorKind::AttentionValue => "attn.v",
        DenseLayerTensorKind::AttentionOutput => "attn.o",
        DenseLayerTensorKind::DenseMlpGate => "ffn.gate",
        DenseLayerTensorKind::DenseMlpUp => "ffn.up",
        DenseLayerTensorKind::DenseMlpDown => "ffn.down",
    };
    format!("layers.{layer}.{suffix}")
}

fn external_meta(tensor: &HfDenseLayerTensorInfo) -> ValueMeta {
    tensor_meta(&tensor.dtype, &tensor.shape)
}

fn tensor_meta(dtype: &str, shape: &[usize]) -> ValueMeta {
    ValueMeta::tensor(dtype_from_hf(dtype), TensorShape::from(shape.to_vec()))
}

fn activation_meta(dims: Vec<Dim>) -> ValueMeta {
    ValueMeta::tensor(DataType::F32, TensorShape::new(dims))
}

fn linear_output_meta(input: &ValueMeta, weight: &ValueMeta) -> ValueMeta {
    let token_dim = input.shape.dims().first().cloned().unwrap_or(Dim::Dynamic);
    let out_dim = weight.shape.dims().first().cloned().unwrap_or(Dim::Dynamic);
    ValueMeta::tensor(DataType::F32, TensorShape::new(vec![token_dim, out_dim]))
}

fn dtype_from_hf(dtype: &str) -> DataType {
    match dtype {
        "BOOL" => DataType::Bool,
        "U8" => DataType::U8,
        "U16" => DataType::U16,
        "U32" => DataType::U32,
        "U64" => DataType::U64,
        "I8" => DataType::I8,
        "I16" => DataType::I16,
        "I32" => DataType::I32,
        "I64" => DataType::I64,
        "F16" => DataType::F16,
        "BF16" => DataType::Bf16,
        "F32" => DataType::F32,
        "F64" => DataType::F64,
        other => DataType::Quantized(other.to_string()),
    }
}

fn attrs<const N: usize>(values: [(&str, AttributeValue); N]) -> AttributeMap {
    values
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn insert_optional_usize(attrs: &mut AttributeMap, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        attrs.insert(key.to_string(), AttributeValue::UInt(value as u64));
    }
}

fn insert_optional_float(attrs: &mut AttributeMap, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        attrs.insert(key.to_string(), AttributeValue::Float(value as f64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ferrule_model::support::tensor_role_for_class;
    use ferrule_model::{
        families, HfSafetensorsShardSummary, ModelFamily, MoeSpec, PolicySet, RouterKind,
        TensorClassCount, TensorRoleCount, TransformerSpec, WeightSource,
    };

    use crate::graph::validation::validate_graph_program;
    use ferrule_model::semantic_plan::{
        AttentionSemantic, ExpertResidency, FeedForwardSemantic, SemanticEpilogue, SemanticPrologue,
    };

    #[test]
    fn semantic_transformer_graph_uses_generic_ops_and_artifact_groups() -> Result<()> {
        let plan = semantic_latent_routed_plan();
        let inventory = semantic_inventory(&plan.family);

        let program = build_semantic_transformer_graph_program(&plan, &inventory)?;
        validate_graph_program(&program)?;

        let op_names = program
            .graph
            .nodes()
            .iter()
            .map(|node| node.op().name())
            .collect::<Vec<_>>();
        assert!(op_names.contains(&"transformer_state_init"));
        assert!(op_names.contains(&"transformer_layer"));
        assert!(op_names.contains(&"output_projection"));
        assert!(op_names
            .iter()
            .all(|name| !name.to_ascii_lowercase().contains("deepseek")));

        assert_has_artifact_group(&program, ArtifactGroupKind::Attention, Some(0));
        assert_has_artifact_group(
            &program,
            ArtifactGroupKind::HyperConnectionAttention,
            Some(0),
        );
        assert_has_artifact_group(
            &program,
            ArtifactGroupKind::HyperConnectionFeedForward,
            Some(0),
        );
        assert_has_artifact_group(&program, ArtifactGroupKind::HyperConnectionHead, None);
        assert_has_artifact_group(&program, ArtifactGroupKind::Router, Some(0));
        assert_has_artifact_group(&program, ArtifactGroupKind::SharedExpert, Some(0));
        assert!(program.bindings.entries().iter().any(|binding| matches!(
            binding.kind,
            ExternalBindingKind::ExpertRegistry
        ) && binding.layer == Some(0)));
        assert!(program
            .bindings
            .entries()
            .iter()
            .any(|binding| matches!(binding.kind, ExternalBindingKind::KvState)));
        assert!(program.bindings.entries().iter().all(|binding| {
            let name = binding.key.name();
            !name.contains("layers.0.attn.wq_a") && !name.contains("model.layers.0")
        }));

        Ok(())
    }

    fn assert_has_artifact_group(
        program: &GraphProgram,
        kind: ArtifactGroupKind,
        layer: Option<usize>,
    ) {
        assert!(
            program.bindings.entries().iter().any(|binding| {
                matches!(binding.kind, ExternalBindingKind::ArtifactGroup(group) if group == kind)
                    && binding.layer == layer
            }),
            "missing artifact group {kind:?} layer={layer:?}"
        );
    }

    fn semantic_latent_routed_plan() -> TransformerSemanticPlan {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("semantic-latent-routed-test".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(4),
            num_layers: Some(1),
            vocab_size: Some(8),
            num_heads: Some(2),
            num_kv_heads: None,
            head_dim: Some(2),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(2),
                num_experts_per_tok: Some(1),
                has_shared_experts: true,
                router: RouterKind::HashAssistedTopK,
            },
            semantics: Default::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        TransformerSemanticPlan {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            hidden_size: spec.hidden_size,
            vocab_size: spec.vocab_size,
            num_heads: spec.num_heads,
            num_kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            prologue: SemanticPrologue {
                token_embedding: TensorRole::TokenEmbedding,
            },
            layers: vec![TransformerLayerSemantic {
                index: 0,
                pre_norm_roles: Vec::new(),
                attention: AttentionSemantic {
                    kind: AttentionKind::MultiLatentAttention,
                    kv_shape: ferrule_model::KvCacheShape::LatentOrCompressed,
                    num_heads: Some(2),
                    num_kv_heads: None,
                    head_dim: Some(2),
                    rope_theta: None,
                    rope_head_dim: None,
                    rope_factor: None,
                    rope_original_max_position_embeddings: None,
                    rope_beta_fast: None,
                    rope_beta_slow: None,
                    compress_rope_theta: None,
                    window_size: Some(2),
                    index_topk: None,
                    index_num_heads: None,
                    index_head_dim: None,
                    compress_ratio: Some(0),
                    required_roles: Vec::new(),
                    optional_roles: Vec::new(),
                    needs_sparse_indices: true,
                    needs_attention_sink: true,
                },
                feed_forward: FeedForwardSemantic {
                    kind: FeedForwardKind::RoutedAndSharedExperts,
                    router: RouterKind::HashAssistedTopK,
                    num_experts: Some(4),
                    num_experts_per_tok: Some(2),
                    required_roles: Vec::new(),
                    optional_roles: Vec::new(),
                    swiglu_limit: None,
                    route_scale: None,
                    expert_residency: ExpertResidency::Streamable,
                    has_shared_experts: true,
                },
                auxiliary_roles: Vec::new(),
                norm_epsilon: 1e-6,
                hyper_connection_epsilon: 1e-6,
                hyper_connection_sinkhorn_iters: 4,
            }],
            epilogue: SemanticEpilogue {
                output_norm: Some(TensorRole::OutputNorm),
                output_head: Some(TensorRole::OutputHead),
            },
            policies: PolicySet::from_spec(&spec),
            attachments: Vec::new(),
        }
    }

    fn semantic_inventory(family: &ModelFamily) -> HfSafetensorsInventory {
        let names = [
            ("token_embd.weight", "BF16", vec![8, 4]),
            ("output_norm.weight", "F32", vec![4]),
            ("output.weight", "BF16", vec![8, 4]),
            ("layers.0.attn.wq_a.weight", "F8_E4M3", vec![2, 4]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn.wq_b.weight", "F8_E4M3", vec![4, 2]),
            ("layers.0.attn.wkv.weight", "F8_E4M3", vec![2, 4]),
            ("layers.0.attn.wo_a.weight", "F8_E4M3", vec![2, 4]),
            ("layers.0.attn.wo_b.weight", "F8_E4M3", vec![4, 2]),
            ("layers.0.attn.q_norm.weight", "F32", vec![2]),
            ("layers.0.attn.kv_norm.weight", "F32", vec![2]),
            ("layers.0.attn.attn_sink", "F32", vec![2]),
            ("layers.0.hc_attn_fn", "F32", vec![4, 4]),
            ("layers.0.hc_attn_scale", "F32", vec![3]),
            ("layers.0.hc_attn_base", "F32", vec![2]),
            ("layers.0.hc_ffn_fn", "F32", vec![4, 4]),
            ("layers.0.hc_ffn_scale", "F32", vec![3]),
            ("layers.0.hc_ffn_base", "F32", vec![2]),
            ("hc_head_fn", "F32", vec![4, 4]),
            ("hc_head_scale", "F32", vec![1]),
            ("hc_head_base", "F32", vec![2]),
            ("layers.0.ffn.gate.weight", "BF16", vec![2, 4]),
            ("layers.0.ffn.gate.tid2eid", "I32", vec![8, 1]),
            (
                "layers.0.ffn.shared_experts.w1.weight",
                "F8_E4M3",
                vec![2, 4],
            ),
            (
                "layers.0.ffn.shared_experts.w2.weight",
                "F8_E4M3",
                vec![4, 2],
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                "F8_E4M3",
                vec![2, 4],
            ),
            ("layers.0.ffn.experts.0.w1.weight", "I8", vec![2, 4]),
            ("layers.0.ffn.experts.0.w2.weight", "I8", vec![4, 2]),
            ("layers.0.ffn.experts.0.w3.weight", "I8", vec![2, 4]),
            ("layers.0.ffn.experts.1.w1.weight", "I8", vec![2, 4]),
            ("layers.0.ffn.experts.1.w2.weight", "I8", vec![4, 2]),
            ("layers.0.ffn.experts.1.w3.weight", "I8", vec![2, 4]),
        ];
        let shard_tensor_count = names.len();
        let mut tensors = Vec::new();
        let mut offset = 0u64;
        for (name, dtype, shape) in names {
            let byte_size = shape.iter().product::<usize>() as u64;
            let class = families::classify_hf_tensor(family, name);
            let role = tensor_role_for_class(&class);
            tensors.push(HfSafetensorsTensorInfo {
                name: name.into(),
                shard: "model-00001-of-00001.safetensors".into(),
                dtype: dtype.into(),
                shape,
                data_offset: offset,
                file_offset: offset,
                byte_size,
                class,
                role,
            });
            offset += byte_size;
        }
        HfSafetensorsInventory {
            family: family.clone(),
            total_size: Some(offset),
            shard_count: 1,
            tensor_count: tensors.len(),
            tensors,
            dtype_counts: Vec::new(),
            class_counts: Vec::<TensorClassCount>::new(),
            role_counts: Vec::<TensorRoleCount>::new(),
            shard_summaries: vec![HfSafetensorsShardSummary {
                shard: "model-00001-of-00001.safetensors".into(),
                tensors: shard_tensor_count,
                bytes: offset,
            }],
            index_only_tensors: Vec::new(),
            header_only_tensors: Vec::new(),
        }
    }
}
