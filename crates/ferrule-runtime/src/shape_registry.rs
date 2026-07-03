//! Shape inference for Ferrule transformer graph dialects.

use ferrule_core::{Error, Result};
use ferrule_graph::{
    AttributeMap, AttributeValue, DataType, Dim, OpKey, ShapeRegistry, TensorShape, ValueMeta,
};

use crate::dialects::domain;

#[derive(Debug, Clone, Default)]
pub struct TransformerShapeRegistry;

impl ShapeRegistry for TransformerShapeRegistry {
    fn infer_outputs(
        &self,
        op: &OpKey,
        inputs: &[ValueMeta],
        _attrs: &AttributeMap,
    ) -> Result<Vec<ValueMeta>> {
        if is_op(op, domain::TRANSFORMER, "token_embedding") {
            return infer_token_embedding(inputs);
        }
        if is_op(op, domain::TRANSFORMER, "linear") {
            return infer_linear(inputs);
        }
        if is_op(op, domain::TRANSFORMER, "rms_norm")
            || is_op(op, domain::TRANSFORMER, "causal_attention")
            || is_op(op, domain::TRANSFORMER, "logits_select")
        {
            return Ok(vec![require_input(inputs, 0, op)?.clone()]);
        }
        if is_op(op, domain::TRANSFORMER, "rope") {
            return Ok(vec![
                require_input(inputs, 0, op)?.clone(),
                require_input(inputs, 1, op)?.clone(),
            ]);
        }
        if is_op(op, domain::TRANSFORMER, "swiglu_ffn") {
            return infer_swiglu_ffn(inputs);
        }
        if is_op(op, domain::TRANSFORMER, "transformer_state_init") {
            return infer_transformer_state_init(inputs, _attrs);
        }
        if is_op(op, domain::TRANSFORMER, "transformer_layer") {
            return Ok(vec![require_input(inputs, 0, op)?.clone()]);
        }
        if is_op(op, domain::TRANSFORMER, "output_projection") {
            return infer_output_projection(inputs, _attrs);
        }
        if is_op(op, domain::TRANSFORMER, "hyper_connection_pre")
            || is_op(op, domain::TRANSFORMER, "latent_attention")
            || is_op(op, domain::TRANSFORMER, "routed_moe")
            || is_op(op, domain::TRANSFORMER, "hyper_connection_post")
        {
            return Ok(vec![require_input(inputs, 0, op)?.clone()]);
        }
        if is_op(op, domain::TENSOR, "residual_add") {
            let lhs = require_input(inputs, 0, op)?;
            let rhs = require_input(inputs, 1, op)?;
            if lhs.shape != rhs.shape {
                return Err(Error::Graph(format!(
                    "residual_add shape mismatch: lhs={:?} rhs={:?}",
                    lhs.shape.dims(),
                    rhs.shape.dims()
                )));
            }
            return Ok(vec![lhs.clone()]);
        }
        Err(Error::Graph(format!(
            "no shape inference registered for op {}::{} v{}",
            op.domain(),
            op.name(),
            op.version()
        )))
    }
}

fn infer_token_embedding(inputs: &[ValueMeta]) -> Result<Vec<ValueMeta>> {
    let token_ids = require_input_by_name(inputs, 0, "token_embedding")?;
    let embedding = require_input_by_name(inputs, 1, "token_embedding")?;
    let hidden = embedding.shape.dims().get(1).cloned().ok_or_else(|| {
        Error::Graph("token_embedding expects embedding weight shape [vocab, hidden]".into())
    })?;
    Ok(vec![ValueMeta::tensor(
        DataType::F32,
        TensorShape::new(vec![first_dim(token_ids), hidden]),
    )])
}

fn infer_linear(inputs: &[ValueMeta]) -> Result<Vec<ValueMeta>> {
    let input = require_input_by_name(inputs, 0, "linear")?;
    let weight = require_input_by_name(inputs, 1, "linear")?;
    let out = weight
        .shape
        .dims()
        .first()
        .cloned()
        .ok_or_else(|| Error::Graph("linear expects weight shape [out, in]".into()))?;
    Ok(vec![ValueMeta::tensor(
        DataType::F32,
        TensorShape::new(vec![first_dim(input), out]),
    )])
}

fn infer_swiglu_ffn(inputs: &[ValueMeta]) -> Result<Vec<ValueMeta>> {
    let input = require_input_by_name(inputs, 0, "swiglu_ffn")?;
    let down = require_input_by_name(inputs, 3, "swiglu_ffn")?;
    let out = down.shape.dims().first().cloned().ok_or_else(|| {
        Error::Graph("swiglu_ffn expects down weight shape [hidden, intermediate]".into())
    })?;
    Ok(vec![ValueMeta::tensor(
        DataType::F32,
        TensorShape::new(vec![first_dim(input), out]),
    )])
}

fn infer_transformer_state_init(
    inputs: &[ValueMeta],
    attrs: &AttributeMap,
) -> Result<Vec<ValueMeta>> {
    let input = require_input_by_name(inputs, 0, "transformer_state_init")?;
    let state_dim = match attrs.get("state_dim") {
        Some(AttributeValue::UInt(value)) => Dim::Known(*value as usize),
        _ => Dim::Dynamic,
    };
    Ok(vec![ValueMeta::tensor(
        DataType::F32,
        TensorShape::new(vec![first_dim(input), state_dim]),
    )])
}

fn infer_output_projection(inputs: &[ValueMeta], attrs: &AttributeMap) -> Result<Vec<ValueMeta>> {
    let input = require_input_by_name(inputs, 0, "output_projection")?;
    let vocab = match attrs.get("vocab_size") {
        Some(AttributeValue::UInt(value)) => Dim::Known(*value as usize),
        _ => Dim::Dynamic,
    };
    Ok(vec![ValueMeta::tensor(
        DataType::F32,
        TensorShape::new(vec![first_dim(input), vocab]),
    )])
}

fn require_input<'a>(inputs: &'a [ValueMeta], index: usize, op: &OpKey) -> Result<&'a ValueMeta> {
    inputs.get(index).ok_or_else(|| {
        Error::Graph(format!(
            "op {}::{} missing input {index}",
            op.domain(),
            op.name()
        ))
    })
}

fn require_input_by_name<'a>(
    inputs: &'a [ValueMeta],
    index: usize,
    op_name: &str,
) -> Result<&'a ValueMeta> {
    inputs
        .get(index)
        .ok_or_else(|| Error::Graph(format!("{op_name} missing input {index}")))
}

fn first_dim(meta: &ValueMeta) -> Dim {
    meta.shape.dims().first().cloned().unwrap_or(Dim::Dynamic)
}

fn is_op(op: &OpKey, domain: &str, name: &str) -> bool {
    op.domain() == domain && op.name() == name
}
