//! Attention preparation for the shared standard layer composition.
use super::*;
use crate::decoder::StandardSequenceState;
use crate::transformer::{GatedDeltaNetAttention, GatedDeltaNetRequest};
use ferrule_backend::cpu::gated_delta::GatedDeltaShape;

#[derive(Debug, Clone)]
pub struct PreparedGatedDeltaNetBlock {
    pub descriptor: GatedDeltaNetAttention,
    #[cfg(feature = "cuda")]
    pub(crate) vector_parameters: [Arc<crate::transformer::PreparedParameter>; 3],
    pub input_norm: PreparedNorm,
    pub qkv: PreparedLinear,
    pub z: PreparedLinear,
    pub beta: PreparedLinear,
    pub a: PreparedLinear,
    pub conv: Arc<[f32]>,
    pub a_log: Arc<[f32]>,
    pub dt_bias: Arc<[f32]>,
    pub norm: PreparedNorm,
    pub output: PreparedLinear,
}
#[derive(Debug, Clone)]
pub enum PreparedAttentionBlock {
    Gqa(PreparedGqaBlock),
    GatedDeltaNet(PreparedGatedDeltaNetBlock),
}
pub(super) enum AttentionBlockRef<'a> {
    Gqa(&'a PreparedGqaBlock),
    GatedDeltaNet(&'a PreparedGatedDeltaNetBlock),
}
impl PreparedAttentionBlock {
    pub(super) fn as_ref(&self) -> AttentionBlockRef<'_> {
        match self {
            Self::Gqa(block) => AttentionBlockRef::Gqa(block),
            Self::GatedDeltaNet(block) => AttentionBlockRef::GatedDeltaNet(block),
        }
    }
}
pub type PreparedStandardLayer = TransformerLayer<
    Connected<AddResidual, PreparedAttentionBlock>,
    Connected<AddResidual, PreparedFeedForwardBlock>,
>;

pub(super) fn validate_cpu_profile(
    spec: &DecoderModelSpec,
    precision: ExecutionPrecisionPolicy,
) -> Result<()> {
    let extended = spec.final_norm().one_plus_weight()
        || spec.layers().iter().any(|l| {
            l.input_norm().one_plus_weight()
                || l.post_attention_norm().one_plus_weight()
                || match l.attention() {
                    Attention::GatedDeltaNet(_) => true,
                    Attention::Gqa(g) => {
                        g.gated_query()
                            || g.query_norm().is_some_and(RmsNorm::one_plus_weight)
                            || g.key_norm().is_some_and(RmsNorm::one_plus_weight)
                    }
                    _ => false,
                }
        });
    if extended && precision != ExecutionPrecisionPolicy::f32() {
        return Err(unsupported_error(UnsupportedOperator::new(
            "hybrid_cpu_profile",
            "hybrid/gated/offset-norm CPU operators support F32 execution only",
        )));
    }
    Ok(())
}

pub(super) fn prepare_standard_layer(
    descriptor: &DecoderLayer,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
    max_positions: usize,
) -> Result<PreparedStandardLayer> {
    if matches!(descriptor.attention(), Attention::Gqa(_)) {
        let layer = prepare_layer(descriptor, resources, materializer, cache, max_positions)?;
        return Ok(TransformerLayer::new(
            layer.index(),
            Connected::new(
                AddResidual,
                PreparedAttentionBlock::Gqa(layer.attention().block().clone()),
            ),
            layer.feed_forward().clone(),
        ));
    }
    let Attention::GatedDeltaNet(d) = descriptor.attention() else {
        return Err(unsupported_error(UnsupportedOperator::new(
            "standard_attention",
            "unsupported attention descriptor",
        )));
    };
    let layer = descriptor.index();
    let input_norm = prepare_layer_norm(
        layer,
        TensorRole::AttentionNorm,
        descriptor.input_norm(),
        resources,
        materializer,
        cache,
    )?;
    let post_norm = prepare_layer_norm(
        layer,
        TensorRole::FeedForwardNorm,
        descriptor.post_attention_norm(),
        resources,
        materializer,
        cache,
    )?;
    let qkv = prepare_layer_linear(
        layer,
        TensorRole::LinearAttentionQkv,
        d.qkv(),
        resources,
        materializer,
        cache,
    )?;
    let z = prepare_layer_linear(
        layer,
        TensorRole::LinearAttentionZ,
        d.z(),
        resources,
        materializer,
        cache,
    )?;
    let beta = prepare_layer_linear(
        layer,
        TensorRole::LinearAttentionBeta,
        d.beta(),
        resources,
        materializer,
        cache,
    )?;
    let a = prepare_layer_linear(
        layer,
        TensorRole::LinearAttentionA,
        d.a(),
        resources,
        materializer,
        cache,
    )?;
    let output = prepare_layer_linear(
        layer,
        TensorRole::AttentionOutput,
        d.output(),
        resources,
        materializer,
        cache,
    )?;
    let mut parameter = |role: TensorRole, shape: &[usize]| {
        materializer.layer_parameter(
            layer,
            resources.require_layer_shape(layer, role, shape)?,
            cache,
        )
    };
    let conv_parameter = parameter(TensorRole::LinearAttentionConv, &d.conv_weight_shape())?;
    let a_log_parameter = parameter(TensorRole::LinearAttentionALog, &[d.num_value_heads()])?;
    let dt_bias_parameter = parameter(TensorRole::LinearAttentionDtBias, &[d.num_value_heads()])?;
    let conv = conv_parameter.values_f32()?.into();
    let a_log = a_log_parameter.values_f32()?.into();
    let dt_bias = dt_bias_parameter.values_f32()?.into();
    let norm = prepare_layer_norm(
        layer,
        TensorRole::LinearAttentionNorm,
        d.norm(),
        resources,
        materializer,
        cache,
    )?;
    let block = PreparedGatedDeltaNetBlock {
        descriptor: d.clone(),
        #[cfg(feature = "cuda")]
        vector_parameters: [conv_parameter, a_log_parameter, dt_bias_parameter],
        input_norm,
        qkv,
        z,
        beta,
        a,
        conv,
        a_log,
        dt_bias,
        norm,
        output,
    };
    Ok(TransformerLayer::new(
        layer,
        Connected::new(AddResidual, PreparedAttentionBlock::GatedDeltaNet(block)),
        Connected::new(
            AddResidual,
            prepare_feed_forward(descriptor, resources, materializer, cache, post_norm)?,
        ),
    ))
}

pub(super) fn execute_delta(
    operators: &mut dyn StandardDecoderOperators,
    block: &PreparedGatedDeltaNetBlock,
    layer: usize,
    hidden: &CpuTransformerHidden,
    states: &mut [&mut dyn StandardSequenceState],
) -> Result<Rows> {
    let d = &block.descriptor;
    let normalized = ready(operators.rms_norm(&block.input_norm, hidden.rows()?, 1, None)?)?;
    let qkv = ready(operators.linear(&block.qkv, &normalized, None)?)?;
    let z = ready(operators.linear(&block.z, &normalized, None)?)?;
    let a = ready(operators.linear(&block.a, &normalized, None)?)?;
    let b = ready(operators.linear(&block.beta, &normalized, None)?)?;
    let mut linear_states = states
        .iter_mut()
        .map(|s| s.linear_state(layer))
        .collect::<Result<Vec<_>>>()?;
    let update = ready(operators.gated_delta_net(GatedDeltaNetRequest {
        layer,
        shape: GatedDeltaShape {
            key_heads: d.num_key_heads(),
            value_heads: d.num_value_heads(),
            key_dim: d.key_head_dim(),
            value_dim: d.value_head_dim(),
            kernel: d.conv_kernel_dim(),
        },
        qkv: &qkv,
        z: &z,
        a: &a,
        b: &b,
        conv: &block.conv,
        a_log: &block.a_log,
        dt_bias: &block.dt_bias,
        norm: &block.norm,
        metadata: &hidden.metadata,
        states: &mut linear_states,
    })?)?;
    ready(operators.linear(&block.output, &update, None)?)
}
