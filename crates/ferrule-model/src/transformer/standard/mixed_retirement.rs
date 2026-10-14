//! Metadata-only replacements for a successfully sealed resident Mixed image.
//! Construction borrows the old image: errors/unwind cannot discard its payloads.
use super::*;
use crate::decoder::MixedSourceRetirement;
use std::collections::BTreeMap;

pub(super) struct MixedRuntimeBindings {
    pub layers: Vec<PreparedStandardLayer>,
    pub output: Option<PreparedStandardOutput>,
    pub deltas: BTreeMap<usize, PreparedGatedDeltaNetBlock>,
}

impl MixedRuntimeBindings {
    pub(super) fn prepare(
        layers: &[PreparedStandardLayer],
        output: Option<&PreparedStandardOutput>,
        completion: &MixedSourceRetirement,
    ) -> Result<Self> {
        if !completion.is_complete() {
            return Err(model_error(
                "Mixed runtime binding requires final claim completion",
            ));
        }
        let layers = layers
            .iter()
            .map(|layer| {
                Ok(TransformerLayer::new(
                    layer.index(),
                    Connected::new(
                        AddResidual,
                        attention(layer.attention().block(), completion)?,
                    ),
                    Connected::new(
                        AddResidual,
                        feed_forward(layer.feed_forward().block(), completion)?,
                    ),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let output = output
            .map(|output| -> Result<PreparedStandardOutput> {
                Ok(OutputPipeline::new(
                    NoReduction,
                    output.norm().clone(),
                    output.head().mixed_runtime_binding(completion)?,
                ))
            })
            .transpose()?;
        // Recurrent execution retains a second descriptor directory. Both must
        // switch together; updating only the layer vector leaves live raw aliases.
        let deltas = layers
            .iter()
            .filter_map(|layer| match layer.attention().block() {
                PreparedAttentionBlock::GatedDeltaNet(delta) => {
                    Some((layer.index(), delta.clone()))
                }
                PreparedAttentionBlock::Gqa(_) => None,
            })
            .collect();
        Ok(Self {
            layers,
            output,
            deltas,
        })
    }
}

fn attention(
    block: &PreparedAttentionBlock,
    completion: &MixedSourceRetirement,
) -> Result<PreparedAttentionBlock> {
    Ok(match block {
        PreparedAttentionBlock::Gqa(block) => {
            let mut runtime = block.clone();
            runtime.query = block.query.mixed_runtime_binding(completion)?;
            runtime.key = block.key.mixed_runtime_binding(completion)?;
            runtime.value = block.value.mixed_runtime_binding(completion)?;
            runtime.output = block.output.mixed_runtime_binding(completion)?;
            PreparedAttentionBlock::Gqa(runtime)
        }
        PreparedAttentionBlock::GatedDeltaNet(block) => {
            let mut runtime = block.clone();
            runtime.qkv = block.qkv.mixed_runtime_binding(completion)?;
            runtime.z = block.z.mixed_runtime_binding(completion)?;
            runtime.a = block.a.mixed_runtime_binding(completion)?;
            runtime.beta = block.beta.mixed_runtime_binding(completion)?;
            runtime.output = block.output.mixed_runtime_binding(completion)?;
            PreparedAttentionBlock::GatedDeltaNet(runtime)
        }
    })
}

fn swiglu(expert: &PreparedSwiGlu, completion: &MixedSourceRetirement) -> Result<PreparedSwiGlu> {
    PreparedSwiGlu::new(
        expert.gate().mixed_runtime_binding(completion)?,
        expert.up().mixed_runtime_binding(completion)?,
        expert.down().mixed_runtime_binding(completion)?,
        expert.activation_limit(),
    )
}

fn feed_forward(
    block: &PreparedFeedForwardBlock,
    completion: &MixedSourceRetirement,
) -> Result<PreparedFeedForwardBlock> {
    let kind = match &block.kind {
        PreparedFeedForwardKind::Dense(expert) => {
            PreparedFeedForwardKind::Dense(swiglu(expert, completion)?)
        }
        PreparedFeedForwardKind::Routed {
            router,
            descriptor,
            shared,
            shared_gate,
            experts,
        } => {
            PreparedFeedForwardKind::Routed {
                router: router.mixed_runtime_binding(completion)?,
                descriptor: descriptor.clone(),
                shared: shared
                    .as_ref()
                    .map(|expert| swiglu(expert, completion))
                    .transpose()?,
                shared_gate: shared_gate
                    .as_ref()
                    .map(|gate| gate.mixed_runtime_binding(completion))
                    .transpose()?,
                // ExpertSemantics contains source metadata, not materialized bytes.
                experts: experts.clone(),
            }
        }
    };
    Ok(PreparedFeedForwardBlock {
        norm: block.norm.clone(),
        kind,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/transformer/mixed_source_retirement.rs"]
mod tests;
