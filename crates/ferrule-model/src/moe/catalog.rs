//! Payload-free semantic validation and lowering for routed expert owners.
//! Placement, devices, transport and physical materialization remain with callers.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use ferrule_common::{Error, Result};

use super::ExpertId;
use crate::TensorRole;
use crate::transformer::{
    BoundDecoderResources, BoundParameter, FeedForward, Moe, PreparedLinear, PreparedSwiGlu,
    StateDictMaterializer, UnsupportedOperator,
};

/// A validated global layer and its model-owned router/shared/routed semantics.
/// The router and shared branch remain on the decoder, never on an EP owner.
#[derive(Debug)]
pub struct ExpertLayerSemantics<'a> {
    layer: usize,
    descriptor: &'a Moe,
    router: &'a BoundParameter,
    router_bias: Option<&'a BoundParameter>,
    shared: Option<[&'a BoundParameter; 3]>,
    shared_output_gate: Option<&'a BoundParameter>,
}

impl<'a> ExpertLayerSemantics<'a> {
    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn descriptor(&self) -> &'a Moe {
        self.descriptor
    }

    pub const fn router(&self) -> &'a BoundParameter {
        self.router
    }

    pub const fn router_bias(&self) -> Option<&'a BoundParameter> {
        self.router_bias
    }

    pub const fn shared(&self) -> Option<&[&'a BoundParameter; 3]> {
        self.shared.as_ref()
    }

    pub const fn shared_output_gate(&self) -> Option<&'a BoundParameter> {
        self.shared_output_gate
    }
}

/// Exact global identity, ordered gate/up/down bindings and activation policy.
/// Cloned bindings preserve canonical/source identity without reading payloads.
#[derive(Debug, Clone)]
pub struct ExpertSemantics {
    id: ExpertId,
    parameters: [BoundParameter; 3],
    activation_limit: Option<f32>,
}

impl ExpertSemantics {
    #[cfg(test)]
    pub(crate) fn read_test_fixture(metadata: &crate::transformer::ExpertMetadata) -> Self {
        let crate::nn::ParameterResidency::Expert { layer, expert } =
            *metadata.parameters()[0].residency()
        else {
            panic!("validated expert metadata required")
        };
        Self {
            id: ExpertId::new(layer, expert),
            parameters: metadata.parameters().clone(),
            activation_limit: metadata.activation_limit(),
        }
    }

    pub const fn id(&self) -> ExpertId {
        self.id
    }

    pub const fn parameters(&self) -> &[BoundParameter; 3] {
        &self.parameters
    }

    pub const fn activation_limit(&self) -> Option<f32> {
        self.activation_limit
    }

    pub(crate) fn prepare_from_read(
        &self,
        materializer: &StateDictMaterializer,
        read: &crate::transformer::expert_reads::CurrentExpertRead,
    ) -> Result<PreparedSwiGlu> {
        if (read.plan.layer, read.plan.expert) != (self.id.layer, self.id.expert) {
            return Err(error(
                "prefetched expert identity differs from semantic lookup",
            ));
        }
        let linear = |index: usize| {
            let binding = &self.parameters[index];
            PreparedLinear::from_parameter(
                materializer.expert_parameter_from_read(read, index, binding)?,
                binding.role().clone(),
            )
        };
        PreparedSwiGlu::new(linear(0)?, linear(1)?, linear(2)?, self.activation_limit)
    }

    pub fn prepare(&self, materializer: &StateDictMaterializer) -> Result<PreparedSwiGlu> {
        let linear = |binding: &BoundParameter| {
            PreparedLinear::from_parameter(
                materializer.expert_parameter(self.id.layer, self.id.expert, binding)?,
                binding.role().clone(),
            )
        };
        PreparedSwiGlu::new(
            linear(&self.parameters[0])?,
            linear(&self.parameters[1])?,
            linear(&self.parameters[2])?,
            self.activation_limit,
        )
    }
}

/// Validates the entire selected range before any selected expert can be lowered.
/// This is a semantic catalog, not a host cache, provider or device admission token.
#[derive(Debug)]
pub struct ValidatedExpertCatalog<'a> {
    layers: Vec<ExpertLayerSemantics<'a>>,
    experts: Vec<ExpertSemantics>,
}

impl<'a> ValidatedExpertCatalog<'a> {
    pub fn new(resources: &'a BoundDecoderResources, range: Range<usize>) -> Result<Self> {
        let catalog = Self::for_decoder_range(resources, range)?;
        if catalog.layers.is_empty() {
            return Err(error("expert group contains no routed layer"));
        }
        Ok(catalog)
    }

    /// Decoder segments may be dense; every routed layer in the range is still
    /// checked before embedding, attention, shared or expert payload I/O.
    pub fn for_decoder_range(
        resources: &'a BoundDecoderResources,
        range: Range<usize>,
    ) -> Result<Self> {
        if range.is_empty() || range.end > resources.spec().layers().len() {
            return Err(error("expert layers outside decoder"));
        }
        let mut layers = Vec::new();
        let mut experts = Vec::new();
        for layer in range {
            let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() else {
                continue;
            };
            let projections = [moe.expert().gate(), moe.expert().up(), moe.expert().down()];
            if projections.iter().any(|linear| linear.has_bias()) {
                return Err(Error::ModelSource {
                    source: Box::new(UnsupportedOperator::new(
                        "routed_expert_lowering",
                        "biased routed experts are unsupported",
                    )),
                });
            }
            let router = resources.require_layer_shape(
                layer,
                TensorRole::RouterLogits,
                &moe.router().weight_shape(),
            )?;
            let router_bias = moe
                .router_spec()
                .selection_bias()
                .then(|| {
                    resources.require_layer_shape(
                        layer,
                        TensorRole::RouterBias,
                        &[moe.router_spec().num_experts()],
                    )
                })
                .transpose()?;
            let shared = moe
                .shared_expert()
                .map(|shared| {
                    let projections = [shared.gate(), shared.up(), shared.down()];
                    Ok::<_, Error>([
                        resources.require_layer_shape(
                            layer,
                            TensorRole::SharedExpertGate,
                            &projections[0].weight_shape(),
                        )?,
                        resources.require_layer_shape(
                            layer,
                            TensorRole::SharedExpertUp,
                            &projections[1].weight_shape(),
                        )?,
                        resources.require_layer_shape(
                            layer,
                            TensorRole::SharedExpertDown,
                            &projections[2].weight_shape(),
                        )?,
                    ])
                })
                .transpose()?;
            let shared_output_gate = moe
                .shared_expert_gate()
                .map(|gate| {
                    resources.require_layer_shape(
                        layer,
                        TensorRole::SharedExpertOutputGate,
                        &gate.weight_shape(),
                    )
                })
                .transpose()?;
            for expert in 0..moe.router_spec().num_experts() {
                let binding = |role, projection: &crate::transformer::Linear| {
                    resources.experts().require_shape(
                        layer,
                        expert,
                        role,
                        &projection.weight_shape(),
                    )
                };
                experts.push(ExpertSemantics {
                    id: ExpertId::new(layer, expert),
                    parameters: [
                        binding(TensorRole::RoutedExpertGate, projections[0])?.clone(),
                        binding(TensorRole::RoutedExpertUp, projections[1])?.clone(),
                        binding(TensorRole::RoutedExpertDown, projections[2])?.clone(),
                    ],
                    activation_limit: moe.expert().activation_limit(),
                });
            }
            layers.push(ExpertLayerSemantics {
                layer,
                descriptor: moe,
                router,
                router_bias,
                shared,
                shared_output_gate,
            });
        }
        Ok(Self { layers, experts })
    }

    pub fn layers(&self) -> &[ExpertLayerSemantics<'a>] {
        &self.layers
    }

    pub fn experts(&self) -> &[ExpertSemantics] {
        &self.experts
    }

    pub fn into_experts(self) -> Box<[ExpertSemantics]> {
        self.experts.into_boxed_slice()
    }

    /// Selection is runtime placement policy; semantic lowering is shared by all
    /// transports and uses the existing materializer unchanged.
    pub fn prepare_selected(
        &self,
        materializer: &StateDictMaterializer,
        mut selected: impl FnMut(ExpertId) -> bool,
    ) -> Result<BTreeMap<ExpertId, Arc<PreparedSwiGlu>>> {
        self.experts
            .iter()
            .filter(|expert| selected(expert.id))
            .map(|expert| Ok((expert.id, Arc::new(expert.prepare(materializer)?))))
            .collect()
    }
}

fn error(message: &str) -> Error {
    Error::Model {
        message: message.into(),
    }
}
