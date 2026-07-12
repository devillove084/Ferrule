//! High-level graph program construction from model descriptors.

use ferrule_common::{Error, Result};
use ferrule_model::{
    AttentionKind, FeedForwardKind, HfSafetensorsInventory, ModelDescriptor, WeightSource,
};

use crate::graph::program::GraphProgram;
use crate::graph::translate::{
    build_dense_decoder_graph_program_with_options,
    build_semantic_transformer_graph_program_with_options, uses_semantic_artifact_groups,
    DenseGraphTranslationOptions, SemanticGraphTranslationOptions,
};
use crate::graph::validation::validate_graph_program;
use ferrule_model::semantic_plan::TransformerSemanticPlan;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphProgramBuildOptions {
    pub dense: DenseGraphTranslationOptions,
    pub semantic: SemanticGraphTranslationOptions,
    pub validate: bool,
}

impl Default for GraphProgramBuildOptions {
    fn default() -> Self {
        Self {
            dense: DenseGraphTranslationOptions::default(),
            semantic: SemanticGraphTranslationOptions::default(),
            validate: true,
        }
    }
}

pub fn build_graph_program_from_descriptor(descriptor: &ModelDescriptor) -> Result<GraphProgram> {
    build_graph_program_from_descriptor_with_options(
        descriptor,
        GraphProgramBuildOptions::default(),
    )
}

pub fn build_graph_program_from_descriptor_with_options(
    descriptor: &ModelDescriptor,
    options: GraphProgramBuildOptions,
) -> Result<GraphProgram> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(Error::Graph(format!(
            "graph program builder currently supports HF safetensors descriptors, got {}",
            descriptor.spec.weight_source
        )));
    }
    let contract = descriptor.support_contract();
    let semantic_plan = TransformerSemanticPlan::from_contract(&contract);
    let inventory = HfSafetensorsInventory::open(&descriptor.path, descriptor.spec.family.clone())?;
    build_graph_program_from_semantic_plan_with_options(&semantic_plan, &inventory, options)
}

pub fn build_graph_program_from_semantic_plan(
    semantic_plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
) -> Result<GraphProgram> {
    build_graph_program_from_semantic_plan_with_options(
        semantic_plan,
        inventory,
        GraphProgramBuildOptions::default(),
    )
}

pub fn build_graph_program_from_semantic_plan_with_options(
    semantic_plan: &TransformerSemanticPlan,
    inventory: &HfSafetensorsInventory,
    options: GraphProgramBuildOptions,
) -> Result<GraphProgram> {
    let program = if is_dense_decoder_plan(semantic_plan) {
        build_dense_decoder_graph_program_with_options(semantic_plan, inventory, options.dense)?
    } else if uses_semantic_artifact_groups(semantic_plan) {
        build_semantic_transformer_graph_program_with_options(
            semantic_plan,
            inventory,
            options.semantic,
        )?
    } else {
        return Err(Error::Graph(format!(
            "no graph translator registered for family={} attention={:?} feed_forward={:?}",
            semantic_plan.family,
            semantic_plan
                .layers
                .first()
                .map(|layer| &layer.attention.kind),
            semantic_plan
                .layers
                .first()
                .map(|layer| &layer.feed_forward.kind)
        )));
    };

    if options.validate {
        validate_graph_program(&program)?;
    }
    Ok(program)
}

fn is_dense_decoder_plan(semantic_plan: &TransformerSemanticPlan) -> bool {
    semantic_plan.layers.iter().all(|layer| {
        matches!(
            layer.attention.kind,
            AttentionKind::DenseMha | AttentionKind::GroupedQuery
        ) && layer.feed_forward.kind == FeedForwardKind::DenseMlp
    })
}
