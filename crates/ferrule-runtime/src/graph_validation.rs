//! Validation passes for runtime graph programs.

use std::collections::BTreeSet;

use ferrule_core::{Error, Result};
use ferrule_graph::{ExternalKey, ShapeRegistry, ValueMeta, ValueOrigin};

use crate::graph_program::GraphProgram;
use crate::shape_registry::TransformerShapeRegistry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphValidationReport {
    pub external_count: usize,
    pub binding_count: usize,
    pub node_count: usize,
    pub shape_checked_nodes: usize,
    pub unused_bindings: Vec<ExternalKey>,
}

impl GraphValidationReport {
    pub fn has_unused_bindings(&self) -> bool {
        !self.unused_bindings.is_empty()
    }
}

pub fn validate_graph_program(program: &GraphProgram) -> Result<GraphValidationReport> {
    validate_graph_program_with_registry(program, &TransformerShapeRegistry)
}

pub fn validate_graph_program_with_registry<R: ShapeRegistry>(
    program: &GraphProgram,
    registry: &R,
) -> Result<GraphValidationReport> {
    program.graph.validate()?;
    let external_keys = graph_external_keys(program)?;
    validate_external_bindings(program, &external_keys)?;
    let shape_checked_nodes = validate_node_shapes(program, registry)?;

    let unused_bindings = program
        .bindings
        .entries()
        .iter()
        .filter(|binding| !external_keys.contains(&binding.key))
        .map(|binding| binding.key.clone())
        .collect::<Vec<_>>();

    Ok(GraphValidationReport {
        external_count: external_keys.len(),
        binding_count: program.bindings.len(),
        node_count: program.graph.nodes().len(),
        shape_checked_nodes,
        unused_bindings,
    })
}

fn graph_external_keys(program: &GraphProgram) -> Result<BTreeSet<ExternalKey>> {
    let mut keys = BTreeSet::new();
    for value in program.graph.values() {
        if let ValueOrigin::External { key } = value.origin() {
            validate_semantic_external_key(key)?;
            keys.insert(key.clone());
        }
    }
    Ok(keys)
}

fn validate_external_bindings(
    program: &GraphProgram,
    external_keys: &BTreeSet<ExternalKey>,
) -> Result<()> {
    for key in external_keys {
        let binding = program.bindings.get(key).ok_or_else(|| {
            Error::Graph(format!(
                "graph external '{}:{}' has no ExternalBindingPlan entry",
                key.namespace(),
                key.name()
            ))
        })?;
        let value = program
            .graph
            .values()
            .iter()
            .find(|value| matches!(value.origin(), ValueOrigin::External { key: value_key } if value_key == key))
            .ok_or_else(|| {
                Error::Graph(format!(
                    "external binding '{}:{}' does not match a graph value",
                    key.namespace(),
                    key.name()
                ))
            })?;
        if value.meta() != &binding.meta {
            return Err(Error::Graph(format!(
                "external binding meta mismatch for '{}:{}': graph={:?} binding={:?}",
                key.namespace(),
                key.name(),
                value.meta(),
                binding.meta
            )));
        }
    }
    Ok(())
}

fn validate_node_shapes<R: ShapeRegistry>(program: &GraphProgram, registry: &R) -> Result<usize> {
    let mut checked = 0usize;
    for node in program.graph.nodes() {
        let input_metas = node
            .inputs()
            .iter()
            .map(|input| {
                program
                    .graph
                    .value(*input)
                    .map(|value| value.meta().clone())
            })
            .collect::<Result<Vec<ValueMeta>>>()?;
        let inferred = registry.infer_outputs(node.op(), &input_metas, node.attrs())?;
        if inferred.len() != node.outputs().len() {
            return Err(Error::Graph(format!(
                "shape registry output count mismatch for node {} {}::{}: inferred={} graph={}",
                node.id().index(),
                node.op().domain(),
                node.op().name(),
                inferred.len(),
                node.outputs().len()
            )));
        }
        for (index, (inferred_meta, output)) in inferred.iter().zip(node.outputs()).enumerate() {
            let graph_meta = program.graph.value(*output)?.meta();
            if graph_meta != inferred_meta {
                return Err(Error::Graph(format!(
                    "shape mismatch for node {} output {} {}::{}: inferred={:?} graph={:?}",
                    node.id().index(),
                    index,
                    node.op().domain(),
                    node.op().name(),
                    inferred_meta,
                    graph_meta
                )));
            }
        }
        checked += 1;
    }
    Ok(checked)
}

fn validate_semantic_external_key(key: &ExternalKey) -> Result<()> {
    if key.namespace().trim().is_empty() || key.name().trim().is_empty() {
        return Err(Error::Graph(
            "external keys must have namespace and name".into(),
        ));
    }
    let name = key.name();
    let raw_artifact_markers = [
        "model.layers.",
        "self_attn.",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        ".safetensors",
    ];
    if raw_artifact_markers
        .iter()
        .any(|marker| name.contains(marker))
    {
        return Err(Error::Graph(format!(
            "external key '{}:{}' appears to contain raw artifact tensor naming",
            key.namespace(),
            key.name()
        )));
    }
    Ok(())
}
