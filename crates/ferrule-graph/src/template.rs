use ferrule_core::{Error, Result};

use crate::{AttributeMap, ComputeGraph, ExternalKey, OpKey, TensorData, ValueId, ValueMeta};

/// Symbolic reference to a value inside a [`GraphTemplate`].
///
/// A template can be instantiated into a fresh [`ComputeGraph`] repeatedly. These
/// references are stable within the template, while concrete [`ValueId`]s are
/// assigned during instantiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TemplateValueRef {
    Input(usize),
    External(usize),
    Constant(usize),
    NodeOutput { node: usize, output: usize },
}

/// A graph input declared by a template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInput {
    pub name: String,
    pub meta: ValueMeta,
}

/// A runtime-managed external declared by a template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateExternal {
    pub name: String,
    pub key: ExternalKey,
    pub meta: ValueMeta,
}

/// A constant declared by a template.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateConstant {
    pub name: String,
    pub tensor: TensorData,
}

/// An opaque graph node declared by a template.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateNode {
    pub op: OpKey,
    pub inputs: Vec<TemplateValueRef>,
    pub attrs: AttributeMap,
    pub outputs: Vec<ValueMeta>,
}

/// Reusable graph recipe.
///
/// This is intentionally lower-level than a model-family adapter. A Qwen/Llama
/// style decoder translator can use one or more templates to stamp repeated
/// layer structure, while DeepSeek-specific pieces can add their own template
/// fragments without changing core graph types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphTemplate {
    name: Option<String>,
    inputs: Vec<TemplateInput>,
    externals: Vec<TemplateExternal>,
    constants: Vec<TemplateConstant>,
    nodes: Vec<TemplateNode>,
    outputs: Vec<TemplateValueRef>,
}

impl GraphTemplate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::default()
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn inputs(&self) -> &[TemplateInput] {
        &self.inputs
    }

    pub fn externals(&self) -> &[TemplateExternal] {
        &self.externals
    }

    pub fn constants(&self) -> &[TemplateConstant] {
        &self.constants
    }

    pub fn nodes(&self) -> &[TemplateNode] {
        &self.nodes
    }

    pub fn outputs(&self) -> &[TemplateValueRef] {
        &self.outputs
    }

    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        meta: ValueMeta,
    ) -> Result<TemplateValueRef> {
        let name = non_empty_name(name.into(), "input")?;
        let index = self.inputs.len();
        self.inputs.push(TemplateInput { name, meta });
        Ok(TemplateValueRef::Input(index))
    }

    pub fn add_external(
        &mut self,
        name: impl Into<String>,
        key: ExternalKey,
        meta: ValueMeta,
    ) -> Result<TemplateValueRef> {
        let name = non_empty_name(name.into(), "external")?;
        let index = self.externals.len();
        self.externals.push(TemplateExternal { name, key, meta });
        Ok(TemplateValueRef::External(index))
    }

    pub fn add_constant(
        &mut self,
        name: impl Into<String>,
        tensor: TensorData,
    ) -> Result<TemplateValueRef> {
        let name = non_empty_name(name.into(), "constant")?;
        let index = self.constants.len();
        self.constants.push(TemplateConstant { name, tensor });
        Ok(TemplateValueRef::Constant(index))
    }

    pub fn add_node(
        &mut self,
        op: OpKey,
        inputs: impl Into<Vec<TemplateValueRef>>,
        attrs: AttributeMap,
        outputs: impl Into<Vec<ValueMeta>>,
    ) -> Result<Vec<TemplateValueRef>> {
        let inputs = inputs.into();
        for input in &inputs {
            self.resolve_ref_meta(*input)?;
        }
        let outputs = outputs.into();
        if outputs.is_empty() {
            return Err(Error::Graph(format!(
                "template node '{}::{}' must declare at least one output",
                op.domain(),
                op.name()
            )));
        }
        let node_index = self.nodes.len();
        let output_refs = (0..outputs.len())
            .map(|output| TemplateValueRef::NodeOutput {
                node: node_index,
                output,
            })
            .collect::<Vec<_>>();
        self.nodes.push(TemplateNode {
            op,
            inputs,
            attrs,
            outputs,
        });
        Ok(output_refs)
    }

    pub fn set_outputs(&mut self, outputs: impl Into<Vec<TemplateValueRef>>) -> Result<()> {
        let outputs = outputs.into();
        if outputs.is_empty() {
            return Err(Error::Graph(
                "graph template must declare at least one output".into(),
            ));
        }
        for output in &outputs {
            self.resolve_ref_meta(*output)?;
        }
        self.outputs = outputs;
        Ok(())
    }

    pub fn instantiate(&self) -> Result<TemplateInstantiation> {
        self.validate()?;
        let mut graph = match &self.name {
            Some(name) => ComputeGraph::with_name(name.clone()),
            None => ComputeGraph::new(),
        };

        let mut input_values = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let id = graph.add_input(input.name.clone(), input.meta.clone())?;
            input_values.push(id);
        }

        let mut external_values = Vec::with_capacity(self.externals.len());
        for external in &self.externals {
            let id = graph.add_external(
                external.name.clone(),
                external.key.clone(),
                external.meta.clone(),
            )?;
            external_values.push(id);
        }

        let mut constant_values = Vec::with_capacity(self.constants.len());
        for constant in &self.constants {
            let id = graph.add_constant(constant.name.clone(), constant.tensor.clone())?;
            constant_values.push(id);
        }

        let mut node_outputs: Vec<Vec<ValueId>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let inputs = node
                .inputs
                .iter()
                .map(|reference| {
                    resolve_instantiated_ref(
                        *reference,
                        &input_values,
                        &external_values,
                        &constant_values,
                        &node_outputs,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let (_node_id, outputs) = graph.add_node(
                node.op.clone(),
                inputs,
                node.attrs.clone(),
                node.outputs.clone(),
            )?;
            node_outputs.push(outputs);
        }

        let graph_outputs = self
            .outputs
            .iter()
            .map(|reference| {
                resolve_instantiated_ref(
                    *reference,
                    &input_values,
                    &external_values,
                    &constant_values,
                    &node_outputs,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        graph.set_outputs(graph_outputs.clone())?;
        graph.validate()?;

        Ok(TemplateInstantiation {
            graph,
            inputs: input_values,
            externals: external_values,
            constants: constant_values,
            node_outputs,
            outputs: graph_outputs,
        })
    }

    pub fn validate(&self) -> Result<()> {
        for node in &self.nodes {
            if node.outputs.is_empty() {
                return Err(Error::Graph(format!(
                    "template node '{}::{}' has no outputs",
                    node.op.domain(),
                    node.op.name()
                )));
            }
            for input in &node.inputs {
                self.resolve_ref_meta(*input)?;
            }
        }
        for output in &self.outputs {
            self.resolve_ref_meta(*output)?;
        }
        Ok(())
    }

    fn resolve_ref_meta(&self, reference: TemplateValueRef) -> Result<&ValueMeta> {
        match reference {
            TemplateValueRef::Input(index) => self
                .inputs
                .get(index)
                .map(|input| &input.meta)
                .ok_or_else(|| Error::Graph(format!("unknown template input {index}"))),
            TemplateValueRef::External(index) => self
                .externals
                .get(index)
                .map(|external| &external.meta)
                .ok_or_else(|| Error::Graph(format!("unknown template external {index}"))),
            TemplateValueRef::Constant(index) => self
                .constants
                .get(index)
                .map(|constant| constant.tensor.meta())
                .ok_or_else(|| Error::Graph(format!("unknown template constant {index}"))),
            TemplateValueRef::NodeOutput { node, output } => self
                .nodes
                .get(node)
                .and_then(|template_node| template_node.outputs.get(output))
                .ok_or_else(|| {
                    Error::Graph(format!(
                        "unknown template node output node={node} output={output}"
                    ))
                }),
        }
    }
}

/// Result of instantiating a [`GraphTemplate`].
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateInstantiation {
    pub graph: ComputeGraph,
    pub inputs: Vec<ValueId>,
    pub externals: Vec<ValueId>,
    pub constants: Vec<ValueId>,
    pub node_outputs: Vec<Vec<ValueId>>,
    pub outputs: Vec<ValueId>,
}

fn resolve_instantiated_ref(
    reference: TemplateValueRef,
    inputs: &[ValueId],
    externals: &[ValueId],
    constants: &[ValueId],
    node_outputs: &[Vec<ValueId>],
) -> Result<ValueId> {
    match reference {
        TemplateValueRef::Input(index) => inputs
            .get(index)
            .copied()
            .ok_or_else(|| Error::Graph(format!("unknown instantiated input {index}"))),
        TemplateValueRef::External(index) => externals
            .get(index)
            .copied()
            .ok_or_else(|| Error::Graph(format!("unknown instantiated external {index}"))),
        TemplateValueRef::Constant(index) => constants
            .get(index)
            .copied()
            .ok_or_else(|| Error::Graph(format!("unknown instantiated constant {index}"))),
        TemplateValueRef::NodeOutput { node, output } => node_outputs
            .get(node)
            .and_then(|outputs| outputs.get(output))
            .copied()
            .ok_or_else(|| {
                Error::Graph(format!(
                    "unknown instantiated node output node={node} output={output}"
                ))
            }),
    }
}

fn non_empty_name(name: String, kind: &str) -> Result<String> {
    if name.trim().is_empty() {
        return Err(Error::Graph(format!(
            "graph template {kind} name cannot be empty"
        )));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use crate::{DataType, ValueMeta};

    use super::*;

    #[test]
    fn template_instantiates_to_compute_graph() {
        let mut template = GraphTemplate::with_name("layer_template");
        let hidden = template
            .add_input("hidden", ValueMeta::tensor(DataType::F32, [1, 16]))
            .unwrap();
        let weight = template
            .add_external(
                "proj_weight",
                ExternalKey::new("weights", "layer.proj").unwrap(),
                ValueMeta::tensor(DataType::Bf16, [16, 16]),
            )
            .unwrap();
        let outputs = template
            .add_node(
                OpKey::new("test.dialect", "opaque_transform", 1).unwrap(),
                vec![hidden, weight],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 16])],
            )
            .unwrap();
        template.set_outputs(outputs.clone()).unwrap();

        let instantiation = template.instantiate().unwrap();
        assert_eq!(instantiation.graph.name(), Some("layer_template"));
        assert_eq!(instantiation.graph.inputs().len(), 1);
        assert_eq!(instantiation.graph.nodes().len(), 1);
        assert_eq!(instantiation.outputs.len(), 1);
        assert_eq!(template.outputs(), outputs.as_slice());
    }

    #[test]
    fn invalid_template_ref_is_rejected() {
        let mut template = GraphTemplate::new();
        let err = template
            .add_node(
                OpKey::new("test.dialect", "bad", 1).unwrap(),
                vec![TemplateValueRef::Input(0)],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1])],
            )
            .unwrap_err();
        assert!(format!("{err}").contains("unknown template input"));
    }
}
