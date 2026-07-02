//! Device-independent compute graph IR for Ferrule.
//!
//! This crate deliberately keeps the core graph free of built-in operator enum
//! variants such as `Linear` or `RmsNorm`. A node stores an opaque [`OpKey`]
//! plus attributes; model-family translators, optimization passes, CPU/CUDA
//! executors, and future autograd code interpret those op keys through their own
//! registries/dialects.

pub mod template;
pub use template::{GraphTemplate, TemplateInstantiation, TemplateValueRef};

use std::collections::BTreeMap;

use ferrule_core::{Error, Result};

/// Stable identifier for a value flowing through a [`ComputeGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(usize);

impl ValueId {
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Stable identifier for a node in a [`ComputeGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Opaque operator key interpreted by dialect registries and backends.
///
/// The graph core never matches on concrete operator names. For example,
/// a Qwen translator may emit a domain/name pair understood by Ferrule's LLM
/// dialect, while a backend may lower that op into one or more CUDA kernels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpKey {
    domain: String,
    name: String,
    version: u32,
}

impl OpKey {
    pub fn new(domain: impl Into<String>, name: impl Into<String>, version: u32) -> Result<Self> {
        let domain = domain.into();
        let name = name.into();
        if domain.trim().is_empty() {
            return Err(Error::Graph("graph op domain cannot be empty".into()));
        }
        if name.trim().is_empty() {
            return Err(Error::Graph("graph op name cannot be empty".into()));
        }
        Ok(Self {
            domain,
            name,
            version,
        })
    }

    pub fn domain(&self) -> &str {
        &self.domain
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.version
    }
}

/// Data type metadata for graph values.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    /// Quantized or packed source type identified by a dialect-specific name.
    Quantized(String),
    /// Runtime/backend-specific data, such as a KV cache handle.
    Opaque(String),
}

/// One dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dim {
    Known(usize),
    Symbol(String),
    Dynamic,
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Self::Known(value)
    }
}

/// Tensor shape metadata. Dynamic/symbolic dimensions are first-class so the
/// same graph can represent prefill, decode, batched decode, and expert batches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorShape {
    dims: Vec<Dim>,
}

impl TensorShape {
    pub fn new(dims: impl Into<Vec<Dim>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn vector(len: usize) -> Self {
        Self {
            dims: vec![Dim::Known(len)],
        }
    }

    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

impl<const N: usize> From<[usize; N]> for TensorShape {
    fn from(value: [usize; N]) -> Self {
        Self {
            dims: value.into_iter().map(Dim::Known).collect(),
        }
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(value: Vec<usize>) -> Self {
        Self {
            dims: value.into_iter().map(Dim::Known).collect(),
        }
    }
}

/// Logical value class. This is intentionally broader than tensors so the IR can
/// route token ids, source handles, KV state, and future rollout/checkpoint state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    Tensor,
    TokenIds,
    ExternalState,
    Opaque,
}

/// Optional layout identifier. Layout semantics are dialect/backend-owned.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutKey(String);

impl LayoutKey {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(Error::Graph("graph layout key cannot be empty".into()));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Shape/type/layout metadata for a graph value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueMeta {
    pub dtype: Option<DataType>,
    pub shape: TensorShape,
    pub layout: Option<LayoutKey>,
    pub kind: ValueKind,
}

impl ValueMeta {
    pub fn tensor(dtype: DataType, shape: impl Into<TensorShape>) -> Self {
        Self {
            dtype: Some(dtype),
            shape: shape.into(),
            layout: None,
            kind: ValueKind::Tensor,
        }
    }

    pub fn token_ids(shape: impl Into<TensorShape>) -> Self {
        Self {
            dtype: Some(DataType::U32),
            shape: shape.into(),
            layout: None,
            kind: ValueKind::TokenIds,
        }
    }

    pub fn external_state(name: impl Into<String>) -> Self {
        Self {
            dtype: Some(DataType::Opaque(name.into())),
            shape: TensorShape::scalar(),
            layout: None,
            kind: ValueKind::ExternalState,
        }
    }

    pub fn with_layout(mut self, layout: LayoutKey) -> Self {
        self.layout = Some(layout);
        self
    }
}

/// External object key for weights, KV cache, source tensors, residency handles,
/// or other runtime-managed state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExternalKey {
    namespace: String,
    name: String,
}

impl ExternalKey {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        let namespace = namespace.into();
        let name = name.into();
        if namespace.trim().is_empty() {
            return Err(Error::Graph("external namespace cannot be empty".into()));
        }
        if name.trim().is_empty() {
            return Err(Error::Graph("external name cannot be empty".into()));
        }
        Ok(Self { namespace, name })
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// How a graph value is produced.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueOrigin {
    Input { index: usize },
    External { key: ExternalKey },
    Constant { tensor: TensorData },
    NodeOutput { node: NodeId, output: usize },
}

/// Value metadata plus origin.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphValue {
    id: ValueId,
    name: Option<String>,
    meta: ValueMeta,
    origin: ValueOrigin,
}

impl GraphValue {
    pub fn id(&self) -> ValueId {
        self.id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn meta(&self) -> &ValueMeta {
        &self.meta
    }

    pub fn origin(&self) -> &ValueOrigin {
        &self.origin
    }
}

/// Attribute value attached to an opaque op.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
    Bools(Vec<bool>),
    Ints(Vec<i64>),
    UInts(Vec<u64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
    Shape(TensorShape),
    DataType(DataType),
}

pub type AttributeMap = BTreeMap<String, AttributeValue>;

/// A graph node: opaque operation key, value inputs/outputs, and attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    id: NodeId,
    op: OpKey,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    attrs: AttributeMap,
}

impl GraphNode {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn op(&self) -> &OpKey {
        &self.op
    }

    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn attrs(&self) -> &AttributeMap {
        &self.attrs
    }
}

/// Host-owned tensor blob used by reference executors and graph IO.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    meta: ValueMeta,
    bytes: Vec<u8>,
}

impl TensorData {
    pub fn new(meta: ValueMeta, bytes: Vec<u8>) -> Self {
        Self { meta, bytes }
    }

    pub fn meta(&self) -> &ValueMeta {
        &self.meta
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Runtime-provided object bound to an [`ExternalKey`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalValue {
    Tensor(TensorData),
    Opaque { kind: String, debug_name: String },
}

/// External bindings supplied at execution time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalHandles {
    values: BTreeMap<ExternalKey, ExternalValue>,
}

impl ExternalHandles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: ExternalKey, value: ExternalValue) -> Option<ExternalValue> {
        self.values.insert(key, value)
    }

    pub fn get(&self, key: &ExternalKey) -> Option<&ExternalValue> {
        self.values.get(key)
    }

    pub fn values(&self) -> &BTreeMap<ExternalKey, ExternalValue> {
        &self.values
    }
}

/// Device-independent graph: values, opaque nodes, graph inputs, graph outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComputeGraph {
    name: Option<String>,
    values: Vec<GraphValue>,
    nodes: Vec<GraphNode>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}

impl ComputeGraph {
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

    pub fn values(&self) -> &[GraphValue] {
        &self.values
    }

    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn value(&self, id: ValueId) -> Result<&GraphValue> {
        self.values
            .get(id.index())
            .ok_or_else(|| Error::Graph(format!("unknown graph value {}", id.index())))
    }

    pub fn node(&self, id: NodeId) -> Result<&GraphNode> {
        self.nodes
            .get(id.index())
            .ok_or_else(|| Error::Graph(format!("unknown graph node {}", id.index())))
    }

    pub fn add_input(&mut self, name: impl Into<String>, meta: ValueMeta) -> Result<ValueId> {
        let id = self.next_value_id();
        let index = self.inputs.len();
        self.values.push(GraphValue {
            id,
            name: Some(non_empty_name(name.into(), "input")?),
            meta,
            origin: ValueOrigin::Input { index },
        });
        self.inputs.push(id);
        Ok(id)
    }

    pub fn add_external(
        &mut self,
        name: impl Into<String>,
        key: ExternalKey,
        meta: ValueMeta,
    ) -> Result<ValueId> {
        let id = self.next_value_id();
        self.values.push(GraphValue {
            id,
            name: Some(non_empty_name(name.into(), "external")?),
            meta,
            origin: ValueOrigin::External { key },
        });
        Ok(id)
    }

    pub fn add_constant(&mut self, name: impl Into<String>, tensor: TensorData) -> Result<ValueId> {
        let id = self.next_value_id();
        let meta = tensor.meta().clone();
        self.values.push(GraphValue {
            id,
            name: Some(non_empty_name(name.into(), "constant")?),
            meta,
            origin: ValueOrigin::Constant { tensor },
        });
        Ok(id)
    }

    pub fn add_node(
        &mut self,
        op: OpKey,
        inputs: impl Into<Vec<ValueId>>,
        attrs: AttributeMap,
        outputs: impl Into<Vec<ValueMeta>>,
    ) -> Result<(NodeId, Vec<ValueId>)> {
        let inputs = inputs.into();
        for input in &inputs {
            self.value(*input)?;
        }

        let output_metas = outputs.into();
        if output_metas.is_empty() {
            return Err(Error::Graph(format!(
                "graph node '{}::{}' must declare at least one output",
                op.domain(),
                op.name()
            )));
        }

        let node_id = NodeId(self.nodes.len());
        let mut output_ids = Vec::with_capacity(output_metas.len());
        for (output_index, meta) in output_metas.into_iter().enumerate() {
            let value_id = self.next_value_id();
            self.values.push(GraphValue {
                id: value_id,
                name: None,
                meta,
                origin: ValueOrigin::NodeOutput {
                    node: node_id,
                    output: output_index,
                },
            });
            output_ids.push(value_id);
        }

        self.nodes.push(GraphNode {
            id: node_id,
            op,
            inputs,
            outputs: output_ids.clone(),
            attrs,
        });
        Ok((node_id, output_ids))
    }

    pub fn set_outputs(&mut self, outputs: impl Into<Vec<ValueId>>) -> Result<()> {
        let outputs = outputs.into();
        if outputs.is_empty() {
            return Err(Error::Graph(
                "graph must declare at least one output".into(),
            ));
        }
        for output in &outputs {
            self.value(*output)?;
        }
        self.outputs = outputs;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        for (index, value) in self.values.iter().enumerate() {
            if value.id.index() != index {
                return Err(Error::Graph(format!(
                    "graph value id mismatch: slot {index}, id {}",
                    value.id.index()
                )));
            }
        }

        for (index, graph_node) in self.nodes.iter().enumerate() {
            if graph_node.id.index() != index {
                return Err(Error::Graph(format!(
                    "graph node id mismatch: slot {index}, id {}",
                    graph_node.id.index()
                )));
            }
            for input in &graph_node.inputs {
                self.value(*input)?;
            }
            for (output_index, output) in graph_node.outputs.iter().enumerate() {
                let value = self.value(*output)?;
                match value.origin() {
                    ValueOrigin::NodeOutput { node, output } => {
                        if *node != graph_node.id() || *output != output_index {
                            return Err(Error::Graph(format!(
                                "graph node {} output {} has inconsistent origin",
                                graph_node.id().index(),
                                output_index
                            )));
                        }
                    }
                    _ => {
                        return Err(Error::Graph(format!(
                            "graph node {} output {} is not a node output value",
                            graph_node.id().index(),
                            output_index
                        )));
                    }
                }
            }
        }

        for input in &self.inputs {
            let value = self.value(*input)?;
            if !matches!(value.origin(), ValueOrigin::Input { .. }) {
                return Err(Error::Graph(format!(
                    "graph input {} does not refer to an input value",
                    input.index()
                )));
            }
        }
        for output in &self.outputs {
            self.value(*output)?;
        }
        Ok(())
    }

    fn next_value_id(&self) -> ValueId {
        ValueId(self.values.len())
    }
}

/// Backend boundary: execute a whole graph, not individual concrete ops.
pub trait GraphBackend {
    fn execute(
        &mut self,
        graph: &ComputeGraph,
        external: &ExternalHandles,
        inputs: &[TensorData],
    ) -> Result<Vec<TensorData>>;
}

/// Shape/type inference registry for an op dialect.
pub trait ShapeRegistry {
    fn infer_outputs(
        &self,
        op: &OpKey,
        inputs: &[ValueMeta],
        attrs: &AttributeMap,
    ) -> Result<Vec<ValueMeta>>;
}

fn non_empty_name(name: String, kind: &str) -> Result<String> {
    if name.trim().is_empty() {
        return Err(Error::Graph(format!("graph {kind} name cannot be empty")));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_core_builds_opaque_nodes() {
        let mut graph = ComputeGraph::with_name("decode_step");
        let hidden = graph
            .add_input("hidden", ValueMeta::tensor(DataType::F32, [1, 16]))
            .unwrap();
        let weight = graph
            .add_external(
                "weight",
                ExternalKey::new("weights", "layer0.proj").unwrap(),
                ValueMeta::tensor(DataType::Bf16, [16, 16]),
            )
            .unwrap();
        let op = OpKey::new("test.dialect", "opaque_transform", 1).unwrap();
        let (_node, outputs) = graph
            .add_node(
                op,
                vec![hidden, weight],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 16])],
            )
            .unwrap();
        graph.set_outputs(outputs.clone()).unwrap();

        graph.validate().unwrap();
        assert_eq!(graph.name(), Some("decode_step"));
        assert_eq!(graph.inputs(), &[hidden]);
        assert_eq!(graph.outputs(), outputs.as_slice());
        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(graph.values().len(), 3);
    }

    #[test]
    fn invalid_node_input_is_rejected() {
        let mut graph = ComputeGraph::new();
        let op = OpKey::new("test.dialect", "bad", 1).unwrap();
        let err = graph
            .add_node(
                op,
                vec![ValueId::from_raw(99)],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1])],
            )
            .unwrap_err();
        assert!(format!("{err}").contains("unknown graph value"));
    }
}
