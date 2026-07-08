//! CPU/reference executor for Ferrule graph programs.
//!
//! This backend is intentionally correctness-first. It executes the opaque graph
//! dialect emitted by the dense decoder translator over host `f32` buffers, while
//! keeping checkpoint/artifact storage outside graph nodes through
//! `BackendObjectStore`.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::graph::{
    AttributeMap, AttributeValue, ComputeGraph, DataType, ExternalKey, TensorData, TensorShape,
    ValueId, ValueMeta, ValueOrigin,
};
use ferrule_common::{Error, Result};
use ferrule_model::{HfSafetensorsInventory, ModelDescriptor};

use crate::artifact_binding::bind_hyper_connection_head_from_artifact_group;

use crate::artifact_tensor::{
    ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice,
};
use crate::backend_object_store::{
    materialize_dense_hf_externals, BackendObject, BackendObjectStore,
};
use crate::dialects::domain;
use crate::expert_executor::CpuReferenceExpertExecutor;
use crate::expert_streaming::{ExpertStreamingPolicy, ExpertStreamingReader};
use crate::graph_builder::{
    build_graph_program_from_descriptor_with_options, GraphProgramBuildOptions,
};
use crate::graph_program::GraphProgram;
use crate::graph_runtime::{ArtifactGroupKind, ExecutionBatch};
use crate::hyper_connection::{hc_head_reference, HyperConnectionConfig};
use crate::layer_binding::{
    bind_layer_artifact_from_graph_objects, new_layer_execution_state_from_graph_objects,
    GraphLayerBindingOptions, LayerArtifactBinding, LayerExecutionState, ReferenceLayerExecutor,
};
use crate::session::SessionId;
use crate::transformer_plan::TransformerLayerPlan;

#[derive(Debug, Clone)]
pub struct ReferenceGraphExecutor {
    program: GraphProgram,
    objects: BackendObjectStore,
    backend: ReferenceGraphBackend,
}

impl ReferenceGraphExecutor {
    pub fn new(
        program: GraphProgram,
        objects: BackendObjectStore,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            program,
            objects,
            backend: ReferenceGraphBackend::new(max_artifact_bytes),
        }
    }

    pub fn from_descriptor(descriptor: &ModelDescriptor, max_artifact_bytes: u64) -> Result<Self> {
        Self::from_descriptor_with_options(
            descriptor,
            max_artifact_bytes,
            GraphProgramBuildOptions::default(),
        )
    }

    pub fn from_descriptor_with_options(
        descriptor: &ModelDescriptor,
        max_artifact_bytes: u64,
        options: GraphProgramBuildOptions,
    ) -> Result<Self> {
        let program = build_graph_program_from_descriptor_with_options(descriptor, options)?;
        let inventory =
            HfSafetensorsInventory::open(&descriptor.path, descriptor.spec.family.clone())?;
        let objects = materialize_dense_hf_externals(&program, &inventory, &descriptor.path)?;
        Ok(Self::new(program, objects, max_artifact_bytes))
    }

    pub fn execute(&mut self, batch: &ExecutionBatch) -> Result<Vec<TensorData>> {
        self.backend
            .execute_program(&self.program, &self.objects, batch)
    }

    pub fn program(&self) -> &GraphProgram {
        &self.program
    }

    pub fn objects(&self) -> &BackendObjectStore {
        &self.objects
    }

    pub fn backend(&self) -> &ReferenceGraphBackend {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut ReferenceGraphBackend {
        &mut self.backend
    }

    pub fn clear_session_kv_state(&mut self, session_id: SessionId) {
        self.backend.clear_session_kv_state(session_id);
    }

    pub fn clear_kv_state(&mut self) {
        self.backend.clear_kv_state();
    }
}

#[derive(Debug, Clone)]
pub struct ReferenceGraphBackend {
    reader: ArtifactTensorReader,
    artifact_cache: BTreeMap<ExternalKey, TensorF32>,
    layer_binding_cache: BTreeMap<usize, Arc<LayerArtifactBinding>>,
    layer_states: BTreeMap<ReferenceLayerStateKey, LayerExecutionState>,
    kv_states: BTreeMap<ReferenceKvKey, ReferenceKvState>,
}

impl ReferenceGraphBackend {
    pub fn new(max_artifact_bytes: u64) -> Self {
        Self {
            reader: ArtifactTensorReader::new(max_artifact_bytes),
            artifact_cache: BTreeMap::new(),
            layer_binding_cache: BTreeMap::new(),
            layer_states: BTreeMap::new(),
            kv_states: BTreeMap::new(),
        }
    }

    pub fn with_reader(reader: ArtifactTensorReader) -> Self {
        Self {
            reader,
            artifact_cache: BTreeMap::new(),
            layer_binding_cache: BTreeMap::new(),
            layer_states: BTreeMap::new(),
            kv_states: BTreeMap::new(),
        }
    }

    pub fn clear_cache(&mut self) {
        self.artifact_cache.clear();
        self.layer_binding_cache.clear();
    }

    pub fn clear_kv_state(&mut self) {
        self.kv_states.clear();
        self.layer_states.clear();
    }

    pub fn clear_session_kv_state(&mut self, session_id: SessionId) {
        self.kv_states
            .retain(|key, _| key.session_id != session_id.0);
        self.layer_states
            .retain(|key, _| key.session_id != session_id.0);
    }

    pub fn kv_state_rows(&self, layer: usize, session_id: SessionId) -> usize {
        self.kv_states
            .get(&ReferenceKvKey {
                layer,
                session_id: session_id.0,
            })
            .map(ReferenceKvState::len)
            .unwrap_or(0)
    }

    pub fn execute_program(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        batch: &ExecutionBatch,
    ) -> Result<Vec<TensorData>> {
        program.graph.validate()?;
        if program.graph.inputs().len() != 2 {
            return Err(Error::Graph(format!(
                "reference graph backend expects token_ids and positions inputs, got {}",
                program.graph.inputs().len()
            )));
        }

        let mut values = BTreeMap::<ValueId, RuntimeValue>::new();
        for node in program.graph.nodes() {
            let inputs = node
                .inputs()
                .iter()
                .map(|input| {
                    self.resolve_value(&program.graph, objects, batch, &mut values, *input)
                })
                .collect::<Result<Vec<_>>>()?;
            let outputs =
                self.execute_node(program, objects, batch, node.op(), node.attrs(), &inputs)?;
            if outputs.len() != node.outputs().len() {
                return Err(Error::Graph(format!(
                    "op {}::{} produced {} outputs, graph expects {}",
                    node.op().domain(),
                    node.op().name(),
                    outputs.len(),
                    node.outputs().len()
                )));
            }
            for (value_id, value) in node.outputs().iter().copied().zip(outputs) {
                values.insert(value_id, value);
            }
        }

        program
            .graph
            .outputs()
            .iter()
            .map(|output| {
                self.resolve_value(&program.graph, objects, batch, &mut values, *output)
                    .and_then(|value| runtime_value_to_tensor_data(&value))
            })
            .collect()
    }

    fn resolve_value(
        &mut self,
        graph: &ComputeGraph,
        objects: &BackendObjectStore,
        batch: &ExecutionBatch,
        values: &mut BTreeMap<ValueId, RuntimeValue>,
        id: ValueId,
    ) -> Result<RuntimeValue> {
        if let Some(value) = values.get(&id) {
            return Ok(value.clone());
        }
        let graph_value = graph.value(id)?;
        let value = match graph_value.origin() {
            ValueOrigin::Input { .. } => input_value_from_batch(graph_value.name(), batch)?,
            ValueOrigin::External { key } => self.external_value(objects, key)?,
            ValueOrigin::Constant { tensor } => tensor_data_to_runtime_value(tensor)?,
            ValueOrigin::NodeOutput { .. } => {
                return Err(Error::Graph(format!(
                    "graph value {} has not been produced yet",
                    id.index()
                )))
            }
        };
        values.insert(id, value.clone());
        Ok(value)
    }

    fn external_value(
        &mut self,
        objects: &BackendObjectStore,
        key: &ExternalKey,
    ) -> Result<RuntimeValue> {
        let object = objects.get(key).ok_or_else(|| {
            Error::Graph(format!(
                "missing backend object for external '{}:{}'",
                key.namespace(),
                key.name()
            ))
        })?;
        match object {
            BackendObject::ArtifactTensor(slice) => {
                Ok(RuntimeValue::F32(self.load_artifact_tensor(key, slice)?))
            }
            BackendObject::ArtifactGroup(_)
            | BackendObject::ExpertRegistry(_)
            | BackendObject::KvState(_)
            | BackendObject::Opaque { .. } => Ok(RuntimeValue::Opaque),
        }
    }

    fn execute_node(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        batch: &ExecutionBatch,
        op: &crate::graph::OpKey,
        attrs: &AttributeMap,
        inputs: &[RuntimeValue],
    ) -> Result<Vec<RuntimeValue>> {
        if is_op(op, domain::TRANSFORMER, "transformer_state_init") {
            return Ok(vec![RuntimeValue::F32(
                self.transformer_state_init(
                    program,
                    objects,
                    inputs
                        .first()
                        .ok_or_else(|| {
                            Error::Graph("transformer_state_init missing hidden".into())
                        })?
                        .f32("transformer_state_init hidden")?,
                )?,
            )]);
        }
        if is_op(op, domain::TRANSFORMER, "transformer_layer") {
            let layer = attr_usize(attrs, "layer")?;
            return Ok(vec![RuntimeValue::F32(
                self.transformer_layer(
                    program,
                    objects,
                    batch,
                    layer,
                    inputs
                        .first()
                        .ok_or_else(|| Error::Graph("transformer_layer missing state".into()))?
                        .f32("transformer_layer state")?,
                    inputs
                        .get(1)
                        .ok_or_else(|| Error::Graph("transformer_layer missing token_ids".into()))?
                        .u32("transformer_layer token_ids")?,
                )?,
            )]);
        }
        if is_op(op, domain::TRANSFORMER, "output_projection") {
            return Ok(vec![RuntimeValue::F32(
                self.output_projection(program, objects, inputs)?,
            )]);
        }
        execute_node_dense(&mut self.kv_states, program, batch, op, attrs, inputs)
    }

    fn transformer_state_init(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        hidden: &TensorF32,
    ) -> Result<TensorF32> {
        let (tokens, hidden_size) = hidden.matrix("transformer_state_init hidden")?;
        let Some(first_layer) = program.runtime_plan.layers.first() else {
            return Ok(hidden.clone());
        };
        let layer_objects = objects.layer_objects(first_layer.index)?;
        let Some(hc_group) = layer_objects.hc_attention else {
            return Ok(hidden.clone());
        };
        let hc_mult = infer_hc_mult_from_group(hc_group)?;
        let hc_dim = hc_mult.checked_mul(hidden_size).ok_or_else(|| {
            Error::Graph(format!(
                "transformer_state_init HC dimension overflow: hc_mult={hc_mult} hidden={hidden_size}"
            ))
        })?;
        let mut state = vec![0.0f32; tokens * hc_dim];
        for token in 0..tokens {
            let src = token * hidden_size;
            let dst = token * hc_dim;
            for copy in 0..hc_mult {
                let copy_dst = dst + copy * hidden_size;
                state[copy_dst..copy_dst + hidden_size]
                    .copy_from_slice(&hidden.data[src..src + hidden_size]);
            }
        }
        TensorF32::new(vec![tokens, hc_dim], state, "transformer_state_init output")
    }

    fn transformer_layer(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        batch: &ExecutionBatch,
        layer: usize,
        state: &TensorF32,
        token_ids: &TensorU32,
    ) -> Result<TensorF32> {
        let layer_plan = layer_plan(program, layer)?;
        let binding = self.graph_layer_binding(program, objects, layer)?;
        let hc_config = binding.inferred_hc_config(
            layer_plan.hyper_connection_epsilon,
            layer_plan.norm_epsilon,
            layer_plan.hyper_connection_sinkhorn_iters,
        )?;
        let executor = ReferenceLayerExecutor::new(
            hc_config,
            ExpertStreamingReader::new(self.reader.max_tensor_bytes()),
            CpuReferenceExpertExecutor::default(),
        );
        let (tokens, hc_dim) = state.matrix("transformer_layer state")?;
        let token_count = token_ids.vector_len("transformer_layer token_ids")?;
        if tokens != batch.len() || token_count != tokens {
            return Err(Error::Graph(format!(
                "transformer_layer token mismatch: state={tokens} token_ids={token_count} batch={}",
                batch.len()
            )));
        }
        if hc_dim != hc_config.hc_hidden_size() {
            return Err(Error::Graph(format!(
                "transformer_layer state width mismatch for layer {layer}: expected {}, got {hc_dim}",
                hc_config.hc_hidden_size()
            )));
        }
        let mut out = Vec::with_capacity(state.data.len());
        for row in 0..tokens {
            let batch_row = batch.rows().get(row).ok_or_else(|| {
                Error::Graph(format!("transformer_layer missing batch row {row}"))
            })?;
            let key = ReferenceLayerStateKey {
                layer,
                session_id: batch_row.session_id.0,
            };
            if let std::collections::btree_map::Entry::Vacant(entry) = self.layer_states.entry(key)
            {
                let layer_objects = objects.layer_objects(layer)?;
                let policy = graph_layer_binding_options(layer_plan).expert_policy;
                let new_state = new_layer_execution_state_from_graph_objects(
                    &layer_objects,
                    binding.as_ref(),
                    policy,
                )?;
                entry.insert(new_state);
            }
            let state_for_layer = self.layer_states.get_mut(&key).expect("inserted above");
            let start = row * hc_dim;
            let step = executor.execute_decode_step(
                binding.as_ref(),
                state_for_layer,
                &state.data[start..start + hc_dim],
                token_ids.data[row],
                &[],
            )?;
            out.extend_from_slice(&step.hc_state);
        }
        TensorF32::new(vec![tokens, hc_dim], out, "transformer_layer output")
    }

    fn output_projection(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        inputs: &[RuntimeValue],
    ) -> Result<TensorF32> {
        let state = inputs
            .first()
            .ok_or_else(|| Error::Graph("output_projection missing state".into()))?
            .f32("output_projection state")?;
        let output_head = inputs
            .iter()
            .rev()
            .find_map(|value| match value {
                RuntimeValue::F32(tensor) if tensor.shape.len() == 2 => Some(tensor),
                _ => None,
            })
            .ok_or_else(|| Error::Graph("output_projection missing output head".into()))?;
        let (vocab, hidden_size) = output_head.matrix("output_projection output_head")?;
        let (tokens, state_width) = state.matrix("output_projection state")?;
        let hidden = if state_width == hidden_size {
            state.clone()
        } else {
            let hc_mult = state_width.checked_div(hidden_size).filter(|value| *value > 0).ok_or_else(|| {
                Error::Graph(format!(
                    "output_projection cannot infer HC multiplier: state_width={state_width} hidden={hidden_size}"
                ))
            })?;
            if hc_mult * hidden_size != state_width {
                return Err(Error::Graph(format!(
                    "output_projection HC state width {state_width} is not divisible by hidden size {hidden_size}"
                )));
            }
            let hc_group = objects
                .artifact_group(ArtifactGroupKind::HyperConnectionHead, None)?
                .ok_or_else(|| {
                    Error::Graph(
                        "output_projection HC state requires hyper-connection head artifacts"
                            .into(),
                    )
                })?;
            let hc_config = HyperConnectionConfig {
                hc_mult,
                hidden_size,
                sinkhorn_iters: program
                    .runtime_plan
                    .layers
                    .first()
                    .map(|layer| layer.hyper_connection_sinkhorn_iters)
                    .unwrap_or(4),
                eps: program
                    .runtime_plan
                    .layers
                    .first()
                    .map(|layer| layer.hyper_connection_epsilon)
                    .unwrap_or(1e-6),
                norm_eps: program
                    .runtime_plan
                    .layers
                    .first()
                    .map(|layer| layer.norm_epsilon)
                    .unwrap_or(1e-6),
            };
            let hc_head =
                bind_hyper_connection_head_from_artifact_group(hc_group, &self.reader, hc_config)?;
            let hidden_data = hc_head_reference(&state.data, tokens, hc_config, &hc_head)?;
            TensorF32::new(
                vec![tokens, hidden_size],
                hidden_data,
                "output_projection hc_head",
            )?
        };
        let normalized = if let Some(output_norm) = inputs.iter().find_map(|value| match value {
            RuntimeValue::F32(tensor) if tensor.shape.len() == 1 => Some(tensor),
            _ => None,
        }) {
            rms_norm(
                &hidden,
                output_norm,
                program
                    .runtime_plan
                    .layers
                    .first()
                    .map(|layer| layer.norm_epsilon)
                    .unwrap_or(1e-6),
            )?
        } else {
            hidden
        };
        let logits = matmul_linear(&normalized, output_head, "output_projection logits")?;
        let (logit_rows, logit_cols) = logits.matrix("output_projection logits")?;
        if logit_rows != tokens || logit_cols != vocab {
            return Err(Error::Graph(format!(
                "output_projection logits shape mismatch: got [{logit_rows}, {logit_cols}], expected [{tokens}, {vocab}]"
            )));
        }
        Ok(logits)
    }

    fn graph_layer_binding(
        &mut self,
        program: &GraphProgram,
        objects: &BackendObjectStore,
        layer: usize,
    ) -> Result<Arc<LayerArtifactBinding>> {
        if let Some(binding) = self.layer_binding_cache.get(&layer) {
            return Ok(binding.clone());
        }
        let layer_plan = layer_plan(program, layer)?;
        let layer_objects = objects.layer_objects(layer)?;
        let options = graph_layer_binding_options(layer_plan);
        let binding = Arc::new(bind_layer_artifact_from_graph_objects(
            &layer_objects,
            layer_plan,
            &self.reader,
            options,
        )?);
        self.layer_binding_cache.insert(layer, binding.clone());
        Ok(binding)
    }

    fn load_artifact_tensor(
        &mut self,
        key: &ExternalKey,
        slice: &ArtifactTensorSlice,
    ) -> Result<TensorF32> {
        if let Some(tensor) = self.artifact_cache.get(key) {
            return Ok(tensor.clone());
        }
        let payload = self.reader.read_slice(slice)?;
        let data = decode_artifact_payload_f32(&payload)?;
        let tensor = TensorF32::new(slice.shape.clone(), data, &slice.name)?;
        self.artifact_cache.insert(key.clone(), tensor.clone());
        Ok(tensor)
    }
}

impl Default for ReferenceGraphBackend {
    fn default() -> Self {
        Self::new(512 * 1024 * 1024)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RuntimeValue {
    F32(TensorF32),
    U32(TensorU32),
    Opaque,
}

impl RuntimeValue {
    fn f32(&self, label: &str) -> Result<&TensorF32> {
        match self {
            Self::F32(value) => Ok(value),
            Self::U32(_) | Self::Opaque => Err(Error::Graph(format!("{label} expects f32 tensor"))),
        }
    }

    fn u32(&self, label: &str) -> Result<&TensorU32> {
        match self {
            Self::U32(value) => Ok(value),
            Self::F32(_) | Self::Opaque => Err(Error::Graph(format!("{label} expects u32 tensor"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TensorF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl TensorF32 {
    fn new(shape: Vec<usize>, data: Vec<f32>, label: &str) -> Result<Self> {
        let expected = element_count(&shape, label)?;
        if data.len() != expected {
            return Err(Error::Graph(format!(
                "{label} data length mismatch: shape {:?} expects {expected}, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    fn matrix(&self, label: &str) -> Result<(usize, usize)> {
        if self.shape.len() != 2 {
            return Err(Error::Graph(format!(
                "{label} expects rank-2 tensor, got shape {:?}",
                self.shape
            )));
        }
        Ok((self.shape[0], self.shape[1]))
    }

    fn vector_len(&self, label: &str) -> Result<usize> {
        if self.shape.len() != 1 {
            return Err(Error::Graph(format!(
                "{label} expects rank-1 tensor, got shape {:?}",
                self.shape
            )));
        }
        Ok(self.shape[0])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorU32 {
    shape: Vec<usize>,
    data: Vec<u32>,
}

impl TensorU32 {
    fn new(shape: Vec<usize>, data: Vec<u32>, label: &str) -> Result<Self> {
        let expected = element_count(&shape, label)?;
        if data.len() != expected {
            return Err(Error::Graph(format!(
                "{label} data length mismatch: shape {:?} expects {expected}, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    fn vector_len(&self, label: &str) -> Result<usize> {
        if self.shape.len() != 1 {
            return Err(Error::Graph(format!(
                "{label} expects rank-1 tensor, got shape {:?}",
                self.shape
            )));
        }
        Ok(self.shape[0])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ReferenceKvKey {
    layer: usize,
    session_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct ReferenceKvEntry {
    position: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct ReferenceKvState {
    key_width: usize,
    value_width: usize,
    entries: Vec<ReferenceKvEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ReferenceLayerStateKey {
    layer: usize,
    session_id: u64,
}

impl ReferenceKvState {
    fn new(key_width: usize, value_width: usize) -> Self {
        Self {
            key_width,
            value_width,
            entries: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn upsert(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.key_width || value.len() != self.value_width {
            return Err(Error::Graph(format!(
                "reference KV width mismatch: expected key={} value={}, got key={} value={}",
                self.key_width,
                self.value_width,
                key.len(),
                value.len()
            )));
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.position == position)
        {
            entry.key.clear();
            entry.key.extend_from_slice(key);
            entry.value.clear();
            entry.value.extend_from_slice(value);
        } else {
            self.entries.push(ReferenceKvEntry {
                position,
                key: key.to_vec(),
                value: value.to_vec(),
            });
            self.entries.sort_by_key(|entry| entry.position);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum AttentionKvRef {
    Persisted(usize),
    Current(usize),
}

fn execute_node_dense(
    kv_states: &mut BTreeMap<ReferenceKvKey, ReferenceKvState>,
    program: &GraphProgram,
    batch: &ExecutionBatch,
    op: &crate::graph::OpKey,
    attrs: &AttributeMap,
    inputs: &[RuntimeValue],
) -> Result<Vec<RuntimeValue>> {
    if is_op(op, domain::TRANSFORMER, "token_embedding") {
        return Ok(vec![RuntimeValue::F32(token_embedding(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("token_embedding missing token ids".into()))?
                .u32("token_embedding token_ids")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("token_embedding missing embedding".into()))?
                .f32("token_embedding embedding")?,
        )?)]);
    }
    if is_op(op, domain::TRANSFORMER, "rms_norm") {
        return Ok(vec![RuntimeValue::F32(rms_norm(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("rms_norm missing input".into()))?
                .f32("rms_norm input")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("rms_norm missing weight".into()))?
                .f32("rms_norm weight")?,
            attr_float(attrs, "epsilon", 1e-6)?,
        )?)]);
    }
    if is_op(op, domain::TRANSFORMER, "linear") {
        return Ok(vec![RuntimeValue::F32(linear(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("linear missing input".into()))?
                .f32("linear input")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("linear missing weight".into()))?
                .f32("linear weight")?,
        )?)]);
    }
    if is_op(op, domain::TRANSFORMER, "rope") {
        let layer = attr_usize(attrs, "layer")?;
        let layer_plan = layer_plan(program, layer)?;
        let theta = attr_float(attrs, "theta", 10_000.0)?;
        let q = apply_rope(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("rope missing q".into()))?
                .f32("rope q")?,
            inputs
                .get(2)
                .ok_or_else(|| Error::Graph("rope missing positions".into()))?
                .u32("rope positions")?,
            required_attention_dim(layer_plan.attention.head_dim, "head_dim", layer)?,
            theta,
            "rope q",
        )?;
        let k = apply_rope(
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("rope missing k".into()))?
                .f32("rope k")?,
            inputs
                .get(2)
                .ok_or_else(|| Error::Graph("rope missing positions".into()))?
                .u32("rope positions")?,
            required_attention_dim(layer_plan.attention.head_dim, "head_dim", layer)?,
            theta,
            "rope k",
        )?;
        return Ok(vec![RuntimeValue::F32(q), RuntimeValue::F32(k)]);
    }
    if is_op(op, domain::TRANSFORMER, "causal_attention") {
        let layer = attr_usize(attrs, "layer")?;
        let layer_plan = layer_plan(program, layer)?;
        return Ok(vec![RuntimeValue::F32(causal_attention(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("causal_attention missing q".into()))?
                .f32("causal_attention q")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("causal_attention missing k".into()))?
                .f32("causal_attention k")?,
            inputs
                .get(2)
                .ok_or_else(|| Error::Graph("causal_attention missing v".into()))?
                .f32("causal_attention v")?,
            batch,
            layer_plan,
            kv_states,
        )?)]);
    }
    if is_op(op, domain::TRANSFORMER, "swiglu_ffn") {
        return Ok(vec![RuntimeValue::F32(swiglu_ffn(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("swiglu_ffn missing input".into()))?
                .f32("swiglu_ffn input")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("swiglu_ffn missing gate".into()))?
                .f32("swiglu_ffn gate")?,
            inputs
                .get(2)
                .ok_or_else(|| Error::Graph("swiglu_ffn missing up".into()))?
                .f32("swiglu_ffn up")?,
            inputs
                .get(3)
                .ok_or_else(|| Error::Graph("swiglu_ffn missing down".into()))?
                .f32("swiglu_ffn down")?,
        )?)]);
    }
    if is_op(op, domain::TRANSFORMER, "logits_select") {
        return Ok(vec![RuntimeValue::F32(logits_select(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("logits_select missing logits".into()))?
                .f32("logits_select logits")?,
            batch,
        )?)]);
    }
    if is_op(op, domain::TENSOR, "residual_add") {
        return Ok(vec![RuntimeValue::F32(residual_add(
            inputs
                .first()
                .ok_or_else(|| Error::Graph("residual_add missing lhs".into()))?
                .f32("residual_add lhs")?,
            inputs
                .get(1)
                .ok_or_else(|| Error::Graph("residual_add missing rhs".into()))?
                .f32("residual_add rhs")?,
        )?)]);
    }
    Err(Error::Graph(format!(
        "reference graph backend does not support op {}::{} v{}",
        op.domain(),
        op.name(),
        op.version()
    )))
}

fn input_value_from_batch(name: Option<&str>, batch: &ExecutionBatch) -> Result<RuntimeValue> {
    match name {
        Some("token_ids") => Ok(RuntimeValue::U32(TensorU32::new(
            vec![batch.len()],
            batch.token_ids().collect(),
            "token_ids",
        )?)),
        Some("positions") => {
            let positions = batch
                .positions()
                .map(|position| {
                    u32::try_from(position).map_err(|_| {
                        Error::Graph(format!("position {position} exceeds u32 graph input range"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(RuntimeValue::U32(TensorU32::new(
                vec![batch.len()],
                positions,
                "positions",
            )?))
        }
        Some(other) => Err(Error::Graph(format!(
            "reference graph backend does not know input '{other}'"
        ))),
        None => Err(Error::Graph("graph input is missing a name".into())),
    }
}

fn token_embedding(token_ids: &TensorU32, embedding: &TensorF32) -> Result<TensorF32> {
    let tokens = token_ids.vector_len("token_embedding token_ids")?;
    let (vocab, hidden) = embedding.matrix("token_embedding embedding")?;
    let mut output = vec![0.0f32; tokens * hidden];
    for (row, token_id) in token_ids.data.iter().copied().enumerate() {
        let token = token_id as usize;
        if token >= vocab {
            return Err(Error::Graph(format!(
                "token id {token} exceeds embedding vocab {vocab}"
            )));
        }
        let src = token * hidden;
        let dst = row * hidden;
        output[dst..dst + hidden].copy_from_slice(&embedding.data[src..src + hidden]);
    }
    TensorF32::new(vec![tokens, hidden], output, "token_embedding output")
}

fn rms_norm(input: &TensorF32, weight: &TensorF32, eps: f32) -> Result<TensorF32> {
    let (rows, cols) = input.matrix("rms_norm input")?;
    let weight_len = weight.vector_len("rms_norm weight")?;
    if weight_len != cols {
        return Err(Error::Graph(format!(
            "rms_norm weight length mismatch: expected {cols}, got {weight_len}"
        )));
    }
    let mut output = vec![0.0f32; input.data.len()];
    for row in 0..rows {
        let base = row * cols;
        let mean_sq = input.data[base..base + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cols as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for col in 0..cols {
            output[base + col] = input.data[base + col] * scale * weight.data[col];
        }
    }
    TensorF32::new(input.shape.clone(), output, "rms_norm output")
}

fn linear(input: &TensorF32, weight: &TensorF32) -> Result<TensorF32> {
    matmul_linear(input, weight, "linear")
}

fn matmul_linear(input: &TensorF32, weight: &TensorF32, label: &str) -> Result<TensorF32> {
    let (rows, in_features) = input.matrix(label)?;
    let (out_features, weight_in) = weight.matrix(label)?;
    if weight_in != in_features {
        return Err(Error::Graph(format!(
            "{label} input width mismatch: input={in_features}, weight={weight_in}"
        )));
    }
    let mut output = vec![0.0f32; rows * out_features];
    for row in 0..rows {
        for out in 0..out_features {
            let mut acc = 0.0f32;
            let input_base = row * in_features;
            let weight_base = out * in_features;
            for col in 0..in_features {
                acc += input.data[input_base + col] * weight.data[weight_base + col];
            }
            output[row * out_features + out] = acc;
        }
    }
    TensorF32::new(vec![rows, out_features], output, label)
}

fn apply_rope(
    input: &TensorF32,
    positions: &TensorU32,
    head_dim: usize,
    theta: f32,
    label: &str,
) -> Result<TensorF32> {
    if head_dim == 0 {
        return Err(Error::Graph(format!("{label} requires non-zero head_dim")));
    }
    let (tokens, width) = input.matrix(label)?;
    let position_count = positions.vector_len("rope positions")?;
    if position_count != tokens {
        return Err(Error::Graph(format!(
            "{label} position count mismatch: tokens={tokens}, positions={position_count}"
        )));
    }
    if !width.is_multiple_of(head_dim) {
        return Err(Error::Graph(format!(
            "{label} width {width} is not divisible by head_dim {head_dim}"
        )));
    }
    let heads = width / head_dim;
    let mut output = input.data.clone();
    let pairs = head_dim / 2;
    for token in 0..tokens {
        let position = positions.data[token] as f32;
        for head in 0..heads {
            let head_base = token * width + head * head_dim;
            for pair in 0..pairs {
                let even = head_base + pair * 2;
                let odd = even + 1;
                let inv_freq = theta.powf(-((pair * 2) as f32) / head_dim as f32);
                let angle = position * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = input.data[even];
                let x1 = input.data[odd];
                output[even] = x0 * cos - x1 * sin;
                output[odd] = x0 * sin + x1 * cos;
            }
        }
    }
    TensorF32::new(input.shape.clone(), output, label)
}

fn causal_attention(
    q: &TensorF32,
    k: &TensorF32,
    v: &TensorF32,
    batch: &ExecutionBatch,
    layer: &TransformerLayerPlan,
    kv_states: &mut BTreeMap<ReferenceKvKey, ReferenceKvState>,
) -> Result<TensorF32> {
    let (tokens, q_width) = q.matrix("causal_attention q")?;
    let (k_tokens, k_width) = k.matrix("causal_attention k")?;
    let (v_tokens, v_width) = v.matrix("causal_attention v")?;
    if tokens != batch.len() || k_tokens != tokens || v_tokens != tokens {
        return Err(Error::Graph(format!(
            "causal_attention token mismatch: batch={} q={tokens} k={k_tokens} v={v_tokens}",
            batch.len()
        )));
    }
    let num_heads = required_attention_dim(layer.attention.num_heads, "num_heads", layer.index)?;
    let kv_heads = layer.attention.num_kv_heads.unwrap_or(num_heads);
    let head_dim = required_attention_dim(layer.attention.head_dim, "head_dim", layer.index)?;
    if q_width != num_heads * head_dim {
        return Err(Error::Graph(format!(
            "causal_attention q width mismatch: expected {}, got {q_width}",
            num_heads * head_dim
        )));
    }
    if k_width != kv_heads * head_dim || v_width != kv_heads * head_dim {
        return Err(Error::Graph(format!(
            "causal_attention kv width mismatch: expected {}, got k={k_width} v={v_width}",
            kv_heads * head_dim
        )));
    }
    if !num_heads.is_multiple_of(kv_heads) {
        return Err(Error::Graph(format!(
            "causal_attention requires num_heads divisible by num_kv_heads, got {num_heads}/{kv_heads}"
        )));
    }

    let scale = (head_dim as f32).powf(-0.5);
    let mut output = vec![0.0f32; tokens * q_width];
    let rows = batch.rows();
    for query_row in 0..tokens {
        let query = rows[query_row];
        let state_key = ReferenceKvKey {
            layer: layer.index,
            session_id: query.session_id.0,
        };
        for head in 0..num_heads {
            let kv_head = head / (num_heads / kv_heads);
            let q_base = query_row * q_width + head * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::<(AttentionKvRef, f32)>::new();

            if let Some(state) = kv_states.get(&state_key) {
                for (entry_index, entry) in state.entries.iter().enumerate() {
                    if entry.position > query.position
                        || batch_has_position(rows, query.session_id.0, entry.position)
                    {
                        continue;
                    }
                    let k_base = kv_head * head_dim;
                    let score = dot(
                        &q.data[q_base..q_base + head_dim],
                        &entry.key[k_base..k_base + head_dim],
                    ) * scale;
                    scores.push((AttentionKvRef::Persisted(entry_index), score));
                    max_score = max_score.max(score);
                }
            }

            for key_row in 0..tokens {
                if rows[key_row].session_id != query.session_id
                    || rows[key_row].position > query.position
                {
                    continue;
                }
                let k_base = key_row * k_width + kv_head * head_dim;
                let score = dot(
                    &q.data[q_base..q_base + head_dim],
                    &k.data[k_base..k_base + head_dim],
                ) * scale;
                scores.push((AttentionKvRef::Current(key_row), score));
                max_score = max_score.max(score);
            }

            if !max_score.is_finite() {
                return Err(Error::Graph(format!(
                    "causal_attention found no visible keys for row {query_row}"
                )));
            }
            let mut denom = 0.0f32;
            for (_, score) in &scores {
                denom += (*score - max_score).exp();
            }
            if denom == 0.0 || !denom.is_finite() {
                return Err(Error::Graph(
                    "causal_attention softmax denominator is invalid".into(),
                ));
            }
            let out_base = query_row * q_width + head * head_dim;
            for (kv_ref, score) in scores {
                let weight = (score - max_score).exp() / denom;
                match kv_ref {
                    AttentionKvRef::Persisted(entry_index) => {
                        let state = kv_states.get(&state_key).ok_or_else(|| {
                            Error::Graph("reference KV state disappeared during attention".into())
                        })?;
                        let entry = &state.entries[entry_index];
                        let v_base = kv_head * head_dim;
                        for dim in 0..head_dim {
                            output[out_base + dim] += weight * entry.value[v_base + dim];
                        }
                    }
                    AttentionKvRef::Current(key_row) => {
                        let v_base = key_row * v_width + kv_head * head_dim;
                        for dim in 0..head_dim {
                            output[out_base + dim] += weight * v.data[v_base + dim];
                        }
                    }
                }
            }
        }
    }

    for row in 0..tokens {
        let row_info = rows[row];
        let state_key = ReferenceKvKey {
            layer: layer.index,
            session_id: row_info.session_id.0,
        };
        let state = kv_states
            .entry(state_key)
            .or_insert_with(|| ReferenceKvState::new(k_width, v_width));
        if state.key_width != k_width || state.value_width != v_width {
            return Err(Error::Graph(format!(
                "reference KV state width changed for layer {} session {}: existing key={} value={}, new key={k_width} value={v_width}",
                layer.index,
                row_info.session_id.0,
                state.key_width,
                state.value_width
            )));
        }
        let k_base = row * k_width;
        let v_base = row * v_width;
        state.upsert(
            row_info.position,
            &k.data[k_base..k_base + k_width],
            &v.data[v_base..v_base + v_width],
        )?;
    }

    TensorF32::new(vec![tokens, q_width], output, "causal_attention output")
}

fn residual_add(lhs: &TensorF32, rhs: &TensorF32) -> Result<TensorF32> {
    if lhs.shape != rhs.shape {
        return Err(Error::Graph(format!(
            "residual_add shape mismatch: lhs={:?} rhs={:?}",
            lhs.shape, rhs.shape
        )));
    }
    Ok(TensorF32 {
        shape: lhs.shape.clone(),
        data: lhs
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect(),
    })
}

fn swiglu_ffn(
    input: &TensorF32,
    gate: &TensorF32,
    up: &TensorF32,
    down: &TensorF32,
) -> Result<TensorF32> {
    let gate_out = matmul_linear(input, gate, "swiglu_ffn gate")?;
    let up_out = matmul_linear(input, up, "swiglu_ffn up")?;
    if gate_out.shape != up_out.shape {
        return Err(Error::Graph(format!(
            "swiglu_ffn gate/up shape mismatch: gate={:?} up={:?}",
            gate_out.shape, up_out.shape
        )));
    }
    let (tokens, intermediate) = gate_out.matrix("swiglu_ffn hidden")?;
    let hidden = TensorF32::new(
        vec![tokens, intermediate],
        gate_out
            .data
            .iter()
            .zip(up_out.data.iter())
            .map(|(gate, up)| silu(*gate) * up)
            .collect(),
        "swiglu_ffn activation",
    )?;
    matmul_linear(&hidden, down, "swiglu_ffn down")
}

fn logits_select(logits: &TensorF32, batch: &ExecutionBatch) -> Result<TensorF32> {
    let (tokens, vocab) = logits.matrix("logits_select")?;
    if tokens != batch.len() {
        return Err(Error::Graph(format!(
            "logits_select token mismatch: logits={tokens}, batch={}",
            batch.len()
        )));
    }
    let selected = batch
        .logits_rows()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(selected.len() * vocab);
    for row in selected.iter().copied() {
        if row >= tokens {
            return Err(Error::Graph(format!(
                "logits_select row {row} exceeds logits rows {tokens}"
            )));
        }
        let base = row * vocab;
        output.extend_from_slice(&logits.data[base..base + vocab]);
    }
    TensorF32::new(vec![selected.len(), vocab], output, "logits_select output")
}

fn tensor_data_to_runtime_value(tensor: &TensorData) -> Result<RuntimeValue> {
    let shape = known_shape(tensor.meta())?;
    match tensor.meta().dtype.as_ref() {
        Some(DataType::F32) => Ok(RuntimeValue::F32(TensorF32::new(
            shape,
            bytes_to_f32(tensor.bytes())?,
            "constant f32",
        )?)),
        Some(DataType::U32) => Ok(RuntimeValue::U32(TensorU32::new(
            shape,
            bytes_to_u32(tensor.bytes())?,
            "constant u32",
        )?)),
        other => Err(Error::Graph(format!(
            "reference graph backend does not support constant dtype {other:?}"
        ))),
    }
}

fn runtime_value_to_tensor_data(value: &RuntimeValue) -> Result<TensorData> {
    match value {
        RuntimeValue::F32(tensor) => Ok(TensorData::new(
            ValueMeta::tensor(DataType::F32, TensorShape::from(tensor.shape.clone())),
            f32_to_bytes(&tensor.data),
        )),
        RuntimeValue::U32(tensor) => Ok(TensorData::new(
            ValueMeta::tensor(DataType::U32, TensorShape::from(tensor.shape.clone())),
            u32_to_bytes(&tensor.data),
        )),
        RuntimeValue::Opaque => Err(Error::Graph(
            "cannot return opaque graph value as TensorData".into(),
        )),
    }
}

fn decode_artifact_payload_f32(payload: &ArtifactTensorPayload) -> Result<Vec<f32>> {
    let elements = payload.slice.element_count()?;
    match payload.slice.dtype {
        ArtifactDType::F32 => {
            ensure_byte_len(payload.bytes.len(), elements, 4, &payload.slice.name)?;
            Ok(bytes_to_f32(&payload.bytes)?)
        }
        ArtifactDType::Bf16 => {
            ensure_byte_len(payload.bytes.len(), elements, 2, &payload.slice.name)?;
            Ok(payload
                .bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect())
        }
        ref other => Err(Error::Graph(format!(
            "artifact tensor '{}' dtype {} cannot be decoded as f32 by reference graph backend",
            payload.slice.name,
            other.as_str()
        ))),
    }
}

fn known_shape(meta: &ValueMeta) -> Result<Vec<usize>> {
    meta.shape
        .dims()
        .iter()
        .map(|dim| match dim {
            crate::graph::Dim::Known(value) => Ok(*value),
            other => Err(Error::Graph(format!(
                "reference graph backend cannot materialize symbolic/dynamic constant shape {other:?}"
            ))),
        })
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Graph(format!(
            "f32 byte buffer length {} is not divisible by 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bytes_to_u32(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Graph(format!(
            "u32 byte buffer length {} is not divisible by 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn ensure_byte_len(actual: usize, elements: usize, element_size: usize, label: &str) -> Result<()> {
    let expected = elements.checked_mul(element_size).ok_or_else(|| {
        Error::Graph(format!(
            "artifact tensor '{label}' byte length overflows: elements={elements} element_size={element_size}"
        ))
    })?;
    if actual != expected {
        return Err(Error::Graph(format!(
            "artifact tensor '{label}' byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn element_count(shape: &[usize], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::Graph(format!("{label} shape {:?} element count overflows", shape))
        })
    })
}

fn attr_float(attrs: &AttributeMap, key: &str, default: f32) -> Result<f32> {
    match attrs.get(key) {
        Some(AttributeValue::Float(value)) => Ok(*value as f32),
        Some(other) => Err(Error::Graph(format!(
            "attribute '{key}' expects float, got {other:?}"
        ))),
        None => Ok(default),
    }
}

fn attr_usize(attrs: &AttributeMap, key: &str) -> Result<usize> {
    match attrs.get(key) {
        Some(AttributeValue::UInt(value)) => usize::try_from(*value)
            .map_err(|_| Error::Graph(format!("attribute '{key}' value {value} exceeds usize"))),
        Some(other) => Err(Error::Graph(format!(
            "attribute '{key}' expects unsigned integer, got {other:?}"
        ))),
        None => Err(Error::Graph(format!("missing required attribute '{key}'"))),
    }
}

fn layer_plan(program: &GraphProgram, layer: usize) -> Result<&TransformerLayerPlan> {
    program
        .runtime_plan
        .layers
        .iter()
        .find(|candidate| candidate.index == layer)
        .ok_or_else(|| Error::Graph(format!("runtime plan has no layer {layer}")))
}

fn graph_layer_binding_options(layer: &TransformerLayerPlan) -> GraphLayerBindingOptions {
    let top_k = layer.feed_forward.num_experts_per_tok.unwrap_or(1).max(1);
    GraphLayerBindingOptions {
        hc_eps: layer.hyper_connection_epsilon,
        norm_eps: layer.norm_epsilon,
        hc_sinkhorn_iters: layer.hyper_connection_sinkhorn_iters,
        swiglu_limit: layer.feed_forward.swiglu_limit.unwrap_or(0.0),
        route_scale: layer.feed_forward.route_scale.unwrap_or(1.0),
        attention_topk: layer.attention.window_size.or(layer.attention.index_topk),
        expert_policy: ExpertStreamingPolicy::quality_first(top_k),
    }
}

fn infer_hc_mult_from_group(
    group: &crate::backend_object_store::ArtifactObjectGroup,
) -> Result<usize> {
    let mix_hc = group
        .tensors
        .iter()
        .filter(|tensor| tensor.shape.len() == 1)
        .filter_map(|tensor| tensor.shape.first().copied())
        .max()
        .ok_or_else(|| {
            Error::Graph(format!(
                "cannot infer HC multiplier from artifact group {} layer={:?}",
                group.kind.as_str(),
                group.layer
            ))
        })?;
    (1..=mix_hc)
        .find(|hc_mult| (hc_mult + 2) * hc_mult == mix_hc)
        .ok_or_else(|| {
            Error::Graph(format!(
                "invalid HC mix length {mix_hc} in artifact group {} layer={:?}",
                group.kind.as_str(),
                group.layer
            ))
        })
}

fn required_attention_dim(value: Option<usize>, name: &str, layer: usize) -> Result<usize> {
    value.ok_or_else(|| Error::Graph(format!("layer {layer} missing attention {name}")))
}

fn batch_has_position(
    rows: &[crate::graph_runtime::ExecutionRow],
    session_id: u64,
    position: usize,
) -> bool {
    rows.iter()
        .any(|row| row.session_id.0 == session_id && row.position == position)
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn is_op(op: &crate::graph::OpKey, domain: &str, name: &str) -> bool {
    op.domain() == domain && op.name() == name
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::graph::{
        AttributeMap, AttributeValue, ComputeGraph, DataType, ExternalKey, ValueMeta,
    };
    use ferrule_model::{
        AttentionKind, FeedForwardKind, KvCacheShape, ModelFamily, MoeSpec, PolicySet, RouterKind,
        TensorRole, TransformerSpec, WeightSource,
    };

    use super::*;
    use crate::artifact_tensor::{ArtifactDType, ArtifactTensorSlice};
    use crate::backend_object_store::{
        ArtifactObjectGroup, BackendObject, BackendObjectStore, ExpertRegistryObject,
    };
    use crate::dialects::transformer_ops;
    use crate::expert_streaming::{
        ExpertId, ExpertLoadSource, ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey,
        ExpertTensorPayload, ExpertTensorSlice,
    };
    use crate::graph_program::{GraphProgram, GraphProgramProfile};
    use crate::graph_runtime::{
        ArtifactGroupKind, ExecutionBatch, ExecutionSegment, ExternalBindingPlan, LogitsSelection,
    };
    use crate::session::SessionId;
    use crate::transformer_plan::{
        AttentionStepPlan, ExpertResidencyMode, FeedForwardStepPlan, RuntimeEpilogue,
        RuntimePrologue, TransformerLayerPlan, TransformerRuntimePlan,
    };

    #[test]
    fn reference_backend_executes_tiny_embedding_head_graph() -> Result<()> {
        let (program, objects, batch, dir) = tiny_embedding_head_fixture()?;
        let mut backend = ReferenceGraphBackend::new(1024);
        let outputs = backend.execute_program(&program, &objects, &batch)?;
        assert_tiny_embedding_head_outputs(&outputs)?;

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn reference_graph_executor_executes_tiny_embedding_head_graph() -> Result<()> {
        let (program, objects, batch, dir) = tiny_embedding_head_fixture()?;
        let mut executor = ReferenceGraphExecutor::new(program, objects, 1024);
        let outputs = executor.execute(&batch)?;
        assert_tiny_embedding_head_outputs(&outputs)?;

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn reference_backend_executes_tiny_semantic_transformer_layer_graph() -> Result<()> {
        let (program, objects, batch, dir) = tiny_semantic_layer_fixture()?;
        let mut backend = ReferenceGraphBackend::new(64 * 1024);
        let outputs = backend.execute_program(&program, &objects, &batch)?;
        assert_eq!(outputs.len(), 1);
        let logits = bytes_to_f32(outputs[0].bytes())?;
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(backend.layer_binding_cache.len(), 1);
        assert_eq!(backend.layer_states.len(), 1);

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    fn tiny_embedding_head_fixture(
    ) -> Result<(GraphProgram, BackendObjectStore, ExecutionBatch, PathBuf)> {
        let dir = unique_temp_dir("ferrule-reference-graph-backend");
        std::fs::create_dir_all(&dir).unwrap();
        let artifact_path = dir.join("tiny.safetensors.payload");
        let embedding = f32_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let head_offset = embedding.len() as u64;
        let head = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let mut bytes = embedding.clone();
        bytes.extend_from_slice(&head);
        std::fs::write(&artifact_path, bytes).unwrap();

        let embedding_key = ExternalKey::new("weights", "token_embedding")?;
        let head_key = ExternalKey::new("weights", "output_head")?;
        let mut objects = BackendObjectStore::new();
        objects.insert_required(
            embedding_key.clone(),
            BackendObject::ArtifactTensor(slice(
                &artifact_path,
                "token_embedding.weight",
                TensorRole::TokenEmbedding,
                0,
                embedding.len() as u64,
                vec![3, 2],
            )),
        )?;
        objects.insert_required(
            head_key.clone(),
            BackendObject::ArtifactTensor(slice(
                &artifact_path,
                "lm_head.weight",
                TensorRole::OutputHead,
                head_offset,
                head.len() as u64,
                vec![2, 2],
            )),
        )?;

        let mut graph = ComputeGraph::with_name("tiny reference graph");
        let token_ids = graph.add_input("token_ids", ValueMeta::token_ids([2]))?;
        let _positions = graph.add_input("positions", ValueMeta::tensor(DataType::U32, [2]))?;
        let embedding_value = graph.add_external(
            "token_embedding",
            embedding_key,
            ValueMeta::tensor(DataType::F32, [3, 2]),
        )?;
        let hidden = graph
            .add_node(
                transformer_ops::token_embedding()?,
                vec![token_ids, embedding_value],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [2, 2])],
            )?
            .1[0];
        let head_value = graph.add_external(
            "output_head",
            head_key,
            ValueMeta::tensor(DataType::F32, [2, 2]),
        )?;
        let logits = graph
            .add_node(
                transformer_ops::linear()?,
                vec![hidden, head_value],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [2, 2])],
            )?
            .1[0];
        let selected = graph
            .add_node(
                transformer_ops::logits_select()?,
                vec![logits],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 2])],
            )?
            .1[0];
        graph.set_outputs(vec![selected])?;

        let program = GraphProgram::new(
            graph,
            ExternalBindingPlan::new(),
            tiny_runtime_plan(),
            GraphProgramProfile::default(),
        )?;
        let batch = ExecutionBatch::from_tokens(
            ExecutionSegment::Prefill,
            SessionId(1),
            0,
            &[0, 2],
            None,
            LogitsSelection::Last,
        )?;
        Ok((program, objects, batch, dir))
    }

    fn tiny_semantic_layer_fixture(
    ) -> Result<(GraphProgram, BackendObjectStore, ExecutionBatch, PathBuf)> {
        let dir = unique_temp_dir("ferrule-reference-semantic-layer");
        std::fs::create_dir_all(&dir).unwrap();
        let artifact_path = dir.join("semantic-layer.bin");
        let expert_path = dir.join("expert.bin");
        let mut writer = ArtifactWriter::new(artifact_path.clone());

        let embedding_key = ExternalKey::new("weights", "token_embedding")?;
        let output_norm_key = ExternalKey::new("weights", "output_norm")?;
        let output_head_key = ExternalKey::new("weights", "output_head")?;
        let attention_key = ExternalKey::new("artifacts", "layers.0.attention_artifacts")?;
        let hc_attention_key = ExternalKey::new("artifacts", "layers.0.hc_attention_artifacts")?;
        let hc_ffn_key = ExternalKey::new("artifacts", "layers.0.hc_feed_forward_artifacts")?;
        let router_key = ExternalKey::new("artifacts", "layers.0.router_artifacts")?;
        let shared_key = ExternalKey::new("artifacts", "layers.0.shared_expert_artifacts")?;
        let expert_key = ExternalKey::new("experts", "layers.0.routed_expert_registry")?;
        let kv_key = ExternalKey::new("state", "layers.0.kv_state")?;
        let hc_head_key = ExternalKey::new("artifacts", "hc_head_artifacts")?;

        let mut objects = BackendObjectStore::new();
        objects.insert_required(
            embedding_key.clone(),
            BackendObject::ArtifactTensor(writer.f32_slice(
                "token_embedding.weight",
                TensorRole::TokenEmbedding,
                &[1.0; 32],
                vec![1, 32],
            )),
        )?;
        objects.insert_required(
            output_norm_key.clone(),
            BackendObject::ArtifactTensor(writer.f32_slice(
                "output_norm.weight",
                TensorRole::OutputNorm,
                &[1.0; 32],
                vec![32],
            )),
        )?;
        objects.insert_required(
            output_head_key.clone(),
            BackendObject::ArtifactTensor(writer.f32_slice(
                "output_head.weight",
                TensorRole::OutputHead,
                &one_hot_rows(2, 32, &[(0, 0, 1.0), (1, 1, 1.0)]),
                vec![2, 32],
            )),
        )?;

        let attention_tensors = vec![
            writer.f32_slice(
                "wq_a.weight",
                TensorRole::AttentionLatentQueryA,
                &one_hot_rows(4, 32, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
                vec![4, 32],
            ),
            writer.f32_slice(
                "wq_b.weight",
                TensorRole::AttentionLatentQueryB,
                &one_hot_rows(32, 4, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
                vec![32, 4],
            ),
            writer.f32_slice(
                "wkv.weight",
                TensorRole::AttentionLatentKv,
                &one_hot_rows(
                    32,
                    32,
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
                vec![32, 32],
            ),
            writer.f32_slice(
                "wo_a.weight",
                TensorRole::AttentionLatentOutputA,
                &one_hot_rows(4, 32, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
                vec![4, 32],
            ),
            writer.f32_slice(
                "wo_b.weight",
                TensorRole::AttentionLatentOutputB,
                &one_hot_rows(32, 4, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]),
                vec![32, 4],
            ),
            writer.f32_slice(
                "q_norm.weight",
                TensorRole::AttentionQueryNorm,
                &[1.0; 4],
                vec![4],
            ),
            writer.f32_slice(
                "kv_norm.weight",
                TensorRole::AttentionKeyValueNorm,
                &[1.0; 32],
                vec![32],
            ),
            writer.f32_slice("attention_sink", TensorRole::AttentionSink, &[0.0], vec![1]),
        ];
        objects.insert_required(
            attention_key.clone(),
            BackendObject::ArtifactGroup(ArtifactObjectGroup {
                kind: ArtifactGroupKind::Attention,
                layer: Some(0),
                tensors: attention_tensors,
            }),
        )?;

        objects.insert_required(
            hc_attention_key.clone(),
            BackendObject::ArtifactGroup(hc_group(
                &mut writer,
                ArtifactGroupKind::HyperConnectionAttention,
            )),
        )?;
        objects.insert_required(
            hc_ffn_key.clone(),
            BackendObject::ArtifactGroup(hc_group(
                &mut writer,
                ArtifactGroupKind::HyperConnectionFeedForward,
            )),
        )?;
        objects.insert_required(
            hc_head_key.clone(),
            BackendObject::ArtifactGroup(hc_head_group(&mut writer)),
        )?;
        objects.insert_required(
            router_key.clone(),
            BackendObject::ArtifactGroup(ArtifactObjectGroup {
                kind: ArtifactGroupKind::Router,
                layer: Some(0),
                tensors: vec![
                    writer.f32_slice(
                        "router.weight",
                        TensorRole::RouterLogits,
                        &[0.0; 32],
                        vec![1, 32],
                    ),
                    writer.i64_slice("router.hash", TensorRole::HashRouterTable, &[0], vec![1, 1]),
                ],
            }),
        )?;
        objects.insert_required(
            shared_key.clone(),
            BackendObject::ArtifactGroup(ArtifactObjectGroup {
                kind: ArtifactGroupKind::SharedExpert,
                layer: Some(0),
                tensors: vec![
                    writer.f32_slice(
                        "shared_gate.weight",
                        TensorRole::SharedExpertGate,
                        &one_hot_rows(1, 32, &[(0, 0, 1.0)]),
                        vec![1, 32],
                    ),
                    writer.f32_slice(
                        "shared_up.weight",
                        TensorRole::SharedExpertUp,
                        &one_hot_rows(1, 32, &[(0, 1, 1.0)]),
                        vec![1, 32],
                    ),
                    writer.f32_slice(
                        "shared_down.weight",
                        TensorRole::SharedExpertDown,
                        &one_hot_rows(32, 1, &[(0, 0, 1.0)]),
                        vec![32, 1],
                    ),
                ],
            }),
        )?;
        writer.finish();

        let expert = ExpertId::new(0, 0);
        objects.insert_required(
            expert_key.clone(),
            BackendObject::ExpertRegistry(ExpertRegistryObject {
                layer: 0,
                experts: BTreeMap::from([(
                    expert,
                    ExpertLoadSource::LocalTensorSet {
                        tensors: write_tiny_expert(&expert_path, expert),
                    },
                )]),
            }),
        )?;
        objects.insert_required(kv_key.clone(), BackendObject::KvState(None))?;

        let mut graph = ComputeGraph::with_name("tiny semantic transformer graph");
        let token_ids = graph.add_input("token_ids", ValueMeta::token_ids([1]))?;
        let positions = graph.add_input("positions", ValueMeta::tensor(DataType::U32, [1]))?;
        let embedding = graph.add_external(
            "token_embedding",
            embedding_key,
            ValueMeta::tensor(DataType::F32, [1, 32]),
        )?;
        let hidden = graph
            .add_node(
                transformer_ops::token_embedding()?,
                vec![token_ids, embedding],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 32])],
            )?
            .1[0];
        let state = graph
            .add_node(
                transformer_ops::transformer_state_init()?,
                vec![hidden, token_ids, positions],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 64])],
            )?
            .1[0];
        let mut layer_inputs = vec![state, token_ids, positions];
        for (name, key) in [
            ("attention_artifacts", attention_key),
            ("hc_attention_artifacts", hc_attention_key),
            ("hc_feed_forward_artifacts", hc_ffn_key),
            ("router_artifacts", router_key),
            ("shared_expert_artifacts", shared_key),
            ("routed_expert_registry", expert_key),
            ("kv_state", kv_key),
        ] {
            layer_inputs.push(graph.add_external(name, key, ValueMeta::external_state(name))?);
        }
        let layer_out = graph
            .add_node(
                transformer_ops::transformer_layer()?,
                layer_inputs,
                attrs_u64("layer", 0),
                vec![ValueMeta::tensor(DataType::F32, [1, 64])],
            )?
            .1[0];
        let output_norm = graph.add_external(
            "output_norm",
            output_norm_key,
            ValueMeta::tensor(DataType::F32, [32]),
        )?;
        let hc_head = graph.add_external(
            "hc_head_artifacts",
            hc_head_key,
            ValueMeta::external_state("hc_head_artifacts"),
        )?;
        let output_head = graph.add_external(
            "output_head",
            output_head_key,
            ValueMeta::tensor(DataType::F32, [2, 32]),
        )?;
        let logits = graph
            .add_node(
                transformer_ops::output_projection()?,
                vec![layer_out, output_norm, hc_head, output_head],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 2])],
            )?
            .1[0];
        let selected = graph
            .add_node(
                transformer_ops::logits_select()?,
                vec![logits],
                AttributeMap::new(),
                vec![ValueMeta::tensor(DataType::F32, [1, 2])],
            )?
            .1[0];
        graph.set_outputs(vec![selected])?;

        let program = GraphProgram::new(
            graph,
            ExternalBindingPlan::new(),
            tiny_semantic_runtime_plan(),
            GraphProgramProfile::default(),
        )?;
        let batch = ExecutionBatch::decode_one(SessionId(7), 0, 0, None)?;
        Ok((program, objects, batch, dir))
    }

    fn assert_tiny_embedding_head_outputs(outputs: &[TensorData]) -> Result<()> {
        assert_eq!(outputs.len(), 1);
        assert_eq!(bytes_to_f32(outputs[0].bytes())?, vec![3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn causal_attention_persists_kv_across_decode_calls() -> Result<()> {
        let layer = tiny_layer_plan();
        let mut kv_states = BTreeMap::new();
        let first = ExecutionBatch::decode_one(SessionId(9), 0, 1, None)?;
        let first_out = causal_attention(
            &TensorF32::new(vec![1, 1], vec![0.0], "q0")?,
            &TensorF32::new(vec![1, 1], vec![1.0], "k0")?,
            &TensorF32::new(vec![1, 1], vec![10.0], "v0")?,
            &first,
            &layer,
            &mut kv_states,
        )?;
        assert_eq!(first_out.data, vec![10.0]);
        assert_eq!(kv_states.len(), 1);
        assert_eq!(kv_states.values().next().unwrap().len(), 1);

        let second = ExecutionBatch::decode_one(SessionId(9), 1, 2, None)?;
        let second_out = causal_attention(
            &TensorF32::new(vec![1, 1], vec![0.0], "q1")?,
            &TensorF32::new(vec![1, 1], vec![1.0], "k1")?,
            &TensorF32::new(vec![1, 1], vec![20.0], "v1")?,
            &second,
            &layer,
            &mut kv_states,
        )?;
        assert!((second_out.data[0] - 15.0).abs() < 1e-6);
        assert_eq!(kv_states.values().next().unwrap().len(), 2);
        Ok(())
    }

    fn tiny_semantic_runtime_plan() -> TransformerRuntimePlan {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("tiny-semantic".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(32),
            num_layers: Some(1),
            vocab_size: Some(2),
            num_heads: Some(1),
            num_kv_heads: None,
            head_dim: Some(32),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(1),
                num_experts_per_tok: Some(1),
                has_shared_experts: true,
                router: RouterKind::HashAssistedTopK,
            },
            semantics: Default::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        TransformerRuntimePlan {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            hidden_size: spec.hidden_size,
            vocab_size: spec.vocab_size,
            num_heads: spec.num_heads,
            num_kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            prologue: RuntimePrologue {
                token_embedding: TensorRole::TokenEmbedding,
            },
            layers: vec![TransformerLayerPlan {
                index: 0,
                pre_norm_roles: Vec::new(),
                attention: AttentionStepPlan {
                    kind: AttentionKind::MultiLatentAttention,
                    kv_shape: KvCacheShape::LatentOrCompressed,
                    num_heads: Some(1),
                    num_kv_heads: None,
                    head_dim: Some(32),
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
                feed_forward: FeedForwardStepPlan {
                    kind: FeedForwardKind::RoutedAndSharedExperts,
                    router: RouterKind::HashAssistedTopK,
                    num_experts: Some(1),
                    num_experts_per_tok: Some(1),
                    required_roles: Vec::new(),
                    optional_roles: Vec::new(),
                    swiglu_limit: None,
                    route_scale: None,
                    expert_residency: ExpertResidencyMode::Streamable,
                    has_shared_experts: true,
                },
                auxiliary_roles: Vec::new(),
                norm_epsilon: 1e-6,
                hyper_connection_epsilon: 1e-6,
                hyper_connection_sinkhorn_iters: 4,
            }],
            epilogue: RuntimeEpilogue {
                output_norm: Some(TensorRole::OutputNorm),
                output_head: Some(TensorRole::OutputHead),
            },
            policies: PolicySet::from_spec(&spec),
            attachments: Vec::new(),
        }
    }

    fn tiny_runtime_plan() -> TransformerRuntimePlan {
        let spec = TransformerSpec {
            family: ModelFamily::Qwen3,
            architecture: Some("tiny".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(2),
            num_layers: Some(0),
            vocab_size: Some(2),
            num_heads: Some(1),
            num_kv_heads: Some(1),
            head_dim: Some(2),
            attention: AttentionKind::DenseMha,
            moe: MoeSpec::none(),
            semantics: Default::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        TransformerRuntimePlan {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            hidden_size: spec.hidden_size,
            vocab_size: spec.vocab_size,
            num_heads: spec.num_heads,
            num_kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            prologue: RuntimePrologue {
                token_embedding: TensorRole::TokenEmbedding,
            },
            layers: Vec::new(),
            epilogue: RuntimeEpilogue {
                output_norm: None,
                output_head: Some(TensorRole::OutputHead),
            },
            policies: PolicySet::from_spec(&spec),
            attachments: Vec::new(),
        }
    }

    fn tiny_layer_plan() -> TransformerLayerPlan {
        TransformerLayerPlan {
            index: 0,
            pre_norm_roles: Vec::new(),
            attention: AttentionStepPlan {
                kind: AttentionKind::DenseMha,
                kv_shape: KvCacheShape::FullKeysValues,
                num_heads: Some(1),
                num_kv_heads: Some(1),
                head_dim: Some(1),
                rope_theta: None,
                rope_head_dim: None,
                rope_factor: None,
                rope_original_max_position_embeddings: None,
                rope_beta_fast: None,
                rope_beta_slow: None,
                compress_rope_theta: None,
                window_size: None,
                index_topk: None,
                index_num_heads: None,
                index_head_dim: None,
                compress_ratio: None,
                required_roles: Vec::new(),
                optional_roles: Vec::new(),
                needs_sparse_indices: false,
                needs_attention_sink: false,
            },
            feed_forward: FeedForwardStepPlan {
                kind: FeedForwardKind::DenseMlp,
                router: RouterKind::None,
                num_experts: None,
                num_experts_per_tok: None,
                required_roles: Vec::new(),
                optional_roles: Vec::new(),
                swiglu_limit: None,
                route_scale: None,
                expert_residency: ExpertResidencyMode::AllResident,
                has_shared_experts: false,
            },
            auxiliary_roles: Vec::new(),
            norm_epsilon: 1e-6,
            hyper_connection_epsilon: 1e-6,
            hyper_connection_sinkhorn_iters: 4,
        }
    }

    fn attrs_u64(key: &str, value: u64) -> AttributeMap {
        let mut attrs = AttributeMap::new();
        attrs.insert(key.into(), AttributeValue::UInt(value));
        attrs
    }

    fn hc_group(writer: &mut ArtifactWriter, kind: ArtifactGroupKind) -> ArtifactObjectGroup {
        ArtifactObjectGroup {
            kind,
            layer: Some(0),
            tensors: vec![
                writer.f32_slice(
                    "hc.function",
                    TensorRole::AuxHiddenCompressor,
                    &[0.0; 512],
                    vec![8, 64],
                ),
                writer.f32_slice(
                    "hc.scale",
                    TensorRole::AuxHiddenCompressor,
                    &[1.0, 1.0, 1.0],
                    vec![3],
                ),
                writer.f32_slice(
                    "hc.base",
                    TensorRole::AuxHiddenCompressor,
                    &[0.0; 8],
                    vec![8],
                ),
            ],
        }
    }

    fn hc_head_group(writer: &mut ArtifactWriter) -> ArtifactObjectGroup {
        ArtifactObjectGroup {
            kind: ArtifactGroupKind::HyperConnectionHead,
            layer: None,
            tensors: vec![
                writer.f32_slice(
                    "hc_head.function",
                    TensorRole::AuxOutputHiddenCompressor,
                    &[0.0; 128],
                    vec![2, 64],
                ),
                writer.f32_slice(
                    "hc_head.scale",
                    TensorRole::AuxOutputHiddenCompressor,
                    &[1.0],
                    vec![1],
                ),
                writer.f32_slice(
                    "hc_head.base",
                    TensorRole::AuxOutputHiddenCompressor,
                    &[0.0, 1.0],
                    vec![2],
                ),
            ],
        }
    }

    fn write_tiny_expert(path: &std::path::Path, expert: ExpertId) -> Vec<ExpertTensorSlice> {
        let payloads = vec![
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Gate, 0x42),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Gate),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Up, 0x43),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Up),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Down, 0x22),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Down),
        ];
        let mut bytes = Vec::new();
        for payload in &payloads {
            bytes.extend_from_slice(&payload.bytes);
        }
        std::fs::write(path, bytes).unwrap();
        let mut offset = 0u64;
        payloads
            .into_iter()
            .map(|payload| {
                let bytes = payload.bytes.len() as u64;
                let slice = ExpertTensorSlice {
                    offset,
                    bytes,
                    ..payload.slice
                };
                offset += bytes;
                slice
            })
            .collect()
    }

    fn tiny_fp4_payload(
        expert: ExpertId,
        path: &std::path::Path,
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
        path: &std::path::Path,
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

    fn one_hot_rows(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut values = vec![0.0f32; rows * cols];
        for &(row, col, value) in entries {
            values[row * cols + col] = value;
        }
        values
    }

    struct ArtifactWriter {
        path: std::path::PathBuf,
        bytes: Vec<u8>,
    }

    impl ArtifactWriter {
        fn new(path: std::path::PathBuf) -> Self {
            Self {
                path,
                bytes: Vec::new(),
            }
        }

        fn f32_slice(
            &mut self,
            name: &str,
            role: TensorRole,
            values: &[f32],
            shape: Vec<usize>,
        ) -> ArtifactTensorSlice {
            let offset = self.bytes.len() as u64;
            self.bytes.extend(f32_bytes(values));
            ArtifactTensorSlice {
                name: name.into(),
                role,
                path: self.path.clone(),
                offset,
                bytes: (values.len() * 4) as u64,
                dtype: ArtifactDType::F32,
                shape,
            }
        }

        fn i64_slice(
            &mut self,
            name: &str,
            role: TensorRole,
            values: &[i64],
            shape: Vec<usize>,
        ) -> ArtifactTensorSlice {
            let offset = self.bytes.len() as u64;
            self.bytes
                .extend(values.iter().flat_map(|value| value.to_le_bytes()));
            ArtifactTensorSlice {
                name: name.into(),
                role,
                path: self.path.clone(),
                offset,
                bytes: (values.len() * 8) as u64,
                dtype: ArtifactDType::I64,
                shape,
            }
        }

        fn finish(self) {
            std::fs::write(self.path, self.bytes).unwrap();
        }
    }

    fn slice(
        path: &std::path::Path,
        name: &str,
        role: TensorRole,
        offset: u64,
        bytes: u64,
        shape: Vec<usize>,
    ) -> ArtifactTensorSlice {
        ArtifactTensorSlice {
            name: name.into(),
            role,
            path: path.to_path_buf(),
            offset,
            bytes,
            dtype: ArtifactDType::F32,
            shape,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }
}
