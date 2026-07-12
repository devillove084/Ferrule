//! Runtime-side bindings for opaque graph externals.
//!
//! Ferrule's sole execution ABI lives in [`ferrule_common::execution`]. This
//! module only describes how graph [`crate::graph::ExternalKey`] values bind to
//! runtime weights, state, artifacts, and other backend-managed objects.
//!
//! ## `ExternalBindingPlan`
//!
//! `crate::graph` stores only opaque `ExternalKey`s. It should not own CUDA
//! buffers, mmap-backed tensors, WeightPack objects, KV pages, or resident expert
//! handles. `ExternalBindingPlan` is the runtime-side bridge from semantic model
//! roles and runtime state to those graph externals.
//!
//! A binding records:
//!
//! - the graph `ExternalKey`
//! - the kind of runtime object, such as weight, KV state, artifact tensor, or
//!   resident expert
//! - optional `TensorRole` and layer metadata
//! - shape/type metadata used by graph validation
//! - preferred residency, such as host, device, streamable, paged, or
//!   backend-managed
//!
//! Advantages:
//!
//! - Keeps raw Hugging Face tensor names and backend storage details out of graph
//!   translators.
//! - Lets CUDA/CPU/artifact backends decide upload, residency, eviction, and
//!   streamable expert policy behind one semantic binding layer.
//! - Makes weights, KV cache, adapters, speculation state, and resident expert
//!   handles visible to the runtime without embedding them in the graph IR.
//! - Provides a natural hook for future graph-program compilation, memory
//!   planning, CUDA graph capture, and autograd state management.
//!

use crate::graph::{ExternalKey, ValueMeta};
use ferrule_common::{Error, Result};
use ferrule_model::TensorRole;

/// Semantic artifact group represented by a graph external.
pub use ferrule_model::ArtifactGroupKind;

/// Runtime/artifact object class represented by a graph external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalBindingKind {
    Weight,
    KvState,
    ArtifactTensor,
    ArtifactGroup(ArtifactGroupKind),
    ExpertRegistry,
    ResidentExpert,
    Adapter,
    Speculation,
    Other(String),
}

/// Preferred residency for an external object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalResidency {
    Host,
    Device,
    Streamable,
    Paged,
    BackendManaged,
}

/// One semantic binding for a graph external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalBinding {
    pub key: ExternalKey,
    pub kind: ExternalBindingKind,
    pub role: Option<TensorRole>,
    pub layer: Option<usize>,
    pub meta: ValueMeta,
    pub residency: ExternalResidency,
}

impl ExternalBinding {
    pub fn weight(
        key: ExternalKey,
        role: TensorRole,
        layer: Option<usize>,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind: ExternalBindingKind::Weight,
            role: Some(role),
            layer,
            meta,
            residency,
        }
    }

    pub fn state(
        key: ExternalKey,
        kind: ExternalBindingKind,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind,
            role: None,
            layer: None,
            meta,
            residency,
        }
    }

    pub fn artifact_group(
        key: ExternalKey,
        group: ArtifactGroupKind,
        layer: Option<usize>,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind: ExternalBindingKind::ArtifactGroup(group),
            role: None,
            layer,
            meta,
            residency,
        }
    }

    pub fn expert_registry(
        key: ExternalKey,
        layer: usize,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind: ExternalBindingKind::ExpertRegistry,
            role: None,
            layer: Some(layer),
            meta,
            residency,
        }
    }
}

/// Graph-facing binding plan produced by model/runtime translators.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalBindingPlan {
    entries: Vec<ExternalBinding>,
}

impl ExternalBindingPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[ExternalBinding] {
        &self.entries
    }

    pub fn push(&mut self, binding: ExternalBinding) -> Result<()> {
        if self.entries.iter().any(|entry| entry.key == binding.key) {
            return Err(Error::Graph(format!(
                "duplicate external binding '{}:{}'",
                binding.key.namespace(),
                binding.key.name()
            )));
        }
        self.entries.push(binding);
        Ok(())
    }

    pub fn get(&self, key: &ExternalKey) -> Option<&ExternalBinding> {
        self.entries.iter().find(|entry| &entry.key == key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{DataType, ValueMeta};

    use super::*;

    #[test]
    fn binding_plan_rejects_duplicate_keys() {
        let key = ExternalKey::new("weights", "layer0.q").unwrap();
        let binding = ExternalBinding::weight(
            key.clone(),
            TensorRole::AttentionQuery,
            Some(0),
            ValueMeta::tensor(DataType::Bf16, [16, 16]),
            ExternalResidency::Device,
        );

        let mut plan = ExternalBindingPlan::new();
        plan.push(binding.clone()).unwrap();
        let err = plan.push(binding).unwrap_err();
        assert!(format!("{err}").contains("duplicate external binding"));
    }

    #[test]
    fn binding_plan_finds_entries_by_key() {
        let key = ExternalKey::new("state", "kv").unwrap();
        let binding = ExternalBinding::state(
            key.clone(),
            ExternalBindingKind::KvState,
            ValueMeta::external_state("kv"),
            ExternalResidency::BackendManaged,
        );

        let mut plan = ExternalBindingPlan::new();
        plan.push(binding).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.get(&key).unwrap().kind, ExternalBindingKind::KvState);
    }
}
