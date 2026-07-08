//! Graph-layer object aggregation for coarse semantic Transformer lowering.
//!
//! The semantic graph initially represents a whole Transformer block with a
//! coarse `transformer_layer` node. This module is the generic bridge from a
//! materialized `BackendObjectStore` to the layer-scoped object bundle that
//! existing attention, hyper-connection, router, MoE, expert, and KV components
//! can consume. It deliberately groups by semantic artifact kinds and layer
//! metadata; it does not match raw checkpoint tensor names or model-family names.

use crate::graph::ExternalKey;
use ferrule_common::{Error, Result};

use crate::backend_object_store::{
    ArtifactObjectGroup, BackendObject, BackendObjectStore, ExpertRegistryObject,
};
use crate::graph::runtime::ArtifactGroupKind;

/// A backend object reference together with the graph external key that produced it.
#[derive(Debug, Clone, Copy)]
pub struct GraphObjectRef<'a> {
    pub key: &'a ExternalKey,
    pub object: &'a BackendObject,
}

/// Layer-scoped graph externals required to lower a coarse `transformer_layer`.
#[derive(Debug, Clone, Copy)]
pub struct GraphLayerObjects<'a> {
    pub layer: usize,
    pub attention: &'a ArtifactObjectGroup,
    pub layer_norms: Option<&'a ArtifactObjectGroup>,
    pub hc_attention: Option<&'a ArtifactObjectGroup>,
    pub hc_feed_forward: Option<&'a ArtifactObjectGroup>,
    pub router: Option<&'a ArtifactObjectGroup>,
    pub shared_expert: Option<&'a ArtifactObjectGroup>,
    pub expert_registry: Option<&'a ExpertRegistryObject>,
    pub kv_state: Option<GraphObjectRef<'a>>,
}

impl GraphLayerObjects<'_> {
    pub fn uses_hyper_connection(&self) -> bool {
        self.hc_attention.is_some() || self.hc_feed_forward.is_some()
    }

    pub fn uses_routed_experts(&self) -> bool {
        self.router.is_some() || self.expert_registry.is_some()
    }

    pub fn uses_shared_expert(&self) -> bool {
        self.shared_expert.is_some()
    }
}

impl BackendObjectStore {
    /// Aggregate semantic layer objects for the given layer.
    ///
    /// This is intentionally a coarse, object-reference bundle. Typed payload
    /// decoding is the next adapter layer: `GraphLayerObjects` plus artifact
    /// readers/policies can be converted into the existing layer binding payloads
    /// without adding model-family-specific graph APIs.
    pub fn layer_objects(&self, layer: usize) -> Result<GraphLayerObjects<'_>> {
        Ok(GraphLayerObjects {
            layer,
            attention: self.required_artifact_group(ArtifactGroupKind::Attention, Some(layer))?,
            layer_norms: self.artifact_group(ArtifactGroupKind::LayerNorm, Some(layer))?,
            hc_attention: self
                .artifact_group(ArtifactGroupKind::HyperConnectionAttention, Some(layer))?,
            hc_feed_forward: self
                .artifact_group(ArtifactGroupKind::HyperConnectionFeedForward, Some(layer))?,
            router: self.artifact_group(ArtifactGroupKind::Router, Some(layer))?,
            shared_expert: self.artifact_group(ArtifactGroupKind::SharedExpert, Some(layer))?,
            expert_registry: self.expert_registry(layer)?,
            kv_state: self.kv_state(layer)?,
        })
    }

    pub fn artifact_group(
        &self,
        kind: ArtifactGroupKind,
        layer: Option<usize>,
    ) -> Result<Option<&ArtifactObjectGroup>> {
        let mut found = None;
        for object in self.objects().values() {
            let BackendObject::ArtifactGroup(group) = object else {
                continue;
            };
            if group.kind == kind && group.layer == layer {
                if found.is_some() {
                    return Err(Error::Graph(format!(
                        "duplicate artifact group {} layer={layer:?} in backend object store",
                        kind.as_str()
                    )));
                }
                found = Some(group);
            }
        }
        Ok(found)
    }

    pub fn required_artifact_group(
        &self,
        kind: ArtifactGroupKind,
        layer: Option<usize>,
    ) -> Result<&ArtifactObjectGroup> {
        self.artifact_group(kind, layer)?.ok_or_else(|| {
            Error::Graph(format!(
                "missing artifact group {} layer={layer:?} in backend object store",
                kind.as_str()
            ))
        })
    }

    pub fn expert_registry(&self, layer: usize) -> Result<Option<&ExpertRegistryObject>> {
        let mut found = None;
        for object in self.objects().values() {
            let BackendObject::ExpertRegistry(registry) = object else {
                continue;
            };
            if registry.layer == layer {
                if found.is_some() {
                    return Err(Error::Graph(format!(
                        "duplicate expert registry for layer {layer} in backend object store"
                    )));
                }
                found = Some(registry);
            }
        }
        Ok(found)
    }

    pub fn kv_state(&self, layer: usize) -> Result<Option<GraphObjectRef<'_>>> {
        let expected_name = format!("layers.{layer}.kv_state");
        let mut found = None;
        for (key, object) in self.objects() {
            if matches!(object, BackendObject::KvState(_))
                && key.namespace() == "state"
                && key.name() == expected_name
            {
                if found.is_some() {
                    return Err(Error::Graph(format!(
                        "duplicate KV state for layer {layer} in backend object store"
                    )));
                }
                found = Some(GraphObjectRef { key, object });
            }
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use crate::graph::ExternalKey;
    use ferrule_model::TensorRole;

    use super::*;
    use ferrule_model::artifact::tensor::{ArtifactDType, ArtifactTensorSlice};
    use ferrule_model::moe::streaming::{ExpertId, ExpertLoadSource};

    #[test]
    fn layer_objects_aggregates_semantic_groups_by_kind_and_layer() {
        let mut store = BackendObjectStore::new();
        insert_group(&mut store, ArtifactGroupKind::Attention, 2, "attention");
        insert_group(&mut store, ArtifactGroupKind::LayerNorm, 2, "layer_norm");
        insert_group(
            &mut store,
            ArtifactGroupKind::HyperConnectionAttention,
            2,
            "hc_attention",
        );
        insert_group(
            &mut store,
            ArtifactGroupKind::HyperConnectionFeedForward,
            2,
            "hc_feed_forward",
        );
        insert_group(&mut store, ArtifactGroupKind::Router, 2, "router");
        insert_group(
            &mut store,
            ArtifactGroupKind::SharedExpert,
            2,
            "shared_expert",
        );
        store
            .insert_required(
                external_key("experts", "layers.2.routed_expert_registry"),
                BackendObject::ExpertRegistry(ExpertRegistryObject {
                    layer: 2,
                    experts: BTreeMap::from([(ExpertId::new(2, 0), ExpertLoadSource::CpuResident)]),
                }),
            )
            .unwrap();
        store
            .insert_required(
                external_key("state", "layers.2.kv_state"),
                BackendObject::KvState(None),
            )
            .unwrap();

        let objects = store.layer_objects(2).unwrap();
        assert_eq!(objects.layer, 2);
        assert_eq!(objects.attention.kind, ArtifactGroupKind::Attention);
        assert_eq!(
            objects.layer_norms.unwrap().kind,
            ArtifactGroupKind::LayerNorm
        );
        assert!(objects.uses_hyper_connection());
        assert!(objects.uses_routed_experts());
        assert!(objects.uses_shared_expert());
        assert_eq!(objects.expert_registry.unwrap().experts.len(), 1);
        let kv_state = objects.kv_state.unwrap();
        assert_eq!(kv_state.key.name(), "layers.2.kv_state");
        assert!(matches!(kv_state.object, BackendObject::KvState(None)));
    }

    #[test]
    fn layer_objects_reports_missing_required_attention_group() {
        let store = BackendObjectStore::new();
        let err = store.layer_objects(0).unwrap_err();
        assert!(format!("{err}").contains("missing artifact group attention layer=Some(0)"));
    }

    fn insert_group(
        store: &mut BackendObjectStore,
        kind: ArtifactGroupKind,
        layer: usize,
        suffix: &str,
    ) {
        store
            .insert_required(
                external_key("artifacts", &format!("layers.{layer}.{suffix}")),
                BackendObject::ArtifactGroup(ArtifactObjectGroup {
                    kind,
                    layer: Some(layer),
                    tensors: vec![artifact_slice(&format!("{suffix}.weight"))],
                }),
            )
            .unwrap();
    }

    fn artifact_slice(name: &str) -> ArtifactTensorSlice {
        ArtifactTensorSlice {
            name: name.to_string(),
            role: TensorRole::Unknown,
            path: PathBuf::from("artifact.safetensors"),
            offset: 0,
            bytes: 4,
            dtype: ArtifactDType::F32,
            shape: vec![1],
        }
    }

    fn external_key(namespace: &str, name: &str) -> ExternalKey {
        ExternalKey::new(namespace, name.to_string()).unwrap()
    }
}
