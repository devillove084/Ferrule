//! Storage object identity.
//!
//! `StorageObjectId` is a structured enum — the variant IS the kind. This avoids
//! the problems of a bare `String` newtype (loss of kind at type level, opaque
//! debug output, inability to reject invalid IDs at construction).
//!
//! The `Display` impl produces a canonical content-addressed string for cache
//! keys (host cache, disk cache, remote cache). Two objects with different
//! model revisions, layout versions, or structural coordinates never collide.

use std::fmt;

// ── Auxiliary newtypes ────────────────────────────────────────────────

/// Content-addressed model identity. Derived from config hash + tokenizer hash
/// + quant policy. Two models with different revisions never share objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelRevision(pub u64);

impl fmt::Display for ModelRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rev{:016x}", self.0)
    }
}

/// Hash of a WeightPack manifest. Identifies a specific packaged artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WeightPackId(pub u64);

impl fmt::Display for WeightPackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wp{:016x}", self.0)
    }
}

/// Semantic role tag for an artifact tensor (e.g. "q_proj", "expert.gate").
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorRole(pub String);

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Matrix kind within an expert bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExpertMatrixKind {
    Gate,
    Up,
    Down,
}

impl fmt::Display for ExpertMatrixKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gate => f.write_str("gate"),
            Self::Up => f.write_str("up"),
            Self::Down => f.write_str("down"),
        }
    }
}

/// Component within an expert tensor (weight, scale, or other).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExpertTensorComponent {
    Weight,
    Scale,
    Other(String),
}

impl fmt::Display for ExpertTensorComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight => f.write_str("weight"),
            Self::Scale => f.write_str("scale"),
            Self::Other(s) => f.write_str(s),
        }
    }
}

// ── StorageObjectId ───────────────────────────────────────────────────

/// Logical identity for a loadable/resident runtime object.
///
/// Structured by design: the variant IS the kind, so the catalog can index by
/// variant without a separate descriptor lookup, and debug output shows what
/// the object is without decoding a string.
///
/// # Cache key format
///
/// `Display` produces canonical strings:
/// ```text
/// artifact:{rev}/tensor/{role}/{dtype}/{shape_hash}
/// artifact:{rev}/tensor/{role}/rows/{start}:{end}/{dtype}
/// expert:{rev}/layer{L}/expert{E}/bundle/v{layout_version}
/// expert:{rev}/layer{L}/expert{E}/{matrix_kind}/{component}
/// output_head:{rev}/chunk{C}/{dtype}
/// weightpack:{wp_id}/chunk/{chunk_id}/v{layout_version}
/// kvpage:{session}/page{page}
/// arena:{device}/slot{slot}
/// opaque:{tag}/{key}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StorageObjectId {
    // ── v1: immutable loadable objects ──
    ArtifactTensor {
        model_revision: ModelRevision,
        tensor_role: TensorRole,
        dtype: String,
        shape_hash: u64,
    },
    ArtifactTensorRows {
        model_revision: ModelRevision,
        tensor_role: TensorRole,
        row_start: u64,
        row_end: u64,
        dtype: String,
    },
    ExpertBundle {
        model_revision: ModelRevision,
        layer: u32,
        expert: u32,
        layout_version: u32,
    },
    ExpertMatrix {
        model_revision: ModelRevision,
        layer: u32,
        expert: u32,
        matrix_kind: ExpertMatrixKind,
        component: ExpertTensorComponent,
    },
    OutputHeadChunk {
        model_revision: ModelRevision,
        chunk: u32,
        dtype: String,
    },
    WeightPackChunk {
        weightpack_id: WeightPackId,
        chunk_id: String,
        layout_version: u32,
    },
    // ── v2+: mutable / ephemeral (not in v1 catalog) ──
    KvPage {
        session: u64,
        page: u64,
    },
    DecodeArenaBuffer {
        device_id: u32,
        slot: u32,
    },
    Opaque {
        tag: String,
        key: String,
    },
}

/// Tag string for quick variant dispatch without matching.
pub const ID_VARIANT_EXPERT_BUNDLE: &str = "expert_bundle";

impl StorageObjectId {
    /// Short variant tag for logging / metrics grouping.
    pub fn variant_tag(&self) -> &'static str {
        match self {
            Self::ArtifactTensor { .. } => "artifact_tensor",
            Self::ArtifactTensorRows { .. } => "artifact_tensor_rows",
            Self::ExpertBundle { .. } => ID_VARIANT_EXPERT_BUNDLE,
            Self::ExpertMatrix { .. } => "expert_matrix",
            Self::OutputHeadChunk { .. } => "output_head_chunk",
            Self::WeightPackChunk { .. } => "weightpack_chunk",
            Self::KvPage { .. } => "kv_page",
            Self::DecodeArenaBuffer { .. } => "decode_arena_buffer",
            Self::Opaque { .. } => "opaque",
        }
    }

    /// True if this object is immutable (safe to cache indefinitely).
    pub fn is_immutable(&self) -> bool {
        matches!(
            self,
            Self::ArtifactTensor { .. }
                | Self::ArtifactTensorRows { .. }
                | Self::ExpertBundle { .. }
                | Self::ExpertMatrix { .. }
                | Self::OutputHeadChunk { .. }
                | Self::WeightPackChunk { .. }
        )
    }

    /// Model revision if this object is model-bound, else `None` (for KvPage,
    /// arena, opaque).
    pub fn model_revision(&self) -> Option<ModelRevision> {
        match self {
            Self::ArtifactTensor { model_revision, .. }
            | Self::ArtifactTensorRows { model_revision, .. }
            | Self::ExpertBundle { model_revision, .. }
            | Self::ExpertMatrix { model_revision, .. }
            | Self::OutputHeadChunk { model_revision, .. } => Some(*model_revision),
            Self::WeightPackChunk { .. }
            | Self::KvPage { .. }
            | Self::DecodeArenaBuffer { .. }
            | Self::Opaque { .. } => None,
        }
    }
}

impl fmt::Display for StorageObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArtifactTensor {
                model_revision,
                tensor_role,
                dtype,
                shape_hash,
            } => {
                write!(
                    f,
                    "artifact:{}/tensor/{}/{}/{:016x}",
                    model_revision, tensor_role, dtype, shape_hash
                )
            }
            Self::ArtifactTensorRows {
                model_revision,
                tensor_role,
                row_start,
                row_end,
                dtype,
            } => {
                write!(
                    f,
                    "artifact:{}/tensor/{}/rows/{}:{}/{}",
                    model_revision, tensor_role, row_start, row_end, dtype
                )
            }
            Self::ExpertBundle {
                model_revision,
                layer,
                expert,
                layout_version,
            } => {
                write!(
                    f,
                    "expert:{}/layer{}/expert{}/bundle/v{}",
                    model_revision, layer, expert, layout_version
                )
            }
            Self::ExpertMatrix {
                model_revision,
                layer,
                expert,
                matrix_kind,
                component,
            } => {
                write!(
                    f,
                    "expert:{}/layer{}/expert{}/{}/{}",
                    model_revision, layer, expert, matrix_kind, component
                )
            }
            Self::OutputHeadChunk {
                model_revision,
                chunk,
                dtype,
            } => {
                write!(f, "output_head:{}/chunk{}/{}", model_revision, chunk, dtype)
            }
            Self::WeightPackChunk {
                weightpack_id,
                chunk_id,
                layout_version,
            } => {
                write!(
                    f,
                    "weightpack:{}/chunk/{}/v{}",
                    weightpack_id, chunk_id, layout_version
                )
            }
            Self::KvPage { session, page } => {
                write!(f, "kvpage:{}/page{}", session, page)
            }
            Self::DecodeArenaBuffer { device_id, slot } => {
                write!(f, "arena:device{}/slot{}", device_id, slot)
            }
            Self::Opaque { tag, key } => {
                write!(f, "opaque:{}/{}", tag, key)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rev(n: u64) -> ModelRevision {
        ModelRevision(n)
    }

    #[test]
    fn expert_bundle_display_format() {
        let id = StorageObjectId::ExpertBundle {
            model_revision: rev(0xABCD),
            layer: 17,
            expert: 42,
            layout_version: 3,
        };
        assert_eq!(
            id.to_string(),
            "expert:rev000000000000abcd/layer17/expert42/bundle/v3"
        );
    }

    #[test]
    fn expert_matrix_display_format() {
        let id = StorageObjectId::ExpertMatrix {
            model_revision: rev(1),
            layer: 5,
            expert: 9,
            matrix_kind: ExpertMatrixKind::Gate,
            component: ExpertTensorComponent::Scale,
        };
        assert_eq!(
            id.to_string(),
            "expert:rev0000000000000001/layer5/expert9/gate/scale"
        );
    }

    #[test]
    fn artifact_tensor_display_format() {
        let id = StorageObjectId::ArtifactTensor {
            model_revision: rev(0xFF),
            tensor_role: TensorRole("q_proj".into()),
            dtype: "BF16".into(),
            shape_hash: 0xDEADBEEF,
        };
        assert_eq!(
            id.to_string(),
            "artifact:rev00000000000000ff/tensor/q_proj/BF16/00000000deadbeef"
        );
    }

    #[test]
    fn weightpack_chunk_display_format() {
        let id = StorageObjectId::WeightPackChunk {
            weightpack_id: WeightPackId(0x1234),
            chunk_id: "layer0_expert3_gate".into(),
            layout_version: 2,
        };
        assert_eq!(
            id.to_string(),
            "weightpack:wp0000000000001234/chunk/layer0_expert3_gate/v2"
        );
    }

    #[test]
    fn artifact_tensor_rows_display_format() {
        let id = StorageObjectId::ArtifactTensorRows {
            model_revision: rev(7),
            tensor_role: TensorRole("lm_head".into()),
            row_start: 128,
            row_end: 255,
            dtype: "F32".into(),
        };
        assert_eq!(
            id.to_string(),
            "artifact:rev0000000000000007/tensor/lm_head/rows/128:255/F32"
        );
    }

    #[test]
    fn kvpage_and_arena_display_format() {
        let kv = StorageObjectId::KvPage {
            session: 3,
            page: 72,
        };
        assert_eq!(kv.to_string(), "kvpage:3/page72");

        let arena = StorageObjectId::DecodeArenaBuffer {
            device_id: 0,
            slot: 4,
        };
        assert_eq!(arena.to_string(), "arena:device0/slot4");
    }

    #[test]
    fn opaque_display_format() {
        let id = StorageObjectId::Opaque {
            tag: "custom".into(),
            key: "thing42".into(),
        };
        assert_eq!(id.to_string(), "opaque:custom/thing42");
    }

    #[test]
    fn output_head_chunk_display_format() {
        let id = StorageObjectId::OutputHeadChunk {
            model_revision: rev(10),
            chunk: 7,
            dtype: "Q8_0".into(),
        };
        assert_eq!(
            id.to_string(),
            "output_head:rev000000000000000a/chunk7/Q8_0"
        );
    }

    #[test]
    fn variant_tag_matches_kind() {
        assert_eq!(
            StorageObjectId::ExpertBundle {
                model_revision: rev(1),
                layer: 0,
                expert: 0,
                layout_version: 1
            }
            .variant_tag(),
            "expert_bundle"
        );
        assert_eq!(
            StorageObjectId::KvPage {
                session: 0,
                page: 0
            }
            .variant_tag(),
            "kv_page"
        );
    }

    #[test]
    fn is_immutable_for_loadable_objects() {
        assert!(StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 1
        }
        .is_immutable());

        assert!(!StorageObjectId::KvPage {
            session: 0,
            page: 0
        }
        .is_immutable());
        assert!(!StorageObjectId::DecodeArenaBuffer {
            device_id: 0,
            slot: 0
        }
        .is_immutable());
    }

    #[test]
    fn model_revision_extraction() {
        let bundle = StorageObjectId::ExpertBundle {
            model_revision: rev(42),
            layer: 1,
            expert: 2,
            layout_version: 1,
        };
        assert_eq!(bundle.model_revision(), Some(rev(42)));

        let kv = StorageObjectId::KvPage {
            session: 0,
            page: 0,
        };
        assert_eq!(kv.model_revision(), None);
    }

    #[test]
    fn two_revisions_never_collide() {
        let a = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 1,
        };
        let b = StorageObjectId::ExpertBundle {
            model_revision: rev(2),
            layer: 0,
            expert: 0,
            layout_version: 1,
        };
        assert_ne!(a, b);
        assert_ne!(a.to_string(), b.to_string());
    }

    #[test]
    fn two_layout_versions_never_collide() {
        let a = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 1,
        };
        let b = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 2,
        };
        assert_ne!(a, b);
        assert_ne!(a.to_string(), b.to_string());
    }

    #[test]
    fn eq_and_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 7,
            expert: 3,
            layout_version: 2,
        };
        let b = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 7,
            expert: 3,
            layout_version: 2,
        };

        assert_eq!(a, b);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }
}
