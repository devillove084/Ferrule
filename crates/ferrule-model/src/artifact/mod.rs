//! Source artifact descriptions.
//!
//! These types describe where weights come from. They intentionally do not imply
//! that the artifact is directly executable by Ferrule.

pub mod hf;
pub mod identity;
pub mod index;
pub mod inventory;
pub mod source;

pub use hf::{HfFilePurpose, HfRepoFile, HfSafetensorsArtifact};
pub use identity::{ArtifactFormat, ArtifactIdentity};
pub use index::HfSafetensorsIndex;
pub use inventory::{
    DtypeCount, HfAttentionTensorInfo, HfHyperConnectionTensorInfo, HfRoutedExpertTensorInfo,
    HfRouterTensorInfo, HfSafetensorsInventory, HfSafetensorsShardSummary, HfSafetensorsTensorInfo,
    HfSharedExpertTensorInfo, TensorRoleCount,
};
pub use source::SourceArtifact;

#[cfg(test)]
mod tests;
