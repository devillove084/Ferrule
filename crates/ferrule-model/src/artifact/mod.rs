//! Input artifact descriptions.
//!
//! These types describe where weights come from. They intentionally do not imply
//! that the artifact is directly executable by Ferrule.

pub mod hf;
pub mod identity;
pub mod index;
pub mod input;
pub mod inventory;

pub use hf::{HfFilePurpose, HfRepoFile, HfSafetensorsArtifact};
pub use identity::{ArtifactFormat, ArtifactIdentity};
pub use index::HfSafetensorsIndex;
pub use input::InputArtifact;
pub use inventory::{
    DtypeCount, HfAttentionTensorInfo, HfDenseLayerTensorInfo, HfHyperConnectionTensorInfo,
    HfRoutedExpertTensorInfo, HfRouterTensorInfo, HfSafetensorsInventory,
    HfSafetensorsShardSummary, HfSafetensorsTensorInfo, HfSharedExpertTensorInfo, TensorRoleCount,
};

#[cfg(test)]
mod tests;
