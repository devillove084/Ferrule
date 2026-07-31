//! Checkpoint indexing, bounded tensor reads, payload decoding, and generic weight formats.
//!
//! This module knows storage formats and tensor byte ranges, but not model-family
//! names or runtime residency. Model adapters interpret checkpoint inventories and
//! produce prepared resource manifests for the materialization runtime.

pub mod encoding;
pub(crate) mod hash;
pub mod index;
pub mod inventory;
mod read_plan;
mod source;
pub mod tensor;
pub mod weight;

pub use encoding::{
    decode_e8m0_scale, decode_fp4_e2m1_nibble, decode_fp4_e2m1_packed_low_first,
    decode_fp8_e4m3fn_byte, dequantize_fp4_e2m1_with_e8m0_scales,
    dequantize_fp8_e4m3fn_with_e8m0_scales, normalized_hadamard_transform_rows_in_place,
    simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};
pub use index::HfSafetensorsIndex;
pub use inventory::{
    DtypeCount, HfAttentionTensorInfo, HfDenseLayerTensorInfo, HfHyperConnectionTensorInfo,
    HfRoutedExpertTensorInfo, HfRouterTensorInfo, HfSafetensorsInventory,
    HfSafetensorsShardSummary, HfSafetensorsTensorInfo, HfSharedExpertTensorInfo, TensorRoleCount,
};
pub use read_plan::{CheckpointPositionedReader, CheckpointReadExtent, CheckpointReadPlan};
pub use source::{CheckpointBundleSource, CheckpointSourceCatalog, CheckpointSourceFileIdentity};
pub(crate) use source::{CheckpointSourceTensor, checkpoint_resource_source};
pub use tensor::{
    CheckpointDType, CheckpointMatrixSlice, CheckpointTensorPayload, CheckpointTensorReader,
    CheckpointTensorSlice,
};
pub use weight::{ActivationQuantization, LinearExecutionPolicy, LinearWeight, LinearWeightFormat};
