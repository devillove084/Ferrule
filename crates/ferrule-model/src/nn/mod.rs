//! Inference-only module and parameter metadata.
//!
//! This module deliberately models immutable parameter schemas rather than
//! trainable tensors. It has no autograd, optimizer, or training-state API.

mod module;
mod parameter;
mod path;

pub use module::{Module, ModuleNode, ModuleTreeError, ModuleVisitError, ModuleVisitor};
pub use parameter::{
    DTypeConstraint, ParameterDType, ParameterId, ParameterPart, ParameterResidency,
    ParameterScaleSpec, ParameterSpec, ParameterSpecError, ParameterTensorSpec, StorageEncoding,
};
pub use path::{ModulePath, ModulePathError};
