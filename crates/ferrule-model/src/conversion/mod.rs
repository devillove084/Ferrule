//! Conversion planning and quantization recipes.
//!
//! Conversion plans are intentionally separate from model support contracts:
//! contracts describe what a model is; conversion plans describe how an input
//! artifact becomes a Ferrule or compatibility artifact.

pub mod plan;
pub mod recipe;

pub use plan::ConversionPlan;
pub use recipe::{
    ArtifactTarget, CalibrationSet, QuantizationFormat, QuantizationRecipe, TensorRoleQuantPolicy,
};

#[cfg(test)]
mod tests;
