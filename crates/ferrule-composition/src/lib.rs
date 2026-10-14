//! Application composition of built-in model bindings and runtime owners.
//! Runtime receives explicit bindings; it does not depend on this crate.

#[cfg(test)]
extern crate self as ferrule_composition;

pub mod model_factory;

pub use model_factory::{
    BackendSelection, BuiltinModelResolver, ModelFactoryOptions, ModelPrecision,
    NumericStoragePolicy, Qwen35NumericBf16MemoryPlan, ResidentModelBuildObservability,
    ResidentModelBuildPlan, ResidentModelPlanner, ResolvedModelBackend, ServingEngineSelection,
    ServingModelOptions,
};

pub use ferrule_runtime::{Error, Result};

#[cfg(all(test, unix))]
#[path = "../tests/support/build_process_child.rs"]
mod build_process_child;

#[cfg(test)]
#[path = "../tests/support/hybrid_fixture.rs"]
mod hybrid_fixture;

#[cfg(unix)]
pub mod process_binding;

#[cfg(unix)]
pub use process_binding::{BUILTIN_PROCESS_BINDINGS, initialize_builtin_process_endpoint};
