//! Boot-selected private endpoint shared by CLI and child integration fixtures.

use super::decoder::{DecoderBoot, DecoderChild};
use super::decoder_expert::ExpertChild;
use super::{
    ProcessBoot, ProcessChildHandler, ProcessFailureKind, ProcessHandlerError, ProcessLaunch,
    ProcessRequest,
};
use crate::parallel::expert_process::NumericBoot;
use ferrule_model::transformer::BoundDecoderResources;
use serde_json::Value;

/// Explicit application composition, not a global registry or a wire field.
/// Functions run on the child only after the corresponding boot is validated.
/// Parallel owns the rebound geometry/source checks and all physical execution.
#[derive(Clone, Copy)]
pub struct DecoderBindings {
    pub decoder: fn(&DecoderBoot) -> ferrule_common::Result<BoundDecoderResources>,
    pub numeric: fn(&NumericBoot) -> ferrule_common::Result<BoundDecoderResources>,
}

// One fixed-size owner allocation per child startup, covered by process metadata
// overhead; never allocated per command. Retaining the endpoint in child_serve's
// ManuallyDrop guard also retains this box and all of its physical custody.
pub enum DecoderEndpoint {
    Stage(Box<DecoderChild>),
    Expert(Box<ExpertChild>),
    Numeric(Box<crate::parallel::expert_process::NumericExpertChild>),
}
impl DecoderEndpoint {
    /// Linear benchmark Boot has `spec`; decoder Boot has `version`, or the
    /// explicit `expert_boot` wrapper. Malformed decoder Boots are not retried
    /// as another workload and no CUDA probing occurs before this selection.
    pub fn accepts(config: &Value) -> bool {
        config.get("version").is_some()
            || config.get("expert_boot").is_some()
            || config.get("numeric_expert_boot").is_some()
    }
    pub fn initialize(
        boot: ProcessBoot,
        child_launch: ProcessLaunch,
        bindings: DecoderBindings,
    ) -> Result<Self, ProcessHandlerError> {
        if boot.config.get("numeric_expert_boot").is_some() {
            crate::parallel::expert_process::NumericExpertChild::initialize(boot, bindings.numeric)
                .map(Box::new)
                .map(Self::Numeric)
        } else if boot.config.get("expert_boot").is_some() {
            ExpertChild::initialize(boot, bindings.decoder)
                .map(Box::new)
                .map(Self::Expert)
        } else {
            DecoderChild::initialize_builtin(boot, child_launch, bindings.decoder)
                .map(Box::new)
                .map(Self::Stage)
        }
    }
}
impl ProcessChildHandler for DecoderEndpoint {
    type Command = Value;
    type Output = Value;
    fn execute(&mut self, request: ProcessRequest<Value>) -> Result<Value, ProcessHandlerError> {
        match self {
            Self::Stage(child) => dispatch(child.as_mut(), request),
            Self::Expert(child) => dispatch(child.as_mut(), request),
            Self::Numeric(child) => dispatch(child.as_mut(), request),
        }
    }
    fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
        match self {
            Self::Stage(child) => child.shutdown(),
            Self::Expert(child) => child.shutdown(),
            Self::Numeric(child) => child.shutdown(),
        }
    }
}
fn dispatch<H: ProcessChildHandler>(
    child: &mut H,
    request: ProcessRequest<Value>,
) -> Result<Value, ProcessHandlerError> {
    let command = serde_json::from_value(request.command).map_err(|e| {
        ProcessHandlerError::fenced(ProcessFailureKind::CommandDecode, e.to_string())
    })?;
    let reply = child.execute(ProcessRequest {
        identity: request.identity,
        session: request.session,
        command,
    })?;
    serde_json::to_value(reply)
        .map_err(|e| ProcessHandlerError::unknown(ProcessFailureKind::OutputEncode, e.to_string()))
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/decoder_endpoint.rs"]
mod tests;
