//! Boot-selected private endpoint shared by CLI and child integration fixtures.

use super::decoder::DecoderChild;
use super::decoder_expert::ExpertChild;
use super::{
    ProcessBoot, ProcessChildHandler, ProcessFailureKind, ProcessHandlerError, ProcessLaunch,
    ProcessRequest,
};
use serde_json::Value;

pub enum DecoderEndpoint {
    Stage(DecoderChild),
    Expert(ExpertChild),
}
impl DecoderEndpoint {
    /// Linear benchmark Boot has `spec`; decoder Boot has `version`, or the
    /// explicit `expert_boot` wrapper. Malformed decoder Boots are not retried
    /// as another workload and no CUDA probing occurs before this selection.
    pub fn accepts(config: &Value) -> bool {
        config.get("version").is_some() || config.get("expert_boot").is_some()
    }
    pub fn initialize(
        boot: ProcessBoot,
        child_launch: ProcessLaunch,
    ) -> Result<Self, ProcessHandlerError> {
        if boot.config.get("expert_boot").is_some() {
            ExpertChild::initialize(boot).map(Self::Expert)
        } else {
            DecoderChild::initialize_builtin(boot, child_launch).map(Self::Stage)
        }
    }
}
impl ProcessChildHandler for DecoderEndpoint {
    type Command = Value;
    type Output = Value;
    fn execute(&mut self, request: ProcessRequest<Value>) -> Result<Value, ProcessHandlerError> {
        match self {
            Self::Stage(child) => dispatch(child, request),
            Self::Expert(child) => dispatch(child, request),
        }
    }
    fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
        match self {
            Self::Stage(child) => child.shutdown(),
            Self::Expert(child) => child.shutdown(),
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
