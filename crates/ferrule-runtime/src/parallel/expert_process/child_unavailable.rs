//! CPU-only facade: a checked Boot never invokes the binder or opens weights.
use super::*;

/// Uninhabited without CUDA: initialization always rejects after Boot checks.
pub struct NumericExpertChild {
    unavailable: std::convert::Infallible,
}

impl NumericExpertChild {
    pub(super) fn initialize_checked(
        _validated: Arc<ValidatedNumericBoot>,
        _bind: impl FnOnce(&NumericBoot) -> Result<BoundDecoderResources>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        Err(startup("numeric expert requires CUDA; no CPU fallback"))
    }
}

impl ProcessChildHandler for NumericExpertChild {
    type Command = NumericCommand;
    type Output = NumericReply;

    fn execute(
        &mut self,
        _request: ProcessRequest<NumericCommand>,
    ) -> std::result::Result<NumericReply, ProcessHandlerError> {
        match self.unavailable {}
    }

    fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
        match self.unavailable {}
    }
}
