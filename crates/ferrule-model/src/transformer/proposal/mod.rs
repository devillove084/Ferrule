//! Model-independent transformer proposal components.

mod mtp;

pub use mtp::{
    MtpAttachment, MtpConfig, MtpHeadOutput, MtpHeads, MtpPhysicalOps, MtpProposalContinuation,
    MtpProposalExecutor, MtpProtocol, MtpStage, MtpStageResume, MtpStageStart,
    PreparedMtpAttachment, PreparedMtpHeads, PreparedMtpStage,
};
