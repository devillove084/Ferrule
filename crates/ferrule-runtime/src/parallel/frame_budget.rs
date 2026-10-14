use super::process::ProcessFrameLimits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSizeOverflow {
    /// The token or bucket payload bound overflowed.
    Payload,
    /// The payload fits usize, but charging the process envelope overflowed.
    Envelope,
}

/// Conservative integer-JSON sizing for a numeric routed bucket in either
/// direction. This preflight only does checked arithmetic; it never constructs
/// tokens, serializes values, or allocates a representative frame.
///
/// Public access on Unix is through
/// `crate::parallel::expert_process::NumericFrameSize`; the sizing implementation
/// and the IPC codec modules remain private.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericFrameSize {
    payload_bytes: usize,
    frame_bytes: usize,
}

impl NumericFrameSize {
    /// Calculate a checked encoded-size bound without validating model geometry.
    ///
    /// The error is `FrameSizeOverflow`, not `ferrule_common::Error`. When
    /// returning `ferrule_common::Result`, map it explicitly before using `?`.
    pub fn new(hidden: usize, max_tokens: usize) -> Result<Self, FrameSizeOverflow> {
        // Each u32 bit pattern uses at most ten digits plus a separator. Keep
        // token metadata, bucket headers and the process envelope charged too.
        let payload_bytes = hidden
            .checked_mul(11)
            .and_then(|n| n.checked_add(384))
            .and_then(|n| n.checked_mul(max_tokens))
            .and_then(|n| n.checked_add(256))
            .ok_or(FrameSizeOverflow::Payload)?;
        let frame_bytes = payload_bytes
            .checked_add(1024)
            .ok_or(FrameSizeOverflow::Envelope)?;
        Ok(Self {
            payload_bytes,
            frame_bytes,
        })
    }

    pub fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    pub fn frame_bytes(self) -> usize {
        self.frame_bytes
    }

    /// Check sizing only; callers must also validate the negotiated limits.
    pub fn fits(self, limits: ProcessFrameLimits) -> bool {
        self.payload_bytes() <= limits.max_command_bytes
            && self.payload_bytes() <= limits.max_output_bytes
            && self.frame_bytes() <= limits.max_frame_bytes
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parallel/frame_budget.rs"]
mod tests;
