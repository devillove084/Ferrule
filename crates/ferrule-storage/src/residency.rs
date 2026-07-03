//! Residency request — what the policy asks the manager to do.

use std::time::Instant;

use crate::id::StorageObjectId;
use crate::placement::Placement;

/// A request to make an object resident at a placement.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyRequest {
    pub object: StorageObjectId,
    /// Precise placement — not a coarse class. The caller knows which device.
    pub desired: Placement,
    pub priority: ResidencyPriority,
    pub deadline: Option<Instant>,
    pub reason: ResidencyReason,
}

impl ResidencyRequest {
    pub fn new(
        object: StorageObjectId,
        desired: Placement,
        priority: ResidencyPriority,
        reason: ResidencyReason,
    ) -> Self {
        Self {
            object,
            desired,
            priority,
            deadline: None,
            reason,
        }
    }

    /// True if this request blocks the current token.
    pub fn is_blocking(&self) -> bool {
        matches!(self.reason, ResidencyReason::ExecuteNow)
    }
}

/// Urgency of a residency request.
///
/// Ordering: `Background < Low < High < Critical`. `Critical` is the most
/// urgent (selected expert, blocks current token).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidencyPriority {
    /// Prefetch, best-effort.
    Background,
    /// Retain-hot, keep if budget allows.
    Low,
    /// Predicted expert, should be ready soon.
    High,
    /// Selected expert, blocks current token.
    Critical,
}

/// Why a residency request was issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyReason {
    /// Selected by router, must be device-resident now.
    ExecuteNow,
    /// Predicted, stage at host or device.
    Prefetch,
    /// Not needed now but high cumulative frequency — keep if possible.
    RetainHot,
    /// Diagnostic / forced residency.
    Debug,
}

impl ResidencyReason {
    /// `ExecuteNow` implies `Critical` priority.
    pub fn default_priority(self) -> ResidencyPriority {
        match self {
            Self::ExecuteNow => ResidencyPriority::Critical,
            Self::Prefetch => ResidencyPriority::Background,
            Self::RetainHot => ResidencyPriority::Low,
            Self::Debug => ResidencyPriority::Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ModelRevision, StorageObjectId};
    use crate::placement::{DeviceMemoryKind, LocalPlacement, Placement};

    fn rev(n: u64) -> ModelRevision {
        ModelRevision(n)
    }

    fn device_placement() -> Placement {
        Placement::Local(LocalPlacement::Device {
            device_id: 0,
            memory: DeviceMemoryKind::Vram,
        })
    }

    fn sample_object() -> StorageObjectId {
        StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 1,
        }
    }

    #[test]
    fn execute_now_is_blocking() {
        let req = ResidencyRequest::new(
            sample_object(),
            device_placement(),
            ResidencyPriority::Critical,
            ResidencyReason::ExecuteNow,
        );
        assert!(req.is_blocking());
    }

    #[test]
    fn prefetch_is_not_blocking() {
        let req = ResidencyRequest::new(
            sample_object(),
            Placement::Local(LocalPlacement::Host { pinned: false }),
            ResidencyPriority::Background,
            ResidencyReason::Prefetch,
        );
        assert!(!req.is_blocking());
    }

    #[test]
    fn reason_default_priority() {
        assert_eq!(
            ResidencyReason::ExecuteNow.default_priority(),
            ResidencyPriority::Critical
        );
        assert_eq!(
            ResidencyReason::Prefetch.default_priority(),
            ResidencyPriority::Background
        );
        assert_eq!(
            ResidencyReason::RetainHot.default_priority(),
            ResidencyPriority::Low
        );
    }

    #[test]
    fn priority_ordering() {
        assert!(ResidencyPriority::Critical > ResidencyPriority::High);
        assert!(ResidencyPriority::High > ResidencyPriority::Low);
        assert!(ResidencyPriority::Low > ResidencyPriority::Background);
    }
}
