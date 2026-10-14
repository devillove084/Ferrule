//! Pure physical-capacity planning; no device initialization or page ownership.
use super::{Result, error};
use ferrule_model::ModelExecutionBackend;

/// Transaction custody, independently of the device's slot representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustodyStrategy {
    InPlace,
    /// One working/shadow slot per logical page; quarantine stays inside 2L.
    Shadowed,
}

/// Pool extent limits, not a choice of transaction custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalSlotLimit {
    U32,
    /// Zero-based signed indices permit i32::MAX + 1 slots.
    I32,
}
impl PhysicalSlotLimit {
    pub const fn for_backend(backend: ModelExecutionBackend) -> Self {
        match backend {
            ModelExecutionBackend::Cpu => Self::U32,
            ModelExecutionBackend::Cuda => Self::I32,
        }
    }
    pub const fn max_slots(self) -> usize {
        match self {
            Self::U32 => u32::MAX as usize,
            Self::I32 => i32::MAX as usize + 1,
        }
    }
}

/// Metadata only: neither an allocator nor a second custody authority.
#[doc = include_str!("../../../tests/pipeline_capacity/compile.md")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalKvLayout {
    pub custody: CustodyStrategy,
    pub slot_limit: PhysicalSlotLimit,
}
impl PhysicalKvLayout {
    pub const fn new(custody: CustodyStrategy, slot_limit: PhysicalSlotLimit) -> Self {
        Self {
            custody,
            slot_limit,
        }
    }
    pub const fn standard(backend: ModelExecutionBackend) -> Self {
        let custody = match backend {
            ModelExecutionBackend::Cpu => CustodyStrategy::InPlace,
            ModelExecutionBackend::Cuda => CustodyStrategy::Shadowed,
        };
        Self::new(custody, PhysicalSlotLimit::for_backend(backend))
    }
    pub const fn hybrid(backend: ModelExecutionBackend) -> Self {
        Self::new(
            CustodyStrategy::Shadowed,
            PhysicalSlotLimit::for_backend(backend),
        )
    }
    pub fn physical_pages(self, logical_pages: usize) -> Result<usize> {
        let multiplier = match self.custody {
            CustodyStrategy::InPlace => 1,
            CustodyStrategy::Shadowed => 2,
        };
        logical_pages
            .checked_mul(multiplier)
            .filter(|&pages| pages > 0 && pages <= self.slot_limit.max_slots())
            .ok_or_else(|| error("pipeline physical slot capacity overflow"))
    }
    pub fn validate_pages(self, logical_pages: usize, physical_pages: usize) -> Result<()> {
        if self.physical_pages(logical_pages)? != physical_pages {
            return Err(error(
                "pipeline physical capacity differs from custody layout",
            ));
        }
        Ok(())
    }
}
