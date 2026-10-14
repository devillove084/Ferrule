//! CPU-testable contract for the internal two-member ordered BF16 reduction.
//! NCCL-private device/host allocations are unknown, never a zero-byte reserve.
use ferrule_common::{Error, Result};

pub const NCCL_VERSION: i32 = 22705;
pub const NCCL_LIBRARY_SHA256: &str =
    "1d9c10322efacde695cf640db1f29fc1586c929f1087f0e1202d2d5e70fb7168";
pub const NCCL_HEADER_SHA256: &str =
    "4b240a1fa79858ca8cdffcf5c2fc40aeaca8b3eaec61728213715f0902aba3a8";
pub const MAX_ROWS: usize = 8;
pub const HIDDEN: usize = 2048;
pub const MAX_COUNT: usize = MAX_ROWS * HIDDEN;
pub const STORAGE_WORDS: usize = MAX_COUNT * 4 + 2;
pub const STORAGE_BYTES: usize = STORAGE_WORDS * 2;
pub const STORAGE_GRANT_BYTES: usize = STORAGE_BYTES.next_power_of_two();
pub const INPUT_NONFINITE: u32 = 1;
pub const OUTPUT_NONFINITE: u32 = 2;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("internal TP ordered BF16: {}", message.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivateAllocationBound {
    Unknown,
}

/// Per-rank owned storage, NOT an admissible total physical resource claim.
/// A model adapter must reject hard-cap admission while a private bound is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpResourceEstimate {
    pub owned_device_bytes: usize,
    pub owned_device_grant_bytes: usize,
    pub nccl_private_device: PrivateAllocationBound,
    pub nccl_private_host: PrivateAllocationBound,
    pub configured_min_ctas: u32,
    pub configured_max_ctas: u32,
    /// NCCL 2.27.5 has no per-communicator ring/sliding-buffer size setter.
    pub configured_sliding_buffer_bytes: Option<usize>,
}
impl TpResourceEstimate {
    pub fn hard_cap_bytes(self) -> Result<usize> {
        Err(invalid(
            "NCCL private workspace is unbounded; not eligible for a hard-cap claim",
        ))
    }
}

/// Preallocated input views retained jointly until one batch quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpInputCustodyEstimate {
    pub queued_sites: usize,
    pub input_view_bytes: usize,
}
pub(crate) fn checked_input_custody(
    slots: usize,
    bytes_per_slot: usize,
    budget: usize,
) -> Result<usize> {
    let bytes = slots
        .checked_mul(bytes_per_slot)
        .filter(|&bytes| slots != 0 && bytes_per_slot != 0 && bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("input custody size overflow/empty capacity"))?;
    if bytes > budget {
        return Err(invalid("input custody host budget insufficient"));
    }
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpReducePlan {
    members: [u64; 2],
}
impl TpReducePlan {
    /// Sealed member order IS the NCCL rank mapping. Never sort actual rank IDs.
    pub fn new(members: [u64; 2], per_rank_storage_budget: usize) -> Result<Self> {
        if members[0] == members[1] {
            return Err(invalid("members must be distinct"));
        }
        if per_rank_storage_budget < STORAGE_GRANT_BYTES {
            return Err(invalid("per-rank owned-storage budget is insufficient"));
        }
        Ok(Self { members })
    }
    pub fn members(self) -> [u64; 2] {
        self.members
    }
    pub fn count(self, rows: usize) -> Result<usize> {
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(invalid("rows must be 1..=8"));
        }
        Ok(rows * HIDDEN)
    }
    pub fn estimate(self) -> TpResourceEstimate {
        TpResourceEstimate {
            owned_device_bytes: STORAGE_BYTES,
            owned_device_grant_bytes: STORAGE_GRANT_BYTES,
            nccl_private_device: PrivateAllocationBound::Unknown,
            nccl_private_host: PrivateAllocationBound::Unknown,
            configured_min_ctas: 2,
            configured_max_ctas: 2,
            configured_sliding_buffer_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpProgress {
    Pending,
    Enqueued,
    Complete,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Initializing,
    Ready,
    Submitting,
    InFlight,
    Rejected,
    Unknown,
}
impl Phase {
    pub(crate) fn can_release(self, completion_proven: bool) -> bool {
        completion_proven && !matches!(self, Self::Unknown | Self::Initializing | Self::Submitting)
    }
}

/// Mirrors HostCollectiveGroup::compute: first member, then second, no initial +0.
#[cfg(test)]
pub(crate) fn ordered_host_word(first: u16, second: u16) -> std::result::Result<u16, u32> {
    let a = f32::from_bits(u32::from(first) << 16);
    let b = f32::from_bits(u32::from(second) << 16);
    if !a.is_finite() || !b.is_finite() {
        return Err(INPUT_NONFINITE);
    }
    let sum = a + b;
    let bits = sum.to_bits();
    let word = (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
    if !sum.is_finite() || word & 0x7f80 == 0x7f80 {
        return Err(OUTPUT_NONFINITE);
    }
    Ok(word)
}
