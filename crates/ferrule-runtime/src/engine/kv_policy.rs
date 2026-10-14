//! Explicit serving policy; selection never grants KV authority.
//!
//! HostKV serving is resident prefill warming followed by synchronous single-row
//! decode through the native executor. The local dense CUDA factory installs the
//! typed dispatch; GDN/numeric/parallel and other uninstalled engines fail before
//! Ready. Runtime borrows its existing manager and reservation for each call;
//! only the selected mode (not an authority or witness) lives in the transaction.
//! Errors use the same prepare-Unknown / execute-abort / retirement protocols.
//! This is not resident-page eviction and does not promise KV capacity savings.
use crate::{Error, Result};
use ferrule_common::execution::{ExecutionBatch, ExecutionIntent, ForwardMode, ForwardPhase};
use ferrule_common::host_kv::placement::PlacementCredits;
pub use ferrule_model::decoder::{IdleKvMemoryPlan, IdleKvPhysicalStats, IdleKvReclaimConfig};

/// Owner-local HostKV scratch limits, shared across every layer and transaction.
/// This does not evict resident KV or reduce the configured physical KV capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostKvDecodeConfig {
    pub block_tokens: usize,
    pub capacity: PlacementCredits,
}
impl Default for HostKvDecodeConfig {
    fn default() -> Self {
        Self {
            block_tokens: 16,
            capacity: PlacementCredits {
                pageable_host_bytes: 64 << 20,
                pinned_host_bytes: 64 << 20,
                staging_bytes: 64 << 20,
                staging_slots: 4,
                transfers: 4,
                consumers: 4,
            },
        }
    }
}
impl HostKvDecodeConfig {
    /// Validate bounded startup budgets without claiming any physical resources.
    pub fn validate(self) -> Result<()> {
        let c = self.capacity;
        if self.block_tokens == 0
            || self.block_tokens > u32::MAX as usize
            || [
                c.pageable_host_bytes,
                c.pinned_host_bytes,
                c.staging_bytes,
                c.staging_slots,
                c.transfers,
                c.consumers,
            ]
            .contains(&0)
        {
            return Err(invalid(
                "HostKV requires nonzero bounded block size and all six budget domains",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvExecutionPolicy {
    #[default]
    Resident,
    /// Resident prefill warming, then synchronous single-token F32 dense GQA
    /// HostKV decode. Unsupported compositions fail at startup, never fallback.
    HostKvDecode(HostKvDecodeConfig),
    /// Device attention with explicit idle-session suspension and true page reclamation.
    /// Never routes active decode through the HostKV prefix bridge.
    IdleReclaim(IdleKvReclaimConfig),
}
impl KvExecutionPolicy {
    pub(crate) fn unsupported(self) -> Result<()> {
        match self {
            Self::Resident => Ok(()),
            Self::IdleReclaim(_) => Err(invalid(
                "idle KV reclaim requires an installed chunked dense F32 CUDA runner",
            )),
            Self::HostKvDecode(_) => Err(invalid(
                "HostKV decode is not installed for this composition; requires local CUDA dense F32 all-GQA, no GDN/numeric/TP/EP",
            )),
        }
    }
    pub(crate) fn select(self, batch: &ExecutionBatch) -> Result<KvExecutionMode> {
        if matches!(self, Self::Resident | Self::IdleReclaim(_)) {
            return Ok(KvExecutionMode::Resident);
        }
        if batch.intent() != ExecutionIntent::Committed || batch.sequences().len() != 1 {
            return Err(invalid(
                "HostKV policy rejects speculation and multisequence batches",
            ));
        }
        let sequence = &batch.sequences()[0];
        match (batch.mode(), sequence.phase) {
            (ForwardMode::Prefill, ForwardPhase::Prefill) if !batch.is_empty() => {
                Ok(KvExecutionMode::ResidentWarming)
            }
            (ForwardMode::Decode, ForwardPhase::Decode)
                if batch.len() == 1 && batch.positions().first().is_some_and(|p| *p > 0) =>
            {
                Ok(KvExecutionMode::HostKvDecode)
            }
            _ => Err(invalid(
                "HostKV decode requires one token and a nonempty prefix; mixed/multiquery batches have no resident fallback",
            )),
        }
    }
}

/// Recorded on the existing transaction; contains no borrowed authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvExecutionMode {
    Resident,
    ResidentWarming,
    HostKvDecode,
}
impl KvExecutionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::ResidentWarming => "resident_warming",
            Self::HostKvDecode => "host_kv_decode",
        }
    }
}

/// Counts completed forward calls, not publication or physical retirement ACKs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvExecutionStats {
    pub resident_warming: u64,
    pub host_kv_decode: u64,
}

pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidRequest {
        message: message.into(),
    }
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata validation is independent of the concrete CUDA installer"
    )
)]
pub fn validate_host_kv_spec(
    spec: &ferrule_model::transformer::DecoderModelSpec,
    numeric: bool,
) -> Result<()> {
    use ferrule_model::transformer::{Attention, FeedForward};
    if spec.layers().is_empty()
        || !spec.attachments().is_empty()
        || numeric
        || spec.layers().iter().any(|l| {
            !matches!(l.attention(), Attention::Gqa(_))
                || !matches!(l.feed_forward(), FeedForward::SwiGlu(_))
        })
    {
        return Err(invalid(
            "HostKV requires dense F32 all-GQA; GDN, numeric and MoE/EP are unsupported",
        ));
    }
    Ok(())
}
