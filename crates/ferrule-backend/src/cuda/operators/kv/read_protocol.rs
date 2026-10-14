//! Completion protocol shared by the CUDA owner and driver-free fault tests.
//! A DMA acknowledgment never clears a submitted consumer's custody.
use crate::cpu::StreamingF32GqaConfig;
use ferrule_common::execution::{KvElementType, KvPlaneDescriptor};
use ferrule_common::host_kv::placement::PlacementCredits;
use ferrule_common::{Error, Result};

pub(super) fn error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("committed HostKV: {}", message.into()),
    }
}

#[derive(Default, Debug)]
pub(super) struct Custody {
    dma: bool,
    consumer: bool,
    unknown: bool,
    canceled: bool,
}
impl Custody {
    pub fn submit_dma(&mut self) -> Result<()> {
        if self.dma || self.unknown || self.canceled {
            return Err(error("DMA admission blocked"));
        }
        self.dma = true;
        Ok(())
    }
    pub fn dma_ack(&mut self) {
        self.dma = false;
    }
    pub fn submit_consumer(&mut self) -> Result<()> {
        if self.dma || self.consumer || self.unknown || self.canceled {
            return Err(error("consumer admission blocked"));
        }
        self.consumer = true;
        Ok(())
    }
    pub fn consumer_ack(&mut self) {
        self.consumer = false;
    }
    pub fn quarantine(&mut self) {
        self.unknown = true;
        self.canceled = true;
    }
    pub fn cancel(&mut self) {
        self.canceled = true;
    }
    pub fn canceled(&self) -> bool {
        self.canceled
    }
    pub fn unknown(&self) -> bool {
        self.unknown
    }
    pub fn quiescent(&self) -> bool {
        !self.dma && !self.consumer && !self.unknown
    }
    /// Both domains are evaluated before this call, even after an earlier
    /// failure. Successful DMA alone never clears unknown consumer custody.
    pub fn complete_reap(&mut self, dma: Result<()>, compute: Result<()>) -> Result<()> {
        if let Err(e) = dma.and(compute) {
            self.quarantine();
            return Err(e);
        }
        self.reaped();
        Ok(())
    }
    fn reaped(&mut self) {
        self.dma = false;
        self.consumer = false;
        self.unknown = false;
    }
}

/// One page-local read in an existing physical KV plane. The descriptor keeps
/// its element type, token width and layer count together at both call sites.
#[derive(Clone, Copy)]
pub(super) struct CommittedChunk {
    pub slot: u32,
    pub layer: usize,
    pub page_tokens: usize,
    pub plane: KvPlaneDescriptor,
    pub row: usize,
    pub tokens: usize,
}

/// Checked F32 addressing, shared with driver-free geometry regression tests.
pub(super) fn committed_chunk_range(
    chunk: CommittedChunk,
    storage_elements: usize,
) -> Result<std::ops::Range<usize>> {
    let CommittedChunk {
        slot,
        layer,
        page_tokens,
        plane,
        row,
        tokens,
    } = chunk;
    let layers = plane.layer_count;
    let width = plane.elements_per_token;
    if plane.element_type != KvElementType::F32
        || page_tokens == 0
        || width == 0
        || layer >= layers
        || tokens == 0
        || row.checked_add(tokens).is_none_or(|end| end > page_tokens)
    {
        return Err(error("invalid committed page/layer chunk"));
    }
    let start = (slot as usize)
        .checked_mul(layers)
        .and_then(|n| n.checked_add(layer))
        .and_then(|n| n.checked_mul(page_tokens))
        .and_then(|n| n.checked_add(row))
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| error("page byte geometry overflow"))?;
    let end = tokens
        .checked_mul(width)
        .and_then(|n| start.checked_add(n))
        .ok_or_else(|| error("chunk extent overflow"))?;
    if end > storage_elements
        || end
            .checked_mul(4)
            .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(error("committed F32 range exceeds allocation"));
    }
    Ok(start..end)
}

/// Fixed payload scratch plus bounded metadata per committed page. Includes both transport
/// slots, K/V staging, partial and output grants, CPU query/partials/merge copies,
/// and transient boxed D2H payloads. Existing resident KV and caller Q are excluded.
pub fn committed_gqa_credits(c: StreamingF32GqaConfig, pages: usize) -> Result<PlacementCredits> {
    c.validate()?;
    if c.rows != 1 {
        return Err(error("only a single committed query row is supported"));
    }
    for n in [c.q_heads, c.kv_heads, c.head_dim, c.block_tokens] {
        u32::try_from(n).map_err(|_| error("geometry exceeds CUDA ABI"))?;
    }
    if c.q_heads > i32::MAX as usize {
        return Err(error("query grid overflow"));
    }
    let q = c.query_elements()?;
    let kv = c.kv_elements(c.block_tokens)?;
    let partial = c
        .q_heads
        .checked_mul(c.head_dim.checked_add(2).ok_or_else(|| error("overflow"))?)
        .ok_or_else(|| error("overflow"))?;
    let chunk = q.max(kv).max(partial);
    let bytes = |n: usize| n.checked_mul(4).ok_or_else(|| error("byte overflow"));
    let mut device = 0usize;
    for n in [kv, kv, partial, q] {
        device = device
            .checked_add(
                crate::cuda_allocation_grant_bytes(bytes(n)?)
                    .ok_or_else(|| error("grant overflow"))?,
            )
            .ok_or_else(|| error("grant sum overflow"))?;
    }
    let host = q
        .checked_mul(3)
        .and_then(|n| n.checked_add(kv.checked_mul(2)?))
        .and_then(|n| n.checked_add(partial.checked_mul(4)?))
        .and_then(|n| n.checked_add(chunk))
        .ok_or_else(|| error("host budget overflow"))?;
    // Bound source descriptors, slot witnesses and owner metadata as well as
    // payloads. This is a conservative reservation, not an RSS measurement.
    let metadata = pages
        .checked_mul(512)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| error("metadata budget overflow"))?;
    let host_bytes = bytes(host)?
        .checked_add(metadata)
        .ok_or_else(|| error("host budget overflow"))?;
    Ok(PlacementCredits {
        pageable_host_bytes: u64::try_from(host_bytes).map_err(|_| error("budget overflow"))?,
        pinned_host_bytes: u64::try_from(
            bytes(chunk)?
                .checked_mul(2)
                .ok_or_else(|| error("pinned overflow"))?,
        )
        .map_err(|_| error("budget overflow"))?,
        staging_bytes: u64::try_from(device).map_err(|_| error("budget overflow"))?,
        staging_slots: 1,
        transfers: 2,
        consumers: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dma_is_not_consumer_ack_and_cancel_is_not_a_fence() {
        let mut c = Custody::default();
        c.submit_dma().unwrap();
        assert!(c.submit_consumer().is_err());
        c.dma_ack();
        c.submit_consumer().unwrap();
        c.cancel();
        c.dma_ack();
        assert!(!c.quiescent());
        assert!(c.submit_dma().is_err());
        c.consumer_ack();
        assert!(c.quiescent());
    }
    #[test]
    fn unknown_retains_custody_until_explicit_two_domain_reap() {
        let mut c = Custody::default();
        c.submit_dma().unwrap();
        c.quarantine();
        c.dma_ack();
        c.consumer_ack();
        assert!(!c.quiescent());
        assert!(c.unknown());
        c.reaped();
        assert!(c.quiescent());
        assert!(c.canceled());
    }
    #[test]
    fn partial_reaper_failure_keeps_pins_and_can_be_retried() {
        for failed_dma in [true, false] {
            let mut c = Custody::default();
            c.submit_consumer().unwrap();
            c.quarantine();
            let dma = if failed_dma {
                Err(error("DMA fence"))
            } else {
                Ok(())
            };
            let compute = if failed_dma {
                Ok(())
            } else {
                Err(error("consumer fence"))
            };
            assert!(c.complete_reap(dma, compute).is_err());
            assert!(!c.quiescent());
            assert!(c.submit_dma().is_err());
            c.complete_reap(Ok(()), Ok(())).unwrap();
            assert!(c.quiescent());
        }
    }
    #[test]
    fn physical_page_layer_byte_geometry_and_tail_are_exact() {
        let chunk = CommittedChunk {
            slot: 2,
            layer: 1,
            page_tokens: 4,
            plane: KvPlaneDescriptor::new("standard_gqa.key", 6, 3, KvElementType::F32),
            row: 2,
            tokens: 2,
        };
        let range = committed_chunk_range(chunk, 216).unwrap();
        assert_eq!(range, 180..192);
        assert_eq!(range.start * 4..range.end * 4, 720..768);
        assert!(committed_chunk_range(CommittedChunk { tokens: 3, ..chunk }, 216).is_err());
        assert!(committed_chunk_range(CommittedChunk { layer: 3, ..chunk }, 216).is_err());
        assert!(committed_chunk_range(chunk, 191).is_err());
        assert!(
            committed_chunk_range(
                CommittedChunk {
                    plane: KvPlaneDescriptor::new(
                        "standard_gqa.key",
                        usize::MAX,
                        3,
                        KvElementType::F32
                    ),
                    tokens: 1,
                    ..chunk
                },
                usize::MAX
            )
            .is_err()
        );
        assert!(committed_chunk_range(CommittedChunk { tokens: 0, ..chunk }, 216).is_err());
        assert!(
            committed_chunk_range(
                CommittedChunk {
                    row: usize::MAX,
                    ..chunk
                },
                216
            )
            .is_err()
        );
        assert!(
            committed_chunk_range(
                CommittedChunk {
                    page_tokens: 0,
                    ..chunk
                },
                216
            )
            .is_err()
        );
        assert!(
            committed_chunk_range(
                CommittedChunk {
                    plane: KvPlaneDescriptor::new("standard_gqa.key", 6, 3, KvElementType::Bf16),
                    ..chunk
                },
                216
            )
            .is_err()
        );
    }
    #[test]
    fn six_domains_include_allocator_grants_and_overflow() {
        let c = StreamingF32GqaConfig {
            rows: 1,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            block_tokens: 5,
            softmax_scale: 0.5,
            sliding_window: None,
        };
        let credits = committed_gqa_credits(c, 2).unwrap();
        assert_eq!(credits.staging_bytes, 4 * 256);
        assert_eq!(credits.pinned_host_bytes, 2 * 30 * 4);
        assert_eq!(
            (credits.staging_slots, credits.transfers, credits.consumers),
            (1, 2, 1)
        );
        assert!(credits.pageable_host_bytes >= (2 * 30 + 12 + 20) * 4);
        assert!(committed_gqa_credits(StreamingF32GqaConfig { rows: 2, ..c }, 2).is_err());
        assert!(
            committed_gqa_credits(
                StreamingF32GqaConfig {
                    head_dim: usize::MAX,
                    ..c
                },
                2
            )
            .is_err()
        );
    }
}
