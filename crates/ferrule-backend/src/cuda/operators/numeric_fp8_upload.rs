//! Opt-in retained upload of three immutable numeric-FP8 linears, in input order.
//!
//! Staging allocates but submits no DMA. Submission enqueues six exact U8 copies,
//! one private, never-re-recorded event and a compute dependency. Only a successful
//! query of that event permits publication and host-source release. Neither the
//! dependency nor a notification is host completion. No production caller opts in.
//!
//! Only submit/poll avoid explicit host fences. Staging performs CPU copies and
//! CUDA allocations, which can block; existing allocator retirement maintenance
//! also has error-only synchronization fallbacks. Explicit drain and Drop fence
//! outstanding work. This leaf makes no end-to-end overlap/performance claim.

use super::{CudaOperators, cu};
use crate::cuda::operators::linear::CudaNumericFp8Artifact;
use crate::cuda::runtime::{
    self, CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer,
};
use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;
use ferrule_common::{Error, Result};
use std::{marker::PhantomData, rc::Rc, sync::Arc};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("retained numeric FP8 upload: {}", message.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteRange {
    offset: usize,
    len: usize,
}

impl ByteRange {
    fn checked(offset: usize, len: usize, allocation: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| invalid("range overflow"))?;
        if len == 0 || end > allocation || allocation > isize::MAX as usize {
            return Err(invalid("range outside pinned slab"));
        }
        Ok(Self { offset, len })
    }
}

#[derive(Debug, Clone, Copy)]
struct SlabLayout {
    ranges: [ByteRange; 6],
    bytes: usize,
}

impl SlabLayout {
    fn new(lengths: [usize; 6]) -> Result<Self> {
        let bytes = lengths.into_iter().try_fold(0usize, |sum, len| {
            sum.checked_add(len)
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or_else(|| invalid("slab allocation size overflow"))
        })?;
        let mut offset = 0;
        let mut ranges = [ByteRange { offset: 0, len: 0 }; 6];
        for (range, len) in ranges.iter_mut().zip(lengths) {
            *range = ByteRange::checked(offset, len, bytes)?;
            offset += len;
        }
        Ok(Self { ranges, bytes })
    }

    fn for_payloads(payloads: &[ImmutableValidatedNumericFp8Payload; 3]) -> Result<Self> {
        for payload in payloads {
            payload
                .layout()
                .validate_lengths(payload.weight_bytes().len(), payload.scale_bytes().len())?;
        }
        Self::new(std::array::from_fn(|i| {
            let payload = &payloads[i / 2];
            if i % 2 == 0 {
                payload.weight_bytes().len()
            } else {
                payload.scale_bytes().len()
            }
        }))
    }

    fn pack(
        &self,
        payloads: &[ImmutableValidatedNumericFp8Payload; 3],
        slab: &mut [u8],
    ) -> Result<()> {
        if slab.len() != self.bytes {
            return Err(invalid("pinned allocation extent mismatch"));
        }
        for (i, range) in self.ranges.iter().enumerate() {
            let payload = &payloads[i / 2];
            let bytes = if i % 2 == 0 {
                payload.weight_bytes()
            } else {
                payload.scale_bytes()
            };
            ByteRange::checked(range.offset, bytes.len(), slab.len())?;
            if bytes.len() != range.len {
                return Err(invalid("payload range length mismatch"));
            }
            slab[range.offset..range.offset + range.len].copy_from_slice(bytes);
        }
        Ok(())
    }
}

/// Explicit per-ticket host storage charge, not RSS or a model-cache admission.
///
/// Charges the full pageable payload even when its immutable Arcs are shared,
/// PLUS the entire requested pinned allocation, PLUS inline ticket metadata.
/// Arc/allocator headers, OS/driver page rounding and allocator-private metadata
/// are not observable here and require separate caller headroom/qualification.
/// In particular this does not qualify memlock or integrate a host64MiB policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaNumericFp8UploadHostCharge {
    pub pageable_payload_bytes: usize,
    pub pinned_allocation_bytes: usize,
    pub ticket_metadata_bytes: usize,
    pub total_bytes: usize,
}

impl CudaNumericFp8UploadHostCharge {
    pub fn for_payloads(payloads: &[ImmutableValidatedNumericFp8Payload; 3]) -> Result<Self> {
        Self::for_bytes(SlabLayout::for_payloads(payloads)?.bytes)
    }

    fn for_bytes(bytes: usize) -> Result<Self> {
        let ticket_metadata_bytes = std::mem::size_of::<CudaNumericFp8UploadTicket>();
        let total_bytes = bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(ticket_metadata_bytes))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("pageable + pinned host charge overflow"))?;
        Ok(Self {
            pageable_payload_bytes: bytes,
            pinned_allocation_bytes: bytes,
            ticket_metadata_bytes,
            total_bytes,
        })
    }

    fn admit(self, host_budget_bytes: usize) -> Result<()> {
        if self.total_bytes > host_budget_bytes {
            return Err(invalid(format!(
                "host budget exceeded: pageable={} pinned={} ticket={} total={} budget={host_budget_bytes}",
                self.pageable_payload_bytes,
                self.pinned_allocation_bytes,
                self.ticket_metadata_bytes,
                self.total_bytes
            )));
        }
        Ok(())
    }
}

fn next_generation(current: u64) -> Result<u64> {
    current
        .checked_add(1)
        .ok_or_else(|| invalid("owner generation exhausted"))
}

fn exact_owner<O, S>(
    owner: &Arc<O>,
    upload: &Arc<S>,
    compute: &Arc<S>,
    expected_owner: &Arc<O>,
    expected_upload: &Arc<S>,
    expected_compute: &Arc<S>,
) -> bool {
    Arc::ptr_eq(owner, expected_owner)
        && Arc::ptr_eq(upload, expected_upload)
        && Arc::ptr_eq(compute, expected_compute)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Staged,
    // Set BEFORE the first driver call; even an unwind may have submitted DMA.
    Copying,
    Recorded,
    InFlight,
    Unknown,
    Retired,
}

// Production orchestration and CPU failure-ordering tests use this same state
// machine. The private implementation owns every DMA operand and driver handle.
trait UploadResources {
    fn copy(&self, index: usize) -> Result<()>;
    fn record(&self) -> Result<()>;
    fn wait_compute(&self) -> Result<()>;
    fn query(&self) -> Result<bool>;
    fn fence_event(&self) -> Result<()>;
    fn fence_stream(&self) -> Result<()>;
}

struct Custody<B: UploadResources> {
    bundle: Option<B>,
    phase: Phase,
}

impl<B: UploadResources> Custody<B> {
    fn new(bundle: B) -> Self {
        Self {
            bundle: Some(bundle),
            phase: Phase::Staged,
        }
    }

    fn submit(&mut self) -> Result<()> {
        if self.phase != Phase::Staged {
            return Err(invalid("ticket is not staged; submission is never retried"));
        }
        self.phase = Phase::Copying;
        let bundle = self.bundle.as_ref().expect("staged custody");
        for i in 0..6 {
            bundle.copy(i)?;
        }
        bundle.record()?;
        self.phase = Phase::Recorded;
        bundle.wait_compute()?;
        self.phase = Phase::InFlight;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<B>> {
        if self.phase != Phase::InFlight {
            return Err(invalid(
                "ticket is not in flight (failed, unknown, staged or retired)",
            ));
        }
        // Query failure OR unwind is sticky. A later fence must not turn Unknown
        // into successful publication or permit destruction of the bundle.
        self.phase = Phase::Unknown;
        match self.bundle.as_ref().expect("in-flight custody").query()? {
            false => {
                self.phase = Phase::InFlight;
                Ok(None)
            }
            true => {
                self.phase = Phase::Retired;
                Ok(self.bundle.take())
            }
        }
    }

    fn drain(&mut self) -> Result<()> {
        let phase = self.phase;
        if phase == Phase::Retired {
            return Ok(());
        }
        if phase == Phase::Unknown {
            return Err(invalid("completion Unknown is terminal; custody retained"));
        }
        self.phase = Phase::Unknown;
        let bundle = self.bundle.as_ref().expect("live custody");
        match phase {
            Phase::Staged => (),
            Phase::Copying => bundle.fence_stream()?,
            Phase::Recorded | Phase::InFlight => bundle.fence_event()?,
            Phase::Unknown | Phase::Retired => unreachable!(),
        }
        self.phase = Phase::Retired;
        self.bundle = None;
        Ok(())
    }
}

impl<B: UploadResources> Drop for Custody<B> {
    fn drop(&mut self) {
        let phase = self.phase;
        if let Some(bundle) = self.bundle.take() {
            crate::cuda::transport::drop_after_proof(bundle, |bundle| {
                let proof = match phase {
                    Phase::Staged => Ok(()),
                    Phase::Copying => bundle.fence_stream(),
                    Phase::Recorded | Phase::InFlight => bundle.fence_event(),
                    Phase::Unknown | Phase::Retired => {
                        Err(invalid("completion Unknown; custody retained"))
                    }
                };
                if let Err(error) = proof {
                    tracing::error!(%error, "numeric FP8 upload teardown quarantined entire bundle; restart worker to recover");
                    false
                } else {
                    true
                }
            });
        }
    }
}

struct UploadBundle {
    payloads: [ImmutableValidatedNumericFp8Payload; 3],
    pinned: PinnedHostBuffer<u8>,
    destinations: [DeviceBuffer<u8>; 6],
    layout: SlabLayout,
    context: Arc<CudaContext>,
    upload: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    event: CudaEvent,
}

impl UploadResources for UploadBundle {
    fn copy(&self, index: usize) -> Result<()> {
        let range = self.layout.ranges[index];
        let destination = &self.destinations[index];
        if destination.len() != range.len {
            return Err(invalid("destination range mismatch"));
        }
        // SAFETY: Custody is installed before this call. Pinned bytes are private
        // and immutable, each destination is exclusive, and the whole bundle is
        // kept until exact completion (or permanently retained on Unknown).
        cu(unsafe {
            runtime::copy_host_to_device_pinned_range(
                &self.upload,
                destination,
                &self.pinned,
                range.offset,
            )
        })
    }
    fn record(&self) -> Result<()> {
        cu(self.event.record(&self.upload))
    }
    fn wait_compute(&self) -> Result<()> {
        cu(self.compute.wait(&self.event))
    }
    fn query(&self) -> Result<bool> {
        cu(self.event.query())
    }
    fn fence_event(&self) -> Result<()> {
        cu(self.event.synchronize())
    }
    fn fence_stream(&self) -> Result<()> {
        cu(self.upload.synchronize())
    }
}

/// Non-cloneable, owner-local ticket. No source, destination or event is exposed
/// before an observed ready query. A submission error leaves custody HERE.
///
/// Prefer explicit drain for error reporting. Dropping a live ticket can block
/// on its exact event, or its upload stream if recording did not succeed. Failed
/// cleanup, query error, or cleanup unwind retains the ENTIRE bundle permanently;
/// there is no registry, worker thread, re-record, or success-path leak.
///
/// Tickets cannot cross owner threads, even though their driver handles can.
/// ```compile_fail,E0277
/// use ferrule_backend::cuda::operators::linear::CudaNumericFp8UploadTicket;
/// fn require_send<T: Send>() {}
/// require_send::<CudaNumericFp8UploadTicket>();
/// ```
/// The completion event cannot be extracted or re-recorded.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::linear::CudaNumericFp8UploadTicket;
/// fn event(ticket: &CudaNumericFp8UploadTicket) { let _ = &ticket.custody; }
/// ```
#[must_use = "submit and poll, or explicitly drain the retained upload"]
pub struct CudaNumericFp8UploadTicket {
    custody: Custody<UploadBundle>,
    generation: u64,
    charge: CudaNumericFp8UploadHostCharge,
    _owner_local: PhantomData<Rc<()>>,
}

impl CudaNumericFp8UploadTicket {
    /// Monotonic within the exact operator owner, not cross-owner authority.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn host_charge(&self) -> CudaNumericFp8UploadHostCharge {
        self.charge
    }
}

impl CudaOperators {
    /// Stage all custody before DMA. Inputs remain in their original packed
    /// E4M3FN + BF16/F32-scale representation; there is no numerical conversion.
    /// The explicit budget charges pageable AND pinned storage, not just P.
    /// Allocation/driver pin qualification can fail independently of admission.
    pub fn stage_numeric_fp8_upload(
        &self,
        payloads: [ImmutableValidatedNumericFp8Payload; 3],
        host_budget_bytes: usize,
    ) -> Result<CudaNumericFp8UploadTicket> {
        let layout = SlabLayout::for_payloads(&payloads)?;
        let charge = CudaNumericFp8UploadHostCharge::for_bytes(layout.bytes)?;
        charge.admit(host_budget_bytes)?;
        self.check_capture_safe("retained numeric FP8 upload staging")?;
        cu(self._ctx.check_graph_ready())?;
        let generation = next_generation(self.numeric_fp8_upload_generation.get())?;
        // All of these operations are allocation-only. In particular do NOT
        // initialize on compute and then race the upload stream's writes.
        let mut pinned = cu(PinnedHostBuffer::<u8>::zeroed(&self._ctx, layout.bytes))?;
        layout.pack(&payloads, pinned.as_mut_slice())?;
        let destinations = [
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[0].len)?,
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[1].len)?,
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[2].len)?,
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[3].len)?,
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[4].len)?,
            self.uninitialized_upload_device_buffer::<u8>(layout.ranges[5].len)?,
        ];
        let event = cu(self._ctx.new_event(false))?;
        let bundle = UploadBundle {
            payloads,
            pinned,
            destinations,
            layout,
            context: Arc::clone(&self._ctx),
            upload: Arc::clone(&self.upload_stream),
            compute: Arc::clone(&self.stream),
            event,
        };
        self.numeric_fp8_upload_generation.set(generation);
        Ok(CudaNumericFp8UploadTicket {
            custody: Custody::new(bundle),
            generation,
            charge,
            _owner_local: PhantomData,
        })
    }

    fn check_numeric_fp8_upload_owner(&self, ticket: &CudaNumericFp8UploadTicket) -> Result<()> {
        if let Some(bundle) = ticket.custody.bundle.as_ref()
            && exact_owner(
                &bundle.context,
                &bundle.upload,
                &bundle.compute,
                &self._ctx,
                &self.upload_stream,
                &self.stream,
            )
            && ticket.generation != 0
            && ticket.generation <= self.numeric_fp8_upload_generation.get()
        {
            return Ok(());
        }
        Err(invalid("foreign or retired ticket"))
    }

    /// Enqueue only: six range copies, one event record, one compute dependency.
    /// An error must not be retried; the caller still owns all possibly live DMA.
    pub fn submit_numeric_fp8_upload(&self, ticket: &mut CudaNumericFp8UploadTicket) -> Result<()> {
        self.check_numeric_fp8_upload_owner(ticket)?;
        self.check_capture_safe("retained numeric FP8 upload submission")?;
        cu(self._ctx.check_graph_ready())?;
        ticket.custody.submit()?;
        self.counters
            .add_host_to_device(ticket.charge.pinned_allocation_bytes as u64);
        Ok(())
    }

    /// Nonblocking host-completion query. Pending retains every operand. Query
    /// errors are terminal Unknown, never successful completion on a later poll.
    /// Published artifacts use the existing owner/shape/alias launch preflight.
    pub fn poll_numeric_fp8_upload(
        &self,
        ticket: &mut CudaNumericFp8UploadTicket,
    ) -> Result<Option<[CudaNumericFp8Artifact; 3]>> {
        self.check_numeric_fp8_upload_owner(ticket)?;
        self.check_capture_safe("retained numeric FP8 upload query")?;
        let Some(bundle) = ticket.custody.poll()? else {
            return Ok(None);
        };
        let layouts = bundle.payloads.each_ref().map(|p| p.layout());
        let [w0, s0, w1, s1, w2, s2] = bundle.destinations;
        let artifacts = [
            (layouts[0], w0, s0),
            (layouts[1], w1, s1),
            (layouts[2], w2, s2),
        ]
        .map(|(layout, weight, scales)| {
            CudaNumericFp8Artifact::from_device_buffers(layout, weight, scales)
        });
        self.counters
            .add_artifact_upload(ticket.charge.pinned_allocation_bytes as u64);
        Ok(Some(artifacts))
    }

    /// Explicit synchronous cancellation/cleanup, NEVER successful publication.
    /// Recorded work fences its exact event; partial/unrecorded submission fences
    /// its upload stream. Any failed proof is sticky Unknown and retains custody.
    pub fn drain_numeric_fp8_upload(&self, ticket: &mut CudaNumericFp8UploadTicket) -> Result<()> {
        self.check_numeric_fp8_upload_owner(ticket)?;
        self.check_capture_safe("retained numeric FP8 upload drain")?;
        if ticket.custody.phase == Phase::Copying {
            self.record_stream_wide_sync(|| ticket.custody.drain())
        } else {
            ticket.custody.drain()
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/numeric_fp8_async_upload/contract.rs"]
mod contract_tests;
