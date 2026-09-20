//! Fixed, owner-local pinned F32 transfer chunks, independent of collectives.
//!
//! A ticket never owns the DMA resources: dropping it does not free or reuse a
//! slot. The transport retains both pinned storage and a device allocation view
//! until a driver fence proves completion. Cancellation is intent, not a fence.
//! None of these APIs may be used inside CUDA graph capture.

use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::Error;

use super::context::{CudaComputeEvent, CudaF32Buffer, cu};
use super::runtime::{self, CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};

static NEXT_TRANSPORT_ID: AtomicU64 = AtomicU64::new(1);
type Result<T> = std::result::Result<T, CudaTransferError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaTransferConfig {
    pub chunk_elements: usize,
    pub tx_slots: usize,
    pub rx_slots: usize,
}

impl CudaTransferConfig {
    /// Validates the complete fixed pinned and slot-metadata budget, without CUDA.
    pub fn validate(self) -> Result<()> {
        self.layout().map(|_| ())
    }

    fn layout(self) -> Result<(usize, usize)> {
        let invalid = CudaTransferError::InvalidConfig;
        if self.chunk_elements == 0 || self.tx_slots == 0 || self.rx_slots == 0 {
            return Err(invalid(
                "chunk_elements, tx_slots and rx_slots must be nonzero",
            ));
        }
        if self.tx_slots > u32::MAX as usize || self.rx_slots > u32::MAX as usize {
            return Err(invalid("slot count exceeds the ticket index range"));
        }
        let slots = self
            .tx_slots
            .checked_add(self.rx_slots)
            .ok_or(invalid("slot count overflow"))?;
        let bytes = self
            .chunk_elements
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_mul(slots))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(invalid("pinned allocation budget overflow"))?;
        slots
            .checked_mul(size_of::<Slot>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(invalid("slot metadata budget overflow"))?;
        Ok((slots, bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaTransferDirection {
    H2D,
    D2H,
}

/// Host-only identity; not a CUDA handle or authority to access a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaTransferId {
    transport: u64,
    direction: CudaTransferDirection,
    slot: u32,
    generation: u64,
}

impl CudaTransferId {
    pub const fn transport(self) -> u64 {
        self.transport
    }
    pub const fn direction(self) -> CudaTransferDirection {
        self.direction
    }
    pub const fn slot(self) -> u32 {
        self.slot
    }
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Non-cloneable and !Send/!Sync. The private event is never re-recorded.
/// Dropping a ticket leaves custody in its transport until drain/Drop.
#[derive(Debug)]
#[must_use = "poll the ticket or drain its transport to retire DMA custody"]
pub struct CudaTransferTicket {
    id: CudaTransferId,
    event: Arc<CudaEvent>,
    _owner_local: PhantomData<Rc<()>>,
}

impl CudaTransferTicket {
    pub const fn id(&self) -> CudaTransferId {
        self.id
    }
}

#[derive(Debug, PartialEq)]
pub enum CudaTransferCompletion {
    H2D {
        id: CudaTransferId,
    },
    D2H {
        id: CudaTransferId,
        values: Box<[f32]>,
    },
    Canceled {
        id: CudaTransferId,
    },
}

#[derive(Debug)]
pub enum CudaTransferError {
    InvalidConfig(&'static str),
    InvalidLength {
        elements: usize,
        capacity: usize,
    },
    Backpressure(CudaTransferDirection),
    IdentityExhausted,
    ForeignTicket,
    StaleTicket,
    WrongDirection,
    Canceled,
    Quarantined,
    CaptureUnsupported,
    Backend(Error),
    /// No ticket was returned. A failed cleanup leaves the slot and allocation
    /// retained in a quarantined transport; retry drain or retain the owner.
    Submission {
        source: Error,
        cleanup: Option<Error>,
    },
}

impl fmt::Display for CudaTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA async transport: ")?;
        match self {
            Self::Backend(error) => error.fmt(f),
            Self::Submission { source, cleanup } => {
                write!(f, "submission failed ({source}); ")?;
                match cleanup {
                    Some(error) => write!(f, "cleanup fence failed ({error}); custody retained"),
                    None => write!(f, "cleanup fence completed"),
                }
            }
            other => write!(f, "{other:?}"),
        }
    }
}

impl std::error::Error for CudaTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(source) | Self::Submission { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<Error> for CudaTransferError {
    fn from(error: Error) -> Self {
        Self::Backend(error)
    }
}

/// A partial drain never loses the identities it has already retired.
#[derive(Debug)]
pub struct CudaTransferDrainError {
    pub completed: Vec<CudaTransferCompletion>,
    pub source: CudaTransferError,
}

impl fmt::Display for CudaTransferDrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(f)
    }
}
impl std::error::Error for CudaTransferDrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaAsyncTransportStats {
    pub tx_slots: usize,
    pub rx_slots: usize,
    pub tx_in_use: usize,
    pub rx_in_use: usize,
    pub tx_high_water: usize,
    pub rx_high_water: usize,
    /// One pinned allocation per slot, made only at construction.
    pub pinned_allocations: usize,
    pub pinned_bytes: usize,
    /// Allocation views held by active DMA, not new device allocations.
    pub device_holds_in_use: usize,
    pub device_holds_high_water: usize,
    /// Admitted attempts, including submissions whose cleanup later failed.
    pub submissions: u64,
    pub failed_submissions: u64,
    pub completions: u64,
    pub cancel_intents: u64,
    pub canceled_completions: u64,
    pub quarantined: bool,
}

struct Slot {
    pinned: PinnedHostBuffer<f32>,
    generation: u64,
    active: Option<ActiveTransfer>,
}

struct ActiveTransfer {
    id: CudaTransferId,
    event: Arc<CudaEvent>,
    // An unrecorded CUDA event may query as ready. Never use it as a fence!
    recorded: bool,
    device: DeviceBuffer<f32>,
    // H2D overwrites must follow previously queued compute, including zeroing.
    predecessor: Option<CudaEvent>,
    canceled: bool,
}

struct TransportInner {
    transport: u64,
    config: CudaTransferConfig,
    context: Arc<CudaContext>,
    compute: Arc<CudaStream>,
    upload: Arc<CudaStream>,
    control: Arc<CudaStream>,
    tx: Vec<Slot>,
    rx: Vec<Slot>,
    stats: CudaAsyncTransportStats,
}

/// Created on the CUDA owner through `CudaOperators::new_async_transport`.
///
/// No CUDA objects cross owner threads. No pinned/device allocations are made
/// on submission; each submission creates its own immutable completion event.
/// Drop fences outstanding work, or leaks the entire resource bundle if it
/// cannot prove DMA quiescence. Explicit drain is preferred for error reporting.
pub struct CudaAsyncTransport {
    inner: Option<TransportInner>,
    _owner_local: PhantomData<Rc<()>>,
}

impl CudaAsyncTransport {
    pub(crate) fn new(
        config: CudaTransferConfig,
        context: Arc<CudaContext>,
        compute: Arc<CudaStream>,
        upload: Arc<CudaStream>,
        control: Arc<CudaStream>,
    ) -> Result<Self> {
        let (pinned_allocations, pinned_bytes) = config.layout()?;
        if context.is_capturing() {
            return Err(CudaTransferError::CaptureUnsupported);
        }
        for stream in [&compute, &upload, &control] {
            if !Arc::ptr_eq(stream.context(), &context) {
                return Err(CudaTransferError::InvalidConfig(
                    "foreign transport stream owner",
                ));
            }
        }
        let transport = next_transport_id(&NEXT_TRANSPORT_ID)?;
        let allocate = |count| -> Result<Vec<Slot>> {
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(count)
                .map_err(|_| CudaTransferError::InvalidConfig("cannot reserve slot metadata"))?;
            for _ in 0..count {
                slots.push(Slot {
                    pinned: cu(PinnedHostBuffer::zeroed(&context, config.chunk_elements))?,
                    generation: 0,
                    active: None,
                });
            }
            Ok(slots)
        };
        let tx = allocate(config.tx_slots)?;
        let rx = allocate(config.rx_slots)?;
        Ok(Self {
            inner: Some(TransportInner {
                transport,
                config,
                context,
                compute,
                upload,
                control,
                tx,
                rx,
                stats: CudaAsyncTransportStats {
                    tx_slots: config.tx_slots,
                    rx_slots: config.rx_slots,
                    pinned_allocations,
                    pinned_bytes,
                    ..Default::default()
                },
            }),
            _owner_local: PhantomData,
        })
    }

    fn inner(&self) -> &TransportInner {
        self.inner.as_ref().expect("live transport")
    }
    fn inner_mut(&mut self) -> &mut TransportInner {
        self.inner.as_mut().expect("live transport")
    }

    pub fn config(&self) -> CudaTransferConfig {
        self.inner().config
    }
    pub fn stats(&self) -> CudaAsyncTransportStats {
        self.inner().stats
    }

    /// Copies CPU input into an exclusive TX pinned slot before returning, then
    /// enqueues DMA. The caller's slice is not a DMA source and may be released.
    /// Previous compute on this owner is ordered before the overwrite. Call
    /// wait_h2d_on_compute before enqueuing a consumer (or poll until complete).
    pub fn submit_h2d_f32(
        &mut self,
        destination: &mut CudaF32Buffer,
        destination_offset: usize,
        values: &[f32],
    ) -> Result<CudaTransferTicket> {
        let inner = self.inner_mut();
        let direction = CudaTransferDirection::H2D;
        let slot = inner.preflight(direction, values.len())?;
        let device = destination.as_device_buffer();
        cu(device.check_context(&inner.context, "transport H2D destination"))?;
        let device = cu(device.slice(destination_offset, values.len()))?;
        let predecessor = cu(inner.context.new_event(false))?;
        let event = Arc::new(cu(inner.context.new_event(false))?);
        let id = inner.install(
            slot,
            direction,
            device,
            Arc::clone(&event),
            Some(predecessor),
        )?;
        // Install custody before the first driver command, including panic paths.
        let submitted = (|| -> ferrule_common::Result<()> {
            let entry = &mut inner.tx[slot];
            entry.pinned.as_mut_slice()[..values.len()].copy_from_slice(values);
            let active = entry.active.as_mut().expect("installed transfer");
            let predecessor = active.predecessor.as_ref().expect("H2D predecessor");
            cu(predecessor.record(&inner.compute))?;
            cu(inner.upload.wait(predecessor))?;
            // SAFETY: slot and allocation view stay exclusively in active custody
            // until a completion event or fallback stream fence proves quiescence.
            cu(unsafe {
                runtime::copy_host_to_device_pinned_range(
                    &inner.upload,
                    &active.device,
                    &entry.pinned,
                    0,
                )
            })?;
            cu(active.event.record(&inner.upload))?;
            active.recorded = true;
            Ok(())
        })();
        inner.finish_submission(direction, slot, submitted)?;
        Ok(CudaTransferTicket {
            id,
            event,
            _owner_local: PhantomData,
        })
    }

    /// Waits for the exact owner-local producer event, then enqueues D2H into a
    /// fixed RX slot. Do not overwrite the source range until completion. The
    /// source allocation itself may be dropped: this operation holds a view.
    pub fn submit_d2h_f32_after(
        &mut self,
        source: &CudaF32Buffer,
        source_offset: usize,
        elements: usize,
        producer: &CudaComputeEvent,
    ) -> Result<CudaTransferTicket> {
        let inner = self.inner_mut();
        let direction = CudaTransferDirection::D2H;
        let slot = inner.preflight(direction, elements)?;
        cu(producer
            .event()
            .check_context(&inner.context, "transport D2H producer"))?;
        let device = source.as_device_buffer();
        cu(device.check_context(&inner.context, "transport D2H source"))?;
        let device = cu(device.slice(source_offset, elements))?;
        let event = Arc::new(cu(inner.context.new_event(false))?);
        let id = inner.install(slot, direction, device, Arc::clone(&event), None)?;
        let submitted = (|| -> ferrule_common::Result<()> {
            let entry = &mut inner.rx[slot];
            let active = entry.active.as_mut().expect("installed transfer");
            cu(inner.control.wait(producer.event()))?;
            // SAFETY: no pinned reads or slot reuse until this exact DMA fences.
            cu(unsafe {
                runtime::copy_device_to_pinned_host_range(
                    &inner.control,
                    &mut entry.pinned,
                    0,
                    &active.device,
                )
            })?;
            cu(active.event.record(&inner.control))?;
            active.recorded = true;
            Ok(())
        })();
        inner.finish_submission(direction, slot, submitted)?;
        Ok(CudaTransferTicket {
            id,
            event,
            _owner_local: PhantomData,
        })
    }

    /// Orders the bound compute stream after this live H2D ticket. No host sync.
    /// Canceled, foreign, stale and quarantined tickets cannot publish a wait.
    pub fn wait_h2d_on_compute(&mut self, ticket: &CudaTransferTicket) -> Result<()> {
        let inner = self.inner_mut();
        let active = inner.validate_ticket(ticket)?;
        if ticket.id.direction != CudaTransferDirection::H2D {
            return Err(CudaTransferError::WrongDirection);
        }
        if active.canceled {
            return Err(CudaTransferError::Canceled);
        }
        inner.ensure_admissible()?;
        cu(active
            .event
            .check_context(inner.compute.context(), "transport compute wait"))?;
        if let Err(error) = cu(inner.compute.wait(&active.event)) {
            inner.quarantine();
            return Err(error.into());
        }
        Ok(())
    }

    /// Only queries the exact event; never synchronizes. Ready D2H data is
    /// materialized directly from pinned memory into the final boxed slice.
    pub fn poll(&mut self, ticket: &CudaTransferTicket) -> Result<Option<CudaTransferCompletion>> {
        let inner = self.inner_mut();
        let active = inner.validate_ticket(ticket)?;
        let ready = match cu(active.event.query()) {
            Ok(ready) => ready,
            Err(error) => {
                inner.quarantine();
                return Err(error.into());
            }
        };
        if !ready {
            return Ok(None);
        }
        Ok(Some(
            inner.complete(ticket.id.direction, ticket.id.slot as usize),
        ))
    }

    /// Intent only. Even a physically ready transfer occupies its slot until
    /// poll/drain consumes completion. Cancellation cannot undo DMA or kernels
    /// already enqueued by the caller, but never publishes a successful payload.
    pub fn cancel(&mut self, ticket: &CudaTransferTicket) -> Result<()> {
        let inner = self.inner_mut();
        inner.validate_ticket(ticket)?;
        inner.cancel_slot(ticket.id.direction, ticket.id.slot as usize);
        Ok(())
    }

    /// Cancels all active transfers, then attempts EVERY fence, even after an
    /// error. Successful entries return Canceled; failed entries retain custody
    /// and quarantine admission. Unrecorded submission failures require their
    /// exact submission stream's fallback fence, never an unrecorded event.
    pub fn drain(
        &mut self,
    ) -> std::result::Result<Vec<CudaTransferCompletion>, CudaTransferDrainError> {
        let inner = self.inner_mut();
        inner.cancel_all();
        let mut completed = Vec::new();
        let mut failure = None;
        for direction in [CudaTransferDirection::H2D, CudaTransferDirection::D2H] {
            for slot in 0..inner.slots(direction).len() {
                if inner.slots(direction)[slot].active.is_none() {
                    continue;
                }
                match inner.fence(direction, slot) {
                    Ok(()) => completed.push(inner.complete(direction, slot)),
                    Err(error) => {
                        inner.quarantine();
                        if failure.is_none() {
                            failure = Some(error);
                        }
                    }
                }
            }
        }
        match failure {
            Some(source) => Err(CudaTransferDrainError { completed, source }),
            None => Ok(completed),
        }
    }
}

impl TransportInner {
    fn slots(&self, direction: CudaTransferDirection) -> &[Slot] {
        match direction {
            CudaTransferDirection::H2D => &self.tx,
            CudaTransferDirection::D2H => &self.rx,
        }
    }
    fn slots_mut(&mut self, direction: CudaTransferDirection) -> &mut [Slot] {
        match direction {
            CudaTransferDirection::H2D => &mut self.tx,
            CudaTransferDirection::D2H => &mut self.rx,
        }
    }
    fn stream(&self, direction: CudaTransferDirection) -> &CudaStream {
        match direction {
            CudaTransferDirection::H2D => &self.upload,
            CudaTransferDirection::D2H => &self.control,
        }
    }
    fn ensure_admissible(&self) -> Result<()> {
        if self.stats.quarantined {
            return Err(CudaTransferError::Quarantined);
        }
        if self.context.is_capturing() {
            return Err(CudaTransferError::CaptureUnsupported);
        }
        Ok(())
    }
    fn preflight(&self, direction: CudaTransferDirection, elements: usize) -> Result<usize> {
        self.ensure_admissible()?;
        if elements == 0 || elements > self.config.chunk_elements {
            return Err(CudaTransferError::InvalidLength {
                elements,
                capacity: self.config.chunk_elements,
            });
        }
        let free = self
            .slots(direction)
            .iter()
            .position(|slot| slot.active.is_none())
            .ok_or(CudaTransferError::Backpressure(direction))?;
        next_generation(self.slots(direction)[free].generation)?;
        Ok(free)
    }
    fn install(
        &mut self,
        slot: usize,
        direction: CudaTransferDirection,
        device: DeviceBuffer<f32>,
        event: Arc<CudaEvent>,
        predecessor: Option<CudaEvent>,
    ) -> Result<CudaTransferId> {
        let id = CudaTransferId {
            transport: self.transport,
            direction,
            slot: slot as u32,
            generation: next_generation(self.slots(direction)[slot].generation)?,
        };
        let entry = &mut self.slots_mut(direction)[slot];
        entry.generation = id.generation;
        entry.active = Some(ActiveTransfer {
            id,
            event,
            recorded: false,
            device,
            predecessor,
            canceled: false,
        });
        self.stats.submissions = self.stats.submissions.saturating_add(1);
        self.update_in_use();
        Ok(id)
    }
    fn update_in_use(&mut self) {
        self.stats.tx_in_use = self.tx.iter().filter(|slot| slot.active.is_some()).count();
        self.stats.rx_in_use = self.rx.iter().filter(|slot| slot.active.is_some()).count();
        self.stats.tx_high_water = self.stats.tx_high_water.max(self.stats.tx_in_use);
        self.stats.rx_high_water = self.stats.rx_high_water.max(self.stats.rx_in_use);
        self.stats.device_holds_in_use = self.stats.tx_in_use + self.stats.rx_in_use;
        self.stats.device_holds_high_water = self
            .stats
            .device_holds_high_water
            .max(self.stats.device_holds_in_use);
    }
    fn finish_submission(
        &mut self,
        direction: CudaTransferDirection,
        slot: usize,
        result: ferrule_common::Result<()>,
    ) -> Result<()> {
        if let Err(source) = result {
            self.stats.failed_submissions = self.stats.failed_submissions.saturating_add(1);
            // A record failure must not query the newly created, unrecorded event.
            let cleanup = cu(self.stream(direction).synchronize());
            let cleanup =
                retire_after_fence(&mut self.slots_mut(direction)[slot].active, cleanup).err();
            if cleanup.is_some() {
                self.quarantine();
            }
            self.update_in_use();
            return Err(CudaTransferError::Submission { source, cleanup });
        }
        Ok(())
    }
    fn validate_ticket(&self, ticket: &CudaTransferTicket) -> Result<&ActiveTransfer> {
        if ticket.id.transport != self.transport {
            return Err(CudaTransferError::ForeignTicket);
        }
        let active = self
            .slots(ticket.id.direction)
            .get(ticket.id.slot as usize)
            .and_then(|slot| slot.active.as_ref())
            .ok_or(CudaTransferError::StaleTicket)?;
        if !matches_ticket(
            active.id,
            active.recorded,
            &active.event,
            ticket.id,
            &ticket.event,
        ) {
            return Err(CudaTransferError::StaleTicket);
        }
        Ok(active)
    }
    fn cancel_slot(&mut self, direction: CudaTransferDirection, slot: usize) {
        if let Some(active) = self.slots_mut(direction)[slot].active.as_mut()
            && !active.canceled
        {
            active.canceled = true;
            self.stats.cancel_intents = self.stats.cancel_intents.saturating_add(1);
        }
    }
    fn cancel_all(&mut self) {
        for direction in [CudaTransferDirection::H2D, CudaTransferDirection::D2H] {
            for slot in 0..self.slots(direction).len() {
                self.cancel_slot(direction, slot);
            }
        }
    }
    fn quarantine(&mut self) {
        self.stats.quarantined = true;
        self.cancel_all();
    }
    fn fence(&self, direction: CudaTransferDirection, slot: usize) -> Result<()> {
        if self.context.is_capturing() {
            return Err(CudaTransferError::CaptureUnsupported);
        }
        let active = self.slots(direction)[slot]
            .active
            .as_ref()
            .expect("active fence");
        cu(fence_recorded_or_stream(
            active.recorded,
            || active.event.synchronize(),
            || self.stream(direction).synchronize(),
        ))?;
        Ok(())
    }
    /// Called only after a query/fence has proved completion. Allocation remains
    /// owned while materializing the result, including allocation panic paths.
    fn complete(
        &mut self,
        direction: CudaTransferDirection,
        slot: usize,
    ) -> CudaTransferCompletion {
        let entry = &mut self.slots_mut(direction)[slot];
        let active = entry.active.as_ref().expect("completed active transfer");
        let id = active.id;
        let canceled = active.canceled;
        let result = if canceled {
            CudaTransferCompletion::Canceled { id }
        } else {
            match direction {
                CudaTransferDirection::H2D => CudaTransferCompletion::H2D { id },
                CudaTransferDirection::D2H => CudaTransferCompletion::D2H {
                    id,
                    values: Box::from(&entry.pinned.as_slice()[..active.device.len()]),
                },
            }
        };
        entry.active = None;
        self.stats.completions = self.stats.completions.saturating_add(1);
        if canceled {
            self.stats.canceled_completions = self.stats.canceled_completions.saturating_add(1);
        }
        self.update_in_use();
        result
    }
    fn fence_for_drop(&self) -> bool {
        let mut proven = true;
        for direction in [CudaTransferDirection::H2D, CudaTransferDirection::D2H] {
            for slot in 0..self.slots(direction).len() {
                if self.slots(direction)[slot].active.is_some()
                    && self.fence(direction, slot).is_err()
                {
                    proven = false;
                }
            }
        }
        proven
    }
}

impl Drop for CudaAsyncTransport {
    fn drop(&mut self) {
        // Keep the entire bundle under a leak guard even if a fence unwinds.
        if let Some(inner) = self.inner.take() {
            drop_after_proof(inner, TransportInner::fence_for_drop);
        }
    }
}

fn next_transport_id(counter: &AtomicU64) -> Result<u64> {
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| CudaTransferError::IdentityExhausted)
}
fn next_generation(generation: u64) -> Result<u64> {
    generation
        .checked_add(1)
        .ok_or(CudaTransferError::IdentityExhausted)
}
fn matches_ticket<E>(
    id: CudaTransferId,
    recorded: bool,
    event: &Arc<E>,
    candidate: CudaTransferId,
    candidate_event: &Arc<E>,
) -> bool {
    recorded && id == candidate && Arc::ptr_eq(event, candidate_event)
}
fn fence_recorded_or_stream<E>(
    recorded: bool,
    event: impl FnOnce() -> std::result::Result<(), E>,
    stream: impl FnOnce() -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    if recorded { event() } else { stream() }
}
fn retire_after_fence<T, E>(
    active: &mut Option<T>,
    proof: std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    proof?;
    *active = None;
    Ok(())
}
fn drop_after_proof<T>(bundle: T, prove: impl FnOnce(&T) -> bool) {
    let mut bundle = std::mem::ManuallyDrop::new(bundle);
    if prove(&bundle) {
        // SAFETY: this is the only drop path, after proof; false or unwinding
        // intentionally leaks ALL fields rather than running partial destructors.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut bundle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn config() -> CudaTransferConfig {
        CudaTransferConfig {
            chunk_elements: 8,
            tx_slots: 2,
            rx_slots: 3,
        }
    }
    fn id() -> CudaTransferId {
        CudaTransferId {
            transport: 1,
            direction: CudaTransferDirection::H2D,
            slot: 0,
            generation: 1,
        }
    }
    #[test]
    fn checked_fixed_budgets_and_id_exhaustion() {
        assert_eq!(config().layout().unwrap(), (5, 160));
        for bad in [
            CudaTransferConfig {
                chunk_elements: 0,
                ..config()
            },
            CudaTransferConfig {
                tx_slots: 0,
                ..config()
            },
            CudaTransferConfig {
                rx_slots: 0,
                ..config()
            },
            CudaTransferConfig {
                chunk_elements: usize::MAX,
                ..config()
            },
            CudaTransferConfig {
                tx_slots: usize::MAX,
                ..config()
            },
            CudaTransferConfig {
                chunk_elements: isize::MAX as usize / 4,
                ..config()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert_eq!(next_generation(0).unwrap(), 1);
        assert!(next_generation(u64::MAX).is_err());
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(next_transport_id(&counter).unwrap(), u64::MAX - 1);
        assert!(next_transport_id(&counter).is_err());
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }
    #[test]
    fn identity_requires_exact_event_generation_owner_direction_and_slot() {
        let event = Arc::new(());
        assert!(matches_ticket(id(), true, &event, id(), &event));
        assert!(!matches_ticket(id(), false, &event, id(), &event));
        assert!(!matches_ticket(id(), true, &event, id(), &Arc::new(())));
        for other in [
            CudaTransferId {
                transport: 2,
                ..id()
            },
            CudaTransferId {
                direction: CudaTransferDirection::D2H,
                ..id()
            },
            CudaTransferId { slot: 1, ..id() },
            CudaTransferId {
                generation: 2,
                ..id()
            },
        ] {
            assert!(!matches_ticket(id(), true, &event, other, &event));
        }
    }
    struct DropProbe(Rc<Cell<usize>>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    #[test]
    fn failed_record_and_failed_fallback_retain_custody_until_retry_proves_completion() {
        let drops = Rc::new(Cell::new(0));
        let mut active = Some(DropProbe(Rc::clone(&drops)));
        let proof = fence_recorded_or_stream(
            false,
            || panic!("unrecorded CUDA event must not prove completion"),
            || Err("injected stream fence failure"),
        );
        assert!(retire_after_fence(&mut active, proof).is_err());
        assert!(active.is_some());
        assert_eq!(drops.get(), 0);
        let proof =
            fence_recorded_or_stream(false, || panic!("unrecorded event"), || Ok::<_, ()>(()));
        retire_after_fence(&mut active, proof).unwrap();
        assert_eq!(drops.get(), 1);
    }
    #[test]
    fn recorded_event_error_does_not_fall_back_to_an_unrelated_fence() {
        let drops = Rc::new(Cell::new(0));
        let mut active = Some(DropProbe(Rc::clone(&drops)));
        let proof = fence_recorded_or_stream(
            true,
            || Err("injected event error"),
            || panic!("not the exact fence"),
        );
        assert!(retire_after_fence(&mut active, proof).is_err());
        assert_eq!(drops.get(), 0);
        retire_after_fence(&mut active, Ok::<_, ()>(())).unwrap();
        assert_eq!(drops.get(), 1);
    }
    #[test]
    fn drop_failure_or_unwind_retains_the_entire_bundle() {
        for unwind in [false, true] {
            let drops = Rc::new(Cell::new(0));
            let bundle = (
                DropProbe(Rc::clone(&drops)),
                DropProbe(Rc::clone(&drops)),
                DropProbe(Rc::clone(&drops)),
            );
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop_after_proof(bundle, |_| {
                    if unwind {
                        panic!("injected fence panic");
                    }
                    false
                });
            }));
            assert_eq!(drops.get(), 0);
        }
        let drops = Rc::new(Cell::new(0));
        drop_after_proof(
            (DropProbe(Rc::clone(&drops)), DropProbe(Rc::clone(&drops))),
            |_| true,
        );
        assert_eq!(drops.get(), 2);
    }
}
