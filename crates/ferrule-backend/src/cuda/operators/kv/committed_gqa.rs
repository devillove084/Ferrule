//! Bounded owner-local DMA of an isolated committed snapshot. Resident pages
//! remain resident: this slice neither evicts pages nor publishes append state.
use super::read_protocol::{Custody, error};
use super::{CudaCommittedRead, committed_gqa_credits};
use crate::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaPartial, StreamingF32GqaRequest,
    streaming_f32_gqa_partial,
};
use crate::cuda::operator_owner::{CudaComputeEvent, CudaF32Buffer, CudaOperators};
use crate::cuda::{
    CudaAsyncTransport, CudaTransferCompletion, CudaTransferConfig, CudaTransferTicket,
};
use ferrule_common::Result;
use ferrule_common::host_kv::CommittedKvSource;
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementReservation};
use std::rc::Rc;

/// Admission precedes physical pins, host metadata and device allocation.
/// Rejection before transfer consumes no physical custody.
pub struct HostPlacementAdmission {
    source: CommittedKvSource,
    config: StreamingF32GqaConfig,
    charge: Option<PlacementReservation>,
}
impl Drop for HostPlacementAdmission {
    fn drop(&mut self) {
        if let Some(charge) = self.charge.take() {
            charge.release();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommittedGqaProgress {
    Pending,
    Ready,
    CancelPending,
    Canceled,
}

/// Physical completion receipt, created only after the owner has proved both
/// DMA and consumer quiescence. No public constructor, Clone, or input ACK API.
///
/// ```compile_fail
/// use ferrule_backend::cuda::operators::kv::committed_gqa::CommittedGqaReceipt;
/// let forged = CommittedGqaReceipt { operation: 1, canceled: false };
/// ```
/// ```compile_fail
/// use ferrule_backend::cuda::operators::kv::committed_gqa::CommittedGqaReceipt;
/// fn replay(receipt: &CommittedGqaReceipt) -> CommittedGqaReceipt { (*receipt).clone() }
/// ```
pub struct CommittedGqaReceipt {
    operation: u64,
    canceled: bool,
    _pool: Rc<()>,
    source: CommittedKvSource,
    cpu_chunks: usize,
    gpu_chunks: usize,
}
impl CommittedGqaReceipt {
    pub fn operation(&self) -> u64 {
        self.operation
    }
    pub fn canceled(&self) -> bool {
        self.canceled
    }
    pub fn source(&self) -> &CommittedKvSource {
        &self.source
    }
    pub fn cpu_chunks(&self) -> usize {
        self.cpu_chunks
    }
    pub fn gpu_chunks(&self) -> usize {
        self.gpu_chunks
    }
}

/// Keeps the returned device output's allocation charge until the caller's
/// last compute consumer is fenced. Dropping without release retains custody.
pub struct CudaCommittedGqaOutput {
    buffer: Option<CudaF32Buffer>,
    receipt: CommittedGqaReceipt,
    charge: Option<PlacementReservation>,
    ops: Rc<CudaOperators>,
}
impl CudaCommittedGqaOutput {
    pub fn buffer(&self) -> Option<&CudaF32Buffer> {
        self.buffer.as_ref()
    }
    pub fn receipt(&self) -> &CommittedGqaReceipt {
        &self.receipt
    }
    pub fn release(&mut self) -> Result<()> {
        self.ops.sync_stream()?;
        self.buffer = None;
        if let Some(charge) = self.charge.take() {
            charge.release();
        }
        Ok(())
    }
}
impl Drop for CudaCommittedGqaOutput {
    fn drop(&mut self) {
        if self.charge.is_some() {
            std::mem::forget(self.buffer.take());
            std::mem::forget(self.charge.take());
            std::mem::forget(self.ops.clone());
        }
    }
}

struct HostSource {
    operation: u64,
    start: usize,
    tokens: usize,
    key: Box<[f32]>,
    value: Box<[f32]>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Allocate,
    Initialized,
    Query,
    Key,
    Value,
    HostReady,
    UploadKey,
    UploadValue,
    Consumer,
    Partial,
    Output,
    Ready,
    Canceled,
}
struct Resources {
    read: CudaCommittedRead,
    ops: Rc<CudaOperators>,
    query: CudaF32Buffer,
    config: StreamingF32GqaConfig,
    query_start: u64,
    cpu_end: usize,
    cursor: usize,
    end: usize,
    chunk_end: usize,
    stage: Stage,
    charge: Option<PlacementReservation>,
    transport: Option<CudaAsyncTransport>,
    buffers: Vec<CudaF32Buffer>, // K, V, partial, output; fixed grants
    ticket: Option<CudaTransferTicket>,
    consumer: Option<CudaComputeEvent>,
    producer: Option<CudaComputeEvent>,
    custody: Custody,
    query_host: Option<Box<[f32]>>,
    key: Option<Box<[f32]>>,
    host: Option<HostSource>,
    total: StreamingF32GqaPartial,
    cpu_chunks: usize,
    gpu_chunks: usize,
}

#[must_use = "poll or cancel and reap; drop is not completion"]
pub struct CudaHostPlacement {
    resources: Option<Box<Resources>>,
}
impl CudaHostPlacement {
    pub fn admit(
        source: &CommittedKvSource,
        config: StreamingF32GqaConfig,
        budget: &PlacementBudget,
    ) -> Result<HostPlacementAdmission> {
        source
            .validate_geometry()
            .map_err(|e| error(e.to_string()))?;
        let credits = committed_gqa_credits(config, source.pages.len())?;
        let charge = budget
            .reserve_resources(credits)
            .map_err(|e| error(e.to_string()))?;
        Ok(HostPlacementAdmission {
            source: source.clone(),
            config,
            charge: Some(charge),
        })
    }
    /// Q ownership moves here. It must already be RoPE-transformed. The caller
    /// keeps the logical read lease until finish/reap returns an opaque receipt.
    pub fn new(
        read: CudaCommittedRead,
        query: CudaF32Buffer,
        config: StreamingF32GqaConfig,
        query_start: u64,
        cpu_tokens: usize,
        mut admission: HostPlacementAdmission,
    ) -> Result<Self> {
        if read.source() != &admission.source || config != admission.config {
            return Err(error("foreign or stale placement admission"));
        }
        read.validate()?;
        let source = read.source();
        let start = source.positions.start;
        let end = source.positions.end;
        if query.len() != config.query_elements()?
            || cpu_tokens > end - start
            || query_start >= source.committed_tokens as u64
            || source.schema.planes()[0].elements_per_token != config.kv_elements(1)?
        {
            return Err(error("query, split, or committed KV geometry mismatch"));
        }
        let ops = read.context();
        query
            .as_device_buffer()
            .check_context(ops.stream_clone().context(), "committed Q")
            .map_err(|e| error(e.to_string()))?;
        let total = StreamingF32GqaPartial::empty(config, query_start)?;
        let charge = admission.charge.take().expect("admitted charge");
        Ok(Self {
            resources: Some(Box::new(Resources {
                read,
                ops,
                query,
                config,
                query_start,
                cpu_end: start + cpu_tokens,
                cursor: start,
                end,
                chunk_end: start,
                stage: Stage::Allocate,
                charge: Some(charge),
                transport: None,
                buffers: Vec::with_capacity(4),
                ticket: None,
                consumer: None,
                producer: None,
                custody: Custody::default(),
                query_host: None,
                key: None,
                host: None,
                total,
                cpu_chunks: 0,
                gpu_chunks: 0,
            })),
        })
    }
    pub fn is_quarantined(&self) -> bool {
        self.resources.as_ref().is_some_and(|r| r.custody.unknown())
    }
    pub fn poll(&mut self) -> Result<CommittedGqaProgress> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| error("placement already finished"))?;
        if r.custody.unknown() {
            return Err(error("placement quarantined; cancel/reap required"));
        }
        if r.custody.canceled() {
            return Ok(if r.stage == Stage::Canceled {
                CommittedGqaProgress::Canceled
            } else {
                CommittedGqaProgress::CancelPending
            });
        }
        let result = r.step();
        if result.is_err() {
            r.custody.quarantine();
        }
        result
    }
    /// Cancellation is intent only. It never frees operands or logical pins.
    pub fn cancel(&mut self) {
        if let Some(r) = self.resources.as_mut() {
            r.custody.cancel();
            if let (Some(transport), Some(ticket)) = (r.transport.as_mut(), r.ticket.as_ref())
                && transport.cancel(ticket).is_err()
            {
                r.custody.quarantine();
            }
        }
    }
    /// Explicit owner-local reaper. Unknown remains charged if either fence
    /// fails. Transport drain alone is insufficient, even if it returns no work.
    pub fn reap(&mut self) -> Result<()> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| error("placement already finished"))?;
        r.custody.cancel();
        let dma = match r.transport.as_mut() {
            Some(t) => t.drain().map(|_| ()).map_err(|e| error(e.to_string())),
            None => Ok(()),
        };
        // An unrecorded/failed consumer needs a stream fence, never a fresh event query.
        let compute = r.ops.sync_stream();
        r.custody.complete_reap(dma, compute)?;
        r.ticket = None;
        r.consumer = None;
        r.stage = Stage::Canceled;
        Ok(())
    }
    /// One-shot custody return. No caller-constructible completion is accepted.
    pub fn finish(&mut self) -> Result<CudaCommittedGqaOutput> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| error("placement already finished"))?;
        if !matches!(r.stage, Stage::Ready | Stage::Canceled)
            || !r.custody.quiescent()
            || (r.custody.canceled() && r.stage != Stage::Canceled)
        {
            return Err(error("DMA/consumer quiescence is not proven"));
        }
        r.read.release_after_quiescence()?;
        let mut r = self.resources.take().expect("validated owner");
        let canceled = r.stage == Stage::Canceled;
        let output = if canceled {
            None
        } else {
            Some(r.buffers.pop().expect("output"))
        };
        let receipt = CommittedGqaReceipt {
            operation: r.read.nonce(),
            canceled,
            _pool: r.read.identity(),
            source: r.read.source().clone(),
            cpu_chunks: r.cpu_chunks,
            gpu_chunks: r.gpu_chunks,
        };
        let charge = r.charge.take().expect("charged resources");
        let ops = r.ops.clone();
        drop(r);
        if canceled {
            charge.release();
            Ok(CudaCommittedGqaOutput {
                buffer: None,
                receipt,
                charge: None,
                ops,
            })
        } else {
            Ok(CudaCommittedGqaOutput {
                buffer: output,
                receipt,
                charge: Some(charge),
                ops,
            })
        }
    }
}
impl Drop for CudaHostPlacement {
    fn drop(&mut self) {
        // Unwinding and abandoning an owner preserve every operand and charge.
        // Recovery needs this object, so callers should keep it in their reaper.
        if let Some(r) = self.resources.take() {
            std::mem::forget(r);
        }
    }
}
impl Resources {
    fn partial_elements(&self) -> usize {
        self.config.q_heads * (self.config.head_dim + 2)
    }
    fn install_ticket(&mut self, ticket: CudaTransferTicket) {
        self.ticket = Some(ticket);
    }
    fn completed(&mut self) -> Result<Option<CudaTransferCompletion>> {
        let ticket = self
            .ticket
            .as_ref()
            .ok_or_else(|| error("missing DMA ticket"))?;
        let expected = ticket.id();
        let completion = self
            .transport
            .as_mut()
            .expect("transport")
            .poll(ticket)
            .map_err(|e| error(e.to_string()))?;
        if let Some(completion) = completion {
            let id = match &completion {
                CudaTransferCompletion::D2H { id, .. }
                | CudaTransferCompletion::H2D { id }
                | CudaTransferCompletion::Canceled { id } => *id,
            };
            if id != expected {
                return Err(error("transport completion identity mismatch"));
            }
            self.ticket = None;
            self.custody.dma_ack();
            Ok(Some(completion))
        } else {
            Ok(None)
        }
    }
    fn downloaded(&mut self) -> Result<Option<Box<[f32]>>> {
        match self.completed()? {
            Some(CudaTransferCompletion::D2H { values, .. }) => Ok(Some(values)),
            None => Ok(None),
            _ => Err(error("expected successful D2H")),
        }
    }
    fn uploaded(&mut self) -> Result<bool> {
        match self.completed()? {
            Some(CudaTransferCompletion::H2D { .. }) => Ok(true),
            None => Ok(false),
            _ => Err(error("expected successful H2D")),
        }
    }
    fn start_chunk(&mut self) -> Result<()> {
        self.read.validate()?;
        let page = self.read.source().schema.page_size();
        self.chunk_end = self
            .cursor
            .saturating_add(self.config.block_tokens)
            .min(self.end)
            .min(self.cursor.saturating_add(page - self.cursor % page));
        if self.cursor < self.cpu_end {
            self.chunk_end = self.chunk_end.min(self.cpu_end);
        }
        self.custody.submit_dma()?;
        let t = self.read.download(
            self.transport.as_mut().expect("transport"),
            0,
            self.cursor,
            self.chunk_end - self.cursor,
            self.producer.as_ref().expect("producer"),
        )?;
        self.install_ticket(t);
        self.stage = Stage::Key;
        Ok(())
    }
    fn advance(&mut self) -> Result<()> {
        self.host = None;
        self.cursor = self.chunk_end;
        if self.cursor != self.end {
            return self.start_chunk();
        }
        let output = self.total.output();
        self.custody.submit_dma()?;
        let t = self
            .transport
            .as_mut()
            .expect("transport")
            .submit_h2d_f32(&mut self.buffers[3], 0, &output)
            .map_err(|e| error(e.to_string()))?;
        self.install_ticket(t);
        self.stage = Stage::Output;
        Ok(())
    }
    fn upload(&mut self, plane: usize) -> Result<()> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| error("host source not installed"))?;
        if host.operation != self.read.nonce()
            || host.start != self.cursor
            || host.tokens != self.chunk_end - self.cursor
        {
            return Err(error("stale host source"));
        }
        let values = if plane == 0 { &host.key } else { &host.value };
        self.custody.submit_dma()?;
        let transport = self.transport.as_mut().expect("transport");
        let t = transport
            .submit_h2d_f32(&mut self.buffers[plane], 0, values)
            .map_err(|e| error(e.to_string()))?;
        // Keep the ticket in owner custody before publishing a compute wait.
        self.ticket = Some(t);
        transport
            .wait_h2d_on_compute(self.ticket.as_ref().expect("ticket"))
            .map_err(|e| error(e.to_string()))
    }
    fn step(&mut self) -> Result<CommittedGqaProgress> {
        match self.stage {
            Stage::Allocate => {
                let c = self.config;
                let chunk = c
                    .query_elements()?
                    .max(c.kv_elements(c.block_tokens)?)
                    .max(self.partial_elements());
                self.transport = Some(self.ops.new_async_transport(CudaTransferConfig {
                    chunk_elements: chunk,
                    tx_slots: 1,
                    rx_slots: 1,
                })?);
                self.custody.submit_consumer()?;
                for n in [
                    c.kv_elements(c.block_tokens)?,
                    c.kv_elements(c.block_tokens)?,
                    self.partial_elements(),
                    c.query_elements()?,
                ] {
                    self.buffers.push(self.ops.zero_f32_buffer(n)?);
                }
                self.consumer = Some(self.ops.record_compute_event()?);
                self.stage = Stage::Initialized;
            }
            Stage::Initialized => {
                if !self
                    .consumer
                    .as_ref()
                    .expect("initialization fence")
                    .is_complete()?
                {
                    return Ok(CommittedGqaProgress::Pending);
                }
                self.custody.consumer_ack();
                self.producer = self.consumer.take();
                self.custody.submit_dma()?;
                let t = self
                    .transport
                    .as_mut()
                    .expect("transport")
                    .submit_d2h_f32_after(
                        &self.query,
                        0,
                        self.query.len(),
                        self.producer.as_ref().expect("producer"),
                    )
                    .map_err(|e| error(e.to_string()))?;
                self.install_ticket(t);
                self.stage = Stage::Query;
            }
            Stage::Query => {
                if let Some(q) = self.downloaded()? {
                    self.query_host = Some(q);
                    self.start_chunk()?;
                }
            }
            Stage::Key => {
                if let Some(key) = self.downloaded()? {
                    self.key = Some(key);
                    self.custody.submit_dma()?;
                    let t = self.read.download(
                        self.transport.as_mut().expect("transport"),
                        1,
                        self.cursor,
                        self.chunk_end - self.cursor,
                        self.producer.as_ref().expect("producer"),
                    )?;
                    self.install_ticket(t);
                    self.stage = Stage::Value;
                }
            }
            Stage::Value => {
                if let Some(value) = self.downloaded()? {
                    let key = self.key.take().expect("completed key");
                    let tokens = self.chunk_end - self.cursor;
                    if key.len() != self.config.kv_elements(tokens)? || value.len() != key.len() {
                        return Err(error("D2H payload extent mismatch"));
                    }
                    self.host = Some(HostSource {
                        operation: self.read.nonce(),
                        start: self.cursor,
                        tokens,
                        key,
                        value,
                    });
                    self.stage = Stage::HostReady;
                }
            }
            Stage::HostReady => {
                if self.cursor < self.cpu_end {
                    let host = self.host.as_ref().expect("installed source");
                    let partial = streaming_f32_gqa_partial(StreamingF32GqaRequest {
                        config: self.config,
                        query: self.query_host.as_deref().expect("query"),
                        key: &host.key,
                        value: &host.value,
                        query_start: self.query_start,
                        kv_start: host.start as u64,
                    })?;
                    self.total.merge(&partial)?;
                    self.cpu_chunks += 1;
                    self.advance()?;
                } else {
                    self.upload(0)?;
                    self.stage = Stage::UploadKey;
                }
            }
            Stage::UploadKey => {
                if self.uploaded()? {
                    self.upload(1)?;
                    self.stage = Stage::UploadValue;
                }
            }
            Stage::UploadValue => {
                if self.uploaded()? {
                    self.custody.submit_consumer()?;
                    let (kv, rest) = self.buffers.split_at_mut(2);
                    self.ops.streaming_staged_partial(
                        self.config,
                        &self.query,
                        &kv[0],
                        &kv[1],
                        &mut rest[0],
                        self.chunk_end - self.cursor,
                        self.query_start,
                        self.cursor as u64,
                    )?;
                    self.consumer = Some(self.ops.record_compute_event()?);
                    self.stage = Stage::Consumer;
                }
            }
            Stage::Consumer => {
                if !self.consumer.as_ref().expect("consumer").is_complete()? {
                    return Ok(CommittedGqaProgress::Pending);
                }
                self.custody.consumer_ack();
                self.custody.submit_dma()?;
                let n = self.partial_elements();
                let t = self
                    .transport
                    .as_mut()
                    .expect("transport")
                    .submit_d2h_f32_after(
                        &self.buffers[2],
                        0,
                        n,
                        self.consumer.as_ref().expect("consumer"),
                    )
                    .map_err(|e| error(e.to_string()))?;
                self.install_ticket(t);
                self.stage = Stage::Partial;
            }
            Stage::Partial => {
                if let Some(state) = self.downloaded()? {
                    let partial = StreamingF32GqaPartial::from_state(
                        self.config,
                        self.query_start,
                        state.into_vec(),
                    )?;
                    self.total.merge(&partial)?;
                    self.gpu_chunks += 1;
                    self.consumer = None;
                    self.advance()?;
                }
            }
            Stage::Output => {
                if self.uploaded()? {
                    self.stage = Stage::Ready;
                }
            }
            Stage::Ready => return Ok(CommittedGqaProgress::Ready),
            Stage::Canceled => return Ok(CommittedGqaProgress::Canceled),
        }
        Ok(if self.stage == Stage::Ready {
            CommittedGqaProgress::Ready
        } else {
            CommittedGqaProgress::Pending
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{KvEndProgress, PagedKvPrepare};
    use crate::cuda::operators::kv::CudaPagedKvPool;
    use ferrule_common::execution::{
        ExecutionTransactionId, KvElementType, KvLayoutSchemaIdentity, KvPageId, KvPlaneDescriptor,
        StateSlot,
    };
    use ferrule_common::host_kv::placement::PlacementCredits;
    use std::time::{Duration, Instant};

    fn fixture() -> (CudaPagedKvPool, CommittedKvSource, StreamingF32GqaConfig) {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let planes = [
            KvPlaneDescriptor::new("standard_gqa.key", 2, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 2, 1, KvElementType::F32),
        ];
        let mut pool = CudaPagedKvPool::new(ops.clone(), &planes, 4, 2).unwrap();
        let tx = pool
            .prepare(PagedKvPrepare {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                new_pages: &[KvPageId(1)],
                writable_pages: &[],
                cow_replacements: &[],
                protected_pages: &[],
            })
            .unwrap();
        ops.sync_stream().unwrap();
        assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
        pool.bind_committed_logical_owner(7).unwrap();
        let source = CommittedKvSource {
            owner: 7,
            state_slot: StateSlot::new(0),
            generation: 1,
            incarnation: 1,
            schema: KvLayoutSchemaIdentity::new(planes.to_vec(), 4, 8),
            committed_tokens: 3,
            positions: 0..3,
            pages: vec![KvPageId(1)],
        };
        let config = StreamingF32GqaConfig {
            rows: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            block_tokens: 2,
            softmax_scale: 0.5,
            sliding_window: None,
        };
        (pool, source, config)
    }

    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn every_submission_stage_requires_cancel_reap_then_finish() {
        let (mut pool, source, config) = fixture();
        let credits = committed_gqa_credits(config, source.pages.len()).unwrap();
        let budget = PlacementBudget::new(credits);
        for stage in [
            Stage::Allocate,
            Stage::Initialized,
            Stage::Query,
            Stage::Key,
            Stage::Value,
            Stage::HostReady,
            Stage::UploadKey,
            Stage::UploadValue,
            Stage::Consumer,
            Stage::Partial,
            Stage::Output,
            Stage::Ready,
        ] {
            let admission = CudaHostPlacement::admit(&source, config, &budget).unwrap();
            let read = pool.acquire_committed_read(&source, 0).unwrap();
            let query = read.context().upload_f32_buffer(&[1.0; 4]).unwrap();
            let mut job = CudaHostPlacement::new(read, query, config, 2, 0, admission).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while job.resources.as_ref().unwrap().stage != stage {
                job.poll().unwrap();
                assert!(
                    Instant::now() < deadline,
                    "target submission stage timed out"
                );
                std::thread::yield_now();
            }
            job.cancel();
            job.cancel();
            assert_eq!(job.poll().unwrap(), CommittedGqaProgress::CancelPending);
            assert!(job.finish().is_err());
            assert!(pool.clone().release(&source.pages).is_err());
            assert!(CudaHostPlacement::admit(&source, config, &budget.clone()).is_err());
            assert_eq!(budget.in_use(), credits);
            job.reap().unwrap();
            assert_eq!(job.poll().unwrap(), CommittedGqaProgress::Canceled);
            assert!(pool.clone().release(&source.pages).is_err());
            assert_eq!(budget.in_use(), credits);
            job.reap().unwrap();
            let output = job.finish().unwrap();
            assert!(output.receipt().canceled());
            assert_eq!(output.receipt().source(), &source);
            assert!(output.buffer().is_none());
            assert_eq!(budget.in_use(), PlacementCredits::default());
            assert!(job.finish().is_err());
            assert!(job.reap().is_err());
            assert!(job.poll().is_err());
        }
        pool.release(&source.pages).unwrap();
        pool.shutdown().unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn unknown_with_submitted_consumer_stays_pinned_until_explicit_reap() {
        let (mut pool, source, config) = fixture();
        let credits = committed_gqa_credits(config, source.pages.len()).unwrap();
        let budget = PlacementBudget::new(credits);
        let admission = CudaHostPlacement::admit(&source, config, &budget).unwrap();
        let read = pool.acquire_committed_read(&source, 0).unwrap();
        let query = read.context().upload_f32_buffer(&[1.0; 4]).unwrap();
        let mut job = CudaHostPlacement::new(read, query, config, 2, 0, admission).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while job.resources.as_ref().unwrap().stage != Stage::Consumer {
            job.poll().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        // Fault injection changes only bookkeeping, never the GPU/context.
        let r = job.resources.as_mut().unwrap();
        r.custody.quarantine();
        r.custody.dma_ack();
        r.custody.consumer_ack();
        for _ in 0..2 {
            job.cancel();
            assert!(job.poll().is_err());
            assert!(job.finish().is_err());
            assert!(job.is_quarantined());
            assert!(pool.clone().release(&source.pages).is_err());
            assert!(pool.clone().shutdown().is_err());
            assert_eq!(budget.in_use(), credits);
        }
        job.reap().unwrap();
        assert!(!job.is_quarantined());
        assert_eq!(budget.in_use(), credits);
        assert!(pool.clone().release(&source.pages).is_err());
        assert!(job.finish().unwrap().receipt().canceled());
        assert_eq!(budget.in_use(), PlacementCredits::default());
        pool.release(&source.pages).unwrap();
        pool.shutdown().unwrap();
    }
}
