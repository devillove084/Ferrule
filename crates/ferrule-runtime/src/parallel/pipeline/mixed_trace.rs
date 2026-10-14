//! Native trace transport. No device ownership or disk I/O crosses the owner callback.
use ferrule_common::{Error, Result};
use ferrule_model::transformer::CudaDiagnosticLimits;
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel},
    },
    time::{Duration, Instant},
};
const META: usize = 256 << 10;
const DEFAULT_INDEX: usize = 64 << 20;
const WRITER_STACK: usize = 2 << 20;
const METADATA_ITEMS: usize = 1024;
// Fixed owner metadata allowance: bounded descriptor/routes/batch scalars plus
// four capped Debug strings, JSON Value nodes and the META serialization sink.
const METADATA_HOST: usize = 2 << 20;
fn error(message: impl ToString) -> Error {
    Error::Execution {
        message: message.to_string(),
    }
}
#[derive(Default, Debug)]
struct Status {
    failed: AtomicBool,
    done: AtomicBool,
    active: AtomicBool,
    invocations: AtomicUsize,
    events: AtomicUsize,
    bytes: AtomicUsize,
    pending: AtomicUsize,
    pending_bytes: AtomicUsize,
    peak_pending_bytes: AtomicUsize,
    failure: OnceLock<CaptureFailure>,
}
#[derive(Debug, serde::Serialize)]
struct CaptureFailure {
    reason: &'static str,
    event: String,
    attempted: usize,
    limit: usize,
    pending_records: usize,
    pending_bytes: usize,
}
impl Status {
    fn fail(&self, reason: &'static str, event: &str, attempted: usize, limit: usize) {
        let failure = CaptureFailure {
            reason,
            event: event.chars().take(512).collect(),
            attempted,
            limit,
            pending_records: self.pending.load(Ordering::Acquire),
            pending_bytes: self.pending_bytes.load(Ordering::Acquire),
        };
        if self.failure.set(failure).is_ok() {
            tracing::warn!(failure=?self.failure.get(), "native capture incomplete");
        }
        self.failed.store(true, Ordering::Release);
    }
}
type WriterStatuses = Arc<Mutex<Vec<(u32, Arc<Status>)>>>;
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct MixedTraceConfig {
    directory: PathBuf,
    pub max_bytes_per_owner: usize,
    pub max_tensor_bytes: usize,
    pub max_events_per_owner: usize,
    pub max_invocations: usize,
    pub max_metadata_bytes_per_owner: usize,
    admitted_pages: Option<usize>,
    admitted_host: Option<u64>,
    writers: WriterStatuses,
    abort: Arc<AtomicBool>,
}
impl MixedTraceConfig {
    pub fn new(
        directory: impl Into<PathBuf>,
        max_bytes_per_owner: usize,
        max_tensor_bytes: usize,
        max_events_per_owner: usize,
        max_invocations: usize,
    ) -> Result<Self> {
        let c = Self {
            directory: directory.into(),
            max_bytes_per_owner,
            max_tensor_bytes,
            max_events_per_owner,
            max_invocations,
            max_metadata_bytes_per_owner: DEFAULT_INDEX,
            admitted_pages: None,
            admitted_host: None,
            writers: Arc::default(),
            abort: Arc::default(),
        };
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        if !self.directory.is_absolute()
            || !(4..=8 << 20).contains(&self.max_tensor_bytes)
            || self.max_bytes_per_owner < self.max_tensor_bytes
            || self.max_bytes_per_owner > 2usize << 30
            || !(1..=100_000).contains(&self.max_events_per_owner)
            || !(1..=64).contains(&self.max_invocations)
            || !(META..=DEFAULT_INDEX).contains(&self.max_metadata_bytes_per_owner)
        {
            return Err(error("invalid native trace bounds"));
        }
        Ok(())
    }
    pub fn limits(&self) -> Result<CudaDiagnosticLimits> {
        self.validate()?;
        CudaDiagnosticLimits::new(self.max_tensor_bytes)
    }
    pub fn device_reserve(&self) -> Result<u64> {
        Ok(self.limits()?.device_scratch_grant_bytes()? as u64)
    }
    pub fn host_reserve(&self, owners: usize) -> Result<u64> {
        self.host_reserve_with_pages(owners, 0)
    }
    pub fn host_reserve_with_pages(&self, owners: usize, pages: usize) -> Result<u64> {
        self.validate()?;
        // Queue + disk writer in-flight, native D2H/F32 view + byte serialization,
        // one descriptor/serialization budget, six source/lease page copies + export map.
        let page_bytes = pages
            .checked_mul(
                6 * std::mem::size_of::<ferrule_common::execution::KvPageId>()
                    + std::mem::size_of::<(ferrule_common::execution::KvPageId, u32, u64)>(),
            )
            .ok_or_else(|| error("trace page budget overflow"))?;
        // Queue every accepted tensor and control record, even if the writer
        // never runs. Both buffers are exact boxed slices, not cap-sized Vecs.
        // Channel slots: Record + sequence stamp/alignment; per-record allowance
        // also covers the two host allocation headers. No per-layer multiplier.
        let records = self.queue_record_limit()?;
        let record_bytes = records
            .checked_mul(std::mem::size_of::<Record>() + 6 * std::mem::size_of::<usize>())
            .ok_or_else(|| error("trace record budget overflow"))?;
        let bytes = self
            .max_bytes_per_owner
            .checked_add(self.max_metadata_bytes_per_owner)
            .and_then(|v| v.checked_add(record_bytes))
            .and_then(|v| v.checked_add(self.max_tensor_bytes))
            .and_then(|v| v.checked_add(self.limits().ok()?.host_tensor_bytes().ok()?))
            .and_then(|v| v.checked_add(METADATA_HOST + WRITER_STACK))
            .and_then(|v| v.checked_add(page_bytes))
            .and_then(|v| v.checked_mul(owners))
            .ok_or_else(|| error("trace host budget overflow"))?;
        Ok(bytes as u64)
    }
    /// One binding, and begin/execution_end/committed_kv_begin per invocation.
    /// Tensor events include committed KV chunks in the same total event cap.
    pub fn queue_record_limit(&self) -> Result<usize> {
        self.validate()?;
        self.max_invocations
            .checked_mul(3)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_add(self.max_events_per_owner))
            .ok_or_else(|| error("trace record count overflow"))
    }
    /// Host allowance must be explicitly charged by the caller before direct preparation.
    pub fn admit_host(mut self, per_owner_capacity: u64, pages: usize) -> Result<Self> {
        if self.host_reserve_with_pages(1, pages)? > per_owner_capacity {
            return Err(error("insufficient trace host budget"));
        }
        self.admitted_pages = Some(pages);
        self.admitted_host = Some(per_owner_capacity);
        Ok(self)
    }
    pub fn validate_admission(
        &self,
        workspace: usize,
        base_workspace: usize,
        pages: usize,
    ) -> Result<()> {
        if self.admitted_pages.is_none_or(|p| p < pages)
            || self.admitted_host.is_none_or(|h| {
                self.host_reserve_with_pages(1, pages)
                    .map_or(true, |required| required > h)
            })
            || base_workspace
                .checked_add(self.device_reserve()? as usize)
                .is_none_or(|n| n > workspace)
        {
            return Err(error(
                "native trace requires admitted host and device scratch budgets",
            ));
        }
        Ok(())
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    /// Host evidence cancellation only. In particular this does not acknowledge
    /// device completion, close worker custody or release a committed-read lease.
    pub fn abort_capture(&self) {
        self.abort.store(true, Ordering::Release);
    }
    /// Parent-side bounded wait only. Owners never join or wait for filesystem progress.
    pub fn wait_for_writers(&self, timeout: Duration) -> Result<()> {
        let states = self
            .writers
            .lock()
            .map_err(error)?
            .iter()
            .map(|(_, s)| s.clone())
            .collect::<Vec<_>>();
        if states.is_empty() {
            return Err(error("trace writer absent"));
        }
        let deadline = Instant::now() + timeout;
        while states.iter().any(|s| !s.done.load(Ordering::Acquire)) {
            if Instant::now() >= deadline {
                self.abort_capture();
                return Err(error("trace writer close deadline; evidence incomplete"));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if states.iter().any(|s| s.failed.load(Ordering::Acquire)) {
            return Err(error("trace capture incomplete"));
        }
        Ok(())
    }
}
pub(super) fn bounded_debug(value: &impl std::fmt::Debug) -> Result<String> {
    struct Text(String);
    impl std::fmt::Write for Text {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            if text.len() > (META / 8).saturating_sub(self.0.len()) {
                return Err(std::fmt::Error);
            }
            self.0.push_str(text);
            Ok(())
        }
    }
    let mut text = Text(String::with_capacity(META / 8));
    std::fmt::write(&mut text, format_args!("{value:?}")).map_err(error)?;
    Ok(text.0)
}
struct Record {
    metadata: Box<[u8]>,
    data: Box<[u8]>,
    status: Arc<Status>,
}
impl Drop for Record {
    fn drop(&mut self) {
        self.status.pending.fetch_sub(1, Ordering::AcqRel);
        self.status
            .pending_bytes
            .fetch_sub(self.metadata.len() + self.data.len(), Ordering::AcqRel);
    }
}
// Serialize into a bounded sink: an oversized record must not first allocate an
// unbounded serialized copy. The metadata Value itself is owner-local/transient.
struct Metadata(Vec<u8>);
impl Write for Metadata {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > META.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("trace metadata cap"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn write_stream(config: MixedTraceConfig, owner: u32, rx: Receiver<Record>, status: Arc<Status>) {
    let root = config.directory.join(format!("rank-{owner}"));
    let fresh = |name: &str| {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(name))
    };
    let result = (|| -> Result<()> {
        std::fs::create_dir(&root)?;
        let mut events = fresh("events.jsonl")?;
        let mut tensors = fresh("tensors.f32le")?;
        loop {
            if config.abort.load(Ordering::Acquire) {
                status.fail("parent_abort", "writer", 0, 0);
                break;
            }
            match rx.recv_timeout(Duration::from_millis(25)) {
                Ok(record) => {
                    events.write_all(&record.metadata)?;
                    tensors.write_all(&record.data)?;
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        events.sync_all()?;
        tensors.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        status.fail("writer_io", &e.to_string(), 0, 0);
    }
    // Never wait for a retained Unknown sender; discard queued host evidence only.
    drop(rx);
    if config.abort.load(Ordering::Acquire) {
        status.fail("parent_abort", "writer", 0, 0);
    }
    let receipt = (|| -> Result<()> {
        let mut file = fresh("capture-status.json")?;
        serde_json::to_writer_pretty(&mut file, &json!({
            "schema":"ferrule.mixed-owner-trace.v2","invocations":status.invocations.load(Ordering::Acquire),
            "events":status.events.load(Ordering::Acquire),"tensor_bytes":status.bytes.load(Ordering::Acquire),
            "complete_available_trace":!status.failed.load(Ordering::Acquire)&&!status.active.load(Ordering::Acquire),
            "capture_error":status.failure.get().map(|f|f.reason),"capture_failure":status.failure.get(),
            "peak_pending_bytes":status.peak_pending_bytes.load(Ordering::Acquire),
            "required_inventory_complete":false,"committed":false,"excluded_events":["gdn.chunk.workspace"]
        })).map_err(error)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = receipt {
        status.fail("writer_status_io", &e.to_string(), 0, 0);
    }
    status.done.store(true, Ordering::Release);
}
pub(super) struct MixedTraceWriter {
    config: MixedTraceConfig,
    tx: Option<SyncSender<Record>>,
    status: Arc<Status>,
    invocation: usize,
    transaction: Option<u64>,
    count: usize,
    bytes: usize,
    index_bytes: usize,
    records: usize,
    active: bool,
}
impl MixedTraceWriter {
    pub(super) fn new(config: MixedTraceConfig, owner: u32) -> Result<Self> {
        config.validate()?;
        let (tx, rx) = sync_channel(config.queue_record_limit()?);
        let status = Arc::new(Status::default());
        {
            let mut writers = config.writers.lock().map_err(error)?;
            if writers.iter().any(|(id, _)| *id == owner) {
                return Err(error(
                    "trace owner already registered; fresh capture required",
                ));
            }
            writers.push((owner, status.clone()));
        }
        let c = config.clone();
        let s = status.clone();
        std::thread::Builder::new()
            .name(format!("mixed-evidence-{owner}"))
            .stack_size(WRITER_STACK)
            .spawn(move || write_stream(c, owner, rx, s))?;
        Ok(Self {
            config,
            tx: Some(tx),
            status,
            invocation: 0,
            transaction: None,
            count: 0,
            bytes: 0,
            index_bytes: 0,
            records: 0,
            active: false,
        })
    }
    fn fail(&self, reason: &'static str, event: &str, attempted: usize, limit: usize) {
        self.status.fail(reason, event, attempted, limit);
    }
    pub(super) fn failed(&self) -> bool {
        self.config.abort.load(Ordering::Acquire) || self.status.failed.load(Ordering::Acquire)
    }
    fn enqueue(&mut self, metadata: Value, data: Vec<u8>) {
        if self.failed() {
            return;
        }
        let name = metadata["event"]
            .as_str()
            .or_else(|| metadata["kind"].as_str())
            .unwrap_or("metadata");
        if self.records
            >= self
                .config
                .queue_record_limit()
                .expect("validated trace config")
        {
            self.fail(
                "record_count",
                name,
                self.records + 1,
                self.config.queue_record_limit().unwrap(),
            );
            return;
        }
        let mut encoded = Metadata(Vec::with_capacity(META));
        if serde_json::to_writer(&mut encoded, &metadata).is_err()
            || encoded.write_all(b"\n").is_err()
        {
            self.fail("record_metadata_bytes", name, META + 1, META);
            return;
        }
        let encoded = encoded.0.into_boxed_slice();
        if self.index_bytes + encoded.len() > self.config.max_metadata_bytes_per_owner {
            self.fail(
                "total_metadata_bytes",
                name,
                self.index_bytes + encoded.len(),
                self.config.max_metadata_bytes_per_owner,
            );
            return;
        }
        self.index_bytes += encoded.len();
        self.records += 1;
        let data = data.into_boxed_slice();
        self.status.pending.fetch_add(1, Ordering::AcqRel);
        let n = encoded.len() + data.len();
        let pending = self.status.pending_bytes.fetch_add(n, Ordering::AcqRel) + n;
        self.status
            .peak_pending_bytes
            .fetch_max(pending, Ordering::AcqRel);
        let record = Record {
            metadata: encoded,
            data,
            status: self.status.clone(),
        };
        match self.tx.as_ref().map(|tx| tx.try_send(record)) {
            Some(Ok(())) => {}
            Some(Err(TrySendError::Full(_))) => self.fail(
                "queue_capacity_invariant",
                name,
                self.records,
                self.config.queue_record_limit().unwrap(),
            ),
            Some(Err(TrySendError::Disconnected(_))) | None => {
                self.fail("writer_disconnected", name, self.records, 0)
            }
        }
    }
    pub(super) fn seal(&mut self, binding: Value) -> Result<()> {
        self.enqueue(json!({"kind":"owner_binding","binding":binding}), vec![]);
        Ok(())
    }
    pub(super) fn begin(&mut self, transaction: u64, identity: Value) {
        if self.active || self.invocation >= self.config.max_invocations {
            self.fail(
                "invocation_count_or_active",
                "begin",
                self.invocation + 1,
                self.config.max_invocations,
            );
            self.active = false;
            return;
        }
        self.invocation += 1;
        self.transaction = Some(transaction);
        self.active = true;
        self.status.active.store(true, Ordering::Release);
        self.status
            .invocations
            .store(self.invocation, Ordering::Release);
        self.enqueue(json!({"kind":"begin","invocation":self.invocation,"identity":identity,"committed":false}),vec![]);
    }
    pub(super) fn accepts_metadata(&self, items: usize, text_bytes: usize) -> bool {
        if self.failed() || items > METADATA_ITEMS || text_bytes > 4096 {
            self.fail(
                "metadata_geometry",
                "descriptor",
                items.max(text_bytes),
                if items > METADATA_ITEMS {
                    METADATA_ITEMS
                } else {
                    4096
                },
            );
            return false;
        }
        true
    }
    pub(super) fn accepts_descriptor(
        &mut self,
        d: &ferrule_model::transformer::CudaDiagnosticDescriptor,
    ) -> bool {
        use ferrule_model::transformer::hybrid_parallel::HybridLocalAttention;
        let items = d
            .layout
            .dimensions
            .len()
            .saturating_add(d.layout.strides.len())
            .saturating_add(
                d.sequence
                    .as_ref()
                    .map_or(0, |s| s.packed_rows.len().saturating_add(s.positions.len())),
            )
            .saturating_add(
                d.parameter_partition
                    .as_ref()
                    .map_or(0, |p| p.rows.len().saturating_mul(2)),
            )
            .saturating_add(match &d.attention_partition {
                Some(HybridLocalAttention::Gqa { kv_owners, .. }) => kv_owners.len(),
                _ => 0,
            });
        self.accepts(d.name, d.layout.exported_f32_bytes())
            && self.accepts_metadata(items, d.name.len())
    }
    /// Invoked by the MODEL pre-allocation filter, not after readback.
    pub(super) fn accepts(&mut self, name: &str, f32_bytes: usize) -> bool {
        if name == "gdn.chunk.workspace" || self.failed() {
            return false;
        }
        let rejection = if !self.active || self.tx.is_none() {
            Some(("inactive_or_closed", 0, 0))
        } else if f32_bytes > self.config.max_tensor_bytes {
            Some(("tensor_bytes", f32_bytes, self.config.max_tensor_bytes))
        } else if f32_bytes > self.config.max_bytes_per_owner.saturating_sub(self.bytes) {
            Some((
                "total_tensor_bytes",
                self.bytes.saturating_add(f32_bytes),
                self.config.max_bytes_per_owner,
            ))
        } else if self.count >= self.config.max_events_per_owner {
            Some((
                "tensor_event_count",
                self.count + 1,
                self.config.max_events_per_owner,
            ))
        } else {
            None
        };
        if let Some((reason, attempted, limit)) = rejection {
            self.fail(reason, name, attempted, limit);
            return false;
        }
        true
    }
    pub(super) fn record(&mut self, mut metadata: Value, data: Vec<u8>) {
        if self.failed() {
            return;
        }
        if !self.accepts(metadata["event"].as_str().unwrap_or(""), data.len()) {
            return;
        }
        metadata["kind"] = json!("tensor");
        metadata["invocation"] = json!(self.invocation);
        metadata["order"] = json!(self.count);
        metadata["offset"] = json!(self.bytes);
        metadata["bytes"] = json!(data.len());
        metadata["storage"] = json!("f32le-native-host-view");
        self.bytes += data.len();
        self.count += 1;
        self.status.events.store(self.count, Ordering::Release);
        self.status.bytes.store(self.bytes, Ordering::Release);
        self.enqueue(metadata, data);
    }
    pub(super) fn finish(&mut self, ok: bool) {
        self.active = false;
        self.status.active.store(false, Ordering::Release);
        self.enqueue(json!({"kind":"execution_end","invocation":self.invocation,"execution_ok":ok,"committed":false}),vec![]);
    }
    pub(super) fn committed_begin(&mut self, transaction: u64, identity: Value) -> Result<()> {
        if self.failed() || self.active || self.transaction != Some(transaction) {
            return Err(error(
                "committed trace transaction differs from captured invocation",
            ));
        }
        self.active = true;
        self.status.active.store(true, Ordering::Release);
        self.enqueue(json!({"kind":"committed_kv_begin","invocation":self.invocation,"identity":identity,"committed":true}),vec![]);
        Ok(())
    }
    pub(super) fn committed_end(&mut self) {
        self.active = false;
        self.status.active.store(false, Ordering::Release);
    }
    pub(super) fn close(&mut self) {
        self.tx.take();
    }
    pub(super) fn config(&self) -> &MixedTraceConfig {
        &self.config
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn paused(c: MixedTraceConfig) -> (MixedTraceWriter, Receiver<Record>) {
        let (tx, rx) = sync_channel(c.queue_record_limit().unwrap());
        (
            MixedTraceWriter {
                config: c,
                tx: Some(tx),
                status: Arc::default(),
                invocation: 0,
                transaction: None,
                count: 0,
                bytes: 0,
                index_bytes: 0,
                records: 0,
                active: false,
            },
            rx,
        )
    }
    fn descriptor(bytes: usize) -> ferrule_model::transformer::CudaDiagnosticDescriptor {
        use ferrule_model::transformer::{
            CudaDiagnosticDescriptor, CudaDiagnosticDtype, CudaDiagnosticLayout,
        };
        CudaDiagnosticDescriptor {
            name: "bf16",
            layer: Some(0),
            layout: CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::Bf16, vec![1, bytes / 4])
                .unwrap(),
            sequence: None,
            parameter_partition: None,
            attention_partition: None,
        }
    }
    #[test]
    fn retained_sender_abort_finishes_incomplete_without_releasing_source_pin() {
        use crate::{KvPageManager, KvReservationCommit};
        use ferrule_common::execution::{KvElementType, StateSlot};
        use ferrule_model::decoder::StandardGqaPlanes;
        let schema = StandardGqaPlanes::new(1, 1, 1, 4, 16, KvElementType::Bf16).unwrap();
        let mut manager = KvPageManager::new(Box::new(schema), 4);
        let slot = StateSlot::new(0);
        manager.alloc_sequence(slot, 0).unwrap();
        let reservation = manager.reserve(slot, 0, 1).unwrap();
        let prepared = manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, 1)])
            .unwrap();
        let retired = manager.publish_commit(prepared);
        manager.confirm_page_retirement(retired).unwrap();
        let source = manager.committed_kv_source(slot, 0, 0..1).unwrap();
        let lease = manager.acquire_committed_read(&source).unwrap();
        let d = tempfile::tempdir().unwrap();
        let c = MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap();
        let mut writer = MixedTraceWriter::new(c.clone(), 0).unwrap();
        writer.begin(1, json!({"transaction":1}));
        // Match Unknown custody: the worker/sender and logical reader stay retained.
        c.abort_capture();
        assert!(c.wait_for_writers(Duration::from_secs(2)).is_err());
        assert!(writer.tx.is_some());
        assert!(writer.status.done.load(Ordering::Acquire));
        let status: Value = serde_json::from_slice(
            &std::fs::read(d.path().join("rank-0/capture-status.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(status["complete_available_trace"], false);
        manager.validate_committed_read(&lease).unwrap();
        drop(lease); // Unknown must not fabricate a physical-completion ACK.
        assert!(manager.free_sequence_pages(slot).is_err());
        assert_eq!(manager.allocated_pages(), 1);
    }
    #[test]
    fn older_publication_after_cancelled_execution_never_uses_latest_invocation() {
        let d = tempfile::tempdir().unwrap();
        let (mut t, rx) = paused(MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap());
        t.begin(11, json!({"transaction":11}));
        t.finish(true);
        while rx.try_recv().is_ok() {}
        t.committed_begin(11, json!({"transaction":11})).unwrap();
        t.committed_end();
        while rx.try_recv().is_ok() {}
        t.begin(12, json!({"transaction":12}));
        t.finish(false); // B executed, then rolled back/cancelled; A is still published.
        while rx.try_recv().is_ok() {}
        assert!(t.committed_begin(11, json!({"transaction":11})).is_err());
        assert_eq!(t.invocation, 2);
        assert!(!t.active);
        assert!(
            rx.try_recv().is_err(),
            "no incorrectly labelled committed record"
        );
    }
    #[test]
    fn rejected_large_no_materialization() {
        let d = tempfile::tempdir().unwrap();
        let (mut t, _rx) =
            paused(MixedTraceConfig::new(d.path(), 32 << 20, 8 << 20, 10, 4).unwrap());
        t.begin(1, json!({}));
        let mut allocated = 0;
        if t.accepts_descriptor(&descriptor((8 << 20) + 4)) {
            allocated += 1;
        }
        assert_eq!(allocated, 0);
        assert!(t.failed());
    }
    #[test]
    fn fifth_call_no_materialization() {
        let d = tempfile::tempdir().unwrap();
        let (mut t, rx) = paused(MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap());
        for _ in 0..4 {
            t.begin(1, json!({}));
            t.finish(true);
            while rx.try_recv().is_ok() {}
        }
        t.begin(1, json!({}));
        assert!(!t.accepts_descriptor(&descriptor(4)));
        assert_eq!(t.invocation, 4);
    }
    #[test]
    fn direct_prepare_rejects_insufficient_budget() {
        let d = tempfile::tempdir().unwrap();
        let c = MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap();
        assert!(c.clone().admit_host(1, 4).is_err());
        assert!(c.validate_admission(999999, 0, 4).is_err());
        let h = c.host_reserve_with_pages(1, 4).unwrap();
        let c = c.admit_host(h, 4).unwrap();
        let required = 64 + c.device_reserve().unwrap() as usize;
        assert!(c.validate_admission(required - 1, 64, 4).is_err());
        c.validate_admission(required, 64, 4).unwrap();
        let mut changed = c;
        changed.max_events_per_owner += 1;
        assert!(changed.validate_admission(required, 64, 4).is_err());
    }
    #[test]
    fn parent_wait_has_a_deadline_and_metadata_serialization_is_bounded() {
        let d = tempfile::tempdir().unwrap();
        let c = MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap();
        c.writers.lock().unwrap().push((0, Arc::default()));
        assert!(c.wait_for_writers(Duration::from_millis(2)).is_err());
        let mut sink = Metadata(Vec::with_capacity(META));
        assert!(sink.write_all(&vec![0; META + 1]).is_err());
        assert!(sink.0.is_empty());
        assert_eq!(sink.0.capacity(), META);
    }
    #[test]
    fn stalled_writer_accepts_whole_capture_then_rejects_total_budget() {
        let d = tempfile::tempdir().unwrap();
        let c = MixedTraceConfig::new(d.path(), 256 * 64, 64, 256, 4).unwrap();
        let (mut t, rx) = paused(c.clone());
        t.seal(json!({})).unwrap();
        for tx in 1..=4 {
            t.begin(tx, json!({"transaction":tx}));
            for _ in 0..64 {
                assert!(t.accepts_descriptor(&descriptor(64)));
                t.record(json!({"event":"legitimate","committed":false}), vec![0; 64]);
            }
            t.finish(true);
            t.committed_begin(tx, json!({"transaction":tx})).unwrap();
            t.committed_end();
        }
        assert!(!t.failed(), "writer did not consume a single record");
        assert_eq!(t.records, c.queue_record_limit().unwrap());
        assert_eq!(t.status.pending.load(Ordering::Acquire), t.records);
        assert_eq!(
            t.status.pending_bytes.load(Ordering::Acquire),
            t.bytes + t.index_bytes
        );
        assert!(c.host_reserve(1).unwrap() > (t.bytes + t.index_bytes) as u64);
        t.active = true;
        assert!(!t.accepts_descriptor(&descriptor(4)));
        assert_eq!(t.status.failure.get().unwrap().reason, "total_tensor_bytes");
        let now = Instant::now();
        t.close();
        assert!(now.elapsed() < Duration::from_millis(100));
        drop(rx);
        assert_eq!(t.status.pending_bytes.load(Ordering::Acquire), 0);
    }
    #[test]
    fn event_and_metadata_caps_report_specific_first_failure() {
        let d = tempfile::tempdir().unwrap();
        let (mut t, _rx) = paused(MixedTraceConfig::new(d.path(), 128, 8, 1, 4).unwrap());
        t.begin(1, json!({}));
        t.record(json!({"event":"x"}), vec![0; 4]);
        assert!(!t.accepts("y", 4));
        assert_eq!(t.status.failure.get().unwrap().reason, "tensor_event_count");
        assert_eq!(t.status.failure.get().unwrap().event, "y");
        let (mut t, _rx) = paused(MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap());
        t.config.max_metadata_bytes_per_owner = META;
        t.index_bytes = META;
        t.begin(1, json!({}));
        assert_eq!(
            t.status.failure.get().unwrap().reason,
            "total_metadata_bytes"
        );
    }
    #[test]
    fn async_writer_preserves_uncommitted_and_exclusive_create() {
        let d = tempfile::tempdir().unwrap();
        let c = MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap();
        let mut t = MixedTraceWriter::new(c.clone(), 0).unwrap();
        t.begin(1, json!({}));
        t.record(
            json!({"event":"x","committed":false}),
            1f32.to_le_bytes().to_vec(),
        );
        t.finish(true);
        t.close();
        c.wait_for_writers(Duration::from_secs(2)).unwrap();
        let lines = std::fs::read_to_string(d.path().join("rank-0/events.jsonl")).unwrap();
        assert!(lines.contains("\"committed\":false"));
        assert!(MixedTraceWriter::new(c.clone(), 0).is_err());
        // An independent config must still not overwrite the old directory.
        let collision_config = MixedTraceConfig::new(d.path(), 128, 8, 10, 4).unwrap();
        let mut collision = MixedTraceWriter::new(collision_config.clone(), 0).unwrap();
        collision.close();
        assert!(
            collision_config
                .wait_for_writers(Duration::from_secs(2))
                .is_err()
        );
    }
}
