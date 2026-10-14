//! Borrowed logical authority + authenticated physical view, not a second owner.
use super::append_dma::AppendResources;
use super::append_read::CudaAppendPrefix;
use super::*;
use crate::cpu::StreamingF32GqaConfig;
use crate::cuda::operator_owner::CudaF32Buffer;
use ferrule_common::host_kv::HostKvAppendReadWitness;
use ferrule_common::host_kv::placement::PlacementBudget;

/// Runtime-to-backend trust boundary. Metadata alone cannot implement this safely.
///
/// # Safety
/// Implementors must retain a borrow of the actual live logical reservation and
/// its unique page manager. `validate` must check reservation identity, source
/// incarnation/frontier and COW projection. The witness/layer must
/// not change while borrowed. Only the runtime reservation adapter implements
/// this contract in production; a witness-only implementation violates it.
/// This is deliberately NOT sealed across crates: another integration may use
/// `unsafe impl` only after auditing those obligations. In particular, `validate`
/// must not merely validate the public witness geometry or a cached success bit.
/// It must revalidate the retained live authority. All witness fields, source
/// incarnation, reservation and transaction identity, and layer must remain
/// stable through begin, polling, output consumption and release (also across
/// shared references/interior mutability). A new layer needs a new borrowed
/// sub-capability, never mutation of the currently installed authority.
///
/// Safe code cannot promote copied witness metadata into this authority:
/// ```compile_fail,E0200
/// use ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority;
/// use ferrule_common::{Result, host_kv::HostKvAppendReadWitness};
/// struct Metadata(HostKvAppendReadWitness);
/// impl AppendPrefixAuthority for Metadata {
///     fn witness(&self) -> &HostKvAppendReadWitness { &self.0 }
///     fn layer(&self) -> usize { 0 }
///     fn validate(&self) -> Result<()> { Ok(()) }
/// }
/// ```
pub unsafe trait AppendPrefixAuthority {
    fn witness(&self) -> &HostKvAppendReadWitness;
    fn layer(&self) -> usize;
    fn validate(&self) -> Result<()>;
}

/// Issued only by an authenticated entered-view callback. No public constructor,
/// Clone, pointer access, commit, or abort authority is exposed.
pub struct CudaAppendTransaction {
    pub(super) view: CudaKvView,
}
pub struct AppendGqaRequest<'a> {
    pub query: &'a CudaF32Buffer,
    pub key: &'a CudaF32Buffer,
    pub value: &'a CudaF32Buffer,
    pub config: StreamingF32GqaConfig,
}
#[must_use = "poll and finish; cancellation/drop never acknowledges custody"]
pub struct CudaAppendGqa<'a> {
    authority: &'a dyn AppendPrefixAuthority,
    read: Option<CudaAppendPrefix<'static>>,
    ready: bool,
    finished: bool,
}
impl CudaAppendTransaction {
    pub fn begin<'a>(
        &self,
        authority: &'a dyn AppendPrefixAuthority,
        operators: &CudaOperators,
        request: AppendGqaRequest<'_>,
        budget: &PlacementBudget,
    ) -> Result<CudaAppendGqa<'a>> {
        authority.validate()?;
        let witness = authority.witness();
        super::append_stream::AppendStream::<u64>::validate_config(witness, request.config)?;
        if request.query.len() != request.config.query_elements()?
            || request.key.len() != request.config.kv_elements(1)?
            || request.value.len() != request.key.len()
        {
            return Err(error("append bridge operand shape mismatch"));
        }
        let transaction = {
            let inner = self.view.inner.borrow();
            inner.available()?;
            if inner.storage().staging_page_limit().is_some() {
                return Err(error(
                    "chunked KV append-prefix bridge unsupported; use transaction staging",
                ));
            }
            inner.no_prefix_readers()?;
            let entry = inner
                .entries
                .get(&self.view.id)
                .ok_or_else(|| error("append transaction absent"))?;
            if !entry.entered
                || entry.phase != Phase::Active
                || entry.nonce != self.view.nonce
                || entry.epoch != self.view.epoch
                || witness.transaction() != self.view.id
                || !std::ptr::eq(operators, inner.context.as_ref())
            {
                return Err(error("append bridge foreign transaction/epoch/operator"));
            }
            for operand in [request.query, request.key, request.value] {
                operand
                    .as_device_buffer()
                    .check_context(operators.stream_clone().context(), "append operand")
                    .map_err(|e| error(e.to_string()))?;
                let base = operand.as_device_buffer().cu_deviceptr();
                let end = base
                    .checked_add(
                        operand
                            .len()
                            .checked_mul(4)
                            .ok_or_else(|| error("append byte overflow"))?
                            as u64,
                    )
                    .ok_or_else(|| error("append address overflow"))?;
                for plane in 0..2 {
                    let storage = inner
                        .storage()
                        .plane_storage(plane)
                        .ok_or_else(|| error("append requires F32"))?;
                    let p = storage.as_device_buffer().cu_deviceptr();
                    let q = p
                        .checked_add(
                            storage
                                .len()
                                .checked_mul(4)
                                .ok_or_else(|| error("KV byte overflow"))?
                                as u64,
                        )
                        .ok_or_else(|| error("KV address overflow"))?;
                    if base < q && p < end {
                        return Err(error("append input aliases KV allocation"));
                    }
                }
            }
            CudaPagedKvTransaction {
                id: self.view.id,
                identity: inner.identity.clone(),
                nonce: entry.nonce,
            }
        };
        // All payload/transport admission precedes physical read registration and
        // tail writes. Existing caller Q/K/V allocations remain caller-accounted.
        let mut resources = AppendResources::new(
            operators,
            witness,
            request.query.prefix(request.query.len())?,
            request.config,
            budget,
        )?;
        let operands = (|| {
            resources.append_operands(
                request.key.prefix(request.key.len())?,
                request.value.prefix(request.value.len())?,
            )
        })();
        if let Err(e) = operands {
            resources.release();
            return Err(e);
        }

        let mut pool = CudaPagedKvPool {
            inner: self.view.inner.clone(),
        };
        let read = match pool.derive_append_prefix(&transaction, witness, authority.layer()) {
            Ok(read) => read,
            Err(e) => {
                resources.release();
                return Err(e);
            }
        };
        let mut read = read.into_owned();
        read.adopt_resources(resources)?;
        let operation = CudaAppendGqa {
            authority,
            read: Some(read),
            ready: false,
            finished: false,
        };
        // Source originals were authenticated before any mutation. Only the
        // exact staged tail slot is writable; whole-plane callbacks stay closed.
        let result = (|| {
            let offsets = [
                operation.read.as_ref().expect("read").tail_range(0)?.start,
                operation.read.as_ref().expect("read").tail_range(1)?.start,
            ];
            let mut inner = operation.read.as_ref().expect("read").inner.borrow_mut();
            let Inner {
                entries, storage, ..
            } = &mut *inner;
            let entry = entries
                .get_mut(&transaction.id)
                .expect("authenticated Entry");
            let (key, value) = storage
                .as_mut()
                .expect("owned storage")
                .f32_transaction_planes()?;
            entry
                .prefix_dma
                .as_mut()
                .expect("installed resources")
                .write_tail(operators, key, value, offsets)
        })();
        if let Err(e) = result {
            operation.quarantine();
            return Err(e);
        }
        Ok(operation)
    }
}
impl CudaAppendGqa<'_> {
    fn quarantine(&self) {
        if let Some(read) = &self.read
            && let Some(entry) = read.inner.borrow().entries.get(&read.transaction.id)
        {
            let _ = entry.prefix_read.mark_unknown(&read.handle);
        }
    }
    pub fn cancel(&mut self) {
        self.quarantine();
    }
    pub fn poll(&mut self) -> Result<bool> {
        if self.finished {
            return Err(error("append operation already finished"));
        }
        if let Err(e) = self.authority.validate() {
            self.quarantine();
            return Err(e);
        }
        match self.read.as_mut().expect("read custody").poll_cpu_prefix() {
            Ok(ready) => {
                self.ready = ready;
                Ok(ready)
            }
            Err(e) => {
                self.quarantine();
                Err(e)
            }
        }
    }
}
impl<'a> CudaAppendGqa<'a> {
    pub fn finish(&mut self) -> Result<CudaAppendGqaOutput<'a>> {
        if !self.ready || self.finished {
            return Err(error("append output not ready or already taken"));
        }
        if let Err(e) = self.authority.validate() {
            self.quarantine();
            return Err(e);
        }
        let (buffer, ops) = {
            let read = self.read.as_ref().expect("read custody");
            read.validate()?;
            let inner = read.inner.borrow();
            let resources = inner
                .entries
                .get(&read.transaction.id)
                .expect("Entry retained")
                .prefix_dma
                .as_ref()
                .expect("ready resources");
            (resources.output_view()?, inner.context.clone())
        };
        self.finished = true;
        Ok(CudaAppendGqaOutput {
            authority: self.authority,
            read: self.read.take(),
            buffer: Some(buffer),
            ops,
            unknown: false,
        })
    }
}

/// The buffer here is only an allocation view. Original output, all DMA operands,
/// charge and unresolved consumer record remain in the same transaction Entry.
#[must_use = "return the output after its last consumer; drop retains Entry custody"]
pub struct CudaAppendGqaOutput<'a> {
    authority: &'a dyn AppendPrefixAuthority,
    read: Option<CudaAppendPrefix<'static>>,
    buffer: Option<CudaF32Buffer>,
    ops: Rc<CudaOperators>,
    unknown: bool,
}
impl CudaAppendGqaOutput<'_> {
    pub fn buffer(&self) -> Result<&CudaF32Buffer> {
        if self.unknown {
            return Err(error("append output completion unknown"));
        }
        self.buffer
            .as_ref()
            .ok_or_else(|| error("append output already returned"))
    }
    pub fn cancel(&mut self) {
        self.unknown = true;
        if let Some(read) = &self.read {
            read.inner.borrow_mut().poisoned = true;
        }
    }
    pub fn release(&mut self) -> Result<()> {
        if self.unknown {
            return Err(error("append output completion unknown"));
        }
        if self.read.is_none() {
            return Err(error("append output already returned"));
        }
        if let Err(e) = self.authority.validate() {
            self.cancel();
            return Err(e);
        }
        self.unknown = true;
        let result = (|| {
            let event = self.ops.record_compute_event()?;
            event.synchronize()?;
            // Return this alias before the owning Entry releases its resources.
            self.buffer = None;
            self.read
                .as_mut()
                .expect("read custody")
                .finish_with_consumer(event)
        })();
        if let Err(e) = result {
            self.cancel();
            return Err(e);
        }
        self.read = None;
        self.unknown = false;
        Ok(())
    }
}
impl Drop for CudaAppendGqaOutput<'_> {
    fn drop(&mut self) {
        if self.read.is_some() {
            self.cancel();
        }
        // Entry retains the original buffer and charge even after this view drops.
    }
}
impl Drop for CudaAppendGqa<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.quarantine();
        }
    }
}
