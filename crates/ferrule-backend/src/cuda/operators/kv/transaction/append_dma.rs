//! Physical resources live in the existing Entry, never in the borrowed handle.
use super::append_stream::{AppendStream, PrefixDownload};
use super::plan::error;
use crate::cpu::{StreamingF32GqaConfig, StreamingF32GqaPartial};
use crate::cuda::operator_owner::{CudaComputeEvent, CudaF32Buffer, CudaOperators};
use crate::cuda::{
    CudaAsyncTransport, CudaTransferCompletion, CudaTransferConfig, CudaTransferId,
    CudaTransferTicket,
};
use ferrule_common::Result;
use ferrule_common::host_kv::HostKvAppendReadWitness;
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementReservation};
use std::ops::Range;

pub(super) enum AppendEvidence {
    Producer(CudaComputeEvent),
    /// Created only after all exact D2H tickets were consumed successfully.
    DmaConsumed,
    /// Synchronous CPU partial consumer returned successfully, not a CUDA ACK.
    CpuConsumed,
}
impl AppendEvidence {
    pub fn is_complete(&self) -> Result<bool> {
        match self {
            Self::Producer(event) => event.is_complete(),
            Self::DmaConsumed | Self::CpuConsumed => Ok(true),
        }
    }
}

pub(super) struct AppendResources {
    stream: AppendStream<CudaTransferId>,
    transport: CudaAsyncTransport,
    query: CudaF32Buffer,
    producer: CudaComputeEvent,
    ticket: Option<CudaTransferTicket>,
    charge: Option<PlacementReservation>,
    interrupted: bool,
    tail: Option<[CudaF32Buffer; 2]>,
    output: Option<CudaF32Buffer>,
    output_ready: bool,
}
impl AppendResources {
    /// Admission is atomic and precedes transport/payload allocation. Reuses the
    /// conservative six-domain committed budget (including unused GPU staging).
    pub fn new(
        ops: &CudaOperators,
        witness: &HostKvAppendReadWitness,
        query: CudaF32Buffer,
        config: StreamingF32GqaConfig,
        budget: &PlacementBudget,
    ) -> Result<Self> {
        AppendStream::<CudaTransferId>::validate_config(witness, config)?;
        if query.len() != config.query_elements()? {
            return Err(error("append query length mismatch"));
        }
        query
            .as_device_buffer()
            .check_context(ops.stream_clone().context(), "append Q")
            .map_err(|e| error(e.to_string()))?;
        let credits = super::super::committed_gqa_credits(config, witness.source().pages.len())?;
        let charge = budget
            .reserve_resources(credits)
            .map_err(|e| error(e.to_string()))?;
        let prepared = (|| {
            let stream = AppendStream::new(witness, config)?;
            let transport = ops.new_async_transport(CudaTransferConfig {
                chunk_elements: config
                    .query_elements()?
                    .max(config.kv_elements(config.block_tokens)?),
                tx_slots: 1,
                rx_slots: 1,
            })?;
            // This is the actual Q/K/V producer predecessor, not DMA evidence.
            let producer = ops.record_compute_event()?;
            Ok((stream, transport, producer))
        })();
        let (stream, transport, producer) = match prepared {
            Ok(parts) => parts,
            Err(e) => {
                charge.release();
                return Err(e);
            }
        };
        Ok(Self {
            stream,
            transport,
            query,
            producer,
            ticket: None,
            charge: Some(charge),
            interrupted: false,
            tail: None,
            output: None,
            output_ready: false,
        })
    }
    pub(super) fn append_operands(
        &mut self,
        key: CudaF32Buffer,
        value: CudaF32Buffer,
    ) -> Result<()> {
        self.stream.include_tail()?;
        self.tail = Some([key, value]);
        Ok(())
    }
    pub(super) fn write_tail(
        &mut self,
        ops: &CudaOperators,
        key: &mut CudaF32Buffer,
        value: &mut CudaF32Buffer,
        offsets: [usize; 2],
    ) -> Result<()> {
        self.interrupted = true;
        let tail = self
            .tail
            .as_ref()
            .ok_or_else(|| error("append operands absent"))?;
        ops.copy_f32_range(&tail[0], 0, key, offsets[0], tail[0].len())?;
        ops.copy_f32_range(&tail[1], 0, value, offsets[1], tail[1].len())?;
        self.producer = ops.record_compute_event()?;
        self.interrupted = false;
        Ok(())
    }
    pub(super) fn output_view(&self) -> Result<CudaF32Buffer> {
        if self.is_unknown() || !self.output_ready || self.ticket.is_some() {
            return Err(error("append output not quiescent"));
        }
        let buffer = self.output.as_ref().expect("ready output");
        buffer.prefix(buffer.len())
    }
    pub fn is_unknown(&self) -> bool {
        self.interrupted || self.stream.is_unknown()
    }
    pub fn request(&self) -> Result<Option<PrefixDownload>> {
        if self.is_unknown() {
            return Err(error("append DMA quarantined"));
        }
        if self.ticket.is_some() {
            return Ok(None);
        }
        self.stream.request()
    }
    pub fn step(
        &mut self,
        ops: &CudaOperators,
        source: Option<(&CudaF32Buffer, Range<usize>)>,
    ) -> Result<bool> {
        if self.is_unknown() {
            return Err(error("append DMA quarantined"));
        }
        // Remains set on driver error, malformed payload, or unwind. Entry's
        // normal lifecycle must never mistake any of those for no submission.
        self.interrupted = true;
        if let Some(ticket) = self.ticket.as_ref() {
            if let Some(completion) = self
                .transport
                .poll(ticket)
                .map_err(|e| error(e.to_string()))?
            {
                match completion {
                    CudaTransferCompletion::D2H { id, values } if id == ticket.id() => {
                        self.stream.complete(id, values)?;
                        self.ticket = None;
                    }
                    CudaTransferCompletion::H2D { id }
                        if id == ticket.id() && self.output.is_some() =>
                    {
                        self.ticket = None;
                        self.output_ready = true;
                    }
                    _ => {
                        return Err(error(
                            "append requires exact successful transfer completion",
                        ));
                    }
                }
            }
        } else if let Some(request) = self.stream.request()? {
            let (buffer, range) = match request {
                PrefixDownload::Query => (&self.query, 0..self.query.len()),
                _ => source.ok_or_else(|| error("append source range absent"))?,
            };
            let ticket = self
                .transport
                .submit_d2h_f32_after(buffer, range.start, range.len(), &self.producer)
                .map_err(|e| error(e.to_string()))?;
            // Transport already retains source/slot even if no ticket returns.
            self.ticket = Some(ticket);
            self.stream
                .submitted(self.ticket.as_ref().expect("installed ticket").id())?;
        }
        if self.stream.ready()? && self.tail.is_some() && self.output.is_none() {
            self.output = Some(ops.zero_f32_buffer(self.query.len())?);
            let mut normalized = None;
            self.stream.consume(|total| {
                normalized = Some(total.output());
                Ok(())
            })?;
            self.ticket = Some(
                self.transport
                    .submit_h2d_f32(
                        self.output.as_mut().expect("output retained"),
                        0,
                        normalized.as_deref().expect("normalized once"),
                    )
                    .map_err(|e| error(e.to_string()))?,
            );
        }
        let ready = if self.tail.is_some() {
            self.output_ready
        } else {
            self.stream.ready()?
        };
        self.interrupted = false;
        Ok(ready)
    }
    pub fn consume(
        &mut self,
        consumer: impl FnOnce(&StreamingF32GqaPartial) -> Result<()>,
    ) -> Result<()> {
        if self.is_unknown() {
            return Err(error("append DMA quarantined"));
        }
        self.stream.consume(consumer)
    }
    pub fn evidence(&self) -> Result<(AppendEvidence, AppendEvidence)> {
        if self.is_unknown() || self.ticket.is_some() || !self.stream.consumed()? {
            return Err(error("append DMA/consumer custody is unresolved"));
        }
        Ok((AppendEvidence::DmaConsumed, AppendEvidence::CpuConsumed))
    }
    pub fn release(mut self) {
        let charge = self.charge.take().expect("admitted resources");
        // Physical resources go away before refunding the shared budget.
        drop(self);
        charge.release();
    }
}
