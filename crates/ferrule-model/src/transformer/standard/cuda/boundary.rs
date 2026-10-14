//! Bounded pinned staging at PP, logits and EP boundaries only.

use ferrule_backend::cuda::operators::linear::{
    CudaBf16Buffer, CudaF32Buffer, CudaOperators, F32ToBf16RowsLayout,
};
use ferrule_backend::cuda::{CudaAsyncTransport, CudaTransferCompletion, CudaTransferTicket};
use ferrule_common::{Error, Result};

use super::cuda_error;
use crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG;

pub(super) struct Boundary {
    transport: CudaAsyncTransport,
    // Owner health, not transfer-slot custody: a missing producer proof stays
    // unknown even if the backend's cleanup or a later drain succeeds.
    poisoned: bool,
    #[cfg(test)]
    pub(super) fail_next_producer_record: Option<Error>,
}

impl Boundary {
    pub fn new(ops: &CudaOperators) -> Result<Self> {
        Ok(Self {
            transport: ops.new_async_transport(DEFAULT_TRANSFER_CONFIG)?,
            poisoned: false,
            #[cfg(test)]
            fail_next_producer_record: None,
        })
    }

    pub fn upload(&mut self, ops: &CudaOperators, values: &[f32]) -> Result<CudaF32Buffer> {
        self.ensure_ready()?;
        let mut output = ops.zero_f32_buffer(values.len())?;
        let chunk = self.transport.config().chunk_elements;
        for (index, values) in values.chunks(chunk).enumerate() {
            let ticket = self
                .transport
                .submit_h2d_f32(&mut output, index * chunk, values)
                .map_err(backend)?;
            self.transport
                .wait_h2d_on_compute(&ticket)
                .map_err(backend)?;
            if !matches!(self.wait(&ticket)?, CudaTransferCompletion::H2D { .. }) {
                return Err(cuda_error("unexpected pinned H2D completion"));
            }
        }
        Ok(output)
    }

    pub fn download(&mut self, ops: &CudaOperators, input: &CudaF32Buffer) -> Result<Vec<f32>> {
        self.ensure_ready()?;
        #[cfg(test)]
        let producer = match self.fail_next_producer_record.take() {
            Some(error) => Err(error),
            None => ops.record_compute_event(),
        };
        #[cfg(not(test))]
        let producer = ops.record_compute_event();
        // Preserve the original typed error (including backend cleanup), but
        // never treat a successful cleanup sync as the missing event proof.
        let producer = producer.inspect_err(|_| self.poisoned = true)?;
        let chunk = self.transport.config().chunk_elements;
        let mut values = Vec::with_capacity(input.len());
        for offset in (0..input.len()).step_by(chunk) {
            let ticket = self
                .transport
                .submit_d2h_f32_after(input, offset, chunk.min(input.len() - offset), &producer)
                .map_err(backend)?;
            match self.wait(&ticket)? {
                CudaTransferCompletion::D2H { values: part, .. } => values.extend_from_slice(&part),
                _ => return Err(cuda_error("unexpected pinned D2H completion")),
            }
        }
        Ok(values)
    }

    /// HostRows represents BF16 as exactly expanded F32. Use the same bounded
    /// pinned transfer slots, then discard the explicit F32 staging allocation.
    pub fn upload_bf16(&mut self, ops: &CudaOperators, values: &[f32]) -> Result<CudaBf16Buffer> {
        let staging = self.upload(ops, values)?;
        ops.f32_rows_to_bf16_rne(&staging, F32ToBf16RowsLayout::packed(1, 1, values.len())?)
    }

    /// Q/logits/PP export expansion is scratch, not a resident F32 tensor mirror.
    pub fn download_bf16(
        &mut self,
        ops: &CudaOperators,
        input: &CudaBf16Buffer,
    ) -> Result<Vec<f32>> {
        let staging = ops.bf16_to_f32(input)?;
        self.download(ops, &staging)
    }

    fn wait(&mut self, ticket: &CudaTransferTicket) -> Result<CudaTransferCompletion> {
        // The backend preserves the original exact-ticket query/yield loop.
        // Yielding is never completion evidence or custody transfer.
        self.transport.wait(ticket).map_err(backend)
    }

    pub fn drain(&mut self) -> Result<()> {
        self.transport.drain().map(|_| ()).map_err(backend)
    }
    /// Copies host-side transfer metadata only; does not query or retire tickets.
    pub fn observation_stats(&self) -> ferrule_backend::cuda::CudaAsyncTransportStats {
        self.transport.stats()
    }

    pub fn needs_quarantine(&self) -> bool {
        self.poisoned || self.transport.stats().quarantined
    }

    fn ensure_ready(&self) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error(
                "CUDA boundary requires quarantine; completion is unknown",
            ));
        }
        Ok(())
    }
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Backend {
        source: Box::new(error),
    }
}
