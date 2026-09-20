//! Bounded pinned staging at PP, logits and EP boundaries only.

use ferrule_backend::cuda::operators::linear::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::{CudaAsyncTransport, CudaTransferCompletion, CudaTransferTicket};
use ferrule_common::{Error, Result};

use super::cuda_error;
use crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG;

pub(super) struct Boundary {
    transport: CudaAsyncTransport,
}

impl Boundary {
    pub fn new(ops: &CudaOperators) -> Result<Self> {
        Ok(Self {
            transport: ops.new_async_transport(DEFAULT_TRANSFER_CONFIG)?,
        })
    }

    pub fn upload(&mut self, ops: &CudaOperators, values: &[f32]) -> Result<CudaF32Buffer> {
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
        let producer = ops.record_compute_event()?;
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

    fn wait(&mut self, ticket: &CudaTransferTicket) -> Result<CudaTransferCompletion> {
        loop {
            if let Some(completion) = self.transport.poll(ticket).map_err(backend)? {
                return Ok(completion);
            }
            std::thread::yield_now();
        }
    }

    pub fn drain(&mut self) -> Result<()> {
        self.transport.drain().map(|_| ()).map_err(backend)
    }
    pub fn needs_quarantine(&self) -> bool {
        self.transport.stats().quarantined
    }
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Backend {
        source: Box::new(error),
    }
}
