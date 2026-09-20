//! Synchronous shard boundary over the backend's single pinned-slot owner.
//!
//! Only the backend tracks slot custody, generations and allocation holds. This
//! adapter retains per-call compute buffers as well, including across unwinding.
//! Host polling is intentional: the current shard API promises no compute /
//! communication overlap. Weight materialization is a separate path.

use std::fmt;
use std::mem::size_of;
use std::rc::Rc;

use ferrule_backend::cuda::context::CudaComputeEvent;
use ferrule_backend::cuda::operators::linear::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::{
    CudaAsyncTransport, CudaAsyncTransportStats, CudaTransferCompletion, CudaTransferConfig,
    CudaTransferTicket,
};
use ferrule_common::{Error, Result};

pub(super) const DEFAULT_TRANSFER_CONFIG: CudaTransferConfig = CudaTransferConfig {
    chunk_elements: 64 * 1024 / size_of::<f32>(),
    tx_slots: 1,
    rx_slots: 1,
};

/// Shard failure with an explicit physical-quiescence / owner-reuse distinction.
///
/// Preserved as `ferrule_common::Error::ModelSource`, not flattened to text.
/// Workers must inspect this error (or the shard's getters) BEFORE translating
/// it to their own error type. `needs_quarantine()` means retain the worker and
/// report an unknown/unsafe completion, never an ordinary failure ACK. A poisoned
/// transport remains unusable even if a later drain proves physical quiescence.
/// No runtime-specific panic or scheduling type is used in model code.
#[derive(Debug)]
pub struct CudaShardError {
    source: Error,
    cleanup: Option<Error>,
    quiescent: bool,
    quarantined: bool,
}

impl CudaShardError {
    pub fn is_quiescent(&self) -> bool {
        self.quiescent
    }

    pub fn needs_quarantine(&self) -> bool {
        !self.quiescent || self.quarantined
    }

    /// Recover the typed failure at the common error boundary.
    pub fn from_error(error: &Error) -> Option<&Self> {
        match error {
            Error::ModelSource { source } => source.downcast_ref(),
            Error::Context { source, .. } => Self::from_error(source),
            Error::Cleanup {
                source, cleanup, ..
            } => Self::from_error(cleanup).or_else(|| Self::from_error(source)),
            _ => None,
        }
    }

    pub(super) fn wrap(
        source: Error,
        cleanup: Option<Error>,
        quiescent: bool,
        quarantined: bool,
    ) -> Error {
        Error::ModelSource {
            source: Box::new(Self {
                source,
                cleanup,
                quiescent,
                quarantined,
            }),
        }
    }
}

impl fmt::Display for CudaShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA shard: {}; quiescent={}, needs_quarantine={}",
            self.source,
            self.is_quiescent(),
            self.needs_quarantine(),
        )?;
        if let Some(cleanup) = &self.cleanup {
            write!(f, "; cleanup: {cleanup}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CudaShardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Backend {
        source: Box::new(error),
    }
}

pub(super) struct CudaShardTransfer {
    transport: CudaAsyncTransport,
    ops: Rc<CudaOperators>,
    input: Option<CudaF32Buffer>,
    output: Option<CudaF32Buffer>,
    producer: Option<CudaComputeEvent>,
    quiescent: bool,
    // Owner health, not a second slot state machine. Backend quarantine is sticky.
    poisoned: bool,
}

impl CudaShardTransfer {
    pub(super) fn new(ops: Rc<CudaOperators>, config: CudaTransferConfig) -> Result<Self> {
        Ok(Self {
            transport: ops.new_async_transport(config)?,
            ops,
            input: None,
            output: None,
            producer: None,
            quiescent: true,
            poisoned: false,
        })
    }

    pub(super) fn config(&self) -> CudaTransferConfig {
        self.transport.config()
    }

    pub(super) fn stats(&self) -> CudaAsyncTransportStats {
        self.transport.stats()
    }

    pub(super) fn is_quiescent(&self) -> bool {
        self.quiescent && self.stats().device_holds_in_use == 0
    }

    pub(super) fn needs_quarantine(&self) -> bool {
        !self.is_quiescent() || self.poisoned || self.stats().quarantined
    }

    pub(super) fn ensure_ready(&self) -> Result<()> {
        if self.needs_quarantine() {
            return Err(self.failure(
                Error::Model {
                    message: "CUDA shard owner requires quarantine".into(),
                },
                None,
            ));
        }
        Ok(())
    }

    pub(super) fn execute(
        &mut self,
        values: &[f32],
        launch: impl FnOnce(&CudaOperators, &CudaF32Buffer) -> Result<CudaF32Buffer>,
    ) -> Result<Vec<f32>> {
        self.ensure_ready()?;
        self.quiescent = false;
        let result = (|| {
            self.input = Some(self.ops.zero_f32_buffer(values.len())?);
            let chunk = self.config().chunk_elements;
            for (index, values) in values.chunks(chunk).enumerate() {
                let ticket = self
                    .transport
                    .submit_h2d_f32(
                        self.input.as_mut().expect("retained input"),
                        index * chunk,
                        values,
                    )
                    .map_err(backend)?;
                self.transport
                    .wait_h2d_on_compute(&ticket)
                    .map_err(backend)?;
                match self.wait(&ticket)? {
                    CudaTransferCompletion::H2D { .. } => {}
                    other => return Err(unexpected(other)),
                }
            }
            self.output = Some(launch(
                &self.ops,
                self.input.as_ref().expect("retained input"),
            )?);
            self.producer = Some(self.ops.record_compute_event()?);
            let len = self.output.as_ref().expect("retained output").len();
            let mut host = Vec::with_capacity(len);
            for offset in (0..len).step_by(chunk) {
                let ticket = self
                    .transport
                    .submit_d2h_f32_after(
                        self.output.as_ref().expect("retained output"),
                        offset,
                        chunk.min(len - offset),
                        self.producer.as_ref().expect("retained producer"),
                    )
                    .map_err(backend)?;
                match self.wait(&ticket)? {
                    CudaTransferCompletion::D2H { values, .. } => host.extend_from_slice(&values),
                    other => return Err(unexpected(other)),
                }
            }
            Ok(host)
        })();
        match result {
            Ok(host) => {
                // Every exact D2H event follows the producer, including the tail.
                // H2D slots were consumed before reuse. No DMA reads this Vec.
                self.quiescent = true;
                self.release_buffers();
                Ok(host)
            }
            Err(source) => {
                let cleanup = self.fence_all().err();
                Err(self.failure(source, cleanup))
            }
        }
    }

    fn wait(&mut self, ticket: &CudaTransferTicket) -> Result<CudaTransferCompletion> {
        loop {
            if let Some(completion) = self.transport.poll(ticket).map_err(backend)? {
                return Ok(completion);
            }
            std::thread::yield_now();
        }
    }

    /// The transport drains EVERY active slot, including control-stream events
    /// and unrecorded-submission fallback fences. Neither stream fence below is
    /// skipped after another error. Unknown custody is never cleared.
    fn fence_all(&mut self) -> Result<()> {
        self.quiescent = false;
        let drain = self.transport.drain().map(|_| ()).map_err(backend);
        let compute = self.ops.sync_stream();
        let upload = self.ops.sync_upload_stream();
        let failures: Vec<_> = [drain, compute, upload]
            .into_iter()
            .filter_map(Result::err)
            .collect();
        self.quiescent = failures.is_empty();
        self.poisoned |= !self.quiescent || self.stats().quarantined;
        if self.quiescent {
            self.release_buffers();
        }
        Error::failures("CUDA shard drain/compute/upload fences", failures)
    }

    pub(super) fn quiesce(&mut self) -> Result<()> {
        let proof = self.fence_all();
        match proof {
            Err(source) => Err(self.failure(source, None)),
            Ok(()) => self.ensure_ready(),
        }
    }

    fn failure(&self, source: Error, cleanup: Option<Error>) -> Error {
        CudaShardError::wrap(
            source,
            cleanup,
            self.is_quiescent(),
            self.needs_quarantine(),
        )
    }

    fn release_buffers(&mut self) {
        self.input = None;
        self.output = None;
        self.producer = None;
    }
}

impl Drop for CudaShardTransfer {
    fn drop(&mut self) {
        // A panic while fencing must also retain the compute resources. The
        // backend transport independently guards/leaks its pinned DMA bundle.
        let proof = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.fence_all()));
        if !matches!(proof, Ok(Ok(()))) {
            std::mem::forget(self.input.take());
            std::mem::forget(self.output.take());
            std::mem::forget(self.producer.take());
            std::mem::forget(Rc::clone(&self.ops));
            std::mem::forget(proof);
        }
    }
}

fn unexpected(completion: CudaTransferCompletion) -> Error {
    Error::Internal {
        message: format!("unexpected shard transfer completion: {completion:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pinned_budget_is_bounded() {
        DEFAULT_TRANSFER_CONFIG.validate().unwrap();
        assert_eq!(
            DEFAULT_TRANSFER_CONFIG.chunk_elements * size_of::<f32>(),
            64 * 1024
        );
        assert_eq!(
            (
                DEFAULT_TRANSFER_CONFIG.tx_slots,
                DEFAULT_TRANSFER_CONFIG.rx_slots
            ),
            (1, 1)
        );
    }

    #[test]
    #[ignore = "requires CUDA and FERRULE_TP_TEST_DEVICE; external timeout and --test-threads=1"]
    fn operation_error_and_panic_keep_resources_until_quiescence() {
        let ordinal: usize = std::env::var("FERRULE_TP_TEST_DEVICE")
            .expect("explicit device required")
            .parse()
            .unwrap();
        std::thread::spawn(move || {
            let ops = Rc::new(CudaOperators::new_on_device(ordinal).unwrap());
            let mut transfer = CudaShardTransfer::new(
                ops,
                CudaTransferConfig {
                    chunk_elements: 3,
                    tx_slots: 1,
                    rx_slots: 1,
                },
            )
            .unwrap();
            let error = transfer
                .execute(&[1.0, 2.0, 3.0, 4.0], |_, _| {
                    Err(Error::Model {
                        message: "model failure after H2D".into(),
                    })
                })
                .unwrap_err();
            let typed = CudaShardError::from_error(&error).unwrap();
            assert!(typed.is_quiescent() && !typed.needs_quarantine());
            assert_eq!(transfer.stats().device_holds_in_use, 0);
            assert!(transfer.input.is_none());

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transfer.execute(&[5.0, 6.0, 7.0, 8.0], |_, _| {
                    panic!("model unwind after H2D")
                })
            }));
            assert!(panic.is_err());
            assert!(!transfer.is_quiescent());
            assert!(transfer.input.is_some());
            assert!(transfer.ensure_ready().is_err());
            transfer.quiesce().unwrap();
            assert!(transfer.is_quiescent() && !transfer.needs_quarantine());
            assert!(transfer.input.is_none());
            assert_eq!(transfer.stats().device_holds_in_use, 0);
            // A successfully fenced operation error/unwind does not poison an
            // otherwise healthy transport; both directions can be used again.
            let output = transfer
                .execute(&[9.0, 10.0, 11.0, 12.0], |ops, input| {
                    ops.zero_f32_buffer(input.len())
                })
                .unwrap();
            assert_eq!(output, [0.0; 4]);
        })
        .join()
        .expect("CUDA transfer owner");
    }

    #[test]
    fn common_error_preserves_quiescence_and_sticky_quarantine() {
        for (quiescent, poisoned) in [(true, false), (false, true), (true, true)] {
            let error = CudaShardError::wrap(
                Error::Model {
                    message: "test failure".into(),
                },
                None,
                quiescent,
                poisoned,
            );
            let typed = CudaShardError::from_error(&error).unwrap();
            assert_eq!(typed.is_quiescent(), quiescent);
            assert_eq!(typed.needs_quarantine(), !quiescent || poisoned);
            let context = Error::context("worker boundary", error);
            assert_eq!(
                CudaShardError::from_error(&context).unwrap().is_quiescent(),
                quiescent
            );
        }
    }
}
