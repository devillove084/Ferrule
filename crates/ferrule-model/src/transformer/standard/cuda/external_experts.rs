//! Routed CUDA completion custody shared by the resident module and PP segment.
//! No state publication, transaction allocation or transport scheduling lives here.
use super::super::cuda_error;
use crate::decoder::{
    CudaKvView, HybridCudaDevice, HybridCudaRoutedExperts, PackedDecoderBatch, WorkProgress,
};
use crate::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use crate::transformer::{OperatorProgress, Rows};
use ferrule_backend::cuda::providers::DeviceBuffer;
use ferrule_common::{Result, execution::ExecutionTransactionId};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertForwardPhase {
    Active,
    Failed,
    Draining,
}

#[derive(Debug)]
pub(crate) struct ExpertForward {
    pub(crate) transaction: ExecutionTransactionId,
    pub(crate) batch: PackedDecoderBatch,
    pub(crate) phase: ExpertForwardPhase,
}
impl ExpertForward {
    pub(crate) fn check(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        if self.transaction != transaction
            || self.phase != ExpertForwardPhase::Active
            || &self.batch != batch
        {
            return Err(cuda_error(
                "external expert transaction/batch/generation/phase is not active",
            ));
        }
        Ok(())
    }
}

/// Freeze stable sequence identities in packed ROW order, not state-slot order.
pub(crate) fn routed_sequences(batch: &PackedDecoderBatch) -> Result<Vec<u64>> {
    if batch.row_to_sequence().len() != batch.len() {
        return Err(cuda_error("external expert packed row count mismatch"));
    }
    batch
        .row_to_sequence()
        .iter()
        .enumerate()
        .map(|(row, &index)| {
            let sequence = batch
                .sequences()
                .get(index)
                .filter(|sequence| sequence.query().contains(&row))
                .ok_or_else(|| cuda_error("external expert packed sequence mapping mismatch"))?;
            Ok(sequence.topology_id().get())
        })
        .collect()
}

pub(crate) struct RoutedCall<'a> {
    pub(crate) executor: &'a mut dyn crate::decoder::HybridCudaRoutedExecutor,
    pub(crate) device: &'a HybridCudaDevice,
    pub(crate) holds: &'a mut Vec<DeviceBuffer<f32>>,
}
impl RoutedCall<'_> {
    fn hold(&mut self, rows: &Rows) -> Result<()> {
        let rows = rows.cuda()?;
        let buffer = rows
            .f32_buffer()
            .ok_or_else(|| cuda_error("external expert requires F32 CUDA rows"))?;
        self.holds.push(
            buffer
                .as_device_buffer()
                .slice(0, rows.shape().elements())?,
        );
        rows.validate_owner(self.device.operators())?;
        Ok(())
    }
}
impl RoutedSwiGluExecutor for RoutedCall<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        if request.sequences.len() != request.input.shape().rows()
            || request.routes.rows() != request.sequences.len()
        {
            return Err(cuda_error("external routed rows/identities mismatch"));
        }
        let transaction = request.context.transaction;
        let shape = request.input.shape();
        let arena = request.arena;
        // Retain aliases BEFORE submission, including failure/panic paths. A root
        // stream fence alone says nothing about a remote consumer of these rows.
        self.hold(request.input)?;
        let result = self.executor.routed_swiglu(request, check_active)?;
        if let OperatorProgress::Ready(rows) = &result {
            self.hold(rows)?;
            if rows.shape() != shape || rows.arena() != arena {
                return Err(cuda_error("external expert result geometry mismatch"));
            }
        }
        check_active(transaction)?;
        Ok(result)
    }
}

#[derive(Default)]
pub(crate) struct ExternalExpertCustody {
    pub(crate) attachment: Option<HybridCudaRoutedExperts>,
    pub(crate) forward: Option<ExpertForward>,
    pub(crate) holds: Vec<DeviceBuffer<f32>>,
    pub(crate) kv: Option<CudaKvView>,
}
impl std::fmt::Debug for ExternalExpertCustody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalExpertCustody")
            .field("attachment", &self.attachment)
            .field("forward", &self.forward)
            .field("activation_holds", &self.holds.len())
            .field("retains_kv", &self.kv.is_some())
            .finish()
    }
}
impl ExternalExpertCustody {
    pub(crate) fn new(
        mut attachment: Option<HybridCudaRoutedExperts>,
        device: &HybridCudaDevice,
    ) -> Self {
        if let Some(experts) = &mut attachment {
            experts.owner = Some(device.clone());
        }
        Self {
            attachment,
            ..Self::default()
        }
    }
    pub(crate) fn begin(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
        kv: &CudaKvView,
    ) -> Result<()> {
        if self.forward.is_some() || transaction != kv.transaction() {
            return Err(cuda_error(
                "external expert forward already active or KV transaction mismatch",
            ));
        }
        if self.attachment.is_some() {
            self.forward = Some(ExpertForward {
                transaction,
                batch: batch.clone(),
                phase: ExpertForwardPhase::Active,
            });
            self.kv = Some(kv.clone());
        }
        Ok(())
    }
    pub(crate) fn check(
        &mut self,
        device: &HybridCudaDevice,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        device.ensure_ready()?;
        if let Some(experts) = &mut self.attachment {
            self.forward
                .as_ref()
                .ok_or_else(|| cuda_error("unknown external expert forward"))?
                .check(transaction, batch)?;
            if let Some(check) = &mut experts.check_active {
                check(transaction)?;
            }
        }
        Ok(())
    }
    // User-supplied transport callbacks cannot unwind past custody or replace
    // the original forward error. A panic means completion evidence was lost.
    fn callback<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self))) {
            Ok(result) => result,
            Err(_) => {
                if let Some(owner) = self.attachment.as_ref().and_then(|a| a.owner.as_ref()) {
                    owner.quarantine();
                    owner.ensure_ready()?;
                }
                Err(cuda_error(
                    "external expert callback panicked; completion unknown",
                ))
            }
        }
    }
    pub(crate) fn fail(&mut self) -> Result<()> {
        self.callback(|this| {
            if let Some(forward) = &mut this.forward
                && forward.phase == ExpertForwardPhase::Active
            {
                forward.phase = ExpertForwardPhase::Failed;
                if let Some(experts) = &mut this.attachment {
                    experts
                        .executor
                        .as_mut()
                        .expect("attached executor")
                        .on_error(forward.transaction);
                }
            }
            Ok(())
        })
    }
    pub(crate) fn begin_drain(&mut self) -> Result<()> {
        self.fail()?;
        if let Some(forward) = &mut self.forward {
            forward.phase = ExpertForwardPhase::Draining;
        }
        Ok(())
    }
    pub(crate) fn drain(&mut self, device: &HybridCudaDevice, shutdown: bool) -> Result<bool> {
        device.ensure_ready()?;
        self.callback(|this| {
            let Some(experts) = &mut this.attachment else {
                return Ok(true);
            };
            if experts.closed {
                return Ok(true);
            }
            let executor = experts.executor.as_mut().expect("attached executor");
            let progress = executor.drain()?;
            let progress = if progress == WorkProgress::Complete && shutdown {
                executor.shutdown()?
            } else {
                progress
            };
            match progress {
                WorkProgress::Complete => {
                    device.fence()?;
                    this.holds.clear();
                    Ok(true)
                }
                WorkProgress::Pending => Ok(false),
                WorkProgress::Unknown => {
                    device.quarantine();
                    device.ensure_ready()?;
                    unreachable!()
                }
            }
        })
    }
    pub(crate) fn require_drain(
        &mut self,
        device: &HybridCudaDevice,
        shutdown: bool,
    ) -> Result<()> {
        if !self.drain(device, shutdown)? {
            return Err(cuda_error(
                "external expert completion pending; retain model/KV custody",
            ));
        }
        Ok(())
    }
    /// Caller has proved root completion and, on success, the state frontier.
    pub(crate) fn finish(&mut self) {
        self.forward = None;
        self.kv = None;
    }
    pub(crate) fn close(&mut self) {
        if let Some(experts) = &mut self.attachment {
            experts.closed = true;
        }
        self.finish();
    }
    pub(crate) fn retain_unknown(&mut self) {
        std::mem::forget(self.attachment.take());
        std::mem::forget(self.kv.take());
        std::mem::forget(std::mem::take(&mut self.holds));
    }
}
