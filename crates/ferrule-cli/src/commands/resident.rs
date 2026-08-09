use std::future::Future;

use ferrule_runtime::ResidentTopKDriverConfig;
use ferrule_runtime::{RequestTerminal, ResidentSchedulerConfig, SequenceState};

/// Run a non-`Send` inference owner on the calling OS thread.
///
/// CUDA context creation, completion reactors, stream callbacks, and every model
/// step therefore remain attached to one owner thread.
pub(crate) fn block_on_local_inference<F, T>(future: F) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, future)
}

/// Default per-position proposal confidence threshold used by every
/// interactive/benchmark entry point until the calibrated batch-wide
/// scheduler lands.
pub(crate) const DEFAULT_PROPOSAL_CONFIDENCE_THRESHOLD: f32 = 0.2;

pub(crate) fn require_finished_request(
    terminal: RequestTerminal,
    context: &str,
) -> anyhow::Result<SequenceState> {
    match terminal {
        RequestTerminal::Finished(sequence) => Ok(sequence),
        RequestTerminal::Cancelled(sequence) => anyhow::bail!(
            "{context} request {:?} was cancelled ({:?})",
            sequence.request_id,
            sequence.finish_reason
        ),
        RequestTerminal::Failed(sequence) => anyhow::bail!(
            "{context} request {:?} failed ({:?})",
            sequence.request_id,
            sequence.finish_reason
        ),
    }
}

/// Driver configuration shared by all resident-model entry points.
pub(crate) fn resident_driver_config(
    ctx_size: usize,
    stop_at_eos: bool,
) -> ResidentTopKDriverConfig {
    ResidentTopKDriverConfig {
        ctx_size,
        stop_at_eos,
        enable_native_proposals: true,
        proposal_confidence_threshold: DEFAULT_PROPOSAL_CONFIDENCE_THRESHOLD,
    }
}

/// Scheduler configuration for the single-sequence interactive/benchmark path.
/// Mixed batching is disabled explicitly so the decision is not left to a
/// default that may drift.
pub(crate) fn single_sequence_scheduler_config(
    prefill_chunk_size: usize,
) -> ResidentSchedulerConfig {
    ResidentSchedulerConfig {
        prefill_chunk_size: prefill_chunk_size.max(1),
        max_active_sequences: 1,
        max_decode_batch: 1,
        allow_mixed_batches: false,
        ..ResidentSchedulerConfig::default()
    }
}
