//! Single-submission upload capability. The provider allocates/recycles frames
//! and decides operation faults/quarantine; this adapter returns the original
//! event/frame custody and notification error without claiming quiescence.

use super::CudaExpertFrame;
use super::payload::PinnedExpertBundle;
use crate::runner::completion_notify_callback;
use ferrule_backend::cuda::operators::moe::{CudaOperators, CudaRoutedExpertMaterialization};
use ferrule_common::{CompletionHub, Error, QuiescenceEvidence, Result};

pub(super) struct CudaExpertUploadTicket {
    materialization: CudaRoutedExpertMaterialization,
    frame: Option<CudaExpertFrame>,
}

impl CudaExpertUploadTicket {
    pub(super) fn is_complete(&self) -> Result<bool> {
        self.materialization.is_complete()
    }

    pub(super) fn drain_into_frame(mut self) -> Result<CudaExpertFrame> {
        self.materialization.synchronize()?;
        self.frame.take().ok_or_else(|| Error::Internal {
            message: "CUDA expert upload lost its frame".into(),
        })
    }
}

impl Drop for CudaExpertUploadTicket {
    fn drop(&mut self) {
        if self.frame.is_some()
            && !matches!(self.materialization.is_complete(), Ok(true))
            && self.materialization.synchronize().is_err()
            && let Some(frame) = self.frame.take()
        {
            std::mem::forget(frame);
        }
    }
}

pub(super) fn submit(
    ops: &CudaOperators,
    completion_hub: &CompletionHub,
    bundle: PinnedExpertBundle,
    frame: CudaExpertFrame,
) -> std::result::Result<(CudaExpertUploadTicket, Option<Error>), (Error, QuiescenceEvidence)> {
    let [
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        down_weight,
        down_scale,
    ] = bundle.into_upload_buffers();
    let (frame, materialization, notify_error) = submit_retaining_frame(
        frame,
        |frame| {
            ops.materialize_routed_expert_from_pinned_async(
                &mut frame.expert,
                gate_weight,
                gate_scale,
                up_weight,
                up_scale,
                down_weight,
                down_scale,
            )
        },
        || ops.notify_upload_stream(completion_notify_callback(completion_hub.clone())),
    )?;
    Ok((
        CudaExpertUploadTicket {
            materialization,
            frame: Some(frame),
        },
        notify_error,
    ))
}

// Keep the native submission/fault boundary independently testable without a GPU.
fn submit_retaining_frame<F, E>(
    mut frame: F,
    submit: impl FnOnce(&mut F) -> Result<E>,
    notify: impl FnOnce() -> Result<()>,
) -> std::result::Result<(F, E, Option<Error>), (Error, QuiescenceEvidence)> {
    let event = match submit(&mut frame) {
        Ok(event) => event,
        Err(error) => {
            // The backend may have submitted a prefix and leaked unknown
            // references. Neither its error nor Drop is a fence proof.
            std::mem::forget(frame);
            return Err((error, QuiescenceEvidence::Unknown));
        }
    };
    let notify_error = notify().err();
    Ok((frame, event, notify_error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[derive(Debug)]
    struct Tracked<'a>(&'a Cell<usize>);
    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    #[ignore = "requires real CUDA; BF16 transfer boundary, not routed FP4 success"]
    fn gpu_bf16_submission_notification_failure_retains_exact_ticket_and_frame() {
        use ferrule_backend::cuda::operators::moe::CudaArtifactLinearShape;
        let ops = CudaOperators::new_on_device(0).expect("real CUDA transfer required");
        let shape = CudaArtifactLinearShape::Bf16Bytes {
            out_features: 32,
            in_features: 32,
        };
        let source = ops.pin_u8_host_buffer(&vec![0x3f; 2048]).unwrap();
        let handle = ops.allocate_artifact_linear_device(shape).unwrap();
        let drops = Cell::new(0);
        let notifications = Cell::new(0);
        ops.reset_counters();
        let (frame, ticket, error) = submit_retaining_frame(
            (handle, Tracked(&drops)),
            |frame| {
                ops.overwrite_artifact_linear_from_pinned_async(
                    &mut frame.0,
                    shape,
                    source.clone(),
                    None,
                )
            },
            || {
                notifications.set(notifications.get() + 1);
                Err(Error::Internal {
                    message: "injected notification failure after H2D".into(),
                })
            },
        )
        .unwrap_or_else(|(error, evidence)| panic!("{error}; {evidence:?}"));
        assert!(error.unwrap().to_string().contains("notification failure"));
        assert_eq!(notifications.get(), 1);
        assert_eq!(drops.get(), 0);
        assert!(
            !source.is_uniquely_owned(),
            "native ticket retains the actual DMA source"
        );
        ticket.synchronize().unwrap();
        assert!(ticket.is_complete().unwrap());
        assert_eq!(ops.counters().host_to_device_copies, 1);
        assert_eq!(ops.counters().host_to_device_bytes, 2048);
        assert_eq!(
            ops.counters().stream_wide_syncs,
            0,
            "exact event, not global sync"
        );
        drop(ticket);
        assert!(source.is_uniquely_owned());
        assert_eq!(drops.get(), 0, "event consumption is not frame recycling");
        drop(frame);
        assert_eq!(drops.get(), 1);
        assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
    }

    #[test]
    fn native_failure_is_unknown_and_never_recycles_frame_or_notifies() {
        let drops = Cell::new(0);
        let error = submit_retaining_frame(
            Tracked(&drops),
            |_| {
                Err::<(), _>(Error::Internal {
                    message: "submitted prefix failed".into(),
                })
            },
            || panic!("notification must not run after failed submission"),
        )
        .unwrap_err();
        assert_eq!(error.1, QuiescenceEvidence::Unknown);
        assert!(error.0.to_string().contains("submitted prefix failed"));
        assert_eq!(drops.get(), 0);
    }

    #[test]
    fn notification_failure_returns_original_event_and_frame_custody() {
        let frame_drops = Cell::new(0);
        let event_drops = Cell::new(0);
        let (frame, event, error) = submit_retaining_frame(
            Tracked(&frame_drops),
            |_| Ok(Tracked(&event_drops)),
            || {
                Err(Error::Internal {
                    message: "notification failure".into(),
                })
            },
        )
        .unwrap();
        assert!(error.unwrap().to_string().contains("notification failure"));
        assert_eq!((frame_drops.get(), event_drops.get()), (0, 0));
        drop((frame, event));
        assert_eq!((frame_drops.get(), event_drops.get()), (1, 1));
    }

    #[test]
    fn successful_submission_notifies_once_after_native_submission() {
        let calls = RefCell::new(Vec::new());
        let (frame, event, error) = submit_retaining_frame(
            17,
            |frame| {
                assert_eq!(*frame, 17);
                calls.borrow_mut().push("submit");
                Ok(23)
            },
            || {
                calls.borrow_mut().push("notify");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!((frame, event), (17, 23));
        assert!(error.is_none());
        assert_eq!(*calls.borrow(), ["submit", "notify"]);
    }
}
