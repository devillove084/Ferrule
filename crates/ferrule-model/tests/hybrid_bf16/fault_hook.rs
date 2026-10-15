//! Private test-only observations at the existing native projection/consumer seam.
//! No new execution path, synchronization, worker polling or public fault switch.
use super::*;
use crate::resource_observability::ExpertCacheObservation;
use crate::transformer::ExpertReadPipelineObservation;
use crate::transformer::standard::cuda::diagnostic::{CudaDiagnosticBuffer, CudaDiagnosticLimits};
use serde::Serialize;
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct OwnerObservation {
    pub image_generation: u64,
    pub admitted_expert: Option<(u64, usize, usize)>,
    pub cache: ExpertCacheObservation,
    pub numeric_workspace: Option<(usize, usize, u64, u64, u64)>,
    pub chunk_workspace_bytes: usize,
    pub bf16_holds: usize,
    pub f32_holds: usize,
    pub route_holds: usize,
    pub artifact_holds: usize,
    pub consumer_proof_attempts: usize,
    pub quarantined: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SubmissionObservation {
    pub layer: usize,
    pub expert: usize,
    pub bf16_elements: usize,
    pub input_compute_launches: u64,
    pub output_compute_launches: u64,
    pub input_h2d_copies: u64,
    pub output_h2d_copies: u64,
    pub reads: ExpertReadPipelineObservation,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PhysicalProofObservation {
    pub compute_event_completed: bool,
    pub upload_sync_completed: bool,
    pub deliberately_discarded: bool,
}

#[derive(Default)]
pub(crate) struct FaultProbe {
    // 0 disarmed, 1 ordinary operation error, 2 discard the next real proof.
    arm: Rc<Cell<u8>>,
    before_proof: Rc<RefCell<Option<OwnerObservation>>>,
    submission: Rc<RefCell<Option<SubmissionObservation>>>,
    physical_proof: RefCell<Option<PhysicalProofObservation>>,
}

impl CudaStandardDecoderOperators {
    fn test_owner_observation(&self) -> OwnerObservation {
        OwnerObservation {
            image_generation: self.image_generation(),
            admitted_expert: self
                .bindings
                .expert_admission
                .as_ref()
                .map(|key| (key.image_generation, key.layer, key.expert)),
            cache: self.expert_cache_stats().expect("bounded fixture").into(),
            numeric_workspace: self.numeric_workspace_stats().map(|s| {
                (
                    s.reserved_bytes,
                    s.allocated_bytes,
                    s.allocations,
                    s.reuses,
                    s.submissions,
                )
            }),
            chunk_workspace_bytes: self.bf16_chunk_workspace_bytes(),
            bf16_holds: self.bf16_hold.len(),
            f32_holds: self.expert_hold.len(),
            route_holds: self.route_hold.len(),
            artifact_holds: self.expert_artifacts.len(),
            consumer_proof_attempts: self.consumer_proofs,
            quarantined: self.needs_quarantine(),
        }
    }

    pub(crate) fn observe_test_consumer_boundary(&self) -> bool {
        let mode = self.fault_probe.arm.replace(0);
        if mode == 0 {
            return false;
        }
        assert!(
            self.fault_probe.before_proof.borrow().is_none(),
            "one-shot fault"
        );
        *self.fault_probe.before_proof.borrow_mut() = Some(self.test_owner_observation());
        mode == 2
    }

    pub(crate) fn observe_test_physical_proof(&self, discarded: bool, compute: bool, upload: bool) {
        if self.fault_probe.before_proof.borrow().is_some()
            && self.fault_probe.physical_proof.borrow().is_none()
        {
            *self.fault_probe.physical_proof.borrow_mut() = Some(PhysicalProofObservation {
                compute_event_completed: compute,
                upload_sync_completed: upload,
                deliberately_discarded: discarded,
            });
        }
    }
}

impl CudaHybridModule {
    pub(crate) fn test_fault_owner_observation(&self) -> OwnerObservation {
        self.operators.test_owner_observation()
    }

    pub(crate) fn test_fault_boundary_observation(
        &self,
    ) -> (
        SubmissionObservation,
        OwnerObservation,
        PhysicalProofObservation,
    ) {
        (
            self.operators
                .fault_probe
                .submission
                .borrow()
                .clone()
                .expect("real routed down observed"),
            self.operators
                .fault_probe
                .before_proof
                .borrow()
                .clone()
                .expect("consumer seam reached"),
            self.operators
                .fault_probe
                .physical_proof
                .borrow()
                .expect("real consumer proof attempted"),
        )
    }

    pub(crate) fn test_install_routed_down_fault(&self, lose_proof: bool) {
        let probe = &self.operators.fault_probe;
        assert!(probe.submission.borrow().is_none());
        let arm = Rc::clone(&probe.arm);
        let submission = Rc::clone(&probe.submission);
        let materializer = Arc::clone(&self.materializer);
        let mut input_counters = None;
        self.operators.set_native_diagnostic_trace(
            CudaDiagnosticLimits::new(1 << 20).unwrap(),
            |d| d.layer == Some(1) && matches!(d.name, "linear.input" | "linear.output"),
            move |event| {
                let Some(parameter) = event.parameter else {
                    return Ok(());
                };
                if parameter.role() != &crate::TensorRole::RoutedExpertDown
                    || submission.borrow().is_some()
                {
                    return Ok(());
                }
                let crate::nn::ParameterResidency::Expert { layer, expert } =
                    *parameter.residency()
                else {
                    panic!("routed parameter identity")
                };
                assert_eq!(layer, 1);
                let counters = event.operators.counters();
                if event.descriptor.name == "linear.input" {
                    input_counters = Some(counters);
                    return Ok(());
                }
                let before = input_counters.take().expect("same real projection input");
                let CudaDiagnosticBuffer::Bf16(buffer) = event.buffer else {
                    panic!("Mixed routed output must be actual BF16 storage")
                };
                assert!(counters.compute_kernel_launches > before.compute_kernel_launches);
                let reads = materializer.expert_read_pipeline_observation();
                let receipt = reads.last_receipt.expect("actual worker read receipt");
                assert!(reads.worker_read_observed);
                assert_eq!(
                    (receipt.identity.layer, receipt.identity.expert),
                    (layer, expert)
                );
                assert_eq!(receipt.outcome, "source_verified");
                assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
                assert!(receipt.taken_ns.is_some());
                *submission.borrow_mut() = Some(SubmissionObservation {
                    layer,
                    expert,
                    bf16_elements: buffer.len(),
                    input_compute_launches: before.compute_kernel_launches,
                    output_compute_launches: counters.compute_kernel_launches,
                    input_h2d_copies: before.host_to_device_copies,
                    output_h2d_copies: counters.host_to_device_copies,
                    reads,
                });
                arm.set(if lose_proof { 2 } else { 1 });
                Err(cuda_error(
                    "scheduled fixture fault after actual Mixed BF16 routed down submission",
                ))
            },
        );
    }
}
