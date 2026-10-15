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

// This witness owns only the existing observation/handoff, not the
// materializer or any device resource. Reading it never joins or refunds.
pub(crate) struct ReadOwnerReleaseWitness {
    owner: std::sync::Weak<crate::transformer::StateDictMaterializer>,
    observation:
        Arc<std::sync::Mutex<crate::transformer::expert_reads::ExpertReadObservationState>>,
    handoff: crate::checkpoint::read_pipeline::ReadJoinHandoff,
}
impl ReadOwnerReleaseWitness {
    fn capture(materializer: &Arc<crate::transformer::StateDictMaterializer>, image: u64) -> Self {
        let before = materializer.expert_read_pipeline_observation();
        assert_eq!(before.status, "idle_retained");
        assert!(before.worker_alive && !before.worker_unknown && !before.worker_join_pending);
        let sequence = materializer
            .expert_read_sequence(image, &[])
            .unwrap()
            .unwrap();
        let witness = ReadOwnerReleaseWitness {
            owner: Arc::downgrade(materializer),
            observation: sequence.stats_sink.as_ref().unwrap().clone(),
            handoff: sequence.observation_test_handoff(),
        };
        drop(sequence);
        let mut expected = before.stats;
        expected.worker_reuses += 1;
        expected.sequences_started += 1;
        expected.sequences_finished += 1;
        let after = witness.snapshot();
        assert_eq!(after.stats, expected);
        assert_eq!(after.current_charge, before.current_charge);
        assert!(after.worker_alive && !after.worker_join_pending && !after.worker_unknown);
        witness
    }

    pub(crate) fn snapshot(&self) -> ExpertReadPipelineObservation {
        let view = self.observation.lock().unwrap().clone();
        view.snapshot()
    }

    pub(crate) fn after_owner_drop(&self) -> ExpertReadPipelineObservation {
        assert!(
            self.owner.upgrade().is_none(),
            "actual final materializer owner must be gone"
        );
        let worker = self.handoff.observe_worker();
        assert!(worker.joined && !worker.alive && !worker.join_pending);
        assert!(!worker.join_failed && !worker.poisoned && !worker.custody_unknown);
        let budget = self.handoff.observe_budget();
        assert!(!budget.ledger_poisoned);
        assert_eq!(budget.budget.requested_host_reservation_bytes(), 0);
        assert_eq!(budget.budget.reserved_slots, 0);
        self.snapshot()
    }
}

impl CudaHybridModule {
    /// Explicit test action, NOT a passive sampler: obtain the existing
    /// worker's handoff via one empty sequence. Account for its reuse and
    /// sequence counters; it creates no worker and submits no checkpoint IO.
    pub(crate) fn test_read_owner_release_witness(&self) -> ReadOwnerReleaseWitness {
        ReadOwnerReleaseWitness::capture(&self.materializer, self.operators.image_generation())
    }

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

#[test]
fn cpu_only_read_release_witness_requires_final_materializer_drop() {
    use crate::transformer::ExpertReadSchedule;
    use crate::transformer::expert_reads::tests::{Fixture, IMAGE};
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let fixture = Fixture::new(1);
        let materializer = Arc::new(fixture.materializer(schedule));
        for _ in 0..2 {
            let mut sequence = materializer
                .expert_read_sequence(IMAGE, &[&fixture.metadata[0]])
                .unwrap()
                .unwrap();
            sequence.demand(0, 0).unwrap();
            drop(sequence);
        }
        let witness = ReadOwnerReleaseWitness::capture(&materializer, IMAGE);
        let idle = witness.snapshot();
        assert_eq!(idle.status, "idle_retained");
        assert_eq!(idle.current_charge.host_charged_bytes, 1280 << 10);
        assert_eq!(
            (
                idle.stats.workers_created,
                idle.stats.worker_reuses,
                idle.stats.workers_joined
            ),
            (1, 2, 0)
        );
        assert_eq!(
            idle.stats.submitted_reads, 2,
            "empty witness checkout submits no IO"
        );
        let final_owner = Arc::clone(&materializer);
        drop(materializer);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| witness.after_owner_drop()))
                .is_err()
        );
        assert_eq!(
            witness.snapshot(),
            idle,
            "observer must not join a live owner's worker"
        );
        drop(final_owner);
        let ended = witness.after_owner_drop();
        assert_eq!(ended.status, "closed");
        assert_eq!(ended.current_charge, Default::default());
        assert_eq!(ended.stats.workers_joined, 1);
        assert_eq!(ended.stats.workers_created, 1);
        assert_eq!(ended.stats.worker_reuses, 2);
        assert_eq!(witness.after_owner_drop(), ended);
    }
}
