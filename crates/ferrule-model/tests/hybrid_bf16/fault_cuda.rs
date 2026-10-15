//! Actual Mixed runner fault cells; compilation and CPU tests are not GPU evidence.
use crate as model;
#[path = "fault_contract.rs"]
mod contract;
#[allow(dead_code)]
mod fixture {
    use crate as ferrule_model;
    include!("fixture.rs");
}

use crate::decoder::*;

use crate::runner::{
    ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, ResidentModelRunner,
    TransactionEndIntent, TransactionEndProgress,
};
use crate::tokenizer::TokenizerHandle;
use crate::transformer::ExpertReadSchedule;
use contract::{CACHE_BYTES, CacheLedger, FaultKind, HOST_BYTES, PRECISION, REPORT_ENV, Report};
use ferrule_common::{Error, execution::*};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

type Runner = GenericDecoderRunner<HybridCudaDecoder>;

fn build(f: &fixture::Fixture, schedule: ExpertReadSchedule) -> (Runner, HybridCudaDevice, usize) {
    let options = contract::options(&f.resources, schedule);
    let estimate = HybridCudaMemoryEstimate::for_resources(&f.resources, &options, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 8,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    // An explicitly labelled allocator observation, NOT a passive snapshot.
    let allocation_baseline = device.operators().allocator_metrics().live_requested_bytes;
    let runner = Runner::hybrid_cuda(
        f.resources.clone(),
        TokenizerHandle::load(&f.dir).unwrap(),
        options,
        device.clone(),
    )
    .unwrap();
    let module = runner.forward_executor().module();
    assert_eq!(module.numeric_fp8_precision(), Some(PRECISION));
    assert!(module.numeric_bf16_profile().is_some());
    assert!(module.bf16_profile().is_none());
    assert!(module.bf16_chunk_workspace_bytes() > 0);
    assert_eq!(
        module.expert_read_pipeline_options().unwrap().schedule,
        schedule
    );
    (runner, device, allocation_baseline)
}

fn request(
    state: &CudaHybridSequenceState,
    continuation: bool,
) -> (ExecutionBatch, Vec<KvReservationView>) {
    let start = state.core().position();
    assert_eq!(start, usize::from(continuation));
    let (tokens, pages, newly, cow) = if continuation {
        (
            vec![2, 3],
            vec![KvPageId(2), KvPageId(3)],
            vec![KvPageId(3)],
            Some(KvCowReplacement {
                logical_page: 0,
                source: KvPageId(1),
                replacement: KvPageId(2),
            }),
        )
    } else {
        (vec![1], vec![KvPageId(1)], vec![KvPageId(1)], None)
    };
    let end = start + tokens.len();
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        tokens,
        (start..end).map(|p| p as u32).collect(),
        (start..end)
            .map(|p| Some(KvWriteSlot::new(pages[p / 2].0 * 2 + (p % 2) as u32)))
            .collect(),
        vec![LogitsRequest::Full; end - start],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..(end - start) as u32,
            start as u32,
            end as u32,
            0..pages.len() as u32,
        )],
        pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservations = vec![KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: start..end,
        newly_allocated: newly,
        generation: state.core().generation(),
        execution_generation: state.core().generation(),
        cow_replacement: cow,
    }];
    (batch, reservations)
}

fn finish(
    runner: &mut Runner,
    states: &mut [CudaHybridSequenceState],
    tx: ExecutionTransactionId,
    intent: TransactionEndIntent,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match runner.end_transaction(tx, states, intent).unwrap() {
            TransactionEndProgress::Complete => return,
            TransactionEndProgress::Pending => {
                assert!(Instant::now() < deadline, "component cleanup deadline");
                std::thread::yield_now();
            }
        }
    }
}

fn known_state_bits(state: &CudaHybridSequenceState) -> (Vec<u32>, Vec<u32>) {
    assert!(
        state.kv_state().layers().is_empty(),
        "no F32 history mirror"
    );
    let (conv, ssm) = state.kv_state().bf16_layers().unwrap()[0]
        .as_ref()
        .unwrap()
        .diagnostic_snapshot()
        .unwrap();
    assert!(
        conv.iter()
            .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
    );
    assert!(ssm.iter().all(|v| v.is_finite()));
    (
        conv.iter().map(|v| v.to_bits()).collect(),
        ssm.iter().map(|v| v.to_bits()).collect(),
    )
}

fn known_kv_bits(runner: &mut Runner) -> Vec<Vec<u16>> {
    // Explicit D2H/reload action. Never called through the passive sampler or on Unknown.
    runner.preempt_kv_pages(&[KvPageId(1)]).unwrap();
    let snapshot = runner
        .backend()
        .pool()
        .typed_host_snapshot(KvPageId(1))
        .unwrap();
    let bits = snapshot
        .planes
        .iter()
        .map(|plane| {
            assert_eq!(plane.element_type(), KvElementType::Bf16);
            assert!(plane.f32_bits().is_none());
            // Only the one committed row is semantic state; the unused page tail is not.
            plane.bf16_words().unwrap()[..8].to_vec()
        })
        .collect();
    runner.restore_kv_pages(&[KvPageId(1)]).unwrap();
    bits
}

fn passive(runner: &Runner, device: &HybridCudaDevice, state: &CudaHybridSequenceState) -> Value {
    let module = runner.forward_executor().module();
    let reads = module.expert_read_pipeline_observation();
    let ledger = CacheLedger::checked(module.expert_cache_stats().unwrap(), CACHE_BYTES).unwrap();
    assert!(contract::host_ledger_valid(
        reads.current_charge,
        HOST_BYTES
    ));
    let logical = runner.observability_snapshot();
    json!({
        "transaction_under_test": 2,
        "committed": {"topology": state.topology_id().get(), "generation": state.core().generation(),
            "frontier": state.core().position(), "state_bytes": state.kv_state().resident_bytes(),
            "gdn_position": state.kv_state().bf16_layers().unwrap()[0].as_ref().unwrap().position()},
        "live_state_bytes_including_default_committed_working": device.live_state_bytes(),
        "image": module.test_fault_owner_observation(),
        "host": reads, "cache_ledger": ledger,
        "runner": {"active_transactions": logical.active_transactions, "commits": logical.commits,
            "aborts": logical.aborts, "completed_batches": logical.completed_batches},
        "kv": {"active_transactions": runner.backend().pool().active_transaction_count(),
            "free_slots": runner.backend().pool().free_slot_count(),
            "committed_page_slot": runner.backend().pool().physical_slot(KvPageId(1))},
        "device_quarantined": device.needs_quarantine(),
        "physical_owner": runner.physical_resource_snapshot(),
        "allocator": "unobserved_passive_getter_must_not_poll_retirement_fences",
        "state_contents": "unobserved_by_passive_sampler"
    })
}

fn sample(
    report: &mut Report,
    label: &str,
    runner: &Runner,
    device: &HybridCudaDevice,
    state: &CudaHybridSequenceState,
) -> Value {
    let before = device.operators().counters();
    let first = passive(runner, device, state);
    assert_eq!(first, passive(runner, device, state));
    assert_eq!(
        device.operators().counters(),
        before,
        "sampling must not submit/query/synchronize CUDA"
    );
    report
        .record(json!({"record": "passive_snapshot", "label": label, "value": first}))
        .unwrap();
    first
}

fn action(report: &mut Report, name: &str) {
    report
        .record(json!({"record": "physical_action_begin", "action": name}))
        .unwrap();
}
fn is_unknown(error: &Error) -> bool {
    ferrule_common::error::ErrorChain::new(error)
        .downcast_ref::<CompletionUnknown>()
        .is_some()
}
fn assert_unknown(error: &Error) {
    assert!(
        is_unknown(error),
        "expected typed CompletionUnknown, got {error:?}"
    );
}

fn run(kind: FaultKind) {
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let dir = std::env::var_os(REPORT_ENV).map(PathBuf::from);
    let mut report = Report::new(kind, dir.as_deref()).unwrap();
    for (schedule, name) in [
        (ExpertReadSchedule::Blocking, "blocking"),
        (ExpertReadSchedule::Lookahead, "lookahead"),
    ] {
        report.record(json!({"record": "case_begin", "schedule": name, "gpu_execution": "running_not_yet_proved"})).unwrap();
        let f = fixture::Fixture::numeric_profile();
        action(
            &mut report,
            "prepare_real_Mixed_CUDA_image_and_measure_allocator_baseline",
        );
        let (mut runner, device, allocator_baseline) = build(&f, schedule);
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let tx1 = ExecutionTransactionId::new(1).unwrap();
        let tx2 = ExecutionTransactionId::new(2).unwrap();
        let (prime, reservations) = request(&states[0], false);
        action(&mut report, "prepare_execute_publish_one_real_token_tx1");
        runner
            .prepare_multi_session_batch(tx1, &mut states, &prime, &reservations)
            .unwrap();
        let MultiSessionBatchProgress::Complete(output) = runner
            .execute_multi_session_batch_progress(tx1, &mut states, &prime)
            .unwrap()
        else {
            panic!("synchronous Mixed runner unexpectedly waits")
        };
        assert!(output.logits.iter().all(|row| matches!(&row.logits, LogitsOutput::Full(v) if v.len() == 32 && v.iter().all(|x| x.is_finite()))));
        finish(&mut runner, &mut states, tx1, TransactionEndIntent::Publish);
        assert_eq!(states[0].core().position(), 1);
        assert_eq!(states[0].kv_state().resident_bytes(), 512);
        let core = states[0].core().clone();
        let topology = states[0].topology_id();
        action(
            &mut report,
            "healthy_committed_GDN_download_and_KV_preempt_restore_baseline",
        );
        let state_bits = known_state_bits(&states[0]);
        let kv_bits = known_kv_bits(&mut runner);
        assert!(state_bits.0.iter().chain(&state_bits.1).any(|&b| b != 0));
        let baseline = sample(
            &mut report,
            "committed_frontier_one",
            &runner,
            &device,
            &states[0],
        );
        let state_baseline = device.live_state_bytes();
        let counters_before = device.operators().counters();
        let reads_before = runner
            .forward_executor()
            .module()
            .expert_read_pipeline_stats();
        let cache_before = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        let (batch, reservations) = request(&states[0], true);
        action(
            &mut report,
            "prepare_tx2_two_token_continuation_with_KV_COW",
        );
        runner
            .prepare_multi_session_batch(tx2, &mut states, &batch, &reservations)
            .unwrap();
        runner
            .forward_executor()
            .module()
            .test_install_routed_down_fault(kind == FaultKind::Unknown);
        sample(
            &mut report,
            "prepared_before_submit",
            &runner,
            &device,
            &states[0],
        );
        action(
            &mut report,
            "execute_tx2_scheduled_error_after_real_layer1_routed_down",
        );
        let error = runner
            .execute_multi_session_batch_progress(tx2, &mut states, &batch)
            .unwrap_err();
        assert_eq!(is_unknown(&error), kind == FaultKind::Unknown, "{error:?}");
        assert!(error.to_string().contains("scheduled fixture fault"));
        let module = runner.forward_executor().module();
        let (submission, before_proof, physical_proof) = module.test_fault_boundary_observation();
        assert!(physical_proof.compute_event_completed && physical_proof.upload_sync_completed);
        assert_eq!(
            physical_proof.deliberately_discarded,
            kind == FaultKind::Unknown
        );
        assert_eq!(
            before_proof.admitted_expert,
            Some((
                before_proof.image_generation,
                submission.layer,
                submission.expert
            ))
        );
        assert_eq!(submission.reads.requested_mode, Some(schedule));
        assert_eq!(submission.reads.effective_mode, Some(schedule));
        assert!(contract::host_ledger_valid(
            submission.reads.current_charge,
            HOST_BYTES
        ));
        assert!(submission.reads.current_charge.host_charged_bytes > 0);
        assert_eq!(
            submission
                .reads
                .last_receipt
                .unwrap()
                .identity
                .image_generation,
            before_proof.image_generation
        );
        assert_eq!(submission.reads.last_receipt.unwrap().identity.layer, 1);
        assert!(before_proof.bf16_holds > 0 && before_proof.route_holds > 0);
        assert!(before_proof.cache.charged_bytes > 0);
        assert!(module.expert_cache_stats().unwrap().uploads > cache_before.uploads);
        report.record(json!({"record": "fault_boundary_before_consumer_proof", "schedule": name,
            "transaction": 2, "submission": submission, "owner": before_proof,
            "physical_proof_before_test_discards_receipt": physical_proof,
            "error_kind": if is_unknown(&error) { "CompletionUnknown" } else { "known_operation_failure" },
            "faults_observed": 1})).unwrap();
        let after_fault = sample(
            &mut report,
            "after_failure_before_abort",
            &runner,
            &device,
            &states[0],
        );
        assert_eq!(states[0].core(), &core);
        assert_eq!(states[0].topology_id(), topology);
        assert_eq!(after_fault["committed"], baseline["committed"]);
        let failed_counters = device.operators().counters();
        let read_delta = module
            .expert_read_pipeline_stats()
            .checked_delta(reads_before)
            .unwrap();
        assert!(read_delta.proved_read_bytes > 0 && read_delta.worker_started_reads > 0);
        assert_eq!(
            read_delta.unknown_reads, 0,
            "device proof loss is not host read Unknown"
        );
        assert_eq!(read_delta.sequences_started, read_delta.sequences_finished);
        assert_eq!(read_delta.submitted_reads, read_delta.completed_reads);
        assert_eq!(read_delta.join_pending_sequences, 0);
        assert_eq!(read_delta.join_failed_sequences, 0);
        assert_eq!(read_delta.unobserved_terminal_reads, 0);
        let host = module.expert_read_pipeline_observation();
        let read_options = module.expert_read_pipeline_options().unwrap();
        contract::worker_contract::assert_idle_retained(&host, read_options);
        assert_eq!(host.current_charge.host_charged_bytes, 1280 << 10);
        assert!(host.stats.worker_reuses > 0);
        assert_eq!(read_delta.workers_created, 0);
        assert_eq!(read_delta.workers_joined, 0);
        assert_eq!(read_delta.worker_reuses, read_delta.sequences_started);
        if schedule == ExpertReadSchedule::Lookahead {
            assert!(submission.reads.pending_read.is_some());
            assert!(read_delta.cancelled_reads > 0);
            assert!(host.last_receipt.unwrap().cancellation_requested);
        } else {
            assert!(submission.reads.pending_read.is_none());
            assert_eq!(read_delta.speculative_reads, 0);
        }
        report
            .record(json!({"record": "host_cleanup_receipt", "schedule": name,
            "action": "wait_take_release_payload_keep_worker",
            "stage": "sequence_finished_materializer_alive",
            "charge_basis": "logical_host_reservation_not_physical_rss",
            "worker_lifetime": host,
            "owner_drop_observed": false,
            "offworker_join_observed": false,
            "submitted_delta": read_delta.submitted_reads,
            "completed_delta": read_delta.completed_reads,
            "cancelled_delta": read_delta.cancelled_reads,
            "sequences_finished_delta": read_delta.sequences_finished,
            "host_charge_after": host.current_charge,
            "last_receipt": host.last_receipt,
            "device_release_authority": false}))
            .unwrap();
        let after_cleanup;
        if kind == FaultKind::Clean {
            assert!(!device.needs_quarantine());
            action(
                &mut report,
                "abort_tx2_consume_real_completion_and_release_transient_working_state",
            );
            finish(&mut runner, &mut states, tx2, TransactionEndIntent::Abort);
            after_cleanup = sample(
                &mut report,
                "clean_abort_complete",
                &runner,
                &device,
                &states[0],
            );
            assert_eq!(after_cleanup["committed"], baseline["committed"]);
            assert_eq!(
                device.live_state_bytes(),
                state_baseline,
                "default plus committed state remain, not zero"
            );
            assert_eq!(runner.observability_snapshot().active_transactions, 0);
            assert_eq!(runner.backend().pool().active_transaction_count(), 0);
            assert_eq!(after_cleanup["kv"], baseline["kv"]);
            let owner = runner
                .forward_executor()
                .module()
                .test_fault_owner_observation();
            assert!(!owner.quarantined);
            assert_eq!(
                (
                    owner.bf16_holds,
                    owner.f32_holds,
                    owner.route_holds,
                    owner.artifact_holds
                ),
                (0, 0, 0, 0)
            );
            assert_eq!(owner.cache.pending_upload_bytes, 0);
            assert_eq!(owner.cache.unknown_quarantine_bytes, 0);
            assert_eq!(
                owner.cache.scratch_bytes, 4096,
                "persistent numeric reservation is not an operation leak"
            );
            assert_eq!(
                owner.chunk_workspace_bytes,
                before_proof.chunk_workspace_bytes
            );
            action(
                &mut report,
                "verify_unchanged_committed_GDN_bytes_and_KV_bytes_after_clean_abort",
            );
            assert_eq!(known_state_bits(&states[0]), state_bits);
            assert_eq!(known_kv_bits(&mut runner), kv_bits);
            action(
                &mut report,
                "retry_same_continuation_tx3_then_abort_without_fault",
            );
            runner.forward_executor().module().clear_diagnostic_trace();
            let tx3 = ExecutionTransactionId::new(3).unwrap();
            runner
                .prepare_multi_session_batch(tx3, &mut states, &batch, &reservations)
                .unwrap();
            assert!(matches!(
                runner
                    .execute_multi_session_batch_progress(tx3, &mut states, &batch)
                    .unwrap(),
                MultiSessionBatchProgress::Complete(_)
            ));
            finish(&mut runner, &mut states, tx3, TransactionEndIntent::Abort);
            assert_eq!(states[0].core(), &core);
        } else {
            assert_unknown(&error);
            assert!(device.needs_quarantine());
            let charged = device.live_state_bytes();
            assert!(
                charged > state_baseline,
                "unpublished working state retained"
            );
            let frozen = runner
                .forward_executor()
                .module()
                .test_fault_owner_observation();
            assert!(frozen.quarantined && frozen.cache.quarantined);
            assert_eq!(
                frozen.cache.unknown_quarantine_bytes,
                frozen.cache.charged_bytes
            );
            assert!(frozen.bf16_holds > 0 && frozen.route_holds > 0);
            for _ in 0..2 {
                action(
                    &mut report,
                    "independent_KV_compute_and_upload_events_synchronize_not_owner_release",
                );
                device
                    .operators()
                    .record_compute_event()
                    .unwrap()
                    .synchronize()
                    .unwrap();
                device
                    .operators()
                    .record_upload_event()
                    .unwrap()
                    .synchronize()
                    .unwrap();
                action(&mut report, "abort_tx2_must_return_typed_sticky_Unknown");
                assert_unknown(
                    &runner
                        .end_transaction(tx2, &mut states, TransactionEndIntent::Abort)
                        .unwrap_err(),
                );
                assert_eq!(
                    runner
                        .forward_executor()
                        .module()
                        .test_fault_owner_observation(),
                    frozen
                );
                assert_eq!(device.live_state_bytes(), charged);
                assert_eq!(runner.observability_snapshot().active_transactions, 1);
                assert_eq!(runner.backend().pool().active_transaction_count(), 1);
            }
            action(
                &mut report,
                "ordinary_reset_sequence_admission_KV_release_and_publish_must_not_discharge_Unknown",
            );
            assert!(runner.reset_sequence_state(&mut states[0]).is_err());
            assert!(runner.reset_session().is_err());
            assert!(runner.create_sequence_state().is_err());
            assert!(
                runner
                    .release_kv_pages(&[KvPageId(1), KvPageId(2), KvPageId(3)])
                    .is_err()
            );
            assert!(
                runner
                    .end_transaction(tx2, &mut states, TransactionEndIntent::Publish)
                    .is_err()
            );
            assert!(runner.shutdown().is_err());
            assert_eq!(states[0].core(), &core);
            assert_eq!(
                runner
                    .forward_executor()
                    .module()
                    .test_fault_owner_observation(),
                frozen
            );
            after_cleanup = sample(
                &mut report,
                "unknown_retained_after_independent_fences_and_rejected_reset",
                &runner,
                &device,
                &states[0],
            );
            assert_eq!(after_cleanup["committed"], baseline["committed"]);
            assert_eq!(after_cleanup["kv"], after_fault["kv"]);
            assert_eq!(device.live_state_bytes(), charged);
            // No post-Unknown device reads or forged bitwise-state proof. The
            // committed topology/core and retained transaction are still observable.
        }
        let evidence = json!({
            "real_compute_launch_delta": failed_counters.compute_kernel_launches - counters_before.compute_kernel_launches,
            "real_h2d_copy_delta": failed_counters.host_to_device_copies - counters_before.host_to_device_copies,
            "artifact_upload_delta": failed_counters.artifact_uploads - counters_before.artifact_uploads,
            "artifact_upload_byte_delta": failed_counters.artifact_upload_bytes - counters_before.artifact_upload_bytes,
            "cuda_counter_scope": "whole_tx2_owner_not_per_read_ticket_attribution",
            "real_read_byte_delta": read_delta.proved_read_bytes,
            "read_statistics_delta": read_delta, "faults_observed": 1,
            "committed_frontier_before": core.position(), "committed_frontier_after": states[0].core().position(),
            "committed_state_content_check": if kind == FaultKind::Clean { "GDN_and_KV_bitwise_unchanged" } else { "unobserved_post_Unknown_no_device_read_or_release_authority" },
            "result": if kind == FaultKind::Clean { "clean_abort_transients_released_retry_succeeded" } else { "typed_Unknown_sticky_host_payloads_released_worker_idle_device_custody_retained" },
            "after_cleanup": after_cleanup
        });
        if kind == FaultKind::Clean {
            let before_witness = runner
                .forward_executor()
                .module()
                .expert_read_pipeline_observation();
            contract::worker_contract::assert_idle_retained(&before_witness, read_options);
            action(
                &mut report,
                "test_only_empty_sequence_checkout_capture_existing_join_witness_no_reads",
            );
            let witness = runner
                .forward_executor()
                .module()
                .test_read_owner_release_witness();
            let idle_before_drop = witness.snapshot();
            contract::worker_contract::assert_idle_retained(&idle_before_drop, read_options);
            report
                .record(
                    json!({"record": "host_final_owner_witness_capture", "schedule": name,
                "physical_action": "empty_sequence_checkout_keep_same_worker",
                "before": before_witness, "after": idle_before_drop,
                "delta": idle_before_drop.stats.checked_delta(before_witness.stats).unwrap(),
                "adds_sequences": 1, "adds_worker_reuses": 1, "adds_workers_created": 0,
                "adds_submitted_reads": 0, "adds_workers_joined": 0}),
                )
                .unwrap();
            action(
                &mut report,
                "release_committed_page_and_explicit_state_then_shutdown_image",
            );
            runner.release_kv_pages(&[KvPageId(1)]).unwrap();
            runner
                .try_release_sequence_state(states.pop().unwrap())
                .unwrap();
            runner.shutdown().unwrap();
            assert_eq!(
                device.live_state_bytes(),
                512,
                "default state lives until runner Drop"
            );
            assert_eq!(
                witness.snapshot(),
                idle_before_drop,
                "shutdown is not materializer Drop"
            );
            let ops = device.operators().clone();
            action(
                &mut report,
                "drop_final_runner_materializer_owner_offworker_then_observe_existing_join_witness",
            );
            drop(runner);
            let host_after_drop = witness.after_owner_drop();
            contract::worker_contract::assert_final_joined(&host_after_drop, &idle_before_drop);
            assert_eq!(
                witness.after_owner_drop(),
                host_after_drop,
                "passive witness is stable"
            );
            report
                .record(
                    json!({"record": "host_final_owner_release", "schedule": name,
                "stage": "final_materializer_owner_dropped_worker_joined",
                "owner_drop_observed": true, "offworker_join_observed": true,
                "host": host_after_drop, "device_release_authority": false,
                "charge_basis": "logical_host_reservation_not_physical_rss"}),
                )
                .unwrap();
            drop(device);
            action(
                &mut report,
                "final_owner_drop_then_explicit_allocator_retirement_trim_and_measure",
            );
            ops.trim_device_allocator().unwrap();
            let reclaimed = ops.allocator_metrics().live_requested_bytes;
            assert_eq!(reclaimed, allocator_baseline);
            report.record(json!({"record": "clean_final_owner_release", "allocator_requested_baseline": allocator_baseline,
                "allocator_requested_after_drop_trim": reclaimed, "scope": "this_fixture_owner_not_device_wide_free_memory"})).unwrap();
        } else {
            let host_retained = runner
                .forward_executor()
                .module()
                .expert_read_pipeline_observation();
            contract::worker_contract::assert_idle_retained(&host_retained, read_options);
            assert_eq!(host_retained.stats, host.stats);
            report.record(json!({"record": "unknown_retention", "action": "retain_runner_states_device_until_process_exit",
                "host_stage": "payloads_released_worker_idle_retained_with_materializer",
                "host": host_retained, "host_after_final_owner_drop": null,
                "host_final_join_witness": null,
                "forced_allocator_zero": false, "owner_release": "unobserved_not_authorized"})).unwrap();
            std::mem::forget(runner);
            std::mem::forget(states);
            std::mem::forget(device);
        }
        report.observed_case(name, evidence).unwrap();
    }
    report.seal().unwrap();
}

#[test]
#[ignore = "sole GPU owner only; actual Mixed BF16 runner/read/upload/down error, Blocking and Lookahead"]
fn mixed_bf16_clean_fault_after_real_expert_submit_both_schedules() {
    run(FaultKind::Clean);
}

#[test]
#[ignore = "sole GPU owner only; actual Mixed BF16 work then lost consumer proof, sticky typed Unknown in both schedules"]
fn mixed_bf16_unknown_after_real_expert_submit_both_schedules() {
    run(FaultKind::Unknown);
}
