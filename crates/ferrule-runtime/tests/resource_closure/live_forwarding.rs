use ferrule_model as model;

#[path = "fixture.rs"]
mod fixture;

use ferrule_runtime::FixedSequenceSlotPool;
use ferrule_runtime::engine::{ResidentResourceShutdownSummary, ResidentTopKDriver};
use fixture::{Control, runner, runner_with_plan};
use model::ResidentModelRunner;
use model::resource_observability::*;
use model::transformer::{
    ExpertCacheStats, ExpertReadHostCharge, ExpertReadHostReceipt, ExpertReadIdentity,
    ExpertReadPipelineObservation, ExpertReadPipelineOptions, ExpertReadPipelineStats,
    ExpertReadSchedule, TransformerModule,
};
use std::rc::Rc;

#[test]
fn actual_runner_executor_module_chain_reads_allocations_and_identity_without_progress() {
    let control = Rc::new(Control::default());
    let runner = runner(control.clone());
    let before = runner.physical_resource_snapshot();
    assert_eq!(before.image_generation, ResourceEvidence::Observed(47));
    assert_eq!(before.prepared_plan_id, ResourceEvidence::Observed(41));
    assert_eq!(before.scope.value().unwrap().generation.get(), 41);
    let state = before.default_state.value().unwrap();
    assert_eq!(
        state.topology_id,
        runner.default_state().topology_id().get()
    );
    assert_eq!(state.generation, runner.default_state().core().generation());
    assert_eq!(state.resident_bytes, ResourceEvidence::Observed(24));
    assert_eq!(
        before.live_model_state_bytes,
        ResourceEvidence::Observed(24)
    );
    let cache = before.expert_cache.value().unwrap();
    assert_eq!(
        (
            cache.resident_bytes,
            cache.pending_upload_bytes,
            cache.scratch_bytes
        ),
        (64, 32, 16)
    );
    assert_eq!(cache.charged_bytes, 112);
    assert_eq!(cache.unknown_quarantine_bytes, 0);
    assert!(!cache.counters_overflowed);
    assert_eq!(cache.diagnostics, ResourceEvidence::Unobserved);
    let workspace = before.numeric_workspace.value().unwrap();
    assert_eq!(
        (workspace.image_generation, workspace.kind),
        (47, "numeric")
    );
    assert_eq!(workspace.allocated_bytes, cache.scratch_bytes);
    assert_eq!(before.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(before.release, ResourceEvidence::Unobserved);
    assert_eq!(before.allocator, ResourceEvidence::Unobserved);
    assert_eq!(before.boundary_transfer, ResourceEvidence::Unobserved);
    for _ in 0..3 {
        assert_eq!(runner.physical_resource_snapshot(), before);
    }
    assert_eq!(control.reads.get(), 4);
    assert_eq!(control.shutdown_calls.get(), 0);
    assert_eq!(runner.forward_executor().arena_stats().installs, 0);
}

#[test]
fn authenticated_quiescence_keeps_live_image_state_and_workspace_retained() {
    let control = Rc::new(Control::default());
    let mut runner = runner(control.clone());
    let before = runner.physical_resource_snapshot();
    runner.shutdown().unwrap();
    let after = runner.physical_resource_snapshot();
    assert_eq!(after.scope, before.scope);
    assert_eq!(after.default_state, before.default_state);
    assert_eq!(after.expert_cache, before.expert_cache);
    assert_eq!(after.numeric_workspace, before.numeric_workspace);
    assert_eq!(
        after.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(
        after.quiescence,
        ResourceEvidence::Verified(ResourceQuiescence::Quiescent)
    );
    assert_eq!(
        after.residency,
        ResourceEvidence::Observed(ResourceResidency::Retained)
    );
    assert_eq!(after.release, ResourceEvidence::Unobserved);
    runner.shutdown().unwrap();
    assert_eq!(control.shutdown_calls.get(), 1);
    assert_eq!(runner.physical_resource_snapshot(), after);
}

#[test]
fn failed_close_is_sticky_across_a_later_authenticated_mock_fence() {
    let control = Rc::new(Control::default());
    control.failures.set(1);
    let mut runner = runner(control.clone());
    let before = runner.physical_resource_snapshot();
    assert!(runner.shutdown().is_err());
    runner.shutdown().unwrap();
    for _ in 0..2 {
        let after = runner.physical_resource_snapshot();
        assert_eq!(after.scope, before.scope);
        assert_eq!(after.close, ResourceEvidence::Unknown);
        assert_eq!(after.quiescence, ResourceEvidence::Unknown);
        assert_eq!(after.release, ResourceEvidence::Unknown);
        assert_eq!(after.expert_cache, before.expert_cache);
        assert_eq!(after.numeric_workspace, before.numeric_workspace);
    }
    assert_eq!(control.shutdown_calls.get(), 2);
}

#[test]
fn unknown_cache_charge_is_a_subset_not_an_additional_charge() {
    let control = Rc::new(Control::default());
    control.failures.set(1);
    control.lose_completion.set(true);
    let mut runner = runner(control.clone());
    assert!(runner.shutdown().is_err());
    let after = runner.physical_resource_snapshot();
    let cache = after.expert_cache.value().unwrap();
    assert_eq!(cache.charged_bytes, 112);
    assert_eq!(cache.unknown_quarantine_bytes, cache.charged_bytes);
    assert!(cache.quarantined);
    assert_eq!(after.quiescence, ResourceEvidence::Unknown);
    assert_eq!(after.release, ResourceEvidence::Unknown);
    // Even a test owner that violates sticky health cannot erase the retained
    // decoder close ambiguity by reporting a later successful observation.
    control.unknown.set(false);
    runner.shutdown().unwrap();
    assert_eq!(
        runner.physical_resource_snapshot().release,
        ResourceEvidence::Unknown
    );
}

#[test]
fn evidence_coverage_and_json_never_label_unsupported_fields_as_zero() {
    let control = Rc::new(Control::default());
    let runner = runner(control);
    let snapshot = runner.physical_resource_snapshot();
    let coverage = snapshot.coverage();
    assert!(coverage.unobserved.contains(&"allocator"));
    assert!(coverage.unobserved.contains(&"other_workspace_bytes"));
    assert!(coverage.unobserved.contains(&"release"));
    assert!(!coverage.verified.contains(&"close"));
    let json: serde_json::Value =
        serde_json::from_str(&resource_summary_json(&snapshot).unwrap()).unwrap();
    assert_eq!(
        json["allocator"],
        serde_json::json!({"state": "unobserved"})
    );
    assert_eq!(json["expert_cache"]["value"]["charged_bytes"], 112);
}

#[test]
fn physical_scope_mismatch_cannot_replace_values_or_clear_unknown() {
    let mut retained = runner(Rc::new(Control::default())).physical_resource_snapshot();
    retained.mark_unknown();
    let mut replacement = retained.clone();
    replacement.scope =
        ResourceEvidence::Observed(ResourceObservationScope::fresh(std::num::NonZeroU64::MIN));
    replacement.resident_parameter_bytes = ResourceEvidence::Observed(0);
    let combined = retained.snapshot_or_unknown(&replacement);
    assert_eq!(combined.scope, ResourceEvidence::Unknown);
    assert_eq!(
        combined.resident_parameter_bytes,
        retained.resident_parameter_bytes
    );
    assert_eq!(combined.release, ResourceEvidence::Unknown);
    assert_eq!(combined.snapshot_or_unknown(&replacement), combined);
}

#[test]
fn runtime_shutdown_reports_actual_forwarded_physical_evidence_and_broker_domains() {
    let mut driver = ResidentTopKDriver::new(
        runner(Rc::new(Control::default())),
        FixedSequenceSlotPool::new(1),
    );
    assert!(driver.resource_shutdown_summary().is_none());
    let before = driver.resource_observability_snapshot();
    assert_eq!(
        before.model.expert_cache.value().unwrap().charged_bytes,
        112
    );
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    let summary = driver.resource_shutdown_summary().unwrap();
    assert_eq!(summary.attempt, 1);
    assert_eq!(summary.physical.scope, before.model.scope);
    assert_eq!(summary.physical.default_state, before.model.default_state);
    assert_eq!(summary.physical.expert_cache, before.model.expert_cache);
    assert_eq!(
        summary.physical.numeric_workspace,
        before.model.numeric_workspace
    );
    assert_eq!(
        summary.physical.resident_parameter_bytes,
        before.model.resident_parameter_bytes
    );
    assert_eq!(
        summary.physical.live_model_state_bytes,
        before.model.live_model_state_bytes
    );
    assert_eq!(
        summary.runtime.pending_operations,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        summary.runtime.active_credit_grants,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        summary.runtime.closure,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(summary.runtime.fence, ResourceEvidence::Unobserved);
    assert_eq!(
        summary.physical.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert!(summary.coverage.runtime.observed.contains(&"closure"));
    assert!(!summary.coverage.runtime.verified.contains(&"closure"));
    assert!(summary.coverage.physical.observed.contains(&"close"));
    assert!(!summary.coverage.physical.verified.contains(&"close"));
    assert_eq!(
        summary.physical.residency,
        ResourceEvidence::Observed(ResourceResidency::Retained)
    );
    assert_eq!(
        summary.physical.quiescence,
        ResourceEvidence::Verified(ResourceQuiescence::Quiescent)
    );
    assert_eq!(summary.physical.release, ResourceEvidence::Unobserved);
    assert!(summary.coverage.physical.verified.contains(&"quiescence"));
    assert!(summary.coverage.physical.unobserved.contains(&"release"));
    assert!(
        summary
            .broker_domains
            .iter()
            .all(|domain| matches!(domain, ResourceEvidence::Observed(_)))
    );
    let json = parsed(summary);
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["outcome"], "complete");
    assert_eq!(
        json["physical"]["expert_cache"]["value"]["charged_bytes"],
        112
    );
    assert_eq!(json["physical"]["release"]["state"], "unobserved");
    assert_eq!(json["runtime"]["closure"]["state"], "observed");
    assert_eq!(
        json["coverage"]["observed_broker_domains"],
        before.credits.len()
    );
}

#[test]
fn runtime_close_success_without_owner_proof_is_only_an_observed_lifecycle_event() {
    let control = Rc::new(Control::default());
    control.omit_quiescence_proof.set(true);
    let mut driver =
        ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
    let before = driver.resource_observability_snapshot();
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    let after = driver.resource_observability_snapshot();
    assert!(driver.physical_shutdown_complete());
    assert_eq!(
        after.runtime.closure,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(after.runtime.fence, ResourceEvidence::Unobserved);
    assert_eq!(
        after.runtime.pending_operations,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        after.runtime.active_credit_grants,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        after.model.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(after.model.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(after.model.release, ResourceEvidence::Unobserved);
    assert_eq!(
        after.model.residency,
        ResourceEvidence::Observed(ResourceResidency::Retained)
    );
    assert_eq!(after.model.scope, before.model.scope);
    assert_eq!(after.model.expert_cache, before.model.expert_cache);
    assert_eq!(
        after.model.numeric_workspace,
        before.model.numeric_workspace
    );
    assert_eq!(after.model.default_state, before.model.default_state);
    let summary = driver.resource_shutdown_summary().unwrap();
    assert_eq!(
        summary.outcome,
        ferrule_runtime::engine::ResourceShutdownOutcome::Complete
    );
    assert!(!summary.coverage.runtime.verified.contains(&"closure"));
    assert!(summary.coverage.physical.unobserved.contains(&"quiescence"));
    assert!(summary.coverage.physical.unobserved.contains(&"release"));
    let json = parsed(summary);
    assert_eq!(json["runtime"]["closure"]["state"], "observed");
    assert_eq!(json["physical"]["close"]["state"], "observed");
    assert_eq!(json["physical"]["quiescence"]["state"], "unobserved");
    assert_eq!(json["physical"]["release"]["state"], "unobserved");
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    assert_eq!(driver.resource_observability_snapshot(), after);
    assert_eq!(control.shutdown_calls.get(), 1);
}

fn parsed(summary: &ResidentResourceShutdownSummary) -> serde_json::Value {
    serde_json::from_str(&summary.to_json().unwrap()).unwrap()
}

fn pipeline_observation(status: &'static str) -> ExpertReadPipelineObservation {
    let current = ExpertReadIdentity {
        image_generation: 47,
        layer: 3,
        expert: 7,
        operation: 501,
    };
    ExpertReadPipelineObservation {
        enabled: true,
        requested_mode: Some(ExpertReadSchedule::Lookahead),
        effective_mode: Some(ExpertReadSchedule::Lookahead),
        status,
        limits: Some(ExpertReadPipelineOptions {
            schedule: ExpertReadSchedule::Lookahead,
            max_host_bytes: 8192,
            max_metadata_bytes: 128,
            worker_stack_bytes: 4096,
        }),
        current_charge: if status == "closed" {
            ExpertReadHostCharge::default()
        } else {
            ExpertReadHostCharge {
                packed_bytes: 80,
                artifact_bytes: 160,
                metadata_bytes: 24,
                requested_stack_bytes: 4096,
                payload_bytes: 240,
                host_charged_bytes: 4648,
                upload_reserved_packed_bytes: 40,
                pinned_upload_bytes: 96,
                upload_ticket_metadata_bytes: 24,
                upload_policy_headroom_bytes: 128,
                unknown_upload_bytes: if status == "quarantined" { 408 } else { 0 },
                unknown_packed_bytes: if status == "quarantined" { 80 } else { 0 },
                unknown_stack_bytes: if status == "quarantined" { 4096 } else { 0 },
                reserved_read_slots: 2,
                unknown_read_slots: usize::from(status == "quarantined"),
            }
        },
        stats: ExpertReadPipelineStats {
            sequences_started: 2,
            submitted_reads: 3,
            speculative_reads: 1,
            unknown_reads: 1,
            unknown_bytes: 80,
            peak_host_charged_bytes: 4700,
            ..Default::default()
        },
        current_read: (status == "active").then_some(current),
        pending_read: (status == "active" || status == "join_pending_at_scope_end").then_some(
            ExpertReadIdentity {
                expert: 8,
                operation: 502,
                ..current
            },
        ),
        last_receipt: Some(ExpertReadHostReceipt {
            identity: ExpertReadIdentity {
                layer: 2,
                expert: 9,
                operation: 499,
                ..current
            },
            outcome: "unknown",
            requested_bytes: 80,
            io_bytes: None,
            cancellation_requested: true,
            submitted_ns: 3,
            worker_started_ns: None,
            io_started_ns: None,
            io_finished_ns: None,
            source_verified_ns: None,
            completed_ns: None,
            taken_ns: None,
        }),
        ..Default::default()
    }
}

fn assert_coverage<T>(
    coverage: &EvidenceCoverage,
    field: &'static str,
    evidence: &ResourceEvidence<T>,
) {
    for (fields, expected) in [
        (
            &coverage.unobserved,
            matches!(evidence, ResourceEvidence::Unobserved),
        ),
        (
            &coverage.observed,
            matches!(evidence, ResourceEvidence::Observed(_)),
        ),
        (
            &coverage.verified,
            matches!(evidence, ResourceEvidence::Verified(_)),
        ),
        (
            &coverage.unknown,
            matches!(evidence, ResourceEvidence::Unknown),
        ),
    ] {
        assert_eq!(
            fields.iter().filter(|name| **name == field).count(),
            usize::from(expected)
        );
    }
}

#[test]
fn cache_diagnostics_forward_disabled_observed_and_overflowed_evidence_without_zero_filling() {
    for (enabled, overflowed) in [(false, false), (true, false), (true, true), (false, true)] {
        let control = Rc::new(Control::default());
        control.cache_stats.set(ExpertCacheStats {
            diagnostics_enabled: enabled,
            counters_overflowed: overflowed,
            misses: 3,
            entry_limit_evictions: 2,
            byte_limit_evictions: 1,
            scratch_limit_evictions: 1,
            admission_stalls: 4,
            ..Default::default()
        });
        let mut driver =
            ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
        let before = driver.resource_observability_snapshot();
        let cache = before.model.expert_cache.value().unwrap();
        assert_eq!(cache.counters_overflowed, overflowed);
        assert_eq!(cache.charged_bytes, 112);
        if !enabled {
            assert_eq!(cache.diagnostics, ResourceEvidence::Unobserved);
        } else if overflowed {
            assert_eq!(cache.diagnostics, ResourceEvidence::Unknown);
        } else {
            let diagnostics = cache.diagnostics.value().unwrap();
            assert_eq!(diagnostics.misses, 3);
            assert_eq!(diagnostics.entry_limit_evictions, 2);
            assert_eq!(diagnostics.byte_limit_evictions, 1);
            assert_eq!(diagnostics.scratch_limit_evictions, 1);
            assert_eq!(diagnostics.admission_stalls, 4);
            for unsupported in [
                diagnostics.cold_misses,
                diagnostics.evicted_misses,
                diagnostics.evicted_selected,
                diagnostics.evicted_future_hint,
            ] {
                assert_eq!(unsupported, ResourceEvidence::Unobserved);
            }
            assert_eq!(diagnostics.history_bytes, 0);
        }
        for _ in 0..3 {
            assert_eq!(driver.resource_observability_snapshot(), before);
        }
        assert_eq!(control.shutdown_calls.get(), 0);
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.physical.expert_cache, before.model.expert_cache);
        let json = parsed(summary);
        let diagnostic_json = &json["physical"]["expert_cache"]["value"]["diagnostics"];
        assert_eq!(
            json["physical"]["expert_cache"]["value"]["counters_overflowed"],
            overflowed
        );
        assert_eq!(
            diagnostic_json,
            &serde_json::to_value(cache.diagnostics).unwrap()
        );
        if !enabled || overflowed {
            assert!(diagnostic_json.get("value").is_none());
        } else {
            for field in [
                "cold_misses",
                "evicted_misses",
                "evicted_selected",
                "evicted_future_hint",
            ] {
                assert_eq!(
                    diagnostic_json["value"][field],
                    serde_json::json!({"state":"unobserved"})
                );
            }
        }
        assert_eq!(summary.physical.release, ResourceEvidence::Unobserved);
    }
}

fn idle_pipeline_observation() -> ExpertReadPipelineObservation {
    ExpertReadPipelineObservation {
        enabled: true,
        requested_mode: Some(ExpertReadSchedule::Lookahead),
        effective_mode: Some(ExpertReadSchedule::Lookahead),
        worker_read_observed: true,
        status: "idle_retained",
        limits: Some(ExpertReadPipelineOptions {
            schedule: ExpertReadSchedule::Lookahead,
            max_host_bytes: 8192,
            max_metadata_bytes: 128,
            worker_stack_bytes: 4096,
        }),
        current_charge: ExpertReadHostCharge {
            metadata_bytes: 128,
            requested_stack_bytes: 4096,
            host_charged_bytes: 4224,
            ..Default::default()
        },
        stats: ExpertReadPipelineStats {
            sequences_started: 2,
            sequences_finished: 2,
            workers_created: 1,
            worker_reuses: 1,
            submitted_reads: 2,
            completed_reads: 2,
            verified_reads: 2,
            worker_started_reads: 2,
            io_started_reads: 2,
            proved_read_bytes: 160,
            peak_metadata_bytes: 128,
            peak_requested_stack_bytes: 4096,
            peak_host_charged_bytes: 4648,
            ..Default::default()
        },
        worker_alive: true,
        ..Default::default()
    }
}

#[test]
fn retained_upload_charges_forward_and_block_owner_close_until_explicit_retirement() {
    for field in [
        "upload_reserved_packed_bytes",
        "pinned_upload_bytes",
        "upload_ticket_metadata_bytes",
        "upload_policy_headroom_bytes",
        "unknown_upload_bytes",
    ] {
        let control = Rc::new(Control::default());
        let mut pipeline = idle_pipeline_observation();
        let charge = &mut pipeline.current_charge;
        let bytes = match field {
            "upload_reserved_packed_bytes" => {
                charge.upload_reserved_packed_bytes = 40;
                40
            }
            "pinned_upload_bytes" => {
                charge.pinned_upload_bytes = 96;
                96
            }
            "upload_ticket_metadata_bytes" => {
                charge.upload_ticket_metadata_bytes = 24;
                24
            }
            "upload_policy_headroom_bytes" => {
                charge.upload_policy_headroom_bytes = 128;
                128
            }
            "unknown_upload_bytes" => {
                charge.artifact_bytes = 64;
                charge.payload_bytes = 64;
                charge.unknown_upload_bytes = 64;
                64
            }
            _ => unreachable!(),
        };
        charge.host_charged_bytes += bytes;
        control.pipeline.set(ResourceEvidence::Observed(pipeline));
        let runner = runner(control.clone());
        let module = runner
            .forward_executor()
            .module()
            .physical_resource_snapshot(runner.default_state());
        assert_eq!(
            module.expert_read_pipeline,
            ResourceEvidence::Observed(pipeline)
        );
        assert_eq!(
            runner.physical_resource_snapshot().expert_read_pipeline,
            module.expert_read_pipeline
        );
        let mut driver = ResidentTopKDriver::new(runner, FixedSequenceSlotPool::new(1));
        let before = driver.resource_observability_snapshot();
        for _ in 0..3 {
            assert_eq!(driver.resource_observability_snapshot(), before);
        }
        assert_eq!(control.pipeline.get(), ResourceEvidence::Observed(pipeline));
        assert_eq!(control.shutdown_calls.get(), 0);
        let error = driver
            .shutdown_and_close_progress(&mut |_| Ok(()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("retains host read/upload custody")
        );
        assert!(!driver.physical_shutdown_complete());
        assert!(!control.closed.get());
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(
            summary.outcome,
            ferrule_runtime::engine::ResourceShutdownOutcome::Error
        );
        assert_eq!(
            summary.physical.expert_read_pipeline,
            ResourceEvidence::Observed(pipeline)
        );
        assert_eq!(summary.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.close, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.quiescence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.release, ResourceEvidence::Unknown);
        let json = parsed(summary);
        let forwarded = &json["physical"]["expert_read_pipeline"]["value"]["current_charge"];
        assert_eq!(forwarded[field], bytes, "{field}");
        assert_eq!(forwarded["host_charged_bytes"], 4224 + bytes);
        assert_eq!(
            json["physical"]["expert_read_pipeline"],
            serde_json::to_value(ResourceEvidence::Observed(pipeline)).unwrap()
        );

        // The test owner explicitly retires the held upload. Snapshot reads and
        // the failed close did not retire it; the real driver retains ambiguity.
        control
            .pipeline
            .set(ResourceEvidence::Observed(idle_pipeline_observation()));
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
        assert!(driver.physical_shutdown_complete());
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.attempt, 2);
        let closed = summary.physical.expert_read_pipeline.value().unwrap();
        assert_eq!(closed.status, "closed");
        assert_eq!(closed.current_charge, ExpertReadHostCharge::default());
        assert_eq!(summary.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.close, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.quiescence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.release, ResourceEvidence::Unknown);
        assert_eq!(control.shutdown_calls.get(), 2);
    }
}

#[test]
fn unknown_pipeline_or_zero_byte_worker_ambiguity_never_becomes_close_proof() {
    let mut zero_unknown = idle_pipeline_observation();
    zero_unknown.current_charge = ExpertReadHostCharge::default();
    zero_unknown.worker_unknown = true;
    let mut zero_join_pending = zero_unknown;
    zero_join_pending.worker_unknown = false;
    zero_join_pending.worker_join_pending = true;
    for pipeline in [
        ResourceEvidence::Unknown,
        ResourceEvidence::Observed(zero_unknown),
        ResourceEvidence::Observed(zero_join_pending),
    ] {
        let control = Rc::new(Control::default());
        control.pipeline.set(pipeline);
        let mut driver =
            ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
        assert_eq!(
            driver
                .resource_observability_snapshot()
                .model
                .expert_read_pipeline,
            pipeline
        );
        assert!(driver.shutdown_and_close_progress(&mut |_| Ok(())).is_err());
        assert!(!driver.physical_shutdown_complete());
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.physical.expert_read_pipeline, pipeline);
        assert_eq!(summary.physical.close, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.quiescence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.release, ResourceEvidence::Unknown);
        assert_eq!(
            parsed(summary)["physical"]["expert_read_pipeline"],
            serde_json::to_value(pipeline).unwrap()
        );
        control
            .pipeline
            .set(ResourceEvidence::Observed(idle_pipeline_observation()));
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.physical.close, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.quiescence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.release, ResourceEvidence::Unknown);
        if matches!(pipeline, ResourceEvidence::Unknown) {
            assert_eq!(
                summary.physical.expert_read_pipeline,
                ResourceEvidence::Unknown
            );
            assert_eq!(
                parsed(summary)["physical"]["expert_read_pipeline"],
                serde_json::json!({"state":"unknown"})
            );
        }
    }
}

#[test]
fn idle_worker_metadata_and_stack_are_retained_until_explicit_close_not_final_owner_release() {
    let control = Rc::new(Control::default());
    let idle = idle_pipeline_observation();
    control.pipeline.set(ResourceEvidence::Observed(idle));
    let mut driver =
        ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
    let before = driver.resource_observability_snapshot();
    assert_eq!(
        before.model.expert_read_pipeline,
        ResourceEvidence::Observed(idle)
    );
    assert_eq!(idle.current_charge.host_charged_bytes, 4224);
    assert_eq!(idle.current_charge.payload_bytes, 0);
    assert!(idle.worker_alive);
    assert_eq!(idle.stats.workers_joined, 0);
    assert_eq!(driver.resource_observability_snapshot(), before);
    assert_eq!(control.shutdown_calls.get(), 0);
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    let summary = driver.resource_shutdown_summary().unwrap();
    let closed = summary.physical.expert_read_pipeline.value().unwrap();
    assert_eq!(closed.status, "closed");
    assert!(!closed.worker_alive);
    assert_eq!(closed.stats.workers_created, 1);
    assert_eq!(closed.stats.workers_joined, 1);
    assert_eq!(closed.stats.worker_reuses, 1);
    assert_eq!(closed.stats.sequences_started, 2);
    assert_eq!(closed.current_charge, ExpertReadHostCharge::default());
    assert_eq!(summary.physical.expert_cache, before.model.expert_cache);
    assert_eq!(summary.physical.default_state, before.model.default_state);
    assert_eq!(
        summary.physical.numeric_workspace,
        before.model.numeric_workspace
    );
    assert_eq!(
        summary.physical.quiescence,
        ResourceEvidence::Verified(ResourceQuiescence::Quiescent)
    );
    assert_eq!(
        summary.physical.residency,
        ResourceEvidence::Observed(ResourceResidency::Retained)
    );
    assert_eq!(summary.physical.release, ResourceEvidence::Unobserved);
    let json = parsed(summary);
    for field in [
        "upload_reserved_packed_bytes",
        "pinned_upload_bytes",
        "upload_ticket_metadata_bytes",
        "upload_policy_headroom_bytes",
        "unknown_upload_bytes",
    ] {
        assert!(
            json["physical"]["expert_read_pipeline"]["value"]["current_charge"]
                .get(field)
                .is_none()
        );
    }
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    assert_eq!(control.shutdown_calls.get(), 1);
}

#[test]
fn generic_pipeline_and_state_evidence_roundtrip_without_observation_progress() {
    let mut configured = pipeline_observation("configured");
    configured.effective_mode = None;
    configured.current_charge = Default::default();
    configured.stats = Default::default();
    configured.last_receipt = None;
    let mut blocking = configured;
    blocking.requested_mode = Some(ExpertReadSchedule::Blocking);
    blocking.limits.as_mut().unwrap().schedule = ExpertReadSchedule::Blocking;
    let cases = [
        ResourceEvidence::Unobserved,
        ResourceEvidence::Unknown,
        ResourceEvidence::Observed(ExpertReadPipelineObservation::default()),
        ResourceEvidence::Observed(configured),
        ResourceEvidence::Observed(blocking),
        ResourceEvidence::Observed(pipeline_observation("active")),
        ResourceEvidence::Observed(pipeline_observation("join_pending_at_scope_end")),
        ResourceEvidence::Observed(pipeline_observation("closed")),
        ResourceEvidence::Observed(pipeline_observation("closed_retained")),
        ResourceEvidence::Observed(pipeline_observation("quarantined")),
    ];
    for pipeline in cases {
        for state_bytes in [
            ResourceEvidence::Unobserved,
            ResourceEvidence::Observed(0),
            ResourceEvidence::Observed(24),
            ResourceEvidence::Unknown,
        ] {
            let control = Rc::new(Control::default());
            control.pipeline.set(pipeline);
            control.state_resident_bytes.set(Some(state_bytes));
            control.omit_quiescence_proof.set(true);
            let runner = runner(control.clone());
            let module = runner
                .forward_executor()
                .module()
                .physical_resource_snapshot(runner.default_state());
            assert_eq!(module.expert_read_pipeline, pipeline);
            let physical = runner.physical_resource_snapshot();
            assert_eq!(physical.expert_read_pipeline, module.expert_read_pipeline);
            let mut driver = ResidentTopKDriver::new(runner, FixedSequenceSlotPool::new(1));
            let before = driver.resource_observability_snapshot();
            assert_eq!(before.model, physical);
            assert_eq!(
                before.model.default_state.value().unwrap().resident_bytes,
                state_bytes
            );
            let runtime_ledger = driver.observability.resources.snapshot();
            let physical_ledger = driver.observability.physical_resources.clone();
            let registry_stats = driver.load_registry().stats();
            let reads = control.reads.get();
            for _ in 0..3 {
                assert_eq!(driver.resource_observability_snapshot(), before);
                assert_eq!(driver.observability.resources.snapshot(), runtime_ledger);
                assert_eq!(driver.observability.physical_resources, physical_ledger);
                assert_eq!(driver.load_registry().stats(), registry_stats);
            }
            assert_eq!(control.reads.get(), reads + 3);
            assert_eq!(control.pipeline.get(), pipeline);
            assert_eq!(control.shutdown_calls.get(), 0);
            assert_eq!(
                driver
                    .executor
                    .runner()
                    .forward_executor()
                    .arena_stats()
                    .installs,
                0
            );
            assert!(driver.resource_shutdown_summary().is_none());
            assert!(!driver.physical_shutdown_complete());
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
            let summary = driver.resource_shutdown_summary().unwrap();
            assert_eq!(summary.physical.expert_read_pipeline, pipeline);
            assert_eq!(summary.physical.default_state, before.model.default_state);
            assert_eq!(summary.physical.scope, before.model.scope);
            assert_eq!(summary.runtime.scope, before.runtime.scope);
            assert_eq!(summary.physical.quiescence, ResourceEvidence::Unobserved);
            assert_eq!(summary.physical.release, ResourceEvidence::Unobserved);
            assert_eq!(summary.coverage.physical, summary.physical.coverage());
            assert_coverage(
                &summary.coverage.physical,
                "expert_read_pipeline",
                &pipeline,
            );
            assert_coverage(
                &summary.coverage.physical,
                "default_state.resident_bytes",
                &state_bytes,
            );
            let json = parsed(summary);
            assert_eq!(json["schema_version"], 1);
            assert_eq!(
                json["physical"],
                serde_json::to_value(&summary.physical).unwrap()
            );
            assert_eq!(
                json["physical"]["expert_read_pipeline"],
                serde_json::to_value(pipeline).unwrap()
            );
            assert_eq!(
                json["coverage"]["physical"],
                serde_json::to_value(&summary.coverage.physical).unwrap()
            );
            assert_eq!(
                json["physical"]["default_state"]["value"]["resident_bytes"],
                serde_json::to_value(state_bytes).unwrap()
            );
            if pipeline.value().is_none() {
                assert!(
                    json["physical"]["expert_read_pipeline"]
                        .get("value")
                        .is_none()
                );
            }
        }
    }
}

#[test]
fn resource_snapshots_neither_poll_provider_progress_nor_drain_queued_completions() {
    use ferrule_common::{CancellationReason, CompletionOutcome};
    let (mut driver, handle, scope) = super::driver_at_read();
    handle.push_outcome(
        scope.operation,
        scope.key,
        scope.stage,
        CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
    );
    let before = driver.resource_observability_snapshot();
    let commands = handle.commands();
    let stats = driver.load_registry().stats();
    for _ in 0..3 {
        assert_eq!(driver.resource_observability_snapshot(), before);
        assert_eq!(driver.load_registry.pending_completions(), 0);
        assert_eq!(driver.load_registry().stats(), stats);
        assert_eq!(handle.commands(), commands);
    }
    // Only explicit progress transfers the provider's ready completion; reading
    // the subsequent queued-completion gauge must not process it either.
    assert_eq!(driver.load_registry.collect_provider_completions(1), 1);
    for _ in 0..3 {
        assert_eq!(driver.resource_observability_snapshot(), before);
        assert_eq!(driver.load_registry.pending_completions(), 1);
        assert_eq!(driver.load_registry().stats(), stats);
        assert_eq!(handle.commands(), commands);
        assert!(driver.load_registry.retirement(scope.operation).is_none());
    }
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
}

#[test]
fn active_zero_host_charge_does_not_verify_quiescence_or_release() {
    let control = Rc::new(Control::default());
    let active = ExpertReadPipelineObservation {
        current_charge: Default::default(),
        current_read: None,
        pending_read: None,
        last_receipt: None,
        ..pipeline_observation("active")
    };
    control.pipeline.set(ResourceEvidence::Observed(active));
    control.omit_quiescence_proof.set(true);
    let mut driver =
        ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
    let before = driver.resource_observability_snapshot();
    assert_eq!(
        before.model.close,
        ResourceEvidence::Observed(ResourceClosure::Open)
    );
    assert_eq!(before.model.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(before.model.release, ResourceEvidence::Unobserved);
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    let summary = driver.resource_shutdown_summary().unwrap();
    assert_eq!(
        summary.physical.expert_read_pipeline,
        ResourceEvidence::Observed(active)
    );
    assert_eq!(summary.physical.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(summary.physical.release, ResourceEvidence::Unobserved);
    assert_eq!(
        summary.physical.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert!(!summary.coverage.physical.verified.contains(&"close"));
    assert!(
        !summary
            .coverage
            .physical
            .verified
            .contains(&"expert_read_pipeline")
    );
    let json = parsed(summary);
    assert_eq!(
        json["physical"]["expert_read_pipeline"]["value"]["status"],
        "active"
    );
    assert_eq!(
        json["physical"]["expert_read_pipeline"]["value"]["current_charge"]["host_charged_bytes"],
        0
    );
    assert_eq!(
        json["physical"]["release"],
        serde_json::json!({"state": "unobserved"})
    );
}

#[test]
fn pipeline_and_nested_state_unknown_remain_sticky_after_close_and_retry() {
    let control = Rc::new(Control::default());
    control.pipeline.set(ResourceEvidence::Unknown);
    control
        .state_resident_bytes
        .set(Some(ResourceEvidence::Unknown));
    control.failures.set(1);
    let mut driver =
        ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
    assert!(driver.shutdown_and_close_progress(&mut |_| Ok(())).is_err());
    let before = driver.resource_observability_snapshot();
    control
        .pipeline
        .set(ResourceEvidence::Observed(pipeline_observation("closed")));
    control
        .state_resident_bytes
        .set(Some(ResourceEvidence::Observed(0)));
    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    for _ in 0..3 {
        let after = driver.resource_observability_snapshot();
        assert_eq!(after.model.scope, before.model.scope);
        assert_eq!(after.model.expert_read_pipeline, ResourceEvidence::Unknown);
        assert_eq!(
            after.model.default_state.value().unwrap().resident_bytes,
            ResourceEvidence::Unknown
        );
        assert_eq!(after.model.close, ResourceEvidence::Unknown);
        assert_eq!(after.model.release, ResourceEvidence::Unknown);
    }
    let summary = driver.resource_shutdown_summary().unwrap();
    assert_eq!(summary.attempt, 2);
    let json = parsed(summary);
    assert_eq!(json["outcome"], "complete");
    assert_eq!(
        json["physical"]["expert_read_pipeline"],
        serde_json::json!({"state": "unknown"})
    );
    assert_coverage(
        &summary.coverage.physical,
        "expert_read_pipeline",
        &ResourceEvidence::<ExpertReadPipelineObservation>::Unknown,
    );
    assert_coverage(
        &summary.coverage.physical,
        "default_state.resident_bytes",
        &ResourceEvidence::<usize>::Unknown,
    );
}

#[test]
fn runtime_summary_retains_standalone_scopes_and_exact_pipeline_identities() {
    let control = Rc::new(Control::default());
    let mut pipeline = pipeline_observation("active");
    // Read source generations and receipt operation IDs are not runtime-owner
    // generations or prepared-plan IDs, even if they happen to share a number.
    pipeline.current_read.as_mut().unwrap().image_generation = 97;
    control.pipeline.set(ResourceEvidence::Observed(pipeline));
    control.omit_quiescence_proof.set(true);
    let mut first = ResidentTopKDriver::new(
        runner_with_plan(control.clone(), 41),
        FixedSequenceSlotPool::new(1),
    );
    let second = ResidentTopKDriver::new(
        runner_with_plan(control.clone(), 42),
        FixedSequenceSlotPool::new(1),
    );
    let a = first.resource_observability_snapshot();
    let b = second.resource_observability_snapshot();
    assert_ne!(a.runtime.scope.owner, b.runtime.scope.owner);
    assert_ne!(
        a.model.scope.value().unwrap().owner,
        b.model.scope.value().unwrap().owner
    );
    assert_ne!(a.runtime.scope.owner, a.model.scope.value().unwrap().owner);
    assert_eq!(a.runtime.scope.generation.get(), 1);
    assert_eq!(a.model.scope.value().unwrap().generation.get(), 41);
    assert_eq!(b.model.scope.value().unwrap().generation.get(), 42);
    assert_eq!(a.model.expert_read_pipeline, b.model.expert_read_pipeline);
    first.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
    let summary = first.resource_shutdown_summary().unwrap();
    let json = parsed(summary);
    assert_eq!(
        json["runtime"]["scope"],
        serde_json::to_value(a.runtime.scope).unwrap()
    );
    assert_eq!(
        json["physical"]["scope"]["value"],
        serde_json::to_value(a.model.scope.value().unwrap()).unwrap()
    );
    assert_eq!(
        json["physical"]["expert_read_pipeline"]["value"]["current_read"]["image_generation"],
        97
    );
    assert_eq!(
        json["physical"]["expert_read_pipeline"]["value"]["last_receipt"]["identity"]["operation"],
        499
    );
}

#[test]
fn overlapping_broker_cache_and_workspace_views_are_not_summed() {
    use crate::scheduling::{ExecutionPhase, PhysicalResourceClaim, ResourceDemand, ResourceKind};
    let mut driver = ResidentTopKDriver::new(
        runner(Rc::new(Control::default())),
        FixedSequenceSlotPool::new(1),
    );
    driver
        .load_registry
        .resources_mut()
        .reconfigure_limit(ResourceKind::ResidentBytes, 64, 0)
        .unwrap();
    let mut grant = driver
        .load_registry
        .acquire_hard_resources(
            123,
            ResourceDemand::required(ExecutionPhase::Decode),
            [PhysicalResourceClaim::new(ResourceKind::ResidentBytes, 64)],
        )
        .unwrap();
    let snapshot = driver.resource_observability_snapshot();
    let result = Ok(crate::engine::ResidentShutdownProgress::Pending);
    let summary = ResidentResourceShutdownSummary::new(1, &result, snapshot.clone());
    let json = parsed(&summary);
    assert_eq!(summary.runtime.scope, snapshot.runtime.scope);
    assert_eq!(summary.physical.scope, snapshot.model.scope);
    assert_ne!(
        summary.runtime.scope.owner,
        summary.physical.scope.value().unwrap().owner
    );
    assert_eq!(json["physical"]["resident_parameter_bytes"]["value"], 64);
    assert_eq!(
        json["physical"]["expert_cache"]["value"]["resident_bytes"],
        64
    );
    assert_eq!(
        json["physical"]["expert_cache"]["value"]["charged_bytes"],
        112
    );
    assert_eq!(
        json["physical"]["numeric_workspace"]["value"]["reserved_bytes"],
        16
    );
    assert_eq!(
        json["physical"]["expert_cache"]["value"]["scratch_bytes"],
        16
    );
    let resident_domain = summary
        .broker_domains
        .iter()
        .filter_map(ResourceEvidence::value)
        .find(|domain| domain.kind == "ResidentBytes")
        .unwrap();
    assert_eq!(resident_domain.in_use, 64);
    assert_eq!(json["runtime"]["resident_bytes"]["value"], 0);
    assert!(json.get("total_bytes").is_none());
    assert!(json["physical"].get("total_bytes").is_none());
    assert!(!grant.is_released());
    driver
        .load_registry
        .release_hard_resources(&mut grant)
        .unwrap();
}

#[test]
fn every_shutdown_attempt_emits_the_versioned_log_and_retains_unknown_after_retry() {
    let control = Rc::new(Control::default());
    // This exercises lifecycle-event logging, not a quiescent owner receipt.
    control.omit_quiescence_proof.set(true);
    control.failures.set(1);
    let pipeline = pipeline_observation("quarantined");
    control.pipeline.set(ResourceEvidence::Observed(pipeline));
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    // Keep two scoped dispatches alive: tracing's single-dispatch fast path
    // otherwise registers a shared callsite against a parallel thread's default.
    let _interest_dispatch = tracing::Dispatch::new(LogCapture(captured.clone()));
    let subscriber = LogCapture(captured.clone());
    tracing::subscriber::with_default(subscriber, || {
        let mut driver =
            ResidentTopKDriver::new(runner(control.clone()), FixedSequenceSlotPool::new(1));
        assert!(driver.shutdown_and_close_progress(&mut |_| Ok(())).is_err());
        assert_eq!(driver.resource_shutdown_summary().unwrap().attempt, 1);
        assert_eq!(
            parsed(driver.resource_shutdown_summary().unwrap())["outcome"],
            "error"
        );
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap();
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.attempt, 2);
        assert_eq!(summary.physical.release, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.quiescence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical.close, ResourceEvidence::Unknown);
        assert_eq!(
            summary.physical.expert_cache.value().unwrap().charged_bytes,
            112
        );
        assert!(driver.physical_shutdown_complete());
    });
    let events = captured.lock().unwrap();
    assert_eq!(events.len(), 2);
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.get("event").unwrap(), "resource_shutdown_summary");
        let summary: serde_json::Value =
            serde_json::from_str(event.get("summary_json").unwrap()).unwrap();
        assert_eq!(summary["attempt"], i + 1);
        assert_eq!(summary["schema_version"], 1);
        assert_eq!(summary["physical"]["release"]["state"], "unknown");
        assert_eq!(
            summary["physical"]["expert_read_pipeline"],
            serde_json::to_value(ResourceEvidence::Observed(pipeline)).unwrap()
        );
        assert!(
            summary["coverage"]["physical"]["observed"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("expert_read_pipeline"))
        );
        let receipt = &summary["physical"]["expert_read_pipeline"]["value"]["last_receipt"];
        assert_eq!(receipt["submitted_ns"], 3);
        for field in [
            "worker_started_ns",
            "io_started_ns",
            "io_finished_ns",
            "source_verified_ns",
            "completed_ns",
            "taken_ns",
            "io_bytes",
        ] {
            assert_eq!(
                receipt.get(field),
                Some(&serde_json::Value::Null),
                "{field}"
            );
        }
    }
    let second: serde_json::Value =
        serde_json::from_str(events[1].get("summary_json").unwrap()).unwrap();
    assert_eq!(second["outcome"], "complete");
    assert_eq!(second["physical"]["close"]["state"], "unknown");
}

type LogFields = std::collections::BTreeMap<String, String>;
struct LogCapture(std::sync::Arc<std::sync::Mutex<Vec<LogFields>>>);
impl tracing::Subscriber for LogCapture {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }

    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::INFO)
    }

    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_resource_observability"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = FieldVisitor(LogFields::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}
struct FieldVisitor(LogFields);
impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
}
