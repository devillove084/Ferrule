use ferrule_model as model;

#[path = "../../../ferrule-model/tests/resource_observability/fixture.rs"]
mod fixture;

use ferrule_runtime::FixedSequenceSlotPool;
use ferrule_runtime::engine::{ResidentResourceShutdownSummary, ResidentTopKDriver};
use fixture::{Control, runner};
use model::ResidentModelRunner;
use model::resource_observability::*;
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

#[test]
fn every_shutdown_attempt_emits_the_versioned_log_and_retains_unknown_after_retry() {
    let control = Rc::new(Control::default());
    control.failures.set(1);
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
