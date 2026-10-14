use crate as model;

#[path = "fixture.rs"]
mod fixture;

use crate::ResidentModelRunner;
use crate::resource_observability::*;
use fixture::{Control, runner};
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
fn successful_close_without_owner_proof_retains_storage_and_unobserved_release() {
    let control = Rc::new(Control::default());
    control.omit_quiescence_proof.set(true);
    let mut runner = runner(control.clone());
    let before = runner.physical_resource_snapshot();
    runner.shutdown().unwrap();
    let after = runner.physical_resource_snapshot();
    assert_eq!(
        after.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(after.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(after.release, ResourceEvidence::Unobserved);
    assert_eq!(after.scope, before.scope);
    assert_eq!(
        after.residency,
        ResourceEvidence::Observed(ResourceResidency::Retained)
    );
    assert_eq!(after.expert_cache, before.expert_cache);
    assert_eq!(after.numeric_workspace, before.numeric_workspace);
    assert_eq!(after.default_state, before.default_state);
    assert_eq!(after.live_model_state_bytes, before.live_model_state_bytes);
    runner.shutdown().unwrap();
    assert_eq!(runner.physical_resource_snapshot(), after);
    assert_eq!(control.shutdown_calls.get(), 1);
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
fn changed_image_or_plan_cannot_refresh_the_same_scope() {
    let retained = runner(Rc::new(Control::default())).physical_resource_snapshot();
    for field in ["image_generation", "prepared_plan_id"] {
        let mut replacement = retained.clone();
        match field {
            "image_generation" => replacement.image_generation = ResourceEvidence::Observed(48),
            _ => replacement.prepared_plan_id = ResourceEvidence::Observed(42),
        }
        replacement.resident_parameter_bytes = ResourceEvidence::Observed(0);
        assert!(matches!(
            retained.snapshot_with_live(&replacement),
            Err(PhysicalResourceObservationMismatch::Identity { field: name, .. }) if name == field
        ));
        let combined = retained.snapshot_or_unknown(&replacement);
        assert_eq!(combined.scope, ResourceEvidence::Unknown);
        assert_eq!(
            combined.resident_parameter_bytes,
            retained.resident_parameter_bytes
        );
        assert_eq!(combined.snapshot_or_unknown(&retained), combined);
    }
}

#[test]
fn failed_upper_close_preserves_only_an_authenticated_lower_owner_fence() {
    let mut observation = runner(Rc::new(Control::default())).physical_resource_snapshot();
    observation.quiescence = ResourceEvidence::Verified(ResourceQuiescence::Quiescent);
    observation.record_close(false);
    observation.record_close(true);
    assert_eq!(
        observation.quiescence,
        ResourceEvidence::Verified(ResourceQuiescence::Quiescent)
    );
    assert_eq!(observation.close, ResourceEvidence::Unknown);
    assert_eq!(observation.release, ResourceEvidence::Unknown);

    let mut reported = PhysicalResourceObservation {
        quiescence: ResourceEvidence::Observed(ResourceQuiescence::Quiescent),
        ..Default::default()
    };
    reported.record_close(false);
    assert_eq!(reported.quiescence, ResourceEvidence::Unknown);
}

#[test]
fn ambiguous_default_state_does_not_report_unobserved_subset_coverage() {
    let observation = PhysicalResourceObservation {
        default_state: ResourceEvidence::Unknown,
        ..Default::default()
    };
    let coverage = observation.coverage();
    assert!(coverage.unknown.contains(&"default_state"));
    assert!(coverage.unknown.contains(&"default_state.resident_bytes"));
    assert!(
        !coverage
            .unobserved
            .contains(&"default_state.resident_bytes")
    );
}

#[test]
fn unsupported_module_forwards_only_known_runner_identity_and_close_event() {
    let control = Rc::new(Control::default());
    control.unsupported.set(true);
    let mut runner = runner(control.clone());
    let before = runner.physical_resource_snapshot();
    assert!(before.scope.value().is_some());
    assert!(before.default_state.value().is_some());
    assert_eq!(
        before.default_state.value().unwrap().resident_bytes,
        ResourceEvidence::Unobserved
    );
    assert_eq!(before.image_generation, ResourceEvidence::Unobserved);
    assert_eq!(before.expert_cache, ResourceEvidence::Unobserved);
    assert_eq!(before.numeric_workspace, ResourceEvidence::Unobserved);
    assert_eq!(before.residency, ResourceEvidence::Unobserved);
    runner.shutdown().unwrap();
    let after = runner.physical_resource_snapshot();
    assert_eq!(after.scope, before.scope);
    assert_eq!(
        after.close,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(after.quiescence, ResourceEvidence::Unobserved);
    assert_eq!(after.release, ResourceEvidence::Unobserved);
}

#[test]
fn default_state_subset_unknown_is_sticky_across_later_known_identity() {
    let mut retained = runner(Rc::new(Control::default())).physical_resource_snapshot();
    let ResourceEvidence::Observed(state) = &mut retained.default_state else {
        panic!("mock default state must be observed");
    };
    state.resident_bytes = ResourceEvidence::Unknown;
    let mut live = retained.clone();
    let ResourceEvidence::Observed(state) = &mut live.default_state else {
        unreachable!();
    };
    state.generation += 1;
    state.resident_bytes = ResourceEvidence::Observed(0);
    let combined = retained.snapshot_with_live(&live).unwrap();
    assert_eq!(
        combined.default_state.value().unwrap().generation,
        live.default_state.value().unwrap().generation
    );
    assert_eq!(
        combined.default_state.value().unwrap().resident_bytes,
        ResourceEvidence::Unknown
    );
}
