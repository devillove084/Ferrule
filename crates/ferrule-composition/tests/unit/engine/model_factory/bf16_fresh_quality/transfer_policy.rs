//! CPU contracts for policy identity in the maintained full-model gates.
use super::*;

fn grouped_profiles() -> [(Full40TestConfig, Full40TestConfig); 2] {
    [
        (
            Full40TestConfig::StorageMiddle64Blocking,
            Full40TestConfig::StorageMiddle64GroupedBlocking,
        ),
        (
            Full40TestConfig::StorageMiddle64Lookahead,
            Full40TestConfig::StorageMiddle64GroupedLookahead,
        ),
    ]
}

#[test]
fn grouped_profiles_bind_exact_policy_and_baseline_host_cap_without_reference_changes() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    assert_eq!(
        ExpertTransferPolicy::default(),
        ExpertTransferPolicy::Synchronous
    );
    assert_eq!((BF16_ATOL, BF16_RTOL), (0.00025, 0.012));
    for (sync, grouped) in grouped_profiles() {
        assert_eq!(sync.capacity(precision), grouped.capacity(precision));
        assert_eq!(sync.scheduler(), grouped.scheduler());
        assert_eq!(sync.expert_read_pipeline(), grouped.expert_read_pipeline());
        assert_eq!(
            sync.manifest(precision)["capacity_and_options"],
            grouped.manifest(precision)["capacity_and_options"]
        );
        assert_eq!(
            sync.expert_transfer_policy(),
            ExpertTransferPolicy::Synchronous
        );
        assert!(
            sync.manifest(precision)
                .get("expert_transfer_policy")
                .is_none()
        );
        assert_eq!(
            grouped.expert_transfer_policy(),
            ExpertTransferPolicy::GroupedSynchronous
        );
        assert_eq!(
            grouped.manifest(precision)["expert_transfer_policy"],
            json!({
                "mode":"grouped_synchronous", "source_custody":"existing_bounded_read",
                "host_quota_unchanged":true, "additional_pinned_bytes":0,
                "physical_overhead_measured":false, "gpu_overlap_claim":false,
            })
        );
        let read = grouped.expert_read_pipeline().unwrap();
        assert_eq!(
            (
                read.max_host_bytes,
                read.max_metadata_bytes,
                read.worker_stack_bytes
            ),
            (64 << 20, 8 << 20, 2 << 20)
        );
        let payload = 3 * (512 * 2048 + 64 * 2);
        let charged = 2 * payload + read.max_metadata_bytes + read.worker_stack_bytes as u64;
        assert!(charged <= read.max_host_bytes);
        assert!(charged + payload <= read.max_host_bytes);
        assert_ne!(sync.manifest(precision), grouped.manifest(precision));
    }
}

fn primary(config: Full40TestConfig, effective: Value) -> Value {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let calls: serde_json::Map<String, Value> = (0..64)
        .map(|i| (i.to_string(), json!({"sample":i})))
        .collect();
    json!({"schema":"ferrule.bf16-fresh-quality-result.v1", "passed":true, "complete":true,
        "cold_owner_restart":false, "test_config":config.manifest(precision), "effective_config":effective,
        "oracle_manifest_sha256":"same-pin", "observer_mode":"diagnostic_routes_and_states",
        "ordinary_no_observer_claim":false,
        "expert_read_quality":{"passed":true,"requested":config.expert_read_pipeline().unwrap()},
        "prediction_fingerprints":{"calls":calls}})
}

#[test]
fn grouped_cold_and_no_observer_reject_cross_policy_missing_and_modified_primary() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let fixture = crate::model_factory::mixed_serving::fixture::Fixture::new();
    for (sync, grouped) in grouped_profiles() {
        let plan = prepare_full40_with_config(fixture.dir.path(), precision, grouped);
        assert_eq!(
            plan.requested_expert_transfer_policy(),
            Some(ServingExpertTransferPolicy::Synchronous)
        );
        assert_eq!(
            plan.expert_transfer_policy(),
            ServingExpertTransferPolicy::Synchronous
        );
        let effective = full40_effective_config(&plan, grouped, precision);
        assert_eq!(
            effective["expert_transfer_policy"],
            grouped.manifest(precision)["expert_transfer_policy"]
        );
        let prior = primary(grouped, effective.clone());
        assert_cold_primary(&prior, grouped, precision, "same-pin");
        expert_read_reporting::assert_diagnostic_primary(
            &prior,
            &grouped.manifest(precision),
            &effective,
            "same-pin",
        );
        for (field, value) in [
            ("mode", json!("synchronous")),
            ("source_custody", json!("unbounded")),
            ("additional_pinned_bytes", json!(1)),
        ] {
            let mut bad = prior.clone();
            bad["test_config"]["expert_transfer_policy"][field] = value;
            assert!(
                std::panic::catch_unwind(|| assert_cold_primary(
                    &bad, grouped, precision, "same-pin"
                ))
                .is_err(),
                "{field}"
            );
            assert!(
                std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                    &bad,
                    &grouped.manifest(precision),
                    &effective,
                    "same-pin"
                ))
                .is_err(),
                "{field}"
            );
        }
        let mut missing = prior.clone();
        missing["test_config"]
            .as_object_mut()
            .unwrap()
            .remove("expert_transfer_policy");
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(
                &missing, grouped, precision, "same-pin"
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                &missing,
                &grouped.manifest(precision),
                &effective,
                "same-pin"
            ))
            .is_err()
        );
        let sync_plan = prepare_full40_with_config(fixture.dir.path(), precision, sync);
        assert_eq!(
            sync_plan.expert_transfer_policy(),
            ServingExpertTransferPolicy::Synchronous
        );
        let sync_effective = full40_effective_config(&sync_plan, sync, precision);
        let sync_prior = primary(sync, sync_effective.clone());
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(
                &sync_prior,
                grouped,
                precision,
                "same-pin"
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(&prior, sync, precision, "same-pin"))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                &sync_prior,
                &grouped.manifest(precision),
                &effective,
                "same-pin"
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                &prior,
                &sync.manifest(precision),
                &sync_effective,
                "same-pin"
            ))
            .is_err()
        );
        let mut wrong_effective = prior.clone();
        wrong_effective["effective_config"] = sync_effective;
        assert!(
            std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                &wrong_effective,
                &grouped.manifest(precision),
                &effective,
                "same-pin"
            ))
            .is_err()
        );
    }
}
