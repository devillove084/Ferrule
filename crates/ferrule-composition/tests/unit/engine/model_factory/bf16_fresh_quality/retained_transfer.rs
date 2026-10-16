//! CPU contracts for policy identity in the maintained full-model gates.
use super::*;

fn profiles() -> [(Full40TestConfig, Full40TestConfig); 2] {
    [
        (
            Full40TestConfig::StorageMiddle64Blocking,
            Full40TestConfig::StorageMiddle64RetainedBlocking,
        ),
        (
            Full40TestConfig::StorageMiddle64Lookahead,
            Full40TestConfig::StorageMiddle64RetainedLookahead,
        ),
    ]
}

#[test]
fn retained_profiles_bind_exact_policy_without_changing_any_quota_or_reference_bound() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    assert_eq!(
        ExpertTransferPolicy::default(),
        ExpertTransferPolicy::Synchronous
    );
    assert_eq!((BF16_ATOL, BF16_RTOL), (0.00025, 0.012));
    for (sync, retained) in profiles() {
        assert_eq!(sync.capacity(precision), retained.capacity(precision));
        assert_eq!(sync.scheduler(), retained.scheduler());
        assert_eq!(sync.expert_read_pipeline(), retained.expert_read_pipeline());
        assert_eq!(
            sync.manifest(precision)["capacity_and_options"],
            retained.manifest(precision)["capacity_and_options"]
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
            retained.expert_transfer_policy(),
            ExpertTransferPolicy::Retained {
                policy_headroom_bytes: 16 << 20,
                readiness_timeout: std::time::Duration::from_secs(30),
            }
        );
        assert_eq!(
            retained.manifest(precision)["expert_transfer_policy"],
            json!({
                "mode":"retained", "policy_headroom_bytes":16 << 20,
                "readiness_timeout_ms":30_000, "host_quota_unchanged":true,
                "physical_overhead_measured":false, "gpu_overlap_claim":false,
            })
        );
        let read = retained.expert_read_pipeline().unwrap();
        assert_eq!(
            (
                read.max_host_bytes,
                read.max_metadata_bytes,
                read.worker_stack_bytes
            ),
            (64 << 20, 8 << 20, 2 << 20)
        );
        let ticket = std::mem::size_of::<
            ferrule_backend::cuda::operators::linear::CudaNumericFp8UploadTicket,
        >();
        let payload = 3 * (512 * 2048 + 64 * 2);
        let charged = 3 * payload + ticket + (16 << 20) + (8 << 20) + (2 << 20);
        assert!(charged <= read.max_host_bytes as usize);
        // Even an extra packed lookahead would fit the logical cap. Production
        // currently serializes it instead; no overlap or physical-memory proof.
        assert!(charged + payload <= read.max_host_bytes as usize);
        eprintln!(
            "RETAINED_TRANSFER_HOST_POLICY {}",
            json!({
                "profile":retained.name(), "payload_bytes":payload, "ticket_inline_bytes":ticket,
                "metadata_bytes":read.max_metadata_bytes, "requested_stack_bytes":read.worker_stack_bytes,
                "policy_headroom_bytes":16 << 20, "readiness_timeout_ms":30_000,
                "aggregate_reservation_bytes":charged, "max_host_bytes":read.max_host_bytes,
                "logical_slack_bytes":read.max_host_bytes as usize - charged,
                "physical_overhead_measured":false, "gpu_overlap_claim":false,
            })
        );
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
fn retained_cold_and_no_observer_reject_cross_policy_missing_and_modified_primary() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let fixture = crate::model_factory::mixed_serving::fixture::Fixture::new();
    for (sync, retained) in profiles() {
        let plan = prepare_full40_with_config(fixture.dir.path(), precision, retained);
        let effective = full40_effective_config(&plan, retained, precision);
        assert_eq!(
            effective["expert_transfer_policy"],
            retained.manifest(precision)["expert_transfer_policy"]
        );
        let prior = primary(retained, effective.clone());
        assert_cold_primary(&prior, retained, precision, "same-pin");
        expert_read_reporting::assert_diagnostic_primary(
            &prior,
            &retained.manifest(precision),
            &effective,
            "same-pin",
        );
        for field in ["mode", "policy_headroom_bytes", "readiness_timeout_ms"] {
            let mut bad = prior.clone();
            bad["test_config"]["expert_transfer_policy"][field] = match field {
                "mode" => json!("synchronous"),
                "policy_headroom_bytes" => json!((16 << 20) + 1),
                "readiness_timeout_ms" => json!(30_001),
                _ => unreachable!(),
            };
            assert!(
                std::panic::catch_unwind(|| assert_cold_primary(
                    &bad, retained, precision, "same-pin"
                ))
                .is_err(),
                "{field}"
            );
            assert!(
                std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                    &bad,
                    &retained.manifest(precision),
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
                &missing, retained, precision, "same-pin"
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                &missing,
                &retained.manifest(precision),
                &effective,
                "same-pin"
            ))
            .is_err()
        );
        let sync_plan = prepare_full40_with_config(fixture.dir.path(), precision, sync);
        let sync_effective = full40_effective_config(&sync_plan, sync, precision);
        let sync_prior = primary(sync, sync_effective.clone());
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(
                &sync_prior,
                retained,
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
                &retained.manifest(precision),
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
                &retained.manifest(precision),
                &effective,
                "same-pin"
            ))
            .is_err()
        );
    }
}
