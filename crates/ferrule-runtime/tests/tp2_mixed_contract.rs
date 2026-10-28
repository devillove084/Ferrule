//! Metadata and synthetic byte-ledger contracts only; never starts CUDA or workers.
use ferrule_common::topology::{ExpertDispatchMembers, ExpertSourceScope};
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_runtime::parallel::pipeline::{HybridDeviceBudget, HybridStageBudget};

fn rank(value: u32) -> ParallelRankId {
    ParallelRankId::new(value)
}

#[test]
fn tp2_mesh_excludes_attached_ep_owners_and_requires_tensor_source_scope() {
    let plan = ParallelismPlan {
        tensor_parallel: 2,
        expert_parallel: 2,
        ..Default::default()
    };
    let mesh =
        ValidatedParallelTopology::new(ParallelTopologyId::new(71), 2, rank(0), plan).unwrap();
    assert_eq!(mesh.world_size(), 2);
    assert_eq!(
        mesh.participants().iter().collect::<Vec<_>>(),
        vec![rank(0), rank(1)]
    );
    assert!(ValidatedParallelTopology::new(ParallelTopologyId::new(71), 4, rank(0), plan).is_err());
    for source in [rank(0), rank(1)] {
        let attachment = ExpertDispatchMembers::new_with_scope(
            &mesh,
            0,
            0,
            ExpertSourceScope::ExternalTensorSource,
            source,
            [rank(2), rank(3)],
        )
        .unwrap();
        assert_eq!(
            attachment.iter().collect::<Vec<_>>(),
            vec![rank(2), rank(3)]
        );
        assert!(!attachment.contains(source));
        assert!(
            ExpertDispatchMembers::new_with_scope(
                &mesh,
                0,
                0,
                ExpertSourceScope::ExternalStage,
                source,
                [rank(2), rank(3)],
            )
            .is_err()
        );
    }
    assert!(
        ExpertDispatchMembers::new_with_scope(
            &mesh,
            0,
            0,
            ExpertSourceScope::ExternalTensorSource,
            rank(2),
            [rank(2), rank(3)],
        )
        .is_err()
    );
    assert!(
        ExpertDispatchMembers::new_with_scope(
            &mesh,
            0,
            0,
            ExpertSourceScope::ExternalTensorSource,
            rank(0),
            [rank(0), rank(1)],
        )
        .is_err()
    );
}

#[test]
fn four_owners_fitting_four_24gb_cards_need_not_fit_two_colocated_cards() {
    // Decimal GB, explicitly synthetic, not a full-model memory estimate.
    const GB: u64 = 1_000_000_000;
    let root = |device| HybridStageBudget {
        device,
        weights: 9 * GB,
        state: GB,
        kv: 2 * GB,
        workspace: GB,
        margin: GB,
    };
    let expert = |device| HybridStageBudget {
        device,
        weights: 10 * GB,
        state: 0,
        kv: 0,
        workspace: GB,
        margin: GB,
    };
    let mut owners = [root(0), root(1), expert(2), expert(3)];
    let four = [(0, 24 * GB), (1, 24 * GB), (2, 24 * GB), (3, 24 * GB)];
    assert_eq!(HybridDeviceBudget::admit(&owners, &four).unwrap().len(), 4);
    owners[2].device = 0;
    owners[3].device = 1;
    assert!(HybridDeviceBudget::admit(&owners, &[(0, 24 * GB), (1, 24 * GB)]).is_err());
    let exact = HybridDeviceBudget::admit(&owners, &[(0, 26 * GB), (1, 26 * GB)]).unwrap();
    assert_eq!(exact.len(), 2);
    assert!(exact.iter().all(|card| card.required == 26 * GB));
    // Capacity on the other card cannot compensate for a one-byte shortfall.
    assert!(HybridDeviceBudget::admit(&owners, &[(0, 26 * GB - 1), (1, 100 * GB)]).is_err());
    assert!(HybridDeviceBudget::admit(&owners, &four).is_err());
    assert!(HybridDeviceBudget::admit(&owners, &[(0, 52 * GB)]).is_err());
    assert!(
        HybridDeviceBudget::admit(&owners, &[(0, 26 * GB), (0, 26 * GB), (1, 26 * GB)]).is_err()
    );
    // State is charged per owner, even when both roots share one physical card.
    owners[1].device = 0;
    let required =
        owners[0].total().unwrap() + owners[1].total().unwrap() + owners[2].total().unwrap();
    let before = HybridDeviceBudget::admit(&owners, &[(0, required), (1, 12 * GB)]).unwrap();
    assert_eq!(
        before
            .iter()
            .find(|card| card.device == 0)
            .unwrap()
            .required,
        40 * GB
    );
    owners[1].state += 1;
    assert!(HybridDeviceBudget::admit(&owners, &[(0, required), (1, 12 * GB)]).is_err());
}

#[cfg(unix)]
mod wire {
    use ferrule_common::topology::ExpertSourceScope;
    use ferrule_model::transformer::NumericFp8Precision;
    use ferrule_runtime::parallel::expert_process::{
        NUMERIC_EXPERT_WIRE_VERSION, NumericBoot, NumericSource,
    };
    use ferrule_runtime::parallel::process::decoder::{DecoderDevice, ExpertPlacementFrame};

    fn boot() -> NumericBoot {
        NumericBoot {
            numeric_expert_boot: NUMERIC_EXPERT_WIRE_VERSION,
            numeric_profile: NumericFp8Precision::F32Tf32x3,
            checkpoint: std::env::current_exe().unwrap(), // Identity only; never opened as weights.
            total_layers: 1,
            layers: 0..1,
            placement: ExpertPlacementFrame {
                source_scope: ExpertSourceScope::ExternalStage,
                source: 0,
                members: vec![2, 3],
                entries: vec![(0, 0, 2), (0, 1, 3)],
                route_entries: None,
                placement_epoch: 0,
                max_tokens: 2,
                max_bytes: 64,
                devices: vec![
                    DecoderDevice::Cuda { ordinal: 0 },
                    DecoderDevice::Cuda { ordinal: 1 },
                ],
                timeout_ms: 1000,
            },
            owner: 2,
            hidden: 8,
            max_rows: 2,
            max_parameter_bytes: 4096,
            host_max_bytes: 8192,
            host_max_experts: 1,
            host_workers: 1,
            host_headroom_bytes: 0,
            device_bytes: 8192,
            workspace_bytes: 4096,
            expected: vec![NumericSource {
                layer: 0,
                expert: 0,
                identity: [1; 32],
                content_hash: [2; 32],
                encoding: 1,
                generation: 1,
                bytes: 128,
            }],
        }
    }

    #[test]
    fn numeric_wire_keeps_legacy_profiles_but_rejects_mixed_and_tensor_source_boots() {
        for (profile, name) in [
            (NumericFp8Precision::F32Tf32x3, "f32"),
            (NumericFp8Precision::W8A16Bf16F32Accumulate, "w8a16-bf16"),
        ] {
            let mut value = boot();
            value.numeric_profile = profile;
            value.group().unwrap();
            let encoded = serde_json::to_value(&value).unwrap();
            assert_eq!(encoded["numeric_profile"], name);
            let restored: NumericBoot = serde_json::from_value(encoded).unwrap();
            assert_eq!(restored.numeric_profile, profile);
            assert_eq!(restored.group().unwrap(), value.group().unwrap());
        }
        let mut mixed = boot();
        mixed.numeric_profile = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
        assert!(mixed.group().is_err());
        assert!(serde_json::to_value(mixed).is_err());
        let mut tensor_source = boot();
        tensor_source.placement.source_scope = ExpertSourceScope::ExternalTensorSource;
        assert!(
            tensor_source.group().is_err(),
            "TP process boot is not enabled by topology alone"
        );
        let mut legacy = serde_json::to_value(boot()).unwrap();
        legacy.as_object_mut().unwrap().remove("numeric_profile");
        assert_eq!(
            serde_json::from_value::<NumericBoot>(legacy.clone())
                .unwrap()
                .numeric_profile,
            NumericFp8Precision::F32Tf32x3
        );
        for unknown in ["mixed", "bf16-model-mixed", "bf16"] {
            legacy["numeric_profile"] = serde_json::json!(unknown);
            assert!(serde_json::from_value::<NumericBoot>(legacy.clone()).is_err());
        }
    }
}
