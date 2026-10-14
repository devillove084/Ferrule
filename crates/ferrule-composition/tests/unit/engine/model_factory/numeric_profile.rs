use super::super::{Qwen35MoeCapacityLimits, Qwen35MoeCapacityPlan, qwen35, tests::options};
use super::*;
use ferrule_model::{ModelFamily, models::qwen35::Qwen35Config};

#[test]
fn numeric_profile_default_and_wire_names_are_explicit() {
    assert_eq!(NumericFp8Profile::default(), NumericFp8Profile::F32);
    assert_eq!(
        NumericFp8Profile::F32.precision(),
        NumericFp8Precision::F32Tf32x3
    );
    assert_eq!(
        NumericFp8Profile::W8A16Bf16.precision(),
        NumericFp8Precision::W8A16Bf16F32Accumulate
    );
    for (profile, wire) in [
        (NumericFp8Profile::F32, "f32"),
        (NumericFp8Profile::W8A16Bf16, "w8a16-bf16"),
    ] {
        assert_eq!(serde_json::to_value(profile).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<NumericFp8Profile>(serde_json::json!(wire)).unwrap(),
            profile
        );
    }
    for unknown in ["bf16", "auto", "W8A16Bf16F32Accumulate", "typo"] {
        assert!(serde_json::from_value::<NumericFp8Profile>(serde_json::json!(unknown)).is_err());
    }
}

#[test]
fn numeric_profile_reports_operands_separately_from_model_boundaries() {
    let profile = NumericFp8Profile::W8A16Bf16;
    for label in [
        profile.resident_backend_name(),
        profile.expert_backend_name(false),
        profile.expert_backend_name(true),
    ] {
        assert!(label.contains("w8a16-bf16-f32-accumulate"));
        assert!(!label.contains("f32-tf32x3"));
    }
    assert_eq!(
        NumericFp8Profile::F32.resident_backend_name(),
        Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE
    );
    let description = profile.arithmetic_description();
    assert!(description.contains("BF16-RNE operands"));
    assert!(description.contains("F32 accumulation/output/model/KV/state boundaries"));
    assert!(description.contains("required global numeric workspace=0"));
    assert!(description.contains("not end-to-end BF16 or same-precision Marlin"));
}

#[test]
fn numeric_profile_family_adapter_preserves_selection_and_rejects_unused_flags() {
    for selection in [
        None,
        Some(NumericFp8Profile::F32),
        Some(NumericFp8Profile::W8A16Bf16),
    ] {
        let mut input = options();
        input.numeric_fp8_profile = selection;
        let resolved = qwen35::resolve_family_options(&ModelFamily::Qwen35Moe, &input).unwrap();
        assert_eq!(
            resolved.qwen35().unwrap().moe().unwrap().numeric_profile,
            selection.unwrap_or_default()
        );
        assert_eq!(input.numeric_fp8_profile, selection);
        for family in [
            ModelFamily::Qwen35,
            ModelFamily::Qwen3,
            ModelFamily::QwenMoe,
        ] {
            let result = qwen35::resolve_family_options(&family, &input);
            if selection.is_some() {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("--numeric-fp8-precision")
                );
            } else {
                result.unwrap();
            }
        }
    }
}

#[test]
fn numeric_profile_capacity_reports_zero_required_scratch_but_keeps_reserved_budget() {
    let config = Qwen35Config::from_value(
        &serde_json::from_str(include_str!(
            "../../../../../ferrule-model/tests/qwen35_35b_fp8_config.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let limits = Qwen35MoeCapacityLimits::default();
    let f32 = Qwen35MoeCapacityPlan::from_config(&config, limits).unwrap();
    let fused = Qwen35MoeCapacityPlan::from_config_with_profile(
        &config,
        limits,
        NumericFp8Profile::W8A16Bf16,
    )
    .unwrap();
    assert_eq!(f32.numeric_profile, NumericFp8Profile::F32);
    assert_eq!(fused.numeric_profile, NumericFp8Profile::W8A16Bf16);
    assert!(f32.minimum_scratch_bytes > 0);
    assert_eq!(fused.minimum_scratch_bytes, 0);
    assert_eq!(fused.limits.scratch_bytes, 64 << 20);
    assert_eq!(fused.total_device_bytes, f32.total_device_bytes);
    assert_eq!(fused.resident_f32_bytes, f32.resident_f32_bytes);
    assert_eq!(fused.state_bytes, f32.state_bytes);
    assert_eq!(fused.kv_bytes, f32.kv_bytes);
    assert_eq!(fused.minimum_workspace_bytes, f32.minimum_workspace_bytes);
}

#[test]
fn numeric_profile_resolved_request_and_observability_agree() {
    use super::super::*;
    for selection in [
        None,
        Some(NumericFp8Profile::F32),
        Some(NumericFp8Profile::W8A16Bf16),
    ] {
        let descriptor = tests::descriptor(ModelFamily::Qwen35Moe, "metadata-only", 40);
        let mut input = options();
        input.max_tensor_mebibytes = 1024;
        input.driver_config.enable_native_proposals = false;
        input.numeric_fp8_profile = selection;
        let profile = selection.unwrap_or_default();
        let entry = ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen3.5-35b-a3b-fp8",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: profile.resident_backend_name(),
            default_chat_template: ChatTemplate::Qwen35,
        };
        // Metadata seam only: public CPU-only builds still reject CUDA execution.
        let request = configure_request(&descriptor, entry, input.clone()).unwrap();
        let mut plan = ResidentModelBuildPlan {
            prepared_source_plan: None,
            model_name: "qwen3.5-35b-a3b-fp8",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: profile.resident_backend_name(),
            chat_template: ChatTemplate::Qwen35,
            requested_backend: BackendSelection::Cuda,
            effective_options: input.clone(),
            topology: resident_topology(),
            provider_adjustments: AdjustmentReport::default(),
            requested_options: input,
            requested_parallel: None,
            request,
        };
        plan.refresh_facts().unwrap();
        assert_eq!(plan.requested_options().numeric_fp8_profile, selection);
        assert_eq!(plan.effective_options().numeric_fp8_profile, selection);
        assert_eq!(plan.numeric_fp8_profile(), Some(profile));
        assert_eq!(plan.observability().numeric_fp8_profile, Some(profile));
        assert!(
            plan.resolution_report()
                .contains(profile.arithmetic_description())
        );
        assert_eq!(
            plan.qwen35_moe_capacity_limits()
                .unwrap()
                .unwrap()
                .scratch_bytes,
            64 << 20
        );
        let frozen = plan.build_with(Ok).unwrap();
        assert_eq!(
            frozen
                .resolved
                .family_options
                .qwen35()
                .unwrap()
                .moe()
                .unwrap()
                .numeric_profile,
            profile
        );
    }
}
