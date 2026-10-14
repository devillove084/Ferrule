use super::*;
#[test]
#[ignore = "CPU source headers only; set FERRULE_NUMERIC_BF16_MODEL and hide GPUs"]
fn mixed_chunk_full_header_reserves_once_and_rejects_workspace_and_device_underbudget() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let model = std::path::PathBuf::from(std::env::var_os("FERRULE_NUMERIC_BF16_MODEL").unwrap());
    let limits = Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 8 << 20,
        workspace_bytes: 2 << 30,
        kv_bytes: 16 << 20,
        device_bytes: 7 << 30,
        allocator_margin_bytes: 512 << 20,
        max_positions: 128,
        max_sequences: 2,
        max_batch_tokens: 8,
    };
    let open = |limits, precision| {
        Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
            &model,
            ModelPrecision::Bf16,
            NumericStoragePolicy::E4M3FnBf16Scales,
            limits,
            1 << 30,
            precision,
        )
    };
    let mixed = open(limits, P::Bf16ModelMixedF32Accumulate).unwrap();
    let tiled = open(limits, P::Bf16ModelTiledF32Accumulate).unwrap();
    let one_row_limits = Qwen35MoeCapacityLimits {
        max_batch_tokens: 1,
        ..limits
    };
    let one_row = open(one_row_limits, P::Bf16ModelMixedF32Accumulate).unwrap();
    assert_eq!(
        ferrule_model::decoder::HybridMemoryPlan::bf16_chunk_workspace_bytes(
            one_row.resources(),
            one_row.runner_options()
        )
        .unwrap(),
        17_850_368
    );
    let one_row_workspace = one_row.model_estimate().workspace_bytes_upper_bound;
    assert!(
        open(
            Qwen35MoeCapacityLimits {
                device_bytes: one_row.total_device_bytes() - 1,
                ..one_row_limits
            },
            P::Bf16ModelMixedF32Accumulate
        )
        .is_err()
    );
    eprintln!(
        "MIXED_M1_HISTORY_BUDGET chunk_logical=17850368 chunk_grant=18874368 workspace_demand={} total={} cap={} headroom={} margin={}",
        one_row_workspace,
        one_row.total_device_bytes(),
        one_row_limits.device_bytes,
        one_row_limits.device_bytes - one_row.total_device_bytes(),
        one_row_limits.allocator_margin_bytes
    );

    assert!(
        open(
            Qwen35MoeCapacityLimits {
                workspace_bytes: one_row_workspace - one_row_limits.scratch_bytes - 1,
                ..one_row_limits
            },
            P::Bf16ModelMixedF32Accumulate
        )
        .is_err()
    );
    assert!(
        open(
            Qwen35MoeCapacityLimits {
                workspace_bytes: one_row_workspace - one_row_limits.scratch_bytes,
                ..one_row_limits
            },
            P::Bf16ModelMixedF32Accumulate
        )
        .is_ok()
    );

    let chunk = ferrule_model::decoder::HybridMemoryPlan::bf16_chunk_workspace_bytes(
        mixed.resources(),
        mixed.runner_options(),
    )
    .unwrap();
    assert_eq!(chunk, 17_850_368);
    let grant = ferrule_backend::cuda_allocation_grant_bytes(chunk).unwrap();
    assert_eq!(grant, 18_874_368);
    let (gemmex, provider) = ferrule_model::decoder::HybridMemoryPlan::bf16_gemmex_bytes(
        mixed.resources(),
        mixed.runner_options(),
    )
    .unwrap();
    assert_eq!(gemmex, 67_272_704);
    assert_eq!(provider, 16_777_216);
    let gemmex_grant = ferrule_backend::cuda_allocation_grant_bytes(gemmex).unwrap();
    assert_eq!(gemmex_grant, 69_206_016);
    let views = ferrule_model::decoder::HybridMemoryPlan::bf16_gemmex_plans(
        mixed.resources(),
        mixed.runner_options(),
    )
    .unwrap();
    assert!(views.iter().all(|p| p.rows() <= 8 && p.layout().n < 262144));
    for rows in [1, 5, 8] {
        assert!(
            views
                .iter()
                .any(|p| p.rows() == rows && p.layout().n == 256 && p.layout().k == 2048),
            "resident router must warm in the existing pool"
        );
        assert!(
            views.iter().any(|p| p.is_resident()
                && p.rows() == rows
                && p.layout().n == 32
                && p.layout().k == 2048),
            "resident A/B must warm in the same unchanged maximum pool"
        );
        assert!(
            views.iter().any(|p| p.is_resident()
                && p.rows() == rows
                && p.layout().n == 1
                && p.layout().k == 2048),
            "resident shared output gate must warm in the unchanged pool"
        );
    }

    assert_eq!(
        mixed.model_estimate().workspace_bytes_upper_bound,
        tiled.model_estimate().workspace_bytes_upper_bound
            + gemmex_grant
            + provider
            + grant
            + ferrule_model::decoder::HybridMemoryPlan::bf16_attention_plan(
                mixed.resources(),
                mixed.runner_options()
            )
            .unwrap()
            .unwrap()
            .total_charge_bytes()
            + ferrule_model::decoder::HybridMemoryPlan::bf16_attention_metadata_bytes(
                mixed.resources(),
                mixed.runner_options()
            )
            .unwrap()
    );
    assert_eq!(mixed.total_device_bytes(), 6_846_743_296);
    assert_eq!(
        limits.device_bytes - mixed.total_device_bytes(),
        669_449_472
    );
    let exact_workspace = Qwen35MoeCapacityLimits {
        workspace_bytes: mixed.model_estimate().workspace_bytes_upper_bound - limits.scratch_bytes,
        ..limits
    };
    assert!(open(exact_workspace, P::Bf16ModelMixedF32Accumulate).is_ok());
    let under_workspace = Qwen35MoeCapacityLimits {
        workspace_bytes: mixed.model_estimate().workspace_bytes_upper_bound
            - limits.scratch_bytes
            - 1,
        ..limits
    };
    assert!(
        open(under_workspace, P::Bf16ModelMixedF32Accumulate)
            .unwrap_err()
            .to_string()
            .contains("workspace")
    );
    let under_device = Qwen35MoeCapacityLimits {
        device_bytes: mixed.total_device_bytes() - 1,
        ..limits
    };
    assert!(
        open(under_device, P::Bf16ModelMixedF32Accumulate)
            .unwrap_err()
            .to_string()
            .contains("device byte cap")
    );
    eprintln!(
        "MIXED_GEMMEX_BUDGET storage={gemmex} gemmex_grant={gemmex_grant} external={provider} chunk_logical={chunk} grant={grant} workspace_demand={} workspace_cap={} total={} headroom={} margin={}",
        mixed.model_estimate().workspace_bytes_upper_bound,
        mixed.model_budget().workspace_bytes,
        mixed.total_device_bytes(),
        limits.device_bytes - mixed.total_device_bytes(),
        limits.allocator_margin_bytes
    );
}
