//! Header-only public factory preflight: never initializes CUDA or reads payloads.
use ferrule_composition::model_factory::{Qwen35MoeCapacityLimits, Qwen35NumericBf16MemoryPlan};
use ferrule_composition::{ModelPrecision, NumericStoragePolicy};
use ferrule_model::transformer::NumericFp8Precision as P;

#[path = "mixed_fixture.rs"]
mod fixture;

#[test]
fn mixed_attention_context_rejected_at_public_factory_admission() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_attention_context_rejected_at_public_factory_admission"
        ),
        || {
            let fixture = fixture::Fixture::new();
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
            let open = |limits, profile| {
                Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
                    fixture.dir.path(),
                    ModelPrecision::Bf16,
                    NumericStoragePolicy::E4M3FnBf16Scales,
                    limits,
                    1 << 30,
                    profile,
                )
            };
            for rows in [1, 8] {
                let plan = open(
                    Qwen35MoeCapacityLimits {
                        max_batch_tokens: rows,
                        ..limits
                    },
                    P::Bf16ModelMixedF32Accumulate,
                )
                .unwrap();
                let attention = ferrule_model::decoder::HybridMemoryPlan::bf16_attention_plan(
                    plan.resources(),
                    plan.runner_options(),
                )
                .unwrap()
                .unwrap();
                assert_eq!(attention.storage_grant_bytes(), 37_748_736);
                assert_eq!(attention.provider_bytes(), 16_777_216);
                assert_eq!(
                    attention.warm_groups().len(),
                    if rows == 1 { 128 } else { 996 }
                );
                let metadata =
                    ferrule_model::decoder::HybridMemoryPlan::bf16_attention_metadata_bytes(
                        plan.resources(),
                        plan.runner_options(),
                    )
                    .unwrap();
                assert_eq!(metadata, 13_056);
                let baseline = if rows == 1 { 595_099_724 } else { 817_594_444 };
                assert_eq!(
                    plan.model_estimate().workspace_bytes_upper_bound,
                    baseline + attention.total_charge_bytes() + metadata
                );
                assert!(
                    open(
                        Qwen35MoeCapacityLimits {
                            max_batch_tokens: rows,
                            workspace_bytes: plan.model_estimate().workspace_bytes_upper_bound
                                - limits.scratch_bytes
                                - 1,
                            ..limits
                        },
                        P::Bf16ModelMixedF32Accumulate
                    )
                    .is_err()
                );
                assert!(
                    open(
                        Qwen35MoeCapacityLimits {
                            max_batch_tokens: rows,
                            workspace_bytes: plan.model_estimate().workspace_bytes_upper_bound
                                - limits.scratch_bytes,
                            ..limits
                        },
                        P::Bf16ModelMixedF32Accumulate
                    )
                    .is_ok()
                );
                assert!(
                    open(
                        Qwen35MoeCapacityLimits {
                            max_batch_tokens: rows,
                            device_bytes: plan.total_device_bytes() - 1,
                            ..limits
                        },
                        P::Bf16ModelMixedF32Accumulate
                    )
                    .is_err()
                );
                assert_eq!(plan.total_device_bytes(), 6_846_743_296);
                assert_eq!(limits.device_bytes - plan.total_device_bytes(), 669_449_472);
                eprintln!(
                    "Mixed attention admitted rows={rows}: workspace_demand={} workspace_cap={} total={} headroom={} margin={}",
                    plan.model_estimate().workspace_bytes_upper_bound,
                    plan.model_budget().workspace_bytes,
                    plan.total_device_bytes(),
                    limits.device_bytes - plan.total_device_bytes(),
                    limits.allocator_margin_bytes
                );
            }
            // Enough KV/budget to isolate provider context admission, not page scarcity.
            let long = Qwen35MoeCapacityLimits {
                max_positions: 1025,
                kv_bytes: 256 << 20,
                device_bytes: 16 << 30,
                ..limits
            };
            let error = open(long, P::Bf16ModelMixedF32Accumulate)
                .unwrap_err()
                .to_string();
            assert!(error.contains("initial image admission"), "{error}");
            assert!(open(long, P::Bf16ModelTiledF32Accumulate).is_ok());
            assert!(
                open(
                    Qwen35MoeCapacityLimits {
                        max_positions: 1024,
                        ..long
                    },
                    P::Bf16ModelMixedF32Accumulate
                )
                .is_ok()
            );
        },
    );
}
