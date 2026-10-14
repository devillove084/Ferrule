use super::*;
use crate::model_factory::tests::{descriptor, options};
use ferrule_model::transformer::{NumericFp8Bf16HybridCudaProfile, NumericFp8Precision};
use ferrule_runtime::engine::KvExecutionPolicy;

#[path = "../../../qwen35_memory_plan/mixed_fixture.rs"]
pub(crate) mod fixture;

fn serving() -> ServingModelOptions {
    ServingModelOptions {
        model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
        ..Default::default()
    }
}
fn bounded_options() -> ModelFactoryOptions {
    let mut options = options();
    options.max_layers = None;
    options.output_head_chunk_rows = 4096;
    options.expert_reader_max_tensor_mebibytes = 64;
    options.driver_config.ctx_size = 128;
    options.scheduler_config.max_active_sequences = 2;
    options.scheduler_config.max_batch_tokens = 8;
    options.scheduler_config.prefill_chunk_size = 8;
    options.kv_cache_mebibytes = Some(16);
    options.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 8 << 20,
        workspace_bytes: 2 << 30,
        device_bytes: 7 << 30,
        ..Default::default()
    });
    options
}

#[test]
fn startup_workspace_is_literal_and_actionable() {
    for value in [None, Some(""), Some(":4096:8"), Some(":0:0 ")] {
        let error = ServingModelProfile::Qwen35MixedBf16
            .validate_startup_environment(value.map(std::ffi::OsStr::new))
            .unwrap_err();
        assert!(error.to_string().contains("CUBLAS_WORKSPACE_CONFIG=:0:0"));
        assert!(error.to_string().contains("restart"));
    }
    ServingModelProfile::Qwen35MixedBf16
        .validate_startup_environment(Some(std::ffi::OsStr::new(":0:0")))
        .unwrap();
}

#[test]
fn mixed_public_dispatch_freezes_existing_factory_and_budgets() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_public_dispatch_freezes_existing_factory_and_budgets"
        ),
        || {
            let f = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
            let plan = ResidentModelPlanner
                .prepare_serving(
                    &config,
                    BackendSelection::Cuda,
                    None,
                    bounded_options(),
                    serving(),
                )
                .unwrap();
            assert_eq!(
                plan.backend_profile(),
                NumericFp8Bf16HybridCudaProfile::MIXED_NAME
            );
            assert_eq!(
                plan.support().storage,
                StorageEncoding::NumericFp8E4m3fnBf16Scales
            );
            assert_eq!(
                plan.selected_implementation().kind,
                SelectedImplementationKind::Resident
            );
            let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
                super::super::bf16_serving::build_numeric_bf16;
            assert!(std::ptr::fn_addr_eq(
                plan.request.execution.construction.builder(),
                expected
            ));
            assert_eq!(plan.numeric_fp8_profile(), None);
            assert!(plan.pipeline_options().is_none());
            assert!(plan.qwen35_expert_placement().is_none());
            let limits = plan.qwen35_moe_capacity_limits().unwrap().unwrap();
            assert_eq!(
                (
                    limits.max_positions,
                    limits.max_sequences,
                    limits.max_batch_tokens
                ),
                (128, 2, 8)
            );
            assert_eq!(
                (
                    limits.device_bytes,
                    limits.workspace_bytes,
                    limits.scratch_bytes,
                    limits.kv_bytes
                ),
                (7 << 30, 2 << 30, 8 << 20, 16 << 20)
            );
            let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
            assert_eq!(
                memory.numeric_precision(),
                NumericFp8Precision::Bf16ModelMixedF32Accumulate
            );
            assert!(memory.total_device_bytes() <= limits.device_bytes);
            assert_eq!(
                plan.qwen35_host_cache_options().unwrap().mode,
                ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy
            );
            let direct = ResidentModelPlanner
                .prepare_numeric_fp8_bf16_serving_with_precision(
                    &config,
                    BackendSelection::Cuda,
                    None,
                    bounded_options(),
                    ServingModelOptions {
                        model_precision: Some(ModelPrecision::Bf16),
                        ..Default::default()
                    },
                    NumericStoragePolicy::E4M3FnBf16Scales,
                    NumericFp8Precision::Bf16ModelMixedF32Accumulate,
                )
                .unwrap();
            assert!(std::ptr::fn_addr_eq(
                plan.request.execution.construction.builder(),
                direct.request.execution.construction.builder()
            ));
            plan.build_with(|request: ModelBuildRequest| -> Result<()> {
                assert_eq!(
                    request.resolved.execution.numeric_bf16_precision,
                    Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate)
                );
                assert_eq!(
                    request.resolved.execution.kv_execution_policy,
                    KvExecutionPolicy::Resident
                );
                Ok(())
            })
            .unwrap();
            for change in 0..4 {
                let mut o = bounded_options();
                match change {
                    0 => o.qwen35_moe_capacity.as_mut().unwrap().device_bytes = 1,
                    1 => o.qwen35_moe_capacity.as_mut().unwrap().workspace_bytes = 1,
                    2 => o.kv_cache_mebibytes = Some(1),
                    _ => o.max_layers = Some(1),
                }
                assert!(
                    ResidentModelPlanner
                        .prepare_serving(&config, BackendSelection::Cuda, None, o, serving())
                        .is_err(),
                    "accepted invalid metadata budget {change}"
                );
            }
            #[cfg(not(feature = "cuda"))]
            assert!(
                direct
                    .build()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("no CPU fallback")
            );
        },
    );
}

#[test]
fn mixed_dispatch_rejects_unsupported_policy_before_checkpoint_io() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_dispatch_rejects_unsupported_policy_before_checkpoint_io"
        ),
        || {
            let config = AutoConfig::from_descriptor(descriptor(
                ModelFamily::Qwen35Moe,
                "must-not-read",
                40,
            ));
            for backend in [BackendSelection::Cpu, BackendSelection::Auto] {
                let error = ResidentModelPlanner
                    .prepare_serving(&config, backend, None, bounded_options(), serving())
                    .err()
                    .unwrap();
                assert!(error.to_string().contains("explicit CUDA"), "{error}");
            }
            for change in 0..12 {
                let mut s = serving();
                let mut o = bounded_options();
                match change {
                    0 => s.parallel = Some(PipelineBuildOptions::default()),
                    1 => s.engine = ServingEngineSelection::Pipeline,
                    2 => {
                        s.kv_execution_policy = KvExecutionPolicy::HostKvDecode(Default::default())
                    }
                    3 => s.model_precision = Some(ModelPrecision::F32),
                    4 => o.numeric_fp8_profile = Some(NumericFp8Profile::W8A16Bf16),
                    5 => o.qwen35_host_cache = Some(Default::default()),
                    6 => s.expert_host_cache_entries = Some(0),
                    7 => s.expert_pinned_cache_entries = Some(0),
                    8 => s.moe_hotset_experts = Some(0),
                    9 => o.scheduler_config.max_batch_tokens = 33,
                    10 => o.driver_config.ctx_size = 1,
                    _ => {
                        s.kv_execution_policy = KvExecutionPolicy::IdleReclaim(
                            ferrule_model::decoder::IdleKvReclaimConfig {
                                staging_page_limit: 1,
                                device_bytes: 1 << 20,
                                host_bytes: 1 << 20,
                                pinned_bytes: 1 << 20,
                            },
                        )
                    }
                }
                let error = ResidentModelPlanner
                    .prepare_serving(&config, BackendSelection::Cuda, None, o, s)
                    .err()
                    .unwrap();
                assert!(
                    error.to_string().contains("BF16")
                        || error.to_string().contains("batch")
                        || error.to_string().contains("precision"),
                    "case {change}: {error}"
                );
            }
        },
    );
}
