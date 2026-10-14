use super::fixture::*;
use ferrule_composition::model_factory::{
    ArithmeticPrecision, BuildabilityStatus, NumericFp8Profile, PipelineBuildOptions,
    PipelineRankBackend, StorageEncoding,
};
use ferrule_composition::{
    BackendSelection, ModelPrecision, ResidentModelPlanner, ServingEngineSelection,
    ServingModelOptions,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::{
    ModelExecutionBackend, ModelFamily, models::qwen3::Qwen3DenseAdapter,
    transformer::LayerSegmentPlan,
};
use ferrule_runtime::KvExecutionPolicy;
use ferrule_runtime::parallel::pipeline::{
    DenseBf16MemoryPlan, PipelineConfig, PipelineStageDescription,
};

#[test]
fn public_factory_selects_real_bf16_builder_and_typed_budget_without_cuda() {
    let f = Fixture::new();
    let plan = ResidentModelPlanner
        .prepare_serving(
            &f.config(),
            BackendSelection::Cuda,
            None,
            options(),
            serving(),
        )
        .unwrap();
    assert_eq!(plan.backend(), ModelExecutionBackend::Cuda);
    assert_eq!(plan.backend_profile(), "cuda-dense-bf16-serial");
    assert_eq!(plan.support().storage, StorageEncoding::NativeBf16);
    assert_eq!(
        plan.support().arithmetic,
        ArithmeticPrecision::Bf16Compatibility
    );
    assert_eq!(
        plan.support().buildability,
        if cfg!(feature = "cuda") {
            BuildabilityStatus::CatalogValidated
        } else {
            BuildabilityStatus::FeatureUnavailable
        }
    );
    assert_eq!(
        plan.pipeline_options().unwrap().parallelism,
        Default::default()
    );
    assert_eq!(plan.numeric_fp8_profile(), None);
    assert_eq!(
        plan.effective_options().scheduler_config.max_decode_batch,
        1
    );
    assert!(
        !plan
            .effective_options()
            .scheduler_config
            .allow_mixed_batches
    );
    assert!(
        !plan
            .effective_options()
            .driver_config
            .enable_native_proposals
    );
    let memory = plan.dense_bf16_memory_plan().unwrap().unwrap();
    let weights: u64 = f.oracle["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["values"].as_array().unwrap().len() as u64 * 2)
        .sum();
    assert_eq!(memory.weight_bytes, weights);
    assert!(memory.bf16_rows_bytes > 0 && memory.f32_work_bytes > 0);
    assert_eq!(memory.gemm_workspace_bytes, 0);
    memory
        .check_available(memory.total_device_bytes as usize)
        .unwrap();
    assert!(
        memory
            .check_available(memory.total_device_bytes as usize - 1)
            .is_err()
    );
    assert!(plan.with_pipeline(PipelineBuildOptions::default()).is_err());
    #[cfg(not(feature = "cuda"))]
    {
        let plan = ResidentModelPlanner
            .prepare_serving(
                &f.config(),
                BackendSelection::Cuda,
                None,
                options(),
                serving(),
            )
            .unwrap();
        assert!(plan.build().err().unwrap().to_string().contains("cuda"));
    }
}

#[test]
fn native_weights_and_read_limit_are_checked_not_torch_dtype_metadata() {
    let f = Fixture::new();
    f.write(true);
    let error = ResidentModelPlanner
        .prepare_serving(
            &f.config(),
            BackendSelection::Cuda,
            None,
            options(),
            serving(),
        )
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("state-dict binding"),
        "raw F32 payload must fail strict BF16 binding: {error}"
    );
    f.write(false);
    let (_, checkpoint) = Qwen3DenseAdapter::bind_hf_metadata(f.dir.path()).unwrap();
    let resources = checkpoint.into_resources();
    let mut config = pipeline_config();
    config.max_parameter_bytes = 1;
    assert!(DenseBf16MemoryPlan::new(&resources, &full_plan(), config).is_err());
}
fn full_plan() -> LayerSegmentPlan {
    LayerSegmentPlan::new(2, 0..2, true, true).unwrap()
}
fn pipeline_config() -> PipelineConfig {
    PipelineConfig {
        page_size: 16,
        max_pages: 2,
        max_positions: 16,
        max_batch_tokens: 3,
        session_capacity: 2,
        max_parameter_bytes: 1 << 20,
        precision: ExecutionPrecisionPolicy::bf16_compatibility(),
        max_ack_polls: 8,
    }
}
#[test]
fn typed_kv_is_half_f32_and_rows_and_work_scale_with_batch_not_layers() {
    let f = Fixture::new();
    let (_, checkpoint) = Qwen3DenseAdapter::bind_hf_metadata(f.dir.path()).unwrap();
    let resources = checkpoint.into_resources();
    let config = pipeline_config();
    let budget = DenseBf16MemoryPlan::new(&resources, &full_plan(), config).unwrap();
    let mut f32 = config;
    f32.precision = ExecutionPrecisionPolicy::f32();
    let desc = PipelineStageDescription {
        plan: full_plan(),
        config: f32,
        physical_pages: 4,
        hidden: 16,
        vocabulary: 16,
        kv_heads: 1,
        head_dim: 8,
        expert_group: None,
    };
    assert_eq!(budget.kv_bytes * 2, desc.physical_bytes().unwrap());
    assert_eq!(budget.kv_bytes, 2 * 2 * 4 * 16 * 8 * 2);
    let mut one = config;
    one.max_batch_tokens = 1;
    let smaller = DenseBf16MemoryPlan::new(&resources, &full_plan(), one).unwrap();
    assert_eq!(budget.bf16_rows_bytes, smaller.bf16_rows_bytes * 3);
    assert_eq!(budget.f32_work_bytes, smaller.f32_work_bytes * 3);
    assert!(
        DenseBf16MemoryPlan::new(
            &resources,
            &LayerSegmentPlan::new(2, 0..1, true, false).unwrap(),
            config
        )
        .is_err()
    );
    let mut overflow = config;
    overflow.max_batch_tokens = usize::MAX;
    assert!(DenseBf16MemoryPlan::new(&resources, &full_plan(), overflow).is_err());
}

#[test]
fn incompatible_policies_and_partial_or_distributed_profiles_fail_closed() {
    let f = Fixture::new();
    for kind in 0..16 {
        let mut o = options();
        let mut s = serving();
        let mut backend = BackendSelection::Cuda;
        match kind {
            0 => backend = BackendSelection::Cpu,
            1 => s.kv_execution_policy = KvExecutionPolicy::HostKvDecode(Default::default()),
            2 => s.engine = ServingEngineSelection::Resident,
            3 => o.max_layers = Some(1),
            4 => o.numeric_fp8_profile = Some(NumericFp8Profile::W8A16Bf16),
            5 => o.numeric_fp8_profile = Some(NumericFp8Profile::F32),
            6 => s.expert_host_cache_entries = Some(0),
            7 => s.moe_hotset_experts = Some(0),
            8 => o.scheduler_config.prefix_cache_capacity_pages = 1,
            15 => {
                s.kv_execution_policy =
                    KvExecutionPolicy::IdleReclaim(ferrule_runtime::IdleKvReclaimConfig {
                        staging_page_limit: 1,
                        device_bytes: 1 << 20,
                        host_bytes: 1 << 20,
                        pinned_bytes: 1 << 20,
                    })
            }
            n => {
                let mut p = PipelineBuildOptions::default();
                match n {
                    9 => p.parallelism.tensor_parallel = 2,
                    10 => p.parallelism.pipeline_parallel = 2,
                    11 => p.parallelism.expert_parallel = 2,
                    12 => p.rank_backend = PipelineRankBackend::Process,
                    13 => p.rank_restarts = 1,
                    _ => p.devices = Some(vec![0, 1]),
                }
                s.parallel = Some(p);
            }
        }
        assert!(
            ResidentModelPlanner
                .prepare_serving(&f.config(), backend, None, o, s)
                .is_err(),
            "accepted case {kind}"
        );
    }
    for family in [
        ModelFamily::Qwen35,
        ModelFamily::Qwen35Moe,
        ModelFamily::QwenMoe,
    ] {
        let mut descriptor = f.config().descriptor().clone();
        descriptor.spec.family = family;
        assert!(
            ResidentModelPlanner
                .prepare_serving(
                    &ferrule_model::AutoConfig::from_descriptor(descriptor),
                    BackendSelection::Cuda,
                    None,
                    options(),
                    serving()
                )
                .is_err()
        );
    }
}

#[test]
fn omitted_default_keeps_cpu_catalog_and_explicit_f32_does_not_lie_about_cpu_bf16() {
    let f = Fixture::new();
    let plan = ResidentModelPlanner
        .prepare_serving(
            &f.config(),
            BackendSelection::Auto,
            None,
            options(),
            ServingModelOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.backend(), ModelExecutionBackend::Cpu);
    assert_eq!(plan.backend_profile(), "cpu-standard-decoder");
    assert_eq!(plan.support().storage, StorageEncoding::ProfileDefined);
    assert!(plan.dense_bf16_memory_plan().unwrap().is_none());
    let s = ServingModelOptions {
        model_precision: Some(ModelPrecision::F32),
        ..Default::default()
    };
    assert!(
        ResidentModelPlanner
            .prepare_serving(&f.config(), BackendSelection::Cpu, None, options(), s)
            .is_err()
    );
}

#[test]
fn tied_native_head_is_counted_once_and_over_budget_kv_fails_before_build() {
    let mut f = Fixture::new();
    f.oracle["config"]["tie_word_embeddings"] = serde_json::json!(true);
    f.oracle["weights"]
        .as_array_mut()
        .unwrap()
        .retain(|w| w["name"] != "lm_head.weight");
    f.write(false);
    let plan = ResidentModelPlanner
        .prepare_serving(
            &f.config(),
            BackendSelection::Cuda,
            None,
            options(),
            serving(),
        )
        .unwrap();
    let expected = f.oracle["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["values"].as_array().unwrap().len() as u64 * 2)
        .sum::<u64>();
    assert_eq!(
        plan.dense_bf16_memory_plan().unwrap().unwrap().weight_bytes,
        expected
    );
    let mut o = options();
    o.scheduler_config.max_active_sequences = 1024;
    let error = ResidentModelPlanner
        .prepare_serving(&f.config(), BackendSelection::Cuda, None, o, serving())
        .err()
        .unwrap();
    assert!(error.to_string().contains("budget"), "{error}");
}
