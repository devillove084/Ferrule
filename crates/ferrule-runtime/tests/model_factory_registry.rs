use std::path::PathBuf;

use ferrule_model::{
    AttentionKind, ChatTemplate, ModelDescriptor, ModelExecutionBackend, ModelFamily, MoeSpec,
    RouterKind, TransformerSemantics, TransformerSpec, WeightSource,
};
use ferrule_runtime::{BackendSelection, BuiltinModelResolver};

fn descriptor(family: ModelFamily, architecture: &str) -> ModelDescriptor {
    ModelDescriptor {
        path: PathBuf::from("model"),
        spec: TransformerSpec {
            family,
            architecture: Some(architecture.into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(64),
            num_layers: Some(48),
            vocab_size: Some(128),
            num_heads: Some(4),
            num_kv_heads: Some(4),
            head_dim: Some(16),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec::none(),
            semantics: TransformerSemantics::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        },
        tensor_classes: Vec::new(),
    }
}

#[test]
fn owner_engine_has_no_send_or_leak_escape_hatch() {
    let source = include_str!("../src/engine/inference.rs");

    assert!(source.contains("pub trait InferenceEngine: 'static"));
    assert!(!source.contains("OwnerLocal"));
    assert!(!source.contains("unsafe impl"));
    assert!(!source.contains("mem::forget"));
}

fn qwen_descriptor() -> ModelDescriptor {
    let mut descriptor = descriptor(ModelFamily::QwenMoe, "qwen3_moe");
    descriptor.spec.attention = AttentionKind::GroupedQuery;
    descriptor.spec.num_kv_heads = Some(1);
    descriptor.spec.moe = MoeSpec {
        num_experts: Some(128),
        num_experts_per_tok: Some(8),
        has_shared_experts: false,
        router: RouterKind::DenseTopK,
    };
    descriptor
}

#[test]
fn qwen_auto_registers_cpu_backend() {
    let entry = BuiltinModelResolver::new()
        .resolve(&qwen_descriptor(), BackendSelection::Auto)
        .unwrap();

    assert_eq!(entry.model_name(), "qwen3-moe");
    assert_eq!(entry.backend(), ModelExecutionBackend::Cpu);
    assert_eq!(entry.backend_profile(), "cpu-standard-decoder");
    assert_eq!(entry.default_chat_template(), ChatTemplate::Qwen3);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn deepseek_auto_is_known_unavailable_without_cuda() {
    let error = BuiltinModelResolver::new()
        .resolve(
            &descriptor(ModelFamily::DeepSeekV4, "deepseek4"),
            BackendSelection::Auto,
        )
        .unwrap_err();

    assert!(error.to_string().contains("requires CUDA"));
    assert!(
        error
            .to_string()
            .contains("compiled without the 'cuda' feature")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_build_registers_deepseek_auto_backend() {
    let entry = BuiltinModelResolver::new()
        .resolve(
            &descriptor(ModelFamily::DeepSeekV4, "deepseek4"),
            BackendSelection::Auto,
        )
        .unwrap();

    assert_eq!(entry.model_name(), "deepseek-v4");
    assert_eq!(entry.backend(), ModelExecutionBackend::Cuda);
    assert_eq!(entry.backend_profile(), "cuda");
    assert_eq!(entry.default_chat_template(), ChatTemplate::DeepSeekV4);
}

mod qwen35_moe_planning {
    use super::*;
    use ferrule_model::models::qwen35::Qwen35Config;
    use ferrule_runtime::engine::model_factory::{
        ExpertCacheOptions, PipelineBuildOptions, PipelineRankBackend, Qwen35MoeCapacityLimits,
        Qwen35MoeCapacityPlan,
    };
    use ferrule_runtime::{ModelFactoryOptions, ResidentModelPlanner};

    fn config() -> Qwen35Config {
        Qwen35Config::from_value(
            &serde_json::from_str(include_str!(
                "../../ferrule-model/tests/qwen35_35b_fp8_config.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn long_context_moe_schema_profile_1k_2k_4k_and_no_overcommit() {
        let config = config();
        for context in [1024, 2048, 4096] {
            for sequences in [1, 4] {
                let defaults = Qwen35MoeCapacityLimits::default();
                let limits = Qwen35MoeCapacityLimits {
                    max_positions: context,
                    max_sequences: sequences,
                    ..defaults
                };
                let physical_bytes = context * sequences * 10 * 2 * 2 * 256 * 4 * 2;
                let default_plan = Qwen35MoeCapacityPlan::from_config(&config, limits);
                assert_eq!(default_plan.is_ok(), physical_bytes <= defaults.kv_bytes);
                let limits = Qwen35MoeCapacityLimits {
                    kv_bytes: physical_bytes,
                    ..limits
                };
                let plan = Qwen35MoeCapacityPlan::from_config(&config, limits).unwrap();
                let profile = plan.context_capacity_profile(16).unwrap();
                for case in 0..4 {
                    let mut malformed = plan.clone();
                    match case {
                        0 => malformed.kv_pages = 0,
                        1 => malformed.kv_bytes = usize::MAX,
                        2 => malformed.limits.max_sequences = usize::MAX,
                        _ => malformed.state_bytes = usize::MAX,
                    }
                    assert!(malformed.context_capacity_profile(16).is_err());
                }
                assert_eq!(profile.ctx_size, context);
                assert_eq!(profile.max_active_sequences, sequences);
                assert_eq!(profile.max_batch_tokens, Some(32));
                assert_eq!(profile.prefill_chunk_size, 16);
                assert_eq!(
                    profile.kv.unwrap().full_capacity_pages,
                    context / 16 * sequences
                );
                assert_eq!(
                    profile.kv.unwrap().configured_pages,
                    context / 16 * sequences * 2
                );
                assert_eq!(profile.logical_kv_bytes, Some((physical_bytes / 2) as u64));
                assert_eq!(
                    profile.kv.unwrap().configured_bytes,
                    Some(physical_bytes as u64)
                );
                assert_eq!(
                    profile.recurrent_bytes,
                    Some((2 * sequences + 1) * 30 * (8192 * 4 + 32 * 128 * 128) * 4)
                );
                assert_eq!(profile.workspace_bytes, Some(limits.workspace_bytes));
                assert_eq!(profile.root_required_bytes, Some(plan.total_device_bytes));
                assert!(profile.validate_request(0, context - 32, 32).is_ok());
                assert!(profile.validate_request(0, context - 32, 33).is_err());
                assert!(
                    Qwen35MoeCapacityPlan::from_config(
                        &config,
                        Qwen35MoeCapacityLimits {
                            kv_bytes: physical_bytes - 1,
                            ..limits
                        }
                    )
                    .is_err()
                );
                assert!(
                    Qwen35MoeCapacityPlan::from_config(
                        &config,
                        Qwen35MoeCapacityLimits {
                            device_bytes: plan.total_device_bytes - 1,
                            ..limits
                        }
                    )
                    .is_err()
                );
                eprintln!("schema arithmetic, not GPU throughput: {profile:?}");
            }
        }
    }

    #[test]
    fn full_model_capacity_keeps_only_bounded_compressed_experts() {
        let config = config();
        let limits = Qwen35MoeCapacityLimits::default();
        let plan = Qwen35MoeCapacityPlan::from_config(&config, limits).unwrap();
        eprintln!("35B planning only: {plan:#?}");
        // Independent geometry, including untied embedding/head and ALL layers.
        let embedding_and_head = 2 * 248_320 * 2048 * 4;
        let common = 40 * (2 * 2048 + 256 * 2048 + 2048) * 4;
        let linear = 30 * (2 * 32 * 2048 + 8192 * 4 + 32 + 32 + 128) * 4;
        assert_eq!(
            plan.resident_f32_bytes,
            embedding_and_head + common + linear + 2048 * 4 + 10 * 2 * 256 * 4
        );
        let projections = 10 * ((8192 + 512 + 512) * 2048 + 2048 * 4096)
            + 30 * (8192 * 2048 + 4096 * 2048 + 2048 * 4096)
            + 40 * 3 * 512 * 2048;
        assert_eq!(
            plan.compressed_projection_bytes,
            projections + projections / 8192
        );
        assert_eq!(plan.one_expert_bytes, 3 * (512 * 2048 + 4 * 16 * 2));
        assert_eq!(
            plan.routed_expert_storage_bytes,
            40 * 256 * plan.one_expert_bytes
        );
        assert_eq!(
            plan.per_sequence_state_bytes,
            30 * (8192 * 4 + 32 * 128 * 128) * 4
        );
        assert_eq!(plan.state_bytes, 9 * plan.per_sequence_state_bytes);
        assert_eq!(plan.kv_pages, 512);
        assert_eq!(plan.kv_bytes, 320 << 20);
        assert_eq!(plan.rotary_bytes, 10 * 1024 * 64 * 4);
        assert_eq!(plan.minimum_scratch_bytes, 4096 * 4);
        assert_eq!(
            plan.total_device_bytes,
            plan.resident_f32_bytes
                + plan.compressed_projection_bytes
                + plan.rotary_bytes
                + plan.state_bytes
                + plan.kv_bytes
                + limits.workspace_bytes
                + limits.max_bytes
                + limits.allocator_margin_bytes
        );
        assert_eq!(limits.max_experts, 1024);
        assert_eq!(limits.max_bytes, 4usize << 30);
        assert_eq!(limits.scratch_bytes, 64 << 20);
        assert_eq!(plan.one_expert_bytes, 3_146_112);
        assert_eq!(
            1024 * plan.one_expert_bytes + limits.scratch_bytes,
            3_288_727_552
        );
        assert!(plan.total_device_bytes < 12usize << 30);
        assert!(plan.routed_expert_storage_bytes > 30usize << 30);
        assert!(!config.supports_execution());
        assert!(Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE.contains("fp8-f32-tf32x3"));
        plan.check_available_device_bytes(plan.total_device_bytes)
            .unwrap();
        assert!(
            plan.check_available_device_bytes(plan.total_device_bytes - 1)
                .is_err()
        );
    }

    #[test]
    fn every_budget_is_checked_without_payload_or_cuda() {
        let config = config();
        let defaults = Qwen35MoeCapacityLimits::default();
        let plan = Qwen35MoeCapacityPlan::from_config(&config, defaults).unwrap();
        for case in 0..17 {
            let mut limits = defaults;
            match case {
                0 => limits.max_experts = 0,
                1 => limits.max_experts = 7,
                2 => limits.max_experts = usize::MAX,
                3 => limits.max_bytes = 0,
                4 => {
                    limits.max_bytes =
                        limits.max_experts * plan.one_expert_bytes + limits.scratch_bytes - 1
                }
                5 => limits.scratch_bytes = plan.minimum_scratch_bytes - 1,
                6 => limits.workspace_bytes = plan.minimum_workspace_bytes - 1,
                7 => limits.kv_bytes = plan.kv_bytes - 1,
                8 => limits.device_bytes = plan.total_device_bytes - 1,
                9 => limits.max_positions = 0,
                10 => limits.max_positions = 262_145,
                11 => limits.max_sequences = 0,
                12 => limits.max_sequences = usize::MAX,
                13 => limits.max_batch_tokens = 0,
                14 => limits.max_batch_tokens = 33,
                15 => limits.allocator_margin_bytes = usize::MAX,
                _ => limits.max_bytes = usize::MAX,
            }
            assert!(
                Qwen35MoeCapacityPlan::from_config(&config, limits).is_err(),
                "case {case}"
            );
        }
        let mut exact = defaults;
        exact.max_bytes = exact.max_experts * plan.one_expert_bytes + exact.scratch_bytes;
        exact.workspace_bytes = plan.minimum_workspace_bytes;
        exact.kv_bytes = plan.kv_bytes;
        assert!(Qwen35MoeCapacityPlan::from_config(&config, exact).is_ok());
        let dense = Qwen35Config::from_value(
            &serde_json::from_str(include_str!(
                "../../ferrule-model/tests/qwen35_08b_config.json"
            ))
            .unwrap(),
        )
        .unwrap();
        assert!(Qwen35MoeCapacityPlan::from_config(&dense, defaults).is_err());
    }

    #[test]
    fn changing_cache_bound_does_not_turn_all_experts_into_resident_weights() {
        let config = config();
        let limits = Qwen35MoeCapacityLimits::default();
        let a = Qwen35MoeCapacityPlan::from_config(&config, limits).unwrap();
        let b = Qwen35MoeCapacityPlan::from_config(
            &config,
            Qwen35MoeCapacityLimits {
                max_experts: 8,
                max_bytes: 8 * a.one_expert_bytes + limits.scratch_bytes,
                ..limits
            },
        )
        .unwrap();
        assert_eq!(a.resident_f32_bytes, b.resident_f32_bytes);
        assert_eq!(a.compressed_projection_bytes, b.compressed_projection_bytes);
        assert_eq!(a.routed_expert_storage_bytes, b.routed_expert_storage_bytes);
        assert_eq!(
            a.total_device_bytes - b.total_device_bytes,
            a.limits.max_bytes - b.limits.max_bytes
        );
    }

    #[test]
    fn policy_is_explicit_bounded_never_keep_all() {
        use ferrule_model::transformer::{ExpertCacheLimits, ExpertCachePolicy};
        let limits = Qwen35MoeCapacityLimits::default();
        assert_eq!(
            limits.expert_cache_policy().unwrap(),
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: limits.max_experts,
                max_bytes: limits.max_bytes,
            })
        );
        for limits in [
            Qwen35MoeCapacityLimits {
                max_experts: 0,
                ..limits
            },
            Qwen35MoeCapacityLimits {
                scratch_bytes: 0,
                ..limits
            },
            Qwen35MoeCapacityLimits {
                max_bytes: limits.scratch_bytes,
                ..limits
            },
        ] {
            assert!(limits.expert_cache_policy().is_err());
        }
    }

    #[cfg(feature = "cuda")]
    mod model_admission {
        use super::*;
        use ferrule_model::models::qwen35::{
            Qwen35Adapter, Qwen35HfNameMapper, Qwen35TensorPartitionKind,
        };
        use ferrule_runtime::engine::model_factory::Qwen35MoeCudaAdmission;
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Headers(PathBuf);
        impl Headers {
            fn new() -> Self {
                static NEXT: AtomicUsize = AtomicUsize::new(0);
                let dir = std::env::temp_dir().join(format!(
                    "qwen35-runtime-admission-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&dir).unwrap();
                let fixture = Self(dir);
                std::fs::write(
                    fixture.0.join("config.json"),
                    include_str!("../../ferrule-model/tests/qwen35_35b_fp8_config.json"),
                )
                .unwrap();
                let mapper = Qwen35HfNameMapper::new(&config());
                let mut header = serde_json::Map::new();
                let mut offset = 0u64;
                for tensor in mapper
                    .tensors()
                    .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
                {
                    let end = offset + tensor.bytes();
                    header.insert(
                        tensor.external_name.clone(),
                        serde_json::json!({
                            "dtype": tensor.dtype.as_str(), "shape": tensor.shape,
                            "data_offsets": [offset, end]
                        }),
                    );
                    offset = end;
                }
                let mut header = serde_json::to_vec(&header).unwrap();
                while !header.len().is_multiple_of(8) {
                    header.push(b' ');
                }
                let mut file = std::fs::File::create(fixture.0.join("model.safetensors")).unwrap();
                file.write_all(&(header.len() as u64).to_le_bytes())
                    .unwrap();
                file.write_all(&header).unwrap();
                // Sparse extents are valid metadata, NOT loaded model weights.
                file.set_len(8 + header.len() as u64 + offset).unwrap();
                fixture
            }
        }
        impl Drop for Headers {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn long_context_owner_profile_uses_model_budget_not_schema_target() {
            let fixture = Headers::new();
            for context in [1024, 2048, 4096] {
                let limits = Qwen35MoeCapacityLimits {
                    max_positions: context,
                    kv_bytes: context * 4 * 10 * 2 * 2 * 256 * 4 * 2,
                    device_bytes: 32usize << 30,
                    ..Default::default()
                };
                let admission =
                    Qwen35MoeCudaAdmission::open_hf(&fixture.0, limits, 1 << 30).unwrap();
                let default_device_bytes = Qwen35MoeCapacityLimits::default().device_bytes;
                if admission.total_device_bytes() > default_device_bytes {
                    assert!(
                        Qwen35MoeCudaAdmission::open_hf(
                            &fixture.0,
                            Qwen35MoeCapacityLimits {
                                device_bytes: default_device_bytes,
                                ..limits
                            },
                            1 << 30
                        )
                        .is_err()
                    );
                }
                let profile = admission.context_capacity_profile(16).unwrap();
                let budget = admission.model_budget();
                let estimate = admission.model_estimate();
                assert_eq!(profile.ctx_size, context);
                assert_eq!(profile.max_batch_tokens, Some(32));
                assert_eq!(profile.prefill_chunk_size, 16);
                assert_eq!(profile.kv.unwrap().configured_pages, budget.kv_pages);
                assert_eq!(
                    profile.kv.unwrap().configured_bytes,
                    Some(estimate.kv_bytes as u64)
                );
                assert_eq!(profile.recurrent_bytes, Some(budget.state_bytes));
                assert_eq!(profile.workspace_bytes, Some(budget.workspace_bytes));
                assert_eq!(
                    profile.root_required_bytes,
                    Some(admission.total_device_bytes())
                );
                assert_eq!(
                    budget.workspace_bytes,
                    limits.workspace_bytes + limits.scratch_bytes
                );
                assert_eq!(
                    admission.total_device_bytes(),
                    budget.state_bytes
                        + budget.workspace_bytes
                        + budget.weight_bytes
                        + estimate.kv_bytes
                        + limits.allocator_margin_bytes
                );
                assert!(
                    admission
                        .check_available_device_bytes(admission.total_device_bytes() - 1)
                        .is_err()
                );
                eprintln!("owner estimator only; no CUDA initialized: {profile:?}");
            }
        }

        #[test]
        fn real_model_options_estimator_and_budget_are_wired_without_cuda_initialization() {
            let fixture = Headers::new();
            let limits = Qwen35MoeCapacityLimits::default();
            let admission = Qwen35MoeCudaAdmission::open_hf(&fixture.0, limits, 1 << 30).unwrap();
            assert_eq!(
                admission
                    .runner_options()
                    .hybrid_cuda_numeric_fp8_precision(),
                Some(ferrule_model::transformer::NumericFp8Precision::F32Tf32x3),
            );
            let (cache, scratch) = admission
                .runner_options()
                .hybrid_cuda_numeric_fp8()
                .unwrap();
            assert_eq!(cache.max_experts, limits.max_experts);
            assert_eq!(cache.max_bytes, limits.max_bytes);
            assert_eq!(scratch, limits.scratch_bytes);
            assert_eq!(
                admission.runner_options().max_positions(),
                limits.max_positions
            );
            assert_eq!(
                admission.runner_options().capabilities().max_sequences,
                limits.max_sequences
            );
            assert_eq!(
                admission.runner_options().capabilities().max_batch_tokens,
                limits.max_batch_tokens
            );
            let (_, resources) = Qwen35Adapter::bind_hf_metadata(&fixture.0).unwrap();
            let model = ferrule_model::decoder::HybridCudaMemoryEstimate::for_resources(
                &resources,
                admission.runner_options(),
                admission.model_budget().kv_pages,
            )
            .unwrap();
            assert_eq!(admission.model_estimate(), model);
            let budget = admission.model_budget();
            assert_eq!(budget.weight_bytes, model.weight_bytes_upper_bound);
            assert_eq!(budget.workspace_bytes, limits.workspace_bytes + scratch);
            assert!(budget.workspace_bytes >= model.workspace_bytes_upper_bound);
            assert_eq!(
                budget.state_bytes,
                (2 * limits.max_sequences + 1) * model.per_sequence_state_bytes
            );
            assert_eq!(
                admission.total_device_bytes(),
                budget.weight_bytes
                    + budget.workspace_bytes
                    + budget.state_bytes
                    + model.kv_bytes
                    + limits.allocator_margin_bytes
            );
            assert!(admission.total_device_bytes() <= limits.device_bytes);
            assert!(admission.total_device_bytes() < 24usize << 30);
            admission
                .check_available_device_bytes(admission.total_device_bytes())
                .unwrap();
            assert!(
                admission
                    .check_available_device_bytes(admission.total_device_bytes() - 1)
                    .is_err()
            );
            assert!(
                Qwen35MoeCudaAdmission::open_hf(
                    &fixture.0,
                    Qwen35MoeCapacityLimits {
                        device_bytes: admission.total_device_bytes() - 1,
                        ..limits
                    },
                    1 << 30
                )
                .is_err()
            );
            assert!(Qwen35MoeCudaAdmission::open_hf(&fixture.0, limits, 128 << 20).is_err());
            eprintln!("35B metadata-only admission: {admission:#?}");
        }

        #[test]
        fn custom_bounded_policy_reaches_estimator_instead_of_using_defaults() {
            let fixture = Headers::new();
            let limits = Qwen35MoeCapacityLimits {
                max_experts: 8,
                max_bytes: 128 << 20,
                scratch_bytes: 32 << 20,
                max_positions: 64,
                max_sequences: 1,
                max_batch_tokens: 4,
                ..Qwen35MoeCapacityLimits::default()
            };
            let admission = Qwen35MoeCudaAdmission::open_hf(&fixture.0, limits, 1 << 30).unwrap();
            assert_eq!(
                admission
                    .runner_options()
                    .hybrid_cuda_numeric_fp8_precision(),
                Some(ferrule_model::transformer::NumericFp8Precision::F32Tf32x3),
            );
            let (cache, scratch) = admission
                .runner_options()
                .hybrid_cuda_numeric_fp8()
                .unwrap();
            assert_eq!(
                (cache.max_experts, cache.max_bytes, scratch),
                (8, 128 << 20, 32 << 20)
            );
            assert_eq!(admission.target().kv_pages, 8);
            assert_eq!(
                admission.model_budget().state_bytes,
                3 * admission.model_estimate().per_sequence_state_bytes
            );
            let model = admission.model_estimate();
            assert!(
                model.weight_bytes_upper_bound < admission.target().routed_expert_storage_bytes
            );
        }
    }

    fn options() -> ModelFactoryOptions {
        ModelFactoryOptions {
            max_layers: None,
            max_tensor_mebibytes: 1024,
            output_head_chunk_rows: 4096,
            expert_reader_max_tensor_mebibytes: 64,
            expert_cache: ExpertCacheOptions::default(),
            qwen35_moe_capacity: None,
            qwen35_host_cache: None,
            moe_hotset_experts: 0,
            kv_cache_mebibytes: Some(1024),
            scheduler_config: Default::default(),
            driver_config: Default::default(),
        }
    }

    #[test]
    fn exact_moe_profile_resolves_cuda_f32_and_rejects_cpu_and_parallel() {
        let descriptor = descriptor(ModelFamily::Qwen35Moe, "Qwen3_5MoeForConditionalGeneration");
        let config = ferrule_model::AutoConfig::from_descriptor(descriptor.clone());
        for selection in [
            BackendSelection::Cpu,
            BackendSelection::Auto,
            BackendSelection::Cuda,
        ] {
            let result = BuiltinModelResolver.resolve(&descriptor, selection);
            if selection == BackendSelection::Cpu || !cfg!(feature = "cuda") {
                let ferrule_runtime::Error::Backend {
                    source: ferrule_common::Error::ModelSource { source },
                } = result.unwrap_err()
                else {
                    panic!("expected typed unsupported")
                };
                let unsupported = source
                    .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
                    .unwrap();
                assert_eq!(
                    unsupported.operator,
                    if selection == BackendSelection::Cpu {
                        "qwen35_moe_cpu"
                    } else {
                        "qwen35_moe_cuda"
                    }
                );
            } else {
                let entry = result.unwrap();
                assert_eq!(entry.backend(), ModelExecutionBackend::Cuda);
                assert_eq!(
                    entry.backend_profile(),
                    Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE
                );
                let mut o = options();
                o.driver_config.enable_native_proposals = false;
                let plan = ResidentModelPlanner
                    .prepare(&config, selection, None, o)
                    .unwrap();
                assert_eq!(plan.chat_template(), ChatTemplate::Qwen35);
                assert_eq!(plan.model_name(), "qwen3.5-35b-a3b-fp8");
                for case in 0..7 {
                    let mut o = options();
                    o.driver_config.enable_native_proposals = false;
                    match case {
                        0 => o.expert_cache.host_entries = 1,
                        1 => o.moe_hotset_experts = 1,
                        2 => o.output_head_chunk_rows = 1,
                        3 => o.expert_reader_max_tensor_mebibytes = 1,
                        4 => o.max_layers = Some(1),
                        5 => o.driver_config.ctx_size = 262145,
                        _ => o.max_tensor_mebibytes = 128,
                    }
                    assert!(
                        ResidentModelPlanner
                            .prepare(&config, selection, None, o)
                            .is_err(),
                        "case {case}"
                    );
                }
            }
            for case in 0..4 {
                let mut pipeline = PipelineBuildOptions::default();
                match case {
                    0 => pipeline.parallelism.tensor_parallel = 2,
                    1 => pipeline.parallelism.pipeline_parallel = 2,
                    2 => pipeline.parallelism.expert_parallel = 2,
                    _ => pipeline.rank_backend = PipelineRankBackend::Process,
                }
                assert!(
                    ResidentModelPlanner
                        .prepare_pipeline(&config, selection, None, options(), pipeline)
                        .is_err()
                );
            }
        }
    }
}

// PR20: descriptor-only resolution deliberately uses a nonexistent model path.
// Success here must not initialize a device or promise that build can succeed.
mod requested_effective {
    use super::*;
    use ferrule_model::AutoConfig;
    use ferrule_runtime::engine::model_factory::*;
    use ferrule_runtime::{ResidentSchedulerConfig, ResidentTopKDriverConfig};

    fn options() -> ModelFactoryOptions {
        ModelFactoryOptions {
            max_layers: None,
            max_tensor_mebibytes: 1024,
            output_head_chunk_rows: 4096,
            expert_reader_max_tensor_mebibytes: 64,
            expert_cache: ExpertCacheOptions::default(),
            qwen35_moe_capacity: None,
            qwen35_host_cache: None,
            moe_hotset_experts: 0,
            kv_cache_mebibytes: None,
            scheduler_config: ResidentSchedulerConfig::default(),
            driver_config: ResidentTopKDriverConfig {
                enable_native_proposals: false,
                ..Default::default()
            },
        }
    }

    #[test]
    fn long_context_effective_profile_family_matrix_keeps_legacy_factory_reports() {
        for family in [
            ModelFamily::Qwen3,
            ModelFamily::QwenMoe,
            ModelFamily::Qwen35,
            ModelFamily::Qwen35Moe,
            ModelFamily::DeepSeekV4,
        ] {
            for context in [1024, 2048, 4096] {
                let config = AutoConfig::from_descriptor(descriptor(family.clone(), "matrix"));
                let mut input = options();
                input.driver_config.ctx_size = context;
                input.scheduler_config.max_active_sequences = 4;
                input.scheduler_config.max_batch_tokens = 512;
                input.scheduler_config.prefill_chunk_size = 128;
                let result = ResidentModelPlanner.prepare(
                    &config,
                    BackendSelection::Auto,
                    None,
                    input.clone(),
                );
                if !cfg!(feature = "cuda")
                    && matches!(family, ModelFamily::Qwen35Moe | ModelFamily::DeepSeekV4)
                {
                    assert!(result.is_err());
                    continue;
                }
                let plan = result.unwrap();
                let report = plan.resolution_report();
                let adjustments = plan.adjustment_report();
                let profile = plan.context_capacity_profile().unwrap();
                let hybrid = matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe);
                assert_eq!(profile.ctx_size, context);
                assert_eq!(profile.max_active_sequences, 4);
                assert_eq!(
                    profile.max_batch_tokens,
                    Some(if hybrid { 32 } else { 512 })
                );
                assert_eq!(profile.prefill_chunk_size, if hybrid { 32 } else { 128 });
                assert_eq!(profile.kv, None);
                assert_eq!(profile.recurrent_bytes, None);
                assert_eq!(profile.workspace_bytes, None);
                assert_eq!(profile.root_required_bytes, None);
                assert_eq!(
                    plan.requested_options().scheduler_config,
                    input.scheduler_config
                );
                assert_eq!(
                    plan.effective_options().scheduler_config.max_batch_tokens,
                    profile.max_batch_tokens.unwrap()
                );
                assert_eq!(plan.resolution_report(), report);
                assert_eq!(plan.adjustment_report(), adjustments);
                eprintln!("metadata only {family:?}: {profile:?}");
            }
        }
    }

    #[test]
    fn requested_effective_family_backend_profile_matrix_without_devices() {
        for (family, name, profile) in [
            (ModelFamily::Qwen3, "qwen3-dense", "cpu-standard-decoder"),
            (ModelFamily::QwenMoe, "qwen3-moe", "cpu-standard-decoder"),
            (
                ModelFamily::Qwen35,
                "qwen3.5-0.8b",
                "cpu-hybrid-f32-qwen35-0.8b",
            ),
            (
                ModelFamily::Qwen35Moe,
                "qwen3.5-35b-a3b-fp8",
                Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            ),
            (ModelFamily::DeepSeekV4, "deepseek-v4", "cuda"),
        ] {
            for selection in [
                BackendSelection::Auto,
                BackendSelection::Cpu,
                BackendSelection::Cuda,
            ] {
                let config = AutoConfig::from_descriptor(descriptor(family.clone(), "matrix"));
                let result = ResidentModelPlanner.prepare(&config, selection, None, options());
                let cuda_only = matches!(family, ModelFamily::Qwen35Moe | ModelFamily::DeepSeekV4);
                let supported = if cuda_only {
                    cfg!(feature = "cuda") && selection != BackendSelection::Cpu
                } else if selection == BackendSelection::Cuda {
                    family == ModelFamily::Qwen35 && cfg!(feature = "cuda")
                } else {
                    true
                };
                if !supported {
                    let error = result
                        .err()
                        .expect("unsupported selection must not fall back");
                    if family == ModelFamily::Qwen35Moe {
                        let ferrule_runtime::Error::Backend {
                            source: ferrule_common::Error::ModelSource { source },
                        } = error
                        else {
                            panic!("lost typed FP8 unsupported error: {error}")
                        };
                        assert!(
                            source
                                .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
                                .is_some()
                        );
                    }
                    continue;
                }
                let plan = result.unwrap();
                assert_eq!(plan.requested_backend(), selection);
                assert_eq!(plan.model_name(), name);
                assert_eq!(
                    plan.backend_profile(),
                    if family == ModelFamily::Qwen35 && selection == BackendSelection::Cuda {
                        "cuda-hybrid-f32-qwen35-0.8b"
                    } else {
                        profile
                    }
                );
                assert_eq!(
                    plan.selected_implementation().kind,
                    SelectedImplementationKind::Resident
                );
                assert_eq!(
                    plan.support().buildability,
                    BuildabilityStatus::CatalogValidated
                );
                assert_eq!(
                    plan.support().admission,
                    AdmissionBoundary::OwnerLiveAdmissionRequired
                );
                assert_eq!(plan.support().cuda_feature_enabled, cfg!(feature = "cuda"));
                assert_eq!(plan.requested_options().max_layers, None);
                assert_eq!(plan.effective_options().max_layers, Some(48));
                assert!(plan.resolution_report().contains("metadata only"));
                assert!(
                    plan.resolution_report()
                        .contains("physical transaction slots P")
                );
                if family == ModelFamily::Qwen35Moe {
                    assert_eq!(
                        plan.support().storage,
                        StorageEncoding::NumericFp8E4m3fnBf16Scales
                    );
                    assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32Tf32x3);
                    assert_eq!(
                        plan.effective_options().qwen35_moe_capacity,
                        plan.qwen35_moe_capacity_limits().unwrap()
                    );
                }
                eprintln!("{}", plan.resolution_report());
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn legacy_moe_options_roundtrip_requested_effective_and_ep_matrix() {
        use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostExpertCacheOptions};
        let mut desc = descriptor(ModelFamily::Qwen35Moe, "qwen35");
        desc.spec.moe.num_experts = Some(256);
        let config = AutoConfig::from_descriptor(desc);
        for backend in [BackendSelection::Auto, BackendSelection::Cuda] {
            for degree in [1, 2, 4, 8] {
                for explicit in [false, true] {
                    for lazy in [false, true] {
                        let mut input = options();
                        input.driver_config.ctx_size = 16;
                        input.scheduler_config.max_active_sequences = 3;
                        input.scheduler_config.max_batch_tokens = 512;
                        input.scheduler_config.prefill_chunk_size = 512;
                        input.kv_cache_mebibytes = Some(64);
                        if explicit {
                            input.qwen35_host_cache = Some(HostExpertCacheOptions {
                                workers: 2,
                                ..Default::default()
                            });
                        }
                        if lazy {
                            input.qwen35_host_cache = Some(HostExpertCacheOptions {
                                mode: ExpertPrewarmMode::Lazy,
                                max_experts: 0,
                                max_bytes: 0,
                                ..Default::default()
                            });
                        }
                        if explicit && degree == 1 {
                            input.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
                                max_experts: 64,
                                max_positions: 999,
                                max_sequences: 99,
                                max_batch_tokens: 99,
                                ..Default::default()
                            });
                        }
                        let result = if degree == 1 {
                            ResidentModelPlanner.prepare(&config, backend, None, input.clone())
                        } else {
                            ResidentModelPlanner.prepare_pipeline(
                                &config,
                                backend,
                                None,
                                input.clone(),
                                PipelineBuildOptions {
                                    parallelism: ferrule_common::ParallelismPlan {
                                        expert_parallel: degree,
                                        ..Default::default()
                                    },
                                    devices: Some((0..degree).rev().collect()),
                                    ..Default::default()
                                },
                            )
                        };
                        if degree != 1 && lazy {
                            let error = result.err().expect("EP still requires full host prewarm");
                            assert!(
                                error
                                    .to_string()
                                    .contains("complete shared full host prewarm")
                            );
                            continue;
                        }
                        let plan = result.unwrap();
                        let requested = plan.requested_options();
                        assert_eq!(requested.qwen35_moe_capacity, input.qwen35_moe_capacity);
                        assert_eq!(requested.qwen35_host_cache, input.qwen35_host_cache);
                        assert_eq!(requested.scheduler_config, input.scheduler_config);
                        let effective = plan.effective_options();
                        assert_eq!(effective.scheduler_config.max_batch_tokens, 16);
                        assert_eq!(effective.scheduler_config.prefill_chunk_size, 16);
                        assert_eq!(
                            effective.qwen35_host_cache,
                            Some(input.qwen35_host_cache.unwrap_or_default())
                        );
                        assert_eq!(
                            plan.qwen35_host_cache_options(),
                            effective.qwen35_host_cache
                        );
                        if degree == 1 {
                            assert_eq!(
                                effective.qwen35_moe_capacity,
                                Some(Qwen35MoeCapacityLimits {
                                    max_positions: 16,
                                    max_sequences: 3,
                                    max_batch_tokens: 16,
                                    kv_bytes: 64 << 20,
                                    ..input.qwen35_moe_capacity.unwrap_or_default()
                                })
                            );
                            assert!(plan.qwen35_expert_placement().is_none());
                            assert_eq!(
                                plan.selected_implementation().kind,
                                SelectedImplementationKind::Resident
                            );
                        } else {
                            assert_eq!(effective.qwen35_moe_capacity, None);
                            let placement = plan.qwen35_expert_placement().unwrap();
                            assert_eq!(placement.expert_parallel(), degree);
                            assert_eq!(placement.root_device(), degree - 1);
                            assert_eq!(
                                plan.selected_implementation().kind,
                                SelectedImplementationKind::DedicatedQwen35ExpertParallel
                            );
                            assert!(plan.pipeline_options().is_none());
                        }
                        assert_eq!(
                            plan.qwen35_moe_capacity_limits().unwrap(),
                            effective.qwen35_moe_capacity
                        );
                        let report = plan.adjustment_report();
                        let fields: Vec<_> = report
                            .adjustments()
                            .iter()
                            .map(|change| change.field)
                            .collect();
                        let mut expected = vec![
                            "max_layers",
                            "scheduler_config.max_batch_tokens",
                            "scheduler_config.prefill_chunk_size",
                        ];
                        if degree == 1 {
                            expected.push("qwen35_moe_capacity");
                        }
                        if input.qwen35_host_cache.is_none() {
                            expected.push("qwen35_host_cache");
                        }
                        assert_eq!(fields, expected);
                        assert_eq!(
                            plan.support().buildability,
                            BuildabilityStatus::CatalogValidated
                        );
                        assert_eq!(
                            plan.support().admission,
                            AdmissionBoundary::OwnerLiveAdmissionRequired
                        );
                        assert!(
                            plan.resolution_report()
                                .contains("metadata only, owner live admission required")
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn public_catalog_build_request_signature_remains_compatible() {
        fn accepts_legacy_builder(
            _: fn(
                ModelBuildRequest,
            )
                -> ferrule_runtime::Result<ferrule_runtime::engine::BoxedSessionInferenceEngine>,
        ) {
        }
        fn assert_legacy_traits<T: Clone + std::fmt::Debug + Send>() {}
        assert_legacy_traits::<ModelBuildRequest>();
        for implementation in MODEL_IMPLEMENTATIONS {
            accepts_legacy_builder(implementation.build);
        }
    }

    #[test]
    fn qwen35_clamp_reason_is_visible_for_32_then_context_and_explicit_defaults() {
        let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "qwen35"));
        for (batch, prefill, context, expected_batch, expected_prefill) in [
            (512, 512, 1024, 32, 32),
            (32, 32, 16, 16, 16),
            (512, 512, 8, 8, 8),
            (8, 4, 64, 8, 4),
            (32, 32, 32, 32, 32),
        ] {
            let mut input = options();
            input.max_layers = Some(48);
            input.scheduler_config.max_batch_tokens = batch;
            input.scheduler_config.prefill_chunk_size = prefill;
            input.driver_config.ctx_size = context;
            let plan = ResidentModelPlanner
                .prepare(&config, BackendSelection::Cpu, None, input)
                .unwrap();
            assert_eq!(
                plan.requested_options().scheduler_config.max_batch_tokens,
                batch
            );
            assert_eq!(
                plan.requested_options().scheduler_config.prefill_chunk_size,
                prefill
            );
            assert_eq!(
                plan.effective_options().scheduler_config.max_batch_tokens,
                expected_batch
            );
            assert_eq!(
                plan.effective_options().scheduler_config.prefill_chunk_size,
                expected_prefill
            );
            let report = plan.adjustment_report();
            assert_eq!(
                report.is_empty(),
                batch == expected_batch && prefill == expected_prefill
            );
            for change in report.adjustments() {
                assert!(change.reason.contains("min(32, context)"));
                assert_ne!(change.requested, change.effective);
            }
            eprintln!("{}", plan.resolution_report());
        }
    }

    #[test]
    fn requested_effective_pipeline_thread_process_and_ep_matrix() {
        for moe in [false, true] {
            let config = AutoConfig::from_descriptor(if moe {
                qwen_descriptor()
            } else {
                descriptor(ModelFamily::Qwen3, "qwen3")
            });
            for backend in [
                BackendSelection::Auto,
                BackendSelection::Cpu,
                BackendSelection::Cuda,
            ] {
                for rank in [PipelineRankBackend::Thread, PipelineRankBackend::Process] {
                    let mut parallel = PipelineBuildOptions {
                        parallelism: ferrule_common::ParallelismPlan {
                            pipeline_parallel: 2,
                            expert_parallel: if moe { 2 } else { 1 },
                            ..Default::default()
                        },
                        rank_backend: rank,
                        ..Default::default()
                    };
                    #[cfg(unix)]
                    if rank == PipelineRankBackend::Process {
                        parallel.process_launch =
                            Some(ferrule_runtime::parallel::process::ProcessLaunch::new(
                                std::env::current_exe().unwrap(),
                            ));
                    }
                    let mut input = options();
                    input.driver_config.ctx_size = 16;
                    input.driver_config.enable_native_proposals = true;
                    input.scheduler_config.max_decode_batch = 4;
                    input.scheduler_config.decode_cohort_target = 4;
                    input.scheduler_config.decode_cohort_max_deferrals = 3;
                    let result = ResidentModelPlanner
                        .prepare_pipeline(&config, backend, None, input, parallel);
                    if (backend == BackendSelection::Cuda && !cfg!(feature = "cuda"))
                        || (rank == PipelineRankBackend::Process && !cfg!(unix))
                    {
                        assert!(result.is_err());
                        continue;
                    }
                    let plan = result.unwrap();
                    assert_eq!(plan.requested_backend(), backend);
                    assert_eq!(
                        plan.selected_implementation().kind,
                        SelectedImplementationKind::GenericPipeline
                    );
                    assert_eq!(
                        plan.requested_parallel_options().unwrap().rank_backend,
                        rank
                    );
                    assert_eq!(plan.topology_placement().rank_backend, Some(rank));
                    assert_eq!(plan.topology_placement().logical.pipeline_parallel, 2);
                    let old_profile = match (rank, plan.backend(), moe) {
                        (PipelineRankBackend::Thread, ModelExecutionBackend::Cpu, false) => {
                            "cpu-pipeline-serial"
                        }
                        (PipelineRankBackend::Thread, ModelExecutionBackend::Cpu, true) => {
                            "cpu-pipeline-ep-f32-serial"
                        }
                        (PipelineRankBackend::Process, ModelExecutionBackend::Cpu, false) => {
                            "cpu-pipeline-process-bf16-serial"
                        }
                        (PipelineRankBackend::Process, ModelExecutionBackend::Cpu, true) => {
                            "cpu-pipeline-process-ep-f32-serial"
                        }
                        (PipelineRankBackend::Thread, ModelExecutionBackend::Cuda, _) => {
                            "cuda-pipeline-f32-serial"
                        }
                        (PipelineRankBackend::Process, ModelExecutionBackend::Cuda, _) => {
                            "cuda-pipeline-process-f32-serial"
                        }
                    };
                    assert_eq!(plan.backend_profile(), old_profile);
                    assert_eq!(
                        plan.requested_options().scheduler_config.max_decode_batch,
                        4
                    );
                    let effective = plan.effective_options();
                    assert_eq!(effective.scheduler_config.max_decode_batch, 1);
                    assert_eq!(effective.scheduler_config.max_batch_tokens, 16);
                    assert_eq!(effective.scheduler_config.prefill_chunk_size, 16);
                    assert_eq!(effective.scheduler_config.decode_cohort_target, 1);
                    assert_eq!(effective.scheduler_config.decode_cohort_max_deferrals, 0);
                    assert!(!effective.scheduler_config.allow_mixed_batches);
                    assert!(!effective.driver_config.enable_native_proposals);
                    assert_eq!(plan.adjustment_report().adjustments().len(), 8);
                    eprintln!("{}", plan.resolution_report());
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dedicated_ep8_reports_root_colocation_source_and_no_local_cache_without_cuda_init() {
        let mut descriptor = descriptor(ModelFamily::Qwen35Moe, "qwen35");
        descriptor.spec.moe.num_experts = Some(256);
        let config = AutoConfig::from_descriptor(descriptor);
        let plan = ResidentModelPlanner
            .prepare_qwen35_expert_parallel(
                &config,
                BackendSelection::Auto,
                None,
                options(),
                PipelineBuildOptions {
                    parallelism: ferrule_common::ParallelismPlan {
                        expert_parallel: 8,
                        ..Default::default()
                    },
                    devices: Some(vec![7, 6, 5, 4, 3, 2, 1, 0]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::DedicatedQwen35ExpertParallel
        );
        assert!(plan.pipeline_options().is_none());
        assert!(plan.effective_options().qwen35_moe_capacity.is_none());
        assert_eq!(
            plan.backend_profile(),
            "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-thread-ep"
        );
        let placement = plan.topology_placement();
        assert_eq!(placement.logical.expert_parallel, 8);
        assert_eq!(placement.logical.tensor_parallel, 1);
        assert_eq!(placement.logical.pipeline_parallel, 1);
        assert_eq!(placement.physical.root_device, Some(7));
        assert_eq!(placement.physical.expert_devices.as_ref().unwrap()[0], 7);
        assert_eq!(
            placement.physical.source_identity,
            Some(ferrule_common::ParallelRankId::new(0))
        );
        assert_eq!(
            plan.support().storage,
            StorageEncoding::NumericFp8E4m3fnBf16Scales
        );
        assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32Tf32x3);
    }
}
