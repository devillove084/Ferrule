use super::{Qwen35MoeCapacityLimits, config};
use ferrule_composition::model_factory::Qwen35MoeMemoryPlan;
use ferrule_model::models::qwen35::{Qwen35Adapter, Qwen35HfNameMapper, Qwen35TensorPartitionKind};
use std::io::Write;
use std::path::PathBuf;
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
            include_str!("../../../ferrule-model/tests/qwen35_35b_fp8_config.json"),
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
        let admission = Qwen35MoeMemoryPlan::open_hf(&fixture.0, limits, 1 << 30).unwrap();
        let default_device_bytes = Qwen35MoeCapacityLimits::default().device_bytes;
        if admission.total_device_bytes() > default_device_bytes {
            assert!(
                Qwen35MoeMemoryPlan::open_hf(
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
    let admission = Qwen35MoeMemoryPlan::open_hf(&fixture.0, limits, 1 << 30).unwrap();
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
    let model = ferrule_model::decoder::HybridMemoryPlan::for_resources(
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
        Qwen35MoeMemoryPlan::open_hf(
            &fixture.0,
            Qwen35MoeCapacityLimits {
                device_bytes: admission.total_device_bytes() - 1,
                ..limits
            },
            1 << 30
        )
        .is_err()
    );
    assert!(Qwen35MoeMemoryPlan::open_hf(&fixture.0, limits, 128 << 20).is_err());
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
    let admission = Qwen35MoeMemoryPlan::open_hf(&fixture.0, limits, 1 << 30).unwrap();
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
    assert!(model.weight_bytes_upper_bound < admission.target().routed_expert_storage_bytes);
}

#[test]
fn numeric_profiles_state_kv_and_device_ledger_have_exact_values() {
    use ferrule_common::execution::KvLayoutSchema;
    use ferrule_composition::model_factory::NumericFp8Profile;
    use ferrule_model::decoder::{HybridMemoryPlan, HybridStateSchema};
    let fixture = Headers::new();
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&fixture.0).unwrap();
    for (profile, scratch, weights, workspace) in [
        (
            NumericFp8Profile::F32,
            67_108_864,
            13_981_856_256,
            1_094_729_728,
        ),
        (
            NumericFp8Profile::W8A16Bf16,
            67_108_864,
            13_981_856_256,
            1_094_729_728,
        ),
    ] {
        let limits = Qwen35MoeCapacityLimits {
            scratch_bytes: scratch,
            ..Default::default()
        };
        let plan = Qwen35MoeMemoryPlan::open_hf_with_profile(&fixture.0, limits, 1 << 30, profile)
            .unwrap();
        let expected = HybridMemoryPlan {
            per_sequence_state_bytes: 66_846_720,
            kv_bytes: 335_544_320,
            weight_bytes_upper_bound: weights,
            workspace_bytes_upper_bound: workspace,
        };
        assert_eq!(plan.model_estimate(), expected);
        assert_eq!(plan.model_budget().state_bytes, 601_620_480);
        assert_eq!(plan.model_budget().kv_pages, 512);
        assert_eq!(plan.total_device_bytes(), 16_596_742_656);
        let schema = HybridStateSchema::from_spec(resources.spec(), 40).unwrap();
        assert_eq!(
            schema.per_sequence_state_bytes().unwrap(),
            expected.per_sequence_state_bytes
        );
        assert_eq!(
            schema.kv_planes(16, 1024).unwrap().checked_page_bytes(),
            Some(655_360)
        );
        assert!(HybridMemoryPlan::for_resources(&resources, plan.runner_options(), 0).is_err());
        assert!(
            HybridMemoryPlan::for_resources(&resources, plan.runner_options(), usize::MAX).is_err()
        );
        #[cfg(feature = "cuda")]
        assert_eq!(
            ferrule_model::decoder::HybridCudaMemoryEstimate::for_resources(
                &resources,
                plan.runner_options(),
                512,
            )
            .unwrap(),
            expected.into(),
        );
    }
}

#[test]
fn root_and_colocated_expert_are_charged_once_per_domain() {
    use ferrule_composition::model_factory::{
        Qwen35ExpertPhysicalPlan, Qwen35ExpertPlacement, Qwen35ExpertRootMemory,
        Qwen35PhysicalCardCapacity,
    };
    use ferrule_model::decoder::HybridMemoryPlan;
    let fixture = Headers::new();
    let plan = Qwen35MoeMemoryPlan::open_hf(&fixture.0, Default::default(), 1 << 30).unwrap();
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&fixture.0).unwrap();
    let root =
        Qwen35ExpertRootMemory::for_resources(&resources, plan.runner_options(), 512).unwrap();
    let external =
        HybridMemoryPlan::for_resources_with_routed_experts(&resources, plan.runner_options(), 512)
            .unwrap();
    assert_eq!(root.weights_bytes, 9_753_997_824);
    assert_eq!(root.state_bytes, 601_620_480);
    assert_eq!(root.kv_bytes, 335_544_320);
    assert_eq!(root.workspace_bytes, external.workspace_bytes_upper_bound);
    assert_eq!(
        plan.model_estimate().weight_bytes_upper_bound - root.weights_bytes,
        4_227_858_432
    );
    let placement = Qwen35ExpertPlacement::new(2, vec![7, 3]).unwrap();
    let physical = Qwen35ExpertPhysicalPlan::new(&placement, root, &[100, 200], 300, 400).unwrap();
    let cards = physical.cards();
    assert_eq!(cards[0].ordinal, 7);
    assert_eq!(cards[0].contexts, 2);
    assert_eq!(cards[0].allocator_margin_bytes, 800);
    assert_eq!(
        cards[0].required_bytes,
        root.weights_bytes + root.state_bytes + root.kv_bytes + root.workspace_bytes + 1200
    );
    assert_eq!(cards[1].ordinal, 3);
    assert_eq!(cards[1].contexts, 1);
    assert_eq!(cards[1].required_bytes, 900);
    let mut capacities = cards
        .iter()
        .map(|c| Qwen35PhysicalCardCapacity {
            ordinal: c.ordinal,
            free_bytes: c.required_bytes,
            limit_bytes: c.required_bytes,
        })
        .collect::<Vec<_>>();
    physical.check_capacity(&capacities).unwrap();
    capacities[0].free_bytes -= 1;
    assert!(physical.check_capacity(&capacities).is_err());
    assert!(physical.check_capacity(&capacities[1..]).is_err());
    assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, &[usize::MAX, 200], 300, 400).is_err());
    assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, &[100, 200], 300, usize::MAX).is_err());
}

fn bf16_limits() -> Qwen35MoeCapacityLimits {
    Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 0,
        workspace_bytes: 2 << 30,
        max_positions: 128,
        max_sequences: 1,
        max_batch_tokens: 8,
        ..Default::default()
    }
}
#[test]
fn numeric_bf16_35b_metadata_has_typed_state_and_bounded_storage() {
    use ferrule_composition::model_factory::{
        ModelPrecision, NumericStoragePolicy, Qwen35NumericBf16MemoryPlan,
    };
    let f = Headers::new();
    let limits = bf16_limits();
    let plan = Qwen35NumericBf16MemoryPlan::open_hf(
        &f.0,
        ModelPrecision::Bf16,
        NumericStoragePolicy::E4M3FnBf16Scales,
        limits,
        1 << 30,
    )
    .unwrap();
    let options = plan.runner_options();
    assert_eq!(
        options.precision(),
        ferrule_model::execution::ExecutionPrecisionPolicy::bf16_compatibility()
    );
    assert_eq!(
        options.hybrid_cuda_numeric_fp8_precision(),
        Some(ferrule_model::transformer::NumericFp8Precision::Bf16ModelF32Accumulate)
    );
    // Header identities only: sparse payloads cannot prove GPU resident storage.
    use ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile as Profile;
    use ferrule_model::{CheckpointDType as D, TensorRole};
    let mut counts = [0; 3];
    for p in plan.resources().state_dict().parameters() {
        let (slot, expected) = match p.role() {
            TensorRole::RouterLogits => (0, Profile::ROUTER_WEIGHT_DTYPE),
            TensorRole::LinearAttentionALog => (1, Profile::GDN_ALOG_NORM_SOURCE_DTYPE),
            TensorRole::LinearAttentionNorm => (2, Profile::GDN_ALOG_NORM_SOURCE_DTYPE),
            _ => continue,
        };
        assert_eq!(p.weight().slice().dtype, expected);
        assert!(p.numeric_fp8_encoding().is_none());
        assert!(p.scale().is_none());
        counts[slot] += 1;
    }
    assert_eq!(counts, [40, 30, 30]);
    assert_eq!(Profile::ROUTER_WEIGHT_DTYPE, D::Bf16);
    assert_eq!(Profile::GDN_ALOG_NORM_SOURCE_DTYPE, D::F32);
    assert_eq!(Profile::GDN_ALOG_NORM_RESIDENT_DTYPE, D::Bf16);
    assert_eq!(Profile::GDN_ALOG_NORM_OPERATOR_DTYPE, D::Bf16);
    let schema =
        ferrule_model::decoder::HybridStateSchema::from_spec(plan.resources().spec(), 40).unwrap();
    let estimate = plan.model_estimate();
    let budget = plan.model_budget();
    assert!(estimate.per_sequence_state_bytes < schema.per_sequence_state_bytes().unwrap());
    assert_eq!(budget.state_bytes, 3 * estimate.per_sequence_state_bytes);
    assert_eq!(budget.kv_pages, 16);
    assert!(plan.total_device_bytes() < limits.device_bytes);
    assert!(budget.weight_bytes < 4usize << 30);
    for (precision, cap, read) in [
        (ModelPrecision::F32, limits, 1 << 30),
        (
            ModelPrecision::Bf16,
            Qwen35MoeCapacityLimits {
                max_bytes: 1,
                ..limits
            },
            1 << 30,
        ),
        (
            ModelPrecision::Bf16,
            Qwen35MoeCapacityLimits {
                workspace_bytes: 1,
                ..limits
            },
            1 << 30,
        ),
        (
            ModelPrecision::Bf16,
            Qwen35MoeCapacityLimits {
                device_bytes: 1,
                ..limits
            },
            1 << 30,
        ),
        (ModelPrecision::Bf16, limits, 1),
    ] {
        assert!(
            Qwen35NumericBf16MemoryPlan::open_hf(
                &f.0,
                precision,
                NumericStoragePolicy::E4M3FnBf16Scales,
                cap,
                read
            )
            .is_err()
        );
    }
}

#[test]
fn numeric_bf16_public_factory_is_explicit_not_a_legacy_w8_alias() {
    use ferrule_composition::model_factory::*;
    let f = Headers::new();
    let config = ferrule_model::AutoConfig::from_pretrained(&f.0).unwrap();
    let limits = bf16_limits();
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: Some(limits),
        numeric_fp8_profile: None,
        qwen35_host_cache: None,
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1024),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
            max_active_sequences: 1,
            max_batch_tokens: 8,
            prefill_chunk_size: 8,
            prefix_cache_capacity_pages: 0,
            ..Default::default()
        },
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 128,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let serving = ServingModelOptions {
        model_precision: Some(ModelPrecision::Bf16),
        ..Default::default()
    };
    let plan = ResidentModelPlanner::new()
        .prepare_numeric_fp8_bf16_serving(
            &config,
            BackendSelection::Cuda,
            None,
            options.clone(),
            serving.clone(),
            NumericStoragePolicy::E4M3FnBf16Scales,
        )
        .unwrap();
    assert_eq!(
        plan.backend_profile(),
        ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile::NAME
    );
    assert_eq!(
        plan.qwen35_host_cache_options().unwrap().mode,
        ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy
    );
    assert_eq!(plan.numeric_fp8_profile(), None);
    assert_eq!(plan.observability().numeric_fp8_profile, None);
    assert_eq!(
        plan.support().arithmetic,
        ArithmeticPrecision::Bf16Compatibility
    );
    assert_eq!(
        plan.support().storage,
        StorageEncoding::NumericFp8E4m3fnBf16Scales
    );
    assert!(plan.numeric_bf16_memory_plan().unwrap().is_some());
    let constraints = plan.support().static_constraints;
    for boundary in [
        "router parameter weights/logits BF16, probabilities/route weights F32",
        "GDN A_log/norm sources BF16 or F32, F32 sources cast once to resident/operator BF16",
        "F32 accumulation/SSM",
        "RoPE inv_freq/trig F32, resident cos/sin/output BF16",
        "full-depth 35B TP1/PP1/EP1/device0",
        "metadata admission does not validate GPU storage",
        "GPU quality acceptance pending",
    ] {
        assert!(
            constraints.contains(boundary),
            "missing boundary: {boundary}"
        );
    }
    assert!(!constraints.contains("F32 accumulation/SSM/router weights"));
    for bad in [
        Some(NumericFp8Profile::F32),
        Some(NumericFp8Profile::W8A16Bf16),
    ] {
        let mut o = options.clone();
        o.numeric_fp8_profile = bad;
        assert!(
            ResidentModelPlanner::new()
                .prepare_numeric_fp8_bf16_serving(
                    &config,
                    BackendSelection::Cuda,
                    None,
                    o,
                    serving.clone(),
                    NumericStoragePolicy::E4M3FnBf16Scales
                )
                .is_err()
        );
    }
    assert!(
        ResidentModelPlanner::new()
            .prepare_numeric_fp8_bf16_serving(
                &config,
                BackendSelection::Cuda,
                None,
                options,
                ServingModelOptions::default(),
                NumericStoragePolicy::E4M3FnBf16Scales
            )
            .is_err()
    );
}

#[test]
#[ignore = "local real checkpoint headers only; no CUDA context or payload reads"]
fn numeric_bf16_actual_35b_checkpoint_metadata() {
    use ferrule_composition::model_factory::{
        ModelPrecision, NumericStoragePolicy, Qwen35NumericBf16MemoryPlan,
    };
    let path = std::path::Path::new("/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8");
    let plan = Qwen35NumericBf16MemoryPlan::open_hf(
        path,
        ModelPrecision::Bf16,
        NumericStoragePolicy::E4M3FnBf16Scales,
        bf16_limits(),
        1 << 30,
    )
    .unwrap();
    eprintln!(
        "numeric BF16 actual 35B: estimate={:?} budget={:?} total={}",
        plan.model_estimate(),
        plan.model_budget(),
        plan.total_device_bytes()
    );
}
