//! Internal candidate admission and trace reuse; never public factory enablement.
use super::*;
use ferrule_model::transformer::BoundDecoderResources;
use ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan;

const CACHE_GRANTS: usize = 64 << 20;

fn margins() -> ParallelOwnerMargins {
    ParallelOwnerMargins {
        context_bytes: 64 << 20,
        allocator_bytes: 64 << 20,
    }
}
fn capacity(c: PipelineConfig) -> ParallelRankCapacity {
    ParallelRankCapacity {
        page_size: c.page_size,
        max_positions: c.max_positions,
        max_rows: c.max_batch_tokens,
        sessions: c.session_capacity,
        logical_pages: c.max_pages,
    }
}
fn limits(resources: &BoundDecoderResources, c: PipelineConfig) -> HostCollectiveLimits {
    HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: c.max_batch_tokens
            * resources
                .spec()
                .vocab_size()
                .div_ceil(2)
                .max(resources.spec().hidden_size()),
        max_host_bytes: 64 << 20,
    }
}
fn small_config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 32,
        max_batch_tokens: 2,
        max_parameter_bytes: 1 << 20,
        ..config()
    }
}
fn plan(resources: &BoundDecoderResources) -> ParallelExecutionPlan {
    ParallelExecutionPlan::describe(resources.spec(), &placement(), contract(), vec![]).unwrap()
}
fn inputs(
    plan: &ParallelExecutionPlan,
    resources: &BoundDecoderResources,
    c: PipelineConfig,
    grants: Option<usize>,
) -> MixedTensorExecutionPlan {
    plan.internal_mixed_execution_inputs(
        resources,
        c,
        margins(),
        &[(0, 22 << 30), (1, 22 << 30)],
        limits(resources, c),
        grants,
    )
    .unwrap()
}

#[test]
fn native_default_off_and_cache_report_claim_share_exact_grant() {
    let fixture = tiny::base::Fixture::new();
    let resources = tiny::resources(&fixture);
    let plan = plan(&resources);
    let c = small_config();
    let native = inputs(&plan, &resources, c, None);
    let cached = inputs(&plan, &resources, c, Some(CACHE_GRANTS));
    let base = plan
        .resource_requirements_from_mixed_inputs(&resources, capacity(c), margins(), &native)
        .unwrap();
    let report = plan
        .resource_requirements_from_mixed_inputs(&resources, capacity(c), margins(), &cached)
        .unwrap();
    assert!(report.resource_complete());
    let legacy = plan
        .resource_requirements_from_model(
            &resources,
            capacity(c),
            margins(),
            &[native.collective().as_ref().clone()],
        )
        .unwrap();
    for (old, unchanged) in legacy.owners().iter().zip(base.owners()) {
        assert_eq!(
            old.device_bytes().unwrap(),
            unchanged.device_bytes().unwrap()
        );
        assert_eq!(
            old.operators.operation_grants,
            unchanged.operators.operation_grants
        );
    }
    report
        .prepare_internal(&[(0, 22 << 30), (1, 22 << 30)], 128 << 30, |_| Ok(()))
        .unwrap();
    for lane in 0..2 {
        assert_eq!(
            native.program(lane).unwrap().decoded_weight_cache_budget(),
            None
        );
        let program = cached.program(lane).unwrap();
        assert_eq!(
            program
                .decoded_weight_cache_budget()
                .unwrap()
                .capacity_grants,
            CACHE_GRANTS
        );
        let owner = &report.owners()[lane];
        assert_eq!(
            owner.device_bytes().unwrap().unwrap(),
            base.owners()[lane].device_bytes().unwrap().unwrap() + CACHE_GRANTS as u64
        );
        assert_eq!(
            owner.operators.operation_grants.unwrap(),
            base.owners()[lane].operators.operation_grants.unwrap() + CACHE_GRANTS as u64
        );
        assert_eq!(
            owner.operators.providers,
            base.owners()[lane].operators.providers
        );
        let (estimate, context) = program
            .cuda_claim_inputs(
                &resources,
                capacity(c).physical_pages().unwrap(),
                lane,
                c.max_parameter_bytes,
                cached.attention_shapes(),
            )
            .unwrap();
        let budget = ferrule_model::decoder::HybridCudaMemoryBudget {
            kv_pages: capacity(c).physical_pages().unwrap(),
            state_bytes: estimate.per_sequence_state_bytes * capacity(c).state_copies().unwrap(),
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        };
        context.validate_device_budget(lane, budget).unwrap();
        context.validate_estimate(estimate).unwrap();
        assert!(
            context
                .validate_device_budget(
                    lane,
                    ferrule_model::decoder::HybridCudaMemoryBudget {
                        workspace_bytes: budget.workspace_bytes - CACHE_GRANTS,
                        ..budget
                    }
                )
                .is_err(),
            "native claim must not admit cache-on preparation"
        );
        assert!(
            program.operation_estimate(&resources, c.max_pages).is_err(),
            "logical pages must not replace physical slots"
        );
    }
    assert!(
        cached
            .clone()
            .with_decoded_weight_cache(&resources, CACHE_GRANTS, [22 << 30; 2], 128 << 20)
            .is_err()
    );
}

#[test]
fn cache_rejects_low_budget_capacity_and_configuration_drift() {
    let fixture = tiny::base::Fixture::new();
    let resources = tiny::resources(&fixture);
    let plan = plan(&resources);
    let c = small_config();
    let make = |grant, caps: &[(usize, u64)], margin| {
        plan.internal_mixed_execution_inputs(
            &resources,
            c,
            margin,
            caps,
            limits(&resources, c),
            Some(grant),
        )
    };
    for invalid_grant in [0, 4 << 20, CACHE_GRANTS - 1, CACHE_GRANTS + 1] {
        assert!(make(invalid_grant, &[(0, 22 << 30), (1, 22 << 30)], margins()).is_err());
    }
    assert!(make(CACHE_GRANTS, &[(0, 1), (1, 22 << 30)], margins()).is_err());
    assert!(make(CACHE_GRANTS, &[(0, 22 << 30)], margins()).is_err());
    assert!(
        make(
            CACHE_GRANTS,
            &[(0, 22 << 30), (0, 22 << 30), (1, 22 << 30)],
            margins()
        )
        .is_err()
    );
    assert!(make(usize::MAX, &[(0, 22 << 30), (1, 22 << 30)], margins()).is_err());
    assert!(
        make(
            CACHE_GRANTS,
            &[(0, 22 << 30), (1, 22 << 30)],
            ParallelOwnerMargins {
                context_bytes: u64::MAX,
                allocator_bytes: 1,
            }
        )
        .is_err()
    );
    let cached = inputs(&plan, &resources, c, Some(CACHE_GRANTS));
    for changed in [
        ParallelRankCapacity {
            logical_pages: c.max_pages + 1,
            ..capacity(c)
        },
        ParallelRankCapacity {
            sessions: c.session_capacity + 1,
            ..capacity(c)
        },
    ] {
        assert!(
            plan.resource_requirements_from_mixed_inputs(&resources, changed, margins(), &cached)
                .is_err()
        );
    }
    let report = plan
        .resource_requirements_from_mixed_inputs(&resources, capacity(c), margins(), &cached)
        .unwrap();
    let total = report.owners()[0].device_bytes().unwrap().unwrap();
    assert!(
        report
            .prepare_internal(&[(0, total - 1), (1, 22 << 30)], 128 << 30, |_| Ok(()))
            .is_err()
    );
    // A changed source is rejected before any CUDA owner can be opened.
    std::fs::write(fixture.dir.join("runtime-mixed-native.bin"), [0]).unwrap();
    assert!(
        plan.resource_requirements_from_mixed_inputs(&resources, capacity(c), margins(), &cached)
            .is_err()
    );
}

#[test]
fn cache_capacity_follows_device_placement_not_lane_number() {
    let fixture = tiny::base::Fixture::new();
    let resources = tiny::resources(&fixture);
    let c = small_config();
    let plan = ParallelExecutionPlan::describe(
        resources.spec(),
        &PipelineBuildOptions {
            devices: Some(vec![5, 3]),
            ..placement()
        },
        contract(),
        vec![],
    )
    .unwrap();
    assert!(
        plan.internal_mixed_execution_inputs(
            &resources,
            c,
            margins(),
            &[(3, 22 << 30), (5, 22 << 30)],
            limits(&resources, c),
            Some(CACHE_GRANTS)
        )
        .is_ok()
    );
    assert!(
        plan.internal_mixed_execution_inputs(
            &resources,
            c,
            margins(),
            &[(0, 22 << 30), (1, 22 << 30)],
            limits(&resources, c),
            Some(CACHE_GRANTS)
        )
        .is_err()
    );
}

fn header_ledger(sessions: usize) {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = std::env::var("FERRULE_MIXED_HEADER_DIR").unwrap();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path))
            .unwrap();
    let plan = plan(&resources);
    let c = PipelineConfig {
        session_capacity: sessions,
        ..config()
    };
    let inputs = inputs(&plan, &resources, c, Some(3 << 30));
    let report = plan
        .resource_requirements_from_mixed_inputs(&resources, capacity(c), margins(), &inputs)
        .unwrap();
    assert!(report.resource_complete());
    report
        .prepare_internal(&[(0, 22 << 30), (1, 22 << 30)], 128 << 30, |_| Ok(()))
        .unwrap();
    for lane in 0..2 {
        let program = inputs.program(lane).unwrap();
        let b = program.decoded_weight_cache_budget().unwrap();
        assert_eq!(b.capacity_grants, 3 << 30);
        assert_eq!(b.pinned_grants, 1340 << 20);
        assert_eq!(b.routed_working_set_grants, 768 << 20);
        assert_eq!(b.routed_entries, 20_480);
        let owner = &report.owners()[lane];
        let total = owner.device_bytes().unwrap().unwrap();
        let total_with_trace = total.checked_add(8 << 20).unwrap();
        assert!(total_with_trace <= 22 << 30);
        if sessions == 2 {
            assert_eq!(total, 22_575_706_112);
        }
        eprintln!(
            "DECODED_CACHE_LEDGER {}",
            serde_json::json!({
                "owner":owner.owner.get(),"device":owner.device,"sessions":sessions,
                "physical_pages":capacity(c).physical_pages().unwrap(),"state_copies":capacity(c).state_copies().unwrap(),
                "cache_grants":b.capacity_grants,"pinned_grants":b.pinned_grants,"routed_working_set_grants":b.routed_working_set_grants,
                "weights":owner.storage.weight_grants,"kv":owner.storage.kv_grants,"state":owner.storage.state.total_grants,
                "operation_grants":owner.operators.operation_grants,"providers":format!("{:?}", owner.operators.providers),
                "auxiliary":format!("{:?}", owner.operators.auxiliary),"boundary":format!("{:?}", owner.operators.boundary),
                "owner_margin":128 << 20,"total_with_trace":total_with_trace,"capacity":22u64 << 30,
                "headroom":(22u64 << 30)-total_with_trace,
            })
        );
    }
}
#[test]
#[ignore = "CPU-only real checkpoint headers; hidden GPUs"]
fn full40_cache_header_ledger() {
    header_ledger(2);
}
#[test]
#[ignore = "CPU-only real checkpoint headers; hidden GPUs; lifecycle admits three sessions"]
fn lifecycle_cache_header_ledger() {
    header_ledger(3);
}

#[test]
#[ignore = "sole GPU owner: explicit 3 GiB model-admitted cache; unchanged full40 boundary oracle"]
fn full40_cache_candidate_boundaries_two_cases_two_phases() {
    full40_with_cache(true, Some(3 << 30));
}

#[test]
fn admitted_cache_programs_survive_runtime_binding_and_reject_owner_geometry_drift() {
    let fixture = tiny::base::Fixture::new();
    let resources = tiny::resources(&fixture);
    let plan = plan(&resources);
    let c = small_config();
    let admitted = plan
        .mixed_programs_with_decoded_weight_cache(
            &resources,
            capacity(c),
            margins(),
            &[(0, 22 << 30), (1, 22 << 30)],
            CACHE_GRANTS,
        )
        .unwrap();
    let native = inputs(&plan, &resources, c, None);
    let bound = native.clone().with_programs(admitted.clone()).unwrap();
    for (lane, original) in admitted.iter().enumerate() {
        assert_eq!(
            bound.program(lane).unwrap().decoded_weight_cache_budget(),
            original.decoded_weight_cache_budget()
        );
    }
    let mut reordered = admitted.clone();
    reordered.swap(0, 1);
    assert!(native.clone().with_programs(reordered).is_err());
    assert!(
        native
            .clone()
            .with_programs(admitted[..1].to_vec())
            .is_err()
    );
    let other = PipelineConfig {
        max_positions: c.max_positions - 1,
        ..c
    };
    assert!(
        inputs(&plan, &resources, other, None)
            .with_programs(admitted)
            .is_err()
    );
}

#[test]
#[ignore = "sole GPU owner: cache-on full40 without numeric observer; fused logits and publication, not 2178 boundary evidence"]
fn full40_cache_unobserved_fused_logits_two_cases_two_phases() {
    full40_with_cache(false, Some(3 << 30));
}
